"""학습 (CLAUDE.md M4).

config 로드 → 학습 → best ckpt + metrics.json + config 사본 저장.
early stopping(val macro-F1), cosine LR, WeightedRandomSampler, 재현 seed.

사용:
    python scripts/05_train.py --config configs/train_proposed.yaml
    python scripts/05_train.py --config configs/train_proposed.yaml --epochs 30 --overfit
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import ViewSelectionDataset, collate  # noqa: E402
from src.data.feature_cache import FeatureCache  # noqa: E402
from src.data.nusc_wrapper import NuscWrapper  # noqa: E402
from src.data.splits import load_split, split_for_names  # noqa: E402
from src.eval.metrics import macro_f1, top1  # noqa: E402
from src.labeling.label_function import LabelConfig  # noqa: E402
from src.models.proposed import (  # noqa: E402
    LossWeights, ModelConfig, ViewSelector, compute_loss,
)


def load_cfgs(train_cfg_path):
    d = os.path.join(os.path.dirname(__file__), "..", "configs")
    with open(os.path.join(d, "base.yaml"), encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(os.path.join(d, "label_v1.yaml"), encoding="utf-8") as f:
        lab = yaml.safe_load(f)
    with open(train_cfg_path, encoding="utf-8") as f:
        tc = yaml.safe_load(f)
    return base, lab, tc


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(base, lab, tc, args):
    w = NuscWrapper(args.dataroot, base["version"],
                    require_image_files=base.get("require_image_files", True))
    fc = FeatureCache(os.path.join(base.get("cache_dir", "cache"), base["version"]))
    df = pd.read_parquet(f"data/labels_{base['version']}.parquet")
    split = split_for_names(load_split(base["splits_file"]),
                            {w.scene_name(t) for t in w.scene_tokens()})
    label_cfg = LabelConfig.from_dict(lab)
    tr = tc["train"]

    def mk(names, is_train):
        return ViewSelectionDataset(
            w, fc, df, label_cfg, names, seq_len=tr.get("seq_len", 4),
            risk_source=tr.get("risk_source", "gt"),
            noise_preset=tr.get("noise_preset", "clean"),
            train_risk_noise=(is_train and tr.get("risk_noise", False)),
            seed=base.get("seed", 42))

    return fc, mk(split["train"], True), mk(split["val"], False)


def to_device(batch, device):
    for k in ("img_feat", "can", "risk", "pad_mask", "soft_label",
              "attention_view", "primary_view", "layout"):
        batch[k] = batch[k].to(device)
    return batch


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    preds, tgts = [], []
    for batch in dl:
        batch = to_device(batch, device)
        out = model(batch)
        preds.append(out["attn_logits"].argmax(1).cpu().numpy())
        tgts.append(batch["attention_view"].cpu().numpy())
    p, t = np.concatenate(preds), np.concatenate(tgts)
    return {"top1": top1(p, t), "macro_f1": macro_f1(p, t)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_proposed.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--overfit", action="store_true", help="train셋 성능도 리포트(sanity)")
    ap.add_argument("--dataroot", default=None)
    args = ap.parse_args()

    base, lab, tc = load_cfgs(args.config)
    args.dataroot = args.dataroot or base["dataroot"]
    seed = base.get("seed", 42)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = tc["train"]
    epochs = args.epochs or tr.get("epochs", 100)

    fc, train_ds, val_ds = build_datasets(base, lab, tc, args)
    print(f"[학습] exp={tc.get('exp_name')} | train {len(train_ds)} / val {len(val_ds)} "
          f"| device={device} | epochs={epochs}")

    sampler = train_ds.make_sampler() if tr.get("balanced_sampler", True) else None
    train_dl = DataLoader(train_ds, batch_size=tr.get("batch_size", 16),
                          sampler=sampler, shuffle=(sampler is None),
                          collate_fn=collate, num_workers=base.get("num_workers", 4))
    val_dl = DataLoader(val_ds, batch_size=tr.get("batch_size", 16), shuffle=False,
                        collate_fn=collate, num_workers=base.get("num_workers", 4))

    mcfg = ModelConfig(feat_dim=fc.feat_dim, **{k: v for k, v in tc["model"].items()
                                                if k in ModelConfig.__dataclass_fields__})
    model = ViewSelector(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  모델 파라미터: {n_params/1e6:.2f}M")
    lw = LossWeights(**{k: v for k, v in tc.get("loss", {}).items()
                        if k in LossWeights.__dataclass_fields__})

    opt = torch.optim.AdamW(model.parameters(), lr=tr.get("lr", 1e-4),
                            weight_decay=tr.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    out_dir = os.path.join(base.get("results_dir", "results"),
                           tc.get("exp_name", "exp"), time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"base": base, "label": lab, "train_cfg": tc}, f, allow_unicode=True)

    best_f1, best_ep, patience = -1.0, -1, tr.get("patience", 10)
    history = []
    for ep in range(epochs):
        model.train()
        losses = []
        for batch in train_dl:
            batch = to_device(batch, device)
            out = model(batch)
            ld = compute_loss(out, batch, lw)
            opt.zero_grad()
            ld["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(ld["total"].item())
        sched.step()

        val = evaluate(model, val_dl, device)
        rec = {"epoch": ep, "loss": float(np.mean(losses)), **val}
        if args.overfit:
            rec["train_top1"] = evaluate(model, DataLoader(
                train_ds, batch_size=32, shuffle=False, collate_fn=collate),
                device)["top1"]
        history.append(rec)

        msg = (f"  ep{ep:02d} loss={rec['loss']:.3f} "
               f"val_top1={val['top1']:.3f} val_f1={val['macro_f1']:.3f}")
        if args.overfit:
            msg += f" train_top1={rec['train_top1']:.3f}"
        print(msg)

        if val["macro_f1"] > best_f1:
            best_f1, best_ep = val["macro_f1"], ep
            torch.save({"model": model.state_dict(), "mcfg": mcfg.__dict__,
                        "epoch": ep, "val": val}, os.path.join(out_dir, "best.ckpt"))
        elif ep - best_ep >= patience:
            print(f"  early stopping (best ep{best_ep}, f1={best_f1:.3f})")
            break

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_f1": best_f1, "best_epoch": best_ep, "history": history}, f, indent=2)
    print(f"\n✅ 학습 완료 — best val macro-F1={best_f1:.3f} (ep{best_ep})\n   → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
