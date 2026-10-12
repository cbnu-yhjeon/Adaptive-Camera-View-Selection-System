"""학습 (CLAUDE.md M4).

config 로드 → 학습 → best ckpt + metrics.json + config 사본 저장.
early stopping(val macro-AP), cosine LR, WeightedRandomSampler, 재현 seed.

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

from src.utils.config import load_cfgs  # noqa: E402

from src.data import contrib_cache  # noqa: E402
from src.data.dataset import ViewSelectionDataset, collate  # noqa: E402
from src.data.feature_cache import FeatureCache  # noqa: E402
from src.data.nusc_wrapper import NuscWrapper  # noqa: E402
from src.data.splits import load_split, split_for_names  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    macro_ap, macro_f1, mean_aux, multilabel_f1, threshold_for_load, top1,
)
from src.labeling.label_function import LabelConfig  # noqa: E402
from src.models.proposed import (  # noqa: E402
    LossWeights, ModelConfig, ViewSelector, compute_loss, select_views,
)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(base, lab, tc, args):
    w = NuscWrapper(args.dataroot, base["version"],
                    require_image_files=base.get("require_image_files", True))
    cache_dir = os.path.join(base.get("cache_dir", "cache"), base["version"])
    fc = FeatureCache(cache_dir)
    df = pd.read_parquet(f"data/labels_{base['version']}.parquet")
    split = split_for_names(load_split(args.splits),
                            {w.scene_name(t) for t in w.scene_tokens()})
    label_cfg = LabelConfig.from_dict(lab)
    tr = tc["train"]
    # 위험 기여·CAN은 실행마다 다시 계산하지 않고 캐시에서 읽는다(없으면 1회 생성).
    cc = contrib_cache.load_or_build(w, w.all_sample_tokens(), label_cfg, cache_dir)

    def mk(names, is_train):
        return ViewSelectionDataset(
            w, fc, df, label_cfg, names, seq_len=tr.get("seq_len", 4),
            risk_source=tr.get("risk_source", "gt"),
            noise_preset=tr.get("noise_preset", "clean"),
            train_risk_noise=(is_train and tr.get("risk_noise", False)),
            seed=base.get("seed", 42), contrib_cache=cc,
            train_noise_mix=tuple(tr.get("noise_mix", ["light", "medium"])))

    return fc, mk(split["train"], True), mk(split["val"], False)


def to_device(batch, device):
    for k in ("img_feat", "can", "risk", "pad_mask", "soft_label",
              "aux_target", "aux_mask",
              "attention_view", "primary_view", "layout"):
        batch[k] = batch[k].to(device)
    return batch


@torch.no_grad()
def evaluate(model, dl, device, aux_max: int = 2):
    """다중 선택 지표가 주(主)이고, 단일 선택 Top-1은 베이스라인 비교용으로 병기한다.

    모델 선택은 macro-AP(임계값 무관)로 한다. F1·표시량은 **라벨과 같은 표시량**이
    되도록 임계값을 맞춘 뒤 읽는다 — 정밀도·재현율이 표시량에 좌우되므로 고정
    임계값으로는 비교가 성립하지 않는다.
    """
    model.eval()
    probs, aux_t, aux_m, arg, single_t = [], [], [], [], []
    for batch in dl:
        batch = to_device(batch, device)
        out = model(batch)
        logits = out["attn_logits"]
        probs.append(torch.sigmoid(logits).cpu().numpy())
        aux_t.append(batch["aux_target"].cpu().numpy())
        aux_m.append(batch["aux_mask"].cpu().numpy())
        # 단일 선택: primary를 제외한 argmax
        masked = logits.masked_fill(
            torch.nn.functional.one_hot(batch["primary_view"], logits.shape[1]).bool(),
            float("-inf"))
        arg.append(masked.argmax(1).cpu().numpy())
        single_t.append(batch["attention_view"].cpu().numpy())
    P = np.concatenate(probs)
    T = np.concatenate(aux_t)
    M = np.concatenate(aux_m).astype(bool)
    a, s1 = np.concatenate(arg), np.concatenate(single_t)

    target_load = float(T.sum(axis=1).mean())
    th = threshold_for_load(P, target_load, M, aux_max)
    keep = P >= th
    keep[~M] = False
    if aux_max is not None:
        order = np.argsort(-np.where(M, P, -1.0), axis=1)
        np.put_along_axis(keep, order[:, aux_max:], False, axis=1)
    return {"ap": macro_ap(P, T, M), "ml_f1": multilabel_f1(keep, T, M),
            "mean_aux": mean_aux(keep), "label_aux": target_load, "thresh": th,
            "top1": top1(a, s1), "macro_f1": macro_f1(a, s1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_proposed.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--overfit", action="store_true", help="train셋 성능도 리포트(sanity)")
    ap.add_argument("--dataroot", default=None)
    # mini 스모크 테스트는 trainval 분할에서 val이 비므로 mini 전용 분할을 쓴다.
    ap.add_argument("--splits", default=None)
    ap.add_argument("--version", default=None, help="base.yaml의 version 덮어쓰기")
    ap.add_argument("--limit_samples", type=int, default=None,
                    help="드라이런 — train·val을 각각 N개로 줄여 파이프라인만 확인(명세 A.3)")
    args = ap.parse_args()

    base, lab, tc = load_cfgs(args.config)
    if args.version:
        base["version"] = args.version
    label_cfg = LabelConfig.from_dict(lab)   # 추론 시 보조 뷰 상한 참조
    args.dataroot = args.dataroot or base["dataroot"]
    args.splits = args.splits or base["splits_file"]
    seed = base.get("seed", 42)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = tc["train"]
    epochs = args.epochs or tr.get("epochs", 100)

    fc, train_ds, val_ds = build_datasets(base, lab, tc, args)
    if args.limit_samples:
        # 결과는 '{exp}_dryrun'에 둔다 — 같은 폴더에 두면 평가가 최신 실행으로 드라이런을 집는다
        tc["exp_name"] = tc.get("exp_name", "exp") + "_dryrun"
        # 여러 scene이 섞이도록 균등 간격으로 고른다(앞에서 N개면 scene 한두 개뿐이다)
        for ds in (train_ds, val_ds):
            step = max(1, len(ds.flat) // args.limit_samples)
            ds.flat = ds.flat[::step][:args.limit_samples]
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

    best_ap, best_ep, patience = -1.0, -1, tr.get("patience", 10)
    history = []
    for ep in range(epochs):
        model.train()
        # epoch마다 다른 노이즈 실현을 뽑게 한다(고정 노이즈 암기 방지 — 강건성의 전제).
        # DataLoader 이터레이터 생성 전에 호출해야 워커에 반영된다.
        train_ds.set_epoch(ep)
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

        val = evaluate(model, val_dl, device, aux_max=label_cfg.aux_max)
        rec = {"epoch": ep, "loss": float(np.mean(losses)), **val}
        if args.overfit:
            rec["train_ap"] = evaluate(model, DataLoader(
                train_ds, batch_size=32, shuffle=False, collate_fn=collate),
                device, aux_max=label_cfg.aux_max)["ap"]
        history.append(rec)

        msg = (f"  ep{ep:02d} loss={rec['loss']:.3f} "
               f"val_AP={val['ap']:.3f} F1={val['ml_f1']:.3f} "
               f"보조={val['mean_aux']:.2f}/{val['label_aux']:.2f}장 "
               f"(τ={val['thresh']:.2f}) | top1={val['top1']:.3f}")
        if args.overfit:
            msg += f" train_AP={rec['train_ap']:.3f}"
        print(msg)

        if val["ap"] > best_ap:
            best_ap, best_ep = val["ap"], ep
            torch.save({"model": model.state_dict(), "mcfg": mcfg.__dict__,
                        "epoch": ep, "val": val}, os.path.join(out_dir, "best.ckpt"))
        elif ep - best_ep >= patience:
            print(f"  early stopping (best ep{best_ep}, AP={best_ap:.3f})")
            break

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        # 2026-10-07 이전 실행은 같은 값을 'best_f1'이라는 이름으로 저장했다(선택 기준을 F1에서
        # AP로 바꿀 때 이름이 남았다). 읽는 쪽은 두 키를 모두 받는다.
        json.dump({"best_val_ap": best_ap, "best_epoch": best_ep, "history": history}, f, indent=2)
    print(f"\n✅ 학습 완료 — best val macro-AP={best_ap:.3f} (ep{best_ep})\n   → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
