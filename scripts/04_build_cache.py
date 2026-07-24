"""특징 캐시 빌드 + Dataset/로더 검증 (CLAUDE.md M3).

- MobileNetV3-small 특징을 keyframe·채널별로 캐시(재실행 시 스킵).
- train/val/test Dataset 조립, 배치 shape 검증.
- 1 epoch 순회 벤치마크 (캐시 사용 시 <1분 목표).

사용:
    python scripts/04_build_cache.py
    python scripts/04_build_cache.py --force        # 캐시 재생성
    python scripts/04_build_cache.py --risk_source noisy --noise medium
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import ViewSelectionDataset, collate  # noqa: E402
from src.data.feature_cache import FeatureCache, build_feature_cache  # noqa: E402
from src.data.nusc_wrapper import NuscWrapper  # noqa: E402
from src.data.splits import load_split, split_for_names  # noqa: E402
from src.labeling.label_function import LabelConfig  # noqa: E402
import pandas as pd  # noqa: E402


def load_cfg():
    d = os.path.join(os.path.dirname(__file__), "..", "configs")
    with open(os.path.join(d, "base.yaml"), encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(os.path.join(d, "label_v1.yaml"), encoding="utf-8") as f:
        lab = yaml.safe_load(f)
    return base, lab


def main() -> int:
    base, lab = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default=base["dataroot"])
    ap.add_argument("--version", default=base["version"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--risk_source", default="gt", choices=["gt", "noisy", "none"])
    ap.add_argument("--noise", default="clean", choices=["clean", "light", "medium", "heavy"])
    ap.add_argument("--batch_size", type=int, default=base.get("batch_size", 16))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = os.path.join(base.get("cache_dir", "cache"), args.version)
    labels_path = f"data/labels_{args.version}.parquet"

    w = NuscWrapper(args.dataroot, args.version,
                    require_image_files=base.get("require_image_files", True))

    # 1) 특징 캐시
    t0 = time.time()
    build_feature_cache(w, cache_dir, img_size=base.get("img_size", 224),
                        device=device, force=args.force)
    fc = FeatureCache(cache_dir)
    print(f"  캐시 로드 OK (feat_dim={fc.feat_dim}), {time.time()-t0:.1f}s\n")

    # 2) 분할 + Dataset
    df = pd.read_parquet(labels_path)
    split = split_for_names(load_split(base["splits_file"]),
                            {w.scene_name(t) for t in w.scene_tokens()})
    label_cfg = LabelConfig.from_dict(lab)
    print(f"분할: train {len(split['train'])} / val {len(split['val'])} / "
          f"test {len(split['test'])} scenes")

    sets = {}
    for name in ("train", "val", "test"):
        ds = ViewSelectionDataset(
            w, fc, df, label_cfg, split[name], seq_len=lab.get("seq_len", 4),
            risk_source=args.risk_source, noise_preset=args.noise,
            train_risk_noise=(name == "train" and args.risk_source == "noisy"))
        sets[name] = ds
        print(f"  {name}: {len(ds)} keyframe")

    # 3) 배치 shape 검증
    print("\n=== 배치 shape 검증 ===")
    dl = DataLoader(sets["train"], batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate, num_workers=0)
    batch = next(iter(dl))
    for k in ("img_feat", "can", "risk", "pad_mask", "soft_label"):
        print(f"  {k:<12} {tuple(batch[k].shape)}")
    for k in ("attention_view", "primary_view", "layout"):
        print(f"  {k:<14} {tuple(batch[k].shape)}  예시 {batch[k][:6].tolist()}")
    B = batch["img_feat"].shape[0]
    assert batch["img_feat"].shape == (B, 4, 6, fc.feat_dim)
    assert batch["can"].shape[-1] == 6
    assert batch["risk"].shape == (B, 4, 6)
    print("  ✅ shape 검증 통과")

    # 4) 1 epoch 벤치마크 (WeightedRandomSampler)
    print("\n=== 1 epoch 순회 벤치마크 (train, sampler) ===")
    sampler = sets["train"].make_sampler()
    dl = DataLoader(sets["train"], batch_size=args.batch_size, sampler=sampler,
                    collate_fn=collate, num_workers=base.get("num_workers", 4))
    t0 = time.time()
    n = sum(b["img_feat"].shape[0] for b in dl)
    dt = time.time() - t0
    print(f"  {n} 샘플, {dt:.1f}s  {'✅ <60s' if dt < 60 else '⚠️ 느림'}")
    print("\n✅ M3 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
