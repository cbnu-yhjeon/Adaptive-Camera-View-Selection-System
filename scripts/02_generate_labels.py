"""의사 라벨 생성 → parquet (CLAUDE.md M2).

사용:
    python scripts/02_generate_labels.py                      # base+label_v1
    python scripts/02_generate_labels.py --limit_scenes 2     # 드라이런
    python scripts/02_generate_labels.py --out data/labels_v1.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.nusc_wrapper import NuscWrapper  # noqa: E402
from src.labeling.label_function import LabelConfig, generate_all_labels  # noqa: E402
from src.utils.constants import CAMERAS, MANEUVERS  # noqa: E402


def load_configs():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    with open(os.path.join(base_dir, "base.yaml"), encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(os.path.join(base_dir, "label_v1.yaml"), encoding="utf-8") as f:
        lab = yaml.safe_load(f)
    return base, lab


def main() -> int:
    base, lab = load_configs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default=base["dataroot"])
    ap.add_argument("--version", default=base["version"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit_scenes", type=int, default=None)
    args = ap.parse_args()

    cfg = LabelConfig.from_dict(lab)
    out = args.out or f"data/labels_{args.version}.parquet"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(f"[라벨생성] {args.version} @ {args.dataroot}")
    print(f"  categories={cfg.categories}, tau={cfg.tau}, lambda={cfg.lambda_m}")
    w = NuscWrapper(args.dataroot, args.version,
                    require_image_files=base.get("require_image_files", True))

    t0 = time.time()
    rows = generate_all_labels(w, cfg, limit_scenes=args.limit_scenes)
    df = pd.DataFrame(rows)
    # 컬럼 순서 정리
    cols = (["sample_token", "scene_token", "timestamp"]
            + [f"s_{c}" for c in range(6)]
            + ["primary_view", "attention_view"]
            + [f"aux_{c}" for c in range(6)] + ["n_aux"]
            + [f"soft_{c}" for c in range(6)]
            + ["maneuver", "layout", "n_objects", "min_ttc"])
    df = df[cols]
    df.to_parquet(out, index=False)
    dt = time.time() - t0
    print(f"  → {len(df)} keyframe, {dt:.1f}s, 저장: {out}\n")

    _report(df)
    return 0


def _dist(df: pd.DataFrame, col: str, title: str):
    n = len(df)
    print(f"=== {title} ({col}) ===")
    vc = df[col].value_counts().sort_index()
    top_pct = 0.0
    for c in range(6):
        cnt = int(vc.get(c, 0))
        pct = 100 * cnt / n
        top_pct = max(top_pct, pct)
        print(f"  [{c}] {CAMERAS[c]:<16} {cnt:>4} ({pct:4.1f}%) {'#' * int(pct / 2)}")
    return top_pct


def _report(df: pd.DataFrame):
    """라벨 분포 리포트 (M2 ✅ 조건: 특정 채널 편중 여부 확인)."""
    n = len(df)
    # primary는 진행방향(결정적)이라 참고용, attention이 학습 타깃(적응형)
    _dist(df, "primary_view", "진행방향 뷰 분포")
    print()
    top = _dist(df, "attention_view", "적응형 보조 뷰 분포")
    print(f"\n  보조 뷰 최대 채널 비율: {top:.1f}%  "
          f"{'⚠️ 85% 초과 — tau↓/lambda↑ 재튜닝 검토' if top > 85 else '✅ 편중 아님'}")

    print("\n=== 기동 분포 ===")
    mvc = df["maneuver"].value_counts()
    for m in MANEUVERS:
        cnt = int(mvc.get(m, 0))
        print(f"  {m:<20} {cnt:>4} ({100*cnt/n:4.1f}%)")

    # 라벨 flicker rate (main_view 전환 중 3키프레임 내 재전환 비율)
    fr = _flicker_rate(df)
    print(f"\n=== 라벨 flicker rate (스무딩 후): {fr:.3f} ===")
    valid_ttc = df[df["min_ttc"] >= 0]["min_ttc"]
    if len(valid_ttc):
        print(f"min_ttc 중앙값: {valid_ttc.median():.1f}s "
              f"(유효 {len(valid_ttc)}/{n} keyframe)")


def _flicker_rate(df: pd.DataFrame) -> float:
    """scene별 시간순 attention_view 전환 후 3키프레임 내 재전환 비율."""
    transitions, flickers = 0, 0
    for _, g in df.groupby("scene_token"):
        mv = g.sort_values("timestamp")["attention_view"].tolist()
        for i in range(1, len(mv)):
            if mv[i] != mv[i - 1]:
                transitions += 1
                for k in range(i + 1, min(i + 4, len(mv))):
                    if mv[k] != mv[i]:
                        flickers += 1
                        break
    return flickers / transitions if transitions else 0.0


if __name__ == "__main__":
    sys.exit(main())
