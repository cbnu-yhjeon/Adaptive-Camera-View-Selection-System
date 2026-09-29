"""라벨 정책의 운영 곡선 — 표시량 vs 안전 (CLAUDE.md B.8).

보조 뷰 임계값 θ를 쓸어가며 (평균 보조 뷰 수, 위험 사건 포착률, 위험 슬롯 정밀도)를
측정한다. 라벨을 θ마다 다시 생성하면 수십 분이 걸리므로, 위험 벡터와 기동 적합도를
한 번만 만들어 두고 슬롯 배정만 다시 돌린다.

'실제 위험'은 label_function.is_hazard 하나만 쓴다 — 라벨·평가·골드셋이 같은 정의를
공유해야 재현율과 정밀도가 같은 모집단을 가리킨다.

사용:
    python scripts/02b_operating_curve.py --version v1.0-trainval
    python scripts/02b_operating_curve.py --thresholds 0.04,0.08,0.12,0.20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.contrib_cache import ContribCache, cache_path  # noqa: E402
from src.labeling.label_function import (  # noqa: E402
    LabelConfig, is_hazard, merge_aux_flicker, risk_vector_from_contribs,
    select_aux_views, smooth_risk,
)
from src.utils.constants import NUM_CAMERAS, maneuver_view_vector  # noqa: E402

DEFAULT_TH = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55]


def load_configs():
    d = os.path.join(os.path.dirname(__file__), "..", "configs")
    with open(os.path.join(d, "base.yaml"), encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(os.path.join(d, "label_v1.yaml"), encoding="utf-8") as f:
        lab = yaml.safe_load(f)
    return base, lab


def build_frames(df: pd.DataFrame, cache: ContribCache, cfg: LabelConfig):
    """scene별 시간순으로 평활한 위험 벡터 + 기동 적합도 + 위험 채널 집합."""
    frames = []
    for scene, g in df.groupby("scene_token", sort=False):
        g = g.sort_values("timestamp")
        toks = [t for t in g["sample_token"] if t in cache]
        if not toks:
            continue
        raw, haz, mvec = [], [], []
        for t in toks:
            contribs = cache.get(t)
            raw.append(risk_vector_from_contribs(contribs))
            ch = set()
            for c in contribs:
                if is_hazard(c, cfg):
                    ch.update(c["vis"])
            haz.append(ch)
        sub = g[g["sample_token"].isin(set(toks))].sort_values("timestamp")
        for mv in sub["maneuver"]:
            mvec.append(np.asarray(maneuver_view_vector(mv), dtype=np.float64))
        prim = sub["primary_view"].to_numpy().astype(int)
        frames.append(dict(risk=smooth_risk(raw, cfg.ema_alpha), m=mvec,
                           haz=haz, primary=prim))
    return frames


def evaluate(frames, cfg: LabelConfig, theta: float) -> dict:
    """한 임계값에서의 표시량·포착률·정밀도."""
    n_aux = n_frames = 0
    n_haz = n_caught = slot_on = slot_hit = 0
    zero = one = two = 0
    for f in frames:
        cfg.aux_threshold = theta
        aux_seq = [select_aux_views(r, m, int(p), cfg)
                   for r, m, p in zip(f["risk"], f["m"], f["primary"])]
        aux_seq = merge_aux_flicker(aux_seq)
        for aux, haz, p in zip(aux_seq, f["haz"], f["primary"]):
            n_frames += 1
            n_aux += len(aux)
            zero += len(aux) == 0
            one += len(aux) == 1
            two += len(aux) >= 2
            sel = set(aux) | {int(p)}
            for ch in haz:
                n_haz += 1
                n_caught += ch in sel
            for ch in aux:
                slot_on += 1
                slot_hit += ch in haz
    return dict(theta=theta, load=n_aux / max(n_frames, 1),
                recall=100 * n_caught / max(n_haz, 1),
                precision=100 * slot_hit / max(slot_on, 1),
                zero=100 * zero / max(n_frames, 1),
                one=100 * one / max(n_frames, 1),
                two=100 * two / max(n_frames, 1),
                n_hazard=n_haz, n_frames=n_frames)


def baseline_always_one(frames, cfg: LabelConfig) -> dict:
    """현행 비교군 — 매 프레임 primary 외 1장을 s(c) 최대 채널로 켠다."""
    n_haz = n_caught = slot_on = slot_hit = 0
    for f in frames:
        for r, m, haz, p in zip(f["risk"], f["m"], f["haz"], f["primary"]):
            s = r + cfg.lambda_m * np.asarray(m)
            s[int(p)] = -np.inf
            pick = int(np.argmax(s))
            sel = {pick, int(p)}
            for ch in haz:
                n_haz += 1
                n_caught += ch in sel
            slot_on += 1
            slot_hit += pick in haz
    return dict(theta=None, load=1.0, recall=100 * n_caught / max(n_haz, 1),
                precision=100 * slot_hit / max(slot_on, 1))


def main() -> int:
    base, lab = load_configs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=base["version"])
    ap.add_argument("--labels", default=None)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--out", default="results/operating_curve.json")
    args = ap.parse_args()

    cfg = LabelConfig.from_dict(lab)
    labels = args.labels or f"data/labels_{args.version}.parquet"
    df = pd.read_parquet(labels)
    cache = ContribCache(cache_path(os.path.join(base["cache_dir"], args.version), cfg))
    ths = ([float(x) for x in args.thresholds.split(",")]
           if args.thresholds else DEFAULT_TH)

    print(f"[운영곡선] {labels}  ({len(df):,} keyframe)")
    frames = build_frames(df, cache, cfg)
    adopted = float(lab["label"]["aux_threshold"])

    rows = [evaluate(frames, cfg, t) for t in sorted(set(ths + [adopted]))]
    b1 = baseline_always_one(frames, cfg)

    print(f"\n'실제 위험' 정의 TTC<{cfg.hazard_ttc:g}s AND DCA<{cfg.d_hazard:g}m"
          f" → {rows[0]['n_hazard']:,} 채널 / {rows[0]['n_frames']:,} keyframe")
    print(f"\n{'θ':>7s} {'평균 표시':>9s} {'포착률':>8s} {'정밀도':>8s} "
          f"{'0장':>6s} {'1장':>6s} {'2장':>6s}")
    for r in rows:
        mark = "  ← 채택" if abs(r["theta"] - adopted) < 1e-9 else ""
        print(f"{r['theta']:7.2f} {r['load']:8.2f}장 {r['recall']:7.1f}% "
              f"{r['precision']:7.1f}% {r['zero']:5.0f}% {r['one']:5.0f}% "
              f"{r['two']:5.0f}%{mark}")
    print(f"{'현행 1장':>7s} {b1['load']:8.2f}장 {b1['recall']:7.1f}% {b1['precision']:7.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(version=args.version, labels=labels,
                       hazard=dict(ttc=cfg.hazard_ttc, dca=cfg.d_hazard,
                                   channels=rows[0]["n_hazard"],
                                   frames=rows[0]["n_frames"]),
                       adopted=adopted, curve=rows, always_one=b1),
                  f, ensure_ascii=False, indent=2)
    print(f"\n✅ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
