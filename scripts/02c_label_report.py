"""라벨 리포트 — 논문·계획서가 인용하는 라벨 수치를 한 곳에서 산출한다 (원칙 4).

문서에 수치를 손으로 옮기면 정의가 바뀔 때 갱신이 빠진다. 실제로 '실제 위험'의
정의를 3초에서 5초로 바꾼 뒤에도 문서에는 3초 시절 값(1,879채널 / 5.3%)이 남아
있었다. 문서 수치는 이 스크립트의 출력(results/label_report.json)에서만 가져온다.

산출 항목
  hazard        '실제 위험' 채널·프레임 수와 분할별 프레임 수
  alt_defs      위험 정의(TTC·DCA)를 바꿨을 때의 비율 — 정의 선택의 근거
  labels        평균 보조 뷰 수, 0/1/2장 비율, 기동 분포, 보조 뷰 집합 flicker
  goldset       골드셋 300장의 자연 추출 vs 층화 추출 시 재현율 신뢰구간
  signal        기동 라벨과 방향지시등(CAN)의 일치도 — 독립 신호에 의한 라벨 검증

사용:
    python scripts/02c_label_report.py --version v1.0-trainval
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
from src.data.nusc_wrapper import NuscWrapper  # noqa: E402
from src.data.splits import load_split  # noqa: E402
from src.eval.metrics import flicker_rate  # noqa: E402
from src.labeling.label_function import LabelConfig, is_hazard  # noqa: E402
from src.utils.constants import MANEUVERS, NUM_CAMERAS  # noqa: E402


def load_configs():
    d = os.path.join(os.path.dirname(__file__), "..", "configs")
    with open(os.path.join(d, "base.yaml"), encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(os.path.join(d, "label_v1.yaml"), encoding="utf-8") as f:
        lab = yaml.safe_load(f)
    return base, lab


def hazard_channels(cache, token, ttc_max, dca_max):
    ch = set()
    for c in cache.get(token):
        t = c["ttc"]
        if np.isfinite(t) and 0.0 <= t < ttc_max and c["dca"] < dca_max:
            ch.update(c["vis"])
    return ch


def recall_ci(n: int, p: float = 0.9) -> float:
    """재현율 p를 n건으로 추정할 때 95% 신뢰구간 반폭(%p)."""
    return 100 * 1.96 * np.sqrt(p * (1 - p) / max(n, 1))


def main() -> int:
    base, lab = load_configs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=base["version"])
    ap.add_argument("--out", default="results/label_report.json")
    ap.add_argument("--no_signal", action="store_true", help="지시등 대조 생략(느림)")
    args = ap.parse_args()

    cfg = LabelConfig.from_dict(lab)
    df = pd.read_parquet(f"data/labels_{args.version}.parquet")
    cache = ContribCache(cache_path(os.path.join(base["cache_dir"], args.version), cfg))
    w = NuscWrapper(base["dataroot"], args.version,
                    require_image_files=base.get("require_image_files", True))
    split = load_split(base["splits_file"])
    name_of = {t: w.scene_name(t) for t in df["scene_token"].unique()}
    df["scene_name"] = df["scene_token"].map(name_of)
    df["split"] = "other"
    for k in ("train", "val", "test"):
        df.loc[df["scene_name"].isin(set(split[k])), "split"] = k
    toks = [t for t in df["sample_token"] if t in cache]
    df = df[df["sample_token"].isin(set(toks))].reset_index(drop=True)
    rep: dict = {"version": args.version, "n_keyframes": int(len(df))}

    # ---- 1) '실제 위험' ----
    haz = [hazard_channels(cache, t, cfg.hazard_ttc, cfg.d_hazard) for t in df["sample_token"]]
    has = np.array([len(h) > 0 for h in haz])
    rep["hazard"] = dict(
        ttc=cfg.hazard_ttc, dca=cfg.d_hazard,
        channels=int(sum(len(h) for h in haz)), frames=int(has.sum()),
        frame_pct=round(100 * has.mean(), 1),
        by_split={k: int(has[(df["split"] == k).to_numpy()].sum()) for k in ("train", "val", "test")},
        n_by_split={k: int((df["split"] == k).sum()) for k in ("train", "val", "test")})

    # ---- 2) 정의를 바꿨을 때 ----
    test_mask = (df["split"] == "test").to_numpy()
    alt = []
    for T in (3.0, 5.0):
        for D in (2.0, 3.0, 4.0):
            h = np.array([len(hazard_channels(cache, t, T, D)) > 0 for t in df["sample_token"]])
            alt.append(dict(ttc=T, dca=D, frame_pct=round(100 * h.mean(), 1),
                            test_frames=int(h[test_mask].sum())))
    rep["alt_defs"] = alt

    # ---- 3) 라벨 분포 ----
    aux = np.stack([df[f"aux_{c}"].to_numpy().astype(int) for c in range(NUM_CAMERAS)], 1)
    n_aux = aux.sum(1)
    seqs = {}
    for sc, g in df.groupby("scene_token"):
        g = g.sort_values("timestamp")
        ids = [int("".join(str(int(g[f"aux_{c}"].iloc[i])) for c in range(NUM_CAMERAS)), 2)
               for i in range(len(g))]
        seqs[sc] = ids
    rep["labels"] = dict(
        mean_aux=round(float(n_aux.mean()), 3),
        aux_share={str(k): round(100 * float((n_aux == k).mean()), 1) for k in (0, 1, 2)},
        maneuver={m: int((df["maneuver"] == m).sum()) for m in MANEUVERS},
        max_channel_pct=round(100 * float(aux.sum(0).max() / max(aux.sum(), 1)), 1),
        flicker_aux_set=round(flicker_rate(seqs, window=3), 3))

    # ---- 4) 골드셋 신뢰구간 ----
    p_test = has[test_mask].mean()
    nat = 300 * p_test
    rep["goldset"] = dict(
        test_hazard_pct=round(100 * float(p_test), 1),
        natural_300_hazard=round(float(nat), 1),
        natural_ci_pp=round(recall_ci(int(round(nat))), 1),
        stratified_100_ci_pp=round(recall_ci(100), 1))

    # ---- 5) 방향지시등 대조 (라벨 판정과 같은 창) ----
    if not args.no_signal:
        back = cfg.maneuver_window_back_s
        fwd = cfg.maneuver_window_fwd_s
        rows = []
        for _, r in df.iterrows():
            msgs = w.get_can_messages(r["scene_token"], "vehicle_monitor")
            if not msgs:
                continue
            ut = int(w.nusc.get("sample", r["sample_token"])["timestamp"])
            sel = [m for m in msgs if ut - back * 1e6 <= m["utime"] <= ut + fwd * 1e6]
            if not sel:
                continue
            L = any(m["left_signal"] for m in sel)
            R = any(m["right_signal"] for m in sel)
            rows.append((r["maneuver"], "L" if L and not R else "R" if R and not L
                         else "LR" if L and R else "-"))
        sd = pd.DataFrame(rows, columns=["mv", "sig"])
        sig = {}
        for m, g in sd.groupby("mv"):
            same = "L" if m.endswith("LEFT") else "R" if m.endswith("RIGHT") else None
            opp = {"L": "R", "R": "L"}.get(same)
            sig[m] = dict(n=int(len(g)),
                          none_pct=round(100 * float((g.sig == "-").mean()), 1),
                          any_pct=round(100 * float((g.sig != "-").mean()), 1),
                          same_pct=(round(100 * float((g.sig == same).mean()), 1) if same else None),
                          opposite_pct=(round(100 * float((g.sig == opp).mean()), 1) if opp else None))
        rep["signal"] = dict(window=[-back, fwd], by_maneuver=sig)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
