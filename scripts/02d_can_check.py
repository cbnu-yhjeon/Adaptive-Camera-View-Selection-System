"""CAN 입력 검증 — 속도와 방향지시등이 제대로 들어오는가.

골드셋 검수 중 "차량이 움직이는 것 같은데 0 km/h"라는 지적(2026-10-06)에서 시작했다.
모델이 받는 값(contrib 캐시의 can_feat)을 CAN 원자료에서 **독립적으로 다시 계산한 값**과
대조하고, 원자료 자체의 성질을 잰다.

  speed   세 소스의 일치 — keyframe ego_pose 차분 · CAN pose.vel(50 Hz) · 속도계(vehicle_monitor, 2 Hz).
          모델 입력 speed(=pose.vel)가 원값과 같은가. 정지 판정이 소스끼리 엇갈리는 프레임 수.
  signal  vehicle_monitor 지시등 값의 성질 — 메시지 주기, 점등 사이 소등 구간 분포(점멸 여부).
          직전 메시지 하나만 쓰던 옛 입력과 유지 판정(constants.SIGNAL_HOLD_S)의 차이.
          캐시의 지시등 특징이 독립 계산한 유지 판정과 같은가. 기동 라벨별 점등률(t0, 인과적).

출력: results/can_check.json

사용: python scripts/02d_can_check.py
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.contrib_cache import ContribCache, cache_path  # noqa: E402
from src.data.nusc_wrapper import NuscWrapper  # noqa: E402
from src.labeling.label_function import LabelConfig  # noqa: E402
from src.utils.config import load_cfgs  # noqa: E402
from src.utils.constants import CAN_FEATURES, CAN_NORM, SIGNAL_HOLD_S  # noqa: E402


def nearest(us: np.ndarray, u: int) -> int:
    j = int(np.searchsorted(us, u))
    if j == 0:
        return 0
    if j == len(us):
        return len(us) - 1
    return j if us[j] - u < u - us[j - 1] else j - 1


def agree(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    return dict(n=int(len(d)), mae=float(d.mean()), p95=float(np.percentile(d, 95)),
                max=float(d.max()), corr=float(np.corrcoef(a, b)[0, 1]),
                frac_over_1mps=float((d > 1.0).mean()))


def main() -> int:
    base, lab, _ = load_cfgs("configs/train_proposed.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=base["version"])
    ap.add_argument("--out", default="results/can_check.json")
    args = ap.parse_args()

    cfg = LabelConfig.from_dict(lab)
    w = NuscWrapper(base["dataroot"], args.version,
                    require_image_files=base.get("require_image_files", True))
    cc = ContribCache(cache_path(os.path.join(base["cache_dir"], args.version), cfg))
    df = pd.read_parquet(f"data/labels_{args.version}.parquet").set_index("sample_token")
    i_speed = CAN_FEATURES.index("speed")
    i_l, i_r = CAN_FEATURES.index("left_signal"), CAN_FEATURES.index("right_signal")
    hold_us = SIGNAL_HOLD_S * 1e6

    by_scene = collections.defaultdict(list)
    for i, t in enumerate(cc.tokens):
        by_scene[w.sample_scene_token(t)].append(i)

    rows, gaps = [], {"left": collections.Counter(), "right": collections.Counter()}
    periods = []
    for sc, idx in by_scene.items():
        has = w.has_can(sc)
        pose = w.get_can_messages(sc, "pose") if has else []
        vm = w.get_can_messages(sc, "vehicle_monitor") if has else []
        pu = np.array([m["utime"] for m in pose], dtype=np.int64)
        pv = np.array([m["vel"][0] for m in pose], dtype=float)
        vu = np.array([m["utime"] for m in vm], dtype=np.int64)
        vs = np.array([m["vehicle_speed"] for m in vm], dtype=float) / 3.6
        sig = {"left": np.array([int(m["left_signal"]) for m in vm]),
               "right": np.array([int(m["right_signal"]) for m in vm])}
        if len(vu) > 1:
            periods += list(np.diff(vu) / 1e6)
            for k, x in sig.items():
                on = np.flatnonzero(x)
                for a, b in zip(on[:-1], on[1:]):
                    gaps[k][int(b - a - 1)] += 1
        for i in idx:
            t = cc.tokens[i]
            ut = int(w.nusc.get("sample", t)["timestamp"])
            r = dict(token=t, scene=w.scene_name(sc), mask=float(cc.can_mask[i]),
                     v_pose=float(np.linalg.norm(w.get_ego_velocity(t))),
                     v_can=float(pv[nearest(pu, ut)]) if len(pu) else np.nan,
                     v_vm=float(vs[nearest(vu, ut)]) if len(vu) else np.nan,
                     f_speed=float(cc.can_feat[i, i_speed] * CAN_NORM["speed"]),
                     f_l=float(cc.can_feat[i, i_l]), f_r=float(cc.can_feat[i, i_r]))
            last = int(np.searchsorted(vu, ut, side="right")) - 1      # 시점 이하 마지막 메시지
            win = (vu > ut - hold_us) & (vu <= ut)
            for k in ("left", "right"):
                r[f"raw_{k}"] = float(sig[k][last]) if last >= 0 else 0.0
                r[f"held_{k}"] = float(sig[k][win].any()) if win.any() else 0.0
            r["maneuver"] = df.loc[t, "maneuver"] if t in df.index else None
            rows.append(r)
    R = pd.DataFrame(rows)
    can = R[R["mask"] > 0]

    # ---- 속도 ----
    va = can["v_can"].abs().to_numpy()
    speed = dict(
        n_frames=int(len(R)), n_with_can=int(len(can)),
        feature_vs_can_pose=dict(max_abs_diff=float(np.abs(can["f_speed"] - can["v_can"]).max())),
        feature_vs_pose_diff_no_can=(dict(n=int((R["mask"] == 0).sum()),
                                          max_abs_diff=float(np.abs(R.loc[R["mask"] == 0, "f_speed"]
                                                                    - R.loc[R["mask"] == 0, "v_pose"]).max()))
                                     if (R["mask"] == 0).any() else None),
        pose_diff_vs_can_pose=agree(can["v_pose"].to_numpy(), va),
        speedometer_vs_can_pose=agree(can["v_vm"].dropna().to_numpy(),
                                      can.loc[can["v_vm"].notna(), "v_can"].abs().to_numpy()),
        stopped_disagree=dict(
            pose_stop_can_moving=int(((can["v_pose"] < 0.3) & (va > 1.0)).sum()),
            pose_moving_can_stop=int(((can["v_pose"] > 1.0) & (va < 0.3)).sum())),
        reverse_frames_can=int((can["v_can"] < -0.3).sum()),
        stopped_frames=dict(pose_lt_0p5=float((R["v_pose"] < 0.5).mean()),
                            can_lt_0p5=float((va < 0.5).mean())),
    )

    # ---- 지시등 ----
    def gap_table(c):
        tot = sum(c.values())
        return {"0": c[0] / tot, "1": c[1] / tot, "2": c[2] / tot,
                "3+": sum(v for g, v in c.items() if g >= 3) / tot, "n": tot}
    held_any = (can["held_left"] + can["held_right"]) > 0
    raw_any = (can["raw_left"] + can["raw_right"]) > 0
    by_man = {}
    for m, g in can.groupby("maneuver"):
        by_man[m] = dict(n=int(len(g)),
                         held_left=float(g["held_left"].mean()), held_right=float(g["held_right"].mean()),
                         raw_left=float(g["raw_left"].mean()), raw_right=float(g["raw_right"].mean()))
    p = np.array(periods)
    signal = dict(
        hold_s=SIGNAL_HOLD_S,
        period_s=dict(median=float(np.median(p)), p5=float(np.percentile(p, 5)),
                      p95=float(np.percentile(p, 95)), max=float(p.max())),
        off_samples_between_on=dict(left=gap_table(gaps["left"]), right=gap_table(gaps["right"])),
        keyframes_held_on=float(held_any.mean()),
        keyframes_raw_on=float(raw_any.mean()),
        missed_by_latest_only=float((held_any & ~raw_any).sum() / max(1, held_any.sum())),
        hazard_lights=float(((can["held_left"] > 0) & (can["held_right"] > 0)).mean()),
        cache_matches_held=float(((can["f_l"] == can["held_left"]) & (can["f_r"] == can["held_right"])).mean()),
        by_maneuver=by_man,
    )
    out = dict(version=args.version, speed=speed, signal=signal)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    s, g = speed, signal
    print(f"[속도] keyframe {s['n_frames']} (CAN {s['n_with_can']})")
    print(f"  모델 입력 speed vs CAN pose.vel   최대 차 {s['feature_vs_can_pose']['max_abs_diff']:.2e} m/s")
    a = s["pose_diff_vs_can_pose"]
    print(f"  ego_pose 차분 vs CAN pose.vel     MAE {a['mae']:.3f}  p95 {a['p95']:.3f}  상관 {a['corr']:.4f}  >1m/s {100*a['frac_over_1mps']:.2f}%")
    a = s["speedometer_vs_can_pose"]
    print(f"  속도계 vs CAN pose.vel            MAE {a['mae']:.3f}  p95 {a['p95']:.3f}  상관 {a['corr']:.4f}  >1m/s {100*a['frac_over_1mps']:.2f}%")
    print(f"  정지 판정 엇갈림  pose정지·CAN주행 {s['stopped_disagree']['pose_stop_can_moving']}  "
          f"pose주행·CAN정지 {s['stopped_disagree']['pose_moving_can_stop']}  후진 {s['reverse_frames_can']}")
    print(f"[지시등] 주기 median {g['period_s']['median']:.3f}s · 유지 {g['hold_s']}s")
    for k in ("left", "right"):
        t = g["off_samples_between_on"][k]
        print(f"  {k:5s} 점등 사이 소등 0:{100*t['0']:.1f}% 1:{100*t['1']:.1f}% 2:{100*t['2']:.1f}% 3+:{100*t['3+']:.1f}%")
    print(f"  점등 keyframe  유지 {100*g['keyframes_held_on']:.1f}%  직전값 {100*g['keyframes_raw_on']:.1f}%  "
          f"→ 직전값이 놓친 비율 {100*g['missed_by_latest_only']:.1f}%")
    print(f"  캐시 특징 = 독립 유지 판정  {100*g['cache_matches_held']:.2f}%")
    for m, v in sorted(g["by_maneuver"].items(), key=lambda x: -x[1]["n"]):
        print(f"  {m:18s} n={v['n']:5d}  좌 {100*v['held_left']:5.1f}% (직전값 {100*v['raw_left']:5.1f}%)  "
              f"우 {100*v['held_right']:5.1f}% (직전값 {100*v['raw_right']:5.1f}%)")
    print(f"\n✅ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
