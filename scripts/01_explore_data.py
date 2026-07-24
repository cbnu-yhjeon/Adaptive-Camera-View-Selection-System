"""데이터 탐색 (CLAUDE.md M1).

임의 sample의 채널별 이미지 경로 + 박스 수 + CAN 스냅숏을 출력하고,
scene별 keyframe 수/기간/CAN 유무 요약 표를 만든다.

사용:
    python scripts/01_explore_data.py
    python scripts/01_explore_data.py --sample_index 0 --categories vehicle human
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.nusc_wrapper import NuscWrapper  # noqa: E402
from src.utils.constants import CAMERAS  # noqa: E402


def load_config():
    with open(os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"),
              encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default=cfg["dataroot"])
    ap.add_argument("--version", default=cfg["version"])
    ap.add_argument("--sample_index", type=int, default=0)
    ap.add_argument("--categories", nargs="*", default=["vehicle", "human", "cyclist"])
    args = ap.parse_args()

    print(f"[탐색] {args.version} @ {args.dataroot}\n")
    w = NuscWrapper(args.dataroot, args.version,
                    require_image_files=cfg.get("require_image_files", True))

    scenes = w.scene_tokens()
    print(f"사용 가능 scene: {len(scenes)}개\n")

    # --- scene별 요약 표 ---
    print(f"{'scene':<12}{'keyframes':>10}{'기간(s)':>9}{'CAN':>6}{'평균박스':>9}")
    print("-" * 46)
    total_kf = 0
    for st in scenes:
        toks = w.sample_tokens_in_scene(st)
        total_kf += len(toks)
        dur = w.sample_timestamp(toks[-1]) - w.sample_timestamp(toks[0])
        nbox = np.mean([len(w.get_boxes(t, args.categories, cfg_maxrange(cfg)))
                        for t in toks])
        print(f"{w.scene_name(st):<12}{len(toks):>10}{dur:>9.1f}"
              f"{'O' if w.has_can(st) else 'X':>6}{nbox:>9.1f}")
    print("-" * 46)
    print(f"{'합계':<12}{total_kf:>10} keyframe\n")

    # --- 특정 sample 상세 ---
    all_toks = w.all_sample_tokens()
    idx = max(0, min(args.sample_index, len(all_toks) - 1))
    st = all_toks[idx]
    print(f"=== sample[{idx}] 상세 (token={st[:8]}...) ===")
    print(f"scene: {w.scene_name(w.sample_scene_token(st))}, "
          f"기동추론용 ego yaw: {np.degrees(w.get_ego_yaw(st)):.1f}°, "
          f"ego 속도: {np.linalg.norm(w.get_ego_velocity(st)):.1f} m/s")

    print("\n6채널 이미지 + 채널별 투영 박스 수:")
    boxes = w.get_boxes(st, args.categories, cfg_maxrange(cfg))
    from src.utils.geometry import point_visible_in_camera
    ego = w.get_ego_pose(st)
    per_cam = {c: 0 for c in CAMERAS}
    for b in boxes:
        for cam in CAMERAS:
            cal = w.get_camera_calib(st, cam)
            if point_visible_in_camera(b["global_center"], ego["translation"],
                                       ego["rotation"], cal["translation"],
                                       cal["rotation"], cal["intrinsic"],
                                       cal["width"], cal["height"]):
                per_cam[cam] += 1
    paths = w.get_image_paths(st)
    for cam in CAMERAS:
        print(f"  {cam:<16} 박스 {per_cam[cam]:>2}개  {os.path.basename(paths[cam])}")
    print(f"  (필터 '{args.categories}' 통과 박스 총 {len(boxes)}개, "
          f"max_range={cfg_maxrange(cfg)}m)")

    # --- CAN 스냅숏 ---
    scene_token = w.sample_scene_token(st)
    print("\nCAN 스냅숏:")
    if w.has_can(scene_token):
        pose = w.get_can_messages(scene_token, "pose")
        steer = w.get_can_messages(scene_token, "steeranglefeedback")
        print(f"  pose 메시지 {len(pose)}개, steeranglefeedback {len(steer)}개")
        if pose:
            print(f"  pose[0] vel={pose[0].get('vel')}, "
                  f"pos={[round(x,1) for x in pose[0].get('pos', [])[:2]]}")
    else:
        print(f"  {w.scene_name(scene_token)}: CAN 없음(blacklist) — ego_pose 차분 사용")

    print("\n✅ M1 탐색 완료")
    return 0


def cfg_maxrange(cfg) -> float:
    return float(cfg.get("label", {}).get("max_range", 50.0))


if __name__ == "__main__":
    sys.exit(main())
