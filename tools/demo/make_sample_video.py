"""업로드 시험용 영상 만들기 — nuScenes scene을 카메라별 mp4 6개 + 2×3 모자이크 mp4 + CAN CSV로 내보낸다.

실제 주행 영상이 없을 때 업로드 경로를 시험하고, 직접 경로(export.py)와 같은 판단이 나오는지
대조하는 데 쓴다. sweeps가 있으면 12 Hz, 없으면 keyframe 2 Hz 영상이 된다(--keyframes면 항상 2 Hz —
영상 프레임이 keyframe과 정확히 겹쳐 업로드 경로 검증에 쓴다).

    python tools/demo/make_sample_video.py --scene scene-0003 --version v1.0-test
    python tools/demo/make_sample_video.py --scene scene-0002 --version v1.0-trainval --keyframes

출력: results/demo/samples/{scene}[_kf]/  cam_{1..6}_{카메라}.mp4 · mosaic.mp4 · can.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import core  # noqa: E402
from core import GRID, ROOT  # noqa: E402
from export import play_frames  # noqa: E402

from src.utils.constants import CAMERAS  # noqa: E402


def can_rows(w, scene_token, t0_us: int, t1_us: int, hz: float = 10.0) -> list[dict]:
    """CAN 원 메시지 → hz 간격 행. 지시등은 램프 점멸 원값 그대로(유지 규칙은 읽는 쪽이 적용)."""
    msgs = {k: w.get_can_messages(scene_token, k) for k in ("pose", "steeranglefeedback", "vehicle_monitor")}
    if not msgs["pose"]:
        return []

    def latest(lst, ut):
        ts = [m["utime"] for m in lst]
        j = int(np.searchsorted(ts, ut, side="right")) - 1
        return lst[j] if j >= 0 else None

    rows = []
    for ut in np.arange(t0_us, t1_us + 1, 1e6 / hz):
        # 장면 시작 직후엔 시점 이전 메시지가 없을 수 있다 — 첫 메시지로 채운다
        p, s, v = (latest(msgs[k], ut) or (msgs[k][0] if msgs[k] else None)
                   for k in ("pose", "steeranglefeedback", "vehicle_monitor"))
        rows.append(dict(t=round((ut - t0_us) / 1e6, 3), speed=round(p["vel"][0], 3),
                         steering=round(s["value"], 2) if s else 0.0,
                         accel_x=round(p["accel"][0], 3), accel_y=round(p["accel"][1], 3),
                         yaw_rate=round(p["rotation_rate"][2], 4),
                         left_signal=int(v["left_signal"]) if v else 0,
                         right_signal=int(v["right_signal"]) if v else 0))
    return rows


def main() -> int:
    import cv2
    from src.data.nusc_wrapper import NuscWrapper

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--version", default="v1.0-test")
    ap.add_argument("--keyframes", action="store_true", help="keyframe만으로 2 Hz 영상")
    ap.add_argument("--width", type=int, default=1600, help="카메라별 영상 폭 (모자이크 타일은 640)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = core.load_cfg()
    w = NuscWrapper(cfg["dataroot"], args.version, require_image_files=True)
    st = next((s for s in w.scene_tokens() if w.scene_name(s) == args.scene), None)
    if st is None:
        print(f"{args.scene}가 {args.version}에 없거나 이미지가 없습니다", file=sys.stderr)
        return 1
    toks = w.sample_tokens_in_scene(st)
    if args.keyframes:
        frames = [dict(t=round(w.sample_timestamp(t) - w.sample_timestamp(toks[0]), 3),
                       img=["img/" + w.nusc.get("sample_data", w.nusc.get("sample", t)["data"][c])["filename"]
                            for c in CAMERAS]) for t in toks]
        fps = 2.0
    else:
        frames, sweeps = play_frames(w.nusc, w.dataroot, toks)
        fps = 12.0 if sweeps else 2.0
    out = args.out or os.path.join(cfg["out_dir"], "samples", args.scene + ("_kf" if args.keyframes else ""))
    os.makedirs(out, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cw, ch_ = args.width, int(round(args.width * 9 / 16))
    tw, th = 640, 360
    writers = {ci: cv2.VideoWriter(os.path.join(out, f"cam_{GRID.index(CAMERAS[ci]) + 1}_{CAMERAS[ci]}.mp4"),
                                   fourcc, fps, (cw, ch_)) for ci in range(len(CAMERAS))}
    mosaic = cv2.VideoWriter(os.path.join(out, "mosaic.mp4"), fourcc, fps, (3 * tw, 2 * th))
    # 영상은 고정 프레임률이다 — 원래 시각 대신 1/fps 간격으로 놓인다(12 Hz sweeps는 거의 등간격).
    for f in frames:
        tiles = {}
        for ci in range(len(CAMERAS)):
            img = cv2.imread(os.path.join(w.dataroot, f["img"][ci][len("img/"):]))
            writers[ci].write(cv2.resize(img, (cw, ch_), interpolation=cv2.INTER_AREA))
            tiles[CAMERAS[ci]] = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        top = np.hstack([tiles[c] for c in GRID[:3]])
        bot = np.hstack([tiles[c] for c in GRID[3:]])
        mosaic.write(np.vstack([top, bot]))
    for v in [*writers.values(), mosaic]:
        v.release()

    s0 = w.nusc.get("sample", toks[0])["timestamp"]
    s1 = w.nusc.get("sample", toks[-1])["timestamp"]
    rows = can_rows(w, st, s0, s1) if w.has_can(st) else []
    if rows:
        with open(os.path.join(out, "can.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    print(f"✅ {os.path.relpath(out, ROOT)} — {len(frames)}프레임 {fps:g} fps, CAN {len(rows)}행")
    return 0


if __name__ == "__main__":
    sys.exit(main())
