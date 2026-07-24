"""nuScenes 데이터 배치 검증 (다운로드/압축해제 직후 실행).

CLAUDE.md B.2 로드 확인 단계를 스크립트화한 것.
데이터가 올바른 위치에 풀렸는지, devkit과 CAN API가 정상 로드되는지,
6채널 이미지·GT 박스·CAN 신호에 접근 가능한지 점검한다.

사용:
    python scripts/00_check_data.py
    python scripts/00_check_data.py --dataroot /경로 --version v1.0-mini
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]


def _load_base_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    base = _load_base_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default=base["dataroot"])
    ap.add_argument("--version", default=base["version"])
    args = ap.parse_args()

    root = os.path.abspath(args.dataroot)
    print(f"[검증] dataroot = {root}")
    print(f"[검증] version  = {args.version}\n")

    # --- 1. 폴더 구조 ---
    ok = True
    expected = ["samples", "sweeps", "maps", args.version, "can_bus"]
    for d in expected:
        p = os.path.join(root, d)
        exists = os.path.isdir(p)
        mark = "OK " if exists else "없음"
        print(f"  [{mark}] {d}/")
        if d in ("samples", args.version) and not exists:
            ok = False  # 필수
        if d == "can_bus" and not exists:
            print("       ⚠️  can_bus 없음 — CAN bus expansion(can_bus.zip) 압축 확인")
    if not ok:
        print("\n❌ 필수 폴더 누락. 압축 해제 위치를 확인하세요.")
        return 1

    # --- 2. devkit 로드 ---
    print("\n[검증] NuScenes devkit 로드 중...")
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    except ImportError as e:
        print(f"❌ devkit import 실패: {e}\n   pip install nuscenes-devkit")
        return 1

    nusc = NuScenes(version=args.version, dataroot=root, verbose=False)
    print(f"  OK  scene 수: {len(nusc.scene)}")
    print(f"  OK  sample(keyframe) 수: {len(nusc.sample)}")

    # --- 3. 첫 sample에서 6채널 이미지 + GT 박스 접근 ---
    sample = nusc.sample[0]
    print("\n[검증] 첫 keyframe 6채널 이미지 경로:")
    missing_img = 0
    for cam in CAMERAS:
        sd = nusc.get("sample_data", sample["data"][cam])
        path = os.path.join(root, sd["filename"])
        exists = os.path.isfile(path)
        if not exists:
            missing_img += 1
        print(f"  [{'OK ' if exists else '없음'}] {cam}: {sd['filename']}")
    if missing_img:
        print(f"  ⚠️  이미지 {missing_img}개 누락 — samples/ 압축 확인")

    ann = sample["anns"]
    print(f"\n[검증] 첫 keyframe GT 박스 수: {len(ann)}")
    if ann:
        box = nusc.get("sample_annotation", ann[0])
        print(f"  예시: category={box['category_name']}, "
              f"위치(global)={[round(x,1) for x in box['translation']]}")

    # --- 4. CAN 로드 ---
    print("\n[검증] CAN bus API...")
    try:
        can = NuScenesCanBus(dataroot=root)
        scene_name = nusc.get("scene", sample["scene_token"])["name"]
        if scene_name in can.can_blacklist:
            print(f"  ⚠️  {scene_name}: CAN blacklist(결손) — 정상, 다른 scene 사용")
        else:
            pose = can.get_messages(scene_name, "pose")
            print(f"  OK  {scene_name}: pose 메시지 {len(pose)}개")
        print(f"  참고: CAN blacklist scene = {sorted(can.can_blacklist)[:5]}...")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  CAN 로드 경고: {type(e).__name__}: {e}")

    print("\n✅ 데이터 검증 통과 — M1(nusc_wrapper) 진행 가능")
    return 0


if __name__ == "__main__":
    sys.exit(main())
