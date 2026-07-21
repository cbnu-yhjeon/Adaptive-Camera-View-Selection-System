"""geometry.py 단위테스트 (CLAUDE.md M2 ✅ 조건: 투영 유틸 5케이스).

실데이터 없이 합성 6채널 카메라 리그로 검증한다.

좌표 규약 (nuScenes)
--------------------
- ego 좌표: x=전방, y=좌, z=상.
- camera 좌표: z=광축(정면), x=오른쪽, y=아래.
- 방위각 az: ego +x(전방)에서 +y(좌)로 재는 각. az=0 전방, az=180 후방,
  az>0 좌측, az<0 우측.

테스트는 pytest 없이도 직접 실행 가능: `python tests/test_geometry.py`
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.geometry import (  # noqa: E402
    project_to_image,
    point_visible_in_camera,
    quaternion_to_rotation_matrix,
)

# --- 합성 카메라 리그 --------------------------------------------------
# 이미지 1600x900, 초점거리 f=1000 → 수평 반화각 atan(800/1000)≈38.7°.
W, H, F = 1600, 900, 1000.0
INTRINSIC = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]], dtype=np.float64)

# 카메라 방향(ego +x 기준 yaw, 도). nuScenes 배치 근사.
CAM_YAW_DEG = {
    "CAM_FRONT": 0.0,
    "CAM_FRONT_LEFT": 55.0,
    "CAM_FRONT_RIGHT": -55.0,
    "CAM_BACK": 180.0,
    "CAM_BACK_LEFT": 125.0,
    "CAM_BACK_RIGHT": -125.0,
}

# FRONT 카메라의 cam→ego 회전: cam z(광축)→ego +x, cam x(우)→ego -y, cam y(아래)→ego -z.
R_FRONT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)


def _rot_z(deg: float) -> np.ndarray:
    """ego 상방(z)축 기준 회전행렬."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def cam_rotation(name: str) -> np.ndarray:
    """카메라 이름 → cam→ego 회전행렬 (yaw만큼 FRONT를 회전)."""
    return _rot_z(CAM_YAW_DEG[name]) @ R_FRONT


def obj_at(azimuth_deg: float, r: float = 10.0) -> np.ndarray:
    """ego 평면에서 방위각/거리로 객체 global 위치 (ego=global 가정)."""
    a = np.radians(azimuth_deg)
    return np.array([r * np.cos(a), r * np.sin(a), 0.0])


# ego는 global 원점에 정지(회전 없음) 가정
EGO_T = np.zeros(3)
EGO_R = np.eye(3)


def _visible(point, cam_name) -> bool:
    return point_visible_in_camera(
        point, EGO_T, EGO_R, np.zeros(3), cam_rotation(cam_name),
        INTRINSIC, W, H,
    )


# --- 테스트 케이스 -----------------------------------------------------
def test_case1_front_object_visible_in_front():
    """전방 10m 객체 → CAM_FRONT에 보인다."""
    assert _visible(obj_at(0.0), "CAM_FRONT")


def test_case2_front_object_not_in_back():
    """전방 객체는 CAM_BACK에는 안 보인다 (카메라 뒤, z<0)."""
    assert not _visible(obj_at(0.0), "CAM_BACK")


def test_case3_rear_object_visible_in_back():
    """후방 객체(az=180) → CAM_BACK에 보인다."""
    assert _visible(obj_at(180.0), "CAM_BACK")
    assert not _visible(obj_at(180.0), "CAM_FRONT")


def test_case4_left_object_in_front_left():
    """좌전방(az=+55) 객체 → CAM_FRONT_LEFT에 보이고 CAM_FRONT엔 안 보인다."""
    assert _visible(obj_at(55.0), "CAM_FRONT_LEFT")
    assert not _visible(obj_at(55.0), "CAM_FRONT")


def test_case5_right_object_in_front_right():
    """우전방(az=-55) 객체 → CAM_FRONT_RIGHT에 보인다."""
    assert _visible(obj_at(-55.0), "CAM_FRONT_RIGHT")
    assert not _visible(obj_at(-55.0), "CAM_FRONT")


def test_projection_center():
    """정면 정중앙 객체는 이미지 중심(주점)에 투영된다."""
    # ego +x 10m 객체를 FRONT 카메라 좌표로: (0,0,10) → 주점 (W/2, H/2).
    pts_cam = np.array([0.0, 0.0, 10.0])
    uv, depth = project_to_image(pts_cam, INTRINSIC)
    assert depth == 10.0
    assert np.allclose(uv, [W / 2, H / 2])


def test_quaternion_identity():
    """단위 쿼터니언 [1,0,0,0] → 항등 회전행렬."""
    assert np.allclose(quaternion_to_rotation_matrix([1, 0, 0, 0]), np.eye(3))


def test_quaternion_90deg_z():
    """z축 90° 회전 쿼터니언 → +x가 +y로 간다."""
    q = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]  # [w,x,y,z], 90° about z
    R = quaternion_to_rotation_matrix(q)
    assert np.allclose(R @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-9)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}  {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {t.__name__}  {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
