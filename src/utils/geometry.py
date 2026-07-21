"""좌표 변환 및 카메라 투영 유틸 (CLAUDE.md M2 / B.4-(1)).

원리
----
nuScenes의 3D 객체 GT 박스는 **global(월드) 좌표계**로 주어진다. "이 객체가
어느 카메라 화면에 보이는가"를 알려면 좌표계를 두 번 갈아탄다:

    global → ego(차량) → camera(개별 카메라) → image(픽셀)

각 단계는 (회전 R, 평행이동 t) 한 쌍으로 정의된다. nuScenes는 회전을
쿼터니언 [w, x, y, z], 이동을 [x, y, z]로 준다.

- global→ego : ego_pose(차량의 global 자세)의 역변환
- ego→camera : calibrated_sensor(카메라의 ego 기준 자세)의 역변환
- camera→image: intrinsic 행렬 K로 원근투영

카메라 좌표계에서 z가 광축(앞) 방향이므로, **z > 0 이고 픽셀 좌표가 이미지
범위 안(0≤u<W, 0≤v<H)** 이면 그 카메라에 "보인다"고 판정한다.

왜 이 프로젝트에서 쓰는가
-------------------------
의사 라벨(M2)은 "각 카메라에 어떤 위험 객체가 있는가"를 세어 채널 점수 s(c)를
만든다. 그 첫 단계가 바로 이 투영이다. 여기가 틀리면 라벨 전체가 틀어지므로
단위테스트로 5개 대표 케이스(전방/후방/좌우/뒤편/화면밖)를 검증한다.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# 회전 표현
# ---------------------------------------------------------------------
def quaternion_to_rotation_matrix(q) -> np.ndarray:
    """단위 쿼터니언 [w, x, y, z] → 3x3 회전행렬.

    nuScenes devkit(pyquaternion)과 동일한 (w, x, y, z) 순서 규약.
    외부 의존성 없이 표준 공식으로 구현하여 단위테스트가 가볍게 돌게 한다.
    """
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError("영(0) 쿼터니언은 회전으로 변환할 수 없다.")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


# ---------------------------------------------------------------------
# 좌표계 변환
# ---------------------------------------------------------------------
def transform_points(points: np.ndarray, rotation: np.ndarray,
                     translation: np.ndarray, inverse: bool = False) -> np.ndarray:
    """포인트들을 (R, t) 자세로 변환.

    Args:
        points: (N, 3) 또는 (3,) 좌표.
        rotation: (3, 3) 회전행렬 R (부모←자식, 즉 자식 좌표를 부모 좌표로 옮김).
        translation: (3,) 이동 t (부모 좌표계에서의 자식 원점 위치).
        inverse: False면 자식→부모 ( p_parent = R·p + t ),
                 True면  부모→자식 ( p_child  = Rᵀ·(p − t) ).

    Returns:
        입력과 같은 shape의 변환된 좌표.
    """
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    if single:
        pts = pts[None, :]
    t = np.asarray(translation, dtype=np.float64).reshape(1, 3)
    R = np.asarray(rotation, dtype=np.float64)
    if inverse:
        out = (pts - t) @ R          # (Rᵀ·xᵀ)ᵀ = x·R
    else:
        out = pts @ R.T + t
    return out[0] if single else out


def global_to_camera(points_global: np.ndarray,
                     ego_translation, ego_rotation,
                     cam_translation, cam_rotation) -> np.ndarray:
    """global 좌표 → camera 좌표 (global→ego→camera 2단 역변환).

    ego_*/cam_* 의 rotation은 쿼터니언 [w,x,y,z] 또는 (3,3) 행렬 모두 허용.
    """
    R_ego = _as_matrix(ego_rotation)
    R_cam = _as_matrix(cam_rotation)
    pts_ego = transform_points(points_global, R_ego, ego_translation, inverse=True)
    pts_cam = transform_points(pts_ego, R_cam, cam_translation, inverse=True)
    return pts_cam


def _as_matrix(rotation) -> np.ndarray:
    """쿼터니언(len 4) 또는 (3,3) 행렬을 회전행렬로 정규화."""
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape == (3, 3):
        return r
    if r.shape == (4,):
        return quaternion_to_rotation_matrix(r)
    raise ValueError(f"회전은 쿼터니언(4,) 또는 행렬(3,3)이어야 함: shape={r.shape}")


# ---------------------------------------------------------------------
# 카메라 투영
# ---------------------------------------------------------------------
def project_to_image(points_cam: np.ndarray, intrinsic: np.ndarray):
    """카메라 좌표 (N,3) → 픽셀 좌표 (N,2) + 깊이 z (N,).

    z ≤ 0 (카메라 뒤)인 점의 픽셀 좌표는 의미가 없으므로 NaN으로 둔다.
    """
    pts = np.asarray(points_cam, dtype=np.float64)
    single = pts.ndim == 1
    if single:
        pts = pts[None, :]
    K = np.asarray(intrinsic, dtype=np.float64)
    depth = pts[:, 2].copy()
    uvw = pts @ K.T                       # (N,3)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvw[:, :2] / uvw[:, 2:3]
    uv[depth <= 0] = np.nan
    if single:
        return uv[0], float(depth[0])
    return uv, depth


def in_image(uv: np.ndarray, depth, width: int, height: int) -> np.ndarray:
    """픽셀 좌표가 이미지 범위 안이고 카메라 앞(z>0)인지 여부 (bool).

    판정: z > 0  AND  0 ≤ u < W  AND  0 ≤ v < H.
    """
    uv = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    single = uv.ndim == 1
    if single:
        uv = uv[None, :]
        depth = depth.reshape(1)
    u, v = uv[:, 0], uv[:, 1]
    ok = (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return bool(ok[0]) if single else ok


def point_visible_in_camera(point_global, ego_translation, ego_rotation,
                            cam_translation, cam_rotation, intrinsic,
                            width: int, height: int) -> bool:
    """global 3D 점 1개가 특정 카메라 화면에 보이는지 판정 (end-to-end).

    라벨 함수가 "이 객체가 이 카메라 채널에 귀속되는가"를 물을 때 쓰는 함수.
    """
    pts_cam = global_to_camera(point_global, ego_translation, ego_rotation,
                               cam_translation, cam_rotation)
    uv, depth = project_to_image(pts_cam, intrinsic)
    return in_image(uv, depth, width, height)
