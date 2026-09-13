"""nuScenes devkit 래퍼 (CLAUDE.md M1 / B.3).

devkit의 저수준 토큰 API를 이 프로젝트에서 반복적으로 쓰는 단위
(scene 순회, 6채널 이미지 경로, ego 좌표 GT 박스, CAN 시계열, 카메라 캘리브)
로 감싼다. 라벨링·데이터셋·평가가 모두 이 래퍼만 통해 데이터에 접근하므로,
devkit 버전이나 좌표 규약이 바뀌어도 수정 지점이 한 곳으로 모인다.

좌표 규약
--------
- GT 박스 위치/속도는 global 좌표로 얻는다(TTC는 프레임 무관, 일관성만 유지).
- 카메라 채널 귀속은 global→ego→camera 투영으로 판정(geometry.py).
- ego 속도는 인접 keyframe ego_pose 차분으로 추정(CAN 없어도 동작).
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from src.utils.constants import CAMERAS


class NuscWrapper:
    def __init__(self, dataroot: str, version: str = "v1.0-mini",
                 require_image_files: bool = True, verbose: bool = False):
        from nuscenes.nuscenes import NuScenes

        self.dataroot = os.path.abspath(os.path.expanduser(dataroot))
        self.version = version
        self.require_image_files = require_image_files
        self.nusc = NuScenes(version=version, dataroot=self.dataroot, verbose=verbose)

        # CAN API는 실패해도 파이프라인이 죽지 않도록 지연·방어 로드
        self._can = None
        self._can_blacklist: set[int] = set()
        try:
            from nuscenes.can_bus.can_bus_api import NuScenesCanBus
            self._can = NuScenesCanBus(dataroot=self.dataroot)
            self._can_blacklist = set(self._can.can_blacklist)
        except Exception:  # noqa: BLE001
            self._can = None

    # ------------------------------------------------------------------
    # scene / sample 순회
    # ------------------------------------------------------------------
    def scene_tokens(self) -> list[str]:
        """사용 가능한 scene 토큰 목록. require_image_files면 이미지 결손 scene 제외."""
        tokens = [s["token"] for s in self.nusc.scene]
        if self.require_image_files:
            tokens = [t for t in tokens if self._scene_has_images(t)]
        return tokens

    def scene_name(self, scene_token: str) -> str:
        return self.nusc.get("scene", scene_token)["name"]

    def sample_tokens_in_scene(self, scene_token: str) -> list[str]:
        """scene 내 keyframe(sample) 토큰을 시간순으로."""
        scene = self.nusc.get("scene", scene_token)
        out, tok = [], scene["first_sample_token"]
        while tok:
            out.append(tok)
            tok = self.nusc.get("sample", tok)["next"]
        return out

    def all_sample_tokens(self) -> list[str]:
        out = []
        for st in self.scene_tokens():
            out.extend(self.sample_tokens_in_scene(st))
        return out

    def sample_scene_token(self, sample_token: str) -> str:
        return self.nusc.get("sample", sample_token)["scene_token"]

    def sample_timestamp(self, sample_token: str) -> float:
        """초 단위 timestamp (devkit은 마이크로초)."""
        return self.nusc.get("sample", sample_token)["timestamp"] / 1e6

    # ------------------------------------------------------------------
    # 이미지 / 캘리브
    # ------------------------------------------------------------------
    def get_image_paths(self, sample_token: str) -> dict[str, str]:
        """채널 이름 → 절대 이미지 경로."""
        sample = self.nusc.get("sample", sample_token)
        paths = {}
        for cam in CAMERAS:
            sd = self.nusc.get("sample_data", sample["data"][cam])
            paths[cam] = os.path.join(self.dataroot, sd["filename"])
        return paths

    def get_camera_calib(self, sample_token: str, cam: str) -> dict:
        """카메라 intrinsic + ego기준 외부파라미터 + 해상도."""
        sample = self.nusc.get("sample", sample_token)
        sd = self.nusc.get("sample_data", sample["data"][cam])
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        return {
            "intrinsic": np.array(cs["camera_intrinsic"], dtype=np.float64),
            "translation": np.array(cs["translation"], dtype=np.float64),
            "rotation": np.array(cs["rotation"], dtype=np.float64),  # 쿼터니언 [w,x,y,z]
            "width": sd["width"],
            "height": sd["height"],
        }

    def get_ego_pose(self, sample_token: str) -> dict:
        """sample의 ego pose (CAM_FRONT 기준). translation/rotation/timestamp."""
        sample = self.nusc.get("sample", sample_token)
        sd = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ego = self.nusc.get("ego_pose", sd["ego_pose_token"])
        return {
            "translation": np.array(ego["translation"], dtype=np.float64),
            "rotation": np.array(ego["rotation"], dtype=np.float64),
            "timestamp": ego["timestamp"] / 1e6,
        }

    def get_ego_yaw(self, sample_token: str) -> float:
        """ego 헤딩(yaw, rad) — global 기준."""
        from pyquaternion import Quaternion
        q = Quaternion(self.get_ego_pose(sample_token)["rotation"])
        return float(q.yaw_pitch_roll[0])

    def get_ego_velocity(self, sample_token: str) -> np.ndarray:
        """인접 keyframe ego_pose 차분으로 추정한 global 속도 (vx, vy)."""
        scene_token = self.sample_scene_token(sample_token)
        toks = self.sample_tokens_in_scene(scene_token)
        i = toks.index(sample_token)
        # 중앙차분 우선, 경계는 전/후진차분
        j0 = max(0, i - 1)
        j1 = min(len(toks) - 1, i + 1)
        if j0 == j1:
            return np.zeros(2)
        p0 = self.get_ego_pose(toks[j0])
        p1 = self.get_ego_pose(toks[j1])
        dt = p1["timestamp"] - p0["timestamp"]
        if dt <= 1e-6:
            return np.zeros(2)
        return (p1["translation"][:2] - p0["translation"][:2]) / dt

    # ------------------------------------------------------------------
    # GT 박스
    # ------------------------------------------------------------------
    def get_boxes(self, sample_token: str, categories: list[str] | None = None,
                  max_range: float | None = None) -> list[dict]:
        """sample의 GT 박스 목록.

        각 박스: {category, global_center(xyz), size, yaw, global_velocity(xy),
                  range(수평거리 m), ann_token}. category prefix 필터 + 거리 상한 적용.
        """
        sample = self.nusc.get("sample", sample_token)
        ego_t = self.get_ego_pose(sample_token)["translation"]
        out = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            cat = ann["category_name"]
            if categories is not None and not any(cat.startswith(p) for p in categories):
                continue
            center = np.array(ann["translation"], dtype=np.float64)
            r = float(np.linalg.norm((center - ego_t)[:2]))
            if max_range is not None and r > max_range:
                continue
            vel = self.nusc.box_velocity(ann_token)  # global (vx,vy,vz), NaN 가능
            out.append({
                "category": cat,
                "global_center": center,
                "size": np.array(ann["size"], dtype=np.float64),
                "rotation": np.array(ann["rotation"], dtype=np.float64),
                "global_velocity": np.array(vel[:2], dtype=np.float64),
                "range": r,
                "ann_token": ann_token,
            })
        return out

    # ------------------------------------------------------------------
    # CAN
    # ------------------------------------------------------------------
    def has_can(self, scene_token: str) -> bool:
        if self._can is None:
            return False
        num = self._scene_number(scene_token)
        return num not in self._can_blacklist

    def get_can_messages(self, scene_token: str, message: str = "pose") -> list:
        """CAN 메시지 리스트 (pose/steeranglefeedback/zoe_veh_info 등). 없으면 []."""
        if not self.has_can(scene_token):
            return []
        try:
            return self._can.get_messages(self.scene_name(scene_token), message)
        except Exception:  # noqa: BLE001
            return []

    def get_can_features(self, sample_token: str):
        """모델 입력용 CAN 특징 (정규화) + 결측 마스크.

        반환: (feat[5], mask) — feat=[speed, steering, accel_x, accel_y, yaw_rate],
        mask=1이면 CAN 존재. CAN 없으면 speed만 ego_pose 차분으로 채우고 나머지 0.
        """
        from src.utils.constants import CAN_NORM

        scene_token = self.sample_scene_token(sample_token)
        speed_fallback = float(np.linalg.norm(self.get_ego_velocity(sample_token)))
        if not self.has_can(scene_token):
            feat = np.array([speed_fallback / CAN_NORM["speed"], 0, 0, 0, 0], dtype=np.float32)
            return feat, 0.0

        utime = int(self.nusc.get("sample", sample_token)["timestamp"])
        pose = self._nearest_can(scene_token, "pose", utime)
        steer = self._nearest_can(scene_token, "steeranglefeedback", utime)
        speed = pose["vel"][0] if pose and "vel" in pose else speed_fallback
        accel = pose["accel"] if pose and "accel" in pose else [0.0, 0.0, 0.0]
        yaw_rate = pose["rotation_rate"][2] if pose and "rotation_rate" in pose else 0.0
        steering = steer["value"] if steer and "value" in steer else 0.0
        feat = np.array([
            speed / CAN_NORM["speed"],
            steering / CAN_NORM["steering"],
            accel[0] / CAN_NORM["accel_x"],
            accel[1] / CAN_NORM["accel_y"],
            yaw_rate / CAN_NORM["yaw_rate"],
        ], dtype=np.float32)
        return feat, 1.0

    def _nearest_can(self, scene_token: str, message: str, utime: int):
        """utime(마이크로초)에 가장 가까운 CAN 메시지."""
        msgs = self.get_can_messages(scene_token, message)
        if not msgs:
            return None
        return min(msgs, key=lambda m: abs(m["utime"] - utime))

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------
    def _scene_number(self, scene_token: str) -> int:
        name = self.scene_name(scene_token)  # 'scene-0061'
        return int(name.split("-")[-1])

    @lru_cache(maxsize=None)
    def _scene_has_images(self, scene_token: str) -> bool:
        """scene의 첫 sample 6채널 이미지가 모두 존재하는지 (부분집합 대응)."""
        first = self.nusc.get("scene", scene_token)["first_sample_token"]
        for path in self.get_image_paths(first).values():
            if not os.path.isfile(path):
                return False
        return True
