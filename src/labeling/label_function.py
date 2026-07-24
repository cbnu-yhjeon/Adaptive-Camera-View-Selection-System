"""의사 라벨 생성 s(c) (CLAUDE.md B.4-(6),(7)) — M2의 최종 결합 단계.

파이프라인
----------
각 keyframe에서:
  1. GT 박스를 카메라에 투영해 채널 귀속 (geometry). 2채널 겹치면 0.5씩.
  2. 박스별 TTC → 위험 가중 w(TTC) 를 귀속 채널에 누적 → 채널 위험도.
  3. ego pose ±3초 윈도우로 기동 추론 (maneuver) → 채널 적합도 m(c).
  4. s(c) = Σ w(TTC_i) + λ·m(c).
scene 단위로:
  5. s(c)에 EMA 스무딩(α) → argmax = main_view.
  6. 1-keyframe 단발 전환 병합(flicker 방지).
  7. soft_label = softmax(s / T_soft), layout = 기동→템플릿.

산출 컬럼: sample_token, scene_token, timestamp, s_0..s_5, main_view,
soft_0..soft_5, maneuver, layout, n_objects, min_ttc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.labeling.maneuver import ManeuverThresholds, infer_maneuver
from src.labeling.ttc import compute_ttc, risk_weight, horizontal_range, TTC_INF
from src.utils.constants import (
    CAMERAS, NUM_CAMERAS, maneuver_to_layout_index, maneuver_view_vector,
)
from src.utils.geometry import point_visible_in_camera


@dataclass
class LabelConfig:
    categories: list[str] = field(default_factory=lambda: ["vehicle", "human.pedestrian"])
    max_range: float = 50.0
    tau: float = 4.0
    w_type: str = "exp"
    w_min_near: float = 0.3
    near_range: float = 10.0
    v_r_min: float = 0.1
    lambda_m: float = 1.0
    T_soft: float = 0.5
    ema_alpha: float = 0.7
    flicker_merge: bool = True
    maneuver_window_s: float = 3.0
    maneuver_th: ManeuverThresholds = field(default_factory=ManeuverThresholds)

    @classmethod
    def from_dict(cls, d: dict) -> "LabelConfig":
        lab = dict(d.get("label", d))
        mvth = lab.pop("maneuver", {}) or {}
        known = {k: lab[k] for k in lab if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        cfg.maneuver_th = ManeuverThresholds(
            **{k: v for k, v in mvth.items()
               if k in ManeuverThresholds.__dataclass_fields__})
        return cfg


def _channel_visibility(wrapper, sample_token, box, ego) -> list[int]:
    """박스가 보이는 채널 인덱스 목록 (0~5)."""
    vis = []
    for ci, cam in enumerate(CAMERAS):
        cal = wrapper.get_camera_calib(sample_token, cam)
        if point_visible_in_camera(box["global_center"], ego["translation"],
                                   ego["rotation"], cal["translation"],
                                   cal["rotation"], cal["intrinsic"],
                                   cal["width"], cal["height"]):
            vis.append(ci)
    return vis


def compute_channel_scores(wrapper, sample_token, cfg: LabelConfig):
    """한 keyframe의 위험도 채널 벡터(6) + n_objects + min_ttc.

    반환: (risk_vec[6], n_objects, min_ttc)
    """
    ego = wrapper.get_ego_pose(sample_token)
    ego_vel = wrapper.get_ego_velocity(sample_token)
    boxes = wrapper.get_boxes(sample_token, cfg.categories, cfg.max_range)

    risk = np.zeros(NUM_CAMERAS, dtype=np.float64)
    min_ttc = TTC_INF
    for box in boxes:
        v_obj = box["global_velocity"]
        if not np.all(np.isfinite(v_obj)):     # box_velocity NaN 폴백: 정지 가정
            v_obj = np.zeros(2)
        ttc = compute_ttc(box["global_center"][:2], ego["translation"][:2],
                          v_obj, ego_vel, v_r_min=cfg.v_r_min)
        r = horizontal_range(box["global_center"], ego["translation"])
        w = risk_weight(ttc, r, tau=cfg.tau, w_type=cfg.w_type,
                        w_min_near=cfg.w_min_near, near_range=cfg.near_range)
        if ttc < min_ttc:
            min_ttc = ttc

        vis = _channel_visibility(wrapper, sample_token, box, ego)
        if not vis:
            continue
        share = w / len(vis)                   # 2채널 겹치면 0.5씩
        for ci in vis:
            risk[ci] += share
    return risk, len(boxes), min_ttc


def infer_sample_maneuver(wrapper, sample_token, cfg: LabelConfig) -> str:
    """±window 초 ego pose 시퀀스로 기동 추론 (미래 참조 허용)."""
    scene_token = wrapper.sample_scene_token(sample_token)
    toks = wrapper.sample_tokens_in_scene(scene_token)
    t0 = wrapper.sample_timestamp(sample_token)

    win = [tk for tk in toks
           if abs(wrapper.sample_timestamp(tk) - t0) <= cfg.maneuver_window_s]
    positions = np.array([wrapper.get_ego_pose(tk)["translation"][:2] for tk in win])
    yaws = np.array([wrapper.get_ego_yaw(tk) for tk in win])
    times = np.array([wrapper.sample_timestamp(tk) for tk in win])
    speeds = np.array([np.linalg.norm(wrapper.get_ego_velocity(tk)) for tk in win])
    return infer_maneuver(positions, yaws, timestamps=times, speeds=speeds,
                          th=cfg.maneuver_th)


def _softmax(x: np.ndarray, T: float) -> np.ndarray:
    z = x / T
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def generate_scene_labels(wrapper, scene_token: str, cfg: LabelConfig) -> list[dict]:
    """scene 전체 keyframe 라벨 (EMA 스무딩 + flicker 병합 적용)."""
    toks = wrapper.sample_tokens_in_scene(scene_token)
    rows = []
    raw_s = []
    for st in toks:
        risk, n_obj, min_ttc = compute_channel_scores(wrapper, st, cfg)
        maneuver = infer_sample_maneuver(wrapper, st, cfg)
        m_vec = np.array(maneuver_view_vector(maneuver), dtype=np.float64)
        s = risk + cfg.lambda_m * m_vec
        raw_s.append(s)
        rows.append({
            "sample_token": st,
            "scene_token": scene_token,
            "timestamp": wrapper.sample_timestamp(st),
            "maneuver": maneuver,
            "layout": maneuver_to_layout_index(maneuver),
            "n_objects": n_obj,
            "min_ttc": float(min_ttc if min_ttc != TTC_INF else -1.0),  # inf→-1 저장
            "_risk": risk,
        })

    # --- EMA 스무딩 ---
    ema = []
    for i, s in enumerate(raw_s):
        ema.append(s if i == 0 else cfg.ema_alpha * ema[-1] + (1 - cfg.ema_alpha) * s)

    main = [int(np.argmax(e)) for e in ema]

    # --- flicker 병합: A B A 패턴에서 가운데 B(1-keyframe)를 A로 ---
    if cfg.flicker_merge:
        for i in range(1, len(main) - 1):
            if main[i - 1] == main[i + 1] and main[i] != main[i - 1]:
                main[i] = main[i - 1]

    # --- 컬럼 채우기 ---
    for i, row in enumerate(rows):
        s_smooth = ema[i]
        soft = _softmax(s_smooth, cfg.T_soft)
        for c in range(NUM_CAMERAS):
            row[f"s_{c}"] = float(s_smooth[c])
            row[f"soft_{c}"] = float(soft[c])
        row["main_view"] = main[i]
        row.pop("_risk", None)
    return rows


def generate_all_labels(wrapper, cfg: LabelConfig, limit_scenes: int | None = None) -> list[dict]:
    """모든 (또는 일부) scene 라벨 생성."""
    scenes = wrapper.scene_tokens()
    if limit_scenes:
        scenes = scenes[:limit_scenes]
    all_rows = []
    for st in scenes:
        all_rows.extend(generate_scene_labels(wrapper, st, cfg))
    return all_rows
