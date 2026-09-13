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
from src.labeling.ttc import (
    TTC_INF, closest_approach, compute_ttc, horizontal_range, path_factor, risk_weight,
)
from src.utils.constants import (
    CAMERAS, CAMERA_INDEX, NUM_CAMERAS, attribute_weight, maneuver_to_layout_index,
    maneuver_view_vector, travel_direction_view,
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
    # 위험 재정의 (B.4-(2),(3))
    ttc_mode: str = "cpa"        # cpa | radial(ablation)
    d_near: float = 2.0          # 최접근 거리가 이보다 가까우면 경로 감쇠 없음
    d_scale: float = 2.0         # 초과분 지수 감쇠 스케일
    rho_min: float = 0.05        # 경로 무관 객체의 하한
    d_hazard: float = 3.0        # '임박 위험 객체' 판정: 최접근 거리가 이보다 가까울 것
    attr_weight: dict = field(default_factory=dict)   # 비면 DEFAULT_ATTR_WEIGHT
    # 보조 뷰 선택 (B.4-(6))
    aux_threshold: float = 0.5   # s(c)가 이 값 이상인 비-primary 채널만 강조
    aux_max: int = 2             # 동시에 강조할 최대 채널 수
    lambda_m: float = 1.0
    T_soft: float = 0.5
    ema_alpha: float = 0.7
    flicker_merge: bool = True
    stop_hold: bool = True       # 정지 시 뷰 유지(극단 위험 제외)
    extreme_ttc: float = 1.5     # 이보다 짧은 min_TTC면 정지 중에도 위험 뷰로 전환
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


def _channel_visibility(calibs: list[dict], box, ego) -> list[int]:
    """박스가 보이는 채널 인덱스 목록 (0~5).

    calibs는 CAMERAS 순서의 캘리브레이션 6개. 캘리브레이션은 프레임 단위로 같으므로
    호출자가 프레임당 한 번만 조회해 넘긴다(박스마다 다시 조회하면 34k keyframe 규모에서
    devkit 조회와 배열 생성이 지배적인 비용이 된다).
    """
    vis = []
    for ci, cal in enumerate(calibs):
        if point_visible_in_camera(box["global_center"], ego["translation"],
                                   ego["rotation"], cal["translation"],
                                   cal["rotation"], cal["intrinsic"],
                                   cal["width"], cal["height"]):
            vis.append(ci)
    return vis


def channel_risk_contributors(wrapper, sample_token, cfg: LabelConfig) -> list[dict]:
    """keyframe의 객체별 위험 기여 목록. 노이즈 시뮬레이터가 재계산에 쓴다.

    각 항목: {vis: [채널 인덱스], r: 거리, ttc: float(inf 가능), w: 위험가중}.
    """
    ego = wrapper.get_ego_pose(sample_token)
    ego_vel = wrapper.get_ego_velocity(sample_token)
    boxes = wrapper.get_boxes(sample_token, cfg.categories, cfg.max_range)
    calibs = [wrapper.get_camera_calib(sample_token, cam) for cam in CAMERAS]

    contribs = []
    for box in boxes:
        v_obj = box["global_velocity"]
        if not np.all(np.isfinite(v_obj)):     # box_velocity NaN 폴백: 정지 가정
            v_obj = np.zeros(2)
        ttc = compute_ttc(box["global_center"][:2], ego["translation"][:2],
                          v_obj, ego_vel, v_r_min=cfg.v_r_min, mode=cfg.ttc_mode)
        r = horizontal_range(box["global_center"], ego["translation"])
        # ρ = 상태 계수 × 경로 관련성. 스쳐 지나갈 객체와 정지 객체를 낮춘다.
        _, dca = closest_approach(box["global_center"][:2], ego["translation"][:2],
                                  v_obj, ego_vel)
        rho = (attribute_weight(box.get("attribute", ""), cfg.attr_weight or None)
               * path_factor(dca, d_near=cfg.d_near, d_scale=cfg.d_scale,
                             rho_min=cfg.rho_min))
        w = risk_weight(ttc, r, tau=cfg.tau, w_type=cfg.w_type,
                        w_min_near=cfg.w_min_near, near_range=cfg.near_range, rho=rho)
        vis = _channel_visibility(calibs, box, ego)
        contribs.append({"vis": vis, "r": float(r), "ttc": float(ttc),
                         "w": float(w), "rho": float(rho), "dca": float(dca)})
    return contribs


def risk_vector_from_contribs(contribs: list[dict]) -> np.ndarray:
    """기여 목록 → 채널 위험도 벡터(6). 2채널 겹치면 0.5씩 분배."""
    risk = np.zeros(NUM_CAMERAS, dtype=np.float64)
    for c in contribs:
        vis = c["vis"]
        if not vis:
            continue
        share = c["w"] / len(vis)
        for ci in vis:
            risk[ci] += share
    return risk


def hazard_ttcs(contribs: list[dict], cfg: LabelConfig) -> list[float]:
    """'임박 위험 객체'의 TTC 목록.

    cpa 정의에서는 옆으로 5m 떨어져 스쳐 갈 객체도 최접근 시각이 0.2s로 나온다.
    시간만으로는 위험을 가릴 수 없으므로 **최접근 거리(DCA)가 d_hazard 이내**인
    객체만 위험으로 센다. 이 정의가 평가의 '위험 객체 가시화율'과도 일치해야 한다.
    """
    return [c["ttc"] for c in contribs
            if np.isfinite(c["ttc"]) and c.get("dca", 0.0) < cfg.d_hazard]


def compute_channel_scores(wrapper, sample_token, cfg: LabelConfig):
    """한 keyframe의 위험도 채널 벡터(6) + n_objects + min_ttc(임박 위험 기준)."""
    contribs = channel_risk_contributors(wrapper, sample_token, cfg)
    risk = risk_vector_from_contribs(contribs)
    ttcs = hazard_ttcs(contribs, cfg)
    min_ttc = min(ttcs) if ttcs else TTC_INF
    return risk, len(contribs), min_ttc


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


def _argmax_excl(scores: np.ndarray, excl: int) -> int:
    """primary(excl) 채널을 제외한 argmax → attention_view."""
    m = scores.copy()
    m[excl] = -np.inf
    return int(np.argmax(m))


def select_aux_views(scores: np.ndarray, primary: int, cfg: LabelConfig) -> list[int]:
    """강조할 보조 채널 목록 (0~aux_max개) — 임계값 기반 다중 선택.

    항상 1장을 고르면 위험도 기동도 없는 프레임에서도 화면이 켜져 오퍼레이터가
    보조 뷰를 무시하게 된다(실측: 16.2%의 프레임이 위험 기여 없이 기동 항만으로 선택됨).
    반대로 차선변경처럼 측방과 후측방이 동시에 중요한 상황은 1장으로 표현할 수 없다.
    s(c) ≥ aux_threshold인 비-primary 채널을 점수 순으로 최대 aux_max개 고른다.

    임계값을 위험 항이 아니라 s(c)=위험+λ·m(c) 전체에 거는 것이 핵심이다. 기동 항
    덕분에 **검출된 객체가 하나도 없어도 차선변경 방향은 켜진다** — 검출 실패 시
    화면이 꺼지는 것이 이 시스템의 가장 위험한 실패 모드이기 때문이다.
    """
    order = [c for c in np.argsort(-scores) if c != primary]
    return [int(c) for c in order[:max(0, cfg.aux_max)]
            if scores[c] >= cfg.aux_threshold]


def _softmax_excl(x: np.ndarray, temp: float, excl: int) -> np.ndarray:
    """primary(excl) 채널을 0으로 마스킹한 softmax (attention 분포, KL 타깃)."""
    z = (x / temp) - np.max(x / temp)
    e = np.exp(z)
    e[excl] = 0.0
    tot = e.sum()
    return e / tot if tot > 0 else e


def _resolve_views(rows: list[dict], ema: list[np.ndarray], cfg: LabelConfig):
    """시간순으로 primary(진행방향) + attention(적응형 보조) 결정.

    - 주행 중: primary=진행방향, attention=argmax_{c≠primary}.
    - 정지(STOP) & 비-극단위험 & stop_hold: 직전 뷰 유지(전환 억제).
    - 정지라도 극단 위험(min_TTC<extreme_ttc)이면 위험 뷰로 전환.
    """
    primary, attention = [], []
    prev_p = CAMERA_INDEX["CAM_FRONT"]
    prev_a: int | None = None
    for i, row in enumerate(rows):
        mv = row["maneuver"]
        min_ttc = row["min_ttc"]                       # inf는 -1로 저장됨
        extreme = 0.0 <= min_ttc < cfg.extreme_ttc
        hold = (mv == "STOP" and cfg.stop_hold and not extreme and i > 0)
        if hold:
            p = prev_p
            a = prev_a if prev_a is not None else _argmax_excl(ema[i], p)
        else:
            p = travel_direction_view(mv)
            a = _argmax_excl(ema[i], p)
        primary.append(p)
        attention.append(a)
        prev_p, prev_a = p, a
    return primary, attention


def generate_scene_labels(wrapper, scene_token: str, cfg: LabelConfig) -> list[dict]:
    """scene 전체 keyframe 라벨 (EMA 스무딩 + primary/attention + flicker 병합)."""
    toks = wrapper.sample_tokens_in_scene(scene_token)
    rows, raw_s = [], []
    for st in toks:
        risk, n_obj, min_ttc = compute_channel_scores(wrapper, st, cfg)
        maneuver = infer_sample_maneuver(wrapper, st, cfg)
        m_vec = np.array(maneuver_view_vector(maneuver), dtype=np.float64)
        raw_s.append(risk + cfg.lambda_m * m_vec)
        rows.append({
            "sample_token": st,
            "scene_token": scene_token,
            "timestamp": wrapper.sample_timestamp(st),
            "maneuver": maneuver,
            "layout": maneuver_to_layout_index(maneuver),
            "n_objects": n_obj,
            "min_ttc": float(min_ttc if min_ttc != TTC_INF else -1.0),  # inf→-1
        })

    # --- EMA 스무딩 ---
    ema = []
    for i, s in enumerate(raw_s):
        ema.append(s if i == 0 else cfg.ema_alpha * ema[-1] + (1 - cfg.ema_alpha) * s)

    primary, attention = _resolve_views(rows, ema, cfg)

    # --- attention flicker 병합: A B A 패턴에서 가운데 B(1-keyframe)를 A로 ---
    if cfg.flicker_merge:
        for i in range(1, len(attention) - 1):
            if attention[i - 1] == attention[i + 1] and attention[i] != attention[i - 1]:
                attention[i] = attention[i - 1]

    # --- 컬럼 채우기 ---
    for i, row in enumerate(rows):
        s_smooth = ema[i]
        soft = _softmax_excl(s_smooth, cfg.T_soft, primary[i])
        aux = select_aux_views(s_smooth, primary[i], cfg)
        for c in range(NUM_CAMERAS):
            row[f"s_{c}"] = float(s_smooth[c])
            row[f"soft_{c}"] = float(soft[c])
            row[f"aux_{c}"] = int(c in aux)      # 다중 선택 타깃(0/1)
        row["primary_view"] = primary[i]
        row["attention_view"] = attention[i]     # 단일 argmax — 베이스라인 비교용 유지
        row["n_aux"] = len(aux)
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
