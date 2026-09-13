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
    TTC_INF, attr_path_gate, closest_approach, compute_ttc, horizontal_range,
    near_floor, path_factor, risk_weight,
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
    tau: float = 3.0
    w_type: str = "exp"
    w_min_near: float = 0.3
    near_range: float = 10.0
    v_r_min: float = 0.1
    # 위험 재정의 (B.4-(2),(3))
    ttc_mode: str = "cpa"        # cpa | radial(ablation)
    d_near: float = 3.0          # 최접근 거리가 이보다 가까우면 경로 감쇠 없음
    d_scale: float = 1.5         # 초과분 지수 감쇠 스케일
    rho_min: float = 0.05        # 경로 무관 객체의 하한
    d_hazard: float = 3.0        # '실제 위험' 판정 최접근 거리 (차폭 감안 여유 약 1.1m)
    hazard_ttc: float = 5.0      # '실제 위험' 판정 시간 지평 (원격주행 지연 감안)
    attr_exempt_touch: float = 1.5   # 이 거리 이내로 최접근하면 상태 감쇠를 완전히 해제
    attr_exempt_scale: float = 2.0   # 해제 정도의 감쇠 스케일 (0이면 예외 없음)
    touch_range: float = 2.0     # 이 거리 이내면 근접 보정을 방향과 무관하게 적용
    attr_weight: dict = field(default_factory=dict)   # 비면 DEFAULT_ATTR_WEIGHT
    # 보조 뷰 선택 (B.4-(6))
    aux_threshold: float = 0.12  # 위험 슬롯 임계 — 순수 위험 항 기준(기동 항 제외)
    aux_max: int = 2             # 동시에 강조할 최대 채널 수
    m_guard: float = 0.9         # 기동 적합도가 이 값 이상인 채널을 사각지대로 보장
    lambda_m: float = 0.4
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
        # 상태 감쇠는 '스쳐 지나갈' 객체용이다. 경로 위(최접근이 차체 폭 수준)면 해제한다.
        a = attribute_weight(box.get("attribute", ""), cfg.attr_weight or None)
        gate = (attr_path_gate(dca, cfg.attr_exempt_touch, cfg.attr_exempt_scale)
                if cfg.attr_exempt_scale > 0 else 0.0)
        rho = ((a + (1.0 - a) * gate)
               * path_factor(dca, d_near=cfg.d_near, d_scale=cfg.d_scale,
                             rho_min=cfg.rho_min))
        # 근접 보정은 진행 방향 쪽 객체에만 — 지나쳐 온 정지 객체가 후방 채널을 켜지 않도록.
        floor = near_floor(box["global_center"][:2], ego["translation"][:2], ego_vel, r,
                           w_min_near=cfg.w_min_near, near_range=cfg.near_range,
                           touch_range=cfg.touch_range,
                           stop_speed=cfg.maneuver_th.stop_speed)
        w = risk_weight(ttc, r, tau=cfg.tau, w_type=cfg.w_type,
                        w_min_near=floor, near_range=cfg.near_range, rho=rho)
        vis = _channel_visibility(calibs, box, ego)
        contribs.append({"vis": vis, "r": float(r), "ttc": float(ttc),
                         "w": float(w), "rho": float(rho), "dca": float(dca),
                         "floor": float(floor)})
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


def is_hazard(contrib: dict, cfg: LabelConfig) -> bool:
    """'실제 위험' 판정 — 라벨·평가·골드셋 추출이 공유하는 단일 정의.

    `hazard_ttc(5s) 안에 d_hazard(3m) 이내로 최접근`. 두 조건이 모두 필요하다:
      - 시간만 보면 옆으로 5m 떨어져 스쳐 갈 객체도 최접근 시각이 0.2s로 나온다.
      - 거리만 보면 10초 뒤에나 가까워질 객체가 지금 위험으로 잡힌다.
    DCA는 중심점 간 거리라 3m는 차폭 감안 실제 여유 약 1.1m다.
    trainval 기준 해당 채널 1,879개(프레임의 5.3%) — 이 희소성 때문에 표시 정책의
    정밀도 상한이 구조적으로 낮다. 자세한 근거는 CLAUDE.md B.8 참조.
    """
    ttc = contrib.get("ttc", np.inf)
    return bool(np.isfinite(ttc) and 0.0 <= ttc < cfg.hazard_ttc
                and contrib.get("dca", 0.0) < cfg.d_hazard)


def hazard_ttcs(contribs: list[dict], cfg: LabelConfig) -> list[float]:
    """'실제 위험' 객체의 TTC 목록 (min_ttc 컬럼 산출용)."""
    return [c["ttc"] for c in contribs if is_hazard(c, cfg)]


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


def select_aux_views(risk: np.ndarray, m_vec: np.ndarray, primary: int,
                     cfg: LabelConfig) -> list[int]:
    """강조할 보조 채널 목록 (0~aux_max개) — **슬롯을 역할로 나눠** 배정한다.

    슬롯 1 (안전 기본값): 기동이 만드는 사각지대 채널. 차선변경이면 후측방, 회전이면
      회전 방향. **검출된 객체가 없어도 켠다** — 사각을 봐야 하는 이유는 "객체가
      검출돼서"가 아니라 "안 보여서"이고, 검출 실패 시 화면이 꺼지는 것이 이 시스템의
      가장 위험한 실패 모드다.
    슬롯 2~ (위험): 순수 위험 항 기준 상위 채널 중 risk ≥ aux_threshold.

    하나의 임계값을 합산 점수 s(c)에 걸면 기동 항이 슬롯을 모두 차지해 실제 위험이
    밀려난다(실측: 차선변경 중 위험 커버 60.6%). 역할을 나누면 사각지대를 보장하면서도
    남은 슬롯을 위험이 가져간다(70.3%).
    """
    sel: list[int] = []
    cand = [c for c in np.argsort(-m_vec) if c != primary and m_vec[c] >= cfg.m_guard]
    if cand:
        sel.append(int(cand[0]))
    for c in np.argsort(-risk):
        if len(sel) >= max(0, cfg.aux_max):
            break
        if c == primary or int(c) in sel:
            continue
        if risk[c] >= cfg.aux_threshold:
            sel.append(int(c))
    return sel[:max(0, cfg.aux_max)]


def merge_aux_flicker(aux_seq: list[list[int]]) -> list[list[int]]:
    """A B A 패턴에서 1-keyframe(0.5s)짜리 단발 변경을 이전 집합으로 병합.

    attention 단일 채널에만 적용하던 병합을 보조 뷰 '집합'으로 확장한다.
    기동 항을 평활하지 않게 되면서 기동 경계에서 집합이 튀는 빈도가 늘었다.
    """
    out = [list(a) for a in aux_seq]
    for i in range(1, len(out) - 1):
        if set(out[i - 1]) == set(out[i + 1]) and set(out[i]) != set(out[i - 1]):
            out[i] = list(out[i - 1])
    return out


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


def smooth_scores(risks: list[np.ndarray], m_vecs: list[np.ndarray],
                  alpha: float, lambda_m: float) -> list[np.ndarray]:
    """s(c) = EMA(위험 항) + λ·m(c) — **기동 항은 평활하지 않는다.**

    EMA는 프레임마다 흔들리는 *검출 위험*을 안정화하려는 장치다. 기동 상태는 이미
    ±3초 ego pose 윈도우로 추론된 안정적인 신호라, 여기서 다시 평활하면 이중 스무딩이
    되어 기동 항이 이웃 프레임에 희석된다. 실제로 STRAIGHT 사이에 낀 차선변경
    프레임에서 후측방 항(λ·m=0.4)이 0.12까지 떨어져 임계값(0.15)을 넘지 못했고,
    "차선변경 중에는 사각지대를 반드시 켠다"는 안전 기본값이 무력화됐다.
    """
    return [e + lambda_m * m for e, m in zip(smooth_risk(risks, alpha), m_vecs)]


def smooth_risk(risks: list[np.ndarray], alpha: float) -> list[np.ndarray]:
    """위험 항 EMA — 검출 지터를 눌러 라벨 flicker를 줄인다."""
    out, ema = [], None
    for r in risks:
        ema = r if ema is None else alpha * ema + (1 - alpha) * r
        out.append(ema)
    return out


def generate_scene_labels(wrapper, scene_token: str, cfg: LabelConfig) -> list[dict]:
    """scene 전체 keyframe 라벨 (위험 항 EMA + primary/aux + flicker 병합)."""
    toks = wrapper.sample_tokens_in_scene(scene_token)
    rows, risks, m_vecs = [], [], []
    for st in toks:
        risk, n_obj, min_ttc = compute_channel_scores(wrapper, st, cfg)
        maneuver = infer_sample_maneuver(wrapper, st, cfg)
        risks.append(risk)
        m_vecs.append(np.array(maneuver_view_vector(maneuver), dtype=np.float64))
        rows.append({
            "sample_token": st,
            "scene_token": scene_token,
            "timestamp": wrapper.sample_timestamp(st),
            "maneuver": maneuver,
            "layout": maneuver_to_layout_index(maneuver),
            "n_objects": n_obj,
            "min_ttc": float(min_ttc if min_ttc != TTC_INF else -1.0),  # inf→-1
        })

    # --- 위험 항만 EMA 스무딩하고 기동 항은 원값으로 더한다 ---
    ema_risk = smooth_risk(risks, cfg.ema_alpha)
    ema = [e + cfg.lambda_m * m for e, m in zip(ema_risk, m_vecs)]

    primary, attention = _resolve_views(rows, ema, cfg)

    # --- attention flicker 병합: A B A 패턴에서 가운데 B(1-keyframe)를 A로 ---
    if cfg.flicker_merge:
        for i in range(1, len(attention) - 1):
            if attention[i - 1] == attention[i + 1] and attention[i] != attention[i - 1]:
                attention[i] = attention[i - 1]

    # --- 보조 뷰 집합 + 단발 변경 병합 ---
    aux_seq = [select_aux_views(ema_risk[i], m_vecs[i], primary[i], cfg)
               for i in range(len(rows))]
    if cfg.flicker_merge:
        aux_seq = merge_aux_flicker(aux_seq)

    # --- 컬럼 채우기 ---
    for i, row in enumerate(rows):
        s_smooth = ema[i]
        soft = _softmax_excl(s_smooth, cfg.T_soft, primary[i])
        aux = aux_seq[i]
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
