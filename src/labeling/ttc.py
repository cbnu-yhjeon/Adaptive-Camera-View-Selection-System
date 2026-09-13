"""TTC(Time-To-Collision) 및 위험 가중 함수 (CLAUDE.md B.4-(2),(3)).

원리
----
TTC는 "지금 상대속도가 유지되면 몇 초 뒤 부딪히는가"이다. 두 정의를 지원한다.

**cpa (기본)** — 최접근(closest point of approach) 기반:

    p_rel = p_obj − p_ego,  v_rel = v_obj − v_ego
    t*    = −(p_rel·v_rel) / ‖v_rel‖²        (최접근 시각)
    DCA   = ‖p_rel + v_rel·t*‖               (그때의 거리)
    TTC   = t*   (접근 중일 때)  |  ∞ (멀어지는 중)

**radial** — 접근 속도만 보는 고전 정의 (ablation·비교용):

    v_r = −(p_rel·v_rel)/‖p_rel‖,  TTC = ‖p_rel‖ / v_r

왜 cpa가 기본인가
-----------------
radial 정의는 **옆으로 스쳐 지나갈 객체를 충돌 임박으로 오판한다.** ego가 10m/s로
주행하면 20m 앞 길가 주차 차량도 접근 속도가 생겨 TTC=2s가 나온다. 실측 결과
전체 위험량의 62.9%가 정지·주차 객체에서 나왔고, 이들을 빼면 선택 뷰가 29.4%의
프레임에서 뒤집혔다. cpa는 "실제로 얼마나 가까워지는가(DCA)"를 분리해내므로
스쳐 지나갈 객체를 경로 관련성 계수로 감쇠할 수 있다.

위험 가중
---------
    w = ρ_attr · ρ_path · max( exp(−TTC/τ), 근접보정 )

  - ρ_attr: 객체 속성 계수. 주차 차량처럼 움직이지 않는 객체를 낮춘다.
  - ρ_path: 경로 관련성 계수. DCA가 클수록(스쳐 지나갈수록) 낮춘다.
채널 점수 s(c) = Σ w + λ·m(c) 의 위험도 항으로 쓰인다.
"""

from __future__ import annotations

import numpy as np

TTC_INF = float("inf")


def closest_approach(p_obj, p_ego, v_obj, v_ego) -> tuple[float, float]:
    """최접근 시각 t*(초)와 그때의 거리 DCA(m).

    멀어지는 중이거나 상대속도가 0이면 지금이 최접근이므로 (0, 현재거리).
    """
    p_rel = _xy(p_obj) - _xy(p_ego)
    v_rel = _xy(v_obj) - _xy(v_ego)
    r = float(np.linalg.norm(p_rel))
    vv = float(np.dot(v_rel, v_rel))
    if vv < 1e-9:                       # 상대정지 — 거리 변화 없음
        return 0.0, r
    t = -float(np.dot(p_rel, v_rel)) / vv
    if t <= 0.0:                        # 이미 멀어지는 중
        return 0.0, r
    return t, float(np.linalg.norm(p_rel + v_rel * t))


def compute_ttc(p_obj, p_ego, v_obj, v_ego,
                v_r_min: float = 0.1, mode: str = "cpa") -> float:
    """TTC(초). 접근 중이 아니면 inf.

    입력은 2D(수평 x,y) 또는 3D 모두 허용하며, 거리는 수평면 기준으로 계산한다.
    mode="cpa"는 최접근 시각을, "radial"은 고전적 접근속도 기반 값을 돌려준다.
    """
    p_rel = _xy(p_obj) - _xy(p_ego)     # ego→obj
    v_rel = _xy(v_obj) - _xy(v_ego)     # 객체의 상대속도
    r = float(np.linalg.norm(p_rel))
    if r < 1e-6:
        return 0.0                      # 완전 겹침 → 즉시 충돌

    v_r = -float(np.dot(p_rel, v_rel)) / r
    if v_r <= v_r_min:                  # 접근 중이 아님(두 정의 공통)
        return TTC_INF
    if mode == "radial":
        return r / v_r
    if mode == "cpa":
        t, _ = closest_approach(p_obj, p_ego, v_obj, v_ego)
        return t if t > 0.0 else 0.0
    raise ValueError(f"알 수 없는 ttc mode: {mode}")


def near_floor(p_obj, p_ego, v_ego, r: float, w_min_near: float = 0.3,
               near_range: float = 10.0, touch_range: float = 2.0,
               stop_speed: float = 0.5) -> float:
    """근접 보정의 실효 하한 — **진행 방향 쪽 객체에만** 적용한다.

    근접 보정은 "저속으로 다가가다 부딪힐 수 있는 객체"를 놓치지 않으려는 장치다
    (후진·주차). 방향 조건 없이 r<near_range 전부에 적용하면, 이미 지나쳐 뒤에
    남은 정지 객체에도 최소 가중이 붙어 후방 채널이 켜진다.

    실측 사례(scene-0533): ego가 1.7m/s로 전진 중인데 뒤쪽 3.6m 주차 차량과 6.3m
    차량이 근접 보정만으로 w=0.145/0.140을 받아 BACK_RIGHT·BACK이 보조 뷰로 뽑혔다.
    둘 다 TTC=inf(접근하지 않음)였다.

    규칙: 접촉 거리(touch_range) 이내면 방향 무관, 그 밖에서는 진행 방향 반평면에
    있는 객체만. 정지 중(|v_ego| < stop_speed)이면 내가 부딪힐 일이 없으므로 미적용.
    """
    if r >= near_range:
        return 0.0
    if r < touch_range:
        return w_min_near                      # 사실상 접촉 — 방향 무관
    v = _xy(v_ego)
    if float(np.linalg.norm(v)) < stop_speed:
        return 0.0                             # 정지 중 — 내가 다가가지 않는다
    p_rel = _xy(p_obj) - _xy(p_ego)
    return w_min_near if float(np.dot(p_rel, v)) > 0.0 else 0.0


def attr_path_gate(dca: float, d_touch: float = 1.5, d_scale: float = 2.0) -> float:
    """경로 위 객체에 대해 상태 감쇠를 해제하는 정도 ∈ [0, 1] (1 = 감쇠 없음).

    ρ_attr(주차 0.2)와 ρ_path를 그냥 곱하면 **경로 위의 주차 차량이 이중으로 깎인다.**
    주차 감쇠는 길가에 늘어서서 스쳐 지나갈 차를 위한 것이고 그건 ρ_path가 이미
    처리한다. 최접근 거리가 차체 폭 수준이면 주차 여부와 무관하게 부딪힐 대상이므로
    감쇠를 풀어야 한다 — 특히 그쪽으로 차선변경 중이라면.

    실측 사례(scene-0526): 7.6m 앞 DCA 3.1m 주차 SUV가 ρ_attr 0.2에 눌려 채널 위험이
    0.097이 되면서 임계값 0.10을 0.003 차이로 넘지 못했다.
    """
    return float(np.exp(-max(0.0, float(dca) - d_touch) / max(d_scale, 1e-6)))


def path_factor(dca: float, d_near: float = 2.0, d_scale: float = 2.0,
                rho_min: float = 0.05) -> float:
    """경로 관련성 계수 ρ_path ∈ [rho_min, 1].

    최접근 거리(DCA)가 d_near 이내면 1(정면 충돌 경로), 그보다 멀면 지수 감쇠한다.
    "옆으로 3m 떨어져 스쳐 지나갈 객체"와 "정면으로 다가오는 객체"를 구분하는 항.
    """
    if not np.isfinite(dca):
        return rho_min
    excess = max(0.0, float(dca) - d_near)
    return float(np.clip(np.exp(-excess / max(d_scale, 1e-6)), rho_min, 1.0))


def horizontal_range(p_obj, p_ego) -> float:
    """ego와 객체의 수평면 거리 r(m)."""
    return float(np.linalg.norm(_xy(p_obj) - _xy(p_ego)))


def risk_weight(ttc: float, r: float, tau: float = 4.0,
                w_type: str = "exp", w_min_near: float = 0.3,
                near_range: float = 10.0, rho: float = 1.0) -> float:
    """위험 가중 w ∈ [0, 1] (CLAUDE.md B.4-(3)).

    Args:
        ttc: compute_ttc 결과(초). inf 가능.
        r:   수평 거리(m) — 근접 보정에 사용.
        tau: exp 감쇠 시간상수(기본 4.0s).
        w_type: "exp" → exp(−TTC/tau) | "inv" → clip(1/TTC, 0, 1).
        w_min_near: r < near_range 인 근접 객체의 최소 가중(저속 근접 누락 방지).
        near_range: 근접 보정 거리 임계(기본 10m).
        rho: 객체 속성·경로 관련성 계수(ρ_attr·ρ_path). 근접 보정에도 함께 적용되어,
             바로 옆 주차 차량이 최소 가중만으로 채널을 지배하지 않게 한다.
    """
    if ttc == TTC_INF:
        w = 0.0
    elif w_type == "exp":
        w = float(np.exp(-ttc / tau))
    elif w_type == "inv":
        w = float(np.clip(1.0 / ttc, 0.0, 1.0)) if ttc > 0 else 1.0
    else:
        raise ValueError(f"알 수 없는 w_type: {w_type}")

    # 근접 보정: 후진/주차 등 저속 근접 객체가 TTC=inf여도 최소 가중 부여
    if r < near_range:
        w = max(w, w_min_near)
    return float(np.clip(rho * w, 0.0, 1.0))


def _xy(p) -> np.ndarray:
    """수평면 좌표(x, y)만 추출."""
    a = np.asarray(p, dtype=np.float64).ravel()
    return a[:2]
