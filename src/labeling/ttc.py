"""TTC(Time-To-Collision) 및 위험 가중 함수 (CLAUDE.md B.4-(2),(3)).

원리
----
TTC는 "지금 상대속도가 유지되면 몇 초 뒤 부딪히는가"이다. radial(반경) 정의:

    r   = ‖p_obj − p_ego‖                    (수평면 거리)
    v_r = −(p_obj − p_ego)·(v_obj − v_ego)/r  (접근 속도, 양수 = 접근 중)
    TTC = r / v_r      (v_r > 0.1 m/s 일 때)
        = ∞            (그 외 — 멀어지거나 정지)

v_r 부호 규약: (p_obj − p_ego)는 ego→obj 방향. 상대속도 (v_obj − v_ego)를
그 방향에 투영해 부호를 뒤집으면, 객체가 다가올 때 v_r > 0 이 된다.

위험 가중 w(TTC)는 TTC가 짧을수록 1에 가깝게, 길수록 0에 가깝게 만든다.
채널 점수 s(c) = Σ w(TTC_i) + λ·m(c) 의 위험도 항으로 쓰인다.

왜 이 프로젝트에서 쓰는가
-------------------------
"어느 카메라가 지금 중요한가"의 핵심 근거가 위험 객체다. TTC가 작은(곧 충돌
위험) 객체가 많은 카메라일수록 s(c)가 커져 메인 뷰로 선택된다.
"""

from __future__ import annotations

import numpy as np

TTC_INF = float("inf")


def compute_ttc(p_obj, p_ego, v_obj, v_ego,
                v_r_min: float = 0.1) -> float:
    """radial TTC(초). 접근 중이 아니면 inf.

    입력은 2D(수평 x,y) 또는 3D 모두 허용하며, 거리는 수평면 기준으로 계산한다.
    """
    p_rel = _xy(p_obj) - _xy(p_ego)     # ego→obj
    v_rel = _xy(v_obj) - _xy(v_ego)     # 객체의 상대속도
    r = float(np.linalg.norm(p_rel))
    if r < 1e-6:
        return 0.0                      # 완전 겹침 → 즉시 충돌
    v_r = -float(np.dot(p_rel, v_rel)) / r
    if v_r > v_r_min:
        return r / v_r
    return TTC_INF


def horizontal_range(p_obj, p_ego) -> float:
    """ego와 객체의 수평면 거리 r(m)."""
    return float(np.linalg.norm(_xy(p_obj) - _xy(p_ego)))


def risk_weight(ttc: float, r: float, tau: float = 4.0,
                w_type: str = "exp", w_min_near: float = 0.3,
                near_range: float = 10.0) -> float:
    """위험 가중 w ∈ [0, 1] (CLAUDE.md B.4-(3)).

    Args:
        ttc: compute_ttc 결과(초). inf 가능.
        r:   수평 거리(m) — 근접 보정에 사용.
        tau: exp 감쇠 시간상수(기본 4.0s).
        w_type: "exp" → exp(−TTC/tau) | "inv" → clip(1/TTC, 0, 1).
        w_min_near: r < near_range 인 근접 객체의 최소 가중(저속 근접 누락 방지).
        near_range: 근접 보정 거리 임계(기본 10m).
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
    return float(np.clip(w, 0.0, 1.0))


def _xy(p) -> np.ndarray:
    """수평면 좌표(x, y)만 추출."""
    a = np.asarray(p, dtype=np.float64).ravel()
    return a[:2]
