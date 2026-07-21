"""기동(maneuver) 상태 추론 (CLAUDE.md B.4-(4)).

원리
----
±3초 윈도우의 ego 자세(위치 + 헤딩) 시퀀스로 현재 주행 기동을 규칙 판정한다.
클래스: STRAIGHT, TURN_LEFT/RIGHT, LANE_CHANGE_LEFT/RIGHT, REVERSE, STOP.

핵심 신호는 두 가지:
- 평균 속도 → STOP 판정.
- 헤딩(yaw) 변화 → 회전/차선변경 판정.
    net  = 윈도우 시작→끝 순 헤딩 변화 (부호 = 좌(+)/우(−))
    peak = 윈도우 내 최대 헤딩 이탈량
  회전은 net 자체가 크고(>25°), 차선변경은 잠깐 틀었다 되돌아오므로
  peak는 중간(5~25°)인데 net은 작다(<10°).

**미래 참조 허용**: 라벨 생성은 오프라인이므로 미래 pose를 봐도 된다(윈도우가
현재 시점 ±3초). 단, 모델 입력에는 과거만 넣는다 — 이 비대칭이 "선제적 전환
학습"의 핵심이다.

왜 이 프로젝트에서 쓰는가
-------------------------
기동 상태는 채널 적합도 m(c)(constants.MANEUVER_VIEW_MAP)와 레이아웃 템플릿을
결정한다. 예: REVERSE면 후방 카메라 가중, TURN_LEFT면 좌전방 가중.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ManeuverThresholds:
    """기동 판정 임계값 (config label.maneuver.* 로 오버라이드)."""
    stop_speed: float = 0.5          # m/s, 이하이면 STOP
    turn_deg: float = 25.0           # net 헤딩변화 이 이상이면 TURN
    lane_change_min_deg: float = 5.0  # peak 이탈 하한
    lane_change_max_deg: float = 25.0  # peak 이탈 상한
    lane_change_net_deg: float = 10.0  # net 이 이하이면 (되돌아옴) 차선변경
    reverse_dot: float = 0.0         # heading·velocity 내적 이 미만이면 REVERSE
    enable_lane_change: bool = True  # False면 차선변경을 TURN으로 흡수(v1 단순화)


def wrap_angle(a: float) -> float:
    """각도를 [-π, π]로 래핑."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def infer_maneuver(positions, yaws, timestamps=None, speeds=None,
                   gear: str | None = None,
                   th: ManeuverThresholds | None = None) -> str:
    """ego 자세 시퀀스 → 기동 클래스 이름.

    Args:
        positions: (T, 2+) 윈도우 내 ego 위치(수평 x,y 사용).
        yaws:      (T,) ego 헤딩(rad). positions와 같은 길이.
        timestamps:(T,) 초 단위 시각(속도 추정용, 없으면 등간격 가정).
        speeds:    (T,) m/s 속도. 주면 STOP/REVERSE에 우선 사용.
        gear:      CAN 기어 문자열('reverse' 등). 있으면 REVERSE 우선 판정.
        th:        임계값. None이면 기본값.

    Returns:
        constants.MANEUVERS 중 하나.
    """
    th = th or ManeuverThresholds()
    pos = np.asarray(positions, dtype=np.float64)[:, :2]
    yaw = np.asarray(yaws, dtype=np.float64).ravel()
    n = len(yaw)
    if n < 2:
        return "STRAIGHT"

    # --- 속도 추정 ---
    if speeds is not None:
        mean_speed = float(np.mean(np.abs(np.asarray(speeds, dtype=np.float64))))
    else:
        if timestamps is not None:
            dt = np.diff(np.asarray(timestamps, dtype=np.float64))
        else:
            dt = np.ones(n - 1)
        dt = np.where(dt <= 1e-6, 1e-6, dt)
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        mean_speed = float(np.mean(step / dt))

    # --- STOP (가장 먼저) ---
    if mean_speed < th.stop_speed:
        return "STOP"

    # --- REVERSE: 진행 방향이 헤딩 반대 ---
    if gear is not None and str(gear).lower().startswith("rev"):
        return "REVERSE"
    move = pos[-1] - pos[0]
    if np.linalg.norm(move) > 1e-3:
        heading_vec = np.array([np.cos(yaw[0]), np.sin(yaw[0])])
        if float(np.dot(heading_vec, move)) < th.reverse_dot:
            return "REVERSE"

    # --- 헤딩 변화 분석 ---
    dev = np.array([wrap_angle(y - yaw[0]) for y in yaw])  # 시작 대비 이탈(rad)
    net_deg = np.degrees(dev[-1])
    peak_idx = int(np.argmax(np.abs(dev)))
    peak_deg = np.degrees(dev[peak_idx])

    # --- TURN: net 이 큼 ---
    if abs(net_deg) > th.turn_deg:
        return "TURN_LEFT" if net_deg > 0 else "TURN_RIGHT"

    # --- LANE_CHANGE: peak 는 중간, net 은 되돌아옴 ---
    if th.enable_lane_change:
        if (th.lane_change_min_deg <= abs(peak_deg) <= th.lane_change_max_deg
                and abs(net_deg) < th.lane_change_net_deg):
            return "LANE_CHANGE_LEFT" if peak_deg > 0 else "LANE_CHANGE_RIGHT"

    return "STRAIGHT"
