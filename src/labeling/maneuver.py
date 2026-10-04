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
    turn_deg: float = 25.0           # net 헤딩변화 이 이상이면 TURN 후보
    turn_radius_max: float = 30.0    # 선회 반경 이 이상이면 곡선로 주행 → TURN 아님
    lane_change_min_deg: float = 5.0  # peak 이탈 하한
    lane_change_max_deg: float = 25.0  # peak 이탈 상한
    lane_change_net_deg: float = 10.0  # net 이 이하이면 (되돌아옴) 차선변경
    lane_change_resid_min: float = 2.0  # 곡선 성분을 뺀 횡이탈 하한(m) — 맵이 없을 때
    lane_change_resid_min_map: float = 1.0  # 맵이 차로 ID 변경을 확인했을 때의 하한(경계 떨림 제거용)
    reverse_dot: float = 0.0         # heading·velocity 내적 이 미만이면 REVERSE
    enable_lane_change: bool = True  # False면 차선변경을 TURN으로 흡수(v1 단순화)


def wrap_angle(a: float) -> float:
    """각도를 [-π, π]로 래핑."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def infer_maneuver(positions, yaws, timestamps=None, speeds=None,
                   gear: str | None = None,
                   th: ManeuverThresholds | None = None,
                   check_state: bool = True,
                   on_lane=None,
                   map_lane_change: bool | None = None) -> str:
    """ego 자세 시퀀스 → 기동 클래스 이름.

    Args:
        positions: (T, 2+) 윈도우 내 ego 위치(수평 x,y 사용).
        yaws:      (T,) ego 헤딩(rad). positions와 같은 길이.
        timestamps:(T,) 초 단위 시각(속도 추정용, 없으면 등간격 가정).
        speeds:    (T,) m/s 속도. 주면 STOP/REVERSE에 우선 사용.
        gear:      CAN 기어 문자열('reverse' 등). 있으면 REVERSE 우선 판정.
        th:        임계값. None이면 기본값.
        check_state: False면 STOP/REVERSE 판정을 건너뛰고 기동(TURN/LANE_CHANGE)만 본다.
                   STOP·REVERSE는 "지금의 상태"라 현재 기준 짧은 창으로 따로 판정해야 하는데,
                   선제 창(미래 5초)으로 보면 5초 안에 출발·전진하는 프레임이 정지·후진에서
                   빠진다(검수 카드 16: 후진 중인데 TURN_LEFT로 바뀌어 진행 방향이 전방이 됐다).
        on_lane:   (T,) 프레임마다 ego가 **일반 차로 위**(교차로·연결부·주차장 밖)에 있는지.
                   주면 TURN을 "일반 차로 밖에서 튼 각도"로 판정한다. 없으면 선회 반경으로 근사.
        map_lane_change: 맵이 판정한 차선변경 여부 — 진행 방향으로 이어지지 않은 나란한
                   차로로 차로 ID가 바뀌었는가. 주면 궤적만의 횡이탈 기준 대신 이것을 쓴다.

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

    if check_state:
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

    # --- 궤적 분해: 시작 헤딩 기준 종/횡 성분 ---
    # 차선변경의 본질은 "옆으로 한 차로 옮기는 것"이다. 헤딩 편차는 그 부산물이라
    # 완만한 곡선로 주행과 섞인다(실측: 헤딩 기준 LANE_CHANGE 라벨의 횡변위 중앙값이
    # 1.3m로 차선폭 3.5m에 한참 못 미쳤다). 횡변위를 직접 재서 가른다.
    head = np.array([np.cos(yaw[0]), np.sin(yaw[0])])
    normal = np.array([-head[1], head[0]])          # 좌측(+) 법선
    lat = (pos - pos[0]) @ normal
    lat_end = float(lat[-1])
    travel = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))

    # 곡선 성분 제거. 시작 헤딩 기준 횡변위는 **차로를 그대로 따라가도** 곡선로에서
    # 커진다 — 반경 R을 θ만큼 돌면 수직 이탈이 R(1−cosθ)다. 실측 사례(scene-0105)는
    # 횡변위 1.81m 중 1.87m가 곡선 성분이어서 실제 차로 이탈이 0에 가까웠는데도
    # 차선변경으로 라벨됐다. 등곡률 원호가 만드는 몫을 빼고 남는 잔차만 센다.
    theta = np.radians(net_deg)
    radius = travel / abs(theta) if abs(theta) > 1e-3 else np.inf
    lat_arc = (np.sign(theta) * radius * (1 - np.cos(theta))
               if np.isfinite(radius) else 0.0)
    lat_resid = lat_end - float(lat_arc)

    # --- TURN ---
    if on_lane is not None and len(on_lane) == n:
        # 맵 기준: 일반 차로 위에서 바뀐 헤딩은 곡선로를 따라간 것이고, 회전은 교차로·
        # 연결부에서 일어난다. 반경으로 근사하면 큰 교차로에서 크게 도는 회전을 곡선로로
        # 버렸다 — 반경 조건으로 STRAIGHT가 된 프레임은 방향 전환의 중앙값 33°가 교차로
        # 안에서 일어났고 일반 차로 위에서는 1°였다. 반대로 scene-0054(반경 43 m)는
        # −25.5° 전부가 일반 차로 위였다(곡선로).
        turn_area = 0.0
        for i in range(1, n):
            if not (on_lane[i - 1] and on_lane[i]):
                turn_area += np.degrees(wrap_angle(yaw[i] - yaw[i - 1]))
        if abs(turn_area) > th.turn_deg:
            return "TURN_LEFT" if turn_area > 0 else "TURN_RIGHT"
        if abs(net_deg) > th.turn_deg:
            return "STRAIGHT"   # 일반 차로 위의 곡선 — 사각지대가 생기지 않는다
    elif abs(net_deg) > th.turn_deg:
        # 맵이 없을 때: 선회 반경으로 교차로 회전(R 6~25 m)과 곡선로를 근사한다.
        if radius <= th.turn_radius_max:
            return "TURN_LEFT" if net_deg > 0 else "TURN_RIGHT"
        return "STRAIGHT"

    # --- LANE_CHANGE ---
    if th.enable_lane_change:
        if map_lane_change is not None:
            # 맵 기준: 이어지지 않은 나란한 차로로 차로 ID가 바뀌어야 한다. 궤적만으로는
            # 교차로를 지나며 생기는 옆 이동이 차선변경으로 잡혔다 — 궤적 기준 LANE_CHANGE
            # 라벨 중 맵으로 확인되는 것은 15.8%뿐이었고, 88.5%가 창 안에서 교차로를 지났다.
            # 횡이탈 하한은 차로 경계에서 ID가 떨리는 경우만 거른다.
            if (map_lane_change and abs(lat_resid) >= th.lane_change_resid_min_map
                    and abs(net_deg) < th.lane_change_net_deg):
                return "LANE_CHANGE_LEFT" if lat_resid > 0 else "LANE_CHANGE_RIGHT"
        elif (abs(lat_resid) >= th.lane_change_resid_min
                and abs(net_deg) < th.lane_change_net_deg
                and abs(peak_deg) <= th.lane_change_max_deg):
            return "LANE_CHANGE_LEFT" if lat_resid > 0 else "LANE_CHANGE_RIGHT"

    return "STRAIGHT"
