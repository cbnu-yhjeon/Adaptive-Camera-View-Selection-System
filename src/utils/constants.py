"""프로젝트 전역 상수 — 단일 진실 공급원.

CLAUDE.md 원칙: "하드코딩 금지 — 경로, 임계값, 채널 순서 참조는 전부
config 또는 상수 모듈 경유". 채널 순서·기동 클래스·레이아웃 템플릿처럼
라벨·모델·평가가 반드시 동일하게 참조해야 하는 값은 전부 여기에 못 박는다.

여기 정의된 순서가 어긋나면 라벨의 채널 3번과 모델 출력의 채널 3번이
서로 다른 카메라를 가리키게 되어 실험 전체가 조용히 오염된다.
"""

from __future__ import annotations

# =====================================================================
# 카메라 채널 순서 — 전 구간 고정 (CLAUDE.md B.2)
#   라벨, 모델 출력, 평가 모두 이 인덱스 순서를 따른다.
# =====================================================================
CAMERAS: list[str] = [
    "CAM_FRONT",        # 0
    "CAM_FRONT_RIGHT",  # 1
    "CAM_BACK_RIGHT",   # 2
    "CAM_BACK",         # 3
    "CAM_BACK_LEFT",    # 4
    "CAM_FRONT_LEFT",   # 5
]
NUM_CAMERAS: int = len(CAMERAS)                       # 6
CAMERA_INDEX: dict[str, int] = {c: i for i, c in enumerate(CAMERAS)}


# =====================================================================
# 기동(maneuver) 클래스 (CLAUDE.md B.4-(4))
# =====================================================================
MANEUVERS: list[str] = [
    "STRAIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "LANE_CHANGE_LEFT",
    "LANE_CHANGE_RIGHT",
    "REVERSE",
    "STOP",
]
NUM_MANEUVERS: int = len(MANEUVERS)                   # 7
MANEUVER_INDEX: dict[str, int] = {m: i for i, m in enumerate(MANEUVERS)}


# =====================================================================
# CAN 특징 정규화 상수 (모델 입력 can 토큰용)
#   순서: [speed, steering, accel_x, accel_y, yaw_rate, left_signal, right_signal]
#   대략적인 스케일로 나눠 대략 [-1,1]에 오도록 (정확할 필요는 없음).
#
#   방향지시등은 운전자의 **의도**라 기동보다 먼저 켜진다. 기동 라벨이 최대 5초 앞서
#   붙으므로(선제적 전환) 과거 2초만 보는 모델에게 가장 직접적인 단서다.
#   라벨 판정에는 쓰지 않는다 — 라벨과 입력이 같은 신호를 공유하면 모델이 영상·위험도
#   대신 지시등만 베끼는 지름길을 배운다(원칙 1과 같은 구조). 라벨 검증 지표로만 쓴다.
#   실측: TURN_LEFT의 83%, TURN_RIGHT의 75%가 같은 방향 지시등과 일치(반대 <2%).
# =====================================================================
CAN_FEATURES: list[str] = ["speed", "steering", "accel_x", "accel_y", "yaw_rate",
                           "left_signal", "right_signal"]
CAN_NORM: dict[str, float] = {
    "speed": 30.0,      # m/s
    "steering": 600.0,  # steeranglefeedback value
    "accel_x": 10.0,    # m/s^2
    "accel_y": 10.0,
    "yaw_rate": 2.0,    # rad/s
    "left_signal": 1.0,   # 0/1
    "right_signal": 1.0,  # 0/1
}
D_CAN: int = len(CAN_FEATURES) + 1                    # +1 = CAN 결측 마스크


# =====================================================================
# 기동 → 채널 적합도 m(c) (CLAUDE.md B.4-(5))
#   각 기동 상태에서 "봐야 할 카메라"에 부여하는 가중치.
#   config(label.maneuver_view_map)로 오버라이드 가능하나, 기본값은 여기.
#   키: 카메라 이름, 값: 가중치. 명시되지 않은 채널은 0.
# =====================================================================
MANEUVER_VIEW_MAP: dict[str, dict[str, float]] = {
    "STRAIGHT":          {"CAM_FRONT": 1.0},
    "TURN_LEFT":         {"CAM_FRONT_LEFT": 1.0, "CAM_FRONT": 0.5, "CAM_BACK_LEFT": 0.3},
    "TURN_RIGHT":        {"CAM_FRONT_RIGHT": 1.0, "CAM_FRONT": 0.5, "CAM_BACK_RIGHT": 0.3},
    "LANE_CHANGE_LEFT":  {"CAM_BACK_LEFT": 1.0, "CAM_FRONT_LEFT": 0.5, "CAM_FRONT": 0.5},
    "LANE_CHANGE_RIGHT": {"CAM_BACK_RIGHT": 1.0, "CAM_FRONT_RIGHT": 0.5, "CAM_FRONT": 0.5},
    "REVERSE":           {"CAM_BACK": 1.0, "CAM_BACK_LEFT": 0.5, "CAM_BACK_RIGHT": 0.5},
    "STOP":              {"CAM_FRONT": 0.5},  # 위험도 항이 지배하도록 낮게
}


# =====================================================================
# 레이아웃 템플릿 (K=5) (CLAUDE.md B.4-(6))
#   기동 클래스 → 화면 배치 템플릿의 결정적 매핑.
#   v1은 단순화: 기동에서 바로 템플릿을 정한다.
# =====================================================================
# nuScenes 주석 상태(attribute) → 위험 가중 계수 ρ_attr.
# 실측: 주차 차량이 전체 객체의 46.6% / 위험량의 42.9%를 차지하고, 정지류 전체가
# 위험량의 62.9%였다. 완전히 제외하지는 않는다 — 주차 차량도 문 열림·보행자 가림
# 위험이 있고, 정지 차량과 서 있는 보행자는 곧 움직일 수 있다.
DEFAULT_ATTR_WEIGHT: dict[str, float] = {
    "vehicle.parked":                0.2,
    "vehicle.stopped":               0.5,   # 신호 대기 등 — 곧 출발할 수 있음
    "pedestrian.standing":           0.5,   # 갑자기 도로로 진입할 수 있음
    "pedestrian.sitting_lying_down": 0.3,
    "cycle.without_rider":           0.2,   # 세워둔 자전거
}


def attribute_weight(attribute: str, table: dict[str, float] | None = None) -> float:
    """객체 상태 → ρ_attr (기본 1.0 = 움직이는 객체)."""
    return (table or DEFAULT_ATTR_WEIGHT).get(attribute, 1.0)


LAYOUTS: list[str] = [
    "FRONT_CENTRIC",   # 0 ①전방 중심
    "TURN_LEFT",       # 1 ②좌회전형
    "TURN_RIGHT",      # 2 ③우회전형
    "REAR",            # 3 ④후방형
    "LANE_CHANGE",     # 4 ⑤차선변경형
]
NUM_LAYOUTS: int = len(LAYOUTS)                       # 5
LAYOUT_INDEX: dict[str, int] = {l: i for i, l in enumerate(LAYOUTS)}

# 기동 클래스 → 레이아웃 이름
LAYOUT_MAP: dict[str, str] = {
    "STRAIGHT":          "FRONT_CENTRIC",
    "STOP":              "FRONT_CENTRIC",
    "TURN_LEFT":         "TURN_LEFT",
    "TURN_RIGHT":        "TURN_RIGHT",
    "REVERSE":           "REAR",
    "LANE_CHANGE_LEFT":  "LANE_CHANGE",
    "LANE_CHANGE_RIGHT": "LANE_CHANGE",
}


def maneuver_to_layout_index(maneuver: str) -> int:
    """기동 클래스 이름 → 레이아웃 인덱스(0~4)."""
    return LAYOUT_INDEX[LAYOUT_MAP[maneuver]]


def travel_direction_view(maneuver: str) -> int:
    """진행 방향 뷰 인덱스 — 이동체는 진행 방향을 항상 주시해야 한다.

    후진(REVERSE)이면 CAM_BACK, 그 외(정지 포함)는 CAM_FRONT.
    이 뷰가 primary_view로 고정되어 UI 메인 위치에 상시 표시된다.
    """
    return CAMERA_INDEX["CAM_BACK"] if maneuver == "REVERSE" else CAMERA_INDEX["CAM_FRONT"]


def maneuver_view_vector(maneuver: str) -> list[float]:
    """기동 클래스 → 채널 순서(CAMERAS)에 맞춘 적합도 벡터 m(c) 6개.

    label_function 의 s(c) = Σ w(TTC) + λ·m(c) 에서 m(c) 항으로 쓰인다.
    """
    weights = MANEUVER_VIEW_MAP[maneuver]
    return [weights.get(cam, 0.0) for cam in CAMERAS]
