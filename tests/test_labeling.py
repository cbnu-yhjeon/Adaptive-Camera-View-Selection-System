"""ttc.py / maneuver.py 단위테스트 (합성 데이터, 실 nuScenes 불필요).

실행: `python tests/test_labeling.py`  (pytest 없이도 동작)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.labeling.ttc import (  # noqa: E402
    TTC_INF, closest_approach, compute_ttc, path_factor, risk_weight,
)
from src.labeling.maneuver import ManeuverThresholds, infer_maneuver, wrap_angle  # noqa: E402
from src.utils.constants import attribute_weight  # noqa: E402


# ============ 위험 재정의: 스쳐 지나갈 객체 vs 충돌 경로 객체 ============
def test_cpa_separates_passing_from_colliding():
    """전방 20m·측방 3m 지점의 객체 — 주차면 스쳐 가고, 횡단 중이면 충돌 경로.

    radial TTC는 둘을 구분하지 못한다(둘 다 약 2s). 이것이 주차 차량이 위험량의
    42.9%를 차지하던 원인이다.
    """
    p_obj, p_ego, v_ego = [20, 3], [0, 0], [10, 0]
    parked = compute_ttc(p_obj, p_ego, [0, 0], v_ego, mode="cpa")
    crossing = compute_ttc(p_obj, p_ego, [0, -1.5], v_ego, mode="cpa")
    assert abs(parked - 2.0) < 0.1 and abs(crossing - 2.0) < 0.1   # 시각은 비슷하지만
    _, dca_parked = closest_approach(p_obj, p_ego, [0, 0], v_ego)
    _, dca_cross = closest_approach(p_obj, p_ego, [0, -1.5], v_ego)
    assert dca_parked > 2.5 and dca_cross < 0.5                     # 최접근 거리가 갈린다


def test_path_factor_attenuates_by_closest_distance():
    assert path_factor(1.0) == 1.0                    # 정면 경로 → 감쇠 없음
    assert path_factor(4.0) < path_factor(2.5) < 1.0  # 멀수록 감쇠
    assert path_factor(50.0) >= 0.05                  # 하한 유지


def test_rho_scales_risk_weight_including_near_floor():
    """근접 보정(w_min_near)에도 ρ가 곱해져야 바로 옆 주차 차량이 채널을 지배하지 않는다."""
    near_parked = risk_weight(TTC_INF, r=5.0, rho=0.2)
    near_moving = risk_weight(TTC_INF, r=5.0, rho=1.0)
    assert abs(near_moving - 0.3) < 1e-9
    assert abs(near_parked - 0.06) < 1e-9


def test_attribute_weight_table():
    assert attribute_weight("vehicle.parked") < attribute_weight("vehicle.stopped")
    assert attribute_weight("vehicle.moving") == 1.0
    assert attribute_weight("") == 1.0                # 속성 없음 → 감쇠하지 않음(안전측)


# ======================= TTC =======================
def test_ttc_head_on_approach():
    """전방 20m 객체가 10m/s로 정면 접근, ego 정지 → TTC=2s."""
    ttc = compute_ttc(p_obj=[20, 0], p_ego=[0, 0], v_obj=[-10, 0], v_ego=[0, 0])
    assert abs(ttc - 2.0) < 1e-6


def test_ttc_receding_is_inf():
    """멀어지는 객체 → TTC=inf."""
    ttc = compute_ttc(p_obj=[20, 0], p_ego=[0, 0], v_obj=[+10, 0], v_ego=[0, 0])
    assert ttc == TTC_INF


def test_ttc_ego_catching_up():
    """ego가 앞차를 따라잡음: 앞차 5m/s, ego 15m/s, 거리 10m → 상대접근 10m/s → TTC=1s."""
    ttc = compute_ttc(p_obj=[10, 0], p_ego=[0, 0], v_obj=[5, 0], v_ego=[15, 0])
    assert abs(ttc - 1.0) < 1e-6


def test_ttc_parallel_no_approach():
    """나란히 같은 속도로 주행 → 접근 성분 0 → inf."""
    ttc = compute_ttc(p_obj=[0, 5], p_ego=[0, 0], v_obj=[10, 0], v_ego=[10, 0])
    assert ttc == TTC_INF


def test_risk_weight_monotonic():
    """TTC가 짧을수록 위험 가중이 크다 (exp)."""
    w_near = risk_weight(1.0, r=20.0)
    w_far = risk_weight(8.0, r=20.0)
    assert 0.0 < w_far < w_near <= 1.0


def test_risk_weight_inf_is_zero_when_far():
    """먼 곳(근접보정 밖) + TTC=inf → 위험 0."""
    assert risk_weight(TTC_INF, r=30.0) == 0.0


def test_risk_weight_near_correction():
    """근접(r<10m) 객체는 TTC=inf여도 최소 가중 0.3 보장."""
    assert risk_weight(TTC_INF, r=5.0, w_min_near=0.3) >= 0.3


# ===================== Maneuver =====================
def _straight_track(n=13, speed=10.0, dt=0.5):
    """직진 궤적: x축으로 등속 전진, yaw=0."""
    t = np.arange(n) * dt
    pos = np.stack([speed * t, np.zeros(n)], axis=1)
    yaw = np.zeros(n)
    return pos, yaw, t


def test_maneuver_straight():
    pos, yaw, t = _straight_track()
    assert infer_maneuver(pos, yaw, timestamps=t) == "STRAIGHT"


def test_maneuver_stop():
    """거의 정지 (속도 < 0.5 m/s)."""
    pos, yaw, t = _straight_track(speed=0.1)
    assert infer_maneuver(pos, yaw, timestamps=t) == "STOP"


def _arc_track(total_deg: float, radius: float, speed: float = 6.0,
               n: int = 13, dt: float = 0.5):
    """반경 radius로 total_deg만큼 도는 원호 궤적 — 헤딩과 위치가 물리적으로 일치한다.

    헤딩만 흔들고 위치는 직선인 픽스처를 쓰면 선회 반경이 0이 되어 실제 주행과
    다른 것을 검증하게 된다. 판정이 반경·횡변위를 보므로 궤적을 제대로 만든다.
    """
    t = np.arange(n) * dt
    omega = np.radians(total_deg) / (t[-1] or 1.0)      # rad/s
    yaw = omega * t
    # 원호 적분: 반경 = speed / |omega| 가 되도록 speed를 맞춘다
    v = abs(omega) * radius
    x = np.concatenate([[0.0], np.cumsum(v * dt * np.cos(yaw[:-1]))])
    y = np.concatenate([[0.0], np.cumsum(v * dt * np.sin(yaw[:-1]))])
    return np.stack([x, y], axis=1), yaw, t


def test_maneuver_turn_left():
    """좌회전: 교차로 수준의 작은 반경(12 m)으로 +60°."""
    pos, yaw, t = _arc_track(+60, radius=12.0)
    assert infer_maneuver(pos, yaw, timestamps=t) == "TURN_LEFT"


def test_maneuver_turn_right():
    pos, yaw, t = _arc_track(-60, radius=12.0)
    assert infer_maneuver(pos, yaw, timestamps=t) == "TURN_RIGHT"


def test_maneuver_curve_is_not_turn():
    """완만한 곡선로 주행(R=80 m)은 헤딩이 40° 변해도 TURN이 아니다.

    차로를 벗어나지 않으므로 사각지대가 생기지 않는다 — 실측에서 이 구분이 없어
    TURN 라벨의 43%가 R>30 m인 곡선 주행이었다.
    """
    pos, yaw, t = _arc_track(+40, radius=80.0)
    assert infer_maneuver(pos, yaw, timestamps=t) == "STRAIGHT"


def test_maneuver_lane_change_left():
    """좌 차선변경: 헤딩은 되돌아오고 **옆으로 한 차로(3.5 m)** 옮긴다."""
    n, dt = 13, 0.5
    t = np.arange(n) * dt
    half = n // 2
    yaw = np.concatenate([
        np.linspace(0, np.radians(15), half),
        np.linspace(np.radians(15), 0, n - half),
    ])
    # 횡변위: S자로 3.5 m 좌측 이동 (전진 속도 10 m/s)
    lat = 3.5 * (1 - np.cos(np.pi * t / (t[-1] or 1.0))) / 2
    pos = np.stack([10 * t, lat], axis=1)
    assert infer_maneuver(pos, yaw, timestamps=t) == "LANE_CHANGE_LEFT"


def test_maneuver_gentle_curve_is_not_lane_change():
    """완만한 곡선로(R=200 m)는 차로를 그대로 따라가도 횡변위가 2 m 가까이 나온다.

    시작 헤딩 기준 횡변위는 곡선에서 R(1−cosθ)만큼 자란다. 생(raw) 횡변위로 판정하면
    이런 프레임이 전부 차선변경이 된다 — 실측 scene-0105가 그랬다(횡변위 1.81 m 중
    1.87 m가 곡선 성분). 곡선 성분을 뺀 잔차는 0에 가까우므로 STRAIGHT여야 한다.
    """
    pos, yaw, t = _arc_track(+8, radius=200.0, n=13, dt=0.5)
    # 생 횡변위는 차로 폭 절반을 넘지만
    h = np.array([np.cos(yaw[0]), np.sin(yaw[0])])
    n_vec = np.array([-h[1], h[0]])
    raw_lat = abs(float(((pos - pos[0]) @ n_vec)[-1]))
    assert raw_lat > 1.75, f"픽스처가 의도와 다르다 (횡변위 {raw_lat:.2f} m)"
    # 잔차 기준에서는 차선변경이 아니다
    assert infer_maneuver(pos, yaw, timestamps=t) == "STRAIGHT"


def test_maneuver_lane_keeping_is_not_lane_change():
    """헤딩만 흔들리고 옆으로 움직이지 않으면 차선변경이 아니다.

    횡변위 없이 헤딩 편차만으로 판정하면 차로 유지 중의 흔들림이 전부
    차선변경이 된다 — 실측 LANE_CHANGE 라벨의 횡변위 중앙값이 1.3 m였다.
    """
    n, dt = 13, 0.5
    t = np.arange(n) * dt
    half = n // 2
    yaw = np.concatenate([
        np.linspace(0, np.radians(15), half),
        np.linspace(np.radians(15), 0, n - half),
    ])
    pos = np.stack([10 * t, np.zeros(n)], axis=1)      # 횡변위 0
    assert infer_maneuver(pos, yaw, timestamps=t) == "STRAIGHT"


def test_maneuver_reverse():
    """후진: 헤딩은 +x인데 실제로는 −x로 이동."""
    n, dt = 13, 0.5
    t = np.arange(n) * dt
    pos = np.stack([-5 * t, np.zeros(n)], axis=1)  # 뒤로 이동
    yaw = np.zeros(n)                              # 정면 응시
    assert infer_maneuver(pos, yaw, timestamps=t) == "REVERSE"


def test_maneuver_gear_reverse_priority():
    """CAN gear=reverse면 즉시 REVERSE (단, STOP 아닌 경우)."""
    pos, yaw, t = _straight_track()
    assert infer_maneuver(pos, yaw, timestamps=t, gear="reverse") == "REVERSE"


def test_maneuver_lane_change_disabled_absorbs_to_straight():
    """enable_lane_change=False면 작은 흔들림은 STRAIGHT로."""
    n, dt = 13, 0.5
    t = np.arange(n) * dt
    half = n // 2
    yaw = np.concatenate([
        np.linspace(0, np.radians(15), half),
        np.linspace(np.radians(15), 0, n - half),
    ])
    pos = np.stack([10 * t, np.zeros(n)], axis=1)
    th = ManeuverThresholds(enable_lane_change=False)
    assert infer_maneuver(pos, yaw, timestamps=t, th=th) == "STRAIGHT"


def test_wrap_angle():
    assert abs(wrap_angle(np.radians(190)) - np.radians(-170)) < 1e-9


def test_attr_discount_lifted_on_collision_path():
    """경로 위 주차 차량은 상태 감쇠를 받지 않아야 한다 (ρ_attr × ρ_path 이중 감쇠 방지)."""
    from src.labeling.ttc import attr_path_gate
    a = 0.2                                        # vehicle.parked
    on_path = a + (1 - a) * attr_path_gate(1.0)    # 최접근 1.0m — 부딪힐 거리
    close   = a + (1 - a) * attr_path_gate(3.1)    # scene-0526의 SUV
    far     = a + (1 - a) * attr_path_gate(10.0)   # 길가에 늘어선 주차 차량
    assert on_path > 0.95                          # 사실상 감쇠 없음
    assert 0.4 < close < 0.7                       # 부분 해제
    assert far < 0.25                              # 원래 감쇠 유지


def test_near_floor_only_in_travel_direction():
    """근접 보정은 진행 방향 쪽 객체에만 — 지나쳐 온 정지 객체는 제외."""
    from src.labeling.ttc import near_floor
    v_fwd = [5.0, 0.0]                                    # +x로 전진 중
    ahead  = near_floor([6.0, 0.0], [0, 0], v_fwd, r=6.0)   # 앞 6m
    behind = near_floor([-3.6, 0.0], [0, 0], v_fwd, r=3.6)  # 뒤 3.6m (scene-0533 사례)
    touch  = near_floor([-1.5, 0.0], [0, 0], v_fwd, r=1.5)  # 뒤 1.5m — 접촉 거리
    far    = near_floor([20.0, 0.0], [0, 0], v_fwd, r=20.0)
    assert ahead == 0.3 and behind == 0.0 and touch == 0.3 and far == 0.0


def test_near_floor_follows_reverse_direction():
    """후진 중이면 뒤쪽 객체에 적용된다 — 보정의 원래 목적(후진·주차)."""
    from src.labeling.ttc import near_floor
    v_rev = [-3.0, 0.0]
    assert near_floor([-5.0, 0.0], [0, 0], v_rev, r=5.0) == 0.3
    assert near_floor([5.0, 0.0], [0, 0], v_rev, r=5.0) == 0.0


def test_near_floor_off_when_stopped():
    """정지 중이면 내가 다가가지 않으므로 미적용(접촉 거리는 예외)."""
    from src.labeling.ttc import near_floor
    assert near_floor([5.0, 0.0], [0, 0], [0.0, 0.0], r=5.0) == 0.0
    assert near_floor([1.2, 0.0], [0, 0], [0.0, 0.0], r=1.2) == 0.3


def test_is_hazard_needs_both_time_and_distance():
    """'실제 위험'은 시간·거리 두 조건을 모두 만족해야 한다."""
    from src.labeling.label_function import LabelConfig, is_hazard
    cfg = LabelConfig(hazard_ttc=5.0, d_hazard=3.0)
    assert is_hazard({"ttc": 2.0, "dca": 1.0}, cfg)          # 곧, 가까이
    assert not is_hazard({"ttc": 0.2, "dca": 5.0}, cfg)      # 곧이지만 5m 옆으로 스쳐 감
    assert not is_hazard({"ttc": 9.0, "dca": 1.0}, cfg)      # 가까워지지만 9초 뒤
    assert not is_hazard({"ttc": float("inf"), "dca": 0.5}, cfg)   # 접근 자체를 안 함


# ============ 보조 뷰 슬롯 배정 ============
def _cfg(**kw):
    from src.labeling.label_function import LabelConfig
    return LabelConfig(aux_threshold=0.10, aux_max=2, m_guard=0.9, **kw)


def test_aux_slot_risk_only_when_no_maneuver():
    """직진(사각지대 없음)이면 위험 항만으로 채운다."""
    from src.labeling.label_function import select_aux_views
    from src.utils.constants import maneuver_view_vector
    m = np.array(maneuver_view_vector("STRAIGHT"))          # FRONT(=primary)만 1.0
    risk = np.array([0.9, 0.4, 0.2, 0.05, 0.0, 0.03])
    assert select_aux_views(risk, m, primary=0, cfg=_cfg()) == [1, 2]


def test_aux_can_be_empty():
    """위험도 사각지대도 없으면 강조하지 않는다 — 인지부하 절감의 핵심."""
    from src.labeling.label_function import select_aux_views
    from src.utils.constants import maneuver_view_vector
    m = np.array(maneuver_view_vector("STRAIGHT"))
    risk = np.array([0.5, 0.02, 0.0, 0.01, 0.0, 0.05])
    assert select_aux_views(risk, m, primary=0, cfg=_cfg()) == []


def test_aux_blind_side_guaranteed_without_any_detection():
    """검출이 0이어도 차선변경 사각지대(후측방)는 반드시 켜진다."""
    from src.labeling.label_function import select_aux_views
    from src.utils.constants import maneuver_view_vector
    m = np.array(maneuver_view_vector("LANE_CHANGE_LEFT"))  # BACK_LEFT=1.0
    aux = select_aux_views(np.zeros(6), m, primary=0, cfg=_cfg())
    assert aux == [4]                                        # 4 = CAM_BACK_LEFT


def test_aux_risk_takes_remaining_slot_during_maneuver():
    """사각지대 1장을 보장하고 남은 슬롯은 위험이 가져간다 (기동 항이 독점하지 않음)."""
    from src.labeling.label_function import select_aux_views
    from src.utils.constants import maneuver_view_vector
    m = np.array(maneuver_view_vector("LANE_CHANGE_LEFT"))   # BACK_LEFT 1.0, FRONT_LEFT 0.5
    risk = np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.1])          # 위험은 FRONT_RIGHT에
    aux = select_aux_views(risk, m, primary=0, cfg=_cfg())
    assert aux == [4, 1], f"기대 [BACK_LEFT, FRONT_RIGHT], 실제 {aux}"


def test_merge_aux_flicker():
    """A B A 패턴의 가운데 단발 변경을 이전 집합으로 병합."""
    from src.labeling.label_function import merge_aux_flicker
    out = merge_aux_flicker([[1], [1], [3], [1], [1]])
    assert [set(a) for a in out] == [{1}, {1}, {1}, {1}, {1}]


def test_maneuver_term_survives_smoothing():
    """STRAIGHT 사이에 낀 차선변경 프레임에서도 기동 항이 온전히 살아 있어야 한다.

    기동 항까지 EMA에 넣으면 (1-α)=0.3배로 희석되어 λ·m=0.4가 0.12로 떨어지고,
    임계값 0.15를 넘지 못해 사각지대 뷰가 꺼진다.
    """
    from src.labeling.label_function import smooth_scores
    from src.utils.constants import maneuver_view_vector
    n = 7
    risks = [np.zeros(6) for _ in range(n)]                 # 검출된 위험 없음
    m = [np.array(maneuver_view_vector("STRAIGHT"))] * n
    m[3] = np.array(maneuver_view_vector("LANE_CHANGE_LEFT"))
    s = smooth_scores(risks, m, alpha=0.7, lambda_m=0.4)
    assert s[3][4] == 0.4                                   # BACK_LEFT = λ·m 온전히
    assert s[3][4] >= 0.15                                  # 임계값 통과 → 사각지대 켜짐
    assert s[2][4] == 0.0                                   # 직진 프레임엔 번지지 않음


def test_risk_term_is_still_smoothed():
    """위험 항은 여전히 평활되어야 한다 — 검출 지터 억제가 EMA의 목적."""
    from src.labeling.label_function import smooth_scores
    risks = [np.zeros(6) for _ in range(4)]
    risks[2][1] = 1.0                                       # 한 프레임만 튐
    m = [np.zeros(6)] * 4
    s = smooth_scores(risks, m, alpha=0.7, lambda_m=0.4)
    assert s[2][1] < 1.0                                    # 스파이크가 눌리고
    assert s[3][1] > 0.0                                    # 뒤 프레임으로 번진다


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}  {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {t.__name__}  {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
