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


def test_maneuver_turn_left():
    """좌회전: 헤딩이 +40° 순증가."""
    n, dt = 13, 0.5
    t = np.arange(n) * dt
    yaw = np.linspace(0, np.radians(40), n)   # +40° net
    pos = np.stack([10 * t, np.zeros(n)], axis=1)  # 전진(속도 확보)
    assert infer_maneuver(pos, yaw, timestamps=t) == "TURN_LEFT"


def test_maneuver_turn_right():
    n, dt = 13, 0.5
    t = np.arange(n) * dt
    yaw = np.linspace(0, np.radians(-40), n)  # −40° net
    pos = np.stack([10 * t, np.zeros(n)], axis=1)
    assert infer_maneuver(pos, yaw, timestamps=t) == "TURN_RIGHT"


def test_maneuver_lane_change_left():
    """좌 차선변경: 헤딩이 +15°까지 갔다가 0°로 복귀 (net≈0, peak 중간)."""
    n, dt = 13, 0.5
    t = np.arange(n) * dt
    half = n // 2
    yaw = np.concatenate([
        np.linspace(0, np.radians(15), half),
        np.linspace(np.radians(15), 0, n - half),
    ])
    pos = np.stack([10 * t, np.zeros(n)], axis=1)
    assert infer_maneuver(pos, yaw, timestamps=t) == "LANE_CHANGE_LEFT"


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


# ============ 보조 뷰 다중 선택 ============
def test_aux_views_threshold_and_cap():
    """임계값 이상만, 최대 aux_max개, primary 제외."""
    from src.labeling.label_function import LabelConfig, select_aux_views
    cfg = LabelConfig(aux_threshold=0.5, aux_max=2)
    s = np.array([3.0, 1.2, 0.8, 0.1, 0.0, 0.6])      # 0번이 primary
    assert select_aux_views(s, primary=0, cfg=cfg) == [1, 2]
    assert 0 not in select_aux_views(s, primary=0, cfg=cfg)


def test_aux_views_can_be_empty():
    """아무것도 임계값을 넘지 못하면 보조 뷰 없음 — 인지부하 절감의 핵심."""
    from src.labeling.label_function import LabelConfig, select_aux_views
    cfg = LabelConfig(aux_threshold=0.5, aux_max=2)
    s = np.array([1.0, 0.2, 0.1, 0.0, 0.05, 0.3])
    assert select_aux_views(s, primary=0, cfg=cfg) == []


def test_aux_views_maneuver_term_keeps_blind_side_on():
    """검출된 객체가 0이어도 기동 항만으로 차선변경 방향은 켜져야 한다.

    검출 실패 시 화면이 꺼지는 것이 가장 위험한 실패 모드다.
    """
    from src.labeling.label_function import LabelConfig, select_aux_views
    from src.utils.constants import maneuver_view_vector
    cfg = LabelConfig(aux_threshold=0.5, aux_max=2)
    s = np.array(maneuver_view_vector("LANE_CHANGE_LEFT"))   # 위험 0, 기동 항만
    aux = select_aux_views(s, primary=0, cfg=cfg)
    assert len(aux) >= 1 and 4 in aux                        # 4 = CAM_BACK_LEFT


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
