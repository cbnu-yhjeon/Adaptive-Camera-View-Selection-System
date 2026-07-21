"""ttc.py / maneuver.py 단위테스트 (합성 데이터, 실 nuScenes 불필요).

실행: `python tests/test_labeling.py`  (pytest 없이도 동작)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.labeling.ttc import TTC_INF, compute_ttc, risk_weight  # noqa: E402
from src.labeling.maneuver import ManeuverThresholds, infer_maneuver, wrap_angle  # noqa: E402


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
