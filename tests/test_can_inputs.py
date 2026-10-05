"""CAN 입력 단위테스트 — 방향지시등 유지 판정 (합성 메시지, 실 nuScenes 불필요).

vehicle_monitor의 left/right_signal은 램프 점멸 상태를 2 Hz로 찍은 값이라, 지시등을 켠
동안에도 1,1,0,0으로 깜빡인다. 직전 메시지 하나만 쓰면 켠 지시등이 소등 순간에 '꺼짐'으로
읽혀 모델 입력과 골드셋 화면이 함께 틀린다(2026-10-06 골드셋 검수 중 발견).

실행: `python tests/test_can_inputs.py`  (pytest 없이도 동작)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import contrib_cache  # noqa: E402
from src.data.nusc_wrapper import NuscWrapper  # noqa: E402
from src.labeling.label_function import LabelConfig  # noqa: E402

US = 1_000_000


def _wrapper(right):
    """0.5초 간격 vehicle_monitor 메시지만 가진 가짜 래퍼."""
    w = NuscWrapper.__new__(NuscWrapper)
    msgs = [{"utime": int(k * 0.5 * US), "left_signal": 0, "right_signal": v}
            for k, v in enumerate(right)]
    w.get_can_messages = lambda scene_token, message: msgs
    return w


def test_signal_state_holds_through_blink_gaps():
    """점멸 중 소등 샘플(2샘플까지)에서도 켜짐이어야 한다."""
    w = _wrapper([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0])
    for k in (0, 1, 2, 3, 4, 5, 6, 7):
        assert w.signal_state("s", int(k * 0.5 * US)) == (0.0, 1.0), k
    # 메시지 사이 시각도 같다 (keyframe은 메시지 시각과 어긋난다)
    assert w.signal_state("s", int(1.7 * US)) == (0.0, 1.0)


def test_signal_state_turns_off_after_hold():
    """마지막 점등(2.5초)에서 유지 시간(1.5초)이 지나면 꺼짐."""
    w = _wrapper([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0])
    assert w.signal_state("s", int(3.9 * US)) == (0.0, 1.0)
    assert w.signal_state("s", int(4.0 * US)) == (0.0, 0.0)


def test_signal_state_is_causal():
    """시점 이후의 메시지는 보지 않는다 — 곧 켜질 지시등을 미리 알면 선제성이 거저 생긴다."""
    w = _wrapper([0, 0, 0, 1, 1])
    assert w.signal_state("s", int(1.4 * US)) == (0.0, 0.0)
    assert w.signal_state("s", int(1.5 * US)) == (0.0, 1.0)
    assert w.signal_state("s", -1) is None                  # 구간에 메시지가 없으면 None


def test_contrib_cache_key_tracks_signal_hold():
    """지시등의 정의(유지 시간)가 바뀌면 캐시 키도 바뀌어야 한다 — 옛 정의의 특징 재사용 방지."""
    k_now = contrib_cache.cache_key(LabelConfig())
    saved = contrib_cache.SIGNAL_HOLD_S
    try:
        contrib_cache.SIGNAL_HOLD_S = saved + 0.5
        assert contrib_cache.cache_key(LabelConfig()) != k_now
    finally:
        contrib_cache.SIGNAL_HOLD_S = saved


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
