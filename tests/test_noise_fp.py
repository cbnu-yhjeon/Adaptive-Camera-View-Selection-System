"""오검출 시뮬레이터 단위테스트 — 정지 잡음 모델 (합성, 실 nuScenes 불필요).

가짜 객체는 정지된 잡음 + 추적 속도 오차다. 위험은 실제 객체와 같은 계산으로 내 차의
움직임에서 나온다. 이전 방식(차량 상태와 무관하게 TTC 1~10초)은 정지 중 위험 신호의
대부분을 가짜로 만들어 모델이 정지 중 위험도를 불신하게 만들었다(2026-10-07 분석).

실행: `python tests/test_noise_fp.py`  (pytest 없이도 동작)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.noise import apply_noise, make_noise_config  # noqa: E402
from src.utils.constants import CAMERA_INDEX  # noqa: E402


def _phantoms(ego_speed, n=400, seed=0, **over):
    nc = make_noise_config("clean", lambda_fp=1.0, **over)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out += [c for c in apply_noise([], rng, nc, ego_speed).contribs if c.get("fp")]
    return out


def test_stopped_ego_static_phantoms_carry_no_risk():
    """내 차가 서 있고 속도 오차가 없으면 가짜 객체는 다가오지 않는다 → 위험 0."""
    ph = _phantoms(0.0, sigma_fp_vel=0.0)
    assert ph and all(c["w"] == 0.0 and not np.isfinite(c["ttc"]) for c in ph)


def test_moving_ego_front_phantom_behaves_like_static_object():
    """주행 중 전방 가짜 객체: 정지물과 같다 — TTC = r·cosθ / v, 위험 > 0인 경우가 있다."""
    ph = [c for c in _phantoms(10.0, sigma_fp_vel=0.0) if c["vis"] == [CAMERA_INDEX["CAM_FRONT"]]]
    assert ph
    for c in ph:
        assert np.isfinite(c["ttc"]) and 0.0 < c["ttc"] <= c["r"] / 10.0 + 1e-9
    assert any(c["w"] > 0.1 for c in ph)


def test_moving_ego_back_phantom_recedes():
    """주행 중 후방 가짜 객체는 멀어진다 → TTC inf, 근접 보정도 없음(진행 방향 반대)."""
    ph = [c for c in _phantoms(10.0, sigma_fp_vel=0.0) if c["vis"] == [CAMERA_INDEX["CAM_BACK"]]]
    assert ph and all(c["w"] == 0.0 for c in ph)


def test_legacy_approaching_mode_is_reproducible():
    """이전 방식은 차량 상태와 무관하게 위험을 만든다 — 지난 결과 재현용으로 남긴다."""
    ph = _phantoms(0.0, fp_mode="approaching")
    assert ph and all(1.0 <= c["ttc"] <= 10.0 and c["w"] > 0.0 for c in ph)


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
