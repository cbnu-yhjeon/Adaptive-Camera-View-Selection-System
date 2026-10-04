"""노이즈 난수 정책 + 기여 캐시 직렬화 단위테스트 (합성 데이터, 실 nuScenes 불필요).

핵심 회귀 방지 대상 — 학습 증강 노이즈가 아이템 인덱스만으로 시드되면 같은 샘플이
100 epoch 내내 **똑같이 손상된 입력**을 보게 되어, 증강이 아니라 고정 노이즈 암기가
된다. 그러면 M7(불완전 입력 강건성)이 성립하지 않는다.

실행: `python tests/test_noise_seeding.py`  (pytest 없이도 동작)
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import contrib_cache  # noqa: E402
from src.data.dataset import item_rng  # noqa: E402
from src.data.noise import apply_noise, make_noise_config  # noqa: E402
from src.labeling.label_function import LabelConfig, risk_vector_from_contribs  # noqa: E402
from src.utils.constants import CAN_FEATURES  # noqa: E402


def _contribs(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return [{"vis": [int(rng.integers(6))], "r": float(rng.uniform(3, 50)),
             "ttc": float(rng.uniform(1, 20)), "w": 0.3} for _ in range(n)]


def _risk(index, epoch, contribs):
    rng = item_rng(42, index, epoch)
    res = apply_noise(contribs, rng, make_noise_config("medium"))
    return risk_vector_from_contribs(res.contribs)


# ======================= 난수 정책 =======================
def test_eval_noise_is_fixed_per_item():
    """평가(epoch=None): 같은 아이템은 항상 같은 손상 입력 — 모델과 B2 비교의 전제."""
    c = _contribs()
    assert np.allclose(_risk(7, None, c), _risk(7, None, c))


def test_train_noise_varies_across_epochs():
    """학습 증강: 같은 아이템이라도 epoch가 다르면 다른 노이즈 실현이어야 한다."""
    c = _contribs()
    same = sum(np.allclose(_risk(7, 0, c), _risk(7, ep, c)) for ep in range(1, 20))
    assert same == 0, f"{same}/19 epoch에서 노이즈가 동일 — 증강이 작동하지 않음"


def test_noise_differs_across_items():
    """같은 epoch 안에서도 아이템마다 독립적인 노이즈여야 한다."""
    c = _contribs()
    assert not np.allclose(_risk(7, 3, c), _risk(8, 3, c))


def test_train_noise_is_reproducible():
    """epoch를 섞어도 (seed, epoch, index)로 결정되므로 재현 가능해야 한다(원칙 4)."""
    c = _contribs()
    assert np.allclose(_risk(7, 5, c), _risk(7, 5, c))


def test_seed_change_changes_noise():
    """seed가 다르면 다른 노이즈 — base_seed가 실제로 반영되는지 확인."""
    c = _contribs()
    r1 = risk_vector_from_contribs(
        apply_noise(c, item_rng(42, 7, 0), make_noise_config("medium")).contribs)
    r2 = risk_vector_from_contribs(
        apply_noise(c, item_rng(43, 7, 0), make_noise_config("medium")).contribs)
    assert not np.allclose(r1, r2)


# ======================= 기여 캐시 직렬화 =======================
class _FakeWrapper:
    """get_can_features만 필요한 최소 스텁."""

    def get_can_features(self, token):
        return np.arange(len(CAN_FEATURES), dtype=np.float32), 1.0


def test_contrib_cache_roundtrip():
    """다채널 가시성·무한대 TTC·객체 0개 프레임이 캐시 왕복에서 보존되어야 한다."""
    src = {
        "tok_a": [{"vis": [0, 3], "r": 12.5, "ttc": 4.25, "w": 0.5},
                  {"vis": [5], "r": 40.0, "ttc": float("inf"), "w": 0.0}],
        "tok_b": [],                                  # 객체 없는 keyframe
        "tok_c": [{"vis": [0, 1, 2, 3, 4, 5], "r": 2.0, "ttc": 0.5, "w": 1.0}],
    }
    original = contrib_cache.channel_risk_contributors
    contrib_cache.channel_risk_contributors = lambda w, tok, cfg: src[tok]
    try:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.npz")
            contrib_cache.build(_FakeWrapper(), list(src), LabelConfig(), path, verbose=False)
            cache = contrib_cache.ContribCache(path)

            for tok, expect in src.items():
                got = cache.get(tok)
                assert len(got) == len(expect), tok
                for g, e in zip(got, expect):
                    assert g["vis"] == e["vis"], f"{tok} 채널 비트마스크 불일치"
                    assert abs(g["r"] - e["r"]) < 1e-4
                    assert abs(g["w"] - e["w"]) < 1e-6
                    assert (g["ttc"] == e["ttc"]) if np.isinf(e["ttc"]) \
                        else abs(g["ttc"] - e["ttc"]) < 1e-4
            feat, mask = cache.can("tok_a")
            assert mask == 1.0 and len(feat) == len(CAN_FEATURES)
    finally:
        contrib_cache.channel_risk_contributors = original


def test_contrib_cache_key_ignores_irrelevant_fields():
    """캐시 키는 contributors 값을 바꾸는 필드에만 반응해야 한다."""
    base = LabelConfig()
    same = LabelConfig(ema_alpha=0.1, T_soft=2.0, lambda_m=9.0)   # 결합 단계 전용
    diff = LabelConfig(tau=8.0)                                   # w를 바꿈
    assert contrib_cache.cache_key(base) == contrib_cache.cache_key(same)
    assert contrib_cache.cache_key(base) != contrib_cache.cache_key(diff)


def test_contrib_cache_key_tracks_can_schema():
    """CAN 특징 구성이 바뀌면 캐시 키도 바뀌어야 한다.

    키가 라벨 설정만 보면, 특징을 늘려도 옛 캐시가 재사용돼 새 특징이 에러 없이
    0으로 들어간다 — 방향지시등을 추가할 때 실제로 그럴 뻔했다.
    """
    k_now = contrib_cache.cache_key(LabelConfig())
    saved = list(contrib_cache.CAN_FEATURES)
    try:
        contrib_cache.CAN_FEATURES.append("_probe")
        assert contrib_cache.cache_key(LabelConfig()) != k_now
    finally:
        contrib_cache.CAN_FEATURES[:] = saved


def test_contrib_cache_rejects_stale_can_width():
    """폭이 다른 옛 캐시를 읽으면 조용히 쓰지 말고 즉시 실패해야 한다."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.npz")
        np.savez(path, tokens=np.array(["t"]), offsets=np.array([0, 0]),
                 vis_mask=np.zeros(0, np.int32), obj_r=np.zeros(0), obj_ttc=np.zeros(0),
                 obj_w=np.zeros(0), can_feat=np.zeros((1, 5), np.float32),
                 can_mask=np.ones(1, np.float32))
        try:
            contrib_cache.ContribCache(path)
        except ValueError:
            return
        raise AssertionError("폭 5짜리 옛 캐시를 그대로 받아들였다")


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
