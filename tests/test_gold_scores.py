"""골드셋 채점 단위테스트 (합성 데이터).

정답은 '진행 방향 외에 추가로 봐야 할 화면' 집합이고, 빈 집합은 '강조 불필요'다.
Top-k는 정답이 있는 프레임만, F1은 빈 정답 프레임까지 포함해 잰다.

실행: `python tests/test_gold_scores.py`  (pytest 없이도 동작)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.metrics import gold_scores  # noqa: E402


def _sel(rows):
    s = np.zeros((len(rows), 6), dtype=bool)
    for i, r in enumerate(rows):
        s[i, list(r)] = True
    return s


def test_perfect_agreement():
    gold = [{5}, {1, 2}, set()]
    sel = _sel(gold)
    rank = sel.astype(float)
    r = gold_scores(sel, rank, gold, primary=np.zeros(3, int))
    assert r["top1"] == 1.0 and r["top2"] == 1.0
    assert r["f1"] == 1.0 and r["exact"] == 1.0 and r["quiet"] == 1.0
    assert r["n_nonempty"] == 2


def test_view_on_empty_gold_frame_is_false_positive():
    """'강조 불필요' 프레임에서 화면을 켜면 F1과 quiet가 내려가고, Top-k에는 영향이 없다."""
    gold = [{5}, set()]
    sel = _sel([{5}, {3}])
    rank = sel.astype(float)
    r = gold_scores(sel, rank, gold, primary=np.zeros(2, int))
    assert r["top1"] == 1.0                     # 빈 정답 프레임은 Top-k 분모에서 빠진다
    assert r["quiet"] == 0.0
    assert r["f1"] < 1.0 and r["exact"] == 0.5


def test_primary_channel_is_ignored():
    """진행 방향은 상시 표시라 정답·예측·순위에서 모두 뺀다."""
    gold = [{0, 5}]                             # 검수자가 진행 방향(0)까지 골랐더라도
    sel = _sel([{0, 5}])
    rank = np.array([[9.0, 0, 0, 0, 0, 1.0]])   # 진행 방향 점수가 가장 높아도
    r = gold_scores(sel, rank, gold, primary=np.array([0]))
    assert r["top1"] == 1.0 and r["exact"] == 1.0 and r["mean_aux"] == 1.0


def test_top2_and_weights():
    gold = [{4}, {4}]
    sel = _sel([set(), set()])
    rank = np.array([[0, 3, 0, 0, 2, 1.0],      # 4는 2위 → top1 실패, top2 성공
                     [0, 0, 0, 0, 5, 1.0]])     # 4가 1위
    r = gold_scores(sel, rank, gold, primary=np.zeros(2, int), weight=np.array([3.0, 1.0]))
    assert abs(r["top1"] - 0.25) < 1e-9          # 가중 평균: (0·3 + 1·1) / 4
    assert r["top2"] == 1.0


def test_method_without_ranking():
    """B1(보조 뷰 없음)은 순위가 없어 Top-k를 정의하지 않는다. F1은 0, quiet는 1."""
    gold = [{5}, set()]
    r = gold_scores(_sel([set(), set()]), None, gold, primary=np.zeros(2, int))
    assert r["top1"] is None and r["top2"] is None
    assert r["f1"] == 0.0 and r["quiet"] == 1.0


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
