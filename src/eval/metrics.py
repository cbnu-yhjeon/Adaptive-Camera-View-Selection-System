"""평가 지표 (CLAUDE.md B.8). M6에서 확장, 여기선 학습/검증 공용 최소 세트.

- top1: 예측 attention_view == 정답 비율
- topk_set: 예측이 정답 집합에 포함되면 정답 (골드셋 복수정답용)
- macro_f1: 6채널 기준 macro F1
- flicker_rate: scene 시간순 예측 전환 후 3키프레임 내 재전환 비율
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from src.utils.constants import NUM_CAMERAS


def top1(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred == target).mean()) if len(pred) else 0.0


def macro_f1(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) == 0:
        return 0.0
    return float(f1_score(target, pred, labels=list(range(NUM_CAMERAS)),
                          average="macro", zero_division=0))


def topk_set(pred_topk: list[list[int]], gold_sets: list[list[int]]) -> float:
    """골드셋 복수정답: 예측 top-k 중 하나라도 정답집합에 있으면 정답."""
    if not pred_topk:
        return 0.0
    hit = sum(1 for p, g in zip(pred_topk, gold_sets) if set(p) & set(g))
    return hit / len(pred_topk)


def flicker_rate(seq_by_scene: dict[str, list[int]], window: int = 3) -> float:
    """scene별 시간순 예측열 → 전환 후 window 내 재전환 비율."""
    transitions, flickers = 0, 0
    for seq in seq_by_scene.values():
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                transitions += 1
                for k in range(i + 1, min(i + 1 + window, len(seq))):
                    if seq[k] != seq[i]:
                        flickers += 1
                        break
    return flickers / transitions if transitions else 0.0
