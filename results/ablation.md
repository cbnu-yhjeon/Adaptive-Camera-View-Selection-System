# Ablation — test 분할, 같은 표시량(0.48장)에서 비교

포착률·정밀도·flicker는 τ 스윕 곡선을 표시량 0.48장에서 보간한 값(B.8). 입력 조건: clean = gt, noisy = noisy-medium(표 1의 공정 비교 조건). Δ는 noisy 포착률의 제안 모델 대비 차이.

| 태그 | 구성 | val AP | test AP (noisy) | 포착률 clean | 포착률 noisy | Δ | 정밀도 noisy | flicker noisy |
|---|---|---|---|---|---|---|---|---|
| 제안 | 전체 (T=4) | 0.797 | 0.778 | 95.9% | 94.2% | — | 10.3% | 0.529 |
| A1 | − CAN 토큰 (지시등 포함) | 0.781 | 0.766 | 95.4% | 93.6% | -0.6%p | 10.1% | 0.541 |
| A2 | − 위험도 토큰 | 0.413 | 0.414 | 74.2% | 74.2% | -20.0%p | 4.5% | 0.569 |
| A3 | − GRU, T=1 (= B4) | 0.575 | 0.578 | 93.3% | 90.3% | -3.8%p | 9.2% | 0.756 |
| A4 | − cross-view attention | 0.774 | 0.783 | 95.6% | 94.2% | -0.0%p | 10.3% | 0.524 |
| A5 | − temporal consistency (β=0) | 0.794 | 0.773 | 92.7% | 92.7% | -1.5%p | 9.9% | 0.519 |
| B3 | 영상만 (− CAN − 위험도) | 0.421 | 0.413 | 73.0% | 73.0% | -21.1%p | 4.1% | 0.622 |
| T10 | 보조 실험: 이력 5초 (T=10) | 0.835 | 0.819 | 96.3% | 94.3% | +0.1%p | 10.3% | 0.515 |

**참고 — 규칙** (학습 없음, AP 해당 없음)

| 방법 | 포착률 clean | 포착률 noisy | 정밀도 noisy | flicker noisy |
|---|---|---|---|---|
| B2 | 96.3% | 94.3% | 10.4% | 0.540 |
| B2c | 97.5% | 96.0% | 10.9% | 0.530 |

체크포인트:

- 제안: `results/proposed/20261007_013727/best.ckpt` (best ep11, T=4)
- A1: `results/ablation_a1_no_can/20261007_015606/best.ckpt` (best ep37, T=4)
- A2: `results/ablation_a2_no_risk/20261006_025406/best.ckpt` (best ep21, T=4)
- A3: `results/b4_single_frame/20261007_014521/best.ckpt` (best ep25, T=1)
- A4: `results/ablation_a4_no_xview/20261007_021219/best.ckpt` (best ep21, T=4)
- A5: `results/ablation_a5_no_temporal/20261007_022112/best.ckpt` (best ep9, T=4)
- B3: `results/b3_img_only/20261007_010524/best.ckpt` (best ep18, T=4)
- T10: `results/proposed_t10/20261007_022823/best.ckpt` (best ep32, T=10)
