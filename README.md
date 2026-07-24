# 원격주행을 위한 딥러닝 기반 적응형 카메라 뷰 선택 시스템

> Deep Learning-based Adaptive Camera View Selection System for Remote Driving

## 📋 연구 개요

원격주행 관제에서는 6채널 카메라를 고정 레이아웃으로 모두 표시하기 때문에 오퍼레이터의
인지 부하가 커진다. 본 연구는 **주행 컨텍스트(영상 + CAN + 객체 위험도 + 기동 상태)를
융합**하여 "지금 가장 중요한 카메라 뷰"를 자동으로 선택하는 딥러닝 모델을 학습하고,
오프라인 실험으로 검증한다.

공개 벤치마크가 없으므로 **nuScenes의 3D 객체 GT + CAN 신호로 '이상적 뷰' 의사 라벨을
자동 생성(weak supervision)**하여 학습에 사용한다.

> **구현의 단일 진실 공급원(Single Source of Truth)은 [`CLAUDE.md`](CLAUDE.md)이다.**
> 아키텍처·하이퍼파라미터·평가 프로토콜·마일스톤의 상세 명세는 모두 그 문서를 따른다.
> 본 README는 요약본이며, 세부 규칙이 충돌할 경우 항상 `CLAUDE.md`가 우선한다.

## 🎯 핵심 주장 (구현이 반드시 뒷받침해야 함)

1. **의사 라벨 자동 생성 방법이 동작한다** — TTC·기동 상태 기반 라벨 함수 (M2)
2. **다중 신호 융합 + 시간 모델이 단일 신호/단일 프레임보다 낫다** — ablation (M4~M6)
3. **불완전한 입력에서 학습 모델이 규칙 기반(B2)보다 강건하다** — 강건성 실험 (M7) ← 최중요

## 🔑 방법론 원칙

- **순환 평가 금지**: 의사 라벨을 만든 규칙(B2)을 그대로 추론에 쓰면 의사 라벨 시험셋에서
  정의상 만점이 나온다. 따라서 표 1의 **주 지표는 사람이 검수한 골드셋 기준**이며,
  의사 라벨 시험셋 수치는 보조 열로만 병기한다.
- **학습 라벨과 추론 입력의 분리**: 라벨 생성에는 GT 박스를 쓰지만, 모델 추론 입력의
  위험도 벡터는 `gt` / `noisy` / `none` 3가지 조건을 지원한다. 강건성 실험의 기본은 `noisy`.
- **분할 누수 금지**: scene 단위 분할(70/15/15)을 `splits/scene_split_v1.json`으로 고정·커밋하고
  이후 변경하지 않는다.
- **재현성**: 모든 실험은 `configs/*.yaml` + `--seed`(기본 42)로 실행하고, 결과는
  `results/{exp_name}/{timestamp}/`에 metrics.json + config 사본 + 체크포인트로 저장한다.

## 🗂️ 데이터셋

- **nuScenes** (CC BY-NC-SA 4.0, 학술 목적 무료 — 계정 등록 후 다운로드)
  - `v1.0-mini` (~4GB, 10 scenes) — 파이프라인 스모크 테스트용 필수 선행
  - `can_bus.zip` — CAN expansion (일부 scene 결손, `can_blacklist`로 관리)
  - `v1.0-trainval` — 본 학습용. 스토리지 부족 시 카메라 keyframe blob 일부만 받아
    "이미지가 존재하는 scene만" 필터링하여 부분집합으로 진행 가능
- **채널 순서 (전 구간 고정)**: `CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT, CAM_BACK,
  CAM_BACK_LEFT, CAM_FRONT_LEFT` (인덱스 0~5)

## 🧠 모델 아키텍처

시퀀스 T=4 keyframes(≈2초, 과거만)를 입력으로 받는 경량 융합 모델(~1M 파라미터).

```
img_feat (MobileNetV3-small GAP, D_f=576, 사전 캐시) ┐
camera positional embedding (6채널)                  ├→ 채널 토큰 6개 (d=128)
risk 벡터 (채널별 Σw(TTC))                            ┘
CAN (speed, steering, accel_x, accel_y, yaw_rate)  → CAN 토큰 1개
  → TransformerEncoder(layers=2, heads=4, d=128, ff=256)
  → 채널별 GRU (시간축 처리)
  → Head1: attention 뷰 분포 (6-way softmax) — 적응형 보조 뷰
  → Head2: 레이아웃 (K=5-way)
```

- **손실**: `CE_focal(main) + α·KL(soft_label) + γ_l·CE(layout) + β·temporal_consistency`
- **학습**: AdamW(lr=1e-4), cosine decay, focal loss + WeightedRandomSampler,
  early stopping(val macro-F1). **`train.risk_noise: true`(검출 오류 증강)가 기본** — 강건성의 핵심.

## 🏷️ 의사 라벨 생성 (weak supervision)

각 keyframe에서 GT 박스를 카메라에 투영하여 채널에 귀속시키고, 위험도와 기동 적합도를
결합해 채널 점수 `s(c)`를 계산한다.

```
s(c) = Σ_{i∈c} w(TTC_i) + λ·m(c)      # w(TTC)=exp(-TTC/τ), 기동→채널 적합도 m(c)
primary_view   = 진행방향 뷰 (전진 FRONT / 후진 BACK) — 상시 주시
attention_view = argmax_{c≠primary} s(c)  — 위험/기동 기반 적응형 보조 뷰
```

> **뷰 정책 (v1)**: 이동체는 진행 방향을 항상 주시해야 하므로 primary(진행방향)를 UI 메인에
> 고정하고, 위험/기동으로 고른 attention 뷰를 보조로 강조한다. 정지(STOP) 시에는 극단 위험이
> 아니면 뷰를 유지해 불필요한 전환을 억제한다.

- **TTC**: radial 정의 — 접근 속도 기반. 근접 객체(r<10m) 최소 가중 보정.
- **기동 상태**: `{STRAIGHT, TURN_L/R, LANE_CHANGE_L/R, REVERSE, STOP}`
  (ego pose 시퀀스 + CAN으로 규칙 추론, 라벨 생성 시에만 미래 pose 참조 허용)
- **레이아웃**: 기동 클래스 → 사전 정의 템플릿(K=5) 결정적 매핑
- 출력: `data/labels_{version}.parquet`

## 🧪 베이스라인 & Ablation

| ID | 정의 |
|---|---|
| B1 | 고정 전방 (항상 CAM_FRONT) |
| B2 | 규칙 기반 (라벨 함수를 추론 입력에 직접 적용) |
| B3 | 영상 단일 (CAN·위험도 토큰 제거) |
| B4 | 단일 프레임 (GRU 제거, T=1) |

Ablation: A1(−CAN) / A2(−위험도) / A3(−GRU) / A4(−cross-view attention) / A5(−temporal 항)

## 📈 평가 지표

- **Top-1 / Top-2** (골드셋, 복수 정답 허용)
- **Macro-F1** (6채널, 골드셋 + 의사 라벨 시험셋)
- **Flicker rate** (전환 후 1.5초 내 재전환 비율)
- **위험 객체 가시화율** (TTC<3s GT 객체가 선택 뷰 화각에 포함된 비율 — GT 기하로 직접 계산)
- **강건성**: 노이즈 유형(miss / FP / pos-noise / 채널 드롭) × 강도(clean~heavy)별
  제안 모델 vs B2 열화 곡선

## 🛠️ 개발 환경

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision   # CUDA 버전에 맞게
pip install nuscenes-devkit timm opencv-python-headless pandas pyarrow \
            numpy scipy scikit-learn matplotlib pyyaml tensorboard tqdm pyquaternion
```

- 백본 특징을 사전 캐시하여 학습하므로(원본 이미지 학습 아님) GPU 메모리 부담이 작다.
  `configs`의 `batch_size` / `img_size`는 GPU 메모리에 맞춰 조정한다.

## 🗺️ 마일스톤

각 마일스톤의 완료 조건(✅)을 통과하기 전에는 다음 단계로 넘어가지 않는다.

- [ ] **M1** — 데이터 접근·탐색 (`nusc_wrapper.py`, `01_explore_data.py`)
- [ ] **M2** ★ — 의사 라벨 파이프라인 (`geometry/ttc/maneuver/label_function`) + 시각화 검수
- [ ] **M3** — Dataset/로더 + 특징 캐시
- [ ] **M4** — 모델 + 학습
- [ ] **M5** — 베이스라인 (B1~B4)
- [ ] **M6** — 평가 + 골드셋(사용자 직접 라벨링) → `results/table1.md`
- [ ] **M7** — Ablation + 강건성 곡선
- [ ] **M8** — 정성 결과(그림 2) + 논문 반영 수치

### 사용자 확인이 필요한 지점

1. **M2 완료 시** — 라벨 시각화 20장 검수 승인
2. **M4 본 학습 전** — trainval 데이터 확보 범위(전체/부분집합) 확인
3. **M6** — 골드셋 300 keyframe 직접 라벨링
4. **M7** — 강건성 결과가 기대와 다를 경우 대응 방향 상의

## ✅ 개발 로드맵 (우선순위별 TODO)

우선순위는 **의존성 순서**를 따른다 (앞 단계가 끝나야 뒤 단계 가능).
🚦 표시는 **사용자 확인/작업이 필요한 게이트**다.

| 등급 | 의미 | 마일스톤 |
|---|---|---|
| **P0** | 블로커·기반 — 없으면 아무것도 못 함 | M1~M2 |
| **P1** | 핵심 학습 파이프라인 | M3~M4 |
| **P2** | 평가 체계 | M5~M6 |
| **P3** | 논문 검증·산출물 | M7~M8 |

계층 구조:

```
[0] 환경·데이터 확보    ← P0 (게이트)
[1] 데이터 접근 계층     ← P0   nusc_wrapper, constants
[2] 라벨링 계층 ★        ← P0   geometry, ttc, maneuver, label_function
[3] 특징·데이터 계층     ← P1   feature_cache, dataset, noise
[4] 모델 계층            ← P1   proposed (Transformer+GRU)
[5] 베이스라인 계층      ← P2   rule_based (B1~B4)
[6] 평가 계층            ← P2   metrics, evaluate, goldset_ui
[7] 검증·산출물 계층     ← P3   robustness, make_figures
```

### P0 — 블로커 & 기반 (M1~M2)

- [ ] 🚦 **nuScenes v1.0-mini + can_bus 다운로드·배치** (사용자 조치 — 모든 작업의 선행 게이트)
- [ ] 개발환경 세팅: `.venv` + requirements 설치·freeze 커밋 + GPU 확인
- [ ] `utils/constants.py` — 채널 순서, 기동 클래스, 레이아웃 템플릿
- [ ] `data/nusc_wrapper.py` — scene 순회, 이미지 경로, GT 박스(ego 변환), CAN, blacklist 처리
- [ ] `scripts/01_explore_data.py` — mini 10 scene 무에러 실행 (✅ M1)
- [ ] `utils/geometry.py` 투영 유틸 + 단위테스트 5케이스
- [ ] `labeling/ttc.py` — radial TTC, box_velocity NaN 폴백
- [ ] `labeling/maneuver.py` — 7개 기동 상태 규칙 추론
- [ ] `labeling/label_function.py` — `s(c)` 계산 → primary/attention/soft/layout → parquet
- [ ] `scripts/02_generate_labels.py` + `03_visualize_labels.py`
- [ ] 🚦 **시각화 20장 사용자 검수 승인 + 라벨 분포 편중(전방>85%) 없을 것 (✅ M2)**

### P1 — 학습 파이프라인 (M3~M4)

- [ ] `splits/scene_split_v1.json` 생성·커밋 (70/15/15, **이후 변경 금지**)
- [ ] `data/feature_cache.py` — MobileNetV3-small GAP → `.npy` memmap, 재실행 스킵
- [ ] `data/noise.py` — 검출 오류 시뮬레이터 (miss/FP/pos-noise/blackout, light~heavy)
- [ ] `data/dataset.py` — 시퀀스 조립, scene 경계 패딩, risk_source 분기, WeightedRandomSampler
- [ ] `scripts/04_build_cache.py` + shape 검증 + 1 epoch <1분 (✅ M3)
- [ ] `models/proposed.py` — 토큰화 → Transformer(2층) → 채널별 GRU → Head1/Head2 (ablation 플래그 공용)
- [ ] 손실: focal + KL + layout CE + temporal consistency
- [ ] `scripts/05_train.py` — config 로드, best ckpt, metrics.json, tensorboard
- [ ] mini 과적합 sanity check (train Top-1 >95%)
- [ ] 🚦 **trainval 확보범위 확인 후 본 학습 (✅ M4)**

### P2 — 평가 체계 (M5~M6)

- [ ] `baselines/rule_based.py` — B2 (label_function 재사용 + risk_source 입력)
- [ ] B1 상수 예측기 + B3/B4 config (모델 플래그 재사용)
- [ ] `eval/metrics.py` — Top-1/2, Macro-F1, flicker, 위험객체 가시화율 + 단위테스트 3케이스
- [ ] `eval/evaluate.py` — 모든 모델 공통 평가 엔트리
- [ ] 골드셋 300개 층화 추출 + `tools/goldset_ui/` 제작
- [ ] 🚦 **사용자 직접 골드셋 라벨링** (Claude는 UI만 제작)
- [ ] `scripts/06_evaluate.py --all` → `results/table1.md` (✅ M6)

### P3 — 논문 검증·산출물 (M7~M8)

- [ ] A1~A5 ablation config 실행 → `results/ablation.md`
- [ ] `scripts/07_robustness.py` → 열화곡선 4종 + md 표
- [ ] 🚦 **강건성 곡선에서 제안 모델 우위 확인 (✅ M7)** — 미확인 시 대응절차 후 상의
- [ ] `scripts/08_make_figures.py` → `results/fig2/` (성공2+실패1 사례)
- [ ] `results/paper_numbers.md` (표1+ablation+강건성 핵심수치)
- [ ] 전체 재현 커맨드 목록 문서화 (✅ M8)

## 📚 참고문헌

1. Caesar, H., et al. (2020). nuScenes: A multimodal dataset for autonomous driving. *CVPR 2020*.
2. Neumeier, S., et al. (2019). Teleoperation: The holy grail to solve the autonomous driving dilemma. *IEEE/RSJ IROS*.
3. Howard, A., et al. (2019). Searching for MobileNetV3. *ICCV 2019*.

## 📧 Contact

- **소속**: 충북대학교 (CBNU)
- **이메일**: yhjeon@avgenius.kr

---

**License**: MIT (코드) / CC BY-NC-SA 4.0 (nuScenes 데이터)
