# 원격주행을 위한 딥러닝 기반 적응형 카메라 뷰 선택 시스템

> Deep Learning-based Adaptive Camera View Selection System for Remote Driving

## 📋 연구 개요

원격주행 관제에서는 6채널 카메라를 고정 레이아웃으로 모두 표시하기 때문에 오퍼레이터의
인지 부하가 커진다. 다중 화면 감시의 한계는 실증되어 있고(동시 감시 16화면에서 이상 사건
탐지 정확도 64%까지 저하, 효과적 감시 한계 약 4뷰), 현장 오퍼레이터는 이미 단일 화면에서
**평균 26.9~62.4초마다 카메라를 수동 전환**하며 감시한다 — 주행 상황이 초 단위로 바뀌는 것을
감안하면 부족한 주기다. 본 연구는 **주행 컨텍스트(영상 + CAN + 객체 위험도 + 기동 상태)를
융합**하여 "지금 강조해야 할 카메라 뷰"를 자동으로 선택하는 딥러닝 모델을 학습하고,
오프라인 실험으로 검증한다.

> **검증 범위**: 한정 자원은 대역폭이 아니라 **오퍼레이터의 주의**다. 표시 방식은
> 하이라이트(6채널 전부 유지)를 기본 가정으로 하며, 전송량·화질 절감이나 1인당 다수 차량
> 감시(처리량)는 본 연구의 검증 대상이 아니다. 상세는 `CLAUDE.md` A.1 참조.

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

## 🏗️ 전체 구조

### 데이터 흐름

라벨 생성과 모델 추론은 **입력이 다르다**. 라벨은 3D GT를 써도 되지만(정답을 만드는
용도), 모델이 받는 위험도 벡터는 실전을 흉내내어 손상된 값이다. 이 분리가 무너지면
강건성 주장(C3)이 성립하지 않는다.

```
                     ┌──────────────────────────────────────┐
  nuScenes           │  data/nusc_wrapper.py                │
  ├ samples/  (이미지)│  scene 순회 · 6채널 경로 · GT 박스    │
  ├ v1.0-*/  (메타)  │  ego pose · CAN · 캘리브 · blacklist  │
  └ can_bus/  (CAN)  └──────────────┬───────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │  [라벨 경로 — GT 허용]  │  [모델 입력 경로]        │
            ▼                       │                       ▼
  labeling/geometry ─ 투영          │            data/feature_cache
  labeling/ttc      ─ TTC · DCA · w │            MobileNetV3-small GAP
  labeling/maneuver ─ 기동 7클래스   │            → cache/{ver}/feat_*.npy
  labeling/label_function           │                       │
   risk(c)=EMA(Σw) · s=risk+λ·m     │            data/contrib_cache
   primary / aux 슬롯 / soft / layout│            위험 기여 + CAN → .npz
            │                       │                       │
            ▼                       │                       ▼
  data/labels_{ver}.parquet ────────┼──────────────► data/dataset.py
   (학습 타깃)                       │                 T=4 시퀀스 조립
                                    │                 risk_source 분기
                                    │                       │
                                    │                 data/noise.py
                                    │                 miss · FP · 위치오차
                                    │                 · 채널 블랙아웃
                                    │                       │
                                    │                       ▼
                                    │            models/proposed.py
                                    │            채널토큰 6 + CAN 1
                                    │            → Transformer(2층)
                                    │            → 채널별 GRU
                                    │            → Head1 뷰 / Head2 레이아웃
                                    │                       │
                                    └───────────────────────┤
                                                            ▼
                                                    eval/metrics.py
                                              Top-1/2 · Macro-F1 · flicker
                                              위험 포착률 · 위험 슬롯 정밀도
```

**순환 평가 차단**: 왼쪽 라벨 경로를 그대로 추론에 쓰는 것이 베이스라인 B2다.
따라서 최종 성능의 주 지표는 사람이 검수한 골드셋이며, 의사 라벨 시험셋 수치는
보조 열로만 병기한다.

### 디렉터리

```
├── CLAUDE.md                   # 구현 명세 (단일 진실 공급원)
├── README.md
├── configs/
│   ├── base.yaml               # 경로 · seed · 배치 · 디바이스
│   ├── label_v1.yaml           # 라벨 함수 하이퍼파라미터
│   └── train_proposed.yaml     # 모델 · 손실 · 학습 (+ ablation 플래그)
├── splits/
│   ├── scene_split_v1.json     # trainval 70/15/15 — 고정, 변경 금지
│   └── scene_split_mini.json   # mini 스모크 테스트용
├── src/
│   ├── utils/     constants · geometry · slide_kit
│   ├── data/      nusc_wrapper · splits · feature_cache · contrib_cache
│   │              · noise · dataset
│   ├── labeling/  ttc · maneuver · label_function
│   ├── models/    proposed
│   └── eval/      metrics
├── scripts/       00~05 파이프라인 + 문서·슬라이드 생성
├── tests/         geometry · labeling · noise_seeding
├── docs/          research-proposal.html (연구계획서 원본)
├── cache/         백본 특징 · 위험 기여 (gitignore)
└── results/       실험 산출물 · 그림 · 슬라이드 (gitignore)
```

### 모듈별 설명

#### `src/utils` — 전역 규약

| 모듈 | 하는 일 | 왜 필요한가 |
|---|---|---|
| `constants.py` | 채널 순서(6), 기동 7클래스, 기동→채널 적합도 `m(c)`, 레이아웃 K=5, 객체 상태 계수 | 라벨의 채널 3번과 모델 출력의 채널 3번이 달라지면 실험 전체가 **조용히** 오염된다. 순서를 한 곳에 못 박는다 |
| `geometry.py` | `global → ego → camera → image` 변환과 투영, 화면 내 판정 | 객체를 어느 채널에 귀속시킬지가 라벨의 출발점이다 |
| `slide_kit.py` | .pptx 공용 부품(도형·표·색 토큰) | 발표 자료용. 연구 파이프라인과 무관 |

#### `src/data` — 데이터 접근과 입력 조립

| 모듈 | 하는 일 | 설계 의도 |
|---|---|---|
| `nusc_wrapper.py` | devkit을 감싸 scene 순회·이미지 경로·GT 박스(속도·attribute 포함)·ego pose·CAN·캘리브를 제공. CAN blacklist와 이미지 결손 scene 필터 | 라벨·데이터셋·평가가 **모두 이 래퍼만** 거친다. devkit 버전이나 좌표 규약이 바뀌어도 수정 지점이 한 곳이다 |
| `splits.py` | scene 이름 기준 70/15/15 결정적 분할, 저장·로드 | 분할 누수 금지(원칙 3). 토큰이 아니라 이름 기준이라 사람이 읽고 검증할 수 있다 |
| `feature_cache.py` | MobileNetV3-small GAP 특징을 카메라별 `.npy` memmap으로 사전 추출 | 학습 중 백본을 재실행하지 않는다. 원본 이미지 학습이 아니므로 1 epoch가 초 단위로 끝난다 |
| `contrib_cache.py` | 위험 기여(투영·TTC·DCA·w)와 CAN 특징을 ragged `.npz`로 캐시 | 34,149 keyframe × 실험 10여 회를 매번 재계산하면 낭비다. 캐시 키는 **기여 값을 바꾸는 LabelConfig 필드만** 반영해 무관한 설정 변경으로 무효화되지 않는다 |
| `noise.py` | 검출 오류 시뮬레이터 — 누락(원거리 가중) · 오검출(Poisson) · 위치/속도 잡음 · 채널 블랙아웃. `clean/light/medium/heavy` 프리셋 | 원칙 2의 구현체. 학습 증강과 강건성 평가 양쪽에서 같은 코드를 쓴다 |
| `dataset.py` | 과거 T=4 시퀀스 조립(scene 경계 패딩 + 마스크), `risk_source` 분기, 난수 시드 정책, `WeightedRandomSampler`, 평가용 `frame_risk` | 모델은 **과거만** 본다. 라벨이 미래를 참조하는 비대칭이 "선제적 전환 학습"의 핵심이다 |
| `map_index.py` | nuScenes 맵 확장(차로·연결부·도로 구간·주차장) STRtree 조회, 차로 연결 탐색 | devkit 조회는 점당 78 ms(전체 44분) → 0.04 ms(전체 약 1초) |

> **난수 정책** — 학습 증강은 `(seed, epoch, index)`(`item_rng`, `set_epoch`), 평가는 **프레임 단위**
> `(seed, 프레임)`(`frame_rng`, `frame_risk`). 학습에서 인덱스만으로 시드하면 고정된 손상 입력을
> 암기해 강건성이 증명되지 않는다. 평가에서는 한 프레임의 검출 결과가 하나여야 전체 이력을 쓰는
> 규칙(B2)과 4프레임을 보는 모델이 같은 세계를 본다. 검출 품질과 블랙아웃 채널은 2초 시퀀스 안에서 유지된다.

#### `src/labeling` — 의사 라벨 생성 (M2, 본 연구의 ★)

| 모듈 | 하는 일 | 핵심 판단 |
|---|---|---|
| `ttc.py` | 최접근(CPA) 기반 TTC와 DCA, 위험 가중 `w`, 경로 계수 `ρ_path`, 상태 감쇠 예외 `ρ_attr_eff`, 진행방향 게이트 근접 보정 | radial 정의(`r/v_r`)는 옆으로 스쳐 갈 객체를 충돌 임박으로 오판한다. 실측상 위험량의 62.9%가 정지·주차 객체에서 나왔다. `ttc_mode: radial`로 되돌릴 수 있고 ablation 항목으로 쓴다 |
| `maneuver.py` | 7개 기동 상태 판정. 상태(STOP·REVERSE)는 ±1초, 기동(TURN·LANE_CHANGE)은 [−1, +5]초 창. **맵 기준**: TURN은 일반 차로 밖(교차로·연결부)에서 튼 각도 > 25°, 차선변경은 이어지지 않은 이웃 차로로 차로 ID가 바뀜 (맵이 없으면 반경·횡이탈 궤적 규칙) | 궤적만으로는 교차로 통과를 차선변경으로, 큰 교차로 회전을 곡선로로 오판했다(궤적 기준 차선변경 중 맵 확인 15.8%). 방향지시등 일치: 차선변경 25~39% → 58~59% |
| `label_function.py` | 채널 귀속 → 채널 위험도 → `s(c) = risk(c) + λ·m(c)` → primary · aux 슬롯 · soft label · layout, scene 단위 EMA 평활과 flicker 병합 | **위험 항만 평활한다.** 기동 상태는 이미 수 초 윈도우로 추론된 안정 신호라 다시 평활하면 이중 스무딩이 되어, 기동 경계에서 안전 기본값이 꺼진다 |

`is_hazard()` 는 **'실제 위험'의 단일 정의**(`TTC < 5s AND DCA < 3m`)이며 라벨·평가·골드셋이
모두 이 함수를 공유한다. 정의가 갈라지면 재현율과 정밀도가 서로 다른 모집단을 가리킨다.

#### `src/models` · `src/eval`

| 모듈 | 하는 일 |
|---|---|
| `proposed.py` | 채널 토큰 6개 + CAN 토큰 1개 → Transformer(2층·4헤드·d=128) → 채널별 GRU → Head1(뷰) / Head2(레이아웃). Head1은 **채널별 독립 sigmoid**(다중 선택 0~2장). 손실은 focal BCE + KL(soft) + layout CE + temporal consistency. 모델 선택은 임계값 무관한 macro-AP. **ablation은 별도 모델이 아니라 같은 코드의 플래그**(`use_can` · `use_risk` · `use_transformer` · `use_gru`)로 구성 |
| `metrics.py` | Top-1, 복수 정답 Top-k, Macro-F1, scene 시간순 flicker rate |

### 실행 순서

```bash
# 0) 데이터 배치 검증
python scripts/00_check_data.py --version v1.0-mini

# 1) 탐색 — scene별 keyframe 수, CAN 스냅숏
python scripts/01_explore_data.py --version v1.0-mini

# 2) 의사 라벨 생성 → data/labels_{version}.parquet
python scripts/02_generate_labels.py --version v1.0-mini

# 3) 검수용 시각화 20장 → results/label_viz/   ← 사용자 승인 게이트 (M2)
python scripts/03_visualize_labels.py --n 20

# 4) 백본 특징 · 위험 기여 캐시 + 로더 검증
python scripts/04_build_cache.py --version v1.0-mini

# 5) 학습
python scripts/05_train.py --config configs/train_proposed.yaml
python scripts/05_train.py --config configs/train_proposed.yaml --limit_samples 100   # 드라이런

# 테스트
python -m pytest tests/ -q
```

> mini 스모크 테스트는 `--version v1.0-mini --splits splits/scene_split_mini.json`.
> 분할 파일을 새로 만드는 `scripts/make_split.py`는 **최초 1회용**이며 다시 실행하지 않는다.

### 구현 현황

| 계층 | 상태 |
|---|---|
| utils · data · labeling | 구현 완료 — 유닛테스트 57개. **M2 승인** (1차 20장 2026-10-04, 2차 표적 16장 2026-10-06, 맵 기준) |
| 특징·데이터 캐시 | trainval 특징 캐시(204,894장, 473 MB) · 기여 캐시(CAN 7종) — **M3 완료** |
| models · 학습 루프 | 채널별 sigmoid + focal BCE, macro-AP 모델 선택, 지시등 입력 (M4) |
| `baselines/rule_based.py` | B1·B2(오라클 기동)·B2c(과거만) — M5 |
| `scripts/06_evaluate.py` | 공통 평가 + `--curve` 운영 곡선(같은 표시량 비교) — M5·M6 |
| `scripts/06b_goldset_sample.py` · `tools/goldset_ui/` | 골드셋 300장 층화 추출 + 검수 UI — M6 (사용자 라벨링 대기) |
| `configs/` B3·B4·A1·A2·A4·A5 | `inherit:` 상속 설정 — M7 |
| `eval/robustness.py` · `scripts/07~08` | 미구현 — M7·M8 |
| `data/interpolation.py` (2Hz→12Hz) | 선택 항목, 미구현 |

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
CAN (speed, steering, accel_x, accel_y, yaw_rate,
     left_signal, right_signal)                    → CAN 토큰 1개
  → TransformerEncoder(layers=2, heads=4, d=128, ff=256)
  → 채널별 GRU (시간축 처리)
  → Head1: 채널별 독립 sigmoid (6) — 보조 뷰 0~2장 다중 선택
  → Head2: 레이아웃 (K=5-way)
```

- **방향지시등**은 운전자의 의도라 기동보다 먼저 켜진다 — 라벨이 최대 5초 앞서는 선제적
  전환의 단서. 시점 이하의 직전 값만 쓰고, **라벨 판정에는 쓰지 않는다**(모델이 지시등만
  베끼는 지름길을 막기 위해). 라벨 검증 지표로만 쓴다.
- **손실**: `BCE_focal(aux, mask) + α·KL(soft_label) + γ_l·CE(layout) + β·temporal_consistency`
  — 진행 방향 채널은 마스크로 제외(타깃 0으로 두면 "앞은 끄는 것"을 배운다)
- **학습**: AdamW(lr=1e-4), cosine decay, WeightedRandomSampler, early stopping(**val macro-AP**).
  F1·표시량은 라벨과 같은 표시량이 되도록 임계값을 맞춘 뒤 읽는다.
  **`train.risk_noise: true`(검출 오류 증강)가 기본** — 강건성의 핵심.

## 🏷️ 의사 라벨 생성 (weak supervision)

각 keyframe에서 GT 박스를 카메라에 투영하여 채널에 귀속시키고, 위험도와 기동 적합도를
결합해 채널 점수 `s(c)`를 계산한다.

```
risk(c) = EMA_α( Σ_{i∈c} w_i )        # 위험 항만 평활 (검출 지터 억제)
s(c)    = risk(c) + λ·m(c)            # λ=0.4 — 기동 항은 평활하지 않음
w       = ρ_attr · ρ_path · max( exp(-TTC/τ), 근접보정 )

primary  = 진행방향 뷰 (전진 FRONT / 후진 BACK) — 상시 표시, 선택 대상 아님
aux      = 슬롯 배정 (최대 2장)
           슬롯 1  안전 기본값: m(c) ≥ m_guard 인 채널 — 검출 0이어도 켠다
           슬롯 2~ 위험:       risk(c) ≥ θ 인 채널을 위험 순으로
```

**왜 슬롯을 역할로 나누는가**: 하나의 임계값을 합산 `s(c)`에 걸면 기동 항이 슬롯을 독점해
실제 위험이 밀려난다(차선변경 중 위험 커버 60.6%). 역할 분리 후 70.5%.
**왜 0장을 허용하는가**: 실제 위험은 프레임의 13.2%뿐이다. 매 프레임 한 장을 강제하면
대부분의 시간에 무의미한 강조가 켜져 오퍼레이터가 보조 뷰를 무시하게 된다.

> **뷰 정책 (v1)**: 이동체는 진행 방향을 항상 주시해야 하므로 primary(진행방향)를 UI 메인에
> 고정하고, 위험/기동으로 고른 attention 뷰를 보조로 강조한다. 정지(STOP) 시에는 극단 위험이
> 아니면 뷰를 유지해 불필요한 전환을 억제한다.

- **TTC**: 최접근(CPA) 정의 — `t* = −(p_rel·v_rel)/‖v_rel‖²`, 그때의 거리가 DCA.
  radial 정의(`r/v_r`)는 옆으로 스쳐 갈 객체를 충돌 임박으로 오판해 위험량의 62.9%가
  정지·주차 객체에서 나왔다. `ttc_mode: radial`로 되돌릴 수 있으며 ablation 항목이다.
- **'실제 위험'의 단일 정의**: `TTC < 5s AND DCA < 3m` (`label_function.is_hazard`) —
  라벨·평가·골드셋이 같은 함수를 공유한다. trainval 기준 4,852채널 / 프레임의 13.2%
  (`scripts/02c_label_report.py` → `results/label_report.json`).
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
- **평균 보조 뷰 수** — 인지 부하의 대리 지표
- **위험 사건 포착률(재현율)** — '실제 위험' 객체가 선택 뷰(primary ∪ aux) 화각에 든 비율.
  GT 기하로 직접 계산하므로 순환이 없다
- **위험 슬롯 정밀도** — 위험으로 켠 채널 중 실제 위험이 있던 비율. **반드시 병기한다**
  (재현율만 보면 "많이 켜면 오르는" 지표라 튜닝이 왜곡된다)
- **운영 곡선** — 임계값을 쓸어가며 (평균 보조 뷰 수 vs 포착률). 강건성도 스칼라가 아니라
  이 곡선의 열화로 보고한다
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

- [x] **M1** — 데이터 접근·탐색 (`nusc_wrapper.py`, `01_explore_data.py`)
- [x] **M2** ★ — 의사 라벨 파이프라인 (`geometry/ttc/maneuver/label_function`) + 시각화 검수
- [x] **M3** — Dataset/로더 + 특징 캐시
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
- [ ] `labeling/ttc.py` — 최접근(CPA) TTC·DCA, box_velocity NaN 폴백
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
- [ ] 손실: focal BCE + KL + layout CE + temporal consistency
- [ ] `scripts/05_train.py` — config 로드, best ckpt, metrics.json, tensorboard
- [ ] mini 과적합 sanity check (train Top-1 >95%)
- [ ] 🚦 **trainval 확보범위 확인 후 본 학습 (✅ M4)**

### P2 — 평가 체계 (M5~M6)

- [ ] `baselines/rule_based.py` — B2 (label_function 재사용 + risk_source 입력)
- [ ] B1 상수 예측기 + B3/B4 config (모델 플래그 재사용)
- [ ] `eval/metrics.py` — Top-1/2, Macro-F1, flicker, 위험 포착률·슬롯 정밀도 + 단위테스트 3케이스
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

1. Caesar, H., et al. (2020). nuScenes: A Multimodal Dataset for Autonomous Driving. *CVPR 2020*.
2. Neumeier, S., Wintersberger, P., Frison, A.-K., Becher, A., Facchi, C., Riener, A. (2019).
   Teleoperation: The Holy Grail to Solve Problems of Automated Driving? Sure, but Latency Matters.
   *AutomotiveUI '19*. DOI 10.1145/3342197.3344534
3. Howard, A., et al. (2019). Searching for MobileNetV3. *ICCV 2019*.
4. Xia, Y., Zhang, D., Kim, J., Nakayama, K., Zipser, K., Whitney, D. (2019).
   Predicting Driver Attention in Critical Situations. *ACCV 2018*, LNCS 11365, Springer.
   DOI 10.1007/978-3-030-20873-8_42
5. Palazzi, A., Abati, D., Solera, F., Cucchiara, R. Predicting the Driver's Focus of Attention:
   The DR(eye)VE Project. *IEEE TPAMI* 41(7):1720–1733.
6. Hofbauer, M., Kuhn, C. B., Khlifi, M., Petrovic, G., Steinbach, E. (2022).
   Traffic-Aware Multi-View Video Stream Adaptation for Teleoperated Driving. *IEEE VTC2022-Spring*.
   — 가장 가까운 선행연구. 뷰 중요도를 **대역폭 배분**에 활용. 본 연구는 **주의 배분**으로
   정식화하고 검출 불확실성 하의 강건성을 주 검증 대상으로 삼는 점이 다르다.
7. Tatler, B. W. (2021). Searching in CCTV: effects of organisation in the multiplex.
   *Cognitive Research: Principles and Implications* 6. DOI 10.1186/s41235-021-00277-2
8. Tickner, A. H., & Poulton, E. C. (1973). Monitoring up to 16 Synthetic Television Pictures
   Showing a Great Deal of Movement. *Ergonomics* 16(4):381–401. DOI 10.1080/00140137308924529
9. Majstorović, D., Hoffmann, S., Pfab, F., Schimpe, A., Wolf, M.-M., Diermeyer, F. (2022).
   Survey on Teleoperation Concepts for Automated Vehicles. *IEEE SMC 2022*, pp. 1290–1296.
   DOI 10.1109/SMC53654.2022.9945267 — 원격 운전/원격 지원 개념 분류 + 산업 활동·특허 동향
10. Bogdoll, D., Orf, S., Töttel, L., Zöllner, J. M. (2022). Taxonomy and Survey on Remote Human
    Input Systems for Driving Automation Systems. *Advances in Information and Communication*, Springer.
11. Kettwich, C., Schrank, A., Oehl, M. (2021). Teleoperation of Highly Automated Vehicles in Public
    Transport: User-Centered Design of an HMI and Its Expert Usability Evaluation.
    *MTI* 5(5):26. DOI 10.3390/mti5050026 — 모니터 6대 중 상단 3대가 영상
12. Wolf, M.-M., Schmidt, H., Christl, M., Fank, J., Diermeyer, F. (2025). A User-Centered
    Teleoperation GUI for Automated Vehicles. *MTI* 9(8):78. DOI 10.3390/mti9080078
    — 단계별 동적 GUI (표시 요소 수 적응)
13. Musicant, O., Botzer, A., Richmond-Hacham, B. (2024). Safety, Efficiency, and Mental Workload
    in Simulated Teledriving of a Vehicle as Functions of Camera Viewpoint.
    *Sensors* 24(18):6134. DOI 10.3390/s24186134 — n=148, 완료시간 14%↓, 충돌률 불변
14. Stainer, M. J., Scott-Brown, K. C., Tatler, B. W. (2013). Looking for Trouble: A Description of
    Oculomotor Search Strategies During Live CCTV Operation.
    *Frontiers in Human Neuroscience* 7:615. DOI 10.3389/fnhum.2013.00615 — 전환 주기 26.9~62.4초
15. Linkov, V., Vanžura, M. (2021). Situation Awareness Measurement in Remotely Controlled Cars.
    *Frontiers in Psychology* 12:592930. DOI 10.3389/fpsyg.2021.592930

## 📧 Contact

- **소속**: 충북대학교 (CBNU)
- **이메일**: yhjeon@avgenius.kr

---

**License**: MIT (코드) / CC BY-NC-SA 4.0 (nuScenes 데이터)
