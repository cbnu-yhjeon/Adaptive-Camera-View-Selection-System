# 원격주행 적응형 카메라 뷰 선택 시스템 — 상세 구현 명세서

> **이 문서는 Claude Code의 단일 진실 공급원(Single Source of Truth)이다.**
> 논문 「원격주행을 위한 딥러닝 기반 적응형 카메라 뷰 선택 시스템」의 오프라인 실험
> (표 1, ablation, 강건성, 그림 2)을 재현 가능하게 구현하는 것이 목표다.
> 마일스톤(M1~M8) 순서대로 진행하며, 각 마일스톤의 **완료 조건(✅)을 통과하기 전에는
> 다음 단계로 넘어가지 않는다.** 판단이 모호한 지점은 이 문서의 규칙을 우선한다.

---

# PART A. 프로젝트 컨텍스트

## A.1 연구 요약 (구현자가 알아야 할 최소 배경)

- 원격주행 관제는 6채널 카메라를 고정 레이아웃으로 표시 → 오퍼레이터 인지 부하 증가.
  근거(2026-09 문헌 조사):
  ① 다중 화면 감시 성능 저하 — 동시 감시 화면 4대 83% / 9대 84% / 16대 64%.
     Tickner, A.H., Poulton, E.C. "Monitoring up to 16 Synthetic Television Pictures Showing
     a Great Deal of Movement." Ergonomics 16(4):381-401, 1973. DOI 10.1080/00140137308924529
     (관찰자 204명, 1시간, 교도소 장면). 배치 구조만으로 탐색 시간이 달라진다는 보고도 있다 —
     Tatler, B.W. "Searching in CCTV: effects of organisation in the multiplex."
     Cogn. Res. Princ. Implic. 6, 2021. DOI 10.1186/s41235-021-00277-2 (27개 장면, 단독 저자).
  ② 현장 CCTV 오퍼레이터는 멀티플렉스를 훑지 않고 **단일 스팟 모니터에서 평균 26.9초·
     62.4초마다 카메라를 능동 전환**한다 → 본 연구가 자동화하려는 과업을 사람이 수동으로 하고
     있고, 주행 상황은 초 단위로 바뀌므로 그 주기로는 부족하다.
     Stainer, M.J., Scott-Brown, K.C., Tatler, B.W. "Looking for Trouble: A Description of
     Oculomotor Search Strategies During Live CCTV Operation." Front. Hum. Neurosci. 7:615,
     2013. DOI 10.3389/fnhum.2013.00615. **주의: 관찰 오퍼레이터 2명** — 모집단 추정치가
     아니라 현장 전략의 근거로만 쓴다.
  ③ 원격 오퍼레이터는 차량과 물리적으로 분리돼 속도·노면 경사·적재 하중을 체감할 수 없고
     엔진음도 듣지 못하며, 전정기관·기계수용기가 받지 못하는 감각만큼 상황인식이 저하된다.
     Linkov, V., Vanžura, M. "Situation Awareness Measurement in Remotely Controlled Cars."
     Front. Psychol. 12:592930, 2021. DOI 10.3389/fpsyg.2021.592930.
     여기에 본 연구의 전제인 다채널 2D 영상(깊이 단서 약함·화각 밖 불가시)이 더해진다.
  ④ 표현을 바꾸면 워크로드가 실제로 움직인다 — 카메라 시점을 실내에서 차량 루프로 올린
     것만으로 경로 완료 시간 14%↓, NASA-TLX 정신적 요구·좌절감 감소, 충돌률·급제동 확률은
     불변. Musicant, O., Botzer, A., Richmond-Hacham, B. "Safety, Efficiency, and Mental
     Workload in Simulated Teledriving of a Vehicle as Functions of Camera Viewpoint."
     Sensors 24(18):6134, 2024. DOI 10.3390/s24186134 (n=148, 여성 80, 평균 25세).
     단 이 연구가 바꾼 것은 카메라의 물리적 설치 위치(고정)이고, 본 연구는 같은 지렛대를
     상황에 따라 동적으로 당긴다.
  ⑤ 그런데 언제 강조할지가 비자명하다 — 본 프로젝트 측정상 '실제 위험'은 전체 프레임의
     **5.3%**에서만 발생한다(B.4-(2) 정의). 화면은 100% 켜져 있다.
- **산업 운용 형태** (Majstorović, D., Hoffmann, S., Pfab, F., Schimpe, A., Wolf, M.-M.,
  Diermeyer, F. "Survey on Teleoperation Concepts for Automated Vehicles." IEEE SMC 2022,
  pp. 1290-1296, DOI 10.1109/SMC53654.2022.9945267 / Bogdoll 등 2022): 원격 운전(직접 제어·공유 제어·
  궤적 유도, 스티어링 휠·페달)과 원격 지원(경유점 유도·대화형 경로 계획·인지 수정, 마우스·
  키보드)으로 나뉘며, 레벨 1–2가 전자, 3–5가 후자다. **본 연구는 원격 운전(직접 제어) 전제.**
- **관제 UI 현황**: 다중 모니터에 영상 채널 고정 배치가 일반적이다 — 공공교통 셔틀 HMI는
  모니터 6대 중 상단 3대가 영상, 중간 행 상세·장애·지도, 하단 터치스크린.
  Kettwich, C., Schrank, A., Oehl, M. MTI 5(5):26, 2021. DOI 10.3390/mti5050026.
  차량 측은 전방 120° + 어안 180° 4대 = 5채널 구성 사례.
  Wolf, M.-M., Schmidt, H., Christl, M., Fank, J., Diermeyer, F. MTI 9(8):78, 2025.
  DOI 10.3390/mti9080078 은 **표시 정보 요소 수**를 원격주행 단계에 따라 바꾸는 동적 GUI를
  제시했으나(전문가 인터뷰 N=9, 클릭더미 N=36), 바뀌는 대상은 정보 요소이지
  **어느 카메라를 강조할지가 아니다**.
  ※ 위 두 논문의 그림은 MDPI CDN이 자동 접근을 403으로 막고 arXiv판은 재배포 라이선스가
  아니다 — 계획서 그림 1(A)는 문헌 기술을 도식으로 **재현**한 것이며 원본 사진이 아니다.
- 본 연구: 주행 컨텍스트(영상 + CAN + 객체 위험도 + 기동 상태)를 융합해
  **"지금 가장 중요한 카메라 뷰"를 자동 선택**하는 모델을 학습.
- 공개 벤치마크가 없으므로 **nuScenes의 3D 객체 GT + CAN으로 '이상적 뷰' 의사 라벨을
  자동 생성**(weak supervision)하여 학습하고, 오프라인으로 검증.
- 논문의 핵심 주장 3가지 — 구현이 반드시 뒷받침해야 함:
  1. 의사 라벨 자동 생성 방법이 동작한다 (M2)
  2. 다중 신호 융합 + 시간 모델이 단일 신호/단일 프레임보다 낫다 (M4~M6 ablation)
  3. **불완전한 입력에서 학습 모델이 규칙 기반(B2)보다 강건하다** (M7) ← 최중요

### 검증 범위 밖 — 배경·기대효과에서도 주장하지 않는다
아래는 그럴듯하지만 **본 연구가 측정하지 않는** 축이다. 배경에 넣으면 실험이 뒷받침하지
못하는 약속이 되므로 쓰지 않는다. 기대효과에 넣더라도 "검증 범위에 포함하지 않는다"를
명기한다.
- **통신 대역폭·전송 품질 절감**: 표시 방식을 하이라이트(6채널 전부 유지)로 확정했으므로
  전송량을 줄이지 않는다. 채널 중요도를 대역 배분에 쓸 수 있다는 것은 향후 확장일 뿐이다.
- **오퍼레이터 1인당 다수 차량 감시(처리량)**: 그것은 remote *assistance* 모델이고, 본
  연구는 remote *driving*(한 대를 실시간 조종·감시) 전제다. 차량 간 배분은 평가하지 않는다.
- **지연(latency)**: 필요성 근거가 아니라 **설계 파라미터 근거**로만 쓴다 — 200ms 수준에서
  워크로드가 증가한다는 보고(Neumeier et al., 2019)가 위험 지평을 3초가 아닌 **5초**로
  잡은 이유다(B.4-(2)).

## A.2 절대 원칙 (위반 시 논문 논리가 무너짐)

### 원칙 1 — 순환 평가 금지
의사 라벨을 만든 규칙(라벨 함수)을 그대로 추론에 쓰는 것이 B2다. 의사 라벨 시험셋에서
B2는 정의상 ~100%가 나온다. 따라서:
- 최종 성능 표(표 1)의 **주 지표는 사람이 검수한 골드셋 기준**으로 보고.
- 의사 라벨 시험셋 수치는 참고용 보조 열로만 병기.

### 원칙 2 — 학습 라벨과 추론 입력의 분리
- **라벨 생성**: nuScenes 3D GT 박스 사용 (정답을 만드는 용도이므로 GT 허용).
- **모델 추론 입력의 위험도 벡터**: 평가 프로토콜에서 3가지 조건을 지원해야 함.
  - `risk_source: gt` — GT 기반 (상한 확인용)
  - `risk_source: noisy` — GT에 검출 오류 시뮬레이션 주입 (강건성 실험의 기본)
  - `risk_source: none` — 위험도 토큰 제로화 (모달리티 결손 실험)
- **학습 시에도 `noisy` 조건으로 증강 학습**하는 옵션을 반드시 구현 (`train.risk_noise: true`).
  이것이 없으면 M7에서 제안 모델이 B2를 이기지 못할 가능성이 높다.

### 원칙 3 — 분할 누수 금지
- scene 단위 분할. nuScenes `log`의 `location` 필드(4개 지역:
  `boston-seaport`, `singapore-onenorth`, `singapore-queenstown`, `singapore-hollandvillage`)를
  기준으로 test에 최소 1개 지역이 train에 없는 구성을 시도하되, 데이터 규모상 어렵다면
  scene 단위 무작위 분할(70/15/15)로 하고 논문에 명시. 분할은 `splits/scene_split_v1.json`으로
  고정하고 git에 커밋. **이후 절대 변경 금지.**

### 원칙 4 — 재현성
- 모든 실험은 `configs/*.yaml` + `--seed` 로 실행. 기본 seed=42.
- 결과는 `results/{exp_name}/{timestamp}/` 아래 metrics.json + config 사본 + 체크포인트로 저장.
- 수치를 수기로 옮기지 않는다. 표는 스크립트가 md로 생성.

## A.3 작업 규칙

- 마일스톤 단위 커밋, 메시지에 `[M2]`처럼 번호 명시.
- 새 코드는 반드시 **v1.0-mini로 스모크 테스트 통과 후** 커밋.
- 하드코딩 금지: 경로, 임계값, 채널 순서 참조는 전부 config 또는 상수 모듈 경유.
- 긴 학습(>30분 예상) 전에는 소규모 드라이런(`--limit_samples 100`)으로 파이프라인 확인.
- 외부에서 결과를 검수해야 하는 산출물(라벨 시각화, 골드셋 UI, 열화 곡선)은
  파일로 저장하고 경로를 보고.

---

# PART B. 환경·데이터·구조

## B.1 개발 환경

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision  # CUDA 버전에 맞게
pip install nuscenes-devkit timm opencv-python-headless pandas pyarrow \
            numpy scipy scikit-learn matplotlib pyyaml tensorboard tqdm pyquaternion
```
- `requirements.txt`를 처음 설치 시점에 freeze하여 커밋.
- GPU 확인: `python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e9)"`
  → 결과에 따라 config의 `batch_size`, `img_size` 조정 (기본: 24GB 기준 batch 16, img 224).

## B.2 데이터 준비

### 단계 1 — v1.0-mini (필수 선행)
- https://www.nuscenes.org/nuscenes 에서 계정 등록 후 다운로드:
  - `v1.0-mini.tgz` (~4GB, 메타데이터 + 10 scenes 전체 센서)
  - `can_bus.zip` (CAN expansion — mini/trainval 공용)
- 배치:
```
{DATAROOT}/
├── maps/  samples/  sweeps/  v1.0-mini/
└── can_bus/          # can_bus.zip 압축 해제 결과
```
- 로드 확인:
```python
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT)
can = NuScenesCanBus(dataroot=DATAROOT)
```

### 단계 2 — trainval (M4 본 학습 전)
- 메타데이터(`v1.0-trainval_meta.tgz`) + **카메라 keyframe blob 우선** 다운로드.
  스토리지가 부족하면 blob 일부(예: part 1~5)만 받아 해당 scene 부분집합으로 진행 —
  코드는 "이미지 파일이 존재하는 scene만 사용"하는 필터를 지원할 것.
- **주의**: CAN expansion은 일부 scene에 없음 (공식 문서 기준 419번 등 결손 scene 존재).
  `NuScenesCanBus.can_blacklist` 를 확인하고 결손 scene은 제외 목록으로 관리.

### 채널 순서 (전 구간 고정 — `src/utils/constants.py`)
```python
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
# 인덱스 0~5. 라벨, 모델 출력, 평가 모두 이 순서.
```

## B.3 디렉터리 구조

```
project/
├── CLAUDE.md
├── requirements.txt
├── configs/
│   ├── base.yaml              # 공통 기본값
│   ├── label_v1.yaml          # 라벨 함수 하이퍼파라미터
│   ├── train_proposed.yaml
│   ├── train_b3_img_only.yaml
│   ├── train_b4_single_frame.yaml
│   └── ablation_*.yaml
├── splits/scene_split_v1.json
├── src/
│   ├── utils/constants.py     # 채널 순서, 기동 클래스, 레이아웃 템플릿
│   ├── utils/geometry.py      # 투영, 좌표 변환
│   ├── data/nusc_wrapper.py   # devkit 래퍼 (scene/sample/CAN 접근)
│   ├── data/interpolation.py  # 2Hz→12Hz 박스 보간 (플래그로 활성화)
│   ├── data/dataset.py        # PyTorch Dataset (시퀀스 조립, risk_source 분기)
│   ├── data/feature_cache.py  # 백본 특징 사전 추출·캐시
│   ├── data/noise.py          # 검출 오류 시뮬레이터 (원칙 2)
│   ├── labeling/ttc.py
│   ├── labeling/maneuver.py
│   ├── labeling/label_function.py  # s(c) 계산 → 라벨 생성
│   ├── models/proposed.py     # 융합 모델 (아래 B.5 명세)
│   ├── baselines/rule_based.py     # B2 (label_function 재사용)
│   ├── eval/metrics.py
│   ├── eval/evaluate.py       # 모든 모델 공통 평가 엔트리
│   └── eval/robustness.py
├── scripts/
│   ├── 01_explore_data.py
│   ├── 02_generate_labels.py
│   ├── 03_visualize_labels.py
│   ├── 04_build_cache.py
│   ├── 05_train.py
│   ├── 06_evaluate.py
│   ├── 07_robustness.py
│   └── 08_make_figures.py
├── tools/goldset_ui/          # M6 골드셋 검수 뷰어
└── results/
```

## B.4 라벨 함수 상세 명세 (M2의 수학적 정의)

### (1) 객체 → 채널 매핑
- 각 keyframe sample에서 GT 박스(카테고리: `vehicle.*`, `human.pedestrian.*`,
  `movable_object.barrier` 등 — config의 `label.categories`로 필터, 기본은
  vehicle + pedestrian + cyclist)를 ego 좌표계로 변환.
- 박스 중심을 각 카메라에 투영(`geometry.py`: global → ego → camera, intrinsic 적용).
  이미지 평면 내(0≤u<W, 0≤v<H) & z>0 이면 해당 채널에 귀속.
- 두 채널에 동시 투영되면 양쪽에 가중 0.5로 귀속.
- 거리 상한: `label.max_range = 50m` (이보다 먼 객체는 무시).

### (2) TTC (최접근 정의, v2 — 2026-09 개정)
```
p_rel = p_obj - p_ego,   v_rel = v_obj - v_ego
t*    = -(p_rel·v_rel)/||v_rel||²        # 최접근 시각
DCA   = ||p_rel + v_rel·t*||             # 그때의 거리
TTC   = t*   (접근 중)  |  ∞ (멀어지는 중)
```
**왜 radial을 버렸나**: radial 정의(`r/v_r`)는 옆으로 스쳐 지나갈 객체를 충돌 임박으로
오판한다. ego가 10m/s로 주행하면 20m 앞 길가 주차 차량도 TTC=2s가 나온다. 실측 결과
전체 위험량의 **62.9%가 정지·주차 객체**에서 나왔고, 이를 빼면 선택 뷰가 29.4%의
프레임에서 뒤집혔다. `ttc_mode: radial`로 되돌릴 수 있으며 ablation 항목으로 쓴다.

**'실제 위험' 정의 (라벨·평가·골드셋 공용, `label_function.is_hazard`)**
    `TTC < hazard_ttc(5s)  AND  DCA < d_hazard(3m)`
- 두 조건이 모두 필요하다. 시간만 보면 옆 5m로 스쳐 갈 객체도 최접근 시각이 0.2s이고,
  거리만 보면 10초 뒤에나 가까워질 객체가 지금 위험으로 잡힌다.
- **거리는 조이고 시간은 늘린다.** DCA 3m는 중심점 간 거리라 차폭 감안 실제 여유가
  약 1.1m — "가까이 지나감"이지 "스쳐 지나감"이 아니다. 4m(여유 2.1m)로 늘리면 옆 차선
  통과가 대량 유입된다(프레임의 16.4%). 반대로 5초는 원격주행의 전송 지연 + 인지·판단·
  조작을 감안하면 최소치다. 3초로 잡으면 test 분할 위험 프레임이 146개로 떨어져
  통계도 안 나온다.
- **이 정의로 trainval 위험 채널은 1,879개(프레임의 5.3%), test 분할 283 프레임.**
- `v_obj`: GT 박스의 velocity (devkit `box_velocity()`), NaN이면 인접 keyframe 차분.
- `v_ego`: CAN pose 또는 ego_pose 차분.

### (3) 위험 가중 함수 (config `label.w_type`)
- 기본: `w = ρ_attr · ρ_path · max( exp(-TTC/tau), 근접보정 )`, `tau = 4.0s`, TTC=∞ → 0.
- **ρ_attr (객체 상태 계수)**: nuScenes 주석의 attribute를 사용. 주차 0.2 / 정지 0.5 /
  서있는 보행자 0.5 / 세워둔 자전거 0.2 / 그 외(움직이는 객체) 1.0. 0으로 만들지는
  않는다 — 주차 차량도 문 열림·보행자 가림 위험이 있다.
- **ρ_path (경로 관련성)**: `exp(-max(0, DCA - d_near)/d_scale)`, 하한 rho_min.
  d_near=4m, d_scale=3m (DCA는 중심점 간 거리이고 인접 차선이 3.5m임을 감안).
- **경로 위 예외**: 두 계수를 그냥 곱하면 *경로 위의* 주차 차량이 이중으로 깎인다.
  상태 감쇠는 길가에 늘어서 스쳐 지나갈 차를 위한 것이고 그건 ρ_path가 이미 처리한다.
  최접근이 차체 폭 수준이면 주차 여부와 무관하게 부딪힐 대상이므로 감쇠를 해제한다:
  `ρ_attr_eff = ρ_attr + (1-ρ_attr)·exp(-max(0, DCA-1.5)/2.0)`.
  실측 사례(scene-0526, 우측 차선변경): 7.6m 앞 DCA 3.1m 주차 SUV가 ρ_attr에 눌려
  채널 위험 0.097로 임계값 0.10을 0.003 차이로 못 넘었다. 예외 적용 후 0.215.
  전체로는 위험 커버 81.1%→88.3%, 표시량 1.19→1.25장, 정지객체 위험 비중 38%
  (예외 없으면 29%, 상태 감쇠를 아예 안 쓰면 58%).
- 대안(실험용): `w = clip(1/TTC, 0, 1)`.
- 근접 보정: r < 10m 인 객체에 최소 `w_min_near = 0.3` 부여. **단 진행 방향 쪽
  객체에만** — 접촉 거리(2m) 이내면 방향 무관, 정지 중(|v_ego|<0.5m/s)이면 미적용.
  방향 조건 없이 적용하면 이미 지나쳐 뒤에 남은 정지 객체가 후방 채널을 켠다
  (실측 scene-0533: ego 1.7m/s 전진 중 뒤쪽 3.6m·6.3m 차량이 TTC=inf인데도
  근접 보정만으로 w=0.145/0.140을 받아 BACK·BACK_RIGHT가 보조 뷰로 뽑혔다).
  이 조건을 넣자 평균 표시가 1.25→1.03장으로 줄면서 위험 커버는 87.8%로 유지됐다.

### (4) 기동 상태 추론 (`maneuver.py`)
클래스: `{STRAIGHT, TURN_LEFT, TURN_RIGHT, LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT, REVERSE, STOP}`
- 입력: ego pose 시퀀스(±3초 윈도우) + CAN(steering, speed, gear가 있으면 사용).
- 규칙 (기본 임계값, config `label.maneuver.*`):
  - `STOP`: 평균 속도 < 0.5 m/s
  - `REVERSE`: 진행 방향과 헤딩의 내적 < 0 (또는 CAN gear=reverse)
  - `TURN_*`: 윈도우 내 누적 헤딩 변화 |Δψ| > 25° (부호로 좌/우)
  - `LANE_CHANGE_*`: |Δψ| ∈ [5°, 25°] 이면서 3초 후 헤딩이 복원(|Δψ_total| < 10°)
    — 판정이 불안정하면 v1에서는 TURN에 흡수시키고 논문에 명시 (완벽할 필요 없음)
  - 그 외: `STRAIGHT`
- **미래 참조 허용**: 라벨 생성은 오프라인이므로 미래 pose를 봐도 된다.
  단, **모델 입력에는 절대 미래 정보를 넣지 않는다** (모델은 과거 2초만 봄 —
  이 비대칭이 "선제적 전환 학습"의 핵심).

### (5) 기동→채널 적합도 m(c) (constants.py, config로 오버라이드 가능)
| 기동 | 가중치 부여 채널 (값) |
|---|---|
| STRAIGHT | FRONT(1.0) |
| TURN_LEFT | FRONT_LEFT(1.0), FRONT(0.5), BACK_LEFT(0.3) |
| TURN_RIGHT | FRONT_RIGHT(1.0), FRONT(0.5), BACK_RIGHT(0.3) |
| LANE_CHANGE_LEFT | BACK_LEFT(1.0), FRONT_LEFT(0.5), FRONT(0.5) |
| LANE_CHANGE_RIGHT | BACK_RIGHT(1.0), FRONT_RIGHT(0.5), FRONT(0.5) |
| REVERSE | BACK(1.0), BACK_LEFT(0.5), BACK_RIGHT(0.5) |
| STOP | FRONT(0.5) — 위험도 항이 지배하도록 낮게 |

### (6) 최종 점수와 라벨 — **v1 뷰 정책 (2026-07 설계 결정)**
```
risk(c) = EMA_α( Σ_{i∈c} w_i )             # 위험 항만 평활 (검출 지터 억제)
s(c)    = risk(c) + λ·m(c)                 # λ 기본 0.4 — 기동 항은 평활하지 않음
primary_view = 진행 방향 뷰                # 전진 CAM_FRONT / 후진(REVERSE) CAM_BACK
aux_views    = 슬롯 배정 (최대 aux_max=2장):
   슬롯 1  안전 기본값: m(c) ≥ m_guard(0.9)인 비-primary 채널 — 검출 0이어도 켠다
   슬롯 2~ 위험:        risk(c) ≥ θ(0.10) 인 채널을 위험 순으로
attention_view = argmax_{c ≠ primary} s(c) # 단일 선택 — 베이스라인 비교용으로 유지
soft_label     = softmax(s / T_soft), primary 채널 마스킹
layout         = 기동 상태 → 템플릿 매핑 (constants.LAYOUT_MAP, K=5)
```
- **왜 임계값 다중 선택인가**: 항상 1장을 고르면 ① 위험도 기동도 없는 프레임에서도
  화면이 켜져 오퍼레이터가 보조 뷰를 무시하게 되고 ② 차선변경처럼 측방과 후측방이
  동시에 중요한 상황을 표현할 수 없다.
- **왜 슬롯을 역할로 나누는가**: 하나의 임계값을 합산 s(c)에 걸면 기동 항이 슬롯을
  독점해 실제 위험이 밀려난다(차선변경 중 위험 커버 60.6%). 역할 분리 후 70.5%.
- **왜 기동 항을 평활하지 않는가**: EMA는 프레임마다 흔들리는 *검출 위험*을 안정화하는
  장치다. 기동 상태는 이미 ±3초 윈도우로 추론된 안정 신호라 다시 평활하면 이중
  스무딩이 된다. 실제로 STRAIGHT 사이에 낀 차선변경 프레임에서 사각지대 항이
  0.4→0.12로 희석돼 안전 기본값이 꺼졌다.
- **실측 비교 (34,149 keyframe, 임박위험 보유 19.6%)**
  | 설계 | 평균 표시 | 0장 | 1장 | 2장 | 위험 커버 |
  |---|---|---|---|---|---|
  | 항상 1장 | 1.00 | 0% | 100% | 0% | 63.1% |
  | 역할 분리 슬롯 | 1.03 | 31% | 35% | 34% | **87.8%** |
  θ를 더 올리는 것은 손해다 — θ=0.15에서 표시 0.84장·커버 84.8%, θ=0.20에서
  0.71장·78.6%로 커버리지가 표시량보다 빠르게 떨어진다.
- **왜 임계값 다중 선택인가 (v2)**: 항상 1장을 고르면 ① 위험도 기동도 없는 프레임에서도
  화면이 켜져 오퍼레이터가 보조 뷰를 무시하게 되고 ② 차선변경처럼 측방과 후측방이 동시에
  중요한 상황을 표현할 수 없다. 실측 비교(동일 부하): 항상 1장 → 평균 1.00장·임박위험
  커버 58.2%, 임계값 방식 → 평균 1.06장·커버 **75.0%**. 보조 0장 30% / 1장 34% / 2장 36%.
- **임계값은 위험 항이 아니라 s(c) 전체에 건다.** 기동 항 덕분에 검출된 객체가 없어도
  차선변경 방향은 켜진다(λ·m: 측방 0.4, 후측방 0.2 > θ=0.15). 검출 실패 시 화면이 꺼지는
  것이 이 시스템의 가장 위험한 실패 모드이기 때문이다.
- **λ=0.4인 이유**: λ=1.0이면 기동 항이 보조 슬롯 2개를 독점해 실제 위험을 밀어냈다
  (차선변경 중 위험 커버 70.7%). λ=0.4에서 83.1%로 회복된다.
- **원칙(이동체 안전)**: 진행 방향(primary)은 UI 메인 위치에 **상시 표시**하고, 위험도·기동으로
  고른 뷰(attention)를 보조 위치에 강조한다. 원래 명세의 단일 `main_view=argmax`를 대체.
  → 논문 기여 프레임이 "단일 메인뷰 선택"에서 **"보조 주의뷰 선택"**으로 이동.
- **정지(STOP) 뷰 유지**: STOP & 비-극단위험(min_TTC ≥ `extreme_ttc`=1.5s) & `stop_hold`이면
  직전 primary/attention을 유지(전환 억제). 극단 위험 시엔 위험 뷰로 즉시 전환.
- 레이아웃 템플릿 v1 (K=5): ①전방 중심 ②좌회전형 ③우회전형 ④후방형 ⑤차선변경형.
  기동 클래스에서 결정적으로 매핑 (v1은 단순화 — 논문 2.1의 "사전 정의 템플릿 분류"와 일치).
- 출력: `data/labels_{version}.parquet`
  컬럼: `sample_token, scene_token, timestamp, s_0..s_5, primary_view, attention_view,
  aux_0..aux_5, n_aux, soft_0..soft_5, maneuver, layout, n_objects, min_ttc`

### (7) 라벨 안정화 (라벨 자체의 flicker 방지)
- scene 내 시간순으로 s(c)에 EMA 스무딩(α=0.7) 적용 후 attention 결정.
- attention_view가 1 keyframe(0.5s)만 유지되고 되돌아가는 단발 전환은 이전 뷰로 병합.
- **보조 뷰 집합에도 같은 병합을 적용**(A B A → A). 기동 항을 평활하지 않게 되면서
  기동 경계에서 집합이 튀는 빈도가 늘었기 때문. 남은 flicker(집합 기준 0.494)는
  모델의 temporal consistency 항(β)과 추론 히스테리시스로 다룬다.

## B.5 모델 상세 명세 (M4)

### 입력 텐서 (배치 B, 시퀀스 T=4 keyframes ≈ 2초)
- `img_feat`: (B, T, 6, D_f) — 캐시된 백본 특징 (MobileNetV3-small, D_f=576 GAP)
- `can`: (B, T, d_can) — [speed, steering, accel_x, accel_y, yaw_rate] 정규화, d_can=5
- `risk`: (B, T, 6) — 채널별 Σw(TTC) (risk_source 조건에 따라 gt/noisy/none)
- 결측 CAN은 0 + 마스크 플래그 1차원 추가 (d_can=6)

### 구조
```
img_feat → Linear(D_f→d) ┐
camera positional emb(6) ─┤→ 채널 토큰 6개 (d=128)
risk → Linear(1→d) 채널별 가산 ┘
can → Linear(d_can→d) → CAN 토큰 1개
[6 채널 토큰 + 1 CAN 토큰] → TransformerEncoder(layers=2, heads=4, d=128, ff=256)
→ 채널 토큰 6개 추출 → 프레임 표현 (B, T, 6, d)
→ 채널별 GRU(d→d, 1층) 시간축 처리 → 마지막 스텝 (B, 6, d)
→ Head1: 채널별 스칼라 스코어 → softmax = attention_view 분포 (6-way, 학습 타깃)
→ Head2: 6채널 표현 mean-pool → MLP → 레이아웃 (K=5-way)
  (primary_view는 진행방향으로 결정적 산출 — 예측 대상 아님)
```
- 파라미터 규모 ~1M 내외 (백본 캐시 제외). 24GB GPU에서 여유.

### 손실
```
L = CE_focal(attention_view; γ=2) + α·KL(soft_label ‖ pred_dist)
  + γ_l·CE(layout) + β·Σ_t ‖p_t − p_{t−1}‖²   (temporal consistency)
기본: α=0.5, γ_l=0.3, β=0.1 (config)
```
- 클래스 불균형: focal loss + WeightedRandomSampler(기동 상태 기준 균형).

### 학습 설정
- AdamW lr=1e-4, weight_decay=1e-4, cosine decay, max 100 epoch,
  early stopping(val macro-F1, patience 10), batch 16, seed 42.
- `train.risk_noise: true`가 기본 — noise.py의 오류 시뮬레이터를 학습 증강으로 적용.

## B.6 검출 오류 시뮬레이터 (`data/noise.py`) — 원칙 2의 구현체

GT 객체 리스트에 다음을 적용해 위험도 벡터를 재계산:
- **누락(miss)**: 각 객체를 확률 p_miss로 제거. 원거리일수록 확률 가중
  (`p = p_miss · min(1, r/30m)` 형태).
- **오검출(FP)**: 프레임당 λ_fp ~ Poisson 개의 가짜 객체를 무작위 채널·거리에 추가.
- **위치/속도 노이즈**: 가우시안 σ_pos(거리 비례), σ_vel.
- **채널 블랙아웃**: 지정 채널의 img_feat를 0으로, 위험도는 유지/제거 옵션.
- 강도 프리셋: `light / medium / heavy` (M7의 x축).

### 난수 정책 (원칙 2·4의 교차점 — 위반 시 M7이 무효)
- **학습 증강**: 시드는 `(seed, epoch, index)`. 같은 샘플이라도 epoch마다 다른 노이즈
  실현을 봐야 한다. 인덱스만으로 시드하면 고정된 손상 입력을 암기하게 되어 강건성이
  증명되지 않는다. 학습 루프는 epoch마다 `dataset.set_epoch(ep)`를 호출한다
  (DataLoader 이터레이터 생성 **전에** 호출해야 워커에 반영된다).
- **평가**: 시드는 `(seed, index)`만. 제안 모델과 B2가 **동일한 손상 입력**을 봐야
  열화 곡선 비교가 성립한다.
- **아이템 단위 고정**: 검출 품질(프리셋)과 카메라 결손(블랙아웃 채널)은 2초 시퀀스
  안에서 유지되는 성질이므로 프레임마다 다시 뽑지 않는다.

## B.7 베이스라인 명세 (M5)

| ID | 정의 | 구현 |
|---|---|---|
| B1 고정 | 항상 CAM_FRONT | 상수 예측기. Top-1 = 전방 라벨 비율, flicker=0 |
| B2 규칙 | 라벨 함수를 추론 입력에 직접 적용 | `label_function`을 risk_source 조건 입력으로 실행. **GT 입력 시 만점이 정상** |
| B3 영상 단일 | 제안 모델 − CAN·위험도 토큰 | config로 토큰 제거 |
| B4 단일 프레임 | 제안 모델 − GRU, T=1 | config로 시퀀스 길이 1 |

## B.8 평가 명세 (M6)

### 지표 (`eval/metrics.py`)
- **Top-1 / Top-2**: 골드셋은 복수 정답 허용 → 예측이 정답 집합에 포함되면 정답.
- **Macro-F1**: 6채널 기준 (골드셋 + 의사 라벨 시험셋 둘 다).
- **Flicker rate**: 뷰 전환 후 1.5초(3 keyframes) 내 재전환된 전환의 비율.
  scene 단위로 시간순 추론 후 계산.
- **평균 보조 뷰 수**: 인지 부하의 대리 지표. 표시 방식과 무관하게 계산된다.
- **위험 사건 포착률 (재현율, n=1,879)**: '실제 위험'(B.4-(2) 정의) 객체가 선택 뷰
  (primary ∪ aux) 화각에 들어있는 비율. GT 기하로 직접 계산 — 순환 없음.
- **위험 슬롯 정밀도**: 위험으로 켠 채널 중 실제 위험이 있던 비율. **반드시 병기한다.**
  재현율만 보면 "많이 켜면 오르는" 지표라 튜닝이 왜곡된다.

### 이 시스템은 충돌 경보기가 아니다 (2026-09 측정으로 확정)
실제 위험은 프레임의 5.3%뿐이다. 프레임마다 1장을 켜면 정밀도 상한이 산술적으로 ~5%대로
묶인다. 실제로 커널(τ·거리)을 아무리 조여도 정밀도는 6.3→6.5%로 움직이지 않았고,
**정밀도를 결정한 것은 표시량**이었다.

| 평균 표시 | 정밀도 | 재현율 |
|---|---|---|
| 1.03장 | 6.5% | 96.9% |
| **0.59장 (채택)** | **13.3%** | **89.4%** |
| 0.30장 | 29.5% | 35.5% |

0.59장을 운영점으로 채택한다 — 그 아래는 재현율이 무너진다. 현행(항상 1장) 대비
표시량 41%↓, 정밀도 3.8→13.3%, 재현율 69.4→89.4%로 세 축 모두 우세하다.
논문에는 **운영 곡선 전체**를 싣고 이 지점을 대표 운영점으로 표시한다.
그리고 이 시스템이 위험 경보기가 아니라 **주의 배분기**임을 본문에 명시한다.
- **운영 곡선**: 추론 임계값 τ를 쓸어가며 (평균 보조 뷰 수 vs 가시화율)을 그린다.
  강건성(M7)도 스칼라가 아니라 이 곡선의 열화로 보고한다.

### 표시 방식 — 두 가정을 병기 (2026-09 결정)
같은 모델 출력을 두 UI 가정으로 해석해 보고한다. 계산은 동일하고 해석만 다르다.
- **하이라이트(정보 손실 없음, 본문 기본 가정)**: 6채널을 모두 표시하되 선택 뷰를 강조.
  놓친 위험 = '주의 유도 실패'(정보는 화면에 남아 있음).
- **전환(선택 뷰만 표시)**: 놓친 위험 = 안전 실패.
- 이 연구의 전제가 "검출은 불완전하다"이므로, 불완전한 검출로 카메라를 끄는 전환 방식은
  자기모순 위험이 있다. 논문 본문은 하이라이트를 기본 가정으로 서술하고 전환은 비교로 둔다.
- **한정 자원은 대역폭이 아니라 오퍼레이터의 주의다.** 6채널 균등 표시는 한정된 주의를
  균등 배분하는 것이고, 본 연구는 그 배분을 상황에 따라 재조정한다. 자원 절감 주장은
  '평균 보조 뷰 수'로만 하며, 전송량·화질 관련 주장은 하지 않는다(A.1 검증 범위 참조).

### 골드셋 절차
1. test 분할에서 **위험 사건을 과대표집**해 300 keyframe 선정:
       위험 100 ('실제 위험' 사건) / 비직진 기동 100 / 무작위 100
   자연 분포로 300개를 뽑으면 위험 사건이 16건뿐이라(5.3%) 재현율 신뢰구간이
   ±20%p를 넘어 안전 관련 주장을 할 수 없다. 위험 층 100건이면 ±8%p 수준이 된다.
   **추출 확률을 함께 기록해 역확률 가중으로 전체 분포 기준 추정치를 복원한다.**
   표 1에는 재가중 추정치를 주 수치로, 층별 수치를 함께 싣는다.
2. `tools/goldset_ui/`: 6채널 이미지를 격자로 보여주고 숫자키 1~6으로 주의(attention) 뷰 선택
   (진행방향 primary는 상시 표시 전제), `0`=복수 정답(추가 선택 모드), 방향키로 이동.
   로컬 HTML+JS 또는 간단한
   matplotlib/OpenCV 키 입력 루프로 구현 — **연구자(사용자)가 직접 라벨링**한다.
   Claude Code는 UI만 만들고 라벨링 작업은 사용자에게 요청할 것.
3. 결과: `data/goldset_v1.json` — {sample_token: [정답 채널 인덱스 목록]}.

### 표 1 생성
- `scripts/06_evaluate.py --all` → `results/table1.md`
- 열: 방법 × {Top-1(gold), Top-2(gold), Macro-F1(gold), Macro-F1(pseudo-test), Flicker, 가시화율}
- 평가 기본 입력 조건: `risk_source=noisy(medium)` — 본 표의 공정 비교 조건.
  `risk_source=gt` 결과는 부록 표로 별도 저장.

## B.9 Ablation & 강건성 (M7)

### Ablation (각각 config 1개, 학습 재실행)
| 태그 | 제거 요소 |
|---|---|
| A1 | − CAN 토큰 |
| A2 | − 위험도 토큰 |
| A3 | − GRU (T=1, B4와 동일) |
| A4 | − cross-view attention (Transformer → 채널별 독립 MLP + concat) |
| A5 | − temporal consistency 항 (β=0) |

### 강건성 프로토콜
- x축: 노이즈 강도 {clean, light, medium, heavy} × 유형 {miss, FP, pos-noise, 채널 드롭 1개}
- 비교: 제안 모델(noise 증강 학습) vs B2(규칙) — 동일 입력 조건.
- 산출: `results/robustness.png` (유형별 4개 서브플롯, y=Top-1 gold셋) + md 표.
- **기대 결과**: clean에서는 B2 ≥ 제안 모델일 수 있으나(정상), 강도가 올라갈수록
  제안 모델의 열화가 완만해야 함. 그렇지 않으면: ① noise 증강 강도 상향
  ② img_feat 기여 확인(A2와 비교) ③ 원인 분석 보고 후 사용자와 상의.

---

# PART C. 마일스톤 실행 계획

## M1 — 데이터 접근·탐색 (0.5~1일)
- [ ] `nusc_wrapper.py`: scene 목록, sample 순회, 6채널 이미지 경로, GT 박스(ego 좌표 변환),
      CAN 시계열(pose/steeranglefeedback/zoe_veh_info) 접근 함수화. CAN blacklist 처리.
- [ ] `scripts/01_explore_data.py`: 임의 sample의 채널별 이미지 경로 + 박스 수 +
      CAN 스냅숏 출력, scene별 keyframe 수/기간 요약 표.
- ✅ mini 10개 scene 전체에서 에러 없이 실행, 출력 로그 저장.

## M2 — 의사 라벨 파이프라인 (2~3일) ★
- [ ] `geometry.py` 투영 유틸 + 단위 테스트 (전방 10m 객체 → CAM_FRONT 판정 등 5케이스).
- [ ] `ttc.py`, `maneuver.py`, `label_function.py` 구현 (B.4 명세 그대로).
- [ ] `scripts/02_generate_labels.py --version v1.0-mini` → parquet 생성.
- [ ] `scripts/03_visualize_labels.py`: 무작위 20 keyframe에 대해 6채널 격자 이미지 +
      투영 박스 + 채널 점수 막대 + 선택 뷰 하이라이트 + 기동 상태 텍스트를 1장 PNG로
      → `results/label_viz/`. **사용자 검수 필수 — 여기서 승인받고 진행.**
- [ ] 라벨 통계 리포트: 채널 분포, 기동 분포, 라벨 flicker rate (스무딩 전/후).
- ✅ 단위 테스트 통과 + 시각화 20장 사용자 승인 + 라벨 분포가 극단 편중
  (전방 > 85%)이 아닐 것. 편중 시 λ·tau 튜닝 후 재생성.

## M3 — Dataset/로더 + 특징 캐시 (1~2일)
- [ ] `feature_cache.py`: MobileNetV3-small(timm, pretrained) GAP 특징을 keyframe별로
      추출해 `cache/{version}/feat_{camera}.npy` (memmap) 저장. 재실행 시 스킵.
- [ ] `dataset.py`: (B.5 입력 명세) 시퀀스 조립, scene 경계 처리(패딩+마스크),
      risk_source 분기, noise.py 연동, WeightedRandomSampler.
- [ ] `data/contrib_cache.py`: 위험 기여(GT 박스 투영) + CAN 특징을 npz로 캐시.
      Dataset이 실행마다 재계산하면 trainval 34k keyframe × 실험 10여 회가 낭비된다.
      캐시 키는 contributors 값을 바꾸는 LabelConfig 필드만 반영.
- [ ] `scripts/04_build_cache.py` + 로더 1 epoch 순회 벤치마크.
- ✅ 배치 shape 검증 테스트 통과, mini 기준 1 epoch < 1분 (캐시 사용 시).

## M4 — 모델 + 학습 (2~3일)
- [ ] `models/proposed.py` (B.5 구조), config 플래그로 토큰/GRU/융합 구조 제거 가능하게
      (ablation·B3·B4 공용 코드).
- [ ] `scripts/05_train.py`: config 로드 → 학습 → best ckpt + metrics 저장, tensorboard.
- [ ] mini 과적합 sanity check (train Top-1 > 95% 도달 확인).
- [ ] trainval(또는 확보된 부분집합) 본 학습 — 시작 전 데이터 준비 상태를 사용자에게 확인.
- ✅ val 곡선 정상 수렴, best ckpt 저장, 학습 로그 보고.

## M5 — 베이스라인 (1일)
- [ ] B1 상수 예측기, B2(`baselines/rule_based.py` — label_function 재사용 + risk_source 입력),
      B3/B4는 config로 학습.
- ✅ 4개 전부 `06_evaluate.py`에서 동일 인터페이스로 평가 가능.

## M6 — 평가 + 골드셋 (2일)
- [ ] `metrics.py` + 단위 테스트 (flicker 계산 등 합성 시퀀스 3케이스).
- [ ] 골드셋 후보 300개 층화 추출 + 검수 UI 제작 → **사용자에게 라벨링 요청** (예상 2~3시간).
- [ ] `06_evaluate.py --all` → `results/table1.md` (+ 부록: gt 입력 조건 표).
- ✅ 표 1 전 칸 채움, 골드셋 라벨링 완료.

## M7 — Ablation + 강건성 (2일)
- [ ] A1~A5 config 실행 → `results/ablation.md`.
- [ ] `07_robustness.py` → 열화 곡선 4종 + md 표.
- ✅ 강건성 곡선에서 제안 모델 우위 확인. 미확인 시 B.9의 대응 절차 수행 후 보고.

## M8 — 정성 결과 + 논문 반영 자료 (1일)
- [ ] `08_make_figures.py`: TURN/REVERSE/LANE_CHANGE 상황별 성공 2 + 실패 1 사례를
      제안/B2/골드 정답 나란히 배치한 그림 (그림 2용) → `results/fig2/`.
- [ ] `results/paper_numbers.md`: 표 1 + ablation + 강건성 핵심 수치 + 실험 설정 요약
      (논문 2.4~2.5절에 그대로 옮길 수 있는 형태).
- ✅ 전체 재현 커맨드 목록(`README.md`) 작성.

---

# PART D. 자주 발생할 문제 & 대응

| 증상 | 원인/대응 |
|---|---|
| CAN 로드 에러 | 해당 scene이 blacklist — 스킵 목록 처리, mini에도 일부 존재 가능 |
| box_velocity가 NaN | 인접 keyframe 차분으로 폴백 (ttc.py에 구현) |
| 라벨이 전방에 90%+ 편중 | tau↓(위험 민감도↑) 또는 λ↑, 기동 임계 완화 후 재생성. M2 ✅ 조건 |
| 학습이 B2 성능에 못 미침(gt 입력) | 정상 범위 — 표 1은 noisy 조건이 기준. gt 조건은 상한 참고 |
| noisy 조건에서도 B2에 짐 | 학습 noise 증강 강도를 평가 강도와 정합시키고, A2 결과로 영상 특징 기여 확인 |
| GPU OOM | batch↓ → grad accumulation, 캐시 특징 사용 확인 (원본 이미지 학습 아님) |
| flicker 높음 | β↑, 라벨 스무딩 α 조정, 추론 시 히스테리시스 후처리(최후 수단, 적용 시 표에 명시) |

# PART E. 사용자 확인이 필요한 결정 지점

Claude Code는 아래 지점에서 반드시 진행을 멈추고 사용자에게 확인을 요청한다:
1. **M2 완료 시**: 라벨 시각화 20장 검수 승인
2. **M4 본 학습 전**: trainval 데이터 확보 범위(전체/부분집합) 확인
3. **M6**: 골드셋 300개 라벨링 (사용자 직접 작업)
4. **M7**: 강건성 결과가 기대와 다를 경우 대응 방향 상의
