# 원격주행을 위한 딥러닝 기반 적응형 카메라 뷰 선택 시스템

> Deep Learning-based Adaptive Camera View Selection System for Remote Driving

## 📋 연구 개요

원격 운전자의 상황 인지를 향상시키기 위해 주행 상황에 따라 자동으로 최적의 카메라 각도를 선택하는 AI 시스템 개발

## 🎯 연구 목표

- 주행 상황별 카메라 뷰 중요도 평가 딥러닝 모델 개발
- 실시간 최적 카메라 뷰 선택 알고리즘 구현
- 원격 운전자의 인지적 부하 감소 및 상황 인지 능력 향상
- 실용 가능한 프로토타입 시스템 개발

## 🗓️ 연구 일정 (6개월, 26주)

| 단계 | 기간 | 주요 활동 |
|------|------|-----------|
| **문헌 조사** | 1-2주 | 관련 연구 서베이, 기술 스택 학습 |
| **Phase 1** | 3-8주 (6주) | 데이터 수집, 전처리, 레이블 생성 |
| **Phase 2** | 9-16주 (8주) | 중요도 평가 모델 개발 및 학습 |
| **Phase 3** | 17-22주 (6주) | 뷰 선택 알고리즘 구현 |
| **Phase 4** | 23-24주 (2주) | 시스템 평가 및 성능 분석 |
| **버퍼** | 25-26주 (2주) | 논문 작성, 최종 점검 |

## 📊 Phase 1: 데이터 수집 및 전처리 (6주)

### 데이터셋
- **nuScenes Full Dataset** (350GB)
  - 1,000+ 주행 시나리오
  - 6방향 카메라 (전방, 전방좌/우, 후방, 후방좌/우)
  - LiDAR, RADAR 센서 데이터
  - 보스턴/싱가포르 도시 환경

### 전처리 작업
1. 6개 방향 카메라 영상 추출 및 동기화
2. 주행 상황 레이블링 (직진, 좌/우회전, 차선변경, 교차로, 정지 등)
3. 영상 정규화 (해상도: 640×480)
4. 중요 객체 어노테이션 활용 (차량, 보행자, 표지판 등)

### 중요도 레이블 생성 전략
```python
# 3가지 접근법 비교
1. 휴리스틱 기반: 주행방향 + 객체거리 + 객체종류
2. Saliency Map: 사전학습 모델의 어텐션 활용
3. 객체 탐지 기반: YOLO 결과로 중요도 계산
```

## 🧠 Phase 2: 중요도 평가 모델 개발 (8주)

### 모델 아키텍처
```
Input: 640×480 RGB 카메라 영상
Backbone: ResNet-50 (ImageNet 사전학습)
Output: 중요도 점수 (0~1, Sigmoid)
```

### 학습 설정
- **손실 함수**: MSE Loss 또는 Focal Loss
- **옵티마이저**: Adam (lr=0.001)
- **데이터 분할**: Train 70% / Valid 15% / Test 15%
- **Early Stopping**: Validation loss 기준

### 실험 계획
1. 베이스라인 모델 구축 (Week 1-2)
2. 하이퍼파라미터 튜닝 (Week 3-4)
3. 데이터 증강 및 개선 (Week 5-6)
4. 최종 모델 검증 (Week 7-8)

## 🎮 Phase 3: 뷰 선택 알고리즘 (6주)

### 규칙 기반 접근법 (Week 1-2)
```python
if situation == "교차로":
    views = [front, front_left, front_right]
elif situation == "좌회전":
    views = [front, front_left, rear_left]
elif situation == "차선변경(좌)":
    views = [front_left, rear_left]
```

### 학습 기반 접근법 (Week 3-6)

#### Option 1: 분류 모델
```
Input: [주행상황, 6개 카메라 중요도]
Output: 최적 카메라 조합 (Multi-label)
```

#### Option 2: 강화학습 (도전 과제)
- **알고리즘**: DQN 또는 PPO
- **State**: 현재 상황 + 중요도 점수
- **Action**: 표시할 카메라 조합
- **Reward**: 중요 객체 포함률 - 뷰 전환 페널티

## 📈 Phase 4: 시스템 평가 (2주)

### 정량적 평가 지표

| 메트릭 | 설명 | 목표 |
|--------|------|------|
| **객체 포함률** | 중요 객체가 선택된 뷰에 포함된 비율 | >90% |
| **처리 속도** | 실시간 처리 가능 여부 | >30 FPS |
| **뷰 전환 빈도** | 불필요한 전환 최소화 | <5회/분 |
| **지연 시간** | 상황 변화부터 뷰 선택까지 | <100ms |

### 비교 기준선
- Random 선택
- 고정 전방 카메라
- 수동 규칙 기반
- 제안 방법 (규칙+학습)

### Ablation Study
- 중요도 모델 유무
- 규칙 vs 학습 vs 하이브리드
- 백본 네트워크 비교

## 🛠️ 기술 스택

### 하드웨어
- **GPU**: NVIDIA RTX 3090 24GB (또는 학교 서버)
- **개발 환경**: 8코어 CPU, 32GB RAM

### 소프트웨어
```bash
# 핵심 라이브러리
Python 3.8+
PyTorch 2.0
OpenCV 4.x
NumPy, Pandas, Matplotlib
```

### 데이터
- nuScenes Dataset (무료, 학술 목적)
- 라이선스: CC BY-NC-SA 4.0

## 💰 예산 계획

### Option 1: 학교 서버 활용 (권장)
- 외장 SSD 2TB: 200,000원
- **총 예산**: ~200,000원

### Option 2: 클라우드 GPU
- Google Colab Pro+: 50,000원/월 × 6 = 300,000원
- 외장 SSD 2TB: 200,000원
- **총 예산**: ~500,000원

## 📦 예상 결과물

1. ✅ **프로토타입 시스템**
   - 실시간 카메라 뷰 선택 데모
   - 웹 또는 데스크톱 UI

2. ✅ **학습된 모델**
   - 중요도 평가 모델 (.pth)
   - 뷰 선택 정책 모델

3. ✅ **성능 평가 결과**
   - 정량적 메트릭 분석
   - 비교 실험 결과
   - 그래프 및 시각화

4. ✅ **데모 영상** (3-5분)
   - 다양한 주행 시나리오
   - 시스템 작동 과정

5. ✅ **소스 코드 & 문서**
   - GitHub 저장소
   - README 및 사용 설명서

6. ✅ **학술 논문**
   - 졸업 논문 (학술 논문 형식)
   - 학회 발표 가능 수준

## 📚 참고문헌

1. Caesar, H., et al. (2020). nuScenes: A multimodal dataset for autonomous driving. *CVPR 2020*.
2. Neumeier, S., et al. (2019). Teleoperation: The holy grail to solve the autonomous driving dilemma. *IEEE/RSJ IROS*.
3. He, K., et al. (2016). Deep residual learning for image recognition. *CVPR 2016*.
4. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540).

## 🎓 기대 효과

### 학술적 기여
- 원격주행 HMI 최적화 새로운 접근법
- CV + RL 융합 실용 사례
- 국제 학회 논문 투고 가능 (ITSC, IV, ICRA)

### 산업적 활용
- 자율주행 원격 제어 시스템
- 건설/농업 무인 장비
- 다중 카메라 모니터링 시스템

### 사회적 기여
- 원격 운전자 안전성 향상
- 자율주행 전환기 안전장치
- 고령자/장애인 운전 보조

## ⚠️ 리스크 관리

| 리스크 | 대응 방안 |
|--------|-----------|
| GPU 리소스 부족 | 학교 서버 우선, Colab 백업 |
| 레이블 생성 어려움 | 3가지 방법 병행, 비교 실험 |
| 강화학습 실패 | 규칙+분류 모델로 폴백 |
| 데이터 다운로드 지연 | Mini-nuScenes로 프로토타입 |
| 평가 기준 모호 | Proxy metrics 다각화 |

## 📝 마일스톤 체크리스트

- [ ] Week 2: 문헌 조사 완료
- [ ] Week 8: 데이터 전처리 완료
- [ ] Week 12: 중요도 모델 1차 완성
- [ ] Week 16: 중요도 모델 최종 완성
- [ ] Week 20: 뷰 선택 알고리즘 구현
- [ ] Week 24: 평가 완료
- [ ] Week 26: 논문 초고 완성

## 📧 Contact

- **연구자**: [이름]
- **이메일**: [이메일]
- **지도교수**: [교수님 성함]
- **소속**: [학교/학과]

---

**License**: MIT (코드) / CC BY-NC-SA 4.0 (문서)
