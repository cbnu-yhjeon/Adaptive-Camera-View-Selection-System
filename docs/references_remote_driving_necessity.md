# 원격 운전의 필요성을 뒷받침하는 문헌 (2026-10-13 조사)

논문·포스터 서론의 배경 근거로 쓰기 위한 정리다.

**확인 방법**
- 학술 문헌은 Crossref API로 DOI를 조회해 저자·게재처·연도·권호·쪽을 확인했다.
- 인용문은 모두 **초록**에서 그대로 옮겼다(Semantic Scholar · OpenAlex · arXiv · 저장소). 본문 서론은 열람하지 못했다.
- 법제도는 정부·기관의 공식 페이지에서 확인했다.

**CLAUDE.md A.1 제약**
- 대역폭 절감, 1인 다수 차량 감시(처리량·인력 절감), 지연은 필요성 근거로 쓰지 않는다.
- 지연은 설계 파라미터(위험 지평 5초) 근거로만 쓴다.
- 본 연구는 원격 **운전**(RD, 직접 제어)을 전제한다. 원격 **지원**(RA)과 구분한다.

유형 표기: (a) 자율주행 한계·엣지 케이스·ODD, (b) 법·규제, (c) 상용화 수단, (d) 원격 오퍼레이터의 인적요인 난제

---

## A. 필요성을 직접 말하는 학술 문헌 (권장 순)

| # | 문헌 | 초록 인용 | 유형 · 모드 | 주의 |
|---|---|---|---|---|
| 1 | Zhao, L., Nybacka, M., Aramrattana, M., Rothhämel, M., Habibovic, A., Drugge, L., Jiang, F. "Remote Driving of Road Vehicles: A Survey of Driving Feedback, Latency, Support Control, and Real Applications." *IEEE Trans. Intelligent Vehicles* 9(10):6086–6107, 2024. DOI 10.1109/TIV.2024.3362597 | "Currently, remote driving is identified as vital for commercialization" | (a)(c) · **RD 전용 서베이** | 지연은 난제 목록의 하나로만 다룬다. 낮은 상황인식을 핵심 난제로 꼽는 점이 본 연구와 맞는다 |
| 2 | Majstorović, D., Hoffmann, S., Pfab, F., Schimpe, A., Wolf, M.-M., Diermeyer, F. "Survey on Teleoperation Concepts for Automated Vehicles." IEEE SMC 2022, pp. 1290–1296. DOI 10.1109/SMC53654.2022.9945267 | "teleoperation can be established as a reliable fallback solution for operational design domain limits and edge cases of AD functions" | (a) · RD·RA 분류 | 없음 |
| 3 | Tener, F., Lanir, J. "Driving from a Distance: Challenges and Guidelines for Autonomous Vehicle Teleoperation Interfaces." CHI '22, pp. 1–13, 2022. DOI 10.1145/3491102.3501827 | "AVs will not be able to resolve all road incidents and that in some situations remote human assistance will be required" / "remote driving is not trivial and introduces many challenges stemming mostly from the physical disconnect of the remote operator" | (a)(d) · RD·RA | 전문가 14명 인터뷰 연구다. 필요성과 본 연구의 문제(물리적 분리)를 한 논문으로 잇는다 |
| 4 | Goodall, N. J. "Non-technological challenges for the remote operation of automated vehicles." *Transportation Research Part A* 142:14–26, 2020. DOI 10.1016/j.tra.2020.09.024 | "No existing automated vehicle can operate in all conditions and environments." | (a)(b) · RD | ⚠ 본문의 '차량군 원격 오퍼레이터 인력 산정' 부분은 인용하지 않는다 |
| 5 | Mutzenich, C., Durant, S., Helman, S., Dalton, P. "Updating our understanding of situation awareness in relation to remote operators of autonomous vehicles." *Cognitive Research: Principles and Implications* 6, Art. 9, 2021. DOI 10.1186/s41235-021-00271-8 | "even entirely driverless vehicles will sometimes require remote human intervention" | (a)(d) · RD | 원격 운전자의 상황인식은 차에 탄 운전과 근본적으로 다르다고 본다 |
| 6 | Amador, O., Aramrattana, M., Vinel, A. "A Survey on Remote Operation of Road Vehicles." *IEEE Access* 10:130135–130154, 2022. DOI 10.1109/ACCESS.2022.3229168 | "a bridge towards driverless mobility by providing human assistance remotely when an automated driving system finds a situation that is ambiguous" | (a)(c) · RD·RA | 없음 |
| 7 | Hans, O., Walter, B. "Remote Assistance or Remote Driving: The Impact of Operational Design Domains on ADS-Supporting Systems Selection." IEEE IAVVC 2025, pp. 1–7. DOI 10.1109/IAVVC61942.2025.11219535 | "High-level Automated Driving Systems (ADS) can handle many situations, but they still encounter situations where human intervention is required." | (a) · RD vs RA 비교 | 본 연구가 원격 지원이 아닌 원격 운전을 전제하는 근거로 쓰기 좋다 |
| 8 | Brecht, D., Gehrke, N., Kerbl, T., Krauss, N., Majstorović, D., Pfab, F., Wolf, M.-M., Diermeyer, F. "Evaluation of Teleoperation Concepts to Solve Automated Vehicle Disengagements." *IEEE Open Journal of Intelligent Transportation Systems* 5:629–641, 2024. DOI 10.1109/OJITS.2024.3468021 | "remotely support highly automated vehicles through a human remote operator whenever a disengagement of the automated driving system is present" | (a) · RD·RA | 해제 사례 문헌으로 개념을 평가한다. 결론은 원격 운전(공유 제어)과 원격 지원 두 개념의 조합이다 |
| 9 | Tener, F., Lanir, J. "Investigating intervention road scenarios for teleoperation of autonomous vehicles." *Multimedia Tools and Applications* 83(21):61103–61119, 2024. DOI 10.1007/s11042-023-17851-z | "the prevailing consensus suggests that AVs will not be capable of addressing every traffic situation, necessitating remote human intervention in certain edge case scenarios" | (a) · 구분 없음 | 없음 |
| 10 | Bogdoll, D., Orf, S., Töttel, L., Zöllner, J. M. "Taxonomy and Survey on Remote Human Input Systems for Driving Automation Systems." FICC 2022, *Advances in Information and Communication*, Springer LNNS, pp. 94–108. DOI 10.1007/978-3-030-98015-3_6 | "Corner cases for driving automation systems can often be detected by the system itself and subsequently resolved by remote humans." | (a) · RD·RA | LNNS 권 번호는 미확인. SAE J3016 정의를 인용하고 있다(C절) |
| 11 | Parr, H., Harvey, C., Burnett, G., Sharples, S. "Investigating levels of remote operation in high-level on-road autonomous vehicles using operator sequence diagrams." *Cognition, Technology & Work* 26(2):207–223, 2024. DOI 10.1007/s10111-024-00762-w | "disengagement from the autonomous driving system by Level 5 vehicles, due to damage, operation outside of expected parameters or software failure" | (a)(b) · 감시·지원·관리·**운전** 4유형 | 없음 |
| 12 | Georg, J.-M., Diermeyer, F. "An Adaptable and Immersive Real Time Interface for Resolving System Limitations of Automated Vehicles with Teleoperation." IEEE SMC 2019, pp. 2659–2664. DOI 10.1109/SMC.2019.8914306 | "Due to the challenges of autonomous driving backup options like teleoperation become a relevant solution for critical scenarios an automated vehicle might face." | (a) · RD | 적응형 카메라 투영 HMI라 **관련 연구로도 가치가 있다** |
| 13 | Hans, O., Walter, B., Adamy, J. *IEEE Open Journal of Intelligent Transportation Systems* 6:722–737, 2025. DOI 10.1109/OJITS.2025.3574692 | "Remote driving has emerged as a solution for enabling human intervention in scenarios where Automated Driving Systems (ADS) face challenges" | (a)(d) · RD | 라스베이거스 실도로 원격 운전자 데이터를 쓴다 |
| 14 | Mutzenich, C., et al. "Situation Awareness in Remote Operators of Autonomous Vehicles: Developing a Taxonomy …" *Frontiers in Psychology* 12:727500, 2021. DOI 10.3389/fpsyg.2021.727500 | "Remote operators will have to build up a mental model of the remote environment facilitated by monitor view and video feed." | (d) | 영상 피드 기반 상황인식을 다뤄 **본 연구 문제와 직결**된다 |
| 15 | Escher, B., Herde, J., Nikolai, F., Riener, A. "Regulating Teleoperation on Public Roads: Key Takeaways From an Expert Workshop." IEEE IV 2025, pp. 2601–2606. DOI 10.1109/IV64158.2025.11097504 | "Teleoperation is considered an interim solution for the introduction of automated vehicles in public transport." | (b)(c) · RD | 규제 공백을 지적한다 |

### 필요성 문장만 쓰고 나머지는 인용하지 말 것

- **Neumeier et al.**, AutomotiveUI '19, pp. 186–197, 2019. DOI 10.1145/3342197.3344534
  - 인용(초록): "many limitations are around that could eventually prevent the deployment of automated driving systems (ADS) beyond SAE level 3"
  - ⚠ 주 기여가 지연 효과다. 지연은 설계 파라미터 근거로만 쓴다.
- **Kang, L., Zhao, W., Qi, B., Banerjee, S.**, HotMobile '18, pp. 19–24, 2018. DOI 10.1145/3177102.3177104
  - 인용(초록): "it is unlikely that such systems are going to ever achieve perfect accuracy under all conditions."
  - ⚠ 본문은 LTE·Wi-Fi 지연 실험이다.
- **Zhang, T.**, *IEEE Internet of Things J.* 7(12):11347–11354, 2020. DOI 10.1109/JIOT.2020.3028766
  - 인용(초록): "playing an important role in helping self-driving cars navigate through tricky situations they cannot handle on their own"
  - ⚠ 5G·클라우드 중심 구상이고 본문은 확인하지 못했다.
- **Kettwich, C., Schrank, A., Avsar, H., Oehl, M.**, *Applied Sciences* 12(9):4350, 2022. DOI 10.3390/app12094350
  - 인용(초록): "Remote operation bears the potential to roll out highly automated vehicles (AVs, SAE Level 4) more safely and quickly."
  - ⚠ 원격 감독자가 차내 안전요원을 대체한다는 인력 절감 논리와 맞닿아 있다.
- **Kamtam et al.**, *Sensors* 24(12):3957, 2024 — 지연 리뷰다. 필요성 근거로 쓰지 않는다.

## B. 자율주행이 아직 사람을 필요로 한다는 실증 근거

원격 운전을 직접 다루지 않는 간접 근거다. "자율주행은 해제가 발생한다"는 사실까지만 쓴다.

- **Dixit, V. V., Chand, S., Nair, D. J.** "Autonomous Vehicles: Disengagements, Accidents and Reaction Times." *PLoS ONE* 11(12):e0168054, 2016. DOI 10.1371/journal.pone.0168054
  - 확인: 서지 ✔, 초록 요지 ✔
  - 내용: 캘리포니아 자율주행 시험 데이터를 분석했다. 해제 시 운전자의 제어 인수 반응시간은 평균 약 0.83초였고, 사고 수는 자율 주행거리와 강하게 상관됐다.
  - 대상은 차내 운전자로의 인계다.
- **Favarò, F., Eurich, S., Nader, N.** "Autonomous vehicles' disengagements: Trends, triggers, and regulatory limitations." *Accident Analysis & Prevention* 110:136–148, 2018. DOI 10.1016/j.aap.2017.11.001
  - 확인: 서지 ✔, 초록 ✗ (공개 DB에 없음)
  - 제목 수준(해제의 추세·원인·규제 한계)으로만 인용한다. 수치는 원문을 확인한 뒤에 쓴다.
- **Kalra, N., Paddock, S. M.** "Driving to safety: How many miles of driving would it take to demonstrate autonomous vehicle reliability?" *Transportation Research Part A* 94:182–193, 2016. DOI 10.1016/j.tra.2016.09.010
  - 확인: 서지 ✔, 초록 ✗
  - 주행 시험만으로 신뢰성을 입증하기 어렵다는 논지의 간접 근거다. 수치는 원문을 확인한 뒤에 쓴다.
- A의 **Brecht et al. 2024**가 해제 문헌과 원격조작을 직접 연결한다. 이 절의 주 근거로 쓴다.

## C. 표준·법제도 (공식 출처 확인)

- **SAE J3016 (2021년 4월 개정, J3016_202104)**: 원격 지원과 원격 운전을 처음 정의했다. 표준 원문이 아니라 Bogdoll et al. 2022에 인용된 문구로 확인했다.
  - 원격 운전(3.24): "real-time performance of part or all of the DDT and/or DDT fallback [...] by a remote driver."
  - 원격 지원(3.23): "event-driven provision, by a remotely located human, of information [...] in order to facilitate trip continuation when the ADS encounters a situation it cannot manage."
  - SAE 웹사이트에서 표준을 무료로 받을 수 있다. 논문에 인용하기 전에 원문 대조를 권장한다.
- **독일 Straßenverkehr-Fernlenk-Verordnung (StVFernLV)**: 독일 연방교통부 보도자료(2025-07-29)에서 확인했다.
  - 공공도로에서 원격 조종 차량을 운행할 법적 틀을 처음 만들었다("schafft erstmalig einen Rechtsrahmen zum Betrieb von ferngelenkten Kraftfahrzeugen auf öffentlichen Straßen").
  - 2025년 7월 공포, **2025년 12월 1일 시행**, 5년 시험 운용("fünfjährige Erprobungsphase").
  - 법률 해설 기준(1차 출처 아님): M·N 차종이 대상이고 원격 운전자는 법적으로 운전자다.
  - 참고: 2021년 독일 자율주행법의 '기술 감독(Technische Aufsicht)'은 원격 운전이 아니라 감독·지원 역할이다. 원격 **운전**의 근거로는 StVFernLV를 쓴다.
- **영국 Law Commission, "Remote driving" 정부 자문**: 2023년 2월 발표했고, 기관 공식 페이지에서 확인했다.
  - 원격 운전 기술은 이미 존재한다("Technology that enables an individual to drive from outside a vehicle already exists").
  - 가시권 밖 원격 운전 사업자에 대한 신규 법정 면허제를 권고했다.
- **한국: 기아 컨소시엄 원격 운전 실증**: 현대차그룹 공식 뉴스(2025-11-27)에서 확인했다.
  - 2025년 4월 국토교통부로부터 원격 운전 규제 샌드박스 실증 특례를 받았다.
  - 2025년 11월 27일 제주 일반도로에서 실증 시연을 했다. 사전 주행은 한 달간 약 70시간, 1,000 km였다.
  - 원격 운전은 "외부 관제 센터에서 4G, 5G 무선 네트워크를 이용해 운전자가 없는 차량을 운행 및 제어하는 기술"로 설명됐다.
  - 컨소시엄은 기아·쏘카·에스유엠·KT다.
  - 국토부 보도자료 원문은 확인하지 못했다.

## D. 서론에 쓸 수 있는 문단 예시 (제약 준수)

> 자율주행 시스템은 아직 모든 교통 상황을 처리하지 못하며[Goodall 2020; Tener·Lanir 2022], 원격 조작은 운행설계영역(ODD) 한계와 엣지 케이스에 대한 대비책으로 제시된다[Majstorović 2022]. 원격 운전은 자율주행 상용화의 필수 요소로 꼽히고[Zhao 2024], 독일은 2025년 12월 공공도로 원격 운전을 허용하는 시행령(StVFernLV)을 시행했다. 국내에서도 2025년 규제 샌드박스로 일반도로 원격 운전 실증이 이뤄졌다. 그러나 원격 운전자는 차량과 물리적으로 분리되어 영상에 의존해 상황을 파악해야 하며[Tener·Lanir 2022; Mutzenich 2021], 본 연구는 이 문제를 다룬다.

## E. 확인하지 못했거나 보류한 것

- **Shi, E., Wolf, M.-M., … Plum, L.** "Towards a Joint Understanding of Remote Operation for Vehicles in Public Road Traffic." arXiv 2606.11336, 2026 — 동료심사를 확인하지 못한 프리프린트다.
- **Andersson, J.** "Legal challenges of remote operation …" Edward Elgar 단행본 챕터, 2024, pp. 193–213. DOI 10.4337/9781035321469.00017 — 서지는 확인했으나 초록을 얻지 못했다.
- 연구계획서(2026-09-13) 참고문헌에서 학술 문헌으로 교체할 대상:
  - [2] Krafcik 발언(언론 보도), [20] EE Times, [21] 시장조사 보고서 → A절의 동료심사 문헌으로 바꾼다.
  - [7]은 저자 표기가 틀렸다(de Winter → **Musicant, Botzer, Richmond-Hacham**, *Sensors* 24(18):6134, 2024).

## 출처

- Crossref API (각 DOI), OpenAlex, Semantic Scholar, arXiv
- 독일 연방교통부 보도자료: https://www.bmv.de/SharedDocs/DE/Pressemitteilungen/2025/038-hirte-fernlenkverordnung.html
- UK Law Commission, Remote driving: https://lawcom.gov.uk/project/remote-driving/
- 현대자동차그룹 뉴스(2025-11-27): https://www.hyundaimotorgroup.com/ko/news/CONT0000000000194590
- Bogdoll et al. (SAE J3016 인용): https://arxiv.org/abs/2109.08599
- SAE J3016 2021 개정 안내: https://sae.org/news/2021/06/sae-revises-levels-of-driving-automation
- StVFernLV 해설: https://www.gvw.com/aktuelles/blog/detail/strassenverkehr-fernlenk-verordnung-der-neue-rechtsrahmen-fuer-remote-driving-seit-dem-1-dezember-2025-in-kraft , https://cms.law/de/deu/legal-updates/fernlenk-verordnung-ermoeglicht-teleoperiertes-fahren
