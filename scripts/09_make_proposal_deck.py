"""연구계획서 발표 덱(.pptx) 생성.

docs/research-proposal.html 의 내용을 발표용 19장으로 옮긴다. 그림은
results/figures/ 의 PNG를 쓰므로 export_figures.py 를 먼저 돌려야 한다.

수치를 이 파일에 손으로 적어 두는 것은 계획서와 어긋날 위험이 있으므로,
예비 결과·운영 곡선 값은 계획서에서 쓰는 것과 같은 출처(표 5)의 값만 옮긴다.

사용:
    python scripts/09_make_proposal_deck.py
    python scripts/09_make_proposal_deck.py --out results/slides/발표.pptx
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.slide_kit import (  # noqa: E402
    ACCENT, ACCENT_BG, BODY, INK, MARGIN, MONO, MUTED, RULE, SIGNAL,
    SIGNAL_BG, SURF2, W, H, WHITE,
    box, new_slide, page_num, para, picture, rich, rule_line, slide_head,
    table, textbox,
)

FIG = "results/figures"


# ---------------------------------------------------------------- 공용 조각
def stat(slide, x, y, w, value, label, *, color=ACCENT, vsize=30):
    """큰 수치 + 설명 한 줄."""
    tf = textbox(slide, x, y, w, 0.52)
    para(tf, value, size=vsize, bold=True, color=color, first=True)
    tf = textbox(slide, x, y + 0.50, w, 0.60)
    para(tf, label, size=11, color=MUTED, line=1.35, first=True)


def bullets(slide, x, y, w, items, *, size=12.5, gap=0.10):
    """· 표식 + 본문. items = [(강조어, 나머지)] 또는 문자열."""
    yy = y
    for it in items:
        head, rest = it if isinstance(it, tuple) else ("", it)
        dot = box(slide, x, yy + 0.09, 0.07, 0.07, shape=MSO_SHAPE.OVAL,
                  fill=ACCENT, line=None)
        dot.shadow.inherit = False
        tf = textbox(slide, x + 0.22, yy, w - 0.22, 0.4)
        parts = ([(head, True, INK)] if head else []) + [(rest, False, INK)]
        rich(tf, parts, size=size, line=1.38, first=True)
        lines = max(1, int(len(head + rest) / (w * 8.2)) + 1)
        yy += 0.24 * lines + gap
    return yy


def note(slide, x, y, w, h, label, text, *, fill=ACCENT_BG, edge=ACCENT):
    sh = box(slide, x, y, w, h, fill=fill, line=edge, line_pt=1.0)
    tf = sh.text_frame
    para(tf, label, size=9.5, bold=True, color=edge, font=MONO, first=True,
         space_after=4)
    para(tf, text, size=11.5, line=1.4, color=INK)
    return sh


# ---------------------------------------------------------------- 슬라이드
def s_cover(prs):
    s = new_slide(prs)
    box(s, 0, 0, W, 2.55, fill=INK, line=None)
    tf = textbox(s, MARGIN, 0.80, 11.0, 0.30)
    para(tf, "2026 연구계획서", size=12, color=RULE, font=MONO, first=True)
    tf = textbox(s, MARGIN, 1.18, 11.6, 0.90)
    para(tf, "원격주행을 위한 딥러닝 기반", size=27, bold=True, color=WHITE, first=True)
    para(tf, "적응형 카메라 뷰 선택 시스템", size=27, bold=True, color=WHITE)

    tf = textbox(s, MARGIN, 3.10, 11.6, 0.60)
    rich(tf, [("6채널을 균등 탐색하는 부담을 줄이되, ", False, INK),
              ("봐야 할 것을 놓치지 않는다", True, SIGNAL)],
         size=17, line=1.4, first=True)

    cards = [("자동 의사 라벨", "3D 객체 GT + CAN으로\n‘이상적 뷰’를 자동 생성"),
             ("다중 신호 융합", "영상 · CAN · 위험도를\nTransformer + GRU로 결합"),
             ("검출 오류 강건성", "불완전한 입력에서\n규칙 기반보다 덜 무너진다")]
    x = MARGIN
    for i, (t, d) in enumerate(cards):
        sh = box(s, x, 4.05, 3.72, 1.42, fill=WHITE, line=RULE)
        tf = sh.text_frame
        para(tf, f"C{i+1}", size=9.5, bold=True, color=ACCENT, font=MONO,
             first=True, space_after=5)
        para(tf, t, size=14, bold=True, space_after=5)
        para(tf, d, size=11, color=MUTED, line=1.35)
        x += 3.94

    rule_line(s, MARGIN, 6.28, W - 2 * MARGIN)
    tf = textbox(s, MARGIN, 6.46, 8.0, 0.50)
    para(tf, "충북대학교", size=12.5, bold=True, first=True)
    para(tf, "2026년 9월", size=11, color=MUTED)
    return s


def s_summary(prs):
    s = new_slide(prs)
    y = slide_head(s, "요약", "한 장으로 보는 연구",
                   sub="배경이 남긴 제약에서 문제를 좁히고, 그 문제를 학습으로 푼 뒤, 불완전한 입력에서 검증한다.")
    cols = [
        ("왜 필요한가", ACCENT, [
            "6채널이 같은 크기로 항상 떠 있다",
            "16화면에서 탐지 정확도 64%",
            "관제사는 27~62초마다 수동 전환",
            "그런데 실제 위험은 프레임의 5.3%",
        ]),
        ("무엇을 하는가", INK, [
            "강조할 보조 뷰 0~2개를 고른다",
            "진행 방향은 끄지 않고 상시 표시",
            "3D GT + CAN으로 의사 라벨 자동 생성",
            "0.46 M 경량 융합 모델",
        ]),
        ("무엇을 증명하는가", SIGNAL, [
            "동일 표시량에서 더 높은 위험 포착률",
            "검출이 나빠질수록 규칙 기반과 격차 확대",
            "골드셋 300장으로 순환 평가 차단",
            "노이즈 4종 × 4단계 열화 곡선",
        ]),
    ]
    x = MARGIN
    for title, col, items in cols:
        sh = box(s, x, y, 3.92, 3.50, fill=WHITE, line=col, line_pt=1.5)
        tf = sh.text_frame
        para(tf, title, size=14, bold=True, color=col, first=True, space_after=10)
        for it in items:
            para(tf, "· " + it, size=12, line=1.4, space_after=8)
        x += 4.13

    note(s, MARGIN, y + 3.74, W - 2 * MARGIN, 0.86, "핵심 주장",
         "clean 조건에서는 규칙 기반이 대등하거나 나을 수 있다 — 그것은 정상이다. "
         "성패는 검출 품질이 떨어질 때 열화 기울기가 더 완만한가에 달려 있다.",
         fill=SIGNAL_BG, edge=SIGNAL)
    return s


def s_industry(prs):
    s = new_slide(prs)
    y = slide_head(s, "1.1 연구 배경", "원격주행은 산업에서 어떻게 운용되는가",
                   sub="원격주행은 하나가 아니라 관여 수준의 스펙트럼이다. 본 연구의 전제를 먼저 고정한다.")
    rows = [
        ["구분", "조작 주체가 하는 일", "입력 장치", "레벨"],
        ["원격 운전", "직접 제어 · 공유 제어 · 궤적 유도", "스티어링 휠 · 페달", "1–2"],
        ["원격 지원", "경유점 유도 · 대화형 경로 계획 · 인지 수정", "마우스 · 키보드", "3–5"],
    ]
    table(s, MARGIN, y, W - 2 * MARGIN, rows, col_w=[0.16, 0.48, 0.24, 0.12])

    note(s, MARGIN, y + 1.20, W - 2 * MARGIN, 0.90, "본 연구의 전제",
         "원격 운전(직접 제어) — 오퍼레이터가 한 대의 차량을 실시간 조종하며 "
         "다채널 영상만으로 상황을 인지하는 조건이다. 1인이 다수 차량을 감시하는 "
         "원격 지원 모델은 전제도 평가 대상도 아니다.")

    y2 = y + 2.34
    stat(s, MARGIN, y2, 3.6, "연 1,000건", "‘Vehicle Teleoperation’ 특허 출원\n2000–2021년 지수적 증가")
    stat(s, MARGIN + 4.20, y2, 4.4, "Zoox · Ottopia\nVW · DriveU",
         "기술을 개발 중인 주요 기업\n(Cruise · Einride · Vay 등 다수)", vsize=17)
    stat(s, MARGIN + 8.80, y2, 3.3, "5채널", "차량 측 카메라 구성 사례\n전방 120° + 어안 180° 4대")

    tf = textbox(s, MARGIN, H - 0.72, 10.0, 0.26)
    para(tf, "출처: Majstorović 등, IEEE SMC 2022 [9] · Bogdoll 등, FICC 2022 [10]",
         size=9.5, color=MUTED, font=MONO, first=True)
    return s


def s_console(prs):
    s = new_slide(prs)
    y = slide_head(s, "1.2 연구 배경", "관제 화면은 실제로 어떻게 구성되는가",
                   sub="문헌이 보고하는 운전석도, 본 연구가 전제하는 조건도 — 영상 채널의 배치와 크기는 고정이다.")
    picture(s, f"{FIG}/fig1_console_layout_nocap.png", MARGIN, y - 0.10,
            W - 2 * MARGIN, 4.30)
    note(s, MARGIN, y + 4.28, W - 2 * MARGIN, 0.84, "여기서 남는 질문",
         "화면 구성을 상황에 따라 바꾸는 시도는 이미 있다. 그러나 바뀌는 대상은 "
         "정보 요소의 개수이지 “어느 카메라 영상을 강조할 것인가”가 아니다.",
         fill=SIGNAL_BG, edge=SIGNAL)
    return s


def s_human(prs):
    s = new_slide(prs)
    y = slide_head(s, "1.3 – 1.5 연구 배경", "사람은 다중 화면을 감시하지 못한다",
                   sub="직관이 아니라 반세기에 걸쳐 축적된 인간공학적 사실이다.")
    cards = [
        ("83% → 64%", "동시 감시 화면 4대에서 83%,\n9대 84%, 16대에서 64%로 탐지 정확도 하락\n(관찰자 204명 · 1시간)", "Tickner & Poulton, Ergonomics 1973 [8]", ACCENT),
        ("26.9 – 62.4초", "관제사는 멀티플렉스를 훑지 않는다.\n단일 스팟 모니터에 카메라를 띄워\n이 주기로 능동 전환한다", "Stainer 등, Front. Hum. Neurosci. 2013 [14]", ACCENT),
        ("감각의 결손", "속도 · 노면 경사 · 적재 하중을\n체감할 수 없고 엔진음도 듣지 못한다.\n전정기관이 받지 못하는 만큼 상황인식 저하", "Linkov & Vanžura, Front. Psychol. 2021 [15]", SIGNAL),
    ]
    x = MARGIN
    for val, desc, src, col in cards:
        sh = box(s, x, y, 3.92, 3.30, fill=WHITE, line=RULE)
        tf = sh.text_frame
        para(tf, val, size=23, bold=True, color=col, first=True, space_after=10)
        para(tf, desc, size=11.5, line=1.45, space_after=10)
        para(tf, src, size=9, color=MUTED, font=MONO, line=1.3)
        x += 4.13

    note(s, MARGIN, y + 3.54, W - 2 * MARGIN, 1.30, "그래서",
         "“어느 뷰를 볼 것인가”는 이미 사람이 수행 중인 과업이다 — 본 연구는 그것을 자동화한다.\n"
         "그리고 주행 상황은 초 단위로 바뀌므로 30초~1분 주기의 수동 전환은 원리적으로 부족하다. "
         "원격 조건에서는 “사람이 알아서 중요한 곳을 보겠지”라는 가정이 실차보다 덜 성립한다.")
    return s


def s_lever(prs):
    s = new_slide(prs)
    y = slide_head(s, "1.6 – 1.7 연구 배경", "표현을 바꾸면 부하는 움직인다 — 그런데 ‘언제’가 비자명하다")

    sh = box(s, MARGIN, y, 6.10, 2.55, fill=WHITE, line=ACCENT, line_pt=1.5)
    tf = sh.text_frame
    para(tf, "1.6  개입 가능성은 실증되어 있다", size=13.5, bold=True, color=ACCENT,
         first=True, space_after=10)
    para(tf, "카메라 시점을 운전자 눈높이에서 차량 루프로 올린 것만으로", size=11.5,
         line=1.4, space_after=8)
    para(tf, "경로 완료 시간 14% 단축 · 정신적 요구와 좌절감 감소", size=13, bold=True,
         color=INK, line=1.4, space_after=6)
    para(tf, "충돌률과 급제동 · 급조향 확률은 변하지 않았다 (n=148)", size=12,
         line=1.4, space_after=10)
    para(tf, "Musicant 등, Sensors 2024 [13] — 다만 바꾼 것은 카메라의 물리적 설치 "
             "위치이며 한 번 정하면 고정된다.", size=9.5, color=MUTED, font=MONO, line=1.35)

    sh = box(s, MARGIN + 6.35, y, 5.75, 2.55, fill=WHITE, line=SIGNAL, line_pt=1.5)
    tf = sh.text_frame
    para(tf, "1.7  그런데 언제 강조할지가 비자명하다", size=13.5, bold=True,
         color=SIGNAL, first=True, space_after=10)
    para(tf, "5.3%", size=34, bold=True, color=SIGNAL, space_after=4)
    para(tf, "본 프로젝트 측정상 ‘실제 위험’이 발생하는 프레임 비율", size=12,
         line=1.4, space_after=8)
    para(tf, "화면은 100% 켜져 있다.", size=13, bold=True, line=1.4)

    picture(s, f"{FIG}/fig2_hazard_rarity_nocap.png", MARGIN, y + 2.78,
            W - 2 * MARGIN, 1.60)
    tf = textbox(s, MARGIN, y + 4.52, 11.0, 0.26)
    para(tf, "‘실제 위험’ = 5초 내 3 m 이내 최접근 (trainval 34,149 keyframe 기준 1,879 채널)",
         size=9.5, color=MUTED, font=MONO, first=True)
    return s


def s_related(prs):
    s = new_slide(prs)
    y = slide_head(s, "2.1 관련 연구", "기존 연구 현황과 비교",
                   sub="‘적응 대상’이 결정적 차이다. 채널 선택 자체를 학습 문제로 정식화한 사례는 확인되지 않는다.")
    rows = [
        ["연구 흐름", "대표 연구", "적응 대상", "결정 방식", "검증"],
        ["영상 스트림 적응", "Hofbauer 등 [6]", "뷰별 비트레이트 · 프레임률 · 해상도\n— 모든 뷰를 계속 전송", "차량 상태 규칙\n+ 픽셀 ROI 예측", "CARLA\n시뮬레이터"],
        ["GUI 요소 적응", "Wolf 등 [12]", "표시 정보 요소의 개수\n— 영상 채널 배치는 고정", "원격주행 단계\n(규칙)", "클릭더미\nN=36"],
        ["운전자 주의 예측", "Xia [4] · Palazzi [5]", "(적응 아님) 픽셀 단위 시선 saliency", "시선 데이터 학습", "단일 전방\n카메라"],
        ["본 연구", "—", "강조할 카메라 채널 0~2개 — 이산 선택", "3D 객체 GT · CAN\n약지도 학습", "34,149 keyframe\n+ 검출 오류"],
    ]
    table(s, MARGIN, y, W - 2 * MARGIN, rows,
          col_w=[0.15, 0.15, 0.34, 0.20, 0.16], row_h=0.78, size=10.5)

    note(s, MARGIN, y + 3.50, W - 2 * MARGIN, 0.96, "검증 조건에서도 차이가 있다",
         "선행 연구는 시뮬레이터나 클릭더미 환경에서 사실상 GT 수준의 인지를 전제하고 평가했다. "
         "본 연구는 반대로 검출이 불완전할 때 무엇이 일어나는가를 주 실험으로 삼는다.",
         fill=SIGNAL_BG, edge=SIGNAL)
    return s


def s_limits(prs):
    s = new_slide(prs)
    y = slide_head(s, "2.1 관련 연구", "기존 기술의 한계 → 본 연구의 검증 항목",
                   sub="차별성을 주장으로만 두지 않기 위해, 각 한계에 그것을 판정할 실험을 미리 대응시킨다.")
    rows = [
        ["", "기존 연구에서 확인된 한계", "이 차이를 검증하는 실험"],
        ["L1", "적응 대상이 전송 품질 — 모든 뷰를 계속 전송하므로 어디를 볼지의 부담은 그대로 남는다", "운영 곡선\n(표시량 vs 포착률)"],
        ["L2", "영상 채널 배치는 고정 — 바뀐 것은 정보 요소의 개수이고, 평가도 클릭더미였다", "골드셋 Top-1 / Top-2\nB1(항상 전방) 대비"],
        ["L3", "시선 데이터에 의존하는 단일 전방 saliency — 다채널 원격주행에 옮길 수 없다", "C1 라벨 검수\nB3 · A1 · A2"],
        ["L4", "규칙 기반이며 검출 불확실성 하의 거동이 미검증 — GT 수준 인지를 전제하고 평가했다", "C3 강건성 곡선\n제안 vs B2"],
    ]
    table(s, MARGIN, y, W - 2 * MARGIN, rows,
          col_w=[0.06, 0.62, 0.32], row_h=0.70, size=11.5)

    note(s, MARGIN, y + 3.28, W - 2 * MARGIN, 1.10, "L4가 핵심이다",
         "L1–L3은 ‘무엇을 적응시키는가’의 차이이고 설계로 해소된다. "
         "그러나 L4는 ‘그 설계가 현실 조건에서 버티는가’의 문제여서 실험으로만 답할 수 있다. "
         "clean 조건에서 규칙 기반이 대등한 것은 정상이며, 성패는 열화 기울기에 달려 있다.",
         fill=SIGNAL_BG, edge=SIGNAL)
    return s


def s_goals(prs):
    s = new_slide(prs)
    y = slide_head(s, "3 연구 목표", "구현과 실험이 반드시 뒷받침해야 할 주장",
                   sub="세 가지로 고정한다. C3이 핵심이며, C1 · C2는 C3을 성립시키기 위한 전제다.")
    items = [
        ("C1", "3D GT와 CAN으로 ‘이상적 뷰’ 의사 라벨을 자동 생성하는 방법이 동작한다",
         "라벨 분포 통계 + 시각화 검수 + 골드셋과의 일치도", ACCENT),
        ("C2", "다중 신호 융합과 시간 모델이 단일 신호 · 단일 프레임보다 낫다",
         "베이스라인 B3 · B4 + ablation A1–A5", ACCENT),
        ("C3", "불완전한 입력에서 학습 모델이 규칙 기반보다 강건하다",
         "노이즈 유형 4종 × 강도 4단계 열화 곡선", SIGNAL),
    ]
    yy = y + 0.10
    for tag, claim, how, col in items:
        sh = box(s, MARGIN, yy, W - 2 * MARGIN, 1.26, fill=WHITE, line=col,
                 line_pt=1.5 if col is SIGNAL else 1.0)
        tf = textbox(s, MARGIN + 0.30, yy + 0.24, 1.0, 0.60)
        para(tf, tag, size=26, bold=True, color=col, font=MONO, first=True)
        tf = textbox(s, MARGIN + 1.40, yy + 0.22, 6.6, 0.80)
        para(tf, claim, size=14.5, bold=True, line=1.35, first=True)
        tf = textbox(s, MARGIN + 8.30, yy + 0.26, 3.4, 0.72)
        para(tf, "검증 방법", size=9, color=MUTED, font=MONO, first=True,
             space_after=4)
        para(tf, how, size=11, color=MUTED, line=1.35)
        yy += 1.26
    note(s, MARGIN, yy + 0.16, W - 2 * MARGIN, 1.05, "왜 C3이 핵심인가",
         "C1과 C2는 설계가 성립하는지를 묻는다 — 라벨이 만들어지는가, 융합이 단일 신호보다 나은가. "
         "그러나 실제 원격주행에서 검출은 불완전하고, 그 조건에서 무너지지 않는다는 것을 보이지 못하면 "
         "앞의 둘은 실험실 안에서만 참인 주장이 된다.",
         fill=SIGNAL_BG, edge=SIGNAL)
    return s


def s_bg_to_form(prs):
    s = new_slide(prs)
    y = slide_head(s, "4 문제 정의", "배경이 남긴 제약이 곧 정식화다",
                   sub="정식화의 각 항목은 임의로 정한 것이 아니라 1절의 사실에서 직접 따라 나온다.")
    rows = [
        ["배경", "1절에서 확인된 사실", "정식화에 들어간 결정"],
        ["1.1", "원격 운전(직접 제어) 전제", "진행 방향 뷰는 선택 대상에서 제외 · 상시 표시  y[p] = 0"],
        ["1.2", "영상 채널은 배치와 크기가 고정", "적응 대상을 채널 그 자체로 —  y ∈ {0,1}^6  이산 선택"],
        ["1.3", "4화면 83% → 16화면 64%", "표시량 최소화  min E[Σy]  · 동시 강조 상한  Σy ≤ 2"],
        ["1.4", "수동 전환 주기 26.9~62.4초", "keyframe마다(0.5초) 갱신 — 사람이 못 내는 주기를 메운다"],
        ["1.5", "속도 · 경사 · 엔진음을 못 느낀다", "영상만이 아니라 CAN과 위험도를 함께 융합"],
        ["1.6", "부하는 내려가고 충돌률은 그대로", "‘부하 감소’는 목적, ‘안전 유지’는 제약으로 분리"],
        ["1.7", "실제 위험은 프레임의 5.3%", "0장을 허용 — 매 프레임 강제하면 보조 뷰는 무시된다"],
    ]
    table(s, MARGIN, y, W - 2 * MARGIN, rows,
          col_w=[0.08, 0.36, 0.56], row_h=0.53, size=12)
    tf = textbox(s, MARGIN, y + 4.34, W - 2 * MARGIN, 0.40)
    rich(tf, [("이 대응은 왼쪽에서 오른쪽으로만 읽는다 — ", False, MUTED),
              ("정식화를 먼저 정하고 근거를 나중에 찾은 것이 아니다.", True, INK)],
         size=11.5, first=True)
    return s


def s_formulation(prs):
    s = new_slide(prs)
    y = slide_head(s, "4 문제 정의", "이산 선택 문제로의 정식화")

    sh = box(s, MARGIN, y, 7.55, 4.25, fill=RGB_PAPER(), line=RULE)
    sh.line.color.rgb = ACCENT
    sh.line.width = Pt(0)
    bar = box(s, MARGIN, y, 0.05, 4.25, fill=ACCENT, line=None)
    bar.shadow.inherit = False
    tf = sh.text_frame
    lines = [
        ("입력   x_t = { 영상특징 F ∈ R^(4×6×576),  CAN c ∈ R^(4×6),", True),
        ("                위험도 r ∈ R^(4×6) }   — 과거 4 keyframe(약 2초)", False),
        ("        위험도는 불완전한 검출기 출력을 가정한다.", False),
        ("", False),
        ("출력   y_t ∈ {0,1}^6          각 채널의 강조 여부", True),
        ("", False),
        ("제약   y_t[p] = 0             진행 방향은 선택 대상이 아님", False),
        ("        Σ y_t ≤ 2              동시 강조 상한", False),
        ("        x_t 는 시각 t 이하만    미래 정보 사용 금지", False),
        ("", False),
        ("목적   min  E[ Σ y_t ]         평균 표시 뷰 수 ↓", True),
        ("        s.t. 위험 사건 포착률 유지 · 향상", True),
    ]
    for i, (t, b) in enumerate(lines):
        para(tf, t, size=12, bold=b, font=MONO, line=1.32, first=(i == 0),
             color=INK if b else MUTED)

    sh = box(s, MARGIN + 7.85, y, 4.25, 4.25, fill=WHITE, line=SIGNAL, line_pt=1.5)
    sh.text_frame.margin_top = Inches(0.10)
    tf = sh.text_frame
    para(tf, "성공 기준", size=13.5, bold=True, color=SIGNAL, first=True,
         space_after=12)
    para(tf, "① 동일한 평균 표시 뷰 수 조건에서\n규칙 기반보다 높은 위험 사건 포착률",
         size=11.5, line=1.38, space_after=3)
    para(tf, "한계 L1 · L2 — 적응 대상이 전송 품질 ·\n정보 요소 수에 머물렀던 문제",
         size=9.5, color=MUTED, font=MONO, line=1.3, space_after=11)
    para(tf, "② 검출 품질이 나빠질수록\n규칙 기반과의 격차가 벌어질 것",
         size=11.5, bold=True, line=1.38, space_after=3)
    para(tf, "한계 L4 · 주장 C3 — 선행 연구가\nGT 수준 인지를 전제하고 평가한 문제",
         size=9.5, color=MUTED, font=MONO, line=1.3, space_after=11)
    para(tf, "①은 설계가 옳았는지를,\n②는 그 설계가 현실에서 버티는지를 묻는다.",
         size=11, bold=True, color=SIGNAL, line=1.35)

    note(s, MARGIN, y + 4.47, W - 2 * MARGIN, 0.90, "정식화가 아닌 것",
         "본 문제는 충돌 경보(collision warning)가 아니다. 출력은 “위험하다”는 경보가 아니라 "
         "“지금은 이쪽을 보는 편이 낫다”는 주의 배분이며, 정밀도는 절대값이 아니라 "
         "동일 표시량 조건에서의 비교로만 해석한다.")
    return s


def RGB_PAPER():
    from src.utils.slide_kit import PAPER
    return PAPER


def s_label(prs):
    s = new_slide(prs)
    y = slide_head(s, "5.2 연구 방법", "의사 라벨 자동 생성",
                   sub="공개 벤치마크가 없으므로 3D 객체 GT와 CAN에서 ‘이상적 뷰’를 자동 생성한다(약지도).")
    picture(s, f"{FIG}/fig3_view_policy_nocap.png", MARGIN, y, 5.05, 4.20)
    x = MARGIN + 5.45
    tf = textbox(s, x, y + 0.05, 6.65, 0.40)
    para(tf, "뷰 정책 — 역할로 나눈 슬롯", size=14, bold=True, first=True)
    bullets(s, x, y + 0.48, 6.65, [
        ("primary  ", "진행 방향 뷰는 항상 메인 위치에 표시한다. 전진이면 CAM_FRONT, 후진이면 CAM_BACK."),
        ("슬롯 1  ", "안전 기본값 — 기동 적합도 m(c)가 높은 채널은 검출이 0이어도 켠다. 검출 실패로 화면이 꺼지는 것이 가장 위험한 실패 모드다."),
        ("슬롯 2  ", "위험 — 평활한 채널 위험 risk(c)가 임계값을 넘는 채널을 위험 순으로."),
    ], size=12)
    note(s, x, y + 2.68, 6.65, 1.26, "왜 슬롯을 역할로 나누는가",
         "하나의 임계값을 합산 점수에 걸면 기동 항이 슬롯을 독점해 실제 위험이 밀려난다 "
         "(차선변경 중 위험 커버 60.6%). 역할을 분리하자 70.5%로 회복됐다.")
    return s


def s_label_example(prs):
    s = new_slide(prs)
    y = slide_head(s, "5.2 연구 방법", "같은 장면에 라벨을 적용한 결과",
                   sub="우측 차선변경 중 — 진행 방향(전방)에 더해 우전측방과 우후측방이 보조 뷰로 선택된다.")
    picture(s, f"{FIG}/fig4_label_example_nocap.png", MARGIN, y, W - 2 * MARGIN, 4.55)
    tf = textbox(s, MARGIN, H - 0.78, 11.5, 0.30)
    para(tf, "nuScenes scene-0538, boston-seaport · CC BY-NC-SA 4.0   |   "
             "라벨 생성은 오프라인이므로 미래 pose를 참조하지만, 모델 입력에는 넣지 않는다.",
         size=10, color=MUTED, font=MONO, line=1.3, first=True)
    return s


def s_model(prs):
    s = new_slide(prs)
    y = slide_head(s, "5.3 연구 방법", "모델 — 0.46 M 경량 융합",
                   sub="과거 4 keyframe(약 2초)을 입력받는다. 백본 특징을 사전 캐시하므로 학습 시 원본 이미지를 읽지 않는다.")
    stages = [
        ("입력 신호", ["6채널 영상 특징 (576-d)", "CAN 5종 + 결측 마스크", "채널별 위험도 벡터"], RULE),
        ("채널 토큰화 + Transformer", ["채널 6 + CAN 1 = 토큰 7개", "2층 · 4헤드 · d=128", "cross-view 상호 참조"], ACCENT),
        ("채널별 GRU", ["T=4 시간축 처리", "선제적 전환", "flicker 억제"], ACCENT),
        ("출력", ["채널별 강조 확률 (6)", "레이아웃 템플릿 (5-way)", "독립 sigmoid → 임계값 조절"], SIGNAL),
    ]
    x = MARGIN
    bw = 2.84
    for i, (title, items, col) in enumerate(stages):
        sh = box(s, x, y + 0.20, bw, 2.10, fill=WHITE, line=col,
                 line_pt=1.5 if col is not RULE else 1.0)
        tf = sh.text_frame
        para(tf, title, size=12.5, bold=True,
             color=col if col is not RULE else INK, first=True, space_after=9)
        for it in items:
            para(tf, "· " + it, size=10.5, color=MUTED, line=1.35, space_after=5)
        if i < 3:
            ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + bw + 0.07),
                                    Inches(y + 1.10), Inches(0.30), Inches(0.16))
            ar.fill.solid(); ar.fill.fore_color.rgb = MUTED
            ar.line.fill.background(); ar.shadow.inherit = False
        x += bw + 0.44

    note(s, MARGIN, y + 2.62, W - 2 * MARGIN, 1.36, "핵심 설계 — 학습 시 검출 오류를 증강으로 주입",
         "위험도 벡터는 학습 중 매번 누락 · 오검출 · 위치 오차가 섞인 값으로 대체된다. "
         "시드는 (seed, epoch, index)로 두어 epoch마다 다른 손상 실현을 보게 한다 — "
         "인덱스만으로 시드하면 고정된 손상 입력을 암기하게 되어 주장 C3을 입증할 수 없다.\n"
         "반대로 평가 시에는 (seed, index)만 사용해 제안 모델과 B2가 동일한 손상 입력을 보도록 고정한다.",
         fill=SIGNAL_BG, edge=SIGNAL)

    tf = textbox(s, MARGIN, y + 4.16, 11.5, 0.30)
    para(tf, "ablation은 별도 모델을 만들지 않고 동일 코드의 플래그"
             "(use_can · use_risk · use_transformer · use_gru)로 구성한다.",
         size=10.5, color=MUTED, font=MONO, first=True)
    return s


def s_eval(prs):
    s = new_slide(prs)
    y = slide_head(s, "6 평가 설계", "순환 평가를 차단하고, 재현율만 보지 않는다")

    sh = box(s, MARGIN, y, 5.95, 2.30, fill=SIGNAL_BG, line=SIGNAL, line_pt=1.5)
    tf = sh.text_frame
    para(tf, "평가의 최대 함정 — 순환 평가", size=13, bold=True, color=SIGNAL,
         first=True, space_after=9)
    para(tf, "의사 라벨을 만든 규칙을 그대로 추론에 적용한 것이 베이스라인 B2다. "
             "따라서 의사 라벨 시험셋에서 B2는 정의상 만점에 가깝다.", size=11.5,
         line=1.42, space_after=9)
    para(tf, "→ 주 지표는 사람이 검수한 골드셋 300 keyframe으로 산출하고, "
             "의사 라벨 수치는 보조 열로만 병기한다.", size=11.5, bold=True, line=1.42)

    sh = box(s, MARGIN + 6.25, y, 5.85, 2.30, fill=WHITE, line=ACCENT, line_pt=1.5)
    tf = sh.text_frame
    para(tf, "골드셋 300장 — 층화 추출", size=13, bold=True, color=ACCENT,
         first=True, space_after=9)
    para(tf, "위험 100 / 비직진 기동 100 / 무작위 100", size=12.5, bold=True,
         line=1.4, space_after=9)
    para(tf, "자연 분포로 뽑으면 위험 장면이 16건뿐이라(5.3%) 재현율 신뢰구간이 "
             "±20%p를 넘어 안전 관련 주장이 불가능하다. 위험 층 100건이면 ±8%p 수준이 된다.",
         size=11, color=MUTED, line=1.4, space_after=8)
    para(tf, "추출 확률을 기록해 역확률 가중으로 전체 분포 기준 추정치를 복원한다.",
         size=11, bold=True, line=1.4)

    rows = [
        ["지표", "정의", "왜 필요한가"],
        ["평균 보조 뷰 수", "동시에 강조한 채널 수", "인지 부하의 대리 지표"],
        ["위험 사건 포착률", "‘실제 위험’ 객체가 선택 뷰 화각에 든 비율", "GT 기하로 직접 계산 — 순환 없음"],
        ["위험 슬롯 정밀도", "위험으로 켠 채널 중 실제 위험이 있던 비율", "반드시 병기 — 재현율만 보면 튜닝이 왜곡"],
        ["Flicker rate", "1.5초 이내 되돌아간 전환의 비율", "UI 안정성"],
    ]
    table(s, MARGIN, y + 2.56, W - 2 * MARGIN, rows,
          col_w=[0.20, 0.44, 0.36], row_h=0.44, size=11.5)
    return s


def s_prelim(prs):
    s = new_slide(prs)
    y = slide_head(s, "7 예비 결과", "표시량과 안전의 운영 곡선",
                   sub="계획서 작성 시점까지 데이터 확보와 의사 라벨 파이프라인을 완료하고, 라벨 설계의 타당성을 정량 검증했다.")
    picture(s, f"{FIG}/fig5_chart_only.png", MARGIN, y - 0.10, 6.70, 4.40)

    x = MARGIN + 6.95
    rows = [
        ["θ", "평균 표시", "포착률", "정밀도"],
        ["0.06", "0.89", "96.4%", "7.9%"],
        ["0.10", "0.67", "93.3%", "11.5%"],
        ["0.12 ←", "0.60", "89.9%", "13.3%"],
        ["0.20", "0.42", "69.7%", "20.0%"],
        ["현행 1뷰", "1.00", "69.4%", "3.8%"],
    ]
    table(s, x, y, 5.15, rows, col_w=[0.28, 0.24, 0.24, 0.24], row_h=0.40, size=11.5)

    note(s, x, y + 2.52, 5.15, 1.72, "채택 운영점 — 임계값 0.12",
         "표시 0.60장에서 포착률 89.9% · 정밀도 13.3%.\n"
         "현행 고정 1뷰(1.00장 · 69.4% · 3.8%) 대비 "
         "표시량 41%↓ · 포착률 +20.5%p · 정밀도 3.5배로 세 축 모두 우세하다. "
         "더 줄이면 포착률이 급격히 무너진다.",
         fill=SIGNAL_BG, edge=SIGNAL)
    return s


def s_schedule(prs):
    s = new_slide(prs)
    y = slide_head(s, "8 추진 일정", "실험은 10월 둘째 주에 종료한다",
                   sub="◆ 는 연구자가 직접 수행하거나 판단이 필요한 지점이다. 연산은 병목이 아니다 — 전체 실험이 약 2시간이다.")
    weeks = ["9/13–9/19", "9/20–9/26", "9/27–10/3", "10/4–10/10",
             "10/11–10/17", "10/18–10/24", "10/25–10/31"]
    tasks = [
        ("데이터 · 환경 구축", 0, 1, "완료", ACCENT),
        ("의사 라벨 전체 생성 ◆", 1, 1, "완료", ACCENT),
        ("특징 캐시 · 데이터로더", 2, 1, "", MUTED),
        ("골드셋 UI → 라벨링 ◆ 300개", 2, 2, "", SIGNAL),
        ("본 학습 (제안 모델)", 3, 1, "", MUTED),
        ("베이스라인 B1–B4", 3, 1, "", MUTED),
        ("성능표 + Ablation", 4, 1, "", MUTED),
        ("강건성 열화 곡선 ◆ 핵심", 4, 1, "", SIGNAL),
        ("예비 대응 + 정성 결과", 5, 1, "", MUTED),
        ("논문 집필 완료", 6, 1, "", ACCENT),
    ]
    x0, tw = MARGIN + 3.30, (W - 2 * MARGIN - 3.30) / 7
    for i, wk in enumerate(weeks):
        tf = textbox(s, x0 + i * tw, y, tw, 0.28)
        para(tf, wk, size=9.5, color=MUTED, font=MONO, first=True,
             align=PP_ALIGN.CENTER)
    yy = y + 0.36
    for name, start, span, badge, col in tasks:
        tf = textbox(s, MARGIN, yy + 0.04, 3.16, 0.30)
        para(tf, name, size=11.5, first=True)
        bar = box(s, x0 + start * tw + 0.04, yy, span * tw - 0.08, 0.28,
                  fill=col if col is not MUTED else SURF2,
                  line=None)
        if badge:
            tfb = bar.text_frame
            tfb.vertical_anchor = MSO_ANCHOR.MIDDLE
            para(tfb, badge, size=9, bold=True, color=WHITE, font=MONO,
                 first=True, align=PP_ALIGN.CENTER)
        yy += 0.36
    rule_line(s, MARGIN, yy + 0.06, W - 2 * MARGIN)
    tf = textbox(s, MARGIN, yy + 0.22, 11.5, 0.40)
    rich(tf, [("일정의 임계 경로는 연산이 아니라 ", False, MUTED),
              ("사람이 수행하는 검수 · 라벨링 단계와, 강건성 결과에 대한 대응 시간", True, INK),
              ("이다. 10월 셋째 주를 대응 기간으로 확보했다.", False, MUTED)],
         size=11.5, first=True)
    return s


def s_risk(prs):
    s = new_slide(prs)
    y = slide_head(s, "9 · 10", "위험 요인과 기대 효과")

    tf = textbox(s, MARGIN, y, 6.10, 0.30)
    para(tf, "위험 요인과 대응", size=14, bold=True, first=True)
    rows = [
        ["위험", "대응"],
        ["강건성에서 규칙 기반에 우위를 보이지 못함", "① 학습 노이즈 강도를 평가 조건과 정합 ② A2로 영상 특징 기여분 확인 ③ 미달이면 ‘어떤 조건에서 규칙 기반이 무너지는가’를 분석 결과로 전환"],
        ["위험 사건의 희소성 (5.3%)", "골드셋 층화로 신뢰구간 ±8%p 확보. 정밀도는 동일 표시량 조건의 비교로만 해석"],
        ["골드셋 라벨의 주관성", "복수 정답 및 ‘강조 불필요’ 허용 + 검수 UI에 과거 프레임 제시"],
        ["CAN 신호 결손 scene", "공식 blacklist 21개 scene 제외, 결손 시 0 + 마스크 플래그"],
    ]
    table(s, MARGIN, y + 0.38, 6.10, rows, col_w=[0.36, 0.64], row_h=0.92, size=10)

    x = MARGIN + 6.45
    tf = textbox(s, x, y, 5.65, 0.30)
    para(tf, "기대 효과 및 활용 방안", size=14, bold=True, first=True)
    bullets(s, x, y + 0.42, 5.65, [
        ("관제 인지 부하 저감 — ", "6채널을 균등 탐색하는 대신 진행 방향과 제안된 주의 뷰에 집중. 예비 결과 기준 표시량 41% 감소."),
        ("수동 뷰 전환의 대체 — ", "관제사가 30초~1분 주기로 수행하던 카메라 전환을 주행 상황에 맞춰 자동화."),
        ("검출기 요구 성능의 역산 — ", "강건성 곡선은 실제 차량에 탑재할 검출기의 요구 수준을 가늠하는 근거가 된다."),
        ("재현 가능한 실험 자산 — ", "라벨 규칙 · 분할 · 설정 · 평가 스크립트를 공개 가능한 형태로 정리."),
    ], size=11.5)

    note(s, x, y + 3.30, 5.65, 1.10, "검증 범위 밖 — 주장하지 않는다",
         "통신 대역폭 · 전송 품질 절감, 1인 다수 차량 감시(처리량). "
         "전자는 하이라이트 방식이라 전송량을 줄이지 않고, 후자는 원격 지원 모델이다.",
         fill=SIGNAL_BG, edge=SIGNAL)
    return s


def s_close(prs):
    s = new_slide(prs)
    box(s, 0, 0, W, H, fill=INK, line=None)
    tf = textbox(s, MARGIN, 2.30, 11.6, 0.40)
    para(tf, "정리", size=12, color=RULE, font=MONO, first=True)
    tf = textbox(s, MARGIN, 2.72, 11.6, 1.80)
    para(tf, "한정된 자원은 대역폭이 아니라", size=26, bold=True, color=WHITE,
         first=True, line=1.3)
    para(tf, "오퍼레이터의 주의다.", size=26, bold=True, color=WHITE, line=1.3)
    tf = textbox(s, MARGIN, 4.60, 11.0, 1.10)
    para(tf, "6채널 균등 표시는 한정된 주의를 균등 배분하는 것이다. "
             "본 연구는 그 배분을 주행 상황에 따라 재조정하고, "
             "검출이 불완전할 때에도 그 재조정이 무너지지 않는지를 검증한다.",
         size=14, color=RULE, line=1.6, first=True)
    return s


BUILDERS = [s_cover, s_summary, s_industry, s_console, s_human, s_lever,
            s_related, s_limits, s_goals, s_bg_to_form, s_formulation,
            s_label, s_label_example, s_model, s_eval, s_prelim,
            s_schedule, s_risk, s_close]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/slides/연구계획서_발표.pptx")
    ap.add_argument("--copy-to", default=None,
                    help="추가로 복사할 경로 (윈도우 폴더 등)")
    args = ap.parse_args()

    missing = [f for f in ("fig1_console_layout", "fig2_hazard_rarity",
                           "fig3_view_policy", "fig4_label_example",
                           "fig5_chart_only")
               if not os.path.exists(f"{FIG}/{f}.png")
               or not os.path.exists(f"{FIG}/{f.replace('_chart_only','_operating_curve')}_nocap.png")]
    if missing:
        print("그림이 없습니다 — scripts/export_figures.py 를 먼저 실행하세요:",
              ", ".join(missing))
        return 1

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    for i, build in enumerate(BUILDERS, 1):
        s = build(prs)
        if i > 1 and build is not s_close:
            page_num(s, i)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    prs.save(args.out)
    print(f"✅ {args.out}  ({len(BUILDERS)}장, {os.path.getsize(args.out)/1024/1024:.1f} MB)")
    if args.copy_to:
        os.makedirs(args.copy_to, exist_ok=True)
        dst = os.path.join(args.copy_to, os.path.basename(args.out))
        shutil.copy2(args.out, dst)
        print(f"   복사: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
