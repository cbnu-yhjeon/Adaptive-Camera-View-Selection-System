"""INDAI 콜로키움 2026 논문(2쪽) — 투고양식 v1.1의 판형·글꼴을 따른 Word 문서.

양식: A4, 여백 15 mm, 제목·저자·초록은 1단, 본문 2단.
글꼴: 제목 HY견명조 13pt / 저자 HY중고딕 10pt / 초록 8pt / 장 제목 HY중고딕 10pt /
본문 HY신명조 9pt / 표·그림 캡션 HY중고딕 8pt (국·영 두 줄) / 참고문헌 8.5pt.
수치는 results/colloquium/numbers.json에서만 읽는다. 한글(.hwp) 양식에는 이 문서의
문단을 붙여 넣으면 된다.

필요: python-docx (프로젝트 .venv에는 없다 — 별도 환경에서 실행)
사용:
    python scripts/colloquium/make_paper.py
"""

from __future__ import annotations

import json
import os
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "results", "colloquium")
FIG = os.path.join(OUT, "fig")

MJ_TITLE, MJ_BODY, GOTHIC, LATIN = "HY견명조", "HY신명조", "HY중고딕", "Times New Roman"
GRAY = RGBColor(0x55, 0x5B, 0x61)


# ---------------------------------------------------------------------------
def set_font(run, ea=MJ_BODY, latin=LATIN, size=9, bold=False, italic=False, color=None, sub=False, sup=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.subscript = sub
    run.font.superscript = sup
    if color is not None:
        run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    rf = rPr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rPr.insert(0, rf)
    rf.set(qn("w:ascii"), latin)
    rf.set(qn("w:hAnsi"), latin)
    rf.set(qn("w:eastAsia"), ea)
    rf.set(qn("w:cs"), latin)
    lang = rPr.find(qn("w:lang"))
    if lang is None:
        lang = OxmlElement("w:lang")
        rPr.append(lang)
    lang.set(qn("w:eastAsia"), "ko-KR")


def para(doc_or_cell, parts, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=9, ea=MJ_BODY, latin=LATIN,
         before=0, after=0, line=1.18, indent_first=None, left=None, hanging=None, keep_next=False):
    """parts: [str | (text, {bold, italic, sub, sup, ea, latin, size, color})]"""
    p = doc_or_cell.add_paragraph()
    pf = p.paragraph_format
    p.alignment = align
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = line
    if indent_first is not None:
        pf.first_line_indent = Mm(indent_first)
    if left is not None:
        pf.left_indent = Mm(left)
    if hanging is not None:
        pf.first_line_indent = Mm(-hanging)
    pf.keep_with_next = keep_next
    for part in parts:
        text, o = (part, {}) if isinstance(part, str) else part
        r = p.add_run(text)
        set_font(r, ea=o.get("ea", ea), latin=o.get("latin", latin), size=o.get("size", size),
                 bold=o.get("bold", False), italic=o.get("italic", False), color=o.get("color"),
                 sub=o.get("sub", False), sup=o.get("sup", False))
    return p


def heading(doc, text, size=10, before=5, after=2):
    return para(doc, [(text, {"bold": True})], align=WD_ALIGN_PARAGRAPH.LEFT, size=size, ea=GOTHIC,
                latin="Arial", before=before, after=after, keep_next=True)


def sub_heading(doc, text):
    return para(doc, [(text, {"bold": True})], align=WD_ALIGN_PARAGRAPH.LEFT, size=9, ea=GOTHIC,
                latin="Arial", before=4, after=1, keep_next=True)


def body(doc, parts, first=True):
    return para(doc, parts, indent_first=3.0 if first else None)


def caption(doc, ko, en, before=2, after=5, table=False):
    """국·영 두 줄 캡션. 표 캡션(table=True)은 표 위에 오므로 표와 떨어지지 않게 영문 줄도 다음과 함께 둔다."""
    para(doc, [(ko, {"bold": True})], align=WD_ALIGN_PARAGRAPH.LEFT, size=8, ea=GOTHIC, latin="Arial",
         before=before, after=0, line=1.05, keep_next=True)
    para(doc, [en], align=WD_ALIGN_PARAGRAPH.LEFT, size=8, ea=GOTHIC, latin="Arial", before=0, after=after, line=1.05,
         keep_next=table)


def figure(doc, path, width_mm, ko, en):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Mm(width_mm))
    caption(doc, ko, en)


def cols(section, n, space_mm=6):
    sectPr = section._sectPr
    c = sectPr.find(qn("w:cols"))
    if c is None:
        c = OxmlElement("w:cols")
        sectPr.append(c)
    c.set(qn("w:num"), str(n))
    c.set(qn("w:space"), str(int(space_mm / 25.4 * 1440)))


def page_setup(section):
    section.page_width, section.page_height = Mm(210), Mm(297)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Mm(15))
    section.header_distance = Mm(8)
    section.footer_distance = Mm(8)


def cell_shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), hexcolor)
    tcPr.append(sh)


TBLPR_ORDER = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize", "tblW", "jc",
               "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook"]


def fix_table_width(tbl, widths_mm):
    """표 너비를 고정값으로 명시한다. '자동'으로 두면 한글(hwp 변환)이 쪽 전체 폭으로 펼쳐 단을 넘는다.

    tblW(dxa)·tblLayout fixed·gridCol 너비를 넣고, tblPr 자식 순서를 OOXML 규격 순서로 정리한다.
    """
    tw = lambda mm: str(int(round(mm / 25.4 * 1440)))
    tbl.autofit = False
    tblPr = tbl._tbl.tblPr
    w = tblPr.find(qn("w:tblW"))
    if w is None:
        w = OxmlElement("w:tblW")
        tblPr.append(w)
    w.set(qn("w:w"), tw(sum(widths_mm)))
    w.set(qn("w:type"), "dxa")
    if tblPr.find(qn("w:tblLayout")) is None:
        lay = OxmlElement("w:tblLayout")
        lay.set(qn("w:type"), "fixed")
        tblPr.append(lay)
    for gc, mm in zip(tbl._tbl.tblGrid.findall(qn("w:gridCol")), widths_mm):
        gc.set(qn("w:w"), tw(mm))
    for row in tbl.rows:
        for c, mm in zip(row.cells, widths_mm):
            c.width = Mm(mm)
    kids = list(tblPr)
    for k in kids:
        tblPr.remove(k)
    rank = lambda el: TBLPR_ORDER.index(el.tag.split("}")[1]) if el.tag.split("}")[1] in TBLPR_ORDER else 99
    for k in sorted(kids, key=rank):
        tblPr.append(k)


def keep_together(tbl):
    """표가 단·쪽 사이에서 나뉘지 않게 마지막 행을 뺀 모든 행의 문단을 '다음 문단과 함께'로 둔다."""
    for row in tbl.rows[:-1]:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True


def table_borders(tbl, color="808080", sz=4):
    tblPr = tbl._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), str(sz))
        e.set(qn("w:color"), color)
        b.append(e)
    tblPr.append(b)


# ---------------------------------------------------------------------------
def build(N: dict, out: str) -> None:
    L, T1, AB, R = N["label"], N["table1"], N["ablation"], N["robust"]
    rows = {r["key"]: r for r in T1["rows"]}
    m, b2, b2c, b3, b4 = (rows[k] for k in ("model", "b2", "b2c", "b3", "b4"))
    a1 = L["always1"]
    ab = {r["tag"]: r for r in AB["rows"]}
    base = ab["제안"]
    n = L["n_split"]
    vs = b2c["vs_model"]
    vs2 = b2["vs_model"]
    lb = rows["label"]
    drop = {k: R["drop"]["heavy"][k] - R["clean"][k] for k in ("model", "B2c", "B2")}
    c3 = R["claim3"]
    sig_ko = lambda pv: "유의하게 " if pv < 0.05 else ""
    flick = {k: rows[k]["flicker_at_load"] for k in ("b2", "b2c", "b3", "b4", "model")}
    lowest_flicker = min(flick, key=flick.get) == "model"

    doc = Document()
    s0 = doc.sections[0]
    page_setup(s0)
    # 기본 문단 간격 제거
    st = doc.styles["Normal"]
    st.paragraph_format.space_after = Pt(0)
    st.paragraph_format.space_before = Pt(0)

    hp = s0.header.paragraphs[0]
    r = hp.add_run("INDAI 콜로키움 2026")
    set_font(r, ea=GOTHIC, latin="Arial", size=8, color=GRAY)

    C = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, [("원격주행을 위한 딥러닝 기반 적응형 카메라 뷰 선택 시스템", {"bold": True})], align=C, size=13,
         ea=MJ_TITLE, before=6, after=6, line=1.1)
    para(doc, ["전양호, OOO", ("*", {}), ", OOO", ("**", {})], align=C, size=10, ea=GOTHIC, latin="Arial", after=2)
    para(doc, ["충북대학교 산업인공지능학과, *충북대학교 산업인공지능연구센터, **충북대학교 OOO학과"], align=C, size=9,
         ea=GOTHIC, latin="Arial", line=1.1)
    para(doc, ["yangho@chungbuk.ac.kr, *OOO@chungbuk.ac.kr, **OOO@chungbuk.ac.kr"], align=C, size=9, ea=GOTHIC,
         latin="Arial", line=1.1, after=8)
    para(doc, [("A Deep Learning-based Adaptive Camera View Selection System for Teleoperated Driving", {"bold": True})],
         align=C, size=13, ea=MJ_TITLE, latin=LATIN, after=6, line=1.1)
    para(doc, ["Yang Ho Jeon, OOO*, OOO**"], align=C, size=10, ea=GOTHIC, latin=LATIN, after=2)
    para(doc, ["Department of Industrial AI, Chungbuk National Univ., *Industrial AI Research Center, Chungbuk National Univ.,"],
         align=C, size=9, latin=LATIN, line=1.1)
    para(doc, ["**Department of OOO, Chungbuk National Univ."], align=C, size=9, latin=LATIN, line=1.1, after=8)
    para(doc, [("Abstract", {"bold": True})], align=C, size=10.5, ea=GOTHIC, latin=LATIN, after=3)
    abstract = (
        "Teleoperated-driving operators watch six camera feeds in a fixed layout, although hazards that call for attention "
        f"appear in only {L['hazard_pct']:.1f}% of frames. We propose an adaptive camera view selection system that keeps the "
        "travel-direction view always visible and highlights zero to two auxiliary views. Because no public benchmark exists, "
        "we automatically generate pseudo labels from the 3D annotations, CAN signals, and map of nuScenes using closest-approach "
        "time-to-collision, map-based maneuver inference, and a role-separated slot policy; with "
        f"{L['mean_aux']:.2f} views per frame the labels cover {L['recall']:.1f}% of hazards, versus {a1['recall']:.1f}% when one "
        "view is always highlighted. A lightweight fusion model (0.46M parameters) combines image features, CAN signals, and "
        "corrupted per-camera risk through a Transformer and per-channel GRUs, and is trained with detection-error augmentation. "
        f"On the test split at equal display load, the model captures {m['recall_at_load']:.1f}% of hazards, on par with the "
        f"rule-based baselines ({b2['recall_at_load']:.1f}–{b2c['recall_at_load']:.1f}%). On a human-labeled gold set of "
        f"{T1['n_labeled']} frames, the pseudo labels agree with human choices in {100 * lb['top1']:.1f}% (Top-1), and under "
        f"clean input and all four heavy corruptions the model agrees with humans {c3['gold_diff'][0]:.0f}–{c3['gold_diff'][1]:.0f} "
        f"points more often than a rule with the same past-only information (Top-1 {100 * m['top1']:.1f}% vs. "
        f"{100 * b2c['top1']:.1f}% under medium noise, p = {vs['p']:.3f}), while both degrade by at most "
        f"{max(abs(c3['deg_model'][0]), abs(c3['deg_b2c'][0])):.0f} points. Camera failures remain a shared weakness."
    )
    para(doc, [abstract], size=8, latin=LATIN, line=1.12, left=6, after=4)
    p = para(doc, [("Key words : ", {"bold": True}),
                   "Teleoperated driving, Adaptive camera view selection, Pseudo label, Multimodal fusion, Robustness"],
             align=WD_ALIGN_PARAGRAPH.LEFT, size=8, ea=GOTHIC, latin="Arial", left=6, after=6)

    # ---------------- 본문 2단 ----------------
    s1 = doc.add_section(WD_SECTION.CONTINUOUS)
    page_setup(s1)
    cols(s1, 2, 6)

    heading(doc, "Ⅰ. 서 론", before=2)
    # 인용문은 원문 초록에서 확인했다(2026-10-13): Goodall 초록 첫 문장, Tener·Lanir와 Zhao 초록의 해당 구절.
    body(doc, ["현존하는 어떤 자율주행차도 모든 조건과 환경에서 운행하지는 못하며[1], 자율주행차가 모든 도로 상황을 스스로 "
               "해결할 수는 없다[2]. 이 공백을 사람이 관제실에서 메우는 것이 원격 운행이고, 그중 원격 운전은 자율주행 상용화에 "
               "필수적인 요소로 꼽힌다[3]. 원격 운행은 오퍼레이터가 차량을 직접 조종하는 원격 운전과 경로·판단만 돕는 원격 지원으로 "
               "나뉘며[4], 본 논문의 원격주행은 전자를 뜻한다."])
    body(doc, ["원격 운전의 어려움은 대부분 오퍼레이터가 차량과 물리적으로 떨어져 있다는 데서 오고[2], 낮은 상황인식이 핵심 난제로 "
               "꼽힌다[3]. 오퍼레이터는 차량 주변을 카메라 영상으로만 파악하는데, 현행 관제석은 여러 영상을 고정된 배치로 동시에 "
               "띄운다. 동시 감시 화면이 4대에서 16대로 늘면 이상 사건 탐지율이 83%에서 64%로 떨어지고[5], 현장 CCTV 요원은 "
               "화면을 훑기보다 한 화면을 골라 평균 26.9~62.4초마다 카메라를 직접 바꾼다[6]. 주행 상황은 초 단위로 바뀌므로 화면 "
               "선택을 자동화할 필요가 있다."])
    body(doc, [f"문제는 언제 무엇을 강조할지가 자명하지 않다는 것이다. 본 연구의 측정에서 실제 위험은 전체 프레임의 {L['hazard_pct']:.1f}%"
               f"에서만 나타나며, 항상 한 화면을 강조하면 정밀도가 {a1['precision']:.1f}%에 그친다. 기존 적응형 연구는 뷰별 전송 품질[7]"
               "이나 GUI 정보 요소의 수[8]를 조정했을 뿐 어느 카메라를 강조할지를 학습 문제로 다루지 않았고, 이를 위한 공개 "
               "벤치마크도 없다."])
    body(doc, ["본 논문은 진행 방향 화면을 항상 표시한 채 나머지 5개 중 0~2개를 보조 뷰로 강조하는 적응형 뷰 선택 시스템을 제안한다. "
               "기여는 (1) 공개 데이터셋 nuScenes[9]의 3D 정답 박스·CAN·지도로 ‘봐야 할 화면’ 의사 라벨을 자동 생성하는 방법, "
               "(2) 영상·CAN·위험도를 융합하고 검출 오류로 증강 학습한 경량 모델, (3) 같은 표시량 비교와 사람이 판정한 골드셋을 "
               "포함한 오프라인 검증이다. 6개 화면을 모두 유지하고 선택한 화면만 강조하는 하이라이트 방식을 전제로 한다."])

    heading(doc, "Ⅱ. 본 론")
    sub_heading(doc, "2.1 의사 라벨 생성")
    body(doc, ["각 keyframe(2 Hz)에서 차량·보행자 정답 박스를 카메라에 투영해 보이는 채널에 배정하고, 객체마다 최접근 시각 t*와 "
               "최접근 거리 DCA로 정의한 TTC를 구해 위험 가중치를 식 (1)로 계산한다."])
    eq = para(doc, ["\t", ("w", {"italic": True}), " = ", ("ρ", {"italic": True}), ("attr", {"sub": True}), " · ",
                    ("ρ", {"italic": True}), ("path", {"sub": True}), "(DCA) · max(exp(−TTC/", ("τ", {"italic": True}), "), ",
                    ("w", {"italic": True}), ("near", {"sub": True}), ")\t(1)"],
              align=WD_ALIGN_PARAGRAPH.LEFT, before=2, after=2)
    tabs = eq.paragraph_format.tab_stops
    tabs.add_tab_stop(Mm(43.5), WD_TAB_ALIGNMENT.CENTER)
    tabs.add_tab_stop(Mm(87), WD_TAB_ALIGNMENT.RIGHT)
    body(doc, ["반경 방향 TTC는 옆으로 스쳐 갈 주차 차량을 충돌 임박으로 오판해 위험량의 62.9%가 정지·주차 객체에서 나왔으므로 "
               "최접근 기하를 쓴다. 채널 위험도 risk(c)는 w의 합을 지수이동평균으로 평활한 값이고, 기동(직진, 회전, 차선변경, 후진, "
               "정지)은 지도의 차로·교차로 정보와 [−1, +5]초 궤적으로 판정한다. 보조 뷰는 역할을 나눈 슬롯으로 고른다. "
               "슬롯 1은 기동이 만드는 사각지대(예: 차선변경 시 후측방)로 검출과 무관하게 켜고, 나머지는 risk(c) ≥ 0.12인 채널을 "
               "위험 순으로 채운다."])
    body(doc, ["실제 위험은 TTC < 5 s이면서 DCA < 3 m인 객체로 정의하며, 5초는 원격주행의 지연과 인지·조작 시간을 고려한 값이다[10]. "
               f"{L['n_keyframes']:,}장의 라벨은 평균 {L['mean_aux']:.2f}장을 켜서 실제 위험의 {L['recall']:.1f}%를 화면에 담는다"
               f"(정밀도 {L['precision']:.1f}%). 라벨은 사람 검수 2회(무작위 20장, 표적 16장)를 통과했고, 라벨 판정에 쓰지 않은 "
               f"방향지시등과의 같은 방향 일치율은 좌회전 {L['signal']['TURN_LEFT']:.1f}%, 우회전 {L['signal']['TURN_RIGHT']:.1f}%였다. "
               f"사람이 판정한 골드셋과는 Top-1 {100 * lb['top1']:.1f}%, Top-2 {100 * lb['top2']:.1f}% 일치했다."])

    sub_heading(doc, "2.2 융합 모델과 학습")
    body(doc, ["모델은 과거 4 keyframe(약 2초)만 본다. 화면별 MobileNetV3-small[11] 특징(576차원)에 카메라 위치 임베딩과 위험도를 "
               "더해 채널 토큰 6개를, CAN 신호(속도, 조향, 가속도, 회전율, 방향지시등)로 토큰 1개를 만들고, 2층 Transformer로 화면 간 "
               "정보를 섞은 뒤 채널별 GRU로 시간축을 처리한다(그림 1). 출력은 화면별 독립 시그모이드로 0~2장을 표현하며, 손실은 "
               "포컬 BCE, soft label KL, 레이아웃 CE, 시간 일관성 항의 합이다. 실제 검출은 불완전하므로 모델 입력의 위험도에는 "
               "누락, 오검출, 위치 오차, 카메라 고장을 주입하고, 학습에서도 에폭마다 다른 오류를 주입한다."])

    figure(doc, os.path.join(FIG, "pipeline_paper.png"), 62, "그림 1 제안 시스템의 구조", "Fig. 1 Overview of the proposed system")

    sub_heading(doc, "2.3 실험 구성")
    body(doc, [f"데이터는 장면 단위로 나눠 학습 595개, 검증 128개, 시험 127개 장면(각 {n['train']:,}장, {n['val']:,}장, {n['test']:,}장)을 썼다. 비교 대상은 "
               "보조 뷰를 쓰지 않는 B1, 라벨 규칙을 손상 입력에 적용한 B2(미래 기동을 아는 상한)와 B2c(과거 궤적만 사용), 영상만 쓰는 "
               "B3, 단일 프레임 B4이다. 규칙을 그대로 쓰면 의사 라벨 기준으로는 정의상 고득점이 나오므로, 정답 박스 기하로 직접 계산한 "
               f"위험 포착률을 같은 표시량({T1['load']:.2f}장)에서 비교하고, 시험 분할에서 층화 추출한 골드셋 300장을 연구자가 직접 "
               f"판정한다(그림 2). 판정을 마친 {T1['n_labeled']}장 중 강조할 화면이 있다고 판정한 장면은 {m['n_nonempty']}장이다."])
    S = N["setup"]
    caption(doc, "표 1 실험 환경", "Table 1 Experimental setup", before=3, after=2, table=True)
    srows = [["항목", "내용"],
             ["데이터", f"nuScenes v1.0-trainval, {L['n_keyframes']:,} keyframe, 장면 단위 분할 595/128/127"],
             ["입력", f"과거 {S['seq_len']} keyframe: 영상 특징 6×576, CAN 8, 화면별 위험도 6"],
             ["모델", f"Transformer {S['tf_layers']}층(d={S['d']}) + 채널별 GRU, {S['params_m']:.2f}M 파라미터"],
             ["학습", f"AdamW {S['lr']:g}, 배치 {S['batch']}, 최대 {S['epochs']} 에폭(조기 종료 {S['patience']}), "
                      f"검출 오류 증강 {'·'.join(S['noise_mix'])}"],
             ["환경", f"{S['gpu'].replace('NVIDIA GeForce ', '')} 16GB, PyTorch {S['torch']}, seed {S['seed']}"]]
    st_ = doc.add_table(rows=len(srows), cols=2)
    st_.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_borders(st_)
    for i, row in enumerate(srows):
        for j, val in enumerate(row):
            cell = st_.cell(i, j)
            cell.width = Mm([14, 70][j])
            cell.paragraphs[0]._p.getparent().remove(cell.paragraphs[0]._p)
            para(cell, [(val, {"bold": i == 0})], align=WD_ALIGN_PARAGRAPH.LEFT if j else WD_ALIGN_PARAGRAPH.CENTER,
                 size=7.5, ea=MJ_BODY if i else GOTHIC, latin=LATIN if i else "Arial", line=1.0)
            if i == 0:
                cell_shade(cell, "E7E9EB")
    keep_together(st_)
    fix_table_width(st_, [14, 70])
    para(doc, [""], size=4)
    figure(doc, os.path.join(FIG, "ui_goldset.png"), 84, "그림 2 골드셋 검수 화면 (실제 판정 장면)",
           "Fig. 2 Gold-set annotation interface")

    sub_heading(doc, "2.4 실험 결과")
    caption(doc, "표 2 시험 분할 비교 (입력 noisy-medium)", "Table 2 Comparison on the test split", before=3, after=2,
            table=True)
    trows = [["방법", "골드 Top-1", "포착률*", "Flicker*"]]
    for key, lab in (("b1", "B1 고정"), ("b2", "B2 규칙(미래 기동)"), ("b2c", "B2c 규칙(과거만)"), ("b3", "B3 영상만"),
                     ("b4", "B4 단일 프레임"), ("model", "제안 모델")):
        r = rows[key]
        trows.append([lab, "—" if r["top1"] is None else f"{100 * r['top1']:.1f}%",
                      "—" if r["recall_at_load"] is None else f"{r['recall_at_load']:.1f}%",
                      "—" if r["flicker_at_load"] is None else f"{r['flicker_at_load']:.2f}"])
    tbl = doc.add_table(rows=len(trows), cols=4)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_borders(tbl)
    widths = [31, 18, 18, 17]
    for i, row in enumerate(trows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.width = Mm(widths[j])
            cell.paragraphs[0]._p.getparent().remove(cell.paragraphs[0]._p)
            hl = trows[i][0] == "제안 모델"
            para(cell, [(val, {"bold": i == 0 or hl})], align=WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER,
                 size=8, ea=MJ_BODY if i else GOTHIC, latin=LATIN if i else "Arial", line=1.0)
            if i == 0:
                cell_shade(cell, "E7E9EB")
    keep_together(tbl)
    fix_table_width(tbl, widths)
    para(doc, [f"* 같은 표시량({T1['load']:.2f}장)에서 보간한 값. 골드 Top-1은 정답이 있는 판정 {m['n_nonempty']}장 기준"],
         align=WD_ALIGN_PARAGRAPH.LEFT, size=7.5, ea=GOTHIC, latin="Arial", before=1, after=4, line=1.05)
    body(doc, [f"제안 모델은 같은 표시량에서 위험의 {m['recall_at_load']:.1f}%를 포착해 규칙({b2['recall_at_load']:.1f}~"
               f"{b2c['recall_at_load']:.1f}%)과 대등했고, 1.5초 안에 강조가 다시 바뀌는 비율(Flicker)은 {m['flicker_at_load']:.2f}"
               + ("로 가장 낮았다(표 2). " if lowest_flicker else "로 규칙과 비슷했다(표 2). ")
               + f"골드셋 Top-1은 {100 * m['top1']:.1f}%로 같은 정보 조건의 B2c({100 * b2c['top1']:.1f}%)보다 {sig_ko(vs['p'])}높았고"
               f"(짝비교 {vs['model_only']}:{vs['other_only']}, McNemar p = {vs['p']:.3f}), 미래 기동을 아는 B2({100 * b2['top1']:.1f}%)와는 "
               + ("유의한 차이가 없었다. " if vs2["p"] >= 0.05 else f"차이가 유의했다(p = {vs2['p']:.3f}). ")
               + f"영상만 쓰는 B3의 포착률은 {b3['recall_at_load']:.1f}%로 크게 낮았고, ablation에서도 위험도 입력을 "
               f"빼면 포착률이 {base['recall_noisy'] - ab['A2']['recall_noisy']:.1f}%p, GRU를 빼면 {base['recall_noisy'] - ab['A3']['recall_noisy']:.1f}%p "
               f"낮아지고 Flicker가 {ab['A3']['flicker_noisy']:.2f}로 커져 위험도와 시간 모델이 위험 포착과 안정성의 핵심임을 확인했다."])
    body(doc, [f"누락, 오검출, 위치 오차, 카메라 고장을 heavy 강도로 주입한 강건성 실험에서, clean과 네 손상 조건 모두 제안 모델의 "
               f"골드셋 Top-1이 B2c보다 {c3['gold_diff'][0]:.1f}~{c3['gold_diff'][1]:.1f}%p 높았고(그림 3) 위험 포착률 차이는 "
               f"{c3['recall_diff'][0]:+.1f}~{c3['recall_diff'][1]:+.1f}%p로 대등했다. 손상에 따른 Top-1 하락 폭은 모델 "
               f"{c3['deg_model'][0]:+.1f}~{c3['deg_model'][1]:+.1f}%p, B2c {c3['deg_b2c'][0]:+.1f}~{c3['deg_b2c'][1]:+.1f}%p로 "
               "차이가 없어, 학습 모델의 이점은 열화 기울기가 아니라 과거 정보만으로 진행 의도를 읽는 데서 오는 수준 차이로 해석된다. "
               f"위험도 입력을 뺀 A2의 Top-1이 {100 * c3['a2_gold_clean']:.1f}%로 가장 높았는데, 사람은 위험 객체보다 진행 의도를 "
               f"기준으로 화면을 고른다는 점을 시사한다. 카메라 3대 고장에는 모든 방법의 포착률이 {abs(max(drop.values())):.0f}%p 이상 떨어졌다."])

    figure(doc, os.path.join(FIG, "robust_paper.png"), 80,
           f"그림 3 손상 조건별 골드셋 Top-1 (heavy 강도, 정답 화면이 있는 {R['gold_n']}장)",
           "Fig. 3 Gold-set Top-1 under heavy input corruptions")
    sub_heading(doc, "2.5 해석 및 한계")
    mv = L["maneuver"]
    lc = mv.get("LANE_CHANGE_LEFT", 0) + mv.get("LANE_CHANGE_RIGHT", 0)
    body(doc, [f"연구자는 판정한 {T1['n_labeled']}장 중 {lb['n_none']}장을 ‘강조 불필요’로 보았는데, 의사 라벨은 그중 "
               f"{100 * (1 - lb['quiet_u']):.0f}%에서 화면을 켰다. 위험 객체의 상당수가 상시 표시되는 전방 화면에 있어 측방 강조가 "
               "필요 없다고 본 것으로, TTC·DCA 기반 위험 정의가 사람의 주의 기준보다 넓다. 차선변경은 학습 데이터의 "
               f"{100 * lc / L['n_keyframes']:.1f}%({lc}프레임)에 불과해 모델이 충분히 배우지 못했고, 후진은 전체 "
               f"{mv.get('REVERSE', 0)}프레임이 모두 학습 분할에 있어 평가하지 못했다. 모든 결과는 오프라인 검증이며 실제 "
               "오퍼레이터의 주의 배분 효과는 측정하지 않았다."])

    heading(doc, "Ⅲ. 결 론")
    body(doc, [f"본 논문은 진행 방향을 유지한 채 보조 뷰 0~2장을 고르는 적응형 뷰 선택 시스템과 공개 데이터 기반 의사 라벨 생성 방법을 "
               f"제안했다. 의사 라벨은 항상 1장을 강조하는 방식보다 {100 * (1 - L['mean_aux']):.0f}% 적게 켜면서 위험을 더 많이 담았고, "
               "학습 모델은 손상 입력 전반에서 규칙과 대등한 위험 포착률을 내면서 과거 정보만 쓰는 규칙보다 사람 판정과 일관되게 "
               "더 잘 일치했다. 향후 첫째, 사람이 실제로 주의를 두는 기준(진행 의도, 잠재 위험)을 라벨에 반영하고, 둘째, 카메라 고장 "
               "같은 채널 결손에 대응하는 학습을 보강하며, 셋째, 실제 오퍼레이터 평가로 주의 배분 효과를 검증할 계획이다."])

    para(doc, [("ACKNOWLEDGMENT", {"bold": True})], align=C, size=10.5, ea=GOTHIC, latin="Arial", before=7, after=2, keep_next=True)
    para(doc, ["이 논문은 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 지역지능화혁신인재양성 "
               "사업임 (IITP-2026-RS-2020-II201462)."], align=WD_ALIGN_PARAGRAPH.LEFT)

    para(doc, [("참 고 문 헌", {"bold": True})], align=C, size=10, ea=GOTHIC, latin="Arial", before=7, after=2, keep_next=True)
    refs = [   # 본문에서 처음 인용된 순서
        "N. Goodall, “Non-Technological Challenges for the Remote Operation of Automated Vehicles,” Transportation Research "
        "Part A: Policy and Practice, vol. 142, pp. 14–26, 2020.",
        "F. Tener and J. Lanir, “Driving from a Distance: Challenges and Guidelines for Autonomous Vehicle Teleoperation "
        "Interfaces,” Proc. CHI ’22, 2022.",
        "L. Zhao et al., “Remote Driving of Road Vehicles: A Survey of Driving Feedback, Latency, Support Control, and Real "
        "Applications,” IEEE Trans. Intelligent Vehicles, vol. 9, no. 10, pp. 6086–6107, 2024.",
        "D. Majstorović, S. Hoffmann, F. Pfab, A. Schimpe, M.-M. Wolf, and F. Diermeyer, “Survey on Teleoperation Concepts "
        "for Automated Vehicles,” Proc. IEEE SMC, pp. 1290–1296, 2022.",
        "A. H. Tickner and E. C. Poulton, “Monitoring up to 16 Synthetic Television Pictures Showing a Great Deal of "
        "Movement,” Ergonomics, vol. 16, no. 4, pp. 381–401, 1973.",
        "M. J. Stainer, K. C. Scott-Brown, and B. W. Tatler, “Looking for Trouble: A Description of Oculomotor Search "
        "Strategies During Live CCTV Operation,” Frontiers in Human Neuroscience, vol. 7, 615, 2013.",
        "M. Hofbauer, C. B. Kuhn, M. Khlifi, G. Petrovic, and E. Steinbach, “Traffic-Aware Multi-View Video Stream Adaptation "
        "for Teleoperated Driving,” Proc. IEEE VTC2022-Spring, 2022.",
        "M.-M. Wolf, H. Schmidt, M. Christl, J. Fank, and F. Diermeyer, “A User-Centered Teleoperation GUI for Automated "
        "Vehicles,” Multimodal Technologies and Interaction, vol. 9, no. 8, 78, 2025.",
        "H. Caesar et al., “nuScenes: A Multimodal Dataset for Autonomous Driving,” Proc. CVPR, 2020.",
        "S. Neumeier et al., “Teleoperation: The Holy Grail to Solve Problems of Automated Driving? Sure, but Latency "
        "Matters,” Proc. AutomotiveUI ’19, 2019.",
        "A. Howard et al., “Searching for MobileNetV3,” Proc. ICCV, 2019.",
    ]
    for i, ref in enumerate(refs, 1):
        para(doc, [f"[{i}] {ref}"], align=WD_ALIGN_PARAGRAPH.LEFT, size=8, latin=LATIN, left=5, hanging=5, line=1.08, after=1)
    # 마지막 쪽의 두 단 높이를 맞춘다 — 연속 구역 나누기 앞에서 Word가 단을 균형 배치한다
    s_end = doc.add_section(WD_SECTION.CONTINUOUS)
    page_setup(s_end)
    cols(s_end, 1)
    doc.save(out)


def main() -> int:
    with open(os.path.join(OUT, "numbers.json"), encoding="utf-8") as f:
        N = json.load(f)
    out = os.path.join(OUT, "paper_INDAI2026.docx")
    build(N, out)
    print("✅", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
