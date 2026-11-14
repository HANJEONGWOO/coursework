#!/usr/bin/env python3
"""Build the minimally edited deck and bilingual notes from scripts.json.

Requires: pymupdf, reportlab, python-docx, pillow.
Run: python build_materials.py
The original PDF is read only. All paths are relative to this file.
"""
from pathlib import Path
import hashlib
import json
import re
from xml.sax.saxutils import escape

import pymupdf as fitz
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
    KeepTogether,
)
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


SUPPORT = Path(__file__).resolve().parent
ROOT = SUPPORT.parent
PREVIEW = SUPPORT / '미리보기'
DATA = json.loads((SUPPORT / 'scripts.json').read_text(encoding='utf-8'))
ORIGINAL = ROOT / '2025-isca-oaken.pdf'
DECK = ROOT / 'Oaken_발표_Jeongwoo_Han.pdf'
NOTES_PDF = ROOT / 'Oaken_발표대본_EN_KO.pdf'
NOTES_DOCX = ROOT / 'Oaken_발표대본_EN_KO.docx'
NOTES_TXT = ROOT / 'Oaken_발표대본_EN_KO.txt'
BLUE = colors.HexColor('#203B64')
GREEN = colors.HexColor('#577D63')
GRAY = colors.HexColor('#526171')
PALE = colors.HexColor('#F0F4F8')
FONT = SUPPORT / 'NanumGothic-Regular.ttf'
FONT_BOLD = SUPPORT / 'NanumGothic-Bold.ttf'
EN_FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
EN_BOLD = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
SECONDS = sum(s['seconds'] for s in DATA['slides'])
WORDS = sum(len(re.findall(r"\b[\w]+(?:[-’'][\w]+)*\b", s['en'])) for s in DATA['slides'])

GUIDE = [
    ('사용 방법', '각 슬라이드의 영문 대본 바로 아래에 한국어 번역을 넣었습니다. 영어 발표에서는 영문 대본을 그대로 읽어도 됩니다. 한국어 번역은 이해를 돕는 자료이며, 회색 발표자 메모는 준비와 질의응답을 위한 참고 내용입니다.'),
    ('시간 계획', f'10분 내외 발표용으로 줄인 대본입니다. 영문 {WORDS:,}단어이며, 분당 110~120단어로 읽으면 대본 약 9분에 슬라이드 전환과 도표 설명을 더해 약 10분입니다. 슬라이드별 권장 시간에는 전환 여유가 포함되며 합계는 {SECONDS // 60}분 {SECONDS % 60}초입니다. 메모를 읽는 시간과 질의응답은 제외합니다.'),
    ('구성', '1~6장 문제와 배경 → 7~12장 알고리즘 → 13~14장 하드웨어 → 15~18장 평가 → 19장 결론과 발표자 의견. 원본 19장 순서를 유지했습니다.'),
    ('수정 범위', '표지의 원저자 목록, 소속과 로고를 정리하고 Presented by Jeongwoo Han, 과목명, 간단한 출처를 넣었습니다. 2~19장은 원본을 유지했습니다. 원본 PDF는 보존했습니다.'),
    ('수치 확인', '12장의 평균 4.4비트 표기와 16장의 1.79배 대상 구성은 논문 본문과 차이가 있어 해당 대본과 메모에 설명했습니다. 17장은 요약 평균보다 실제 표의 개별 수치를 중심으로 설명합니다.'),
    ('평가 범위', 'Oaken 성능은 하드웨어 시뮬레이션, 면적은 RTL 합성 결과입니다. GPU 기준은 A100 실행 결과입니다. 이를 구현한 Oaken 칩의 실측이나 모바일 NPU에서의 검증 결과로 소개하지 않습니다.'),
]


def stamp(seconds):
    return f'{seconds // 60:02d}:{seconds % 60:02d}'


def build_deck():
    before = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
    doc = fitz.open(ORIGINAL)
    p = doc[0]
    # The original title banner ends at y=198. The original author/affiliation
    # content begins below it. True redaction removes that content, not just its
    # appearance under a white rectangle.
    p.add_redact_annot(fitz.Rect(0, 200, 960, 540), fill=(1, 1, 1))
    p.apply_redactions(images=2, graphics=2, text=0)
    # Redaction also removes the full-page white background path. Restore it
    # beneath retained content to preserve the original banner's compositing.
    p.draw_rect(p.rect, color=None, fill=(1, 1, 1), overlay=False)
    p.insert_font(fontname='PresenterSans', fontfile=str(EN_FONT))
    p.insert_font(fontname='PresenterBold', fontfile=str(EN_BOLD))
    p.insert_font(fontname='PresenterKorean', fontfile=str(FONT))
    navy = (0.125, 0.23, 0.39)
    muted = (0.37, 0.42, 0.47)
    p.insert_text((34, 249), 'PAPER PRESENTATION  |  ISCA 2025',
                  fontname='PresenterSans', fontsize=16, color=navy)
    p.insert_text((34, 306), 'Presented by', fontname='PresenterSans',
                  fontsize=20, color=muted)
    p.insert_text((31, 357), DATA['presenter'], fontname='PresenterBold',
                  fontsize=37, color=navy)
    p.insert_text((34, 403), DATA['course'], fontname='PresenterKorean',
                  fontsize=20, color=muted)
    p.draw_line((34, 466), (926, 466), color=(0.80, 0.84, 0.87), width=0.8)
    p.insert_text((34, 493), 'Paper: Minsu Kim et al., ISCA 2025',
                  fontname='PresenterSans', fontsize=13, color=muted)
    p.insert_text((34, 515), "Based on the authors' original presentation.",
                  fontname='PresenterSans', fontsize=12, color=muted)
    p.insert_link({'kind': fitz.LINK_URI, 'from': fitz.Rect(32, 477, 440, 499),
                   'uri': DATA['paper_url']})
    p.insert_link({'kind': fitz.LINK_URI, 'from': fitz.Rect(32, 501, 450, 520),
                   'uri': DATA['slides_url']})
    doc.set_metadata({
        'title': 'Oaken — Paper Presentation',
        'author': 'Jeongwoo Han (presenter); original slides by Minsu Kim et al.',
        'subject': 'ISCA 2025 paper presentation; title page adapted; slides 2–19 unchanged',
        'keywords': 'Oaken, KV cache, ISCA 2025, bilingual speaker notes',
        'creator': 'Prepared for Jeongwoo Han',
    })
    doc.subset_fonts()
    doc.save(DECK, garbage=4, deflate=True)
    doc.close()
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == before
    with fitz.open(DECK) as deck:
        for i, p in enumerate(deck):
            p.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False).save(
                PREVIEW / f'presentation_{i + 1:02}.png')


def paragraph(text, style):
    return Paragraph(escape(text).replace('\n', '<br/>'), style)


def build_pdf():
    pdfmetrics.registerFont(TTFont('KR', str(FONT)))
    pdfmetrics.registerFont(TTFont('KR-Bold', str(FONT_BOLD)))
    pdfmetrics.registerFont(TTFont('EN', str(EN_FONT)))
    pdfmetrics.registerFont(TTFont('EN-Bold', str(EN_BOLD)))
    styles = {
        'title': ParagraphStyle('title', fontName='EN-Bold', fontSize=20,
                                leading=26, textColor=BLUE, spaceAfter=8),
        'sub': ParagraphStyle('sub', fontName='KR', fontSize=10,
                              leading=15, textColor=GRAY, spaceAfter=8, wordWrap='CJK'),
        'label': ParagraphStyle('label', fontName='EN-Bold', fontSize=10,
                                leading=14, textColor=GREEN, spaceBefore=10, spaceAfter=5),
        'en': ParagraphStyle('en', fontName='EN', fontSize=10.6,
                             leading=15.6, spaceAfter=5),
        'ko': ParagraphStyle('ko', fontName='KR', fontSize=10.5,
                             leading=16, spaceAfter=6, wordWrap='CJK'),
        'note': ParagraphStyle('note', fontName='KR', fontSize=8.6,
                               leading=12.8, textColor=GRAY, wordWrap='CJK'),
        'guide_label': ParagraphStyle('guide_label', fontName='KR-Bold',
                                      fontSize=11, leading=16, textColor=BLUE,
                                      spaceBefore=9, spaceAfter=4),
        'source': ParagraphStyle('source', fontName='EN', fontSize=8,
                                 leading=12, textColor=GRAY, wordWrap='CJK'),
    }

    def footer(canvas, document):
        canvas.saveState()
        w, h = A4
        canvas.setStrokeColor(colors.HexColor('#D5DDE5'))
        canvas.line(42, 36, w - 42, 36)
        canvas.setFont('EN', 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(42, 23, 'OAKEN  /  Jeongwoo Han  /  EN + KO')
        canvas.drawRightString(w - 42, 23, str(document.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(NOTES_PDF), pagesize=A4,
                            leftMargin=42, rightMargin=42,
                            topMargin=38, bottomMargin=48,
                            title='Oaken — 10-Minute English and Korean Speaker Notes',
                            author='Jeongwoo Han', pageCompression=1)
    story = [paragraph('Oaken | 10-Minute Script', styles['title']),
             paragraph('Jeongwoo Han · 컴퓨터구조설계특론 · ISCA 2025', styles['sub']),
             Spacer(1, 6)]
    for heading, body in GUIDE:
        story.extend([paragraph(heading, styles['guide_label']),
                      paragraph(body, styles['ko'])])
    story.append(paragraph('출처', styles['guide_label']))
    story.append(paragraph('Minsu Kim et al. Oaken: Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization. ISCA 2025. DOI: 10.1145/3695053.3731019.', styles['source']))
    for label, url in [('Paper', DATA['paper_url']), ('Original slides', DATA['slides_url'])]:
        story.append(Paragraph(f'<link href="{escape(url)}" color="#203B64">{label}: {escape(url)}</link>', styles['source']))
    story.append(paragraph('슬라이드의 내용과 도표를 바탕으로 새로 작성한 발표 대본입니다. 논문은 수치·알고리즘·평가 조건의 교차 확인에 사용했습니다. 출처 확인: 2026-11-14.', styles['note']))
    elapsed = 0
    for s in DATA['slides']:
        start = elapsed
        elapsed += s['seconds']
        story.append(PageBreak())
        story.append(paragraph(f"{s['number']:02d} / 19  ·  {s['title']}", styles['title']))
        image = Image(str(PREVIEW / f"presentation_{s['number']:02}.png"),
                      width=232, height=130.5)
        side = [paragraph(f"권장 {s['seconds']}초\n누적 {stamp(start)}–{stamp(elapsed)}", styles['sub']),
                Spacer(1, 8), paragraph('설명 순서', styles['guide_label']),
                paragraph(s['cue'], styles['ko'])]
        block = Table([[image, side]], colWidths=[246, 265])
        block.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(block)
        story.append(paragraph('ENGLISH SCRIPT', styles['label']))
        story.append(paragraph(s['en'], styles['en']))
        story.append(paragraph('KOREAN TRANSLATION', styles['label']))
        story.append(paragraph(s['ko'], styles['ko']))
        story.append(Spacer(1, 9))
        note = Table([[paragraph('발표자 메모 · 준비 및 질의응답 참고\n' + s['note'], styles['note'])]], colWidths=[511])
        note.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), PALE),
            ('BOX', (0, 0), (-1, -1), 0.4, colors.HexColor('#DDE5ED')),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ]))
        story.append(note)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def set_word_font(style, name='NanumGothic', size=10.5, bold=False):
    style.font.name = name
    style.font.size = Pt(size)
    style.font.bold = bold
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.rFonts
    if fonts is None:
        fonts = OxmlElement('w:rFonts')
        rpr.insert(0, fonts)
    fonts.set(qn('w:eastAsia'), 'NanumGothic')


def build_docx():
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.2677)
    section.page_height = Inches(11.6929)
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.6)
    section.right_margin = Inches(0.6)
    for style in ['Normal', 'Body Text']:
        set_word_font(doc.styles[style])
        doc.styles[style].paragraph_format.space_after = Pt(7)
        doc.styles[style].paragraph_format.line_spacing = 1.12
    set_word_font(doc.styles['Title'], size=25, bold=True)
    set_word_font(doc.styles['Heading 1'], size=17, bold=True)
    set_word_font(doc.styles['Heading 2'], size=10.5, bold=True)
    for style in ['Title', 'Heading 1', 'Heading 2']:
        doc.styles[style].font.color.rgb = RGBColor.from_string('203B64')
    doc.core_properties.title = 'Oaken — 10-Minute English and Korean Speaker Notes'
    doc.core_properties.author = 'Jeongwoo Han'
    doc.core_properties.subject = '컴퓨터구조설계특론 발표 대본'
    doc.add_heading('Oaken | 10-Minute Script', 0)
    doc.add_paragraph('Jeongwoo Han · 컴퓨터구조설계특론 · ISCA 2025')
    for heading, body in GUIDE:
        doc.add_heading(heading, 2)
        doc.add_paragraph(body)
    doc.add_heading('출처', 2)
    p = doc.add_paragraph('Minsu Kim et al., ISCA 2025. DOI: 10.1145/3695053.3731019\n'
                          + DATA['paper_url'] + '\n' + DATA['slides_url'])
    for r in p.runs:
        r.font.size = Pt(8)
    elapsed = 0
    for s in DATA['slides']:
        doc.add_page_break()
        doc.add_heading(f"{s['number']:02d} / 19 · {s['title']}", 1)
        start = elapsed
        elapsed += s['seconds']
        p = doc.add_paragraph(f"권장 {s['seconds']}초 | 누적 {stamp(start)}–{stamp(elapsed)} | {s['cue']}")
        for r in p.runs:
            r.font.size = Pt(9)
        doc.add_picture(str(PREVIEW / f"presentation_{s['number']:02}.png"), width=Inches(3.0))
        doc.add_heading('ENGLISH SCRIPT', 2)
        doc.add_paragraph(s['en'])
        doc.add_heading('KOREAN TRANSLATION', 2)
        doc.add_paragraph(s['ko'])
        doc.add_heading('발표자 메모 · 준비 및 질의응답 참고', 2)
        p = doc.add_paragraph(s['note'])
        for r in p.runs:
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor.from_string('526171')
    footer = section.footer.paragraphs[0]
    footer.text = 'OAKEN / Jeongwoo Han / EN + KO                                      '
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    for run in footer.runs:
        run.font.size = Pt(8)
    doc.save(NOTES_DOCX)


def build_txt():
    lines = ['Oaken — 10-Minute English / Korean Speaker Notes',
             'Presenter: Jeongwoo Han', 'Course: 컴퓨터구조설계특론', '']
    for heading, body in GUIDE:
        lines.extend([f'[{heading}]', body, ''])
    lines.extend(['[출처]', 'Minsu Kim et al., ISCA 2025. DOI: 10.1145/3695053.3731019',
                  DATA['paper_url'], DATA['slides_url'], ''])
    elapsed = 0
    for s in DATA['slides']:
        start = elapsed
        elapsed += s['seconds']
        lines.extend(['=' * 72, f"SLIDE {s['number']:02d} / 19 — {s['title']}",
                      f"권장 {s['seconds']}초 | 누적 {stamp(start)}–{stamp(elapsed)}",
                      '설명 순서: ' + s['cue'], '', '[ENGLISH SCRIPT]', s['en'], '',
                      '[KOREAN TRANSLATION]', s['ko'], '', '[발표자 메모 · 준비 및 질의응답 참고]',
                      s['note'], ''])
    NOTES_TXT.write_text('\n'.join(lines), encoding='utf-8')


def validate():
    report = []
    with fitz.open(ORIGINAL) as original, fitz.open(DECK) as deck:
        assert len(original) == len(deck) == 19
        report.append('PASS: presentation has 19 slides in original order.')
        title_text = deck[0].get_text()
        assert DATA['presenter'] in title_text
        assert DATA['course'] in title_text
        assert 'KAIST' not in title_text and 'HyperAccel' not in title_text
        assert 'Minsu Kim et al.' in title_text
        report.append('PASS: title page has presenter/course/compact credit; affiliations removed.')
        # Preserve the original banner exactly and every subsequent page visually.
        banner = fitz.Rect(0, 0, 960, 198)
        assert original[0].get_pixmap(clip=banner).samples == deck[0].get_pixmap(clip=banner).samples
        for i in range(1, 19):
            a = original[i].get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
            b = deck[i].get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
            assert a.samples == b.samples, f'Body slide changed: {i + 1}'
            assert original[i].get_text() == deck[i].get_text()
        report.append('PASS: title banner and slides 2–19 render pixel-identically to original.')
    with fitz.open(NOTES_PDF) as notes:
        assert len(notes) == 20, f'Expected guide + 19 script pages, got {len(notes)}'
        for n in range(1, 20):
            text = notes[n].get_text()
            assert 'ENGLISH SCRIPT' in text and 'KOREAN TRANSLATION' in text
            assert '발표자 메모' in text
            assert '�' not in text
            for block in notes[n].get_text('blocks'):
                assert block[0] >= 0 and block[1] >= 0
                assert block[2] <= notes[n].rect.width + 1
                assert block[3] <= notes[n].rect.height + 1
        for n in [0, 1, 9, 11, 12, 16, 17, 19]:
            notes[n].get_pixmap(matrix=fitz.Matrix(1.3, 1.3), alpha=False).save(
                PREVIEW / f'notes_page_{n + 1:02}.png')
        report.append('PASS: notes PDF has one guide plus 19 pages, each containing EN then KO and a presenter note.')
    word = Document(NOTES_DOCX)
    paragraphs = [p.text for p in word.paragraphs]
    for s in DATA['slides']:
        assert s['en'] in paragraphs and s['ko'] in paragraphs
    assert len(word.inline_shapes) == 19
    report.append('PASS: Word file includes all 19 English/Korean script pairs and 19 slide previews.')
    report.append(f'INFO: English word count {WORDS}; planned speaking time {stamp(SECONDS)}.')
    report.append('INFO: source slides SHA-256 ' + hashlib.sha256(ORIGINAL.read_bytes()).hexdigest())
    report.append('INFO: PDF content validated visually and with text extraction; DOCX content validated structurally (Word rendering not available).')
    (SUPPORT / '검수결과.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report))


if __name__ == '__main__':
    PREVIEW.mkdir(parents=True, exist_ok=True)
    build_deck()
    build_pdf()
    build_docx()
    build_txt()
    validate()
