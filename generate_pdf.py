"""Convert PAPER_SHORT.md to a clean letter-format PDF using reportlab.

Robust handling of long tables (word-wrap, auto-sized columns), inline
markdown stripping, and a clean two-tone heading hierarchy.
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable,
)
from reportlab.lib.enums import TA_LEFT


SRC = Path("PAPER_SHORT.md")
DST = Path("PAPER_SHORT.pdf")


def _strip_md(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`(.*?)`", r'<font face="Courier">\1</font>', text)
    return text


def build_styles() -> dict:
    base = getSampleStyleSheet()
    BLACK = colors.HexColor("#000000")
    body = ParagraphStyle(
        "Body",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13.5,
        alignment=TA_LEFT,
        spaceAfter=4,
        textColor=BLACK,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=BLACK,
        spaceBefore=0,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=BLACK,
        spaceBefore=10,
        spaceAfter=5,
    )
    h3 = ParagraphStyle(
        "H3",
        parent=base["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=BLACK,
        spaceBefore=6,
        spaceAfter=3,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=14,
        bulletIndent=4,
        spaceAfter=2,
        textColor=BLACK,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=body,
        fontSize=8.5,
        leading=11,
        spaceAfter=0,
        spaceBefore=0,
        textColor=BLACK,
    )
    table_head = ParagraphStyle(
        "TableHead",
        parent=table_cell,
        fontName="Helvetica-Bold",
        textColor=BLACK,
    )
    return {
        "body": body,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "bullet": bullet,
        "tcell": table_cell,
        "thead": table_head,
    }


def make_table(rows: list[list[str]], styles: dict, usable_w: float):
    if not rows:
        return None
    n_cols = max(len(r) for r in rows)
    # Pad short rows with empty cells
    norm = [r + [""] * (n_cols - len(r)) for r in rows]
    head = [Paragraph(_strip_md(c), styles["thead"]) for c in norm[0]]
    body = [
        [Paragraph(_strip_md(c), styles["tcell"]) for c in row]
        for row in norm[1:]
    ]
    data = [head] + body
    col_w = usable_w / n_cols
    table = Table(data, colWidths=[col_w] * n_cols, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def render(md: str) -> list:
    styles = build_styles()
    flow: list = []
    page_w, _ = LETTER
    margin = 0.7 * inch
    usable_w = page_w - 2 * margin

    lines = md.splitlines()
    in_table = False
    table_rows: list[list[str]] = []
    in_intro = True  # before first ---

    def flush_table_now() -> None:
        nonlocal table_rows
        if not table_rows:
            return
        tbl = make_table(table_rows, styles, usable_w)
        if tbl is not None:
            flow.append(tbl)
            flow.append(Spacer(1, 6))
        table_rows = []

    first_h1_seen = False

    for raw in lines:
        line = raw.rstrip()

        # Table parsing
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            # skip separator row
            if all(set(c) <= set("-: ") for c in cells if c):
                continue
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(cells)
            continue
        else:
            if in_table:
                flush_table_now()
                in_table = False

        # Headings
        if line.startswith("# "):
            if first_h1_seen:
                flow.append(PageBreak())
            first_h1_seen = True
            flow.append(Paragraph(_strip_md(line[2:]), styles["h1"]))
        elif line.startswith("## "):
            flow.append(Paragraph(_strip_md(line[3:]), styles["h2"]))
        elif line.startswith("### "):
            flow.append(Paragraph(_strip_md(line[4:]), styles["h3"]))
        elif line.startswith("---"):
            flow.append(Spacer(1, 4))
            flow.append(
                HRFlowable(
                    width="100%",
                    thickness=0.4,
                    color=colors.black,
                    spaceBefore=2,
                    spaceAfter=4,
                )
            )
        elif line.startswith("- "):
            flow.append(
                Paragraph(_strip_md(line[2:]), styles["bullet"], bulletText="\u2022")
            )
        elif line.strip() == "":
            flow.append(Spacer(1, 3))
        else:
            flow.append(Paragraph(_strip_md(line), styles["body"]))

    if in_table:
        flush_table_now()

    return flow


def main() -> None:
    md = SRC.read_text(encoding="utf-8")
    doc = SimpleDocTemplate(
        str(DST),
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="Hazard-Maze RL Benchmark - Short Paper",
        author="Tejas Naladala",
    )
    story = render(md)
    doc.build(story)
    size_kb = DST.stat().st_size // 1024
    print(f"Wrote {DST.resolve()} ({size_kb} KB)")


if __name__ == "__main__":
    main()
