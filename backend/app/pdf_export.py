import html
import re
from pathlib import Path

from fpdf import FPDF


FONT_LOCATIONS = [
    (
        Path(__file__).resolve().parents[2] / "fonts" / "DejaVuSans.ttf",
        Path(__file__).resolve().parents[2] / "fonts" / "DejaVuSans-Bold.ttf",
    ),
    (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ),
    (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
]


def _font_paths() -> tuple[Path, Path]:
    for regular, bold in FONT_LOCATIONS:
        if regular.is_file() and bold.is_file():
            return regular, bold
    raise RuntimeError("No Unicode TTF font was found for PDF export")


def _write_inline(pdf: FPDF, text: str) -> None:
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    parts = re.split(r"(\*\*.+?\*\*)", text)
    for part in parts:
        if not part:
            continue
        bold = part.startswith("**") and part.endswith("**")
        value = part[2:-2] if bold else part
        pdf.set_font("ResearchFont", style="B" if bold else "", size=11)
        pdf.write(6, html.unescape(value))


def _write_report(pdf: FPDF, report: str) -> None:
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            _write_inline(pdf, " ".join(paragraph).strip())
            pdf.ln(8)
            paragraph.clear()

    for line in report.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        bullet = re.match(r"^(\s*)[-*+]\s+(.+?)\s*$", line)

        if heading:
            flush_paragraph()
            level = min(len(heading.group(1)), 3)
            pdf.set_font("ResearchFont", style="B", size={1: 16, 2: 13, 3: 11}[level])
            pdf.multi_cell(0, 8, html.unescape(heading.group(2)))
            pdf.ln(2)
        elif bullet:
            flush_paragraph()
            level = min(len(bullet.group(1)) // 2, 4)
            pdf.set_x(pdf.l_margin + level * 8)
            _write_inline(pdf, "\u2022 ")
            _write_inline(pdf, bullet.group(2))
            pdf.ln(7)
        elif line.strip():
            paragraph.append(line.strip())
        else:
            flush_paragraph()

    flush_paragraph()


def create_report_pdf(topic: str, report: str) -> bytes:
    regular_font, bold_font = _font_paths()
    pdf = FPDF(format="letter")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)
    pdf.add_font("ResearchFont", "", str(regular_font))
    pdf.add_font("ResearchFont", "B", str(bold_font))
    pdf.add_page()

    pdf.set_font("ResearchFont", style="B", size=18)
    pdf.multi_cell(0, 10, html.unescape(topic))
    pdf.ln(5)
    _write_report(pdf, report)
    return bytes(pdf.output())