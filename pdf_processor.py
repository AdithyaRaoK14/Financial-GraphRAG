"""
pdf_processor.py
================

High-quality PDF processor for financial reports.

Features
--------
✓ Preserves reading order (row AND column)
✓ Preserves tables, with bounding boxes for title/order detection
✓ Preserves page numbers
✓ Preserves headings (font-size AND bold based)
✓ OCR-ready architecture, with failure isolation
✓ JSON-ready output
✓ Embedding-friendly text generation
✓ Normalized section labels

Dependencies
------------
pip install pymupdf pdfplumber pandas camelot-py opencv-python

OCR (optional)
--------------
pip install pytesseract pillow
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz
import pdfplumber

import config

try:
    import pytesseract
    from PIL import Image

    # pytesseract's default lookup just searches PATH, which is correct
    # on Linux/Mac (this container included — /usr/bin/tesseract is on
    # PATH) and on most Windows installs too. The hardcoded path below
    # is ONLY needed on a Windows machine where Tesseract wasn't added
    # to PATH — setting it unconditionally (as this used to do) breaks
    # OCR entirely everywhere else: pytesseract.tesseract_cmd gets
    # pointed at a path that doesn't exist on Linux/Mac, so every OCR
    # call — the full-page _ocr_page() fallback AND per-cell repair —
    # raises TesseractNotFoundError, silently swallowed by the
    # try/except around each call site, so OCR just quietly never runs.
    _WINDOWS_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    if not shutil.which("tesseract"):
        if os.name == "nt" and Path(_WINDOWS_TESSERACT_PATH).exists():
            pytesseract.pytesseract.tesseract_cmd = _WINDOWS_TESSERACT_PATH
        else:
            raise RuntimeError("tesseract binary not found on PATH")

    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# Data Classes
# -------------------------------------------------------------


@dataclass
class TextBlock:
    page: int
    text: str
    bbox: tuple
    font_size: float
    is_heading: bool = False
    is_bold: bool = False
    section: str | None = None


@dataclass
class TableBlock:
    page: int
    table_index: int

    headers: list[str]

    rows: list[list[str]]

    bbox: tuple | None = None

    title: str = ""


@dataclass
class ProcessedPage:
    page: int

    text_blocks: list[TextBlock] = field(default_factory=list)

    tables: list[TableBlock] = field(default_factory=list)


# -------------------------------------------------------------
# Main Processor
# -------------------------------------------------------------


class PDFProcessor:
    # Bold text below the "real heading" font size is still treated as a
    # heading if it's short and doesn't look like a sentence — many
    # quarterly reports use bold subsection labels instead of larger fonts.
    BOLD_HEADING_MIN_FONT = 10
    BOLD_HEADING_MAX_LEN = 90

    # Raw heading text -> canonical section label. Keeps the graph clean
    # instead of storing whatever exact heading wording a given report used.
    SECTION_PATTERNS = {
        "management discussion and analysis": "Management Discussion & Analysis",
        "management discussion": "Management Discussion & Analysis",
        "financial highlights": "Financial Highlights",
        "financial results": "Financial Results",
        "balance sheet": "Balance Sheet",
        "statement of assets": "Balance Sheet",
        "statement of liabilities": "Balance Sheet",
        "income statement": "Income Statement",
        "statement of profit": "Income Statement",
        "profit and loss": "Income Statement",
        "cash flow": "Cash Flow",
        "notes to accounts": "Notes",
        "notes": "Notes",
        "risk factors": "Risk Factors",
        "guidance": "Guidance",
        "outlook": "Outlook",
        "segment": "Segment Information",
        "share capital": "Share Capital",
        "earnings call": "Earnings Call",
        "question and answer": "Q&A",
        "conference call": "Conference Call",
    }

    # Keyword scoring for classifying what KIND of page this is — lets us
    # skip low-value boilerplate (auditor legalese, security-cover
    # certificates) instead of indexing it alongside actual financial data.
    # A page needs >=2 keyword hits to be classified as anything other
    # than the default "financial_statement", so a stray mention doesn't
    # misclassify a real results page.
    PAGE_TYPE_PATTERNS = {
        "auditor_report": [
            "independent auditor",
            "review report",
            "sre 2410",
            "chartered accountants",
            "udin",
            "we have reviewed the accompanying",
            "auditor's responsibility",
            "review engagements",
            "audit opinion",
        ],
        "security_cover": [
            "statement of security cover",
            "debenture trustee",
            "security cover",
        ],
        "board_notice": [
            "outcome of board meeting",
            "intimation under regulation",
            "trading window",
        ],
    }

    # Short blocks dominated by these patterns (registered office, CIN,
    # phone/email/fax) repeat on almost every page and add noise rather
    # than retrievable content.
    BOILERPLATE_PATTERN = re.compile(
        r"regd\.?\s*&?\s*corporate office|registered office|corporate office|"
        r"\bcin\s*:|\bwebsite\s*:|\bemail\s*:|\bphone\s*:|\bfax\s*:|\btel\s*:",
        re.IGNORECASE,
    )

    def __init__(self):

        self.heading_font_threshold = 13

    # ---------------------------------------------------------
    # PROCESS (opens the PDF once — via fitz AND pdfplumber — and
    # reuses both handles for every page instead of reopening
    # pdfplumber per page via the fragile `page.parent.name` path)
    # ---------------------------------------------------------

    def process(self, pdf_path: str | Path):

        pdf_path = Path(pdf_path)

        logger.info("Processing %s", pdf_path)

        document = fitz.open(pdf_path)
        plumber_doc = pdfplumber.open(pdf_path)

        pages = []

        try:
            for page_number in range(len(document)):
                page = document.load_page(page_number)
                plumber_page = plumber_doc.pages[page_number]

                processed = self._process_page(page, plumber_page)

                pages.append(processed)
        finally:
            document.close()
            plumber_doc.close()

        return pages

    # ---------------------------------------------------------

    def _process_page(self, page: fitz.Page, plumber_page):

        processed = ProcessedPage(page=page.number + 1)

        processed.text_blocks = self._extract_text_blocks(page)

        processed.tables = self._extract_tables(plumber_page, page)

        if not processed.tables and self._page_is_image_only(page):
            ocr_tables = self._ocr_reconstruct_tables(page)
            if ocr_tables:
                processed.tables = ocr_tables

        return processed

    def _page_is_image_only(self, page) -> bool:
        """Same threshold as _needs_ocr() — true when this page has
        essentially no native vector text at all. Used to gate the OCR
        table reconstruction below, which is meaningfully more
        expensive than the free pdfplumber attempt that already ran, so
        it's only worth trying on a page pdfplumber genuinely had
        nothing to work with (a scanned/rasterized page), not every
        page where it simply didn't find a table."""
        return len(page.get_text().strip()) < 40

    # ---------------------------------------------------------
    # TEXT EXTRACTION
    # ---------------------------------------------------------

    def _extract_text_blocks(self, page):

        blocks = []

        data = page.get_text("dict")

        for block in data["blocks"]:
            if "lines" not in block:
                continue

            text = []

            max_font = 0
            is_bold = False

            for line in block["lines"]:
                for span in line["spans"]:
                    txt = span["text"].strip()

                    if not txt:
                        continue

                    text.append(txt)

                    max_font = max(max_font, span["size"])

                    if span.get("flags", 0) & 16:
                        is_bold = True

            joined = " ".join(text).strip()

            if not joined:
                continue

            cleaned = self._clean_ocr_garbage(joined)

            cleaned = self._normalize_numbers(cleaned)

            if not cleaned:
                continue

            is_heading = self._looks_like_heading(
                cleaned,
                max_font,
                is_bold,
            )

            blocks.append(
                TextBlock(
                    page=page.number + 1,
                    text=cleaned,
                    bbox=tuple(block["bbox"]),
                    font_size=max_font,
                    is_heading=is_heading,
                    is_bold=is_bold,
                )
            )

        blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))

        if self._needs_ocr(blocks):
            logger.info(
                "Page %d appears scanned. OCR available: %s",
                page.number + 1,
                OCR_AVAILABLE,
            )

            if OCR_AVAILABLE:
                ocr_blocks = self._ocr_page(page)

                if ocr_blocks:
                    return ocr_blocks

        return blocks

    # ---------------------------------------------------------

    def _looks_like_heading(
        self,
        text: str,
        font_size: float,
        is_bold: bool = False,
    ) -> bool:

        # Reject gibberish before anything else — stylized text (stamps,
        # signature blocks, watermarks) sometimes gets picked up by PyMuPDF
        # at a large font size and would otherwise pass the size check
        # below. Low alphabetic content or a heavy concentration of
        # special characters means "not a real heading", regardless of
        # font size or boldness.
        if not self._is_plausible_heading_text(text):
            return False

        if font_size >= self.heading_font_threshold:
            return True

        if (
            is_bold
            and font_size >= self.BOLD_HEADING_MIN_FONT
            and len(text) < self.BOLD_HEADING_MAX_LEN
            and not text.endswith(".")
        ):
            return True

        if text.isupper() and len(text) < 80:
            return True

        if re.match(r"^\d+(\.\d+)*\s", text):
            return True

        return False

    def _is_plausible_heading_text(self, text: str) -> bool:

        stripped = text.strip()

        if not stripped:
            return False

        letters = sum(1 for c in stripped if c.isalpha())

        if letters == 0:
            return False

        alpha_ratio = letters / len(stripped)

        if alpha_ratio < 0.6:
            return False

        allowed_punct = set(".,&()-'\"/%")
        specials = sum(
            1
            for c in stripped
            if not (c.isalnum() or c.isspace() or c in allowed_punct)
        )

        if specials / len(stripped) > 0.15:
            return False

        words = re.findall(r"[A-Za-z]+", stripped)
        alpha_words = [w for w in words if len(w) >= 3]

        if not alpha_words:
            return False

        # OCR/stamp garbage tends to have (a) mid-word case switches like
        # "lcAt" / "BANGALOREIuT", and (b) consonant runs with no vowels
        # like "BKtsLv". Real headings essentially never look like this,
        # so a heading dominated by such "words" is rejected even if the
        # overall alpha/special-char ratios looked fine.
        bad_words = 0

        for w in alpha_words:
            inner = w[1:-1] if len(w) > 2 else w
            has_internal_case_switch = bool(
                re.search(r"[a-z][A-Z]|[A-Z]{2,}[a-z]", inner)
            )
            has_vowel = bool(re.search(r"[aeiouAEIOU]", w))

            if has_internal_case_switch or not has_vowel:
                bad_words += 1

        if bad_words / len(alpha_words) > 0.3:
            return False

        return True

    # ---------------------------------------------------------
    # OCR GARBAGE REMOVAL (word-level, applied within prose — unlike
    # the heading-level check above, this runs on real paragraph text
    # sitting next to legitimate content, so it's deliberately more
    # conservative: it must NOT delete real words, including camelCase
    # brand names like "CreditAccess" or "eBay".)
    # ---------------------------------------------------------

    def _looks_like_garbage_token(self, word: str) -> bool:

        stripped = word.strip(".,;:()[]{}\"'")

        if not stripped or len(stripped) < 4:
            return False

        letters = sum(1 for c in stripped if c.isalpha())

        if letters == 0:
            return False

        alpha_ratio = letters / len(stripped)
        specials = sum(1 for c in stripped if not c.isalnum())

        # Heavy symbol contamination mixed with only a few letters
        # (e.g. "$ri;].", "ir$,--$") — this is the main OCR-garbage shape.
        if alpha_ratio < 0.7 and specials >= 2:
            return True

        # A long letter-only run with zero vowels. Real words/brand names
        # of this length always have a vowel; short vowel-less
        # abbreviations like PVT/LLP/CIN are exempted by the length
        # threshold. Deliberately does NOT flag mid-word case switches —
        # that pattern is indistinguishable from real camelCase brand
        # names (CreditAccess, eBay, iPhone) at the single-token level.
        only_letters = "".join(c for c in stripped if c.isalpha())

        if len(only_letters) >= 5 and not re.search(r"[aeiouAEIOU]", only_letters):
            return True

        return False

    def _clean_ocr_garbage(self, text: str) -> str:

        lines = text.split("\n")

        cleaned_lines = []

        for line in lines:
            tokens = line.split(" ")

            kept = [t for t in tokens if not self._looks_like_garbage_token(t)]

            cleaned_lines.append(" ".join(kept))

        return "\n".join(cleaned_lines).strip()

    def _normalize_numbers(self, text: str) -> str:
        """
        Best-effort cleanup of common OCR corruption in numbers, e.g.
        "1 ,363.'t 7" instead of "1,363.17". This is NOT a full fix (that
        would need per-digit OCR confidence data this pipeline doesn't
        have) — it just collapses the most common stray-whitespace and
        stray-quote patterns around digits.
        """

        text = re.sub(r"(\d)\s+,", r"\1,", text)
        text = re.sub(r",\s+(\d)", r",\1", text)
        text = re.sub(r"(\d)\s+\.\s+(\d)", r"\1.\2", text)
        text = re.sub(r"(\d)'(\d)", r"\1\2", text)

        return text

    # ---------------------------------------------------------

    def _needs_ocr(self, blocks):

        if len(blocks) == 0:
            return True

        total = sum(len(b.text) for b in blocks)

        return total < 40

    # ---------------------------------------------------------
    # OCR (failures on one page no longer kill the whole document)
    # ---------------------------------------------------------

    def _ocr_page(self, page):

        if not OCR_AVAILABLE:
            return []

        try:
            from pytesseract import Output

            pix = page.get_pixmap(dpi=300)

            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples,
            )

            try:
                osd = pytesseract.image_to_osd(img, output_type=Output.DICT)

                rotation = osd["rotate"]

                if rotation:
                    logger.info(
                        "Rotating page %d by %d degrees",
                        page.number + 1,
                        rotation,
                    )

                    img = img.rotate(rotation, expand=True)

            except Exception:
                logger.debug(
                    "Orientation detection failed on page %d",
                    page.number + 1,
                )

            text = pytesseract.image_to_string(
                img,
                config="--psm 6",
            )
        except Exception:
            logger.warning(
                "OCR failed on page %d — continuing without it",
                page.number + 1,
                exc_info=True,
            )
            return []

        if not text.strip():
            return []

        # The OCR branch used to return the WHOLE page as one TextBlock.
        # That broke two things downstream:
        #   1. remove_boilerplate_blocks only checks blocks <=300 chars —
        #      a whole-page block is always longer, so letterhead/CIN/
        #      phone/fax text never got a chance to be filtered out.
        #   2. _clean_ocr_garbage/_normalize_numbers (built specifically
        #      for OCR noise) were never applied to OCR output, since
        #      this function returns before that cleaning step runs.
        # Splitting into per-line blocks — and cleaning each one — gives
        # OCR'd pages the same filtering born-digital pages already get.
        blocks = []

        for line in text.split("\n"):
            cleaned = self._clean_ocr_garbage(line)
            cleaned = self._normalize_numbers(cleaned)

            if not cleaned.strip():
                continue

            blocks.append(
                TextBlock(
                    page=page.number + 1,
                    text=cleaned,
                    bbox=(0, 0, pix.width, pix.height),
                    font_size=12,
                    is_heading=False,
                )
            )

        return blocks

    # ---------------------------------------------------------
    # OCR TABLE RECONSTRUCTION — full-page OCR (_ocr_page above)
    # recovers PROSE fine on a scanned page, but flattens everything
    # into paragraph-shaped text: any table on that page has no
    # row/column structure left for chunker.py to recognize, even
    # though every number is still sitting right there in the OCR
    # output (confirmed directly: pdfplumber and even fitz's own
    # get_text() return literally zero characters on a page whose
    # content is one full-page scanned image — there is no vector text
    # for either of them to find a table in, so _extract_tables()
    # always returns [] on pages like this, silently).
    #
    # This rebuilds table structure straight from Tesseract's
    # word-level bounding boxes instead: words are grouped into lines
    # via Tesseract's own line numbering, a contiguous run of
    # "row-like" lines (enough clean numeric tokens relative to total
    # words on the line — excludes ordinary sentences that happen to
    # contain a date or two) is treated as one table, and each row's
    # numeric tokens are assigned to a column by nearest x-position to
    # a set of column anchors built by clustering every numeric
    # token's RIGHT edge across the whole region. Numbers in financial
    # tables are right-aligned, so right edges line up per column even
    # when a row is missing a value OCR couldn't read — confirmed on a
    # real scanned filing: recovered every value OCR read correctly,
    # placed in the correct column, while a handful of individually
    # unreadable digit runs were correctly left blank rather than
    # guessed.
    #
    # Only ever ADDS structure pdfplumber already failed to find at
    # all — _process_page only calls this when _extract_tables()
    # returned nothing AND the page has ~zero native text, so it never
    # overrides or competes with a table pdfplumber did extract.
    # ---------------------------------------------------------

    _OCR_TABLE_COLUMN_GAP_PX = 40  # at 300 dpi; see calibration note below

    def _ocr_clean_numeric_token(self, tok: str) -> str | None:
        """Cleans one OCR'd word into a plain numeric string, or None
        if it isn't confidently one — same fail-closed rule as the rest
        of this pipeline: an unreadable/ambiguous token is dropped, not
        guessed. Handles parens/braces as negative, a trailing %, comma
        thousands-separators, and one OCR-specific repair: a lone comma
        followed by exactly 2 digits at the end of a token with no real
        decimal point elsewhere almost always means Tesseract misread a
        period as a comma ('27,66' for '27.66') — Indian financial
        statements only ever use commas for digit-grouping, never as a
        decimal separator, so this shape is unambiguous."""
        if not tok:
            return None

        t = tok.strip().replace("{", "(").replace("}", ")")

        negative = False
        m = re.fullmatch(r"\((.*)\)", t)
        if m:
            negative = True
            t = m.group(1)

        t = t.rstrip("%")

        if "." not in t and re.search(r",\d{2}$", t):
            idx = t.rindex(",")
            t = t[:idx] + "." + t[idx + 1 :]

        t = t.replace(",", "")

        if not t or not re.fullmatch(r"-?\d+\.?\d*", t):
            return None

        return f"-{t}" if negative else t

    def _ocr_reconstruct_tables(self, page) -> list[TableBlock]:
        if not OCR_AVAILABLE:
            return []

        try:
            from pytesseract import Output

            pix = page.get_pixmap(dpi=300)

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            data = pytesseract.image_to_data(
                img, output_type=Output.DICT, config="--psm 6"
            )
        except Exception:
            logger.warning(
                "OCR table reconstruction failed on page %d",
                page.number + 1,
                exc_info=True,
            )
            return []

        lines: dict[tuple, list[tuple]] = {}
        n = len(data.get("text", []))

        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not txt:
                continue

            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(
                (
                    txt,
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )
            )

        if not lines:
            return []

        ordered_keys = sorted(lines, key=lambda k: min(w[2] for w in lines[k]))

        row_data = []

        for key in ordered_keys:
            words = sorted(lines[key], key=lambda w: w[1])
            numeric = [(w, self._ocr_clean_numeric_token(w[0])) for w in words]
            numeric = [(w, v) for w, v in numeric if v is not None]

            # Require BOTH a minimum count and a minimum density —
            # count alone would treat an ordinary sentence that happens
            # to mention two numbers ("...for the quarter ended June
            # 30, 2023") as a table row; density alone would treat a
            # short two-word numeric fragment as one. Real table rows
            # in these filings are dense with numbers relative to their
            # (short) label, so both together is a reliable signal.
            is_row = len(numeric) >= 2 and len(numeric) / len(words) >= 0.25

            row_data.append((words, numeric, is_row))

        row_indices = [i for i, (_, _, is_row) in enumerate(row_data) if is_row]

        if len(row_indices) < 3:
            # Too little confidently-numeric content on this page to
            # justify treating any part of it as a table — same
            # "return nothing rather than guess" contract as everywhere
            # else in this file.
            return []

        lo, hi = min(row_indices), max(row_indices)
        table_rows = row_data[lo : hi + 1]

        right_edges = sorted(
            w[1] + w[3] for _, numeric, _ in table_rows for w, _ in numeric
        )
        if not right_edges:
            return []

        # Column-anchor clustering. Calibrated against a real scanned
        # filing at 300 dpi: right-edge gaps within the same column
        # were 0-2px; gaps between columns were 178px+ — a wide,
        # unambiguous separation, so a fixed mid-range threshold is
        # safe rather than needing per-document tuning.
        clusters = [[right_edges[0]]]
        for edge in right_edges[1:]:
            if edge - clusters[-1][-1] > self._OCR_TABLE_COLUMN_GAP_PX:
                clusters.append([edge])
            else:
                clusters[-1].append(edge)
        anchors = [sum(c) / len(c) for c in clusters]

        rows = []

        for words, numeric, _ in table_rows:
            label_tokens = [
                w[0] for w in words if self._ocr_clean_numeric_token(w[0]) is None
            ]
            label = " ".join(label_tokens).strip()

            row = [""] * len(anchors)

            for w, value in numeric:
                right = w[1] + w[3]
                col = min(range(len(anchors)), key=lambda i: abs(anchors[i] - right))

                if row[col]:
                    # Two tokens from the same row landed in the same
                    # column — the clustering got this row wrong
                    # somewhere; drop both rather than silently keep
                    # whichever arrived second.
                    row[col] = ""
                    continue

                row[col] = value

            rows.append([label] + row)

        if not rows:
            return []

        logger.info(
            "Reconstructed a %d-row/%d-col table via OCR on page %d "
            "(pdfplumber found no vector text to work with)",
            len(rows),
            len(anchors) + 1,
            page.number + 1,
        )

        return [
            TableBlock(
                page=page.number + 1,
                table_index=0,
                headers=[""] * (len(anchors) + 1),
                rows=rows,
                bbox=None,
            )
        ]

    # ---------------------------------------------------------
    # TABLE EXTRACTION (reuses the already-open plumber page —
    # no more reopening the PDF per page — and records each
    # table's bbox so title/reading-order detection can use it)
    # ---------------------------------------------------------

    # pdfplumber's default table-finding strategy ("lines" for both axes)
    # requires visible ruling lines in the PDF. A lot of financial
    # statements — especially Indian quarterly filings — use whitespace-
    # aligned columns with no visible grid, which the default strategy
    # finds NOTHING for (observed: some reports produced zero table
    # chunks at all, everything falling through to plain text). Trying
    # several strategies per page and keeping whichever produces the
    # most coherent result catches both fully-borderless tables and the
    # common mixed case (ruled columns, borderless rows or vice versa).
    _TABLE_SETTINGS_CANDIDATES = [
        None,  # pdfplumber default: lines/lines
        {"vertical_strategy": "text", "horizontal_strategy": "text"},
        {"vertical_strategy": "lines", "horizontal_strategy": "text"},
        {"vertical_strategy": "text", "horizontal_strategy": "lines"},
    ]

    def _cell_looks_like_real_content(self, cell) -> bool:
        """True for a cell that looks like genuine content — a clean
        number, or a plausible word/short label — as opposed to garbled
        noise ('$.r"XS', '=TF', '1nded,,,r,;,1i;q\'*@'). Used to weight
        table-candidate scoring by actual quality, not just cell count:
        a table that's mostly garbled punctuation-heavy fragments
        should lose to a smaller, cleaner one even if it has more
        cells overall."""
        if not cell:
            return False
        s = str(cell).strip()
        if not s:
            return False
        if self._cell_looks_clean_numeric(s):
            return True
        letters = sum(c.isalpha() for c in s)
        return len(s) >= 2 and letters / len(s) >= 0.6

    def _score_extracted_tables(self, raw_tables) -> float:
        """Higher is better: rewards tables that are bigger, have more
        than one real column (a single-column dump means column
        boundaries weren't actually detected — exactly the garbled
        one-column-per-row failure mode this is meant to catch), are
        densely filled rather than mostly blank cells, AND — critically
        — whose filled cells actually look like real content rather
        than garbled noise. Size alone isn't enough: a 10-column table
        that's mostly letterhead fragments and split numbers ('1,' /
        '105.17' as separate cells) can have more raw cells than a
        correctly-reconstructed 4-column one, and used to outscore it
        for exactly that reason — quality_ratio is what stops that."""
        score = 0.0
        for table in raw_tables or []:
            if not table or len(table) < 2:
                continue
            num_cols = max((len(row) for row in table), default=0)
            if num_cols < 2:
                continue  # a "table" with only one column isn't one
            total_cells = sum(len(row) for row in table)
            if total_cells == 0:
                continue
            non_empty = [
                cell for row in table for cell in row if cell and str(cell).strip()
            ]
            if not non_empty:
                continue
            fill_ratio = len(non_empty) / total_cells
            quality_ratio = sum(
                1 for c in non_empty if self._cell_looks_like_real_content(c)
            ) / len(non_empty)
            score += len(table) * num_cols * fill_ratio * quality_ratio
        return score

    def _looks_like_letterhead_row(self, row) -> bool:
        """True for a 'header' row that's actually page letterhead/company
        name text that leaked into the table's bounding box — exactly one
        long, sentence-like non-empty cell (>40 chars, multiple words)
        while every other cell is blank. A real header row has several
        short column labels, not one long one."""
        non_empty = [c for c in row if c and str(c).strip()]
        if len(non_empty) != 1:
            return False
        cell = str(non_empty[0]).strip()
        return len(cell) > 40 and len(cell.split()) >= 4

    # ---------------------------------------------------------
    # WORD-CLUSTERING TABLE CANDIDATE — an additional table-detection
    # strategy alongside the pdfplumber-settings candidates above.
    #
    # Root cause this addresses: on dense, borderless "financial
    # results" layouts, pdfplumber's word-GAP-based column detection
    # (the "text" strategy) decides column boundaries independently per
    # row. On these filings that's unreliable enough that it routinely
    # merges the page's letterhead into the same table as the real
    # data and finds inconsistent column counts row-to-row — confirmed
    # directly: a value as simple as "1,105.17" was coming out split
    # across two cells as "1," and "105.17" because a row-local
    # word-gap heuristic guessed a column boundary in the middle of a
    # single number.
    #
    # This instead groups words into lines by y-position (so a row's
    # full text — label AND every value — is one unit, never split
    # mid-number), then clusters column boundaries from the RIGHT EDGE
    # of every numeric token across the WHOLE table region at once,
    # not per-row. Financial figures are right-aligned, so their right
    # edges line up consistently across rows even when pdfplumber's
    # per-row heuristic would guess differently row to row. Verified
    # against a real filing: recovered "1,105.17 964.79 736.23
    # 3,327.13" as four correctly-separated, correctly-aligned values
    # where the pdfplumber-strategy candidates had produced "1," and
    # "105.17" as separate garbled cells.
    #
    # This is scored via the same _score_extracted_tables() as every
    # other candidate below — it doesn't automatically win, it has to
    # earn it. On pages with a genuinely well-ruled table, a
    # pdfplumber-lines-strategy candidate is likely to still score
    # higher (and there's nothing wrong with letting it).
    # ---------------------------------------------------------

    def _group_words_into_lines(self, words, y_tolerance: float = 3.0):
        """Groups a flat list of word dicts (each with 'text', 'x0',
        'top', 'x1') into lines by y-position proximity, sorted into
        reading order (top-to-bottom, left-to-right within a line)."""
        if not words:
            return []

        ordered = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))

        lines = []
        current = []
        current_top = None

        for w in ordered:
            if current_top is None or abs(w["top"] - current_top) <= y_tolerance:
                current.append(w)
                current_top = w["top"] if current_top is None else current_top
            else:
                lines.append(sorted(current, key=lambda x: x["x0"]))
                current = [w]
                current_top = w["top"]

        if current:
            lines.append(sorted(current, key=lambda x: x["x0"]))

        return lines

    def _cluster_table_from_lines(self, lines, column_gap: float):
        """Shared reconstruction core: given lines of words (see
        _group_words_into_lines), classifies "row-like" lines (dense
        with clean numeric tokens — excludes an ordinary sentence that
        happens to mention a date), takes the contiguous envelope of
        those as the table region, clusters column anchors from
        numeric tokens' right edges within it (dropping any cluster
        under 30% of the largest — these are stray incidental numbers,
        like a street-address digit or a footnote marker, not a real
        column), and assigns each row's numeric tokens to their nearest
        anchor. `column_gap` is in whatever unit the word bboxes use
        (PDF points for vector text, pixels for OCR) — the two callers
        pass different values since they're different unit scales, not
        because the algorithm differs.

        Returns (headers, rows, cell_bboxes) — cell_bboxes is a grid
        the same shape as rows, holding each value cell's own bbox (or
        None) so a caller with OCR available can repair individual
        cells afterward. Returns (None, None, None) if nothing
        table-shaped was found.
        """
        row_data = []

        for line in lines:
            numeric = [(w, self._ocr_clean_numeric_token(w["text"])) for w in line]
            numeric = [(w, v) for w, v in numeric if v is not None]
            is_row = len(numeric) >= 2 and len(numeric) / len(line) >= 0.25
            row_data.append((line, numeric, is_row))

        row_indices = [i for i, (_, _, is_row) in enumerate(row_data) if is_row]
        if len(row_indices) < 3:
            return None, None, None

        lo, hi = min(row_indices), max(row_indices)
        table_lines = row_data[lo : hi + 1]

        right_edges = sorted(
            w["x1"] for _, numeric, _ in table_lines for w, _ in numeric
        )
        if not right_edges:
            return None, None, None

        clusters = [[right_edges[0]]]
        for edge in right_edges[1:]:
            if edge - clusters[-1][-1] > column_gap:
                clusters.append([edge])
            else:
                clusters[-1].append(edge)

        max_size = max(len(c) for c in clusters)
        kept = [c for c in clusters if len(c) >= 0.3 * max_size]
        if not kept:
            return None, None, None
        anchors = [sum(c) / len(c) for c in kept]

        def nearest_anchor(x):
            return min(range(len(anchors)), key=lambda i: abs(anchors[i] - x))

        headers = [""] * (len(anchors) + 1)
        rows = []
        cell_bboxes = []

        # Anything positioned at or past the leftmost value column is in
        # the "value region" of the row, even if it didn't parse as a
        # clean number — a corrupted cell like '71.U' sits exactly
        # where a real value belongs and fails _ocr_clean_numeric_token,
        # but it is NOT label text, and gluing it onto the label (as a
        # simpler "everything non-numeric is the label" rule would)
        # buries it somewhere _repair_corrupted_cells_from_bbox_grid can
        # never find it. Only text genuinely to the LEFT of where
        # values start is the real row label.
        label_boundary = min(anchors) - column_gap

        for line, numeric, _ in table_lines:
            numeric_word_ids = {id(w) for w, _ in numeric}

            label_tokens = []
            suspect_words = []

            for w in line:
                if id(w) in numeric_word_ids:
                    continue
                if w["x1"] < label_boundary:
                    label_tokens.append(w["text"])
                else:
                    suspect_words.append(w)

            label = " ".join(label_tokens).strip()

            row = [""] * len(anchors)
            bboxes = [None] * len(anchors)

            for w, value in numeric:
                ci = nearest_anchor(w["x1"])
                # Too far from every real anchor to trust — drop it
                # rather than force it into the nearest (wrong) column.
                if abs(anchors[ci] - w["x1"]) > column_gap * 1.5:
                    continue
                if row[ci]:
                    # Two tokens on the same row landed in the same
                    # column — something about this row's alignment
                    # didn't match the rest of the table; drop both
                    # rather than silently keep whichever arrived last.
                    row[ci] = ""
                    bboxes[ci] = None
                    continue
                row[ci] = value
                bboxes[ci] = (w["x0"], w["top"], w["x1"], w["bottom"])

            for w in suspect_words:
                ci = nearest_anchor(w["x1"])
                if abs(anchors[ci] - w["x1"]) > column_gap * 1.5:
                    continue
                if row[ci]:
                    # A real value already landed here — this suspect
                    # token is more likely trailing text than a second
                    # value for the same cell; leave the real value in
                    # place rather than blank it out.
                    continue
                # Leave the cell blank (nothing here parsed cleanly) but
                # keep its bbox — this is exactly the shape
                # _repair_corrupted_cells_from_bbox_grid needs to find
                # and re-OCR a font-corrupted value like '71.U' that
                # native text extraction genuinely can't read correctly.
                bboxes[ci] = (w["x0"], w["top"], w["x1"], w["bottom"])

            if not any(row):
                continue

            rows.append([label] + row)
            cell_bboxes.append([None] + bboxes)

        if not rows:
            return None, None, None

        return headers, rows, cell_bboxes

    def _vector_text_table_candidate(self, plumber_page):
        """Builds a table candidate directly from the page's native
        vector text (extract_words()) using the shared word-clustering
        reconstruction above, instead of trusting pdfplumber's own
        row-local word-gap column detection. Returns
        (raw_rows, cell_bboxes) in the same [headers, *data_rows] shape
        _extract_tables()'s other candidates produce, or ([], None) if
        nothing table-shaped was found."""
        try:
            words = plumber_page.extract_words()
        except Exception:
            logger.debug(
                "extract_words() failed on page %s",
                getattr(plumber_page, "page_number", "?"),
                exc_info=True,
            )
            return [], None

        if not words:
            return [], None

        lines = self._group_words_into_lines(words)

        # Calibrated against a real filing: right-edge gaps within the
        # same column were under 10pt; genuine column-to-column gaps
        # started around 17pt — a fixed mid-range threshold in PDF
        # points is safe without per-document tuning, the same way the
        # OCR path's pixel-space threshold was calibrated separately
        # (points and pixels are different scales, not different
        # algorithms).
        headers, rows, cell_bboxes = self._cluster_table_from_lines(
            lines, column_gap=18.0
        )

        if not rows:
            return [], None

        return [headers] + rows, cell_bboxes

    def _repair_corrupted_cells_from_bbox_grid(
        self, headers, rows, cell_bboxes, fitz_page
    ):
        """Same repair contract as _repair_corrupted_cells (a column
        only counts as suspect-worthy if most of its OTHER cells are
        already clean, and a repair is only accepted if it itself comes
        out clean), but for the word-clustering candidate, which
        already has each cell's own bbox from extract_words() rather
        than needing to look one up from a pdfplumber Table object."""
        if not OCR_AVAILABLE or not rows or not cell_bboxes:
            return headers, rows

        num_cols = max((len(r) for r in rows), default=0)
        if num_cols == 0:
            return headers, rows

        numeric_cols = set()
        for col in range(num_cols):
            values = [row[col] for row in rows if col < len(row) and row[col]]
            if len(values) < 2:
                continue
            clean = sum(1 for v in values if self._cell_looks_clean_numeric(v))
            if clean / len(values) >= 0.5:
                numeric_cols.add(col)

        if not numeric_cols:
            return headers, rows

        # A suspect is any cell in a numeric column that isn't already
        # clean AND has a recorded bbox — that includes a dirty
        # non-empty value (a rarer case for this candidate, since
        # numeric-column building already separates those out) and,
        # more commonly here, a BLANK cell whose bbox was recorded
        # because a token sat there but failed to parse cleanly (e.g.
        # '71.U') — see _cluster_table_from_lines' suspect_words
        # handling for where that bbox comes from.
        suspects = [
            (r_idx, col)
            for r_idx, row in enumerate(rows)
            for col in numeric_cols
            if col < len(row)
            and not self._cell_looks_clean_numeric(row[col])
            and r_idx < len(cell_bboxes)
            and col < len(cell_bboxes[r_idx])
            and cell_bboxes[r_idx][col] is not None
        ]

        if not suspects:
            return headers, rows

        try:
            pix = fitz_page.get_pixmap(dpi=400)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        except Exception:
            logger.debug(
                "Cell-repair render failed on page %d",
                fitz_page.number + 1,
                exc_info=True,
            )
            return headers, rows

        scale = 400 / 72.0
        repaired = 0

        for r_idx, col in suspects:
            if r_idx >= len(cell_bboxes) or col >= len(cell_bboxes[r_idx]):
                continue
            bbox = cell_bboxes[r_idx][col]
            if bbox is None:
                continue

            repaired_text = self._ocr_repair_bbox_value(img, scale, bbox)
            if repaired_text is not None:
                rows[r_idx][col] = repaired_text
                repaired += 1

        if repaired:
            logger.info(
                "Repaired %d font-corrupted cell(s) on page %d via targeted OCR "
                "(word-clustering candidate)",
                repaired,
                fitz_page.number + 1,
            )

        return headers, rows

    def _extract_tables(self, plumber_page, fitz_page=None):

        tables = []

        best_raw, best_found, best_score = [], [], -1.0
        best_settings = "unset"
        best_is_custom = False
        best_cell_bboxes = None

        for settings in self._TABLE_SETTINGS_CANDIDATES:
            try:
                if settings:
                    raw = plumber_page.extract_tables(table_settings=settings)
                    found = plumber_page.find_tables(table_settings=settings)
                else:
                    raw = plumber_page.extract_tables()
                    found = plumber_page.find_tables()
            except Exception:
                logger.debug(
                    "Table extraction failed for settings=%s on page %s",
                    settings,
                    getattr(plumber_page, "page_number", "?"),
                    exc_info=True,
                )
                continue

            score = self._score_extracted_tables(raw)
            if score > best_score:
                best_score = score
                best_raw = raw
                best_found = found
                best_settings = settings
                best_is_custom = False
                best_cell_bboxes = None

        # Did pdfplumber's winning strategy find real ruled lines to
        # define COLUMN boundaries, or did it have to guess them from
        # word gaps? Ruled lines are exact; word-gap guessing is a
        # row-local heuristic that can (and does, on dense borderless
        # filings) guess a different boundary row to row — the direct
        # cause of a value like "1,105.17" coming out split across two
        # cells as "1," and "105.17". `best_settings is None` is
        # pdfplumber's default (lines/lines); an explicit
        # vertical_strategy of "lines" also means real lines defined
        # columns even if rows used text-gaps. Only the remaining case
        # — vertical_strategy explicitly "text" — is the risky one this
        # candidate exists for.
        columns_from_ruled_lines = best_settings is None or (
            best_settings.get("vertical_strategy") == "lines"
        )

        if not columns_from_ruled_lines:
            try:
                custom_raw, custom_cell_bboxes = self._vector_text_table_candidate(
                    plumber_page
                )
            except Exception:
                logger.debug(
                    "Word-clustering table candidate failed on page %s",
                    getattr(plumber_page, "page_number", "?"),
                    exc_info=True,
                )
                custom_raw, custom_cell_bboxes = [], None

            # Prefer it outright (not via a numeric score comparison —
            # see this method's docstring for why a raw score contest
            # doesn't reliably favor the correct one here) once it's
            # found a reasonably-sized table: a header row plus at
            # least 3 data rows.
            if custom_raw and len(custom_raw) >= 4:
                best_raw = [custom_raw]
                best_found = [None]
                best_is_custom = True
                best_cell_bboxes = custom_cell_bboxes

        for idx, table in enumerate(best_raw):
            if not table:
                continue

            if best_is_custom:
                # Already row/column-clean by construction (built from
                # the row-like envelope directly, not a bounding box
                # that could have snagged letterhead text above it) —
                # nothing to trim.
                headers = list(table[0])
                rows = [list(r) for r in table[1:]]
                dropped = 0
            else:
                # Drop up to 2 leading rows that are letterhead/
                # company-name text rather than real column headers —
                # some pages' table bounding box starts a line or two
                # above the actual table, capturing the company name as
                # if it were the header row.
                trimmed = list(table)
                dropped = 0
                while (
                    trimmed
                    and dropped < 2
                    and self._looks_like_letterhead_row(trimmed[0])
                ):
                    trimmed.pop(0)
                    dropped += 1

                if not trimmed:
                    continue

                headers = [self._clean_cell(x) for x in trimmed[0]]
                rows = [[self._clean_cell(x) for x in row] for row in trimmed[1:]]

            plumber_table = best_found[idx] if idx < len(best_found) else None

            if fitz_page is not None:
                if best_is_custom:
                    headers, rows = self._repair_corrupted_cells_from_bbox_grid(
                        headers, rows, best_cell_bboxes, fitz_page
                    )
                else:
                    headers, rows = self._repair_corrupted_cells(
                        headers, rows, plumber_table, dropped, fitz_page
                    )

            bbox = plumber_table.bbox if plumber_table is not None else None

            tables.append(
                TableBlock(
                    page=plumber_page.page_number,
                    table_index=idx,
                    headers=headers,
                    rows=rows,
                    bbox=bbox,
                )
            )

        return tables

    # ---------------------------------------------------------
    # CELL-LEVEL OCR REPAIR (font-encoding corruption, NOT a scan/OCR
    # quality problem — some source PDFs mis-map a subset of glyph
    # codes in their embedded font, so fitz.get_text() itself returns
    # wrong characters for a handful of cells even with zero OCR
    # involved anywhere: verified directly — '71.64' extracts natively
    # as '71.U', 'Other expenses' as 'Olhcr cYbensar', with the exact
    # same font/encoding declared for the surrounding text that DOES
    # extract correctly. No text-cleanup regex can fix this, because
    # the wrong character was never the right one to begin with; only
    # re-reading the actual rendered pixels can. pdfplumber's table
    # detection (row/column layout) is trustworthy even when this
    # happens — it's the DECODED TEXT for a minority of cells that's
    # wrong — so this repairs individual cell values in place rather
    # than distrusting the whole table or falling back to full-page OCR.
    # ---------------------------------------------------------

    _CELL_OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789.,()%-"

    def _cell_looks_clean_numeric(self, value) -> bool:
        if not value:
            return False
        stripped = str(value).strip()
        if not stripped or not any(c.isdigit() for c in stripped):
            return False
        return bool(self._PURE_NUMBER_RE.fullmatch(stripped))

    def _ocr_repair_bbox_value(self, img, scale: float, bbox) -> str | None:
        """Shared crop+OCR+validate core: given a cell's bbox in PDF
        points (x0, top, x1, bottom) and a page image already rendered
        at `scale` (pixels per point), re-reads just that region and
        returns a cleaned numeric string, or None if the crop is
        invalid or OCR didn't produce something confidently numeric.
        Used both by the pdfplumber-Table-based repair below and by the
        word-clustering table candidate, which has its own per-cell
        bboxes from extract_words() and needs the same repair logic
        without a pdfplumber Table object to look them up from."""
        x0, top, x1, bottom = bbox
        pad = 2
        box = (
            max(0, int((x0 - pad) * scale)),
            max(0, int((top - pad) * scale)),
            min(img.width, int((x1 + pad) * scale)),
            min(img.height, int((bottom + pad) * scale)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            return None

        try:
            crop = img.crop(box)
            repaired_text = pytesseract.image_to_string(
                crop, config=self._CELL_OCR_CONFIG
            ).strip()
        except Exception:
            logger.debug("Cell OCR repair failed for bbox %s", bbox, exc_info=True)
            return None

        # Only accept the repair if it produced something that itself
        # looks like a clean number — a miss on a tiny crop (blank, more
        # garbage) is common enough that keeping the original
        # (already known-bad) value is safer than swapping in a second,
        # unverified guess.
        if repaired_text and self._cell_looks_clean_numeric(repaired_text):
            return repaired_text
        return None

    def _repair_corrupted_cells(
        self, headers, rows, plumber_table, header_row_offset, fitz_page
    ):
        if not OCR_AVAILABLE or plumber_table is None or not rows:
            return headers, rows

        num_cols = max((len(r) for r in rows), default=0)
        if num_cols == 0:
            return headers, rows

        # A column only counts as "numeric" — and therefore its outliers
        # as suspect — if most of its OTHER cells already parse as clean
        # numbers. Judging each cell against its own column's actual
        # behavior (rather than a fixed list of known-bad substrings) is
        # what lets this catch corruption it's never seen before.
        numeric_cols = set()
        for col in range(num_cols):
            values = [row[col] for row in rows if col < len(row) and row[col]]
            if len(values) < 2:
                continue
            clean = sum(1 for v in values if self._cell_looks_clean_numeric(v))
            if clean / len(values) >= 0.5:
                numeric_cols.add(col)

        if not numeric_cols:
            return headers, rows

        suspects = [
            (r_idx, col)
            for r_idx, row in enumerate(rows)
            for col in numeric_cols
            if col < len(row)
            and row[col]
            and not self._cell_looks_clean_numeric(row[col])
        ]

        if not suspects:
            return headers, rows

        try:
            pix = fitz_page.get_pixmap(dpi=400)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        except Exception:
            logger.debug(
                "Cell-repair render failed on page %d",
                fitz_page.number + 1,
                exc_info=True,
            )
            return headers, rows

        scale = 400 / 72.0
        repaired = 0

        for r_idx, col in suspects:
            table_row_idx = header_row_offset + 1 + r_idx
            if table_row_idx >= len(plumber_table.rows):
                continue

            cells = plumber_table.rows[table_row_idx].cells
            if col >= len(cells) or cells[col] is None:
                continue

            repaired_text = self._ocr_repair_bbox_value(img, scale, cells[col])
            if repaired_text is not None:
                rows[r_idx][col] = repaired_text
                repaired += 1

        if repaired:
            logger.info(
                "Repaired %d font-corrupted cell(s) on page %d via targeted OCR",
                repaired,
                fitz_page.number + 1,
            )

        return headers, rows

    # ---------------------------------------------------------

    def _clean_cell(self, value):

        if value is None:
            return ""

        value = str(value)

        value = re.sub(r"\s+", " ", value)

        return value.strip()

    # ---------------------------------------------------------
    # REMOVE HEADER / FOOTER
    # ---------------------------------------------------------

    def remove_headers_and_footers(self, pages):
        """
        Detect repeated header/footer text across pages and remove it.

        This dramatically improves embeddings and graph extraction because
        company names, page numbers and disclaimer text won't appear in
        every chunk.
        """

        if len(pages) < 2:
            return pages

        top_counter = {}
        bottom_counter = {}

        for page in pages:
            if page.text_blocks:
                top = page.text_blocks[0].text

                bottom = page.text_blocks[-1].text

                top_counter[top] = top_counter.get(top, 0) + 1

                bottom_counter[bottom] = bottom_counter.get(bottom, 0) + 1

        repeated_headers = {
            x for x, c in top_counter.items() if c >= max(2, len(pages) // 2)
        }

        repeated_footers = {
            x for x, c in bottom_counter.items() if c >= max(2, len(pages) // 2)
        }

        for page in pages:
            filtered = []

            for block in page.text_blocks:
                if block.text in repeated_headers:
                    continue

                if block.text in repeated_footers:
                    continue

                filtered.append(block)

            page.text_blocks = filtered

        return pages

    # ---------------------------------------------------------
    # BOILERPLATE REMOVAL (registered-office / CIN / contact-info blocks
    # — these repeat near-verbatim across pages but aren't always in the
    # exact same position, so the exact-match header/footer remover above
    # doesn't always catch them)
    # ---------------------------------------------------------

    def remove_boilerplate_blocks(self, pages):

        for page in pages:
            filtered = []

            for block in page.text_blocks:
                text = block.text

                if (
                    len(text) <= 300
                    and len(self.BOILERPLATE_PATTERN.findall(text)) >= 2
                ):
                    continue

                filtered.append(block)

            page.text_blocks = filtered

        return pages

    # ---------------------------------------------------------
    # BOILERPLATE TABLE REMOVAL (letterhead/contact-info cells — address,
    # phone, email, website — that pdfplumber's grid-finder sometimes
    # groups into a "table" the same way real ruled/whitespace-aligned
    # data is. remove_boilerplate_blocks() above only ever sees TEXT
    # blocks, so a table shaped like this slips past it entirely and
    # gets indexed as if it were financial data.
    # ---------------------------------------------------------

    # BOILERPLATE_PATTERN above requires an inline colon ("Email:
    # x@y.com") because it was built for prose text blocks. A letterhead
    # laid out as table CELLS often splits label and value into separate
    # cells instead — one cell just says "Phone", the next holds the
    # number, with no colon anywhere — so that pattern alone misses it.
    # This catches that shape: a cell whose content, on its own, IS one
    # of these labels rather than containing it inline.
    _BOILERPLATE_CELL_KEYWORDS = (
        "regd",
        "corporate office",
        "registered office",
        "cin",
        "website",
        "email",
        "phone",
        "fax",
        "tel",
        "gstin",
        "toll free",
    )

    def _is_boilerplate_table(self, table) -> bool:
        """Mirrors remove_boilerplate_blocks()'s text-block check —
        same BOILERPLATE_PATTERN, same >=2-match threshold — applied to
        a table's flattened cell text instead of a paragraph, plus a
        second check for the grid-cell shape above. Real financial
        tables (even short ones) run well past 300 chars once every
        row/column label and number is flattened together, so this only
        ever catches genuinely small, contact-info-shaped tables, not a
        legitimate table that happens to mention "email" once in a
        note."""
        cells = [
            str(c).strip()
            for row in ([table.headers] + list(table.rows))
            for c in row
            if c and str(c).strip()
        ]
        if not cells:
            return False

        flattened = " ".join(cells)

        if len(flattened) > 300:
            return False

        if len(self.BOILERPLATE_PATTERN.findall(flattened)) >= 2:
            return True

        keyword_cells = sum(
            1
            for c in cells
            if any(kw in c.lower() for kw in self._BOILERPLATE_CELL_KEYWORDS)
        )

        return keyword_cells >= 2

    def remove_boilerplate_tables(self, pages):
        for page in pages:
            page.tables = [t for t in page.tables if not self._is_boilerplate_table(t)]

        return pages

    # ---------------------------------------------------------
    # PAGE TYPE CLASSIFICATION (lets low-value boilerplate pages —
    # auditor legalese, security-cover certificates — be tagged and
    # optionally skipped instead of indexed alongside real financial data)
    # ---------------------------------------------------------

    def classify_page(self, page) -> str:

        text = " ".join(b.text for b in page.text_blocks).lower()

        scores = {
            ptype: sum(1 for kw in keywords if kw in text)
            for ptype, keywords in self.PAGE_TYPE_PATTERNS.items()
        }

        best_type, best_score = max(scores.items(), key=lambda kv: kv[1])

        return best_type if best_score >= 2 else "financial_statement"

    # ---------------------------------------------------------
    # TABLE TITLES
    # ---------------------------------------------------------

    def detect_table_titles(self, pages):
        """
        Finds text immediately above each table and stores it as
        table.title.
        """

        for page in pages:
            if not page.tables:
                continue

            for table in page.tables:
                if not table.bbox:
                    continue

                nearest = None

                nearest_distance = 999999

                for block in page.text_blocks:
                    if block.bbox[3] > table.bbox[1]:
                        continue

                    distance = table.bbox[1] - block.bbox[3]

                    if distance < nearest_distance:
                        nearest = block

                        nearest_distance = distance

                if nearest:
                    table.title = nearest.text

        return pages

    # ---------------------------------------------------------
    # TABLE TO TEXT
    # ---------------------------------------------------------

    _PURE_NUMBER_RE = re.compile(r"[-+(]?[\d,]*\.?\d*[)%]?")

    def _looks_like_period_label(self, value: str) -> bool:
        """True for short strings that name a reporting period — '30 Sep
        2025', 'FY26', 'Q2 FY26' — as opposed to an actual numeric data
        value like '1591.04' or '(500)'. The distinguishing signal is
        letters, not digit density: dates are digit-HEAVY ('30 Sep 2025'
        is 6 of 11 chars digits) but are still labels, not data — so a
        digit-ratio threshold misclassifies them. A pure number (however
        it's formatted: commas, decimals, %, parens for negatives) has no
        letters at all; a period label always does."""
        if not value:
            return False
        stripped = value.strip()
        if not stripped or len(stripped) > 25:
            return False
        if self._PURE_NUMBER_RE.fullmatch(stripped):
            return False  # e.g. "1,591.04", "(500)", "12%" — data, not a label
        return any(c.isalpha() for c in stripped)

    def _resolve_table_headers(self, headers, rows):
        """
        Financial statements routinely have a multi-row header — e.g. one
        merged "As at" cell spanning two date columns, with the actual
        dates ("30 Sep 2025" / "31 Mar 2025") on the row underneath it,
        and sometimes a THIRD row underneath that marking audit status
        ("Unaudited" / "Audited") per column. pdfplumber flattens all of
        this into the table body as ordinary rows, so without resolving
        it, table_to_text() has no way to tell which number belongs to
        which period — and, if only the first such row is consumed, the
        second (e.g. the audit-status row) leaks through as if it were
        real numeric data ("30 Jun 2025 = Unaudited").

        Consumes up to 2 leading rows as long as every non-empty cell in
        them looks like a label rather than data. The FIRST row only gets
        consumed if the header row itself was ambiguous (blank/duplicate
        labels — the actual signal that a multi-row header exists);
        without that gate, a genuinely simple table whose first real row
        happens to be all-text would get eaten by mistake. The second row
        has no such gate (headers are already resolved and thus no longer
        "ambiguous" by then) but is still bounded to just one extra row,
        keeping false positives rare.
        """
        if not headers or not rows:
            return headers, rows

        resolved = list(headers)
        remaining = list(rows)
        consumed = 0
        max_leading_label_rows = 2

        while remaining and consumed < max_leading_label_rows:
            candidate_row = remaining[0]
            non_empty_cells = [v for v in candidate_row if v]
            if not non_empty_cells or not all(
                self._looks_like_period_label(v) for v in non_empty_cells
            ):
                break

            if consumed == 0:
                non_blank = [h for h in resolved if h]
                has_ambiguous_headers = len(set(non_blank)) < len(non_blank) or any(
                    not h for h in resolved
                )
                if not has_ambiguous_headers:
                    break

            new_resolved = []
            for h, sub in zip(resolved, candidate_row):
                if not sub:
                    new_resolved.append(h)
                elif consumed == 0:
                    new_resolved.append(sub)
                elif h and h != sub:
                    new_resolved.append(f"{h} ({sub})")
                else:
                    new_resolved.append(sub)
            resolved = new_resolved
            remaining = remaining[1:]
            consumed += 1

        return resolved, remaining

    def table_to_text(self, table):
        """
        Generates embedding-friendly text while preserving the
        structured table.

        Tables with one label column and 2+ value columns (the shape
        every financial-statement balance sheet/P&L table has — metric
        name + one column per reporting period) are rendered as one block
        per row, each period on its own "label = value" line, instead of
        a single "; "-joined sentence. Flattening a multi-period row into
        one line forces the LLM to mentally re-align each number back to
        the header it belongs to; that's exactly where extraction on
        these tables went wrong. Simple 2-column key/value tables keep
        the original compact inline format, since there's no
        period-alignment ambiguity to resolve there.
        """

        lines = []

        if table.title:
            lines.append(table.title)

        headers, rows = self._resolve_table_headers(table.headers, table.rows)

        if headers:
            lines.append("Columns: " + ", ".join(h for h in headers if h))

        multi_period = len(headers) > 2

        for row in rows:
            if not any(row):
                continue

            if multi_period:
                label = row[0] if row else ""
                period_lines = [
                    f"{h} = {value}" for h, value in zip(headers[1:], row[1:]) if value
                ]
                if not (label or period_lines):
                    continue
                block = ([label] if label else []) + period_lines
                lines.append("\n".join(block))
            else:
                pairs = [f"{h}: {value}" for h, value in zip(headers, row) if value]
                if pairs:
                    lines.append("; ".join(pairs))

        return "\n\n".join(lines) if multi_period else "\n".join(lines)

    # ---------------------------------------------------------
    # PAGE TO JSON
    # ---------------------------------------------------------

    def page_to_json(self, page):

        page_json = {"page": page.page, "content": []}

        for block in page.text_blocks:
            page_json["content"].append(
                {
                    "type": "heading" if block.is_heading else "text",
                    "text": block.text,
                    "bbox": block.bbox,
                    "font_size": block.font_size,
                }
            )

        for table in page.tables:
            page_json["content"].append(
                {
                    "type": "table",
                    "title": table.title,
                    "headers": table.headers,
                    "rows": table.rows,
                    "embedding_text": self.table_to_text(table),
                }
            )

        return page_json

    # ---------------------------------------------------------
    # EXPORT
    # ---------------------------------------------------------

    def export(self, pdf_path):
        """
        Main public API.

        Returns structured JSON ready for chunking.
        """

        pages = self.process(pdf_path)

        pages = self.remove_headers_and_footers(pages)

        pages = self.detect_table_titles(pages)

        output = []

        for page in pages:
            output.append(self.page_to_json(page))

        return output

    # ---------------------------------------------------------
    # SECTION DETECTION (normalizes raw heading text to a
    # canonical label instead of storing it verbatim)
    # ---------------------------------------------------------

    def assign_sections(self, pages):

        current_section = "Unknown"

        for page in pages:
            for block in page.text_blocks:
                txt = block.text.lower()

                for pattern, canonical in self.SECTION_PATTERNS.items():
                    if pattern in txt:
                        current_section = canonical

                        break

                block.section = current_section

        return pages

    # ---------------------------------------------------------
    # DUPLICATE REMOVAL
    # ---------------------------------------------------------

    def remove_duplicate_blocks(self, pages, window=3):
        """
        Removes exact-duplicate text blocks, but only within a nearby
        window of pages — not the whole document. The old version used a
        single `seen` set spanning every page, so any block whose text
        happened to match one seen ANYWHERE earlier (even 50 pages back)
        got silently dropped. Two problems with that:

        1. Headings are structural, not noise. A report can legitimately
           reuse the same heading text twice (e.g. "Balance Sheet" for
           Standalone results, then again for Consolidated results later
           in the same PDF). Deleting the second occurrence removes the
           heading `chunker.py` needs to correctly restart section/heading
           tracking — every chunk after it then inherits the WRONG
           section/heading from wherever tracking last left off, silently
           mislabeling citations, even though the actual numbers/text in
           those chunks are untouched and still fully correct.
        2. Non-heading content that happens to repeat far apart (a
           disclaimer sentence in the intro and again near the end, say)
           is more likely a coincidence than true boilerplate — collapsing
           it discards real content for no real gain, when
           remove_headers_and_footers()/remove_boilerplate_blocks() have
           already handled the actual repeated cruft earlier in the
           pipeline.

        `window` pages of lookback keeps this catching genuinely-local
        repeats (e.g. a caption line repeated on consecutive pages of a
        multi-page table) while leaving headings alone entirely and no
        longer nuking unrelated far-apart content.
        """

        recent_seen = {}  # text -> page list-index it was last kept at

        for page_index, page in enumerate(pages):
            filtered = []

            for block in page.text_blocks:
                if block.is_heading:
                    # Headings are never deduped — see rationale above.
                    filtered.append(block)
                    continue

                key = block.text.strip()
                last_seen_index = recent_seen.get(key)

                if (
                    last_seen_index is not None
                    and (page_index - last_seen_index) <= window
                ):
                    continue  # duplicate of a NEARBY page — drop it

                recent_seen[key] = page_index
                filtered.append(block)

            page.text_blocks = filtered

        return pages

    # ---------------------------------------------------------
    # NORMALIZE TABLES
    # ---------------------------------------------------------

    def normalize_tables(self, pages):
        """
        Cleans empty rows and columns.
        """

        for page in pages:
            for table in page.tables:
                rows = []

                for row in table.rows:
                    cleaned = []

                    for cell in row:
                        cleaned.append(self._clean_cell(cell))

                    if any(cleaned):
                        rows.append(cleaned)

                table.rows = rows

        return pages

    # ---------------------------------------------------------
    # READING ORDER (sorts by y, then x, so multi-column layouts
    # don't interleave left/right columns out of order)
    # ---------------------------------------------------------

    def merge_reading_order(self, page):
        """
        Merge text and tables into reading order.

        Returned list contains dictionaries.
        """

        items = []

        for block in page.text_blocks:
            items.append(
                {
                    "y": block.bbox[1],
                    "x": block.bbox[0],
                    "kind": "text",
                    "object": block,
                }
            )

        for table in page.tables:
            y = table.bbox[1] if table.bbox else 999999
            x = table.bbox[0] if table.bbox else 999999

            items.append({"y": y, "x": x, "kind": "table", "object": table})

        items.sort(key=lambda item: (item["y"], item["x"]))

        return items

    # ---------------------------------------------------------
    # MAIN EXTRACTION
    # ---------------------------------------------------------

    def extract(self, pdf_path):

        pages = self.process(pdf_path)

        pages = self.remove_headers_and_footers(pages)

        pages = self.remove_boilerplate_blocks(pages)

        pages = self.remove_boilerplate_tables(pages)

        pages = self.remove_duplicate_blocks(pages)

        pages = self.assign_sections(pages)

        pages = self.normalize_tables(pages)

        pages = self.detect_table_titles(pages)

        output = []

        for page in pages:
            page_type = self.classify_page(page)

            if page_type in config.SKIP_PAGE_TYPES:
                logger.info("Skipping page %d (page_type=%s)", page.page, page_type)
                continue

            merged = self.merge_reading_order(page)

            page_json = {"page": page.page, "page_type": page_type, "content": []}

            for item in merged:
                if item["kind"] == "text":
                    block = item["object"]

                    page_json["content"].append(
                        {
                            "type": "heading" if block.is_heading else "text",
                            "section": block.section or "Unknown",
                            "text": block.text,
                            "bbox": block.bbox,
                            "font_size": block.font_size,
                        }
                    )

                else:
                    table = item["object"]

                    page_json["content"].append(
                        {
                            "type": "table",
                            "title": table.title,
                            "headers": table.headers,
                            "rows": table.rows,
                            "embedding_text": self.table_to_text(table),
                        }
                    )

            output.append(page_json)

        return output


# ---------------------------------------------------------
# Convenience API
# ---------------------------------------------------------

_processor = PDFProcessor()


def extract_pdf(pdf_path):
    """
    Public function used by data_processing.py
    """

    return _processor.extract(pdf_path)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("Usage:")

        print("python pdf_processor.py report.pdf")

        raise SystemExit(1)

    result = extract_pdf(sys.argv[1])

    print(json.dumps(result, indent=2))
