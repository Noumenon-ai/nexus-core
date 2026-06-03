"""nexus-pdf-docs MCP server — Phase 2.5a Server 2.

PDF generation, manipulation, form-filling. Reuses Phase 4 path safety boundary.

Dependencies:
  - reportlab    PDF generation
  - pypdf        merge/split/form-fill/page operations
  - pypdfium2    render PDF pages to PNG (pure-python wheel, no poppler needed)
  - pdfplumber   text + metadata extraction (already used by nexus-filesystem)
  - Pillow       signature image overlay

Templates live in /home/user/nexus_templates/ (configurable via NEXUS_PDF_TEMPLATES_DIR).
Each tool returns a dict; failures return {'ok': False, 'reason': str, 'detail': str}.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from services.file_system_tools import FileSystemSecurityError, _validate_path
from mcp_servers.nexus_filesystem import _validate_write_target


mcp = FastMCP("nexus-pdf-docs")

_TEMPLATES_DIR = Path(os.environ.get("NEXUS_PDF_TEMPLATES_DIR", "/home/user/nexus_templates"))
_PDF_MAX_BYTES = 50 * 1024 * 1024     # 50 MB hard cap for inputs
_PDF_TO_IMAGE_MAX_PAGES = 50          # don't render more than 50 pages at once


def _ensure_pdf_input(path: str) -> Path | dict:
    """Validate that path is a readable PDF inside ALLOWED_ROOTS. Returns Path or error dict."""
    try:
        resolved = _validate_path(path, must_exist=True)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "path_denied", "detail": str(exc)}
    if not resolved.is_file():
        return {"ok": False, "reason": "not_a_file", "detail": str(resolved)}
    if resolved.suffix.lower() != ".pdf":
        return {"ok": False, "reason": "not_a_pdf", "detail": resolved.name}
    if resolved.stat().st_size > _PDF_MAX_BYTES:
        return {"ok": False, "reason": "input_too_large", "detail": f"{resolved.stat().st_size} bytes"}
    return resolved


def _ensure_pdf_output(path: str) -> Path | dict:
    """Validate that path is a permissible write target with .pdf extension."""
    try:
        final = _validate_write_target(path)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "output_path_denied", "detail": str(exc)}
    if final.suffix.lower() != ".pdf":
        return {"ok": False, "reason": "output_must_be_pdf", "detail": final.name}
    return final


# ---------- generation ----------

@mcp.tool(meta={"approval_required": True, "approval_template": "Generate PDF from text?"})
def create_pdf_from_text(content: str, output_path: str, options: dict | None = None) -> dict:
    """Generate a PDF from text. Options: font_name, font_size, page_size ('letter'|'a4'), margin_pt."""
    if not isinstance(content, str) or not content:
        return {"ok": False, "reason": "empty_content", "detail": ""}
    if len(content) > 500_000:
        return {"ok": False, "reason": "content_too_long", "detail": f"{len(content)} chars"}
    out = _ensure_pdf_output(output_path)
    if isinstance(out, dict):
        return out
    opts = options or {}
    font_name = opts.get("font_name", "Helvetica")
    font_size = float(opts.get("font_size", 11))
    page_size_name = (opts.get("page_size") or "letter").lower()
    margin_pt = float(opts.get("margin_pt", 54))

    from reportlab.lib.pagesizes import LETTER, A4
    from reportlab.pdfgen import canvas
    page_size = A4 if page_size_name == "a4" else LETTER

    c = canvas.Canvas(str(out), pagesize=page_size)
    c.setFont(font_name, font_size)
    width, height = page_size
    y = height - margin_pt
    line_h = font_size * 1.3
    for raw_line in content.splitlines():
        # Wrap long lines.
        line = raw_line
        max_w = width - 2 * margin_pt
        while line:
            # crude width budget — split at 1.6 chars/pt heuristic
            chars_per_pt = 1.6
            approx_chars = max(10, int(max_w * chars_per_pt / font_size))
            chunk, line = line[:approx_chars], line[approx_chars:]
            c.drawString(margin_pt, y, chunk)
            y -= line_h
            if y < margin_pt:
                c.showPage()
                c.setFont(font_name, font_size)
                y = height - margin_pt
    c.save()
    return {"ok": True, "path": str(out), "size_bytes": out.stat().st_size, "page_size": page_size_name}


@mcp.tool()
def list_templates() -> dict:
    """List PDF templates available in NEXUS_PDF_TEMPLATES_DIR."""
    if not _TEMPLATES_DIR.exists():
        return {"ok": True, "templates_dir": str(_TEMPLATES_DIR), "templates": [], "warning": "templates_dir_missing"}
    if not _TEMPLATES_DIR.is_dir():
        return {"ok": False, "reason": "templates_dir_not_a_directory", "detail": str(_TEMPLATES_DIR)}
    templates = []
    for entry in sorted(_TEMPLATES_DIR.iterdir()):
        if entry.suffix.lower() == ".pdf" and entry.is_file():
            templates.append({
                "name": entry.stem,
                "filename": entry.name,
                "size_bytes": entry.stat().st_size,
            })
    return {"ok": True, "templates_dir": str(_TEMPLATES_DIR), "count": len(templates), "templates": templates}


@mcp.tool(meta={"approval_required": True, "approval_template": "Generate PDF from template?"})
def create_pdf_from_template(template_name: str, fields: dict, output_path: str) -> dict:
    """Fill a named PDF template's AcroForm fields and write to output_path."""
    if not isinstance(template_name, str) or not template_name.strip():
        return {"ok": False, "reason": "empty_template_name", "detail": ""}
    if not isinstance(fields, dict):
        return {"ok": False, "reason": "fields_must_be_dict", "detail": repr(type(fields).__name__)}
    template_path = _TEMPLATES_DIR / f"{template_name}.pdf"
    if not template_path.exists():
        return {"ok": False, "reason": "template_not_found", "detail": str(template_path)}
    return fill_pdf_form(input_path=str(template_path), fields=fields, output_path=output_path)


# ---------- forms ----------

@mcp.tool()
def extract_pdf_form_fields(path: str) -> dict:
    """Return AcroForm field names + types for a PDF."""
    inp = _ensure_pdf_input(path)
    if isinstance(inp, dict):
        return inp
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        return {"ok": False, "reason": "pypdf_not_installed", "detail": str(exc)}
    try:
        reader = PdfReader(str(inp))
        fields = reader.get_fields() or {}
    except Exception as exc:
        return {"ok": False, "reason": "pdf_read_failed", "detail": str(exc)[:300]}
    if not fields:
        return {"ok": True, "path": str(inp), "has_form": False, "fields": []}
    out = []
    for name, info in fields.items():
        out.append({
            "name": name,
            "type": str(info.get("/FT", "")) if info else "",
            "value": str(info.get("/V", "")) if info else "",
        })
    return {"ok": True, "path": str(inp), "has_form": True, "field_count": len(out), "fields": out}


@mcp.tool(meta={"approval_required": True, "approval_template": "Fill PDF form?"})
def fill_pdf_form(input_path: str, fields: dict, output_path: str) -> dict:
    """Fill AcroForm fields in input_path PDF and save to output_path."""
    if not isinstance(fields, dict) or not fields:
        return {"ok": False, "reason": "empty_or_invalid_fields", "detail": "fields must be a non-empty dict"}
    inp = _ensure_pdf_input(input_path)
    if isinstance(inp, dict):
        return inp
    out = _ensure_pdf_output(output_path)
    if isinstance(out, dict):
        return out
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:
        return {"ok": False, "reason": "pypdf_not_installed", "detail": str(exc)}
    try:
        reader = PdfReader(str(inp))
        writer = PdfWriter(clone_from=reader)
        for page in writer.pages:
            writer.update_page_form_field_values(page, fields)
        with open(out, "wb") as f:
            writer.write(f)
    except Exception as exc:
        return {"ok": False, "reason": "fill_failed", "detail": str(exc)[:300]}
    return {"ok": True, "path": str(out), "size_bytes": out.stat().st_size, "fields_attempted": list(fields.keys())}


# ---------- merge / split ----------

@mcp.tool(meta={"approval_required": True, "approval_template": "Merge PDFs?"})
def merge_pdfs(paths: list[str], output_path: str) -> dict:
    """Concatenate multiple PDFs into output_path, in order."""
    if not isinstance(paths, list) or len(paths) < 1:
        return {"ok": False, "reason": "empty_paths_list", "detail": ""}
    if len(paths) > 100:
        return {"ok": False, "reason": "too_many_inputs", "detail": f"{len(paths)} (max 100)"}
    inputs: list[Path] = []
    for p in paths:
        v = _ensure_pdf_input(p)
        if isinstance(v, dict):
            return {"ok": False, "reason": "input_rejected", "detail": v.get("detail", ""), "rejected_path": p}
        inputs.append(v)
    out = _ensure_pdf_output(output_path)
    if isinstance(out, dict):
        return out
    try:
        from pypdf import PdfWriter
    except ImportError as exc:
        return {"ok": False, "reason": "pypdf_not_installed", "detail": str(exc)}
    try:
        writer = PdfWriter()
        for src in inputs:
            writer.append(str(src))
        with open(out, "wb") as f:
            writer.write(f)
    except Exception as exc:
        return {"ok": False, "reason": "merge_failed", "detail": str(exc)[:300]}
    return {"ok": True, "path": str(out), "size_bytes": out.stat().st_size, "input_count": len(inputs)}


def _parse_ranges(ranges: list[str], page_count: int) -> list[tuple[int, int]]:
    """Parse human-style ranges like ['1-3', '5', '7-9'] (1-based, inclusive) to 0-based half-open pairs."""
    out: list[tuple[int, int]] = []
    for spec in ranges:
        if not isinstance(spec, str):
            raise ValueError(f"range must be a string, got {type(spec).__name__}")
        s = spec.strip()
        if "-" in s:
            a, b = s.split("-", 1)
            start = int(a)
            end = int(b)
        else:
            start = end = int(s)
        if start < 1 or end < 1 or start > page_count or end > page_count or start > end:
            raise ValueError(f"out-of-bounds range {spec!r} for page_count={page_count}")
        out.append((start - 1, end))
    return out


@mcp.tool(meta={"approval_required": True, "approval_template": "Split PDF?"})
def split_pdf(path: str, ranges: list[str], output_dir: str) -> dict:
    """Split a PDF into segments. ranges = ['1-3','5','8-10']. One file per range, named <stem>__<i>.pdf in output_dir."""
    inp = _ensure_pdf_input(path)
    if isinstance(inp, dict):
        return inp
    if not isinstance(ranges, list) or not ranges:
        return {"ok": False, "reason": "empty_ranges", "detail": ""}
    try:
        out_dir = _validate_path(output_dir, must_exist=True)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "output_dir_denied", "detail": str(exc)}
    if not out_dir.is_dir():
        return {"ok": False, "reason": "output_dir_not_a_directory", "detail": str(out_dir)}
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:
        return {"ok": False, "reason": "pypdf_not_installed", "detail": str(exc)}
    try:
        reader = PdfReader(str(inp))
        pairs = _parse_ranges(ranges, len(reader.pages))
    except ValueError as exc:
        return {"ok": False, "reason": "invalid_ranges", "detail": str(exc)}
    out_paths: list[str] = []
    try:
        for i, (start, end) in enumerate(pairs, 1):
            writer = PdfWriter()
            for p_idx in range(start, end):
                writer.add_page(reader.pages[p_idx])
            out_path = out_dir / f"{inp.stem}__{i}.pdf"
            with open(out_path, "wb") as f:
                writer.write(f)
            out_paths.append(str(out_path))
    except Exception as exc:
        return {"ok": False, "reason": "split_failed", "detail": str(exc)[:300]}
    return {"ok": True, "input": str(inp), "output_dir": str(out_dir), "outputs": out_paths, "count": len(out_paths)}


# ---------- render ----------

@mcp.tool()
def pdf_to_images(path: str, output_dir: str, dpi: int = 150) -> dict:
    """Render each PDF page as PNG via pypdfium2. Caps at 50 pages per call."""
    inp = _ensure_pdf_input(path)
    if isinstance(inp, dict):
        return inp
    try:
        dpi = int(dpi)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_dpi", "detail": repr(dpi)}
    if dpi < 36 or dpi > 600:
        return {"ok": False, "reason": "dpi_out_of_range", "detail": "dpi must be 36..600"}
    try:
        out_dir = _validate_path(output_dir, must_exist=True)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "output_dir_denied", "detail": str(exc)}
    if not out_dir.is_dir():
        return {"ok": False, "reason": "output_dir_not_a_directory", "detail": str(out_dir)}
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        return {"ok": False, "reason": "pypdfium2_not_installed", "detail": str(exc)}
    pdf = pdfium.PdfDocument(str(inp))
    if len(pdf) > _PDF_TO_IMAGE_MAX_PAGES:
        return {"ok": False, "reason": "too_many_pages", "detail": f"{len(pdf)} pages (cap {_PDF_TO_IMAGE_MAX_PAGES})"}
    out_paths: list[str] = []
    scale = dpi / 72.0
    try:
        for i, page in enumerate(pdf, 1):
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            out_path = out_dir / f"{inp.stem}__page{i:03d}.png"
            pil_image.save(out_path, "PNG")
            out_paths.append(str(out_path))
    except Exception as exc:
        return {"ok": False, "reason": "render_failed", "detail": str(exc)[:300]}
    return {"ok": True, "input": str(inp), "output_dir": str(out_dir), "page_count": len(out_paths), "outputs": out_paths, "dpi": dpi}


# ---------- signature overlay ----------

@mcp.tool(meta={"approval_required": True, "approval_template": "Add signature to PDF?"})
def add_signature_to_pdf(path: str, signature_image_path: str, location: dict, output_path: str) -> dict:
    """Overlay a signature image on a specific page + (x,y) of a PDF. Visual overlay only — NOT cryptographic.
    location = {'page': 1-based-int, 'x': pt, 'y': pt, 'width': pt, 'height': pt}."""
    inp = _ensure_pdf_input(path)
    if isinstance(inp, dict):
        return inp
    try:
        sig = _validate_path(signature_image_path, must_exist=True)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "signature_path_denied", "detail": str(exc)}
    if not sig.is_file():
        return {"ok": False, "reason": "signature_not_a_file", "detail": str(sig)}
    out = _ensure_pdf_output(output_path)
    if isinstance(out, dict):
        return out
    if not isinstance(location, dict):
        return {"ok": False, "reason": "invalid_location", "detail": "must be dict"}
    try:
        page_num = int(location["page"])
        x = float(location["x"])
        y = float(location["y"])
        w = float(location["width"])
        h = float(location["height"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "reason": "invalid_location_fields", "detail": str(exc)}
    if w <= 0 or h <= 0:
        return {"ok": False, "reason": "non_positive_dimensions", "detail": f"{w}x{h}"}

    try:
        from pypdf import PdfReader, PdfWriter
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        return {"ok": False, "reason": "missing_dependency", "detail": str(exc)}

    try:
        reader = PdfReader(str(inp))
    except Exception as exc:
        return {"ok": False, "reason": "pdf_read_failed", "detail": str(exc)[:300]}
    if page_num < 1 or page_num > len(reader.pages):
        return {"ok": False, "reason": "page_out_of_bounds", "detail": f"page {page_num} of {len(reader.pages)}"}
    src_page = reader.pages[page_num - 1]
    page_width = float(src_page.mediabox.width)
    page_height = float(src_page.mediabox.height)

    overlay_buf = io.BytesIO()
    c = canvas.Canvas(overlay_buf, pagesize=(page_width, page_height))
    try:
        c.drawImage(str(sig), x, y, width=w, height=h, mask='auto')
    except Exception as exc:
        return {"ok": False, "reason": "overlay_failed", "detail": str(exc)[:300]}
    c.save()
    overlay_buf.seek(0)
    overlay_reader = PdfReader(overlay_buf)
    overlay_page = overlay_reader.pages[0]

    # Build writer first so the target page is owned by the writer when we merge.
    writer = PdfWriter(clone_from=reader)
    writer.pages[page_num - 1].merge_page(overlay_page)
    with open(out, "wb") as f:
        writer.write(f)
    return {"ok": True, "path": str(out), "page": page_num, "signature": str(sig), "location": location}


# ---------- extraction ----------

@mcp.tool()
def pdf_word_count(path: str) -> dict:
    """Word + character + page count for a PDF."""
    inp = _ensure_pdf_input(path)
    if isinstance(inp, dict):
        return inp
    try:
        import pdfplumber
    except ImportError as exc:
        return {"ok": False, "reason": "pdfplumber_not_installed", "detail": str(exc)}
    words = 0
    chars = 0
    try:
        with pdfplumber.open(inp) as pdf:
            for p in pdf.pages:
                text = p.extract_text() or ""
                chars += len(text)
                words += len(text.split())
            page_count = len(pdf.pages)
    except Exception as exc:
        return {"ok": False, "reason": "pdf_extract_failed", "detail": str(exc)[:300]}
    return {"ok": True, "path": str(inp), "page_count": page_count, "word_count": words, "char_count": chars}


@mcp.tool()
def pdf_summary(path: str, first_page_chars: int = 1000) -> dict:
    """Return PDF metadata + first-page text preview + page count."""
    inp = _ensure_pdf_input(path)
    if isinstance(inp, dict):
        return inp
    try:
        first_page_chars = int(first_page_chars)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_preview_size", "detail": ""}
    first_page_chars = max(0, min(first_page_chars, 10_000))
    try:
        import pdfplumber
    except ImportError as exc:
        return {"ok": False, "reason": "pdfplumber_not_installed", "detail": str(exc)}
    try:
        with pdfplumber.open(inp) as pdf:
            page_count = len(pdf.pages)
            metadata = dict(pdf.metadata or {})
            first_text = (pdf.pages[0].extract_text() or "") if pdf.pages else ""
    except Exception as exc:
        return {"ok": False, "reason": "pdf_extract_failed", "detail": str(exc)[:300]}
    # Strip bytes values for JSON-safety.
    safe_meta = {k: (v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v) for k, v in metadata.items()}
    return {
        "ok": True,
        "path": str(inp),
        "page_count": page_count,
        "metadata": safe_meta,
        "first_page_preview": first_text[:first_page_chars],
    }


@mcp.tool()
def pdf_to_text(path: str, output_path: str | None = None) -> dict:
    """Extract all text from a PDF. If output_path given, also save to .txt."""
    inp = _ensure_pdf_input(path)
    if isinstance(inp, dict):
        return inp
    try:
        import pdfplumber
    except ImportError as exc:
        return {"ok": False, "reason": "pdfplumber_not_installed", "detail": str(exc)}
    pages: list[str] = []
    try:
        with pdfplumber.open(inp) as pdf:
            for p in pdf.pages:
                pages.append(p.extract_text() or "")
    except Exception as exc:
        return {"ok": False, "reason": "pdf_extract_failed", "detail": str(exc)[:300]}
    text = "\n\n".join(pages)
    saved_path: str | None = None
    if output_path:
        try:
            from mcp_servers.nexus_filesystem import _validate_write_target
            target = _validate_write_target(output_path)
        except FileSystemSecurityError as exc:
            return {"ok": False, "reason": "output_path_denied", "detail": str(exc)}
        if target.suffix.lower() not in {".txt", ".md"}:
            return {"ok": False, "reason": "output_must_be_text", "detail": target.suffix}
        try:
            target.write_text(text, encoding="utf-8")
            saved_path = str(target)
        except OSError as exc:
            return {"ok": False, "reason": "write_failed", "detail": str(exc)[:200]}
    return {"ok": True, "path": str(inp), "page_count": len(pages), "char_count": len(text), "text": text, "saved_to": saved_path}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
