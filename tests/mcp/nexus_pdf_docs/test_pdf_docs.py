"""Adversarial tests for nexus-pdf-docs MCP server (Phase 2.5a Server 2)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


SANDBOX_PREFIX = "_mcp_pdf_test_"


def _allow_tmp_workspace(monkeypatch, tmp_path: Path) -> Path:
    workspace = tmp_path.resolve()
    monkeypatch.setattr("services.file_system_tools.ALLOWED_ROOTS", (workspace,))
    monkeypatch.setattr("mcp_servers.nexus_filesystem.ALLOWED_ROOTS", (workspace,))
    return workspace


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    workspace = _allow_tmp_workspace(monkeypatch, tmp_path)
    d = workspace / SANDBOX_PREFIX.rstrip("_")
    d.mkdir()
    return d


@pytest.fixture
def templates_dir(tmp_path_factory, monkeypatch):
    """Isolate templates dir per test inside ALLOWED_ROOTS."""
    base = tmp_path_factory.mktemp("mcp_pdf_templates").resolve()
    monkeypatch.setattr("mcp_servers.nexus_pdf_docs._TEMPLATES_DIR", base)
    return base


from mcp_servers.nexus_pdf_docs import (
    add_signature_to_pdf,
    create_pdf_from_template,
    create_pdf_from_text,
    extract_pdf_form_fields,
    fill_pdf_form,
    list_templates,
    merge_pdfs,
    pdf_summary,
    pdf_to_images,
    pdf_to_text,
    pdf_word_count,
    split_pdf,
)


def _make_simple_pdf(target: Path, text: str = "Hello PDF") -> None:
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(target))
    c.drawString(72, 720, text)
    c.showPage()
    c.save()


def _make_multipage_pdf(target: Path, page_count: int = 3) -> None:
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(target))
    for i in range(page_count):
        c.drawString(72, 720, f"Page {i+1}")
        c.showPage()
    c.save()


# ---------- create_pdf_from_text ----------

def test_create_pdf_from_text_empty_content_rejected(sandbox):
    r = create_pdf_from_text(content="", output_path=str(sandbox / "x.pdf"))
    assert r["ok"] is False
    assert r["reason"] == "empty_content"


def test_create_pdf_from_text_output_outside_root_rejected():
    r = create_pdf_from_text(content="hi", output_path="/etc/x.pdf")
    assert r["ok"] is False


def test_create_pdf_from_text_non_pdf_extension_rejected(sandbox):
    r = create_pdf_from_text(content="hi", output_path=str(sandbox / "x.txt"))
    assert r["ok"] is False
    assert r["reason"] == "output_must_be_pdf"


def test_create_pdf_from_text_happy(sandbox):
    out = sandbox / "doc.pdf"
    r = create_pdf_from_text(content="Hello\nWorld", output_path=str(out))
    assert r["ok"] is True
    assert out.exists()
    assert out.stat().st_size > 200
    with open(out, "rb") as f:
        assert f.read(4) == b"%PDF"


def test_create_pdf_from_text_content_too_long_rejected(sandbox):
    r = create_pdf_from_text(content="x" * 600_000, output_path=str(sandbox / "x.pdf"))
    assert r["ok"] is False
    assert r["reason"] == "content_too_long"


# ---------- list_templates / create_pdf_from_template ----------

def test_list_templates_missing_dir_returns_empty(templates_dir):
    shutil.rmtree(templates_dir, ignore_errors=True)
    r = list_templates()
    assert r["ok"] is True
    assert r["templates"] == []
    assert r.get("warning") == "templates_dir_missing"


def test_list_templates_finds_pdfs(templates_dir):
    _make_simple_pdf(templates_dir / "lease.pdf")
    _make_simple_pdf(templates_dir / "invoice.pdf")
    (templates_dir / "notes.txt").write_text("ignored")
    r = list_templates()
    assert r["ok"] is True
    names = {t["name"] for t in r["templates"]}
    assert names == {"lease", "invoice"}


def test_create_pdf_from_template_missing_template_rejected(templates_dir, sandbox):
    r = create_pdf_from_template(template_name="nonexistent", fields={}, output_path=str(sandbox / "out.pdf"))
    assert r["ok"] is False
    assert r["reason"] == "template_not_found"


def test_create_pdf_from_template_invalid_fields_rejected(templates_dir, sandbox):
    _make_simple_pdf(templates_dir / "demo.pdf")
    r = create_pdf_from_template(template_name="demo", fields="not_a_dict", output_path=str(sandbox / "out.pdf"))
    assert r["ok"] is False
    assert r["reason"] == "fields_must_be_dict"


# ---------- form fields ----------

def test_extract_pdf_form_fields_no_form(sandbox):
    target = sandbox / "noform.pdf"
    _make_simple_pdf(target)
    r = extract_pdf_form_fields(path=str(target))
    assert r["ok"] is True
    assert r["has_form"] is False


def test_extract_pdf_form_fields_outside_root_rejected():
    r = extract_pdf_form_fields(path="/etc/something.pdf")
    assert r["ok"] is False
    assert r["reason"] == "path_denied"


def test_fill_pdf_form_empty_fields_rejected(sandbox):
    target = sandbox / "noform.pdf"
    _make_simple_pdf(target)
    r = fill_pdf_form(input_path=str(target), fields={}, output_path=str(sandbox / "out.pdf"))
    assert r["ok"] is False
    assert r["reason"] == "empty_or_invalid_fields"


# ---------- merge ----------

def test_merge_pdfs_happy(sandbox):
    a = sandbox / "a.pdf"; _make_simple_pdf(a, "AAA")
    b = sandbox / "b.pdf"; _make_simple_pdf(b, "BBB")
    out = sandbox / "merged.pdf"
    r = merge_pdfs(paths=[str(a), str(b)], output_path=str(out))
    assert r["ok"] is True
    assert out.exists()


def test_merge_pdfs_empty_list_rejected(sandbox):
    r = merge_pdfs(paths=[], output_path=str(sandbox / "out.pdf"))
    assert r["ok"] is False
    assert r["reason"] == "empty_paths_list"


def test_merge_pdfs_too_many_inputs_rejected(sandbox):
    r = merge_pdfs(paths=["x"] * 200, output_path=str(sandbox / "out.pdf"))
    assert r["ok"] is False
    assert r["reason"] == "too_many_inputs"


def test_merge_pdfs_outside_root_input_rejected(sandbox):
    r = merge_pdfs(paths=["/etc/somefile.pdf"], output_path=str(sandbox / "out.pdf"))
    assert r["ok"] is False
    assert r["reason"] == "input_rejected"


# ---------- split ----------

def test_split_pdf_happy(sandbox):
    src = sandbox / "multi.pdf"
    _make_multipage_pdf(src, page_count=5)
    out_dir = sandbox / "split_out"
    out_dir.mkdir()
    r = split_pdf(path=str(src), ranges=["1-2", "4"], output_dir=str(out_dir))
    assert r["ok"] is True
    assert r["count"] == 2
    assert all(Path(p).exists() for p in r["outputs"])


def test_split_pdf_invalid_range_rejected(sandbox):
    src = sandbox / "multi.pdf"
    _make_multipage_pdf(src, page_count=3)
    out_dir = sandbox / "split_out"
    out_dir.mkdir()
    r = split_pdf(path=str(src), ranges=["1-99"], output_dir=str(out_dir))
    assert r["ok"] is False
    assert r["reason"] == "invalid_ranges"


def test_split_pdf_empty_ranges_rejected(sandbox):
    src = sandbox / "multi.pdf"; _make_multipage_pdf(src, page_count=3)
    out_dir = sandbox / "split_out"; out_dir.mkdir()
    r = split_pdf(path=str(src), ranges=[], output_dir=str(out_dir))
    assert r["ok"] is False
    assert r["reason"] == "empty_ranges"


def test_split_pdf_output_dir_outside_root_rejected(sandbox):
    src = sandbox / "multi.pdf"; _make_multipage_pdf(src, page_count=3)
    r = split_pdf(path=str(src), ranges=["1-2"], output_dir="/etc")
    assert r["ok"] is False
    assert r["reason"] == "output_dir_denied"


# ---------- render ----------

def test_pdf_to_images_happy(sandbox):
    src = sandbox / "render.pdf"
    _make_simple_pdf(src)
    out_dir = sandbox / "img_out"
    out_dir.mkdir()
    r = pdf_to_images(path=str(src), output_dir=str(out_dir), dpi=72)
    assert r["ok"] is True
    assert r["page_count"] >= 1
    for p in r["outputs"]:
        assert Path(p).exists()
        with open(p, "rb") as f:
            assert f.read(4) == b"\x89PNG"


def test_pdf_to_images_dpi_out_of_range_rejected(sandbox):
    src = sandbox / "x.pdf"; _make_simple_pdf(src)
    out_dir = sandbox / "out"; out_dir.mkdir()
    r = pdf_to_images(path=str(src), output_dir=str(out_dir), dpi=10)
    assert r["ok"] is False


# ---------- signature overlay ----------

def test_add_signature_happy(sandbox):
    src = sandbox / "doc.pdf"; _make_simple_pdf(src)
    from PIL import Image
    sig_path = sandbox / "sig.png"
    Image.new("RGBA", (100, 40), (0, 0, 0, 0)).save(sig_path)
    out = sandbox / "signed.pdf"
    r = add_signature_to_pdf(
        path=str(src),
        signature_image_path=str(sig_path),
        location={"page": 1, "x": 100, "y": 100, "width": 100, "height": 40},
        output_path=str(out),
    )
    assert r["ok"] is True
    assert out.exists()


def test_add_signature_invalid_page_rejected(sandbox):
    src = sandbox / "doc.pdf"; _make_simple_pdf(src)
    from PIL import Image
    sig_path = sandbox / "sig.png"
    Image.new("RGB", (50, 20), "white").save(sig_path)
    r = add_signature_to_pdf(
        path=str(src),
        signature_image_path=str(sig_path),
        location={"page": 99, "x": 10, "y": 10, "width": 20, "height": 20},
        output_path=str(sandbox / "out.pdf"),
    )
    assert r["ok"] is False
    assert r["reason"] == "page_out_of_bounds"


def test_add_signature_missing_location_field_rejected(sandbox):
    src = sandbox / "doc.pdf"; _make_simple_pdf(src)
    from PIL import Image
    sig_path = sandbox / "sig.png"
    Image.new("RGB", (50, 20), "white").save(sig_path)
    r = add_signature_to_pdf(
        path=str(src),
        signature_image_path=str(sig_path),
        location={"page": 1, "x": 10},   # missing y/width/height
        output_path=str(sandbox / "out.pdf"),
    )
    assert r["ok"] is False
    assert r["reason"] == "invalid_location_fields"


# ---------- extraction ----------

def test_pdf_word_count_happy(sandbox):
    src = sandbox / "wc.pdf"
    _make_simple_pdf(src, "alpha beta gamma delta")
    r = pdf_word_count(path=str(src))
    assert r["ok"] is True
    assert r["word_count"] >= 4
    assert r["page_count"] == 1


def test_pdf_summary_happy(sandbox):
    src = sandbox / "sum.pdf"; _make_simple_pdf(src, "First page preview content")
    r = pdf_summary(path=str(src))
    assert r["ok"] is True
    assert "First page preview" in r["first_page_preview"]
    assert r["page_count"] == 1


def test_pdf_to_text_no_output(sandbox):
    src = sandbox / "txt.pdf"; _make_simple_pdf(src, "extract this text")
    r = pdf_to_text(path=str(src))
    assert r["ok"] is True
    assert "extract this text" in r["text"]
    assert r["saved_to"] is None


def test_pdf_to_text_with_save(sandbox):
    src = sandbox / "txt.pdf"; _make_simple_pdf(src, "extracted text")
    out_txt = sandbox / "extracted.txt"
    r = pdf_to_text(path=str(src), output_path=str(out_txt))
    assert r["ok"] is True
    assert out_txt.exists()
    assert "extracted text" in out_txt.read_text()


def test_pdf_to_text_invalid_output_extension(sandbox):
    src = sandbox / "txt.pdf"; _make_simple_pdf(src)
    r = pdf_to_text(path=str(src), output_path=str(sandbox / "x.bin"))
    assert r["ok"] is False
    assert r["reason"] == "output_must_be_text"


# ---------- input safety / shared ----------

def test_input_outside_root_rejected():
    for fn in [pdf_word_count, pdf_summary, pdf_to_text, extract_pdf_form_fields]:
        r = fn(path="/etc/passwd")
        assert r["ok"] is False
        assert r["reason"] in {"path_denied", "not_a_pdf"}


def test_non_pdf_input_rejected(sandbox):
    target = sandbox / "x.txt"; target.write_text("hi")
    for fn in [pdf_word_count, pdf_summary, pdf_to_text]:
        r = fn(path=str(target))
        assert r["ok"] is False
        assert r["reason"] == "not_a_pdf"
