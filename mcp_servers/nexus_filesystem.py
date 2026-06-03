"""nexus-filesystem MCP server — Phase 2.5a Server 1.

Read, search, write files on the user's laptop. Reuses Phase 4 security boundary
(ALLOWED_ROOTS, DENIED_*, _validate_path) from services.file_system_tools.

Destructive operations (write/append/move/delete) execute immediately when called
— Claude CLI is expected to gate them at the user-permission layer. Each
destructive tool description explicitly notes the mutation.

Every tool returns a dict; on failure returns {'ok': False, 'reason': str, 'detail': str}.
"""
from __future__ import annotations

import base64
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from services.file_system_tools import (
    ALLOWED_ROOTS,
    BINARY_PREVIEW_BYTES,
    DENIED_DIR_PREFIXES,
    FileSystemSecurityError,
    TEXT_MAX_BYTES,
    _is_likely_binary,
    _is_under,
    _matches_denied_filename,
    _matches_denied_substring,
    _validate_path,
)
from services.file_system_tools import (
    list_directory as _phase4_list_directory,
    read_file as _phase4_read_file,
    search_files as _phase4_search_files,
)


mcp = FastMCP("nexus-filesystem")

_VIEW_IMAGE_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_FIND_CONTENT_MAX_HITS = 50


# ---------- internal helpers ----------

def _validate_write_target(path: str) -> Path:
    """Validate a write/append/delete/move target. Parent must exist & be allowed.

    Returns the resolved final path (parent_resolved / filename).
    Raises FileSystemSecurityError on any policy violation.
    """
    if not isinstance(path, str) or not path.strip():
        raise FileSystemSecurityError("path must be a non-empty string")
    if "\x00" in path:
        raise FileSystemSecurityError("path contains null byte")

    if path.startswith("~/"):
        target = Path("/home/user") / path[2:]
    elif path == "~":
        raise FileSystemSecurityError("cannot write to home dir itself")
    elif path.startswith("~"):
        raise FileSystemSecurityError("~user expansion not allowed")
    else:
        target = Path(path)
    if not target.is_absolute():
        raise FileSystemSecurityError(f"path must be absolute: {path!r}")

    parent_resolved = _validate_path(str(target.parent), must_exist=True)
    if not parent_resolved.is_dir():
        raise FileSystemSecurityError(f"parent is not a directory: {parent_resolved}")

    final_path = parent_resolved / target.name
    if _matches_denied_filename(final_path.name):
        raise FileSystemSecurityError(f"filename pattern denied: {final_path.name}")
    if _matches_denied_substring(str(final_path)):
        raise FileSystemSecurityError("path matches denied substring")

    # Reject final path inside any denied prefix.
    for denied_prefix in DENIED_DIR_PREFIXES:
        try:
            denied_resolved = denied_prefix.resolve()
        except (OSError, RuntimeError):
            denied_resolved = denied_prefix
        if _is_under(final_path, denied_resolved):
            raise FileSystemSecurityError(f"path is in denied directory: {denied_prefix}")

    if not any(_is_under(final_path, root) for root in ALLOWED_ROOTS):
        raise FileSystemSecurityError(f"path is outside allowed roots: {final_path}")

    return final_path


def _toolresult_to_dict(tr: Any) -> dict:
    """Adapt Phase 4 ToolResult into the MCP {ok, ...} dict shape."""
    if tr is None:
        return {"ok": False, "reason": "no_result", "detail": ""}
    data = getattr(tr, "data", None) or {}
    if "read" in data:
        if data["read"] is True:
            return {"ok": True, **data}
        return {"ok": False, "reason": data.get("reason", "denied"), "detail": data.get("detail", ""), **data}
    if "listed" in data:
        if data["listed"] is True:
            return {"ok": True, **data}
        return {"ok": False, "reason": data.get("reason", "denied"), "detail": data.get("detail", ""), **data}
    if "found" in data or "searched" in data:
        if data.get("found") is True or data.get("searched") is True:
            return {"ok": True, **data}
        return {"ok": False, "reason": data.get("reason", "denied"), "detail": data.get("detail", ""), **data}
    return {"ok": True, **data}


# ---------- read tools ----------

@mcp.tool()
def read_text_file(path: str) -> dict:
    """Read a UTF-8 text file. Reuses Phase 4 security boundary. Returns contents + metadata."""
    tr = _phase4_read_file(user_id="mcp", path=path)
    return _toolresult_to_dict(tr)


@mcp.tool()
def read_pdf(path: str) -> dict:
    """Extract text content from a PDF file via pdfplumber."""
    try:
        resolved = _validate_path(path, must_exist=True)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "path_denied", "detail": str(exc)}
    if not resolved.is_file():
        return {"ok": False, "reason": "not_a_file", "detail": str(resolved)}
    if resolved.suffix.lower() != ".pdf":
        return {"ok": False, "reason": "not_a_pdf", "detail": resolved.name}
    try:
        import pdfplumber
    except ImportError as exc:
        return {"ok": False, "reason": "pdfplumber_not_installed", "detail": str(exc)}
    pages: list[str] = []
    try:
        with pdfplumber.open(resolved) as pdf:
            for p in pdf.pages:
                pages.append(p.extract_text() or "")
    except Exception as exc:
        return {"ok": False, "reason": "pdf_extract_failed", "detail": str(exc)[:300]}
    text = "\n\n".join(pages)
    return {
        "ok": True,
        "path": str(resolved),
        "page_count": len(pages),
        "char_count": len(text),
        "text": text,
    }


@mcp.tool()
def read_docx(path: str) -> dict:
    """Extract text from a Word (.docx) document."""
    try:
        resolved = _validate_path(path, must_exist=True)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "path_denied", "detail": str(exc)}
    if not resolved.is_file():
        return {"ok": False, "reason": "not_a_file", "detail": str(resolved)}
    if resolved.suffix.lower() != ".docx":
        return {"ok": False, "reason": "not_a_docx", "detail": resolved.name}
    try:
        from docx import Document
    except ImportError as exc:
        return {"ok": False, "reason": "python_docx_not_installed", "detail": str(exc)}
    try:
        doc = Document(str(resolved))
    except Exception as exc:
        return {"ok": False, "reason": "docx_open_failed", "detail": str(exc)[:300]}
    paragraphs = [p.text for p in doc.paragraphs]
    text = "\n".join(paragraphs)
    return {
        "ok": True,
        "path": str(resolved),
        "paragraph_count": len(paragraphs),
        "char_count": len(text),
        "text": text,
    }


@mcp.tool()
def read_spreadsheet(path: str, sheet: str | None = None, max_rows: int = 1000) -> dict:
    """Read .xlsx or .csv. Returns header + rows (up to max_rows, hard cap 10_000)."""
    try:
        resolved = _validate_path(path, must_exist=True)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "path_denied", "detail": str(exc)}
    if not resolved.is_file():
        return {"ok": False, "reason": "not_a_file", "detail": str(resolved)}
    try:
        max_rows = int(max_rows)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_max_rows", "detail": repr(max_rows)}
    max_rows = max(1, min(max_rows, 10_000))
    ext = resolved.suffix.lower()
    if ext == ".csv":
        import csv
        rows: list[list[str]] = []
        try:
            with open(resolved, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    rows.append(row)
        except OSError as exc:
            return {"ok": False, "reason": "read_failed", "detail": str(exc)[:300]}
        header = rows[0] if rows else []
        body = rows[1:] if len(rows) > 1 else []
        return {"ok": True, "kind": "csv", "path": str(resolved), "header": header, "rows": body, "row_count": len(body)}
    if ext == ".xlsx":
        try:
            import openpyxl
        except ImportError as exc:
            return {"ok": False, "reason": "openpyxl_not_installed", "detail": str(exc)}
        try:
            wb = openpyxl.load_workbook(str(resolved), read_only=True, data_only=True)
        except Exception as exc:
            return {"ok": False, "reason": "xlsx_open_failed", "detail": str(exc)[:300]}
        ws = wb[sheet] if sheet else wb.active
        rows = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i >= max_rows:
                break
            rows.append(list(row))
        wb.close()
        header = rows[0] if rows else []
        body = rows[1:] if len(rows) > 1 else []
        return {"ok": True, "kind": "xlsx", "path": str(resolved), "sheet": (sheet or (ws.title if ws else "")),
                "header": header, "rows": body, "row_count": len(body)}
    return {"ok": False, "reason": "unsupported_extension", "detail": ext or "(none)"}


@mcp.tool()
def read_image_metadata(path: str) -> dict:
    """Return image size, format, mode, and EXIF (if any). No image bytes."""
    try:
        resolved = _validate_path(path, must_exist=True)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "path_denied", "detail": str(exc)}
    if not resolved.is_file():
        return {"ok": False, "reason": "not_a_file", "detail": str(resolved)}
    try:
        from PIL import Image, ExifTags
    except ImportError as exc:
        return {"ok": False, "reason": "pillow_not_installed", "detail": str(exc)}
    try:
        with Image.open(resolved) as img:
            exif_raw = img.getexif() if hasattr(img, "getexif") else None
            exif = {}
            if exif_raw:
                for tag_id, val in exif_raw.items():
                    name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if isinstance(val, bytes):
                        val = val[:64].hex()
                    exif[name] = val
            return {
                "ok": True,
                "path": str(resolved),
                "format": img.format,
                "mode": img.mode,
                "width": img.width,
                "height": img.height,
                "size_bytes": resolved.stat().st_size,
                "exif": exif,
            }
    except Exception as exc:
        return {"ok": False, "reason": "image_open_failed", "detail": str(exc)[:300]}


@mcp.tool()
def view_image(path: str) -> dict:
    """Return base64-encoded image bytes for Claude vision. Max 5 MB."""
    try:
        resolved = _validate_path(path, must_exist=True)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "path_denied", "detail": str(exc)}
    if not resolved.is_file():
        return {"ok": False, "reason": "not_a_file", "detail": str(resolved)}
    size = resolved.stat().st_size
    if size > _VIEW_IMAGE_MAX_BYTES:
        return {"ok": False, "reason": "too_large", "detail": f"{size} bytes (cap {_VIEW_IMAGE_MAX_BYTES})"}
    try:
        with open(resolved, "rb") as f:
            data = f.read()
    except OSError as exc:
        return {"ok": False, "reason": "read_failed", "detail": str(exc)[:300]}
    import mimetypes
    mime, _ = mimetypes.guess_type(str(resolved))
    return {
        "ok": True,
        "path": str(resolved),
        "size_bytes": size,
        "mime_type": mime or "application/octet-stream",
        "base64": base64.b64encode(data).decode("ascii"),
    }


@mcp.tool()
def list_directory(path: str, max_entries: int = 200) -> dict:
    """List files + subdirectories. Hides denied entries (e.g. .ssh, .config)."""
    tr = _phase4_list_directory(user_id="mcp", path=path, max_entries=max_entries)
    return _toolresult_to_dict(tr)


@mcp.tool()
def find_file(query: str, search_root: str | None = None, content: bool = False) -> dict:
    """Find files by name (substring). If content=True, also greps text files for the query.
    search_root defaults to /home/user."""
    if not isinstance(query, str) or not query.strip():
        return {"ok": False, "reason": "empty_query", "detail": ""}
    root = search_root or "/home/user"
    if not content:
        tr = _phase4_search_files(user_id="mcp", root=root, pattern=query, max_results=200)
        return _toolresult_to_dict(tr)

    # Content search: walk allowed root, grep text files.
    try:
        root_resolved = _validate_path(root, must_exist=True)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "path_denied", "detail": str(exc)}
    if not root_resolved.is_dir():
        return {"ok": False, "reason": "not_a_directory", "detail": str(root_resolved)}

    hits: list[dict] = []
    q_lower = query.lower()
    files_scanned = 0
    hard_cap = 3000
    for dirpath, dirnames, filenames in os.walk(root_resolved, followlinks=False):
        # Prune denied directories.
        dirnames[:] = [
            d for d in dirnames
            if not _matches_denied_filename(d)
            and not _matches_denied_substring(str(Path(dirpath) / d))
            and not any(_is_under(Path(dirpath) / d, dp) for dp in DENIED_DIR_PREFIXES)
        ]
        for fname in filenames:
            if files_scanned >= hard_cap or len(hits) >= _FIND_CONTENT_MAX_HITS:
                break
            full = Path(dirpath) / fname
            if _matches_denied_filename(fname):
                continue
            if _matches_denied_substring(str(full)):
                continue
            if full.suffix.lower() not in {".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".csv", ".log", ".html", ".css", ""}:
                continue
            files_scanned += 1
            try:
                with open(full, "rb") as f:
                    head = f.read(64 * 1024)
                if _is_likely_binary(head):
                    continue
                if q_lower in head.decode("utf-8", errors="replace").lower():
                    hits.append({"path": str(full), "size_bytes": full.stat().st_size})
            except OSError:
                continue
        if files_scanned >= hard_cap or len(hits) >= _FIND_CONTENT_MAX_HITS:
            break
    return {"ok": True, "query": query, "search_root": str(root_resolved), "files_scanned": files_scanned, "hits": hits}


@mcp.tool()
def recent_files(extension: str | None = None, days: int = 7, search_root: str | None = None, max_results: int = 100) -> dict:
    """Files modified within the last N days. Optionally filter by extension (e.g. '.pdf')."""
    try:
        days = int(days)
        max_results = int(max_results)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "non_integer_argument", "detail": "days/max_results must be ints"}
    if days < 1 or days > 3650:
        return {"ok": False, "reason": "days_out_of_range", "detail": "days must be 1..3650"}
    max_results = max(1, min(max_results, 500))
    cutoff = time.time() - days * 86400
    ext_filter = None
    if extension:
        ext_filter = extension.lower()
        if not ext_filter.startswith("."):
            ext_filter = "." + ext_filter
    root = search_root or "/home/user"
    try:
        root_resolved = _validate_path(root, must_exist=True)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "path_denied", "detail": str(exc)}
    if not root_resolved.is_dir():
        return {"ok": False, "reason": "not_a_directory", "detail": str(root_resolved)}

    results: list[dict] = []
    files_scanned = 0
    hard_cap = 10_000
    for dirpath, dirnames, filenames in os.walk(root_resolved, followlinks=False):
        dirnames[:] = [
            d for d in dirnames
            if not _matches_denied_filename(d)
            and not _matches_denied_substring(str(Path(dirpath) / d))
            and not any(_is_under(Path(dirpath) / d, dp) for dp in DENIED_DIR_PREFIXES)
        ]
        for fname in filenames:
            if files_scanned >= hard_cap or len(results) >= max_results:
                break
            files_scanned += 1
            full = Path(dirpath) / fname
            if _matches_denied_filename(fname):
                continue
            if _matches_denied_substring(str(full)):
                continue
            if ext_filter and full.suffix.lower() != ext_filter:
                continue
            try:
                st = full.stat()
            except OSError:
                continue
            if st.st_mtime < cutoff:
                continue
            results.append({
                "path": str(full),
                "size_bytes": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
            })
        if files_scanned >= hard_cap or len(results) >= max_results:
            break
    results.sort(key=lambda r: r["modified"], reverse=True)
    return {"ok": True, "search_root": str(root_resolved), "days": days, "extension_filter": ext_filter,
            "files_scanned": files_scanned, "result_count": len(results), "results": results}


@mcp.tool()
def file_info(path: str) -> dict:
    """Stat: path, size, modified, mime type, owner uid, is_file/is_dir."""
    try:
        resolved = _validate_path(path, must_exist=True)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "path_denied", "detail": str(exc)}
    try:
        st = resolved.stat()
    except OSError as exc:
        return {"ok": False, "reason": "stat_failed", "detail": str(exc)[:200]}
    import mimetypes
    mime, _ = mimetypes.guess_type(str(resolved))
    return {
        "ok": True,
        "path": str(resolved),
        "size_bytes": st.st_size,
        "modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
        "created": datetime.fromtimestamp(st.st_ctime).isoformat(),
        "owner_uid": st.st_uid,
        "mime_type": mime,
        "is_file": resolved.is_file(),
        "is_dir": resolved.is_dir(),
        "is_symlink": resolved.is_symlink(),
    }


# ---------- write / mutate tools ----------

@mcp.tool(meta={"approval_required": True, "approval_template": "Write file?"})
def write_text_file(path: str, content: str) -> dict:
    """WRITE: create or replace a text file at path. Parent must already exist inside ALLOWED_ROOTS.
    Atomic write via tempfile + rename. Denied filename patterns (.env, *.pem, *.key, *credentials*.json) are rejected."""
    if not isinstance(content, str):
        return {"ok": False, "reason": "non_string_content", "detail": repr(type(content).__name__)}
    try:
        final = _validate_write_target(path)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "path_denied", "detail": str(exc)}
    data = content.encode("utf-8")
    if len(data) > TEXT_MAX_BYTES:
        return {"ok": False, "reason": "content_too_large", "detail": f"{len(data)} bytes (cap {TEXT_MAX_BYTES})"}
    tmp = final.with_name(f".{final.name}.mcp-tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, final)
    except OSError as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return {"ok": False, "reason": "write_failed", "detail": str(exc)[:200]}
    return {"ok": True, "path": str(final), "bytes_written": len(data)}


@mcp.tool(meta={"approval_required": True, "approval_template": "Append to file?"})
def append_to_file(path: str, content: str) -> dict:
    """APPEND text to an existing or new file. Same security boundary as write_text_file."""
    if not isinstance(content, str):
        return {"ok": False, "reason": "non_string_content", "detail": repr(type(content).__name__)}
    try:
        final = _validate_write_target(path)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "path_denied", "detail": str(exc)}
    data = content.encode("utf-8")
    try:
        existing = final.stat().st_size if final.exists() else 0
    except OSError:
        existing = 0
    if existing + len(data) > TEXT_MAX_BYTES * 4:
        return {"ok": False, "reason": "would_exceed_size_cap", "detail": f"existing {existing} + {len(data)} > cap {TEXT_MAX_BYTES*4}"}
    try:
        with open(final, "ab") as f:
            f.write(data)
    except OSError as exc:
        return {"ok": False, "reason": "append_failed", "detail": str(exc)[:200]}
    return {"ok": True, "path": str(final), "bytes_appended": len(data), "new_size_bytes": existing + len(data)}


@mcp.tool(meta={"approval_required": True, "approval_template": "Move/rename file?"})
def move_file(from_path: str, to_path: str) -> dict:
    """MOVE/RENAME a file. Both paths must validate. Refuses to overwrite an existing file at destination."""
    try:
        src = _validate_path(from_path, must_exist=True)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "from_path_denied", "detail": str(exc)}
    if not src.is_file():
        return {"ok": False, "reason": "from_not_a_file", "detail": str(src)}
    try:
        dst = _validate_write_target(to_path)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "to_path_denied", "detail": str(exc)}
    if dst.exists():
        return {"ok": False, "reason": "destination_exists", "detail": str(dst)}
    try:
        os.rename(src, dst)
    except OSError as exc:
        return {"ok": False, "reason": "rename_failed", "detail": str(exc)[:200]}
    return {"ok": True, "from": str(src), "to": str(dst)}


@mcp.tool(meta={"approval_required": True, "approval_template": "Delete file?"})
def delete_file(path: str) -> dict:
    """DELETE a single file. Refuses directories (no recursive delete). Path must validate."""
    try:
        resolved = _validate_path(path, must_exist=True)
    except FileSystemSecurityError as exc:
        return {"ok": False, "reason": "path_denied", "detail": str(exc)}
    if resolved.is_dir():
        return {"ok": False, "reason": "is_directory", "detail": "delete_file refuses directories"}
    if not resolved.is_file():
        return {"ok": False, "reason": "not_a_file", "detail": str(resolved)}
    if _matches_denied_filename(resolved.name):
        return {"ok": False, "reason": "filename_denied", "detail": resolved.name}
    if _matches_denied_substring(str(resolved)):
        return {"ok": False, "reason": "path_substring_denied", "detail": str(resolved)}
    size = resolved.stat().st_size
    try:
        os.remove(resolved)
    except OSError as exc:
        return {"ok": False, "reason": "delete_failed", "detail": str(exc)[:200]}
    return {"ok": True, "path": str(resolved), "size_bytes": size}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
