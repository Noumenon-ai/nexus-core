"""Nexus full-regression orchestrator (H2-041).

Single entry point. Runs every group in tests/regression/regression_config.py
as its own pytest invocation, parses the JUnit XML each one produces, and
emits a JSON + Markdown report under .data/regression/.

Exits 0 on full green, 1 on any failure. Logs progress to stdout (which
systemd / journald captures when invoked from a unit).

Usage:
    .venv/bin/python scripts/run_full_regression.py              # all groups
    .venv/bin/python scripts/run_full_regression.py phase_2_brain_router    # one
    .venv/bin/python scripts/run_full_regression.py --no-aux     # skip aux groups
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.regression.regression_config import (
    AUX_GROUPS,
    PHASE_GROUPS,
    all_groups,
    validate_paths,
)

REPORT_DIR = PROJECT_ROOT / '.data' / 'regression'
JSON_REPORT = REPORT_DIR / 'last_run.json'
MD_REPORT = REPORT_DIR / 'last_run.md'


@dataclass
class TestFailure:
    test: str
    file: str
    line: int | None
    error: str


@dataclass
class GroupResult:
    group: str
    paths: list[str]
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_sec: float = 0.0
    pytest_exit_code: int = 0
    failures: list[TestFailure] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors + self.skipped

    @property
    def is_green(self) -> bool:
        # Skipped is fine (intentional). Failed/errors are not.
        return self.failed == 0 and self.errors == 0 and self.pytest_exit_code in (0, 5)


def run_pytest_group(group_name: str, paths: list[str], junit_path: Path) -> GroupResult:
    """Invoke pytest for one group, write JUnit XML to junit_path, return
    parsed results. Uses subprocess so a crash in one group doesn't take down
    the orchestrator."""
    result = GroupResult(group=group_name, paths=list(paths))
    junit_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PROJECT_ROOT / '.venv' / 'bin' / 'python'), '-m', 'pytest',
        *paths,
        f'--junitxml={junit_path}',
        '--tb=short',
        '-q',
        '--no-header',
    ]
    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    result.duration_sec = time.monotonic() - start
    result.pytest_exit_code = proc.returncode

    if junit_path.exists():
        _parse_junit(junit_path, result)
    else:
        # No XML written — pytest blew up before any test could run.
        result.errors = 1
        result.failures.append(TestFailure(
            test=f'{group_name}: pytest collection failed',
            file='',
            line=None,
            error=(proc.stderr or proc.stdout)[-2000:],
        ))

    return result


def _parse_junit(xml_path: Path, result: GroupResult) -> None:
    """Read a pytest --junitxml file and fold counts + failures into result."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        result.errors += 1
        result.failures.append(TestFailure(
            test=f'{result.group}: malformed JUnit XML',
            file=str(xml_path),
            line=None,
            error=str(exc),
        ))
        return

    root = tree.getroot()
    # pytest writes a top-level <testsuites> wrapping one <testsuite>.
    suites = root.findall('testsuite') if root.tag == 'testsuites' else [root]
    for suite in suites:
        result.passed += int(suite.get('tests', 0)) - (
            int(suite.get('failures', 0)) + int(suite.get('errors', 0))
            + int(suite.get('skipped', 0))
        )
        result.failed += int(suite.get('failures', 0))
        result.errors += int(suite.get('errors', 0))
        result.skipped += int(suite.get('skipped', 0))
        for case in suite.findall('testcase'):
            classname = case.get('classname', '')
            name = case.get('name', '')
            test_id = f'{classname}::{name}' if classname else name
            for tag in ('failure', 'error'):
                node = case.find(tag)
                if node is not None:
                    result.failures.append(TestFailure(
                        test=test_id,
                        file=case.get('file', ''),
                        line=int(case.get('line', 0) or 0) or None,
                        error=(node.get('message') or '')[:500],
                    ))


def emit_json(group_results: list[GroupResult], started_iso: str, duration: float) -> None:
    total_passed = sum(r.passed for r in group_results)
    total_failed = sum(r.failed for r in group_results)
    total_errors = sum(r.errors for r in group_results)
    total_skipped = sum(r.skipped for r in group_results)
    payload = {
        'started_iso': started_iso,
        'duration_sec': round(duration, 2),
        'total_tests': total_passed + total_failed + total_errors + total_skipped,
        'passed': total_passed,
        'failed': total_failed,
        'errors': total_errors,
        'skipped': total_skipped,
        'green': total_failed == 0 and total_errors == 0,
        'groups': [
            {
                'group': r.group,
                'paths': r.paths,
                'passed': r.passed,
                'failed': r.failed,
                'errors': r.errors,
                'skipped': r.skipped,
                'duration_sec': round(r.duration_sec, 2),
                'pytest_exit_code': r.pytest_exit_code,
                'is_green': r.is_green,
                'failures': [asdict(f) for f in r.failures],
            }
            for r in group_results
        ],
    }
    JSON_REPORT.parent.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(payload, indent=2))


def emit_markdown(group_results: list[GroupResult], started_iso: str, duration: float) -> None:
    total_passed = sum(r.passed for r in group_results)
    total_failed = sum(r.failed for r in group_results)
    total_errors = sum(r.errors for r in group_results)
    total_skipped = sum(r.skipped for r in group_results)
    green = total_failed == 0 and total_errors == 0

    lines = [
        f'# Nexus full regression — {started_iso}',
        '',
        f'**Status:** {"GREEN" if green else "RED"}  ',
        f'**Duration:** {duration:.1f}s  ',
        f'**Totals:** {total_passed} passed · {total_failed} failed · '
        f'{total_errors} errors · {total_skipped} skipped',
        '',
        '## Group results',
        '',
        '| Group | Passed | Failed | Errors | Skipped | Duration | Status |',
        '|-------|-------:|-------:|-------:|--------:|---------:|--------|',
    ]
    for r in group_results:
        status = 'green' if r.is_green else 'RED'
        lines.append(
            f'| {r.group} | {r.passed} | {r.failed} | {r.errors} | '
            f'{r.skipped} | {r.duration_sec:.1f}s | {status} |'
        )

    failures = [(r, f) for r in group_results for f in r.failures]
    if failures:
        lines.extend(['', '## Failures', ''])
        for r, f in failures:
            lines.append(f'### `{r.group}` → `{f.test}`')
            if f.file:
                line_suffix = f':{f.line}' if f.line else ''
                lines.append(f'**Location:** `{f.file}{line_suffix}`  ')
            lines.append(f'**Error:** `{f.error or "(no message)"}`')
            lines.append('')

    MD_REPORT.write_text('\n'.join(lines))


def run() -> int:
    parser = argparse.ArgumentParser(description='Nexus full regression runner')
    parser.add_argument('groups', nargs='*', help='Specific group names to run (default: all)')
    parser.add_argument('--no-aux', action='store_true',
                        help='Skip aux_* groups (legacy phases)')
    parser.add_argument('--list-groups', action='store_true',
                        help='List configured group names and exit')
    args = parser.parse_args()

    if args.list_groups:
        for g in all_groups().keys():
            print(g)
        return 0

    missing = validate_paths()
    if missing:
        print('CONFIG ERROR — paths missing on disk:', file=sys.stderr)
        for m in missing:
            print(f'  {m}', file=sys.stderr)
        return 1

    if args.groups:
        groups = {g: all_groups()[g] for g in args.groups if g in all_groups()}
        if not groups:
            print(f'No matching groups. Available: {list(all_groups().keys())}',
                  file=sys.stderr)
            return 1
    elif args.no_aux:
        groups = dict(PHASE_GROUPS)
    else:
        groups = dict(all_groups())

    started = datetime.now(timezone.utc)
    started_iso = started.isoformat()
    overall_start = time.monotonic()

    # Clean prior per-group junit files so a stale one can't pollute totals.
    junit_dir = REPORT_DIR / 'junit'
    if junit_dir.exists():
        shutil.rmtree(junit_dir)
    junit_dir.mkdir(parents=True, exist_ok=True)

    print(f'[regression] starting {len(groups)} group(s) at {started_iso}', flush=True)
    group_results: list[GroupResult] = []
    for group_name, paths in groups.items():
        junit_path = junit_dir / f'{group_name}.xml'
        print(f'[regression] {group_name}: pytest {" ".join(paths)}', flush=True)
        result = run_pytest_group(group_name, paths, junit_path)
        group_results.append(result)
        verdict = 'green' if result.is_green else 'RED'
        print(
            f'[regression] {group_name}: {result.passed}p / {result.failed}f / '
            f'{result.errors}e / {result.skipped}s in {result.duration_sec:.1f}s '
            f'[{verdict}]',
            flush=True,
        )

    duration = time.monotonic() - overall_start
    emit_json(group_results, started_iso, duration)
    emit_markdown(group_results, started_iso, duration)

    total_failed = sum(r.failed + r.errors for r in group_results)
    total_tests = sum(r.passed + r.failed + r.errors + r.skipped for r in group_results)
    print('', flush=True)
    print(f'[regression] DONE — {total_tests} tests in {duration:.1f}s', flush=True)
    print(f'[regression] JSON: {JSON_REPORT}', flush=True)
    print(f'[regression] MD:   {MD_REPORT}', flush=True)
    if total_failed == 0:
        print('[regression] ALL GREEN', flush=True)
        return 0
    print(f'[regression] {total_failed} failure(s) — see {MD_REPORT}', flush=True)
    return 1


if __name__ == '__main__':
    sys.exit(run())
