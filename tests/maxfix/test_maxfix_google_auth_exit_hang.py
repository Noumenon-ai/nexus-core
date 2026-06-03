from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_google_auth_flow_process_exits_cleanly():
    """Regression repro: Google OAuth completion must not leave pytest hung
    after the async test body already passed."""
    repo_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import asyncio
        from pathlib import Path

        from tests.phase_10_google.helpers import (
            build_google_auth_service,
            configure_google_test_env,
            pasted_url_for,
        )


        async def main():
            from _pytest.monkeypatch import MonkeyPatch

            monkeypatch = MonkeyPatch()
            try:
                tmp_path = Path.cwd() / ".pytest_google_auth_exit_hang_tmp"
                tmp_path.mkdir(exist_ok=True)
                container = configure_google_test_env(monkeypatch, tmp_path)
                user = container.users_repository.get_or_create(111)
                service, _client, _holder = build_google_auth_service(
                    container,
                    granted_scopes=(
                        "https://www.googleapis.com/auth/calendar",
                        "https://www.googleapis.com/auth/tasks.readonly",
                        "https://www.googleapis.com/auth/contacts",
                    ),
                )

                _url, future = await service.begin_connect(user.id)
                state = service._pending[user.id].state
                await service.submit_pasted_url(user.id, pasted_url_for(state=state))
                await future
                print("FLOW_OK", flush=True)
            finally:
                monkeypatch.undo()


        asyncio.run(main())
        print("EXITED", flush=True)
        """
    )

    proc = subprocess.run(
        [sys.executable, "-u", "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "FLOW_OK" in proc.stdout
    assert "EXITED" in proc.stdout
