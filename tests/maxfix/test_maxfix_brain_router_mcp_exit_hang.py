from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_generate_with_tools_process_exits_cleanly(tmp_path):
    """Regression repro: MCP routing must not leave the interpreter hung
    after the coroutine result is already produced."""
    fake_config = tmp_path / "mcp.json"
    fake_config.write_text("{}")

    repo_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import asyncio
        from unittest.mock import MagicMock, patch

        from services.brain_router import BrainRouter


        def _make_completed(stdout: str):
            proc = MagicMock()
            proc.returncode = 0
            proc.communicate.return_value = (stdout.encode(), b"")
            return proc


        async def main():
            router = BrainRouter()
            with patch(
                "services.brain_router.subprocess.Popen",
                return_value=_make_completed("ok"),
            ):
                result = await router.generate_with_tools(
                    user_id="u1",
                    system_prompt="sys",
                    contents=[
                        {
                            "role": "user",
                            "parts": [{"text": "list the files in my Documents folder"}],
                        }
                    ],
                    tool_catalog=[
                        {
                            "name": "list_directory",
                            "description": "l",
                            "parameters": {},
                        }
                    ],
                )
                print("RESULT", result["text"], flush=True)


        asyncio.run(main())
        print("EXITED", flush=True)
        """
    )

    env = os.environ.copy()
    env["NEXUS_MCP_CONFIG"] = str(fake_config)
    proc = subprocess.run(
        [sys.executable, "-u", "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "RESULT ok" in proc.stdout
    assert "EXITED" in proc.stdout
