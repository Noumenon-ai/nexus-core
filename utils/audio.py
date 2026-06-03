from __future__ import annotations

import subprocess
from pathlib import Path


def transcode_to_ogg_opus(input_path: Path, output_path: Path) -> Path:
    command = [
        'ffmpeg',
        '-y',
        '-i',
        str(input_path),
        '-c:a',
        'libopus',
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return output_path


def transcode_wav_to_ogg_opus(input_path: Path, output_path: Path) -> Path:
    return transcode_to_ogg_opus(input_path, output_path)
