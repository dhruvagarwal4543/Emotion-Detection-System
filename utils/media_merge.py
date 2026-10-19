"""Merges a recorded video track and a recorded audio track into one MP4.

Uses the locally installed ``ffmpeg`` binary - nothing is uploaded anywhere.

Synchronisation
---------------
The dashboard starts its video recorder and its audio recorder as two separate
calls, so one always begins a little before the other. The browser reports both
start times and the offset between them is passed to ffmpeg with ``-itsoffset``
on whichever stream started later. Without that the audio can sit a few hundred
milliseconds out, which is very audible on speech.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

# A 4-minute 720p clip merges in ~8 s on an M-series Mac; 15 min leaves room for slow machines.
FFMPEG_TIMEOUT_SECONDS = 900


class FFmpegNotAvailable(RuntimeError):
    """Raised when no ffmpeg binary can be found on PATH."""


def ffmpeg_path() -> Optional[str]:
    """Returns the path to ffmpeg, or ``None`` when it is not installed."""
    return shutil.which("ffmpeg")


def ffmpeg_version() -> Optional[str]:
    """Returns the first line of ``ffmpeg -version``, or ``None``."""
    binary = ffmpeg_path()
    if not binary:
        return None
    try:
        out = subprocess.run(
            [binary, "-version"], capture_output=True, text=True, timeout=20, check=True
        )
        return out.stdout.splitlines()[0] if out.stdout else None
    except (subprocess.SubprocessError, OSError):
        return None


def merge_video_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
    offset_seconds: float = 0.0,
) -> str:
    """Muxes ``video_path`` and ``audio_path`` into an H.264/AAC MP4.

    ``offset_seconds`` is *audio start minus video start*. A positive value
    means the audio began later than the video, so the audio is delayed by that
    much; a negative value delays the video instead.

    The source files are never modified. Returns ``output_path``.
    """
    binary = ffmpeg_path()
    if not binary:
        raise FFmpegNotAvailable(
            "ffmpeg was not found on PATH. Install it with 'brew install ffmpeg'."
        )
    for path, what in ((video_path, "video"), (audio_path, "audio")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {what} file for merge: {path}")

    command = [binary, "-y", "-hide_banner", "-loglevel", "error"]

    # -itsoffset has to come *before* the input it applies to.
    if offset_seconds > 0.001:
        command += ["-i", video_path, "-itsoffset", f"{offset_seconds:.3f}", "-i", audio_path]
    elif offset_seconds < -0.001:
        command += ["-itsoffset", f"{abs(offset_seconds):.3f}", "-i", video_path, "-i", audio_path]
    else:
        command += ["-i", video_path, "-i", audio_path]

    command += [
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",      # required for QuickTime and most players
        "-c:a", "aac",
        "-b:a", "160k",
        "-movflags", "+faststart",
        # WebM from MediaRecorder has no fixed frame rate; letting ffmpeg
        # regenerate timestamps stops the output drifting out of sync.
        "-fps_mode", "cfr",
        "-r", "25",
        # Stop at whichever track ends first so the file has no trailing
        # stretch of silence or frozen video.
        "-shortest",
        output_path,
    ]

    try:
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("ffmpeg timed out while merging the recording") from exc

        if result.returncode != 0:
            detail = (result.stderr or "").strip().splitlines()
            raise RuntimeError(
                "ffmpeg failed to merge the recording: "
                + (detail[-1] if detail else f"exit code {result.returncode}")
            )

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("ffmpeg produced an empty file")
    except Exception:
        # A half-written MP4 would show up in History as a valid recording.
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

    return output_path
