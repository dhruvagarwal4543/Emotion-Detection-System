"""AI Emotion Detection System - Flask application.

Everything runs on this machine: the webcam stream is captured by the browser,
frames are POSTed to this server, and both the facial and the speech models are
local ONNX / scikit-learn files under ``models/``.  No cloud service, no
external API and no remote storage is involved at any point.

Start with::

    python app.py

then open http://127.0.0.1:5000
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_VIDEOS_DIR = os.path.join(BASE_DIR, "saved_videos")
SAVED_AUDIO_DIR = os.path.join(BASE_DIR, "saved_audio")
# Merged A/V output and PDF reports live beside the originals, which are always
# kept: merging never consumes or replaces the source recordings.
SAVED_MERGED_DIR = os.path.join(BASE_DIR, "saved_recordings")
SAVED_REPORTS_DIR = os.path.join(BASE_DIR, "saved_reports")

for _directory in (SAVED_VIDEOS_DIR, SAVED_AUDIO_DIR, SAVED_MERGED_DIR, SAVED_REPORTS_DIR):
    os.makedirs(_directory, exist_ok=True)

VIDEO_EXTENSIONS = {".webm", ".mp4", ".mkv", ".mov"}
AUDIO_EXTENSIONS = {".wav", ".webm", ".ogg", ".mp3", ".m4a"}
MERGED_EXTENSIONS = {".mp4"}
REPORT_EXTENSIONS = {".pdf", ".csv"}

# Every downloadable kind the app serves, in one place.
KIND_DIRS = {
    "video": (SAVED_VIDEOS_DIR, VIDEO_EXTENSIONS),
    "audio": (SAVED_AUDIO_DIR, AUDIO_EXTENSIONS),
    "merged": (SAVED_MERGED_DIR, MERGED_EXTENSIONS),
    "report": (SAVED_REPORTS_DIR, REPORT_EXTENSIONS),
}

# Only ever accept names this app generated itself: timestamped, no separators.
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # 256 MB per upload
app.config["JSON_SORT_KEYS"] = False
# The app runs with debug=False (see main()), and Jinja would otherwise cache
# templates for the life of the process - editing index.html would appear to do
# nothing until the server was restarted.
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Populated lazily on first use; a failure is remembered so the dashboard can
# explain what is missing instead of throwing a 500 on every request.
_video_error: Optional[str] = None
_audio_error: Optional[str] = None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def unique_filename(directory: str, prefix: str, extension: str, stamp: Optional[str] = None) -> str:
    """Builds ``prefix_YYYYmmdd_HHMMSS.ext``, suffixed if that name is taken.

    The timestamp only has one-second resolution, so two recordings finished in
    the same second would otherwise silently overwrite each other.
    """
    stamp = stamp or timestamp()
    candidate = f"{prefix}_{stamp}{extension}"
    counter = 2
    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{prefix}_{stamp}_{counter}{extension}"
        counter += 1
    return candidate


def stamp_of(filename: str) -> str:
    """``video_20261019_101500_2.webm`` -> ``20261019_101500_2``; links a recording to its reports."""
    return os.path.splitext(filename)[0].split("_", 1)[1]


def clean_samples(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    samples = session.get("video_samples")
    return [x for x in samples if _valid_sample(x)] if isinstance(samples, list) else []


def clean_duration(session: Dict[str, Any]) -> float:
    try:
        return max(0.0, float(session.get("duration_seconds") or 0))
    except (TypeError, ValueError):
        return 0.0


def write_reports(report_data: Dict[str, Any], prefix: str, stamp: str):
    """Builds the CSV and the PDF independently; returns ``(files, warnings)``.

    A failure in one never blocks the other, and no half-written file is left.
    """
    from utils.report import build_csv, build_report

    files: Dict[str, Any] = {}
    warnings: List[str] = []
    for key, extension, build, label in (
        ("report_csv", ".csv", build_csv, "CSV"),
        ("report", ".pdf", build_report, "PDF"),
    ):
        name = unique_filename(SAVED_REPORTS_DIR, prefix, extension, stamp)
        try:
            build(report_data, os.path.join(SAVED_REPORTS_DIR, name))
            files[key] = describe_file(SAVED_REPORTS_DIR, name, "report")
        except Exception as exc:
            app.logger.error("%s report failed: %s", label, traceback.format_exc())
            warnings.append(f"Could not build the {label} report: {exc}")
            try:
                os.remove(os.path.join(SAVED_REPORTS_DIR, name))
            except OSError:
                pass
    return files, warnings


def safe_join(directory: str, filename: str) -> Optional[str]:
    """Resolves ``filename`` inside ``directory`` or returns ``None``.

    Guards against ``..`` traversal and absolute paths in user-supplied names.
    """
    if not filename or not SAFE_NAME.match(filename):
        return None
    root = os.path.realpath(directory)
    resolved = os.path.realpath(os.path.join(directory, filename))
    # "." resolves to the folder itself; only real children are allowed.
    if resolved == root or os.path.commonpath([resolved, root]) != root:
        return None
    return resolved


def get_video_detector():
    global _video_error
    from utils.video_emotion import get_detector

    try:
        detector = get_detector()
        _video_error = None
        return detector
    except Exception as exc:
        _video_error = str(exc)
        raise


def get_audio_detector():
    global _audio_error
    from utils.audio_emotion import get_detector

    try:
        detector = get_detector()
        _audio_error = None
        return detector
    except Exception as exc:
        _audio_error = str(exc)
        raise


def decode_data_url(data_url: str) -> np.ndarray:
    """Decodes a ``data:image/jpeg;base64,...`` string into a BGR frame."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Frame is not valid base64 data") from exc

    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Frame could not be decoded as an image")
    return frame


def _valid_sample(sample: Any) -> bool:
    """A facial sample from the browser: needs an emotion and a numeric time."""
    number = (int, float)
    return (
        isinstance(sample, dict)
        and isinstance(sample.get("emotion"), str)
        and isinstance(sample.get("t"), number)
        and isinstance(sample.get("confidence", 0), number)
    )


def describe_file(directory: str, filename: str, kind: str) -> Dict[str, Any]:
    path = os.path.join(directory, filename)
    stat = os.stat(path)
    return {
        "name": filename,
        "kind": kind,
        "size_bytes": stat.st_size,
        "size_human": human_size(stat.st_size),
        "created": datetime.fromtimestamp(stat.st_mtime).strftime("%d %b %Y, %H:%M:%S"),
        "created_ts": stat.st_mtime,
        "url": f"/recordings/{kind}/{filename}",
    }


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def list_recordings() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for kind, (directory, extensions) in KIND_DIRS.items():
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if name.startswith("."):
                continue
            if os.path.splitext(name)[1].lower() not in extensions:
                continue
            try:
                items.append(describe_file(directory, name, kind))
            except OSError:
                continue
    items.sort(key=lambda item: item["created_ts"], reverse=True)
    return items


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Reports which models loaded, so the UI can show an honest banner."""
    status = {"video": {"ready": False}, "audio": {"ready": False}}

    try:
        get_video_detector()
        status["video"] = {"ready": True, "model": "FER+ (ONNX) + YuNet face detector"}
    except Exception as exc:
        status["video"] = {"ready": False, "error": str(exc)}

    try:
        detector = get_audio_detector()
        status["audio"] = {
            "ready": True,
            "model": detector.metadata.get("model", "scikit-learn"),
            "dataset": detector.metadata.get("dataset", "RAVDESS"),
            "accuracy": detector.metadata.get("held_out_accuracy"),
        }
    except Exception as exc:
        status["audio"] = {"ready": False, "error": str(exc)}

    from utils.media_merge import ffmpeg_version

    version = ffmpeg_version()
    status["ffmpeg"] = (
        {"ready": True, "version": version}
        if version
        else {"ready": False, "error": "ffmpeg not found on PATH (brew install ffmpeg)"}
    )

    return jsonify(status)


# --------------------------------------------------------------------------
# video
# --------------------------------------------------------------------------

@app.route("/api/video/analyze", methods=["POST"])
def api_video_analyze():
    """Classifies the facial emotion in a single webcam frame."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"success": False, "error": "Body must be a JSON object"}), 400
    image = payload.get("image")
    if not image:
        return jsonify({"success": False, "error": "No image supplied"}), 400

    try:
        frame = decode_data_url(image)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    try:
        detector = get_video_detector()
    except Exception as exc:
        return jsonify({"success": False, "error": f"Model unavailable: {exc}"}), 503

    try:
        sid = str(payload.get("sid") or "default")[:64]
        result = detector.predict(frame, smooth=bool(payload.get("smooth", True)), sid=sid)
    except Exception as exc:
        app.logger.error("Frame analysis failed: %s", traceback.format_exc())
        return jsonify({"success": False, "error": str(exc)}), 500

    result["success"] = True
    result["frame_size"] = {"w": int(frame.shape[1]), "h": int(frame.shape[0])}
    return jsonify(result)


@app.route("/api/video/save", methods=["POST"])
def api_video_save():
    """Stores a recorded webcam clip in ``saved_videos/``."""
    upload = request.files.get("video")
    if upload is None or not upload.filename:
        return jsonify({"success": False, "error": "No video file in the request"}), 400

    extension = os.path.splitext(upload.filename)[1].lower()
    if extension not in VIDEO_EXTENSIONS:
        extension = ".webm"  # what MediaRecorder produces in Chrome/Safari

    filename = unique_filename(SAVED_VIDEOS_DIR, "video", extension)
    destination = os.path.join(SAVED_VIDEOS_DIR, filename)
    upload.save(destination)

    size = os.path.getsize(destination)
    if size == 0:
        os.remove(destination)
        return jsonify({"success": False, "error": "The recording was empty"}), 400

    app.logger.info("Saved video %s (%s)", filename, human_size(size))
    response: Dict[str, Any] = {
        "success": True,
        "file": describe_file(SAVED_VIDEOS_DIR, filename, "video"),
        "path": os.path.join("saved_videos", filename),
        "files": {},
        "warnings": [],
    }

    # Video-only reports: only when the browser sent the facial predictions it collected.
    try:
        session = json.loads(request.form.get("session") or "null")
    except json.JSONDecodeError:
        session = None
    if isinstance(session, dict):
        response["files"], response["warnings"] = write_reports(
            {
                "mode": "video",
                "started_at": session.get("started_at"),
                "duration_seconds": clean_duration(session),
                "video_samples": clean_samples(session),
                "files": {"video": destination},
            },
            "video_report", stamp_of(filename),
        )
    elif request.form.get("session"):
        response["warnings"].append("The session data was not valid, so no report was built.")
    return jsonify(response)


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

@app.route("/api/audio/analyze", methods=["POST"])
def api_audio_analyze():
    """Saves a microphone recording and classifies the speaker's emotion."""
    upload = request.files.get("audio")
    if upload is None or not upload.filename:
        return jsonify({"success": False, "error": "No audio file in the request"}), 400

    extension = os.path.splitext(upload.filename)[1].lower()
    if extension not in AUDIO_EXTENSIONS:
        extension = ".wav"

    filename = unique_filename(SAVED_AUDIO_DIR, "audio", extension)
    destination = os.path.join(SAVED_AUDIO_DIR, filename)
    upload.save(destination)

    if os.path.getsize(destination) == 0:
        os.remove(destination)
        return jsonify({"success": False, "error": "The recording was empty"}), 400

    try:
        detector = get_audio_detector()
    except Exception as exc:
        # The clip is still on disk, so the user does not lose the recording.
        return (
            jsonify(
                {
                    "success": False,
                    "saved": True,
                    "file": describe_file(SAVED_AUDIO_DIR, filename, "audio"),
                    "error": f"Speech model unavailable: {exc}",
                }
            ),
            503,
        )

    try:
        from utils.audio_emotion import InsufficientAudioError

        result = detector.predict_file(destination)
    except InsufficientAudioError as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "saved": True,
                    "file": describe_file(SAVED_AUDIO_DIR, filename, "audio"),
                    "error": str(exc),
                }
            ),
            400,
        )
    except Exception as exc:
        app.logger.error("Audio analysis failed: %s", traceback.format_exc())
        return (
            jsonify(
                {
                    "success": False,
                    "saved": True,
                    "file": describe_file(SAVED_AUDIO_DIR, filename, "audio"),
                    "error": str(exc),
                }
            ),
            500,
        )

    result["success"] = True
    result["file"] = describe_file(SAVED_AUDIO_DIR, filename, "audio")
    result["path"] = os.path.join("saved_audio", filename)
    result["files"], result["warnings"] = {}, []

    # Audio-only reports come from real per-window predictions, never from the single result above.
    try:
        analysis = detector.predict_windows(destination)
        speech = dict(result, file=filename, at=datetime.now().isoformat())
        result["files"], result["warnings"] = write_reports(
            {
                "mode": "audio",
                "started_at": (datetime.now() - timedelta(seconds=analysis["audio_seconds"])).isoformat(),
                "duration_seconds": analysis["audio_seconds"],
                "audio_windows": analysis["windows"],
                "audio_skipped": analysis["skipped"],
                "audio_results": [speech],
                "files": {"audio": destination},
            },
            "audio_report", stamp_of(filename),
        )
    except Exception as exc:
        app.logger.error("Audio window analysis failed: %s", traceback.format_exc())
        result["warnings"].append(f"Could not build the audio report: {exc}")
    app.logger.info("Analysed %s -> %s (%.1f%%)", filename, result["emotion"], result["confidence"])
    return jsonify(result)


# --------------------------------------------------------------------------
# combined session: originals + merged MP4 + PDF report
# --------------------------------------------------------------------------

@app.route("/api/session/save", methods=["POST"])
def api_session_save():
    """Finishes a combined recording session.

    Accepts the video track, the microphone track and the facial-emotion
    samples collected while recording, then:

    1. saves the video and the audio as their own files (both are kept),
    2. classifies the speech emotion from the audio,
    3. muxes the two into one synchronised MP4 with ffmpeg,
    4. renders the PDF emotion report.

    Steps 3 and 4 are best-effort: if ffmpeg is missing the originals and the
    report are still returned rather than failing the whole request.
    """
    session_raw = request.form.get("session") or "{}"
    try:
        session = json.loads(session_raw)
    except json.JSONDecodeError:
        return jsonify({"success": False, "error": "session field is not valid JSON"}), 400
    if not isinstance(session, dict):
        return jsonify({"success": False, "error": "session field must be a JSON object"}), 400

    video_upload = request.files.get("video")
    audio_upload = request.files.get("audio")
    if video_upload is None and audio_upload is None:
        return jsonify({"success": False, "error": "No video or audio in the request"}), 400

    result: Dict[str, Any] = {"success": True, "files": {}, "warnings": []}
    video_path = audio_path = None
    stamp = timestamp()  # one stamp for every file of this session, so History can link them

    # ------------------------------------------------------------- originals
    if video_upload is not None and video_upload.filename:
        extension = os.path.splitext(video_upload.filename)[1].lower()
        if extension not in VIDEO_EXTENSIONS:
            extension = ".webm"
        name = unique_filename(SAVED_VIDEOS_DIR, "video", extension, stamp)
        video_path = os.path.join(SAVED_VIDEOS_DIR, name)
        video_upload.save(video_path)
        stamp = stamp_of(name)
        if os.path.getsize(video_path) == 0:
            os.remove(video_path)
            video_path = None
            result["warnings"].append("The video track was empty and was discarded.")
        else:
            result["files"]["video"] = describe_file(SAVED_VIDEOS_DIR, name, "video")

    if audio_upload is not None and audio_upload.filename:
        extension = os.path.splitext(audio_upload.filename)[1].lower()
        if extension not in AUDIO_EXTENSIONS:
            extension = ".wav"
        name = unique_filename(SAVED_AUDIO_DIR, "audio", extension, stamp)
        audio_path = os.path.join(SAVED_AUDIO_DIR, name)
        audio_upload.save(audio_path)
        if video_path is None:
            stamp = stamp_of(name)
        if os.path.getsize(audio_path) == 0:
            os.remove(audio_path)
            audio_path = None
            result["warnings"].append("The audio track was empty and was discarded.")
        else:
            result["files"]["audio"] = describe_file(SAVED_AUDIO_DIR, name, "audio")

    # --------------------------------------------------- speech emotion
    audio_results = [r for r in (session.get("audio_results") or []) if isinstance(r, dict)]
    if audio_path:
        try:
            from utils.audio_emotion import InsufficientAudioError

            detector = get_audio_detector()
            speech = detector.predict_file(audio_path)
            speech["file"] = os.path.basename(audio_path)
            speech["at"] = datetime.now().isoformat()
            audio_results.append(speech)
            result["audio_emotion"] = speech
        except InsufficientAudioError as exc:
            result["warnings"].append(f"Speech not analysed: {exc}")
        except Exception as exc:
            app.logger.error("Session speech analysis failed: %s", traceback.format_exc())
            result["warnings"].append(f"Speech not analysed: {exc}")

    # ------------------------------------------------------------ merged MP4
    if video_path and audio_path:
        from utils.media_merge import FFmpegNotAvailable, merge_video_audio

        merged_name = unique_filename(SAVED_MERGED_DIR, "session", ".mp4", stamp)
        merged_path = os.path.join(SAVED_MERGED_DIR, merged_name)
        try:
            offset = float(session.get("audio_offset_seconds") or 0.0)
        except (TypeError, ValueError):
            offset = 0.0
        # Guard against a nonsense offset from a stalled browser tab.
        offset = max(-5.0, min(5.0, offset))
        try:
            merge_video_audio(video_path, audio_path, merged_path, offset_seconds=offset)
            result["files"]["merged"] = describe_file(SAVED_MERGED_DIR, merged_name, "merged")
            app.logger.info("Merged session -> %s (offset %.3fs)", merged_name, offset)
        except FFmpegNotAvailable as exc:
            result["warnings"].append(
                f"{exc} The separate video and audio files were still saved."
            )
        except Exception as exc:
            app.logger.error("Merge failed: %s", traceback.format_exc())
            result["warnings"].append(f"Could not merge the recording: {exc}")
    elif video_path or audio_path:
        result["warnings"].append(
            "Only one track was recorded, so there was nothing to merge into an MP4."
        )

    # ------------------------------------------------ CSV + PDF reports
    report_data = {
        "started_at": session.get("started_at"),
        "duration_seconds": clean_duration(session),
        "video_samples": clean_samples(session),
        "audio_results": audio_results,
        "files": {
            "video": video_path,
            "audio": audio_path,
            "merged": os.path.join(SAVED_MERGED_DIR, result["files"]["merged"]["name"])
            if "merged" in result["files"] else None,
        },
    }
    files, warnings = write_reports(report_data, "report", stamp)
    result["files"].update(files)
    result["warnings"].extend(warnings)

    return jsonify(result)


# --------------------------------------------------------------------------
# recordings
# --------------------------------------------------------------------------

@app.route("/api/recordings")
def api_recordings():
    return jsonify({"success": True, "recordings": list_recordings()})


@app.route("/recordings/<kind>/<path:filename>")
def serve_recording(kind: str, filename: str):
    directory = KIND_DIRS.get(kind, (None, None))[0]
    if directory is None:
        return jsonify({"success": False, "error": "Unknown recording type"}), 404
    if safe_join(directory, filename) is None:
        return jsonify({"success": False, "error": "Invalid filename"}), 400
    if not os.path.exists(os.path.join(directory, filename)):
        return jsonify({"success": False, "error": "Recording not found"}), 404
    return send_from_directory(
        directory, filename, conditional=True,
        as_attachment=request.args.get("download") == "1",  # real Save-As with the right name
    )


@app.route("/api/recordings/<kind>/<path:filename>", methods=["DELETE"])
def delete_recording(kind: str, filename: str):
    directory = KIND_DIRS.get(kind, (None, None))[0]
    if directory is None:
        return jsonify({"success": False, "error": "Unknown recording type"}), 404

    path = safe_join(directory, filename)
    if path is None:
        return jsonify({"success": False, "error": "Invalid filename"}), 400
    if not os.path.exists(path):
        return jsonify({"success": False, "error": "Recording not found"}), 404

    if not os.path.isfile(path):
        return jsonify({"success": False, "error": "Invalid filename"}), 400
    os.remove(path)
    app.logger.info("Deleted %s recording %s", kind, filename)
    return jsonify({"success": True})


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------

@app.errorhandler(413)
def too_large(_error):
    return jsonify({"success": False, "error": "The upload is larger than the 256 MB limit"}), 413


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Endpoint not found"}), 404
    return render_template("index.html"), 200


def main() -> None:
    port = int(os.environ.get("PORT", "5000"))
    print("\n" + "=" * 62)
    print("  AI EMOTION DETECTION SYSTEM")
    print("=" * 62)
    print(f"  Dashboard : http://127.0.0.1:{port}")
    print(f"  Videos    : {SAVED_VIDEOS_DIR}")
    print(f"  Audio     : {SAVED_AUDIO_DIR}")
    print("  All processing is local - nothing leaves this machine.")
    print("=" * 62 + "\n")
    # debug=False: the reloader would load both ONNX models twice.
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
