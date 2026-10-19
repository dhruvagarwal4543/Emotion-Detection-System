"""Builds the downloadable PDF emotion report for a recording session.

Everything is rendered locally with ReportLab. The report covers what the brief
asks for: the facial emotions seen over the session, the speech emotions, the
confidence behind each one, the dominant emotion for each channel, and the
timestamps.
"""

from __future__ import annotations

import csv
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Same palette the dashboard uses, so the report matches what was on screen.
EMOTION_COLORS = {
    "Happy": colors.HexColor("#f5b301"),
    "Sad": colors.HexColor("#3b82f6"),
    "Angry": colors.HexColor("#ef4444"),
    "Fear": colors.HexColor("#a855f7"),
    "Surprise": colors.HexColor("#06b6d4"),
    "Neutral": colors.HexColor("#94a3b8"),
    "Disgust": colors.HexColor("#10b981"),
}

INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#6b7280")
RULE = colors.HexColor("#d1d5db")
BAND = colors.HexColor("#f3f4f6")


TITLES = {"combined": "Emotion Detection Report", "video": "Video Emotion Report",
          "audio": "Audio Emotion Report"}


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=20, leading=24,
                                textColor=INK, alignment=TA_LEFT, spaceAfter=2),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=9.5, textColor=MUTED,
                              spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=12.5, textColor=INK,
                             spaceBefore=13, spaceAfter=6),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.5, leading=13.5,
                               textColor=INK),
        "note": ParagraphStyle("n", parent=base["Normal"], fontSize=8.5, leading=11.5,
                               textColor=MUTED),
    }


def _fmt_clock(value: Optional[str]) -> str:
    """Renders an ISO timestamp as a readable local time."""
    if not value:
        return datetime.now().strftime("%d %B %Y, %H:%M:%S")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime(
            "%d %B %Y, %H:%M:%S"
        )
    except ValueError:
        return value


def _bar(fraction: float, colour, width: float = 46 * mm) -> Table:
    """A thin proportional bar, drawn as a two-cell table."""
    fraction = max(0.0, min(1.0, fraction))
    filled = max(0.4 * mm, width * fraction)
    bar = Table([[""]], colWidths=[filled], rowHeights=[3.6 * mm])
    bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colour),
                             ("LINEBELOW", (0, 0), (-1, -1), 0, colors.white)]))
    rest = Table([[""]], colWidths=[max(0.1 * mm, width - filled)], rowHeights=[3.6 * mm])
    rest.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e5e7eb"))]))
    holder = Table([[bar, rest]], colWidths=[filled, max(0.1 * mm, width - filled)])
    holder.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return holder


def _table_style(header: bool = True) -> TableStyle:
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        commands += [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), BAND),
            ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
        ]
    return TableStyle(commands)


def summarise(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates per-frame emotion samples into shares and mean confidence."""
    counts = Counter()
    confidence = defaultdict(list)
    for sample in samples:
        emotion = sample.get("emotion")
        if not emotion:
            continue
        counts[emotion] += 1
        try:
            confidence[emotion].append(float(sample.get("confidence") or 0.0))
        except (TypeError, ValueError):
            pass

    total = sum(counts.values())
    rows = []
    for emotion, count in counts.most_common():
        values = confidence.get(emotion) or [0.0]
        rows.append({
            "emotion": emotion,
            "count": count,
            "share": (count / total) if total else 0.0,
            "avg_confidence": sum(values) / len(values),
            "peak_confidence": max(values),
        })
    return {"total": total, "rows": rows,
            "dominant": rows[0]["emotion"] if rows else None,
            "dominant_share": rows[0]["share"] if rows else 0.0,
            "dominant_confidence": rows[0]["avg_confidence"] if rows else 0.0}


def build_report(session: Dict[str, Any], output_path: str) -> str:
    """Writes the PDF for one session and returns ``output_path``."""
    style = _styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        title=TITLES[_mode(session)], author="AI Emotion Detection System",
    )

    mode = _mode(session)
    video_samples = session.get("video_samples") or [] if mode != "audio" else []
    audio_results = session.get("audio_results") or [] if mode != "video" else []
    windows = session.get("audio_windows") or []
    files = session.get("files") or {}
    generated = datetime.now()
    video = summarise(video_samples)

    story: List[Any] = _template_story(session, style)
    story.append(PageBreak())
    story.append(Paragraph(TITLES[mode], style["title"]))
    story.append(Paragraph(
        f"Generated {generated.strftime('%d %B %Y at %H:%M:%S')} &nbsp;·&nbsp; "
        f"processed entirely on this machine", style["sub"]))
    story.append(HRFlowable(width="100%", thickness=0.8, color=RULE, spaceAfter=8))

    # ---------------------------------------------------------------- session
    duration = session.get("duration_seconds")
    overview = [
        ["Session started", _fmt_clock(session.get("started_at"))],
        ["Recording length", f"{float(duration):.1f} seconds" if duration else "not recorded"],
    ]
    if mode != "audio":
        overview.append(["Facial samples analysed", str(video["total"])])
    if mode == "combined":
        overview.append(["Voice clips analysed", str(len(audio_results))])
    if mode == "audio":
        overview.append(["Speech windows analysed", f"{len(windows)} (4 s each)"])
    for key, label in (("video", "Video file"), ("audio", "Audio file"), ("merged", "Merged MP4")):
        if files.get(key):
            overview.append([label, os.path.basename(files[key])])

    table = Table(overview, colWidths=[45 * mm, None])
    table.setStyle(_table_style(header=False))
    story.append(Paragraph("Session", style["h2"]))
    story.append(table)

    # -------------------------------------------------------------- dominant
    story.append(Paragraph("Dominant emotions", style["h2"]))
    dominant_rows = [["Channel", "Dominant emotion", "Share of session", "Mean confidence"]]
    if mode == "audio":
        speech = summarise(windows)
        dominant_rows.append([
            "Speech (audio)", speech["dominant"] or "no speech analysed",
            f"{speech['dominant_share']*100:.1f}% of windows" if speech["dominant"] else "-",
            f"{speech['dominant_confidence']:.1f}%" if speech["dominant"] else "-",
        ])
    elif video["dominant"]:
        dominant_rows.append([
            "Facial (video)", video["dominant"],
            f"{video['dominant_share']*100:.1f}%", f"{video['dominant_confidence']:.1f}%",
        ])
    else:
        dominant_rows.append(["Facial (video)", "no face detected", "-", "-"])

    if mode != "combined":
        pass
    elif audio_results:
        best = max(audio_results, key=lambda r: float(r.get("confidence") or 0))
        counts = Counter(r.get("emotion") for r in audio_results if r.get("emotion"))
        top_audio = counts.most_common(1)[0][0] if counts else "-"
        dominant_rows.append([
            "Speech (audio)", top_audio,
            f"{counts[top_audio]}/{len(audio_results)} clips",
            f"{float(best.get('confidence') or 0):.1f}% peak",
        ])
    else:
        dominant_rows.append(["Speech (audio)", "no voice clip recorded", "-", "-"])

    table = Table(dominant_rows, colWidths=[32 * mm, 45 * mm, 40 * mm, None])
    table.setStyle(_table_style())
    story.append(table)

    # ------------------------------------------------------- video breakdown
    if mode != "audio":
        story.append(Paragraph("Facial emotion distribution", style["h2"]))
    if mode == "audio":
        pass
    elif video["rows"]:
        rows = [["Emotion", "Share", "Frames", "Mean conf.", "Peak conf."]]
        for row in video["rows"]:
            rows.append([
                row["emotion"],
                _bar(row["share"], EMOTION_COLORS.get(row["emotion"], MUTED), 30 * mm),
                str(row["count"]),
                f"{row['avg_confidence']:.1f}%",
                f"{row['peak_confidence']:.1f}%",
            ])
            rows[-1][1] = Table([[rows[-1][1], Paragraph(
                f"<font size=8>{row['share']*100:.1f}%</font>", style["body"])]],
                colWidths=[30 * mm, 14 * mm])
        table = Table(rows, colWidths=[30 * mm, 48 * mm, 20 * mm, 25 * mm, None])
        table.setStyle(_table_style())
        story.append(table)
    else:
        story.append(Paragraph(
            "No face was detected during this session, so there is no facial "
            "emotion data to report.", style["body"]))

    # ------------------------------------------------------------- audio rows
    if mode != "video":
        story.append(Paragraph("Speech emotion results", style["h2"]))
    if mode == "video":
        pass
    elif mode == "audio":
        rows = [["#", "Window (s)", "Emotion", "Confidence"]]
        for index, w in enumerate(windows, start=1):
            rows.append([str(index), f"{float(w['start']):.1f}-{float(w['end']):.1f}", w["emotion"],
                         f"{float(w['confidence']):.1f}%"])
        if windows:
            table = Table(rows, colWidths=[10 * mm, 38 * mm, 40 * mm, None], repeatRows=1)
            table.setStyle(_table_style())
            story.append(table)
        else:
            story.append(Paragraph("No speech window could be analysed.", style["body"]))
        for note in _audio_window_notes(session)[:2]:
            story.append(Paragraph(note, style["note"]))
        for k in session.get("audio_skipped") or []:
            story.append(Paragraph(f"Skipped {float(k['start']):.1f}-{float(k['end']):.1f} s: {k['reason']}",
                                   style["note"]))
    elif audio_results:
        rows = [["#", "Time", "Emotion", "Confidence", "Clip"]]
        for index, result in enumerate(audio_results, start=1):
            rows.append([
                str(index),
                _fmt_clock(result.get("at")).split(", ")[-1],
                result.get("emotion", "-"),
                f"{float(result.get('confidence') or 0):.1f}%",
                os.path.basename(result.get("file", "-")),
            ])
        table = Table(rows, colWidths=[10 * mm, 28 * mm, 30 * mm, 28 * mm, None])
        table.setStyle(_table_style())
        story.append(table)
    else:
        story.append(Paragraph("No voice clip was recorded in this session.", style["body"]))

    # ---------------------------------------------------------- full timeline
    if video_samples:
        story.append(PageBreak())
        story.append(Paragraph("Facial emotion timeline", style["h2"]))
        story.append(Paragraph(
            "Every frame the detector analysed during the session, in order.",
            style["note"]))
        story.append(Spacer(1, 4))
        rows = [["Time", "Emotion", "Confidence"]]
        for sample in video_samples:
            offset = sample.get("t")
            rows.append([
                f"{float(offset):6.2f} s" if offset is not None else "-",
                sample.get("emotion", "-"),
                f"{float(sample.get('confidence') or 0):.1f}%",
            ])
        table = Table(rows, colWidths=[30 * mm, 45 * mm, None], repeatRows=1)
        table.setStyle(_table_style())
        story.append(table)

    # ----------------------------------------------------------------- footer
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceAfter=6))
    story.append(Paragraph(
        "Facial emotions: FER+ (ONNX) with YuNet face detection, corrected for the "
        "model's neutral/happiness bias. Speech emotions: scikit-learn trained on "
        "RAVDESS. Both models ran locally; no data left this machine.<br/>"
        "These models read facial expression and vocal tone, not inner feeling. "
        "Confidence is the model's own output and is not a guarantee of correctness.",
        style["note"]))

    doc.build(story)
    return output_path


# --------------------------------------------------------------------------
# CSV in the observer template (Emotion_timeline.csv)
# --------------------------------------------------------------------------

TEMPLATE_EMOTIONS = ["Happy", "Sad", "Neutral", "Anger", "Fear", "Disgust", "Surprise"]
INTERVAL_SECONDS = 20
_RENAME = {"Angry": "Anger"}  # the models say "Angry", the template says "Anger"


def _interval_label(index: int) -> str:
    start = 0 if index == 0 else index * INTERVAL_SECONDS + 1
    return f"{start}-{(index + 1) * INTERVAL_SECONDS}"


def interval_dominants(samples: List[Dict[str, Any]], duration: float) -> List[Optional[str]]:
    """Most frequent emotion per 20 s interval (ties: higher mean confidence).

    Intervals follow the template: 0-20, 21-40, ... A sample at t seconds goes
    to the interval whose range holds ceil(t). ``None`` means no usable sample.
    """
    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        try:
            t = float(sample["t"])
        except (KeyError, TypeError, ValueError):
            continue
        if sample.get("emotion") and t >= 0:
            buckets[max(0, math.ceil(t) - 1) // INTERVAL_SECONDS].append(sample)
    last = max([*buckets, max(0, math.ceil(duration or 0) - 1) // INTERVAL_SECONDS, 5])
    out: List[Optional[str]] = []
    for i in range(last + 1):
        rows = summarise(buckets.get(i, []))["rows"]
        best = max(rows, key=lambda r: (r["count"], r["avg_confidence"]), default=None)
        out.append(_RENAME.get(best["emotion"], best["emotion"]) if best else None)
    return out


def _mode(session: Dict[str, Any]) -> str:
    """"combined" (default, the original format), "video" or "audio"."""
    return session.get("mode") if session.get("mode") in ("video", "audio") else "combined"


def _grid_samples(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What fills the template grid: speech windows for audio-only, facial samples otherwise."""
    key = "audio_windows" if _mode(session) == "audio" else "video_samples"
    return session.get(key) or []


def _missing_text(session: Dict[str, Any]) -> str:
    return "no speech analysed" if _mode(session) == "audio" else "no face detected"


AUDIO_CLASSES_NOTE = ("Speech model classes: Neutral, Happy, Sad, Angry (shown as Anger), Fear. "
                      "Disgust and Surprise are not supported, so those rows stay empty.")


def _audio_window_notes(session: Dict[str, Any]) -> List[str]:
    windows = session.get("audio_windows") or []
    skipped = session.get("audio_skipped") or []
    total = float(session.get("duration_seconds") or 0)
    analysed = sum(float(w.get("end", 0)) - float(w.get("start", 0)) for w in windows)
    notes = [
        f"Speech analysed in consecutive 4-second windows: {len(windows)} predicted, {len(skipped)} skipped, "
        f"{analysed:.1f} s analysed of a {total:.1f} s recording. Each window is classified on its own; "
        "an interval's emotion is the most frequent window emotion (ties: higher mean confidence).",
        AUDIO_CLASSES_NOTE,
    ]
    notes += [f"Window {float(w['start']):.1f}-{float(w['end']):.1f} s: {w['emotion']} "
              f"({float(w['confidence']):.1f}% model probability)" for w in windows]
    notes += [f"Skipped {float(k['start']):.1f}-{float(k['end']):.1f} s: {k['reason']}" for k in skipped]
    return notes


def _speech_notes(session: Dict[str, Any]) -> List[str]:
    """Speech results, kept apart from the facial grid and never split by interval."""
    if _mode(session) == "video":
        return ["Video-only session: no audio was recorded or analysed."]
    if _mode(session) == "audio":
        return _audio_window_notes(session)
    duration = float(session.get("duration_seconds") or 0)
    notes = []
    for result in session.get("audio_results") or []:
        window = float(result.get("duration") or 0)
        notes.append(
            f"Speech model ({os.path.basename(str(result.get('file', 'clip')))}): "
            f"{result.get('emotion', '-')} ({float(result.get('confidence') or 0):.1f}% model probability). "
            f"One prediction from a {window:.1f} s window of the speech in a {duration:.0f} s recording; "
            "it describes that window only and is not split by interval."
        )
    return notes or ["Speech model: no voice clip analysed."]


def _duration_note(session: Dict[str, Any]) -> str:
    duration = float(session.get("duration_seconds") or 0)
    count = len(_grid_samples(session))
    what = "speech windows" if _mode(session) == "audio" else "facial predictions"
    return (f"Recording length: {duration:.1f} seconds, {len(interval_dominants([], duration))} intervals, "
            f"{count} {what}.")


def build_csv(session: Dict[str, Any], output_path: str) -> str:
    """Writes the observer-template CSV: rows are emotions, columns 20 s intervals."""
    dominants = interval_dominants(_grid_samples(session), session.get("duration_seconds") or 0)
    n = len(dominants)
    pad = lambda row: row + [""] * (n + 2 - len(row))  # noqa: E731 - label + n cells + comment
    rows = [
        pad(["Instructions to Observer:"]),
        pad(["1. Please watch the video in 20 seconds interval."]),
        pad(["2. Tick the most prevalent emotion in 20 second clip."]),
        pad(["3.If you notice anything unusual, please share your observations in the comments."]),
        pad([]),
        pad([]),
        pad(["Emotions", "", "", "Timeline (seconds)"])[:-1] + ["Any comments"],  # as in the template
        pad([""] + [_interval_label(i) for i in range(n)]),
    ]
    for emotion in TEMPLATE_EMOTIONS:
        rows.append([emotion] + ["X" if d == emotion else "" for d in dominants] + [""])

    comments = [
        f"{_interval_label(i)}: {_missing_text(session)} (no data)" for i, d in enumerate(dominants) if d is None
    ]
    rows.append(pad([]))
    rows.append(pad([_duration_note(session)]))
    rows.append(pad([
        "Source: speech emotion model (scikit-learn, RAVDESS); a model estimate of vocal tone, not a verified emotional state."
        if _mode(session) == "audio" else
        "Source: facial emotion model (FER+), most frequent emotion per interval; not a verified emotional state."]))
    for note in _speech_notes(session):
        rows.append(pad([note]))
    for line in comments:
        rows.append(pad([line]))

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return output_path


def _template_story(session: Dict[str, Any], style: Dict[str, ParagraphStyle]) -> List[Any]:
    """Page 1 of the PDF: the observer template, filled in from real predictions.

    Same fields as the CSV; intervals are laid out six per table, like the sheet.
    """
    dominants = interval_dominants(_grid_samples(session), session.get("duration_seconds") or 0)
    story: List[Any] = [Paragraph("Instructions to Observer:", style["h2"])]
    for line in ("1. Please watch the video in 20 seconds interval.",
                 "2. Tick the most prevalent emotion in 20 second clip.",
                 "3. If you notice anything unusual, please share your observations in the comments."):
        story.append(Paragraph(line, style["body"]))
    story.append(Spacer(1, 8))

    for start in range(0, len(dominants), 6):
        chunk = dominants[start:start + 6]
        # Same two header rows as the sheet: "Emotions / Timeline / Any comments", then the intervals.
        rows = [["Emotions", "Timeline (seconds)"] + [""] * (len(chunk) - 1) + ["Any comments"],
                [""] + [_interval_label(start + i) for i in range(len(chunk))] + [""]]
        for emotion in TEMPLATE_EMOTIONS:
            rows.append([emotion] + ["X" if d == emotion else "" for d in chunk] + [""])
        missing = [_interval_label(start + i) for i, d in enumerate(chunk) if d is None]
        if missing:
            rows[2][-1] = Paragraph(_missing_text(session).capitalize() + " (no data): " + ", ".join(missing), style["note"])
        widths = [24 * mm] + [16 * mm] * len(chunk) + [None]
        table = Table(rows, colWidths=widths, repeatRows=2)
        table.setStyle(_table_style())
        table.setStyle(TableStyle([
            ("GRID", (0, 1), (-1, -1), 0.4, RULE),
            ("ALIGN", (1, 2), (len(chunk), -1), "CENTER"),
            ("FONTNAME", (1, 2), (len(chunk), -1), "Helvetica-Bold"),
            ("SPAN", (1, 0), (len(chunk), 0)),
            ("ALIGN", (1, 0), (len(chunk), 0), "CENTER"),
            ("SPAN", (-1, 2), (-1, -1)),
            ("BACKGROUND", (0, 0), (-1, 1), BAND),
        ]))
        story.append(table)
        story.append(Spacer(1, 10))

    source = ("Speech emotion: speech model estimate per 4-second window, not a verified emotional state."
              if _mode(session) == "audio" else
              "Facial emotion: most frequent FER+ prediction in each interval (ties: higher mean confidence). "
              "A model estimate of expression, not a verified emotional state.")
    notes = [_duration_note(session), source] + _speech_notes(session)
    for note in notes:
        story.append(Paragraph(note, style["note"]))
    return story
