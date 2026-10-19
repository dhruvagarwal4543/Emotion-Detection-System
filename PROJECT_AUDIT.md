# Project Audit

Last updated 2026-10-19. Everything below was checked on this Mac unless it is listed under "Not verified".

## Architecture

| Part | Where | What it does |
|---|---|---|
| Web app / routes | `app.py` | Flask, bound to 127.0.0.1. Upload, list, download, delete, session save. |
| Face emotion | `utils/video_emotion.py` | YuNet finds the face, FER+ (ONNX) classifies 7 emotions. Per-tab smoothing history. |
| Speech emotion | `utils/audio_emotion.py`, `train_audio_model.py` | 623 hand-built audio features into a scikit-learn MLP trained on RAVDESS (5 classes). |
| Merge | `utils/media_merge.py` | Local `ffmpeg` mux to H.264/AAC MP4. |
| Reports | `utils/report.py` | PDF (ReportLab) and CSV, both in the observer-template layout. |
| UI | `templates/index.html`, `static/` | Light, restrained dashboard. Plain HTML, CSS, JS, no build step. |
| Tests | `tests/test_app.py` | 42 `unittest` tests, synthetic data plus one RAVDESS clip and local ffmpeg. |

Nothing leaves the machine: no cloud service, no external API, no database.

## Models (as found)

- Face: `emotion-ferplus-8.onnx` + `face_detection_yunet_2023mar.onnx`. Output order is fixed by the network (`FERPLUS_LABELS`). "Contempt" is dropped, leaving the 7 classes Neutral, Happy, Surprise, Sad, Angry, Disgust, Fear. A prior correction (`PRIOR_STRENGTH = 1.1`) divides out the model's neutral/happy bias.
- Speech: `audio_emotion_model.joblib`, 5 classes (Neutral, Happy, Sad, Angry, Fear). Disgust and Surprise are **not** supported and are never reported. Held-out accuracy stored in the model file: **59.47%** (speaker-independent split by actor, from the training script). The scores are `predict_proba` outputs of an MLP, which are not calibrated probabilities.
- The face accuracy figures in code comments (49.7% to 58.8% recall, 56.8% to 60.5% accuracy) come from earlier work. They were **not re-measured** in this audit.

## Changes made

1. Reports follow the supplied `Emotion_timeline.csv`. Rows are Happy, Sad, Neutral, Anger, Fear, Disgust, Surprise. Columns are 0-20, 21-40, ... in 20 s steps and grow with recording length.
   - An `X` marks the most frequent facial emotion in the interval (ties go to the higher mean confidence).
   - The comment column lists intervals with no face.
   - The speech result is a single 4 s window of the whole recording, so it is a note, not split by interval.
   - The PDF's first page is the same grid (six intervals per table, like the sheet), followed by the earlier detail pages.
2. CSV and PDF are built independently. If one fails, the other is still saved, the failure is returned as a warning, and no half-written file is left.
3. Session data from the browser is validated: non-object JSON, bad samples, and non-numeric duration no longer cause errors or 500s.
4. `safe_join` rejects `.` (the folder itself). Delete only removes regular files.
5. Face smoothing is kept per browser tab (`sid`), so two tabs do not mix. At most 16 tabs are tracked.
6. Removed dead code (`probe_duration`, an unused import).
7. UI: light theme, section navigation, skip link, visible focus states, reduced-motion support, an on-page disclaimer, CSV download card. All element ids used by the JS are unchanged.

## Verification

Run with: `venv/bin/python -m unittest discover tests`

- 42 unit and integration tests: **pass** (last run during this audit). They include a 240 s recording giving 12 consecutive intervals, interval boundaries, CSV/PDF download headers and bytes, a failed PDF or CSV, a failed merge (originals and reports kept, no partial MP4) and malformed sessions.
- Real audio model on a RAVDESS clip: returns a valid class and scores that sum to 100.
- Real ffmpeg merge: synthetic WebM video + RAVDESS audio with a 0.5 s offset gives one video and one audio stream, with the duration expected from `-shortest`. A corrupt video file raises a clear error.
- Real merge of existing recordings (`video_20260813_125459.webm` + `audio_20260813_125459.wav`) written to a temp file: H.264 + AAC, both streams start at 0, duration 10.75 s (the audio length; MediaRecorder WebM files report no duration). Source files were not modified.
- Flask app started locally: `/`, static files, `/api/status` (both models and ffmpeg ready) and `/api/recordings` return 200.
- JS syntax: `node --check` passes.
- Browser run (Chrome driven by puppeteer-core with Chrome's **fake** camera and microphone), against the running app:
  - Page loads with no JS errors; both model badges show ready.
  - Camera starts in two tabs at once, and the server keeps one smoothing history per tab id.
  - A 7 s combined session saves the video, audio, merged MP4, PDF and CSV with no warnings. The result shows five download cards, including the CSV one.
  - The History tab lists the new files, and the Reports filter shows the PDF and CSV.
  - The CSV header matches `Emotion_timeline.csv` exactly. The PDF's first page shows the same grid (checked by rendering it).
  - The fake camera has no face, so every interval was correctly marked "no face detected". An interval holding a real "X" is covered by the unit tests, not the browser run.
  - Files created by this run were deleted afterwards. Existing recordings and reports were not touched.

## Not verified

- Live webcam and microphone with a real person. The browser check used Chrome's **fake** camera and microphone, which show a test pattern with no face. Pause/resume does not exist in this app, so it was not added or tested.
- Face-emotion accuracy and audio/video sync judged by ear or eye. Sync was only checked through stream timestamps and durations.
- Safari and Firefox. Windows and Linux.
- Long recordings (ffmpeg timeout is 300 s).

## Known limitations

- Both models estimate expression and vocal tone, not feelings. The audio model is trained on 24 acted RAVDESS speakers and is likely to do worse on other voices, accents and noisy rooms.
- Only the middle 4 seconds of a speech recording are analysed.
- Face results lean towards Neutral and Happy in this model family; the prior correction reduces this but does not remove it.
- Recordings and reports are never deleted automatically. The folder can grow large.
- The history list shows the PDF and CSV of a session as separate cards.
- `saved_*`, `models/` and `data/` are git-ignored, but this folder is not a git repository yet.

## Long recordings (3-4 minutes)

- A synthetic 240 s, 1280x720 recording (90 MB WebM + 23 MB WAV) merged in about 8 s on this Mac with a 0.3 s audio offset applied (audio stream starts at 0.279 s). The ffmpeg timeout was raised from 300 s to 900 s for slower machines. Uploads of this size are well under the 256 MB limit.
- Facial intervals: every sample is placed by its timestamp, with no cap on the number of samples, so a 240 s session gives 12 columns (`0-20` ... `221-240`) and a sample that arrives after the nominal end adds a 13th, never dropped.
- Speech: the model looks at **one 4 s window** (the middle of the speech after silence is trimmed). For a long recording that is a small part of what was said. The report states this and the recording length next to the result and does not split it by interval. Analysing the whole recording in 4 s windows would be a separate feature.
- Browsers slow down timers in a background tab. Keep the dashboard tab visible while recording, or the face sampling gets sparse (not measured).
- The 240 s recording itself was synthetic. A real 3-4 minute recording with a person was not made.

## Report types (combined, video-only, audio-only)

| Workflow | Button text | Files in `saved_reports/` | Data used |
|---|---|---|---|
| Combined session | Download PDF Report / Download CSV Report | `report_<stamp>.pdf/.csv` | Format unchanged: facial grid plus one speech note |
| Video-only (Record, then Stop & Save) | Download Video Emotion PDF Report / CSV Report | `video_report_<stamp>.pdf/.csv` | Facial predictions timed from the start of the clip; no speech content |
| Audio-only (Start Recording, then Stop & Analyze) | Download Audio Emotion PDF Report / CSV Report | `audio_report_<stamp>.pdf/.csv` | One real prediction per consecutive 4 s window, with start and end times |

- `<stamp>` is the timestamp of the recording the report describes, so History can show each recording's reports next to it. A video card never links to an audio-only report or the other way round.
- Audio-only reports fill the same observer grid with window predictions (an interval's emotion is the most frequent window emotion). Windows shorter than 1 s or quieter than a fixed loudness gate (`MIN_WINDOW_RMS`) are listed as skipped, not guessed. Disgust and Surprise rows stay empty because the speech model does not support them, and the report says so.
- The audio-only response still shows the single whole-clip prediction in the panel; the report uses the per-window predictions instead.
- If one report fails, the other still downloads, a warning is shown, and no partial file is left. If audio windowing fails, the single prediction is still shown and no audio report is offered.
- Verified with Chrome's **fake** camera and microphone: all three workflows showed their buttons, six downloads were byte-identical to the files on disk, and History linked each recording to its own reports. The fake microphone plays a beep, which the speech model labelled "Fear" at 100%; that shows the model's confidence is not a sign of correctness, and it is not a real speech result.
- Not verified: a real face and a real voice.
