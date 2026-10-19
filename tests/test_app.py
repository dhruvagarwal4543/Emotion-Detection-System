"""Run: venv/bin/python -m unittest discover tests  (synthetic data only)."""
import csv
import io
import json
import os
import glob
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

import app as app_module
from utils import media_merge
from utils.audio_emotion import InsufficientAudioError, preprocess
from utils.report import _interval_label, build_csv, interval_dominants


def s(t, emotion="Happy", confidence=80):
    return {"t": t, "emotion": emotion, "confidence": confidence}


class Paths(unittest.TestCase):
    def test_safe_join(self):
        d = tempfile.mkdtemp()
        open(os.path.join(d, "a.wav"), "w").close()
        self.assertIsNotNone(app_module.safe_join(d, "a.wav"))
        for bad in ("..", ".", "../x", "/etc/passwd", "a/b", ""):
            self.assertIsNone(app_module.safe_join(d, bad), bad)

    def test_unique_filename(self):
        d = tempfile.mkdtemp()
        first = app_module.unique_filename(d, "v", ".webm")
        open(os.path.join(d, first), "w").close()
        self.assertNotEqual(first, app_module.unique_filename(d, "v", ".webm"))


class Routes(unittest.TestCase):
    def setUp(self):
        self.c = app_module.app.test_client()

    def test_bad_requests(self):
        self.assertEqual(self.c.post("/api/video/analyze", json=[1]).status_code, 400)
        self.assertEqual(self.c.post("/api/video/analyze", json={"image": "xx"}).status_code, 400)
        self.assertEqual(self.c.post("/api/session/save", data={"session": "[]"}).status_code, 400)
        self.assertEqual(self.c.post("/api/session/save", data={"session": "{"}).status_code, 400)
        self.assertEqual(self.c.post("/api/session/save", data={"session": "{}"}).status_code, 400)

    def test_traversal_and_directory_delete(self):
        self.assertEqual(self.c.delete("/api/recordings/video/.").status_code, 400)
        self.assertEqual(self.c.delete("/api/recordings/video/..%2Fapp.py").status_code, 400)
        self.assertEqual(self.c.delete("/api/recordings/nope/a.webm").status_code, 404)
        self.assertEqual(self.c.get("/recordings/video/..").status_code, 400)
        self.assertTrue(os.path.exists(app_module.__file__))

    def test_list(self):
        self.assertTrue(self.c.get("/api/recordings").get_json()["success"])


class Audio(unittest.TestCase):
    def test_silent_and_short(self):
        with self.assertRaises(InsufficientAudioError):
            preprocess(np.zeros(22050, dtype=np.float32))
        with self.assertRaises(InsufficientAudioError):
            preprocess(np.array([], dtype=np.float32))
        with self.assertRaises(InsufficientAudioError):
            preprocess(np.random.default_rng(0).uniform(-1, 1, 1000).astype(np.float32))


class Timeline(unittest.TestCase):
    def test_labels(self):
        self.assertEqual([_interval_label(i) for i in range(3)], ["0-20", "21-40", "41-60"])

    def test_boundaries_and_gaps(self):
        # t=0 and t=20 fall in 0-20 (Sad wins on confidence); 20.5 and 40 fall in 21-40.
        d = interval_dominants([s(0), s(20, "Sad", 90), s(20.5, "Angry"), s(40, "Angry")], 30)
        self.assertEqual(d[0], "Sad")
        self.assertEqual(d[1], "Anger")
        self.assertIsNone(d[2])
        self.assertEqual(len(d), 6)  # always at least the template's six intervals

    def test_240_second_recording_gives_12_intervals(self):
        emotions = ["Happy", "Sad", "Neutral", "Angry", "Fear", "Disgust", "Surprise", "Happy", "Sad", "Neutral", "Angry", "Fear"]
        # one real-looking prediction every 5 s; interval i is dominated by emotions[i]
        samples = [s(i * 20 + off, emotions[i]) for i in range(12) for off in (2, 7, 12, 17)]
        samples.append(s(240, "Fear"))  # the very last tick belongs to 221-240
        d = interval_dominants(samples, 240)
        self.assertEqual(len(d), 12)
        self.assertEqual(d, [{"Angry": "Anger"}.get(e, e) for e in emotions])

    def test_every_prediction_lands_in_its_own_interval(self):
        d = interval_dominants([s(20, "Sad"), s(21, "Happy"), s(220, "Fear"), s(221, "Angry"), s(240, "Angry")], 240)
        self.assertEqual((d[0], d[1], d[10], d[11]), ("Sad", "Happy", "Fear", "Anger"))
        self.assertEqual(len(d), 12)
        self.assertEqual(len(interval_dominants([s(240.4, "Sad")], 240)), 13)  # a late sample is kept, not dropped

    def test_csv_240_seconds(self):
        p = os.path.join(tempfile.mkdtemp(), "r.csv")
        samples = [s(i * 20 + 10, "Happy" if i % 2 else "Sad") for i in range(12)]
        build_csv({"video_samples": samples, "duration_seconds": 240, "audio_results": [
            {"file": "a.wav", "emotion": "Fear", "confidence": 61.5, "duration": 4.0}]}, p)
        with open(p) as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[7][1:13], [_interval_label(i) for i in range(12)])
        self.assertEqual(rows[6][0], "Emotions")
        self.assertEqual(rows[6][-1], "Any comments")
        grid = {r[0]: r[1:13] for r in rows[8:15]}
        self.assertEqual(grid["Sad"], ["X" if i % 2 == 0 else "" for i in range(12)])
        self.assertEqual(grid["Happy"], ["X" if i % 2 else "" for i in range(12)])
        text = " ".join(c for r in rows for c in r)
        self.assertIn("240.0 seconds, 12 intervals, 12 facial predictions", text)
        self.assertIn("4.0 s window of the speech in a 240 s recording", text)
        self.assertNotIn("no face detected", text)

    def test_extends_for_long_recording(self):
        self.assertEqual(len(interval_dominants([], 250)), 13)

    def test_csv_matches_template(self):
        p = os.path.join(tempfile.mkdtemp(), "r.csv")
        build_csv({"video_samples": [s(5, "Angry"), s(30, "Neutral")], "duration_seconds": 60,
                   "audio_results": []}, p)
        with open(p) as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0][0], "Instructions to Observer:")
        self.assertEqual(rows[7][1:7], ["0-20", "21-40", "41-60", "61-80", "81-100", "101-120"])
        by = {r[0]: r for r in rows[8:15]}
        self.assertEqual(list(by), ["Happy", "Sad", "Neutral", "Anger", "Fear", "Disgust", "Surprise"])
        self.assertEqual(by["Anger"][1], "X")
        self.assertEqual(by["Neutral"][2], "X")
        self.assertTrue(any("41-60: no face" in r[0] for r in rows))


WAV = (sorted(glob.glob(os.path.join(os.path.dirname(app_module.__file__), "data/ravdess/Actor_12/*.wav"))) or [None])[0]


class SessionSave(unittest.TestCase):
    """Posts to /api/session/save with every output folder redirected to a temp dir."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        names = ("SAVED_VIDEOS_DIR", "SAVED_AUDIO_DIR", "SAVED_MERGED_DIR", "SAVED_REPORTS_DIR")
        self.patches = []
        for name in names:
            path = os.path.join(self.tmp, name)
            os.makedirs(path)
            patch = mock.patch.object(app_module, name, path)
            patch.start()
            self.patches.append(patch)
        kinds = {k: (getattr(app_module, d), e) for k, (d, e) in
                 zip(("video", "audio", "merged", "report"),
                     ((n, app_module.KIND_DIRS[k][1]) for n, k in zip(names, ("video", "audio", "merged", "report"))))}
        patch = mock.patch.object(app_module, "KIND_DIRS", kinds)
        patch.start()
        self.patches.append(patch)
        self.c = app_module.app.test_client()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        shutil.rmtree(self.tmp)

    def post(self, session, **files):
        data = {"session": json.dumps(session), **files}
        return self.c.post("/api/session/save", data=data, content_type="multipart/form-data")

    def reports(self):
        return sorted(os.listdir(app_module.SAVED_REPORTS_DIR))

    def test_pdf_failure_keeps_csv(self):
        with mock.patch("utils.report.build_report", side_effect=RuntimeError("boom")):
            r = self.post({"duration_seconds": 30, "video_samples": [s(3)]},
                          audio=(io.BytesIO(b"not audio"), "a.wav"))
        j = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertIn("report_csv", j["files"])
        self.assertNotIn("report", j["files"])
        self.assertTrue(any("PDF" in w for w in j["warnings"]))
        self.assertEqual([n for n in self.reports() if n.endswith(".pdf")], [])
        self.assertEqual(self.c.get(j["files"]["report_csv"]["url"]).status_code, 200)

    def test_csv_failure_keeps_pdf(self):
        with mock.patch("utils.report.build_csv", side_effect=RuntimeError("boom")):
            j = self.post({"video_samples": [s(3)]}, audio=(io.BytesIO(b"x"), "a.wav")).get_json()
        self.assertIn("report", j["files"])
        self.assertNotIn("report_csv", j["files"])

    def test_malformed_session_data(self):
        junk = {"video_samples": [None, "x", 5, {"t": "bad", "emotion": "Happy"}, s(2)],
                "duration_seconds": "abc", "audio_results": [None, 3, "x"]}
        j = self.post(junk, audio=(io.BytesIO(b"x"), "a.wav")).get_json()
        self.assertIn("report_csv", j["files"])
        self.assertIn("report", j["files"])
        self.assertEqual(self.post({"video_samples": {"a": 1}}, audio=(io.BytesIO(b"x"), "a.wav")).status_code, 200)

    def test_downloads_are_attachments_with_the_right_name(self):
        j = self.post({"duration_seconds": 30, "video_samples": [s(3)]},
                      audio=(io.BytesIO(b"x"), "a.wav")).get_json()
        for key, mime in (("report", "application/pdf"), ("report_csv", "text/csv")):
            info = j["files"][key]
            r = self.c.get(info["url"] + "?download=1")
            self.assertEqual(r.status_code, 200)
            self.assertIn(mime, r.headers["Content-Type"])
            self.assertIn(f'filename={info["name"]}', r.headers["Content-Disposition"])
            self.assertIn("attachment", r.headers["Content-Disposition"])
            on_disk = os.path.join(app_module.SAVED_REPORTS_DIR, info["name"])
            self.assertEqual(r.data, open(on_disk, "rb").read())
        listed = {i["name"] for i in self.c.get("/api/recordings").get_json()["recordings"]}
        self.assertTrue({j["files"]["report"]["name"], j["files"]["report_csv"]["name"]} <= listed)

    def test_no_samples_still_gives_both_reports(self):
        j = self.post({"duration_seconds": 100}, audio=(io.BytesIO(b"x"), "a.wav")).get_json()
        self.assertIn("report", j["files"])
        self.assertIn("report_csv", j["files"])
        with open(os.path.join(app_module.SAVED_REPORTS_DIR, j["files"]["report_csv"]["name"])) as f:
            self.assertIn("0-20: no face detected", f.read())

    def test_merge_failure_keeps_originals_and_reports(self):
        with mock.patch("utils.media_merge.merge_video_audio", side_effect=RuntimeError("ffmpeg died")):
            j = self.post({"duration_seconds": 30, "video_samples": [s(3)]},
                          video=(io.BytesIO(b"v"), "v.webm"), audio=(io.BytesIO(b"a"), "a.wav")).get_json()
        self.assertTrue(j["success"])
        self.assertNotIn("merged", j["files"])
        self.assertTrue(any("merge" in w.lower() for w in j["warnings"]))
        self.assertEqual(len(os.listdir(app_module.SAVED_VIDEOS_DIR)), 1)
        self.assertEqual(len(os.listdir(app_module.SAVED_AUDIO_DIR)), 1)
        self.assertEqual(len(self.reports()), 2)

    def test_empty_upload_discarded_and_originals_kept(self):
        j = self.post({}, audio=(io.BytesIO(b""), "a.wav"), video=(io.BytesIO(b"v"), "v.webm")).get_json()
        self.assertIn("video", j["files"])
        self.assertNotIn("audio", j["files"])
        self.assertTrue(os.listdir(app_module.SAVED_VIDEOS_DIR))

    @unittest.skipUnless(WAV, "no RAVDESS clip available")
    def test_real_speech_prediction(self):
        j = self.post({"duration_seconds": 4}, audio=(open(WAV, "rb"), "a.wav")).get_json()
        self.assertIn(j["audio_emotion"]["emotion"], ["Neutral", "Happy", "Sad", "Angry", "Fear"])
        self.assertAlmostEqual(sum(j["audio_emotion"]["scores"].values()), 100, delta=0.5)


def read(path):
    with open(path, "rb") as f:
        return f.read()


class SeparateReports(SessionSave):
    """Video-only and audio-only workflows, each with its own PDF and CSV."""

    def get_report(self, info, mime):
        r = self.c.get(info["url"] + "?download=1")
        self.assertEqual(r.status_code, 200)
        self.assertIn(mime, r.headers["Content-Type"])
        self.assertIn(f'attachment; filename={info["name"]}', r.headers["Content-Disposition"])
        self.assertEqual(r.data, read(os.path.join(app_module.SAVED_REPORTS_DIR, info["name"])))
        return r.data

    def post_video(self, session, name="v.webm"):
        data = {"video": (io.BytesIO(b"video-bytes"), name)}
        if session is not None:
            data["session"] = json.dumps(session)
        return self.c.post("/api/video/save", data=data, content_type="multipart/form-data")

    def test_video_only_reports(self):
        samples = [s(i * 5 + 1, "Happy" if i < 8 else "Angry") for i in range(16)]
        j = self.post_video({"duration_seconds": 80, "video_samples": samples}).get_json()
        stamp = app_module.stamp_of(j["file"]["name"])
        self.assertEqual(j["files"]["report"]["name"], f"video_report_{stamp}.pdf")
        self.assertEqual(j["files"]["report_csv"]["name"], f"video_report_{stamp}.csv")
        self.assertEqual(self.get_report(j["files"]["report"], "application/pdf")[:4], b"%PDF")
        csv_text = self.get_report(j["files"]["report_csv"], "text/csv").decode()
        rows = list(csv.reader(io.StringIO(csv_text)))
        grid = {r[0]: r[1:5] for r in rows[8:15]}
        # Happy at 1-36 s fills 0-20 and 21-40; Angry at 41-76 s fills 41-60 and 61-80.
        self.assertEqual(grid["Happy"], ["X", "X", "", ""])
        self.assertEqual(grid["Anger"], ["", "", "X", "X"])
        self.assertIn("Video-only session: no audio", csv_text)
        self.assertNotIn("Speech model", csv_text)
        self.assertEqual(os.listdir(app_module.SAVED_AUDIO_DIR), [])

    def test_video_only_without_session_has_no_reports(self):
        j = self.post_video(None).get_json()
        self.assertTrue(j["success"])
        self.assertEqual(j["files"], {})
        self.assertEqual(self.reports(), [])

    def test_video_only_pdf_failure_keeps_csv_and_video(self):
        with mock.patch("utils.report.build_report", side_effect=RuntimeError("boom")):
            j = self.post_video({"duration_seconds": 30, "video_samples": [s(3)]}).get_json()
        self.assertIn("report_csv", j["files"])
        self.assertNotIn("report", j["files"])
        self.assertTrue(any("PDF" in w for w in j["warnings"]))
        self.assertEqual(len(os.listdir(app_module.SAVED_VIDEOS_DIR)), 1)

    def test_video_only_garbage_session(self):
        j = self.post_video({"duration_seconds": "x", "video_samples": [None, "a", {"t": "z"}]}).get_json()
        self.assertEqual(sorted(j["files"]), ["report", "report_csv"])
        self.assertEqual(self.post_video([1, 2]).get_json()["files"], {})

    @unittest.skipUnless(WAV, "no RAVDESS clip available")
    def test_audio_only_reports(self):
        import soundfile
        data, rate = soundfile.read(WAV, dtype="float32")
        path = os.path.join(self.tmp, "clip.wav")
        soundfile.write(path, np.concatenate([data, np.zeros(rate * 6, dtype=np.float32), data]), rate)
        j = self.c.post("/api/audio/analyze", data={"audio": (open(path, "rb"), "a.wav")},
                        content_type="multipart/form-data").get_json()
        self.assertTrue(j["success"])
        self.assertEqual(j["warnings"], [])
        stamp = app_module.stamp_of(j["file"]["name"])
        self.assertEqual(j["files"]["report_csv"]["name"], f"audio_report_{stamp}.csv")
        self.assertEqual(self.get_report(j["files"]["report"], "application/pdf")[:4], b"%PDF")
        text = self.get_report(j["files"]["report_csv"], "text/csv").decode()
        self.assertIn("consecutive 4-second windows", text)
        self.assertIn("Disgust and Surprise are not supported", text)
        self.assertIn("silent or too quiet", text)
        self.assertNotIn("Video-only", text)
        self.assertNotIn("facial", text.split("Source:")[1].split("\n")[0])
        rows = list(csv.reader(io.StringIO(text)))
        grid = {r[0]: r[1:7] for r in rows[8:15]}
        self.assertEqual(grid["Disgust"], [""] * 6)
        self.assertEqual(grid["Surprise"], [""] * 6)
        self.assertEqual(sum(1 for r in grid.values() if r[0] == "X"), 1)

    @unittest.skipUnless(WAV, "no RAVDESS clip available")
    def test_audio_window_failure_keeps_prediction_and_recording(self):
        with mock.patch("utils.audio_emotion.AudioEmotionDetector.predict_windows", side_effect=RuntimeError("x")):
            j = self.c.post("/api/audio/analyze", data={"audio": (open(WAV, "rb"), "a.wav")},
                            content_type="multipart/form-data").get_json()
        self.assertTrue(j["success"])
        self.assertIn(j["emotion"], ["Neutral", "Happy", "Sad", "Angry", "Fear"])
        self.assertEqual(j["files"], {})
        self.assertTrue(any("audio report" in w for w in j["warnings"]))
        self.assertEqual(len(os.listdir(app_module.SAVED_AUDIO_DIR)), 1)

    def test_unreadable_audio_builds_no_report(self):
        r = self.c.post("/api/audio/analyze", data={"audio": (io.BytesIO(b"not audio"), "a.wav")},
                        content_type="multipart/form-data")
        self.assertNotEqual(r.status_code, 200)
        self.assertEqual(self.reports(), [])
        self.assertEqual(len(os.listdir(app_module.SAVED_AUDIO_DIR)), 1)  # recording kept

    def test_combined_files_share_one_stamp_and_keep_their_names(self):
        j = self.post({"duration_seconds": 30, "video_samples": [s(3)]},
                      video=(io.BytesIO(b"v"), "v.webm"), audio=(io.BytesIO(b"a"), "a.wav")).get_json()
        stamp = app_module.stamp_of(j["files"]["video"]["name"])
        self.assertEqual(j["files"]["audio"]["name"], f"audio_{stamp}.wav")
        self.assertEqual(j["files"]["report"]["name"], f"report_{stamp}.pdf")
        self.assertEqual(j["files"]["report_csv"]["name"], f"report_{stamp}.csv")
        text = self.get_report(j["files"]["report_csv"], "text/csv").decode()
        self.assertIn("facial emotion model", text)  # the combined format is unchanged
        self.assertNotIn("Video-only", text)

    def test_audio_csv_covers_240_seconds(self):
        p = os.path.join(self.tmp, "a.csv")
        windows = [{"start": i * 4, "end": i * 4 + 4, "t": i * 4 + 2, "emotion": "Angry" if i < 30 else "Sad",
                    "confidence": 70} for i in range(60)]
        build_csv({"mode": "audio", "duration_seconds": 240, "audio_windows": windows, "audio_skipped": []}, p)
        rows = list(csv.reader(open(p)))
        self.assertEqual(rows[7][1:13], [_interval_label(i) for i in range(12)])
        grid = {r[0]: r[1:13] for r in rows[8:15]}
        self.assertEqual(grid["Anger"], ["X"] * 6 + [""] * 6)
        self.assertEqual(grid["Sad"], [""] * 6 + ["X"] * 6)
        self.assertEqual(grid["Disgust"], [""] * 12)


class Merge(unittest.TestCase):
    def test_missing_inputs_and_no_ffmpeg(self):
        with self.assertRaises(FileNotFoundError):
            media_merge.merge_video_audio("/no/v.webm", "/no/a.wav", "/tmp/o.mp4")
        with mock.patch.object(media_merge, "ffmpeg_path", return_value=None):
            with self.assertRaises(media_merge.FFmpegNotAvailable):
                media_merge.merge_video_audio("/no/v.webm", "/no/a.wav", "/tmp/o.mp4")

    def test_failed_merge_leaves_no_partial_mp4(self):
        tmp = tempfile.mkdtemp()
        video, audio, out = (os.path.join(tmp, n) for n in ("v.webm", "a.wav", "o.mp4"))
        open(video, "wb").write(b"v")
        open(audio, "wb").write(b"a")

        def fake_ffmpeg(*args, **kwargs):  # writes a partial file, then fails
            open(out, "wb").write(b"half")
            return mock.Mock(returncode=1, stderr="boom")

        with mock.patch.object(media_merge, "ffmpeg_path", return_value="ffmpeg"), \
                mock.patch.object(media_merge.subprocess, "run", side_effect=fake_ffmpeg):
            with self.assertRaises(RuntimeError):
                media_merge.merge_video_audio(video, audio, out)
        self.assertFalse(os.path.exists(out))
        self.assertTrue(os.path.exists(video) and os.path.exists(audio))
        shutil.rmtree(tmp)

    def test_timeout_covers_a_four_minute_recording(self):
        self.assertGreaterEqual(media_merge.FFMPEG_TIMEOUT_SECONDS, 600)

    @unittest.skipUnless(shutil.which("ffmpeg") and WAV, "needs ffmpeg and a RAVDESS clip")
    def test_real_merge_and_bad_input(self):
        tmp = tempfile.mkdtemp()
        video = os.path.join(tmp, "v.webm")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc=size=160x120:rate=15:duration=5", "-c:v", "libvpx", video], check=True)
        out = os.path.join(tmp, "o.mp4")
        media_merge.merge_video_audio(video, WAV, out, offset_seconds=0.5)

        def probe(entry):
            return subprocess.run(["ffprobe", "-v", "error", "-show_entries", entry, "-of",
                                   "default=nw=1:nk=1", out], capture_output=True, text=True).stdout.split()
        self.assertEqual(sorted(probe("stream=codec_type")), ["audio", "video"])
        audio_len = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                          "default=nw=1:nk=1", WAV], capture_output=True, text=True).stdout)
        # -shortest: audio (delayed 0.5 s) is the shorter track, so it sets the length.
        self.assertAlmostEqual(float(probe("format=duration")[0]), audio_len + 0.5, delta=0.4)

        bad = os.path.join(tmp, "bad.webm")
        open(bad, "wb").write(b"garbage")
        with self.assertRaises(RuntimeError):
            media_merge.merge_video_audio(bad, WAV, os.path.join(tmp, "o2.mp4"))
        shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
