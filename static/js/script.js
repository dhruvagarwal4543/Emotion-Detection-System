/* ==========================================================================
   AI Emotion Detection System - dashboard behaviour
   ==========================================================================
   Media capture happens in the browser (getUserMedia + MediaRecorder), and
   frames / clips are POSTed to the local Flask server for inference. Nothing
   is sent anywhere else.

   Audio is recorded through the Web Audio API and encoded to WAV here in the
   browser rather than using MediaRecorder's WebM/Opus output. That keeps the
   server side free of any ffmpeg dependency: librosa reads WAV natively.
   ========================================================================== */

(() => {
    "use strict";

    /* ---------------------------------------------------------------- config */

    // Inference costs 4-15 ms, so the poll rate is the only thing limiting how
    // quickly a change of expression shows up. 350 ms with the detector's
    // 3-frame smoothing means roughly a one-second response.
    const FRAME_INTERVAL_MS = 350;   // how often a frame is sent for analysis
    const FRAME_WIDTH       = 480;   // frames are downscaled before upload
    const JPEG_QUALITY      = 0.72;
    const MAX_TIMELINE      = 40;
    const TAB_ID            = Math.random().toString(36).slice(2); // keeps tabs' smoothing apart

    // The seven emotions the detector reports.
    const EMOTION_COLORS = {
        Happy: "#f5b301", Sad: "#3b82f6", Angry: "#ef4444", Fear: "#a855f7",
        Surprise: "#06b6d4", Neutral: "#94a3b8", Disgust: "#10b981"
    };

    const $ = (id) => document.getElementById(id);

    /* ----------------------------------------------------------------- state */

    const state = {
        cameraStream: null,
        analysing: false,
        analyseTimer: null,
        inFlight: false,

        videoRecorder: null,
        videoChunks: [],
        videoRecStart: 0,
        videoRecTimer: null,
        videoSamples: [],
        videoStopMs: 0,

        micStream: null,
        audioCtx: null,
        processor: null,
        analyser: null,
        sourceNode: null,
        pcmChunks: [],
        recordingAudio: false,
        audioRecStart: 0,
        audioRecTimer: null,
        waveformRaf: null,

        lastVideoEmotion: null,
        historyFilter: "all",
        recordings: [],

        // Combined session: both recorders run at once, neither uploads on its
        // own, and the blobs are handed to /api/session/save together.
        sessionActive: false,
        sessionStart: 0,
        sessionTimer: null,
        sessionSamples: [],
        videoStartMs: 0,
        audioStartMs: 0,
        resolveVideoBlob: null,
        resolveAudioBlob: null
    };

    /* ---------------------------------------------------------------- toasts */

    function toast(message, kind = "info", ttl = 4200) {
        const host = $("toastHost");
        const el = document.createElement("div");
        el.className = `toast ${kind}`;
        el.textContent = message;
        host.appendChild(el);
        setTimeout(() => {
            el.classList.add("fade");
            setTimeout(() => el.remove(), 320);
        }, ttl);
    }

    function pad2(n) { return String(n).padStart(2, "0"); }

    function formatDuration(ms) {
        const total = Math.floor(ms / 1000);
        return `${pad2(Math.floor(total / 60))}:${pad2(total % 60)}`;
    }

    /* --------------------------------------------------------- model status */

    async function loadStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();

            const videoBadge = $("statusVideo");
            if (data.video.ready) {
                videoBadge.dataset.state = "ready";
                videoBadge.textContent = "Face model ready";
                videoBadge.title = data.video.model || "";
            } else {
                videoBadge.dataset.state = "error";
                videoBadge.textContent = "Face model missing";
                videoBadge.title = data.video.error || "";
                toast("Face model failed to load. Run: python download_models.py", "error", 9000);
            }

            const audioBadge = $("statusAudio");
            if (data.audio.ready) {
                audioBadge.dataset.state = "ready";
                const acc = data.audio.accuracy
                    ? ` · ${(data.audio.accuracy * 100).toFixed(0)}% held-out`
                    : "";
                audioBadge.textContent = "Speech model ready";
                audioBadge.title = `${data.audio.model} on ${data.audio.dataset}${acc}`;
            } else {
                audioBadge.dataset.state = "error";
                audioBadge.textContent = "Speech model missing";
                audioBadge.title = data.audio.error || "";
                toast("Speech model missing. Run: python train_audio_model.py", "error", 9000);
            }
        } catch (err) {
            toast(`Could not reach the server: ${err.message}`, "error");
        }
    }

    /* ------------------------------------------------------- result rendering */

    function renderResult(prefix, emotion, confidence, hint) {
        const card = $(`${prefix}Result`);
        card.dataset.emotion = emotion || "none";

        $(`${prefix}Emotion`).textContent = emotion || "—";
        $(`${prefix}ConfValue`).textContent = `${(confidence || 0).toFixed(1)}%`;
        $(`${prefix}ConfBar`).style.width = `${Math.max(0, Math.min(100, confidence || 0))}%`;
        if (hint !== undefined) $(`${prefix}Hint`).textContent = hint;

        card.classList.add("bump");
        setTimeout(() => card.classList.remove("bump"), 260);
    }

    function renderBars(containerId, scores) {
        const container = $(containerId);
        const entries = Object.entries(scores || {}).sort((a, b) => b[1] - a[1]);
        if (!entries.length) { container.innerHTML = ""; return; }

        container.innerHTML = entries.map(([name, value], index) => `
            <div class="bar-row ${index === 0 ? "top" : ""}">
                <span>${name}</span>
                <span class="bar-track">
                    <i class="bar-fill" style="width:${value}%;background:${EMOTION_COLORS[name] || "#6366f1"}"></i>
                </span>
                <span class="bar-value">${value.toFixed(1)}%</span>
            </div>`).join("");
    }

    function addTimelineEntry(kind, emotion, confidence) {
        const timeline = $("timeline");
        const placeholder = timeline.querySelector(".empty-note");
        if (placeholder) placeholder.remove();

        const item = document.createElement("div");
        item.className = "tl-item";
        item.style.setProperty("--c", EMOTION_COLORS[emotion] || "#94a3b8");
        const now = new Date();
        item.innerHTML = `
            <span class="tl-kind">${kind === "video" ? "🎥" : "🎙️"}</span>
            <b>${emotion}</b>
            <span>${confidence.toFixed(0)}%</span>
            <time>${pad2(now.getHours())}:${pad2(now.getMinutes())}:${pad2(now.getSeconds())}</time>`;

        timeline.prepend(item);
        while (timeline.children.length > MAX_TIMELINE) timeline.lastChild.remove();
    }

    /* =========================================================== VIDEO ===== */

    const preview = $("preview");
    const overlay = $("overlay");
    const overlayCtx = overlay.getContext("2d");
    const grabCanvas = document.createElement("canvas");
    const grabCtx = grabCanvas.getContext("2d", { willReadFrequently: true });

    async function startCamera() {
        if (!navigator.mediaDevices?.getUserMedia) {
            toast("This browser does not support camera access.", "error");
            return;
        }
        $("btnCamStart").disabled = true;
        try {
            state.cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
                audio: false
            });
            preview.srcObject = state.cameraStream;
            await preview.play();

            $("stageEmpty").hidden = true;
            $("btnCamStop").disabled = false;
            $("btnRecStart").disabled = false;
            $("videoHint").textContent = "Analysing live video…";

            overlay.width = preview.videoWidth || 640;
            overlay.height = preview.videoHeight || 480;

            startAnalysisLoop();
            toast("Camera started — analysis is running locally.", "ok");
        } catch (err) {
            $("btnCamStart").disabled = false;
            const denied = err.name === "NotAllowedError";
            toast(
                denied
                    ? "Camera permission denied. Allow it in System Settings ▸ Privacy & Security ▸ Camera."
                    : `Could not start the camera: ${err.message}`,
                "error", 8000
            );
        }
    }

    function stopCamera() {
        stopAnalysisLoop();
        if (state.videoRecorder && state.videoRecorder.state === "recording") stopVideoRecording();

        if (state.cameraStream) {
            state.cameraStream.getTracks().forEach((track) => track.stop());
            state.cameraStream = null;
        }
        preview.srcObject = null;
        overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

        $("stageEmpty").hidden = false;
        $("btnCamStart").disabled = false;
        $("btnCamStop").disabled = true;
        $("btnRecStart").disabled = true;
        $("btnRecStop").disabled = true;
        $("videoFps").textContent = "idle";
        $("videoFps").classList.remove("live");

        state.lastVideoEmotion = null;
        renderResult("video", null, 0, "Camera is off");
        $("videoEmoji").textContent = "🙂";
        renderBars("videoBars", {});
    }

    function startAnalysisLoop() {
        state.analysing = true;
        $("videoFps").textContent = "live";
        $("videoFps").classList.add("live");
        tick();
    }

    function stopAnalysisLoop() {
        state.analysing = false;
        clearTimeout(state.analyseTimer);
    }

    async function tick() {
        if (!state.analysing) return;

        if (!state.inFlight && preview.videoWidth > 0) {
            state.inFlight = true;
            try {
                await analyseCurrentFrame();
            } catch (err) {
                /* transient failures should not kill the loop */
            } finally {
                state.inFlight = false;
            }
        }
        state.analyseTimer = setTimeout(tick, FRAME_INTERVAL_MS);
    }

    async function analyseCurrentFrame() {
        const scale = FRAME_WIDTH / preview.videoWidth;
        grabCanvas.width = FRAME_WIDTH;
        grabCanvas.height = Math.round(preview.videoHeight * scale);
        grabCtx.drawImage(preview, 0, 0, grabCanvas.width, grabCanvas.height);

        const dataUrl = grabCanvas.toDataURL("image/jpeg", JPEG_QUALITY);

        const res = await fetch("/api/video/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataUrl, sid: TAB_ID })
        });
        const data = await res.json();
        if (!data.success) {
            if (res.status === 503) {
                stopAnalysisLoop();
                toast(data.error || "Face model unavailable.", "error", 8000);
            }
            return;
        }

        drawFaceBoxes(data.faces || [], data.frame_size, data.emotion);

        if (!data.face_detected) {
            $("videoEmoji").textContent = "🫥";
            renderResult("video", null, 0, "No face in view — move into the frame");
            renderBars("videoBars", {});
            state.lastVideoEmotion = null;
            return;
        }

        $("videoEmoji").textContent = data.emoji || "🙂";
        renderResult("video", data.emotion, data.confidence,
                     `${data.faces.length} face${data.faces.length === 1 ? "" : "s"} detected`);
        renderBars("videoBars", data.scores);

        // Every frame counts towards the session report, even repeats - the
        // report needs the real distribution, not just the changes.
        if (!state.sessionActive && state.videoRecorder && state.videoRecorder.state === "recording") {
            // Video-only recording: same samples, timed from the start of the clip.
            state.videoSamples.push({
                t: Number(((Date.now() - state.videoRecStart) / 1000).toFixed(2)),
                emotion: data.emotion,
                confidence: data.confidence
            });
        }
        if (state.sessionActive) {
            state.sessionSamples.push({
                t: Number(((Date.now() - state.sessionStart) / 1000).toFixed(2)),
                emotion: data.emotion,
                confidence: data.confidence
            });
            $("sessionSamples").textContent =
                `${state.sessionSamples.length} facial sample${state.sessionSamples.length === 1 ? "" : "s"}`;
        }

        // Only log a timeline entry when the dominant emotion actually changes,
        // otherwise the list fills up with duplicates within seconds.
        if (data.emotion !== state.lastVideoEmotion) {
            state.lastVideoEmotion = data.emotion;
            addTimelineEntry("video", data.emotion, data.confidence);
        }
    }

    function drawFaceBoxes(faces, frameSize, emotion) {
        if (!frameSize) return;

        // The analysed frame is downscaled, so scale the boxes back up to the
        // overlay, which matches the camera's native resolution.
        if (overlay.width !== preview.videoWidth || overlay.height !== preview.videoHeight) {
            overlay.width = preview.videoWidth;
            overlay.height = preview.videoHeight;
        }
        const sx = overlay.width / frameSize.w;
        const sy = overlay.height / frameSize.h;

        overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
        if (!faces.length) return;

        const color = EMOTION_COLORS[emotion] || "#22d3ee";
        const largest = faces.reduce((a, b) => (a.w * a.h >= b.w * b.h ? a : b));

        faces.forEach((face) => {
            const x = face.x * sx, y = face.y * sy;
            const w = face.w * sx, h = face.h * sy;
            const primary = face === largest;

            overlayCtx.strokeStyle = primary ? color : "rgba(255,255,255,0.35)";
            overlayCtx.lineWidth = primary ? Math.max(3, overlay.width / 300) : 2;
            overlayCtx.shadowColor = primary ? color : "transparent";
            overlayCtx.shadowBlur = primary ? 14 : 0;

            // Corner brackets read more cleanly than a full rectangle.
            const c = Math.min(w, h) * 0.22;
            overlayCtx.beginPath();
            overlayCtx.moveTo(x, y + c);          overlayCtx.lineTo(x, y);          overlayCtx.lineTo(x + c, y);
            overlayCtx.moveTo(x + w - c, y);      overlayCtx.lineTo(x + w, y);      overlayCtx.lineTo(x + w, y + c);
            overlayCtx.moveTo(x + w, y + h - c);  overlayCtx.lineTo(x + w, y + h);  overlayCtx.lineTo(x + w - c, y + h);
            overlayCtx.moveTo(x + c, y + h);      overlayCtx.lineTo(x, y + h);      overlayCtx.lineTo(x, y + h - c);
            overlayCtx.stroke();
            overlayCtx.shadowBlur = 0;
        });
    }

    /* ------------------------------------------------------ video recording */

    function pickVideoMimeType() {
        const candidates = [
            "video/webm;codecs=vp9,opus",
            "video/webm;codecs=vp8,opus",
            "video/webm",
            "video/mp4"            // Safari
        ];
        return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
    }

    function startVideoRecording() {
        if (!state.cameraStream) return;
        const mimeType = pickVideoMimeType();
        if (!mimeType) {
            toast("This browser cannot record video.", "error");
            return;
        }

        try {
            state.videoRecorder = new MediaRecorder(state.cameraStream, { mimeType });
        } catch (err) {
            toast(`Recorder failed to start: ${err.message}`, "error");
            return;
        }

        state.videoChunks = [];
        state.videoSamples = [];
        $("videoReports").hidden = true;
        state.videoRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) state.videoChunks.push(event.data);
        };
        state.videoRecorder.onstop = finishVideoRecording;
        state.videoRecorder.start(250);

        state.videoStartMs = Date.now();
        state.videoRecStart = Date.now();
        $("videoRecPill").hidden = false;
        $("btnRecStart").disabled = true;
        $("btnRecStop").disabled = false;
        state.videoRecTimer = setInterval(() => {
            $("videoRecTime").textContent = formatDuration(Date.now() - state.videoRecStart);
        }, 250);

        toast("Recording video…", "ok", 2200);
    }

    function stopVideoRecording() {
        state.videoStopMs = Date.now();
        if (state.videoRecorder && state.videoRecorder.state !== "inactive") {
            state.videoRecorder.stop();
        }
        clearInterval(state.videoRecTimer);
        $("videoRecPill").hidden = true;
        $("videoRecTime").textContent = "00:00";
        $("btnRecStop").disabled = true;
        $("btnRecStart").disabled = !state.cameraStream;
    }

    async function finishVideoRecording() {
        const mimeType = state.videoRecorder?.mimeType || "video/webm";
        const blob = new Blob(state.videoChunks, { type: mimeType });
        state.videoChunks = [];

        // In session mode the blob is uploaded together with the audio, so it
        // is handed back to stopSession() instead of being sent on its own.
        if (state.resolveVideoBlob) {
            const resolve = state.resolveVideoBlob;
            state.resolveVideoBlob = null;
            resolve(blob.size ? blob : null);
            return;
        }

        if (blob.size === 0) {
            toast("Nothing was recorded.", "error");
            return;
        }

        const extension = mimeType.includes("mp4") ? "mp4" : "webm";
        const form = new FormData();
        form.append("video", blob, `recording.${extension}`);
        form.append("session", JSON.stringify({
            started_at: new Date(state.videoRecStart).toISOString(),
            duration_seconds: Number(((state.videoStopMs - state.videoRecStart) / 1000).toFixed(2)),
            video_samples: state.videoSamples
        }));

        toast("Saving video…", "info", 1800);
        try {
            const res = await fetch("/api/video/save", { method: "POST", body: form });
            const data = await res.json();
            if (data.success) {
                toast(`Saved to saved_videos/${data.file.name}`, "ok", 5000);
                showPanelReports("videoReports", data, "Video Emotion ");
                loadRecordings();
            } else {
                toast(data.error || "Saving the video failed.", "error");
            }
        } catch (err) {
            toast(`Upload failed: ${err.message}`, "error");
        }
    }

    /* =========================================================== AUDIO ===== */

    const waveform = $("waveform");
    const waveCtx = waveform.getContext("2d");

    async function startMicRecording() {
        $("audioReports").hidden = true;
        if (!navigator.mediaDevices?.getUserMedia) {
            toast("This browser does not support microphone access.", "error");
            return;
        }
        $("btnMicStart").disabled = true;

        try {
            state.micStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: false   // AGC would flatten the loudness cues
                }
            });
        } catch (err) {
            $("btnMicStart").disabled = false;
            const denied = err.name === "NotAllowedError";
            toast(
                denied
                    ? "Microphone permission denied. Allow it in System Settings ▸ Privacy & Security ▸ Microphone."
                    : `Could not start the microphone: ${err.message}`,
                "error", 8000
            );
            return;
        }

        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        state.audioCtx = new AudioContextClass();
        if (state.audioCtx.state === "suspended") await state.audioCtx.resume();

        state.sourceNode = state.audioCtx.createMediaStreamSource(state.micStream);
        state.analyser = state.audioCtx.createAnalyser();
        state.analyser.fftSize = 2048;
        state.sourceNode.connect(state.analyser);

        // ScriptProcessorNode is deprecated but is still the most portable way
        // to get raw PCM out of Safari and Chrome without shipping a worklet.
        state.processor = state.audioCtx.createScriptProcessor(4096, 1, 1);
        state.pcmChunks = [];
        state.processor.onaudioprocess = (event) => {
            if (!state.recordingAudio) return;
            state.pcmChunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
        };
        state.sourceNode.connect(state.processor);
        // A zero-gain sink keeps the processor pumping without echoing to the speakers.
        const mute = state.audioCtx.createGain();
        mute.gain.value = 0;
        state.processor.connect(mute);
        mute.connect(state.audioCtx.destination);

        state.recordingAudio = true;
        state.audioStartMs = Date.now();
        state.audioRecStart = Date.now();

        $("micIdle").style.display = "none";
        $("audioRecPill").hidden = false;
        $("btnMicStop").disabled = false;
        $("audioState").textContent = "recording";
        $("audioState").classList.add("live");

        state.audioRecTimer = setInterval(() => {
            $("audioRecTime").textContent = formatDuration(Date.now() - state.audioRecStart);
        }, 250);

        drawWaveform();
    }

    function drawWaveform() {
        if (!state.recordingAudio || !state.analyser) return;

        const buffer = new Uint8Array(state.analyser.fftSize);
        state.analyser.getByteTimeDomainData(buffer);

        const rect = waveform.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        if (waveform.width !== Math.round(rect.width * dpr)) {
            waveform.width = Math.round(rect.width * dpr);
            waveform.height = Math.round(rect.height * dpr);
        }
        const w = waveform.width, h = waveform.height;

        waveCtx.clearRect(0, 0, w, h);
        waveCtx.lineWidth = 2 * dpr;

        const gradient = waveCtx.createLinearGradient(0, 0, w, 0);
        gradient.addColorStop(0, "#6366f1");
        gradient.addColorStop(0.5, "#22d3ee");
        gradient.addColorStop(1, "#a855f7");
        waveCtx.strokeStyle = gradient;

        waveCtx.beginPath();
        const step = w / buffer.length;
        let peak = 0;
        for (let i = 0; i < buffer.length; i += 1) {
            const value = (buffer[i] - 128) / 128;
            peak = Math.max(peak, Math.abs(value));
            const y = h / 2 + value * (h / 2) * 0.85;
            if (i === 0) waveCtx.moveTo(0, y);
            else waveCtx.lineTo(i * step, y);
        }
        waveCtx.stroke();

        $("micLevel").style.width = `${Math.min(100, peak * 140)}%`;
        state.waveformRaf = requestAnimationFrame(drawWaveform);
    }

    async function stopMicRecording() {
        if (!state.recordingAudio) return;
        state.recordingAudio = false;

        clearInterval(state.audioRecTimer);
        cancelAnimationFrame(state.waveformRaf);

        const sampleRate = state.audioCtx.sampleRate;
        const chunks = state.pcmChunks;
        state.pcmChunks = [];

        // Tear the audio graph down before the upload so the mic light goes out.
        try {
            state.processor.disconnect();
            state.sourceNode.disconnect();
            state.analyser.disconnect();
            await state.audioCtx.close();
        } catch (err) { /* already closed */ }
        state.micStream.getTracks().forEach((track) => track.stop());
        state.micStream = null;
        state.audioCtx = null;

        $("audioRecPill").hidden = true;
        $("audioRecTime").textContent = "00:00";
        $("btnMicStop").disabled = true;
        $("btnMicStart").disabled = false;
        $("micIdle").style.display = "";
        $("micLevel").style.width = "0%";
        waveCtx.clearRect(0, 0, waveform.width, waveform.height);
        $("audioState").textContent = "analysing…";
        $("audioState").classList.remove("live");

        const totalSamples = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const longEnough = totalSamples >= sampleRate * 0.4;
        const wavBlob = longEnough ? encodeWav(chunks, totalSamples, sampleRate) : null;

        // Session mode: hand the WAV back rather than analysing it on its own.
        if (state.resolveAudioBlob) {
            const resolve = state.resolveAudioBlob;
            state.resolveAudioBlob = null;
            $("audioState").textContent = "idle";
            resolve(wavBlob);
            return;
        }

        if (!longEnough) {
            $("audioState").textContent = "idle";
            toast("That recording was too short — hold it for a second or two.", "error");
            return;
        }

        await uploadAudio(wavBlob);
    }

    /** Flattens Float32 chunks into a 16-bit PCM mono WAV blob. */
    function encodeWav(chunks, totalSamples, sampleRate) {
        const samples = new Float32Array(totalSamples);
        let offset = 0;
        for (const chunk of chunks) { samples.set(chunk, offset); offset += chunk.length; }

        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        const writeString = (pos, text) => {
            for (let i = 0; i < text.length; i += 1) view.setUint8(pos + i, text.charCodeAt(i));
        };

        writeString(0, "RIFF");
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(8, "WAVE");
        writeString(12, "fmt ");
        view.setUint32(16, 16, true);          // PCM header size
        view.setUint16(20, 1, true);           // format = PCM
        view.setUint16(22, 1, true);           // mono
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);  // byte rate
        view.setUint16(32, 2, true);           // block align
        view.setUint16(34, 16, true);          // bits per sample
        writeString(36, "data");
        view.setUint32(40, samples.length * 2, true);

        let pos = 44;
        for (let i = 0; i < samples.length; i += 1) {
            const clamped = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(pos, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
            pos += 2;
        }
        return new Blob([view], { type: "audio/wav" });
    }

    async function uploadAudio(blob) {
        const form = new FormData();
        form.append("audio", blob, "recording.wav");

        try {
            const res = await fetch("/api/audio/analyze", { method: "POST", body: form });
            const data = await res.json();
            $("audioState").textContent = "idle";

            if (!data.success) {
                toast(data.error || "Audio analysis failed.", "error", 6500);
                if (data.saved) loadRecordings();
                return;
            }

            $("audioEmoji").textContent = data.emoji || "🎧";
            renderResult("audio", data.emotion, data.confidence,
                         `${data.duration}s analysed · saved as ${data.file.name}`);
            renderBars("audioBars", data.scores);
            addTimelineEntry("audio", data.emotion, data.confidence);
            toast(`Voice emotion: ${data.emotion} (${data.confidence.toFixed(1)}%)`, "ok");
            showPanelReports("audioReports", data, "Audio Emotion ");
            loadRecordings();
        } catch (err) {
            $("audioState").textContent = "idle";
            toast(`Upload failed: ${err.message}`, "error");
        }
    }

    /* ========================================================= SESSION ===== */

    async function startSession() {
        if (state.sessionActive) return;
        $("btnSessionStart").disabled = true;

        // The camera has to be live: the session records what the preview sees
        // and the facial samples come from the running analysis loop.
        if (!state.cameraStream) {
            await startCamera();
            if (!state.cameraStream) {
                $("btnSessionStart").disabled = false;
                return;
            }
        }

        state.sessionSamples = [];
        state.sessionStart = Date.now();
        state.sessionActive = true;

        const videoBlob = new Promise((resolve) => { state.resolveVideoBlob = resolve; });
        const audioBlob = new Promise((resolve) => { state.resolveAudioBlob = resolve; });
        state.pendingVideoBlob = videoBlob;
        state.pendingAudioBlob = audioBlob;

        startVideoRecording();
        await startMicRecording();

        if (!state.recordingAudio) {
            // Microphone was refused - abandon the session cleanly rather than
            // recording a video the user did not ask for on its own.
            state.sessionActive = false;
            state.resolveVideoBlob = null;
            state.resolveAudioBlob = null;
            stopVideoRecording();
            $("btnSessionStart").disabled = false;
            return;
        }

        $("sessionStatus").hidden = false;
        $("sessionOutput").hidden = true;
        $("sessionStatusText").textContent = "Recording camera + microphone…";
        $("sessionState").textContent = "recording";
        $("sessionState").classList.add("live");
        $("btnSessionStop").disabled = false;

        state.sessionTimer = setInterval(() => {
            $("sessionTime").textContent = formatDuration(Date.now() - state.sessionStart);
        }, 250);

        toast("Session recording started.", "ok", 2500);
    }

    async function stopSession() {
        if (!state.sessionActive) return;
        $("btnSessionStop").disabled = true;
        clearInterval(state.sessionTimer);

        const durationSeconds = (Date.now() - state.sessionStart) / 1000;
        // ffmpeg needs to know which track began first, in seconds.
        const offsetSeconds = (state.audioStartMs - state.videoStartMs) / 1000;

        $("sessionStatusText").textContent = "Merging with FFmpeg and building the report…";
        $("sessionState").textContent = "processing";

        stopVideoRecording();
        await stopMicRecording();

        const [video, audio] = await Promise.all([state.pendingVideoBlob, state.pendingAudioBlob]);
        const samples = state.sessionSamples.slice();

        state.sessionActive = false;
        $("sessionState").classList.remove("live");

        const form = new FormData();
        if (video) form.append("video", video, video.type.includes("mp4") ? "rec.mp4" : "rec.webm");
        if (audio) form.append("audio", audio, "rec.wav");
        form.append("session", JSON.stringify({
            started_at: new Date(state.sessionStart).toISOString(),
            duration_seconds: Number(durationSeconds.toFixed(2)),
            audio_offset_seconds: Number(offsetSeconds.toFixed(3)),
            video_samples: samples
        }));

        try {
            const res = await fetch("/api/session/save", { method: "POST", body: form });
            const data = await res.json();
            renderSessionOutput(data, samples, durationSeconds);
            if (data.success) {
                toast("Session saved, merged and reported.", "ok", 5000);
                loadRecordings();
            } else {
                toast(data.error || "Saving the session failed.", "error", 7000);
            }
        } catch (err) {
            toast(`Session upload failed: ${err.message}`, "error", 7000);
            $("sessionStatusText").textContent = "Upload failed.";
        } finally {
            $("sessionStatus").hidden = true;
            $("sessionState").textContent = "idle";
            $("sessionTime").textContent = "00:00";
            $("btnSessionStart").disabled = false;
            $("btnSessionStop").disabled = true;
        }
    }

    function renderSessionOutput(data, samples, durationSeconds) {
        const host = $("sessionOutput");
        host.hidden = false;

        const files = data.files || {};
        const counts = {};
        samples.forEach((s) => { counts[s.emotion] = (counts[s.emotion] || 0) + 1; });
        const ranked = Object.entries(counts).sort((a, b) => b[1] - a[1]);
        const dominant = ranked.length ? ranked[0][0] : null;

        const cards = [
            files.merged && {
                icon: "🎞️", title: "Merged MP4", sub: "video + audio, synchronised",
                name: files.merged.name, url: files.merged.url, size: files.merged.size_human
            },
            files.video && {
                icon: "🎬", title: "Original video", sub: "unmodified source",
                name: files.video.name, url: files.video.url, size: files.video.size_human
            },
            files.audio && {
                icon: "🔊", title: "Original audio", sub: "unmodified source",
                name: files.audio.name, url: files.audio.url, size: files.audio.size_human
            }
        ].filter(Boolean);

        const speech = data.audio_emotion;
        const summary = `
            <div class="session-summary">
                <div><span>Length</span><b>${durationSeconds.toFixed(1)}s</b></div>
                <div><span>Facial samples</span><b>${samples.length}</b></div>
                <div><span>Dominant face</span><b style="color:${EMOTION_COLORS[dominant] || "#e5e7eb"}">
                    ${dominant || "—"}</b></div>
                <div><span>Voice emotion</span><b style="color:${EMOTION_COLORS[speech?.emotion] || "#e5e7eb"}">
                    ${speech ? `${speech.emotion} ${speech.confidence.toFixed(0)}%` : "—"}</b></div>
            </div>`;

        const warnings = (data.warnings || []).length
            ? `<p class="session-warn">⚠️ ${data.warnings.join("<br>⚠️ ")}</p>`
            : "";

        const reports = reportButtons(files, "", "PDF", "CSV");

        host.innerHTML = summary + warnings + reports + `<div class="session-files">` + cards.map((c) => `
            <a class="session-file" href="${c.url}?download=1" download="${c.name}">
                <span class="session-file-icon">${c.icon}</span>
                <span class="session-file-meta">
                    <b>${c.title}</b>
                    <span>${c.sub}</span>
                    <code>${c.name} · ${c.size}</code>
                </span>
                <span class="session-file-dl">Download</span>
            </a>`).join("") + `</div>`;
    }

    // Only a report that was really written gets an active button; a missing one
    // is shown as unavailable, with the reason in the warnings.
    function reportButtons(files, prefix, pdfLabel, csvLabel) {
        const one = (file, label) => file
            ? `<button type="button" class="btn btn-primary report-btn" data-report-url="${file.url}"
                       data-report-name="${file.name}">
                   <span class="ico">⬇</span> Download ${prefix}${label} Report
                   <small>${file.name} · ${file.size_human}</small></button>`
            : `<button type="button" class="btn report-btn" disabled>
                   <span class="ico">⚠</span> ${prefix}${label} report unavailable</button>`;
        return `<div class="report-actions" role="group" aria-label="${prefix}reports">
            ${one(files.report, pdfLabel)}${one(files.report_csv, csvLabel)}</div>`;
    }

    // Video-only and audio-only results show their own buttons right under the controls.
    function showPanelReports(hostId, data, prefix) {
        const host = $(hostId);
        const warnings = (data.warnings || []).map((w) => `<p class="session-warn">⚠️ ${w}</p>`).join("");
        host.hidden = false;
        host.innerHTML = warnings + reportButtons(data.files || {}, prefix, "PDF", "CSV");
    }

    // Fetches the file first so a failure shows a message instead of a dead tab,
    // then saves it under its real name.
    async function downloadReport(button) {
        const url = button.dataset.reportUrl;
        const name = button.dataset.reportName;
        const label = button.innerHTML;
        button.disabled = true;
        button.textContent = "Preparing…";
        try {
            const res = await fetch(`${url}?download=1`);
            if (!res.ok) throw new Error(`server answered ${res.status}`);
            const blobUrl = URL.createObjectURL(await res.blob());
            const link = Object.assign(document.createElement("a"), { href: blobUrl, download: name });
            document.body.appendChild(link);
            link.click();
            link.remove();
            setTimeout(() => URL.revokeObjectURL(blobUrl), 10000);
            toast(`Downloaded ${name}`, "ok");
            button.textContent = "✓ Downloaded - click to download again";
        } catch (err) {
            toast(`Could not download ${name}: ${err.message}`, "error", 7000);
            button.textContent = "Download failed - click to retry";
        } finally {
            button.disabled = false;
            setTimeout(() => { button.innerHTML = label; }, 4000);
        }
    }

    /* ========================================================= HISTORY ===== */

    async function loadRecordings() {
        try {
            const res = await fetch("/api/recordings");
            const data = await res.json();
            state.recordings = data.recordings || [];
            renderRecordings();
        } catch (err) {
            $("history").innerHTML = `<p class="empty-note">Could not load recordings: ${err.message}</p>`;
        }
    }

    // Report files are named <type>_<stamp>.pdf|csv and share the stamp of the recording they describe.
    const REPORT_NAME = /^(report|video_report|audio_report)_(.+)\.(pdf|csv)$/;
    const REPORT_TYPE = {
        report: "Combined", video_report: "Video-only", audio_report: "Audio-only"
    };
    // Which report types may be linked from which kind of recording; never cross over.
    const LINKS_FROM = {
        video: ["report", "video_report"], audio: ["report", "audio_report"], merged: ["report"]
    };

    function reportLinksFor(item) {
        const wanted = LINKS_FROM[item.kind];
        const match = /^(?:video|audio|session)_(.+)\.[^.]+$/.exec(item.name);
        if (!wanted || !match) return "";
        const links = state.recordings.filter((r) => {
            const m = r.kind === "report" && REPORT_NAME.exec(r.name);
            return m && m[2] === match[1] && wanted.includes(m[1]);
        }).map((r) => {
            const m = REPORT_NAME.exec(r.name);
            return `<a class="mini-btn" href="${r.url}?download=1" download="${r.name}">${
                REPORT_TYPE[m[1]]} ${m[3].toUpperCase()}</a>`;
        });
        return links.length ? `<div class="rec-card-reports">${links.join("")}</div>` : "";
    }

    function reportNote(item) {
        const m = REPORT_NAME.exec(item.name);
        return m ? `${REPORT_TYPE[m[1]]} ${m[3].toUpperCase()} emotion report` : "Emotion report";
    }

    function renderRecordings() {
        const container = $("history");
        const items = state.historyFilter === "all"
            ? state.recordings
            : state.recordings.filter((item) => item.kind === state.historyFilter);

        if (!items.length) {
            container.innerHTML = `<p class="empty-note">No ${
                state.historyFilter === "all" ? "" : state.historyFilter + " "
            }recordings saved yet.</p>`;
            return;
        }

        const ICONS = { video: "🎬", audio: "🔊", merged: "🎞️", report: "📄" };
        const PLAYABLE = {
            video: (url) => `<video src="${url}" controls preload="metadata"></video>`,
            merged: (url) => `<video src="${url}" controls preload="metadata"></video>`,
            audio: (url) => `<audio src="${url}" controls preload="metadata"></audio>`,
            report: (url, item) => `<p class="rec-card-note">${reportNote(item)} — open or download below.</p>`
        };

        container.innerHTML = items.map((item) => `
            <div class="rec-card" data-name="${item.name}" data-kind="${item.kind}">
                <div class="rec-card-head">
                    <div class="rec-card-icon">${ICONS[item.kind] || "📁"}</div>
                    <div class="rec-card-meta">
                        <div class="rec-card-name">${item.name}</div>
                        <div class="rec-card-sub">${item.created} · ${item.size_human}</div>
                    </div>
                </div>
                ${(PLAYABLE[item.kind] || PLAYABLE.report)(item.url, item)}
                ${reportLinksFor(item)}
                <div class="rec-card-actions">
                    <a class="mini-btn" href="${item.url}?download=1" download="${item.name}">Download</a>
                    <button class="mini-btn danger" data-action="delete">Delete</button>
                </div>
            </div>`).join("");
    }

    async function deleteRecording(kind, name) {
        if (!window.confirm(`Delete ${name}? This removes the file from disk.`)) return;
        try {
            const res = await fetch(`/api/recordings/${kind}/${encodeURIComponent(name)}`, {
                method: "DELETE"
            });
            const data = await res.json();
            if (data.success) {
                toast(`Deleted ${name}`, "ok");
                loadRecordings();
            } else {
                toast(data.error || "Delete failed.", "error");
            }
        } catch (err) {
            toast(`Delete failed: ${err.message}`, "error");
        }
    }

    /* ========================================================== WIRING ===== */

    $("btnCamStart").addEventListener("click", startCamera);
    $("btnCamStop").addEventListener("click", stopCamera);
    $("btnRecStart").addEventListener("click", startVideoRecording);
    $("btnRecStop").addEventListener("click", stopVideoRecording);

    $("btnMicStart").addEventListener("click", startMicRecording);
    $("btnMicStop").addEventListener("click", stopMicRecording);

    $("btnSessionStart").addEventListener("click", startSession);
    $("btnSessionStop").addEventListener("click", stopSession);

    $("btnRefreshHistory").addEventListener("click", loadRecordings);
    $("btnClearTimeline").addEventListener("click", () => {
        $("timeline").innerHTML = '<p class="empty-note">Detections from this session will appear here.</p>';
    });

    $("historyFilters").addEventListener("click", (event) => {
        const button = event.target.closest(".filter");
        if (!button) return;
        document.querySelectorAll(".filter").forEach((el) => {
            el.classList.remove("active");
            el.setAttribute("aria-pressed", "false");
        });
        button.classList.add("active");
        button.setAttribute("aria-pressed", "true");
        state.historyFilter = button.dataset.filter;
        renderRecordings();
    });

    document.addEventListener("click", (event) => {
        const button = event.target.closest("[data-report-url]");
        if (button) downloadReport(button);
    });

    $("history").addEventListener("click", (event) => {
        const button = event.target.closest('[data-action="delete"]');
        if (!button) return;
        const card = button.closest(".rec-card");
        deleteRecording(card.dataset.kind, card.dataset.name);
    });

    // Release the camera and microphone if the tab is closed mid-session.
    window.addEventListener("beforeunload", () => {
        state.cameraStream?.getTracks().forEach((track) => track.stop());
        state.micStream?.getTracks().forEach((track) => track.stop());
    });

    loadStatus();
    loadRecordings();
})();
