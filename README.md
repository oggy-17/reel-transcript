# Instagram Reel → Transcript (CLI + API)

This small app takes one or more Instagram **Reels URLs**, downloads the audio with `yt-dlp`, and transcribes it locally with **faster‑whisper** (no external API required). You can run it as a **CLI** or as a **FastAPI** web service.

> ⚠️ **Important:** Only process content you own or have permission to use. Respect Instagram/Meta **Terms of Service**, copyright laws, and privacy rules. Private reels require your own session cookies.

---

## 1) Quick Start (CLI)

### Prerequisites
- Python 3.10+
- `ffmpeg` installed on your system
- `pip install -r requirements.txt`

### Run
```bash
python app.py "https://www.instagram.com/reel/XXXXXXXXX/"
```

**Options**
- Force language: `--language en` (otherwise auto-detect)
- Use cookies (for private reels): `--cookies /path/to/cookies.txt`
- Model size: `--model-size small` (tiny/base/small/medium/large-v3)
- Compute type: `--compute-type int8` (int8/int8_float16/float16/float32)

The CLI prints the plain‑text transcript and saves an `.srt` (subtitles) file next to the downloaded media.

---

## 2) Run as an API

```bash
uvicorn app:app --reload --port 8000
# or: python app.py --serve
```

### Request
`POST /transcribe` with JSON:
```json
{
  "urls": ["https://www.instagram.com/reel/XXXXXXXX/"],
  "language": null,
  "cookies_path": null,
  "model_size": "small",
  "compute_type": "int8"
}
```

### Response (excerpt)
```json
{
  "results": [{
    "url": "...",
    "duration": 14.26,
    "language": "en",
    "text": "Full transcript text...",
    "segments": [
      {"id": 0, "start": 0.0, "end": 3.12, "text": "Hello everyone ..."},
      {"id": 1, "start": 3.12, "end": 6.75, "text": "In this video ..."}
    ],
    "srt_path": "/tmp/igdl_xxxxx/abc123.srt"
  }]
}
```

---

## 3) Docker (optional)

```bash
docker build -t ig-reel-transcriber .
docker run --rm -p 8000:8000 -e MODEL_SIZE=small -e COMPUTE_TYPE=int8 ig-reel-transcriber
```

Then call `POST /transcribe` as above.

To use cookies in Docker, mount them:
```bash
docker run --rm -p 8000:8000 \
  -v /absolute/path/cookies.txt:/cookies.txt \
  -e MODEL_SIZE=small -e COMPUTE_TYPE=int8 \
  ig-reel-transcriber
```
And set `"cookies_path": "/cookies.txt"` in the request JSON.

---

## 4) Notes & Tips

- **Private reels**: Export your cookies to a `cookies.txt` (Netscape format) from your browser and pass `--cookies` (CLI) or `cookies_path` (API). Only use your own account and comply with ToS.
- **Accuracy vs speed**: `small` is a good default. Use `base` for faster, `medium`/`large-v3` for better accuracy (slower & heavier).
- **GPU**: This image/app is CPU‑only by default. If you have a GPU, faster‑whisper can use it; see its docs for enabling `device="cuda"` and different `compute_type`.
- **Multiple URLs**: CLI accepts multiple; API accepts an array.
- **Output files**: The `.srt` is saved next to the downloaded media file in a temporary folder.

---

## 5) Legal & Ethical

Transcribing someone else’s content may require permission. This tool is provided “as is” for lawful use only. You are responsible for complying with all applicable agreements and laws.
