#!/usr/bin/env python3
import argparse
import os
import re
import tempfile
from typing import List, Optional, Dict, Any
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
import uvicorn
import yt_dlp
from faster_whisper import WhisperModel

# ----------------- cookie config -----------------
# For Render (cloud):
#  - Use one of these env vars:
#    COOKIES_FILE=/etc/secrets/cookies.txt      (preferred: Secret File path)
#    COOKIES_TEXT="<entire cookies.txt content>" (fallback: env var)
#
# For local:
#    COOKIES_BROWSER=chrome|edge|firefox
#
COOKIES_FILE = os.environ.get("COOKIES_FILE", "").strip()
COOKIES_TEXT = os.environ.get("COOKIES_TEXT", "")
LOCAL_COOKIES_BROWSER = os.environ.get("COOKIES_BROWSER", "").strip().lower()

# If COOKIES_TEXT is set, materialize it to a temp file once:
_COOKIES_TEXT_FILE = None
if COOKIES_TEXT:
    _tmp = tempfile.mkdtemp(prefix="cookies_env_")
    _COOKIES_TEXT_FILE = os.path.join(_tmp, "cookies.txt")
    with open(_COOKIES_TEXT_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write(COOKIES_TEXT)

# ---------- URL cleaning & validation ----------
INSTAGRAM_RE = re.compile(
    r"^https?://(www\.)?instagram\.com/reels?/[A-Za-z0-9_\-]+/?$",
    re.IGNORECASE,
)

def _clean_instagram_reel_url(url: str) -> str:
    parsed = urlsplit(url)
    parts = [p for p in parsed.path.split("/") if p]
    try:
        if "reel" in parts:
            i = parts.index("reel")
        elif "reels" in parts:
            i = parts.index("reels")
        else:
            raise ValueError
        reel_id = parts[i + 1]
        cleaned_path = f"/reel/{reel_id}"
    except Exception:
        raise ValueError("URL does not contain a valid Instagram reel id.")
    return urlunsplit((parsed.scheme or "https", parsed.netloc or "www.instagram.com", cleaned_path, "", ""))

# ---------- Whisper model (lazy) ----------
_MODEL = None
def get_model(model_size: str = "small", compute_type: str = "int8") -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = WhisperModel(model_size, device="cpu", compute_type=compute_type)
    return _MODEL

# ---------- Core functions ----------
def _resolve_cookiefile(explicit_path: Optional[str]) -> Optional[str]:
    """Pick the best cookie source based on env/args for this process."""
    # 1) explicit path from API/CLI
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    # 2) COOKIES_FILE (Render Secret File)
    if COOKIES_FILE and os.path.exists(COOKIES_FILE):
        return COOKIES_FILE
    # 3) COOKIES_TEXT materialized
    if _COOKIES_TEXT_FILE and os.path.exists(_COOKIES_TEXT_FILE):
        return _COOKIES_TEXT_FILE
    # 4) no file; maybe we'll use cookiesfrombrowser locally
    return None

def download_audio(url: str, cookies_path: Optional[str] = None) -> str:
    url = _clean_instagram_reel_url(url)
    if not INSTAGRAM_RE.match(url):
        raise ValueError("This does not look like a valid Instagram Reel URL.")

    tmpdir = tempfile.mkdtemp(prefix="igdl_")
    outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")

    ydl_opts: Dict[str, Any] = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": outtmpl,
        "quiet": True,
        "nocheckcertificate": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    }

    cookiefile = _resolve_cookiefile(cookies_path)

    if cookiefile:
        ydl_opts["cookiefile"] = cookiefile
    elif LOCAL_COOKIES_BROWSER in {"chrome", "edge", "firefox"}:
        # Only works on your local machine; on cloud this does nothing.
        ydl_opts["cookiesfrombrowser"] = (LOCAL_COOKIES_BROWSER,)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    return filename

def transcribe(audio_path: str, language: Optional[str], model_size: str, compute_type: str) -> Dict[str, Any]:
    model = get_model(model_size=model_size, compute_type=compute_type)
    segments, info = model.transcribe(audio_path, language=language, beam_size=1)
    result_segments = []
    full_text_parts = []
    for seg in segments:
        text = seg.text.strip()
        result_segments.append({
            "id": seg.id,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": text
        })
        if text:
            full_text_parts.append(text)
    return {
        "duration": round(getattr(info, "duration", 0.0), 2) if hasattr(info, "duration") else None,
        "language": getattr(info, "language", None) or language,
        "segments": result_segments,
        "text": " ".join(full_text_parts).strip()
    }

def write_srt(segments: List[Dict[str, Any]], out_path: str) -> str:
    def fmt(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
        ms = int(round((t - int(t)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\n{seg['text']}\n\n")
    return out_path

# ---------- FastAPI ----------
class TranscribeRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    urls: List[str]
    language: Optional[str] = Field(default=None)
    cookies_path: Optional[str] = Field(default=None)  # optional override
    model_size: str = Field(default=os.environ.get("MODEL_SIZE", "small"))
    compute_type: str = Field(default=os.environ.get("COMPUTE_TYPE", "int8"))

class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str

class TranscribeResult(BaseModel):
    url: str
    duration: Optional[float]
    language: Optional[str]
    text: str
    segments: List[Segment]
    srt_path: Optional[str] = None

class BatchResponse(BaseModel):
    results: List[TranscribeResult]

app = FastAPI(title="Instagram Reel Transcriber", version="1.4.0")

FORM_HTML = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reel → Transcript</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:720px;margin:40px auto;padding:0 16px}
h1{font-size:1.6rem;margin-bottom:0.5rem}
form{display:grid;gap:12px}
input,button{font:inherit;padding:10px;border-radius:10px;border:1px solid #ccc}
button{cursor:pointer}
small{color:#666}
pre{white-space:pre-wrap;background:#f6f6f6;border-radius:10px;padding:12px}
</style>
<h1>Instagram Reel → Transcript</h1>
<form method="post" action="/submit">
  <label>Reel URL
    <input name="url" placeholder="https://www.instagram.com/reel/XXXXXXXX/">
  </label>
  <label>Language (optional)
    <input name="language" placeholder="en, it, ...">
  </label>
  <button type="submit">Transcribe</button>
  <small>Only process content you’re allowed to use.</small>
</form>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return FORM_HTML

@app.get("/submit")
def submit_get_redirect():
    return RedirectResponse(url="/")

@app.post("/submit", response_class=HTMLResponse)
def submit(url: str = Form(...), language: Optional[str] = Form(None)):
    try:
        media_path = download_audio(url, None)
        data = transcribe(media_path, language, os.environ.get("MODEL_SIZE","small"), os.environ.get("COMPUTE_TYPE","int8"))
        srt_path = os.path.splitext(media_path)[0] + ".srt"
        write_srt(data["segments"], srt_path)
        cleaned = _clean_instagram_reel_url(url)
        transcript = data.get("text","")
        return HTMLResponse(
            FORM_HTML + f"<h2>Transcript</h2><pre>{transcript}</pre><p><small>URL: {cleaned}</small></p>"
        )
    except Exception as e:
        return HTMLResponse(FORM_HTML + f"<p style='color:#c00'>Error: {e}</p>", status_code=400)

@app.post("/transcribe", response_model=BatchResponse)
def api_transcribe(req: TranscribeRequest):
    out_results: List[TranscribeResult] = []
    for raw_url in req.urls:
        try:
            media_path = download_audio(raw_url, req.cookies_path)
            data = transcribe(media_path, req.language, req.model_size, req.compute_type)
            srt_path = os.path.splitext(media_path)[0] + ".srt"
            write_srt(data["segments"], srt_path)
            out_results.append(TranscribeResult(
                url=_clean_instagram_reel_url(raw_url),
                duration=data.get("duration"),
                language=data.get("language"),
                text=data.get("text", ""),
                segments=[Segment(**s) for s in data.get("segments", [])],
                srt_path=srt_path
            ))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed for {raw_url}: {e}")
    return BatchResponse(results=out_results)

# ---------- CLI ----------
def cli():
    parser = argparse.ArgumentParser(description="Instagram Reel -> Transcript")
    parser.add_argument("urls", nargs="*", help="One or more Instagram Reel URLs")
    parser.add_argument("--language","-l", default=None, help="Force language code (e.g., en, it). Default: auto-detect")
    parser.add_argument("--cookies", default=None, help="Path to cookies.txt for private reels (optional).")
    parser.add_argument("--model-size", default=os.environ.get("MODEL_SIZE","small"), help="Whisper model size (tiny/base/small/medium/large-v3)")
    parser.add_argument("--compute-type", default=os.environ.get("COMPUTE_TYPE","int8"), help="Compute type (int8/int8_float16/float16/float32)")
    parser.add_argument("--serve", action="store_true", help="Run the API/web server instead of the CLI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    if args.serve:
        uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
        return

    if not args.urls:
        parser.error("You must pass at least one URL unless using --serve")

    for raw_url in args.urls:
        try:
            print(f"Downloading: {raw_url}")
            media_path = download_audio(raw_url, args.cookies)
            print(f"Saved media: {media_path}")
            print("Transcribing...")
            data = transcribe(media_path, args.language, args.model_size, args.compute_type)
            srt_path = os.path.splitext(media_path)[0] + ".srt"
            write_srt(data["segments"], srt_path)
            print("----- TRANSCRIPT (plain text) -----")
            print(data["text"])
            print("-----------------------------------")
            print(f"SRT saved to: {srt_path}")
        except Exception as e:
            print(f"[ERROR] {raw_url}: {e}")

if __name__ == "__main__":
    cli()
