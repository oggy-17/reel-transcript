# Instagram Reel Transcriber

A web application that:
- Downloads audio from Instagram Reels
- Transcribes audio using [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- Provides both a **web UI** and a **FastAPI endpoint**

## Local Development

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # on Windows

# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app:app --host 0.0.0.0 --port 8000
