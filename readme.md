# JarvisChat

A lightweight, self-hosted replacement for Open-WebUI designed for local hardware.
Zero bloat. Minimal dependencies. Optimized for Python 3.13 on Debian 13 (Trixie).

## Features

- **Persistent User Profile** — Automatically injects user context (hardware specs, rules of engagement, identity) into every chat.
- **System Prompt Presets** — Toggle between "Coding Companion", "Sysadmin", or custom personas instantly.
- **Streaming Chat** — Real-time token streaming with stop-token support to prevent hallucination.
- **Conversation History** — Stored locally in SQLite (`jarvischat.db`). No external DB required.
- **Model Switcher** — Select any installed Ollama model (DeepSeek, Llama3, Mistral) per request.
- **Responsive UI** — Full Markdown support, syntax highlighting, and copy-to-clipboard for code blocks.
- **LAN Accessible** — Host on `jarvis`, access from `llamadev` (Windows/WSL), mobile, or any network node.

## Requirements

- **OS:** Linux (Tested on Debian 13 Trixie / Ubuntu 24.04)
- **Python:** 3.11+ (Compatible with Python 3.13)
- **Backend:** Ollama running locally (`localhost:11434`)

## Installation

### Method 1: Quick Install (Script)

```bash
# On jarvis (as your standard user):
cd /opt/jarvischat
./install.sh
```

### Method 2: Manual Install (Virtual Environment)

Since Debian 13 enforces PEP 668, you must use a virtual environment.

```bash
# 1. Prepare directory
mkdir -p ~/jarvischat
cp app.py ~/jarvischat/
cd ~/jarvischat

# 2. Create and activate venv
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies (FastAPI, Uvicorn, HTTPX)
pip install fastapi uvicorn httpx

# 4. Run manually to test
./venv/bin/python3 app.py
```

*Access the UI via browser at:* `http://<IP-ADDRESS>:8080`

## Systemd Service (Auto-Start)

To run JarvisChat as a background service:

1. Copy `jarvischat.service` to `/etc/systemd/system/`.
2. Update the `User=` and `ExecStart=` paths in the file.
3. Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now jarvischat
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web Interface |
| `GET` | `/api/models` | List available Ollama models |
| `GET` | `/api/ps` | Show currently loaded model |
| `GET/PUT` | `/api/profile` | **Get/Update user profile context** |
| `GET/PUT` | `/api/settings` | **Get/Update global settings** |
| `GET/POST` | `/api/presets` | **Manage system prompt presets** |
| `GET` | `/api/conversations` | List chat history |
| `POST` | `/api/conversations` | Start new conversation |
| `POST` | `/api/chat` | Send message (SSE Streaming) |

## Files Structure

- `app.py` — Single-file application (FastAPI backend + Embedded HTML/JS frontend).
- `jarvischat.db` — SQLite database (Created automatically on first run).
- `jarvischat.service` — Systemd unit file.
- `requirements.txt` — Dependency list.
