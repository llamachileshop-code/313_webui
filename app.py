#!/usr/bin/env python3
"""
JarvisChat - Lightweight Ollama Coding Companion
A minimal replacement for Open-WebUI that actually runs on Python 3.13
Talks to Ollama API on localhost:11434

Features:
  - Persistent profile/memory injected into every conversation
  - Saved system prompt presets (coding assistant, sysadmin, general, custom)
  - Streaming chat with conversation history
  - Model switching between all installed Ollama models
  - Copy-to-clipboard on code blocks
  - Token count estimates
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

# --- Configuration ---
OLLAMA_BASE = "http://localhost:11434"
DB_PATH = Path(__file__).parent / "jarvischat.db"
DEFAULT_MODEL = "deepseek-coder:6.7b"

# --- Default Profile ---
DEFAULT_PROFILE = """You are a coding companion running locally on a machine called "jarvis".

## Environment
- jarvis: Debian 13 (trixie) x86_64, AMD Ryzen 5 5600X, 16GB RAM, AMD RX 6600 XT (8GB VRAM), IP varies
- llamadev: Windows 11, primary development machine, IP 192.168.50.108, user "alphaalpaca"
- Corsair: Windows 11, gaming/streaming rig
- pivault: RPi 5, 8GB RAM, Debian 13, 11TB RAID5 NAS at /mnt/pivault, IP 192.168.50.159
- Router: ASUS ROG Rapture GT-BE98 Pro "BigBlinkyRouter" at 192.168.50.1
- Ollama runs on jarvis with GPU acceleration (ROCm), serving models on port 11434

## About the User
- Experienced developer, BS in Computer Science (Oklahoma State), coding since 1981 (TRS-80)
- Deep Unix/Linux background — wrote device drivers at SCO during Xenix era (1990s)
- Currently learning Rust, transitioning from decades of PHP
- Building a WW2 mobile game in Godot Engine for Android
- Runs a YouTube series: "Building a Professional Dev Environment with Local AI"
- Working on "Sysadmin's Wizard's Notebook" app concept in Rust
- Veteran on fixed income — prefers free/open-source solutions
- Home lab enthusiast with Z-Wave and Tapo smart home devices
- Streams Fortnite on a regular schedule

## How to Respond
- Be direct and concise — no hand-holding, this user knows what they're doing
- When showing code, prefer complete working examples over snippets
- Default to command-line solutions over GUI when possible
- Consider resource constraints (fixed income, specific hardware limits)
- Use Rust, Python, or bash unless another language is specifically needed
- Explain trade-offs when multiple approaches exist
- Don't repeat information the user clearly already knows"""

# --- Default System Prompt Presets ---
DEFAULT_PRESETS = [
    {
        "name": "Coding Companion",
        "prompt": "You are a senior software engineer and coding companion. Focus on writing clean, efficient, well-documented code. Provide complete working examples. Explain architectural decisions and trade-offs. Prefer Rust, Python, and bash."
    },
    {
        "name": "Linux Sysadmin",
        "prompt": "You are an experienced Linux systems administrator. Focus on command-line solutions, systemd services, networking, storage, and security. Prefer Debian/Ubuntu conventions. Be concise and direct."
    },
    {
        "name": "General Assistant",
        "prompt": "You are a helpful general-purpose assistant. Be clear and concise."
    }
]

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT 'New Chat',
            model TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS system_presets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            prompt TEXT NOT NULL,
            is_default INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profile (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            content TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    # Seed default profile if empty
    existing = conn.execute("SELECT id FROM profile WHERE id = 1").fetchone()
    if not existing:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("INSERT INTO profile (id, content, updated_at) VALUES (1, ?, ?)",
                      (DEFAULT_PROFILE, now))

    # Seed default presets if empty
    existing_presets = conn.execute("SELECT COUNT(*) as c FROM system_presets").fetchone()
    if existing_presets["c"] == 0:
        now = datetime.now(timezone.utc).isoformat()
        for preset in DEFAULT_PRESETS:
            conn.execute(
                "INSERT INTO system_presets (id, name, prompt, is_default, created_at) VALUES (?, ?, ?, 1, ?)",
                (str(uuid.uuid4()), preset["name"], preset["prompt"], now)
            )

    # Default settings
    defaults = {
        "profile_enabled": "true",
        "default_model": DEFAULT_MODEL,
    }
    for key, value in defaults.items():
        existing = conn.execute("SELECT key FROM settings WHERE key = ?", (key,)).fetchone()
        if not existing:
            conn.execute("INSERT INTO settings (key, value) VALUES (?, ?)", (key, value))

    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

# --- App Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="JarvisChat", lifespan=lifespan)

# --- API Routes ---

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.get("/api/models")
async def list_models():
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
            return resp.json()
        except httpx.ConnectError:
            raise HTTPException(status_code=502, detail="Cannot connect to Ollama. Is it running?")

@app.get("/api/ps")
async def running_models():
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{OLLAMA_BASE}/api/ps", timeout=10)
            return resp.json()
        except httpx.ConnectError:
            raise HTTPException(status_code=502, detail="Cannot connect to Ollama.")

# --- Profile ---

@app.get("/api/profile")
async def get_profile():
    db = get_db()
    row = db.execute("SELECT content, updated_at FROM profile WHERE id = 1").fetchone()
    db.close()
    if row:
        return {"content": row["content"], "updated_at": row["updated_at"]}
    return {"content": "", "updated_at": ""}

@app.put("/api/profile")
async def update_profile(request: Request):
    body = await request.json()
    now = datetime.now(timezone.utc).isoformat()
    db = get_db()
    db.execute("UPDATE profile SET content = ?, updated_at = ? WHERE id = 1",
               (body["content"], now))
    db.commit()
    db.close()
    return {"status": "ok", "updated_at": now}

@app.get("/api/profile/default")
async def get_default_profile():
    return {"content": DEFAULT_PROFILE}

# --- Settings ---

@app.get("/api/settings")
async def get_settings():
    db = get_db()
    rows = db.execute("SELECT key, value FROM settings").fetchall()
    db.close()
    return {row["key"]: row["value"] for row in rows}

@app.put("/api/settings")
async def update_settings(request: Request):
    body = await request.json()
    db = get_db()
    for key, value in body.items():
        db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
    db.commit()
    db.close()
    return {"status": "ok"}

# --- System Presets ---

@app.get("/api/presets")
async def list_presets():
    db = get_db()
    rows = db.execute("SELECT * FROM system_presets ORDER BY is_default DESC, name ASC").fetchall()
    db.close()
    return [dict(r) for r in rows]

@app.post("/api/presets")
async def create_preset(request: Request):
    body = await request.json()
    preset_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    db = get_db()
    db.execute(
        "INSERT INTO system_presets (id, name, prompt, is_default, created_at) VALUES (?, ?, ?, 0, ?)",
        (preset_id, body["name"], body["prompt"], now)
    )
    db.commit()
    db.close()
    return {"id": preset_id, "name": body["name"], "prompt": body["prompt"]}

@app.put("/api/presets/{preset_id}")
async def update_preset(preset_id: str, request: Request):
    body = await request.json()
    db = get_db()
    db.execute("UPDATE system_presets SET name = ?, prompt = ? WHERE id = ?",
               (body["name"], body["prompt"], preset_id))
    db.commit()
    db.close()
    return {"status": "ok"}

@app.delete("/api/presets/{preset_id}")
async def delete_preset(preset_id: str):
    db = get_db()
    db.execute("DELETE FROM system_presets WHERE id = ? AND is_default = 0", (preset_id,))
    db.commit()
    db.close()
    return {"status": "ok"}

# --- Conversation CRUD ---

@app.get("/api/conversations")
async def list_conversations():
    db = get_db()
    rows = db.execute("SELECT * FROM conversations ORDER BY updated_at DESC").fetchall()
    db.close()
    return [dict(r) for r in rows]

@app.post("/api/conversations")
async def create_conversation(request: Request):
    body = await request.json()
    conv_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    model = body.get("model", DEFAULT_MODEL)
    title = body.get("title", "New Chat")
    db = get_db()
    db.execute(
        "INSERT INTO conversations (id, title, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (conv_id, title, model, now, now)
    )
    db.commit()
    db.close()
    return {"id": conv_id, "title": title, "model": model, "created_at": now, "updated_at": now}

@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    db = get_db()
    conv = db.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    if not conv:
        db.close()
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = db.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC", (conv_id,)
    ).fetchall()
    db.close()
    return {"conversation": dict(conv), "messages": [dict(m) for m in messages]}

@app.put("/api/conversations/{conv_id}")
async def update_conversation(conv_id: str, request: Request):
    body = await request.json()
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    if "title" in body:
        db.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (body["title"], now, conv_id))
    if "model" in body:
        db.execute("UPDATE conversations SET model = ?, updated_at = ? WHERE id = ?",
                    (body["model"], now, conv_id))
    db.commit()
    db.close()
    return {"status": "ok"}

@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    db = get_db()
    db.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    db.commit()
    db.close()
    return {"status": "ok"}

# --- Chat (streaming) ---

def build_system_prompt(db, extra_prompt=""):
    """Build the full system prompt: profile + preset/custom prompt"""
    parts = []

    # Check if profile is enabled
    settings = {row["key"]: row["value"] for row in db.execute("SELECT key, value FROM settings").fetchall()}
    if settings.get("profile_enabled", "true") == "true":
        profile = db.execute("SELECT content FROM profile WHERE id = 1").fetchone()
        if profile and profile["content"].strip():
            parts.append(profile["content"].strip())

    if extra_prompt and extra_prompt.strip():
        parts.append(extra_prompt.strip())

    return "\n\n---\n\n".join(parts) if parts else ""

@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    conv_id = body.get("conversation_id")
    user_message = body.get("message", "").strip()
    model = body.get("model", DEFAULT_MODEL)
    preset_prompt = body.get("system_prompt", "")

    if not user_message:
        raise HTTPException(status_code=400, detail="Empty message")

    db = get_db()
    now = datetime.now(timezone.utc).isoformat()

    # Auto-create conversation if needed
    if not conv_id:
        conv_id = str(uuid.uuid4())
        title = user_message[:80] + ("..." if len(user_message) > 80 else "")
        db.execute(
            "INSERT INTO conversations (id, title, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (conv_id, title, model, now, now)
        )
    else:
        db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))

    # Save user message
    db.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (conv_id, "user", user_message, now)
    )
    db.commit()

    # Build message history
    history_rows = db.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conv_id,)
    ).fetchall()

    # Build system prompt (profile + preset)
    system_prompt = build_system_prompt(db, preset_prompt)
    db.close()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for row in history_rows:
        messages.append({"role": row["role"], "content": row["content"]})

    ollama_payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "stop": ["User", "User:", "Assistant:", "\nUser"]
        }
    }

    async def stream_response():
        full_response = []
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE}/api/chat",
                    json=ollama_payload,
                    timeout=httpx.Timeout(300.0, connect=10.0)
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.strip():
                            try:
                                chunk = json.loads(line)
                                if "message" in chunk and "content" in chunk["message"]:
                                    token = chunk["message"]["content"]
                                    full_response.append(token)
                                    yield f"data: {json.dumps({'token': token, 'conversation_id': conv_id})}\n\n"
                                if chunk.get("done"):
                                    assistant_msg = "".join(full_response)
                                    db2 = get_db()
                                    db2.execute(
                                        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                                        (conv_id, "assistant", assistant_msg, datetime.now(timezone.utc).isoformat())
                                    )
                                    db2.commit()
                                    db2.close()
                                    yield f"data: {json.dumps({'done': True, 'conversation_id': conv_id})}\n\n"
                            except json.JSONDecodeError:
                                pass
            except httpx.ConnectError:
                yield f"data: {json.dumps({'error': 'Cannot connect to Ollama. Is it running?'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")

# =====================================================================
# FRONTEND
# =====================================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JarvisChat</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {
    --bg-primary: #0a0e14;
    --bg-secondary: #111820;
    --bg-tertiary: #1a2230;
    --bg-hover: #1e2a3a;
    --text-primary: #c8d6e5;
    --text-secondary: #7f8fa6;
    --text-muted: #4a5568;
    --accent: #48b5e0;
    --accent-dim: #2a6f8a;
    --accent-glow: rgba(72, 181, 224, 0.15);
    --danger: #e74c3c;
    --danger-hover: #c0392b;
    --success: #2ecc71;
    --border: #1e2a3a;
    --scrollbar: #2a3a4a;
    --radius: 8px;
    --font-body: 'IBM Plex Sans', -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', 'Consolas', monospace;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: var(--font-body); background: var(--bg-primary); color: var(--text-primary); height: 100vh; overflow: hidden; display: flex; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--scrollbar); border-radius: 3px; }

/* Sidebar */
.sidebar { width: 280px; min-width: 280px; background: var(--bg-secondary); border-right: 1px solid var(--border); display: flex; flex-direction: column; height: 100vh; }
.sidebar-header { padding: 20px 16px 12px; border-bottom: 1px solid var(--border); }
.sidebar-header h1 { font-family: var(--font-mono); font-size: 18px; font-weight: 600; color: var(--accent); letter-spacing: 1px; margin-bottom: 4px; }
.sidebar-header .subtitle { font-size: 11px; color: var(--text-muted); font-family: var(--font-mono); margin-bottom: 12px; }
.btn-row { display: flex; gap: 6px; }
.new-chat-btn, .settings-btn { padding: 10px 14px; background: var(--accent-glow); border: 1px solid var(--accent-dim); border-radius: var(--radius); color: var(--accent); font-family: var(--font-body); font-size: 13px; font-weight: 500; cursor: pointer; transition: all 0.2s; }
.new-chat-btn { flex: 1; }
.settings-btn { padding: 10px 12px; }
.new-chat-btn:hover, .settings-btn:hover { background: var(--accent-dim); color: #fff; }
.conversation-list { flex: 1; overflow-y: auto; padding: 8px; }
.conv-item { padding: 10px 12px; border-radius: var(--radius); cursor: pointer; margin-bottom: 2px; display: flex; justify-content: space-between; align-items: center; transition: background 0.15s; font-size: 13px; color: var(--text-secondary); }
.conv-item:hover { background: var(--bg-hover); color: var(--text-primary); }
.conv-item.active { background: var(--bg-tertiary); color: var(--text-primary); }
.conv-item .conv-title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.conv-item .conv-delete { opacity: 0; color: var(--danger); cursor: pointer; padding: 2px 6px; font-size: 16px; }
.conv-item:hover .conv-delete { opacity: 0.7; }
.conv-item .conv-delete:hover { opacity: 1; }
.sidebar-footer { padding: 12px 16px; border-top: 1px solid var(--border); font-size: 11px; color: var(--text-muted); font-family: var(--font-mono); }

/* Main */
.main { flex: 1; display: flex; flex-direction: column; height: 100vh; min-width: 0; }
.topbar { display: flex; align-items: center; justify-content: space-between; padding: 12px 20px; border-bottom: 1px solid var(--border); background: var(--bg-secondary); gap: 12px; }
.topbar-left { display: flex; align-items: center; gap: 12px; }
.topbar-right { display: flex; align-items: center; gap: 8px; }
.topbar select { background: var(--bg-tertiary); border: 1px solid var(--border); color: var(--text-primary); font-family: var(--font-mono); font-size: 13px; padding: 6px 10px; border-radius: var(--radius); cursor: pointer; }
.topbar-label { font-size: 12px; color: var(--text-muted); font-family: var(--font-mono); text-transform: uppercase; letter-spacing: 1px; }
.profile-badge { font-size: 11px; padding: 4px 10px; border-radius: 12px; font-family: var(--font-mono); cursor: pointer; border: none; transition: all 0.2s; }
.profile-badge.on { background: rgba(46,204,113,0.15); color: var(--success); border: 1px solid rgba(46,204,113,0.3); }
.profile-badge.off { background: rgba(231,76,60,0.15); color: var(--danger); border: 1px solid rgba(231,76,60,0.3); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); display: inline-block; animation: pulse 2s infinite; }
.status-dot.offline { background: var(--danger); animation: none; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* Modal */
.modal-overlay { display:none; position:fixed; top:0;left:0;right:0;bottom:0; background:rgba(0,0,0,0.7); z-index:1000; align-items:center; justify-content:center; }
.modal-overlay.visible { display:flex; }
.modal { background:var(--bg-secondary); border:1px solid var(--border); border-radius:12px; width:90%; max-width:700px; max-height:85vh; overflow-y:auto; }
.modal-header { display:flex; justify-content:space-between; align-items:center; padding:20px 24px 16px; border-bottom:1px solid var(--border); position:sticky; top:0; background:var(--bg-secondary); z-index:1; }
.modal-header h2 { font-family:var(--font-mono); font-size:16px; color:var(--accent); }
.modal-close { background:none; border:none; color:var(--text-muted); font-size:24px; cursor:pointer; }
.modal-close:hover { color:var(--text-primary); }
.modal-body { padding: 20px 24px; }
.modal-section { margin-bottom: 24px; }
.modal-section h3 { font-family:var(--font-mono); font-size:13px; color:var(--text-secondary); text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }
.modal-section p.desc { font-size:12px; color:var(--text-muted); margin-bottom:10px; line-height:1.5; }
.modal-section textarea { width:100%; background:var(--bg-tertiary); border:1px solid var(--border); color:var(--text-primary); font-family:var(--font-mono); font-size:12px; padding:12px; border-radius:var(--radius); resize:vertical; line-height:1.6; }
.modal-section textarea:focus { outline:none; border-color:var(--accent-dim); }
.token-count { font-size:11px; color:var(--text-muted); font-family:var(--font-mono); margin-top:4px; text-align:right; }
.toggle-row { display:flex; align-items:center; justify-content:space-between; padding:8px 0; }
.toggle-label { font-size:13px; }
.toggle-switch { position:relative; width:44px; height:24px; background:var(--bg-tertiary); border:1px solid var(--border); border-radius:12px; cursor:pointer; transition:background 0.2s; }
.toggle-switch.on { background:var(--accent-dim); border-color:var(--accent-dim); }
.toggle-switch::after { content:''; position:absolute; top:2px; left:2px; width:18px; height:18px; background:var(--text-primary); border-radius:50%; transition:transform 0.2s; }
.toggle-switch.on::after { transform:translateX(20px); }
.btn-small { padding:6px 14px; border-radius:var(--radius); font-family:var(--font-mono); font-size:12px; cursor:pointer; border:1px solid var(--border); transition:all 0.2s; }
.btn-save { background:var(--accent-dim); color:#fff; border-color:var(--accent-dim); }
.btn-save:hover { background:var(--accent); }
.btn-reset { background:transparent; color:var(--text-muted); }
.btn-reset:hover { color:var(--danger); border-color:var(--danger); }
.btn-bar { display:flex; gap:8px; margin-top:10px; }
.preset-item { display:flex; align-items:center; gap:8px; padding:8px 10px; background:var(--bg-tertiary); border-radius:var(--radius); margin-bottom:6px; font-size:13px; }
.preset-item .preset-name { flex:1; color:var(--text-primary); }
.preset-item button { background:none; border:none; color:var(--text-muted); cursor:pointer; font-size:13px; padding:2px 4px; }
.preset-item button:hover { color:var(--text-primary); }

/* Chat */
.chat-container { flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:16px; }
.welcome-screen { flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; color:var(--text-muted); text-align:center; gap:12px; }
.welcome-screen .logo { font-family:var(--font-mono); font-size:48px; color:var(--accent-dim); opacity:0.5; }
.welcome-screen p { font-size:14px; max-width:420px; line-height:1.6; }
.message { display:flex; gap:12px; max-width:900px; width:100%; margin:0 auto; animation:fadeIn 0.2s ease; }
@keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
.message .avatar { width:32px; height:32px; min-width:32px; border-radius:6px; display:flex; align-items:center; justify-content:center; font-family:var(--font-mono); font-size:13px; font-weight:600; margin-top:2px; }
.message.user .avatar { background:#1a3a5c; color:var(--accent); }
.message.assistant .avatar { background:var(--accent-dim); color:#fff; }
.message .content { flex:1; min-width:0; }
.message .content .role-label { font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px; color:var(--text-muted); font-family:var(--font-mono); }
.message .content .text { font-size:14px; line-height:1.65; word-wrap:break-word; overflow-wrap:break-word; }
.message .content .text pre { background:var(--bg-primary); border:1px solid var(--border); border-radius:var(--radius); padding:12px; margin:8px 0; overflow-x:auto; font-family:var(--font-mono); font-size:13px; line-height:1.5; position:relative; }
.message .content .text code { font-family:var(--font-mono); background:var(--bg-primary); padding:2px 5px; border-radius:3px; font-size:13px; }
.message .content .text pre code { background:none; padding:0; }
.copy-btn { position:absolute; top:6px; right:6px; background:var(--bg-tertiary); border:1px solid var(--border); color:var(--text-muted); font-family:var(--font-mono); font-size:11px; padding:3px 8px; border-radius:4px; cursor:pointer; }
.copy-btn:hover { color:var(--text-primary); }
.typing-indicator { display:inline-flex; gap:4px; padding:4px 0; }
.typing-indicator span { width:6px; height:6px; background:var(--accent-dim); border-radius:50%; animation:blink 1.4s infinite; }
.typing-indicator span:nth-child(2) { animation-delay:0.2s; }
.typing-indicator span:nth-child(3) { animation-delay:0.4s; }
@keyframes blink { 0%,80%,100%{opacity:0.3} 40%{opacity:1} }

/* Input */
.input-area { padding:16px 20px; border-top:1px solid var(--border); background:var(--bg-secondary); }
.input-row-top { max-width:900px; margin:0 auto 8px; display:flex; gap:8px; align-items:center; }
.input-row-top select { background:var(--bg-tertiary); border:1px solid var(--border); color:var(--text-secondary); font-family:var(--font-mono); font-size:11px; padding:4px 8px; border-radius:var(--radius); cursor:pointer; }
.input-row-top .preset-label { font-size:11px; color:var(--text-muted); font-family:var(--font-mono); }
.input-wrapper { max-width:900px; margin:0 auto; display:flex; gap:10px; align-items:flex-end; }
.input-wrapper textarea { flex:1; background:var(--bg-tertiary); border:1px solid var(--border); color:var(--text-primary); font-family:var(--font-body); font-size:14px; padding:12px 14px; border-radius:var(--radius); resize:none; min-height:44px; max-height:200px; line-height:1.5; }
.input-wrapper textarea:focus { outline:none; border-color:var(--accent-dim); }
.input-wrapper textarea::placeholder { color:var(--text-muted); }
.send-btn { padding:12px 20px; background:var(--accent-dim); border:none; border-radius:var(--radius); color:#fff; font-family:var(--font-mono); font-size:13px; font-weight:600; cursor:pointer; white-space:nowrap; }
.send-btn:hover { background:var(--accent); }
.stop-btn { padding:12px 20px; background:var(--danger); border:none; border-radius:var(--radius); color:#fff; font-family:var(--font-mono); font-size:13px; font-weight:600; cursor:pointer; }
.stop-btn:hover { background:var(--danger-hover); }

@media (max-width:768px) {
    .sidebar { display:none; }
    .topbar { padding:10px 14px; }
    .chat-container { padding:12px; }
    .input-area { padding:10px 12px; }
}
</style>
</head>
<body>

<aside class="sidebar" id="sidebar">
    <div class="sidebar-header">
        <h1>&#9889; JarvisChat</h1>
        <div class="subtitle">local coding companion</div>
        <div class="btn-row">
            <button class="new-chat-btn" onclick="newChat()">+ New Chat</button>
            <button class="settings-btn" onclick="openSettings()">&#9881;</button>
        </div>
    </div>
    <div class="conversation-list" id="convList"></div>
    <div class="sidebar-footer">
        <span id="ollamaStatus"><span class="status-dot offline"></span> checking...</span>
    </div>
</aside>

<!-- Settings Modal -->
<div class="modal-overlay" id="settingsModal">
    <div class="modal">
        <div class="modal-header">
            <h2>Settings</h2>
            <button class="modal-close" onclick="closeSettings()">&times;</button>
        </div>
        <div class="modal-body">
            <div class="modal-section">
                <h3>Profile / Memory</h3>
                <p class="desc">This context is injected as a system prompt into every conversation. It tells the model who you are, your environment, and how you want responses. Edit freely.</p>
                <div class="toggle-row">
                    <span class="toggle-label">Inject profile into all chats</span>
                    <div class="toggle-switch on" id="profileToggle" onclick="toggleProfile()"></div>
                </div>
                <textarea id="profileEditor" rows="18" spellcheck="false"></textarea>
                <div class="token-count" id="profileTokenCount"></div>
                <div class="btn-bar">
                    <button class="btn-small btn-save" onclick="saveProfile()">Save Profile</button>
                    <button class="btn-small btn-reset" onclick="resetProfile()">Reset to Default</button>
                </div>
            </div>

            <div class="modal-section">
                <h3>System Prompt Presets</h3>
                <p class="desc">Presets add extra instructions on top of your profile. Select one in the chat to specialize behavior.</p>
                <div id="presetList"></div>
                <div class="btn-bar" style="margin-top:12px;">
                    <button class="btn-small btn-save" onclick="addPreset()">+ Add Preset</button>
                </div>
            </div>

            <div class="modal-section">
                <h3>General</h3>
                <div class="toggle-row">
                    <span class="toggle-label">Default model</span>
                    <select id="defaultModelSetting" onchange="saveDefaultModel()"></select>
                </div>
            </div>
        </div>
    </div>
</div>

<main class="main">
    <div class="topbar">
        <div class="topbar-left">
            <span class="topbar-label">Model</span>
            <select id="modelSelect"></select>
        </div>
        <div class="topbar-right">
            <button class="profile-badge on" id="profileBadge" onclick="toggleProfile()" title="Toggle profile injection">PROFILE ON</button>
        </div>
    </div>

    <div class="chat-container" id="chatContainer">
        <div class="welcome-screen" id="welcomeScreen">
            <div class="logo">&#9889;</div>
            <p>JarvisChat &mdash; your local coding companion.<br>Profile context is injected automatically.<br>Pick a model and start building.</p>
        </div>
    </div>

    <div class="input-area">
        <div class="input-row-top">
            <span class="preset-label">PRESET</span>
            <select id="presetSelect">
                <option value="">None (profile only)</option>
            </select>
        </div>
        <div class="input-wrapper">
            <textarea id="userInput" placeholder="Type a message... (Shift+Enter for new line)" rows="1" autofocus></textarea>
            <button class="send-btn" id="sendBtn" onclick="sendMessage()">SEND</button>
        </div>
    </div>
</main>

<script>
let currentConvId = null;
let isStreaming = false;
let abortController = null;
let profileEnabled = true;
let presets = [];

document.addEventListener('DOMContentLoaded', async () => {
    await loadModels();
    await loadSettings();
    await loadProfile();
    await loadPresets();
    await loadConversations();
    checkOllamaStatus();
    setInterval(checkOllamaStatus, 30000);
});

async function checkOllamaStatus() {
    try {
        const resp = await fetch('/api/ps');
        const data = await resp.json();
        const el = document.getElementById('ollamaStatus');
        const models = data.models || [];
        el.innerHTML = models.length > 0
            ? '<span class="status-dot"></span> ' + models.map(m => m.name).join(', ')
            : '<span class="status-dot"></span> Ollama ready';
    } catch {
        document.getElementById('ollamaStatus').innerHTML = '<span class="status-dot offline"></span> Ollama offline';
    }
}

async function loadModels() {
    try {
        const resp = await fetch('/api/models');
        const data = await resp.json();
        const select = document.getElementById('modelSelect');
        const settingSelect = document.getElementById('defaultModelSetting');
        select.innerHTML = '';
        settingSelect.innerHTML = '';
        (data.models || []).forEach(m => {
            const gb = (m.size / (1024*1024*1024)).toFixed(1);
            select.add(new Option(m.name + ' (' + gb + 'GB)', m.name));
            settingSelect.add(new Option(m.name, m.name));
        });
    } catch {}
}

async function loadSettings() {
    try {
        const resp = await fetch('/api/settings');
        const s = await resp.json();
        profileEnabled = s.profile_enabled !== 'false';
        updateProfileUI();
        if (s.default_model) {
            document.getElementById('modelSelect').value = s.default_model;
            document.getElementById('defaultModelSetting').value = s.default_model;
        }
    } catch {}
}

async function saveSettings() {
    await fetch('/api/settings', {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ profile_enabled: profileEnabled ? 'true' : 'false' })
    });
}

async function saveDefaultModel() {
    const model = document.getElementById('defaultModelSetting').value;
    await fetch('/api/settings', {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ default_model: model })
    });
}

async function loadProfile() {
    try {
        const resp = await fetch('/api/profile');
        const data = await resp.json();
        document.getElementById('profileEditor').value = data.content || '';
        updateTokenCount();
    } catch {}
}

async function saveProfile() {
    const content = document.getElementById('profileEditor').value;
    await fetch('/api/profile', {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ content })
    });
    updateTokenCount();
    // Flash the save button
    const btn = event.target;
    btn.textContent = 'Saved!';
    setTimeout(() => btn.textContent = 'Save Profile', 1500);
}

async function resetProfile() {
    if (!confirm('Reset profile to default? This overwrites your current profile.')) return;
    try {
        const resp = await fetch('/api/profile/default');
        const data = await resp.json();
        document.getElementById('profileEditor').value = data.content;
        await saveProfile();
    } catch {}
}

function toggleProfile() {
    profileEnabled = !profileEnabled;
    updateProfileUI();
    saveSettings();
}

function updateProfileUI() {
    const badge = document.getElementById('profileBadge');
    const toggle = document.getElementById('profileToggle');
    badge.className = 'profile-badge ' + (profileEnabled ? 'on' : 'off');
    badge.textContent = profileEnabled ? 'PROFILE ON' : 'PROFILE OFF';
    if (toggle) toggle.className = 'toggle-switch' + (profileEnabled ? ' on' : '');
}

function updateTokenCount() {
    const text = document.getElementById('profileEditor').value;
    const tokens = Math.round(text.length / 4);
    document.getElementById('profileTokenCount').textContent = '~' + tokens + ' tokens';
}

document.getElementById('profileEditor')?.addEventListener('input', updateTokenCount);

async function loadPresets() {
    try {
        const resp = await fetch('/api/presets');
        presets = await resp.json();
        renderPresetList();
        renderPresetSelect();
    } catch {}
}

function renderPresetList() {
    const container = document.getElementById('presetList');
    container.innerHTML = '';
    presets.forEach(p => {
        const div = document.createElement('div');
        div.className = 'preset-item';
        div.innerHTML = '<span class="preset-name">' + escapeHtml(p.name) + '</span>' +
            '<div class="preset-actions">' +
            '<button onclick="editPreset(\'' + p.id + '\')" title="Edit">&#9998;</button>' +
            (p.is_default ? '' : '<button onclick="deletePreset(\'' + p.id + '\')" title="Delete">&times;</button>') +
            '</div>';
        container.appendChild(div);
    });
}

function renderPresetSelect() {
    const select = document.getElementById('presetSelect');
    const current = select.value;
    select.innerHTML = '<option value="">None (profile only)</option>';
    presets.forEach(p => select.add(new Option(p.name, p.id)));
    select.value = current;
}

async function addPreset() {
    const name = prompt('Preset name:');
    if (!name) return;
    const p = prompt('System prompt text:');
    if (!p) return;
    await fetch('/api/presets', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name, prompt:p}) });
    await loadPresets();
}

async function editPreset(id) {
    const preset = presets.find(p => p.id === id);
    if (!preset) return;
    const name = prompt('Preset name:', preset.name);
    if (!name) return;
    const p = prompt('System prompt:', preset.prompt);
    if (p === null) return;
    await fetch('/api/presets/' + id, { method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name, prompt:p}) });
    await loadPresets();
}

async function deletePreset(id) {
    if (!confirm('Delete this preset?')) return;
    await fetch('/api/presets/' + id, { method:'DELETE' });
    await loadPresets();
}

function getSelectedPresetPrompt() {
    const id = document.getElementById('presetSelect').value;
    if (!id) return '';
    const p = presets.find(x => x.id === id);
    return p ? p.prompt : '';
}

function openSettings() { document.getElementById('settingsModal').classList.add('visible'); loadProfile(); }
function closeSettings() { document.getElementById('settingsModal').classList.remove('visible'); }
document.getElementById('settingsModal')?.addEventListener('click', e => { if (e.target.id === 'settingsModal') closeSettings(); });

async function loadConversations() {
    try {
        const resp = await fetch('/api/conversations');
        const convs = await resp.json();
        const list = document.getElementById('convList');
        list.innerHTML = '';
        convs.forEach(c => {
            const div = document.createElement('div');
            div.className = 'conv-item' + (c.id === currentConvId ? ' active' : '');
            div.innerHTML = '<span class="conv-title" onclick="loadConversation(\'' + c.id + '\')">' + escapeHtml(c.title) + '</span>' +
                '<span class="conv-delete" onclick="event.stopPropagation(); deleteConversation(\'' + c.id + '\')">&times;</span>';
            list.appendChild(div);
        });
    } catch {}
}

async function loadConversation(convId) {
    try {
        const resp = await fetch('/api/conversations/' + convId);
        const data = await resp.json();
        currentConvId = convId;
        document.getElementById('modelSelect').value = data.conversation.model;
        const container = document.getElementById('chatContainer');
        container.innerHTML = '';
        data.messages.forEach(msg => appendMessage(msg.role, msg.content, false));
        scrollToBottom();
        await loadConversations();
    } catch {}
}

async function deleteConversation(convId) {
    if (!confirm('Delete this conversation?')) return;
    await fetch('/api/conversations/' + convId, { method:'DELETE' });
    if (currentConvId === convId) { currentConvId = null; showWelcome(); }
    await loadConversations();
}

function newChat() {
    currentConvId = null;
    showWelcome();
    document.querySelectorAll('.conv-item').forEach(el => el.classList.remove('active'));
}

function showWelcome() {
    document.getElementById('chatContainer').innerHTML =
        '<div class="welcome-screen" id="welcomeScreen">' +
        '<div class="logo">&#9889;</div>' +
        '<p>JarvisChat &mdash; your local coding companion.<br>Profile context is injected automatically.<br>Pick a model and start building.</p>' +
        '</div>';
}

async function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    if (!message || isStreaming) return;

    const model = document.getElementById('modelSelect').value;
    const presetPrompt = getSelectedPresetPrompt();

    const welcome = document.getElementById('welcomeScreen');
    if (welcome) welcome.remove();

    appendMessage('user', message, true);
    input.value = '';
    input.style.height = 'auto';

    const assistantDiv = appendMessage('assistant', '', true);
    const textEl = assistantDiv.querySelector('.text');
    textEl.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    setStreamingState(true);

    try {
        abortController = new AbortController();
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({ conversation_id: currentConvId, message, model, system_prompt: presetPrompt }),
            signal: abortController.signal
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        let firstToken = true;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const text = decoder.decode(value, { stream: true });
            for (const line of text.split('\n')) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const data = JSON.parse(line.slice(6));
                    if (data.error) { textEl.textContent = 'Error: ' + data.error; setStreamingState(false); return; }
                    if (data.conversation_id && !currentConvId) { currentConvId = data.conversation_id; await loadConversations(); }
                    if (data.token) {
                        if (firstToken) { textEl.innerHTML = ''; firstToken = false; }
                        fullText += data.token;
                        textEl.innerHTML = renderMarkdown(fullText);
                        scrollToBottom();
                    }
                    if (data.done) { addCopyButtons(assistantDiv); setStreamingState(false); await loadConversations(); checkOllamaStatus(); }
                } catch {}
            }
        }
    } catch (e) {
        if (e.name === 'AbortError') textEl.innerHTML += '<br><em style="color:var(--text-muted)">[stopped]</em>';
        else textEl.textContent = 'Error: ' + e.message;
        setStreamingState(false);
    }
}

function setStreamingState(streaming) {
    isStreaming = streaming;
    const btn = document.getElementById('sendBtn');
    if (streaming) {
        btn.textContent = 'STOP'; btn.className = 'stop-btn';
        btn.onclick = () => { if (abortController) abortController.abort(); setStreamingState(false); };
    } else {
        btn.textContent = 'SEND'; btn.className = 'send-btn'; btn.onclick = sendMessage;
    }
}

function appendMessage(role, content, animate) {
    const container = document.getElementById('chatContainer');
    const div = document.createElement('div');
    div.className = 'message ' + role;
    if (!animate) div.style.animation = 'none';
    div.innerHTML = '<div class="avatar">' + (role==='user'?'YOU':'AI') + '</div>' +
        '<div class="content"><div class="role-label">' + role + '</div>' +
        '<div class="text">' + (content ? renderMarkdown(content) : '') + '</div></div>';
    container.appendChild(div);
    if (content && role === 'assistant') addCopyButtons(div);
    scrollToBottom();
    return div;
}

function renderMarkdown(text) {
    let h = escapeHtml(text);
    h = h.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre data-lang="$1"><code>$2</code></pre>');
    h = h.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
    h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    h = h.replace(/\*(.+?)\*/g, '<em>$1</em>');
    h = h.replace(/\n/g, '<br>');
    h = h.replace(/\\n/g, '<br>');
    return h;
}

function addCopyButtons(msgDiv) {
    msgDiv.querySelectorAll('pre').forEach(pre => {
        if (pre.querySelector('.copy-btn')) return;
        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.textContent = 'copy';
        btn.onclick = () => {
            navigator.clipboard.writeText(pre.querySelector('code')?.textContent || pre.textContent)
                .then(() => { btn.textContent = 'copied!'; setTimeout(() => btn.textContent = 'copy', 1500); });
        };
        pre.style.position = 'relative';
        pre.appendChild(btn);
    });
}

function escapeHtml(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }
function scrollToBottom() { const c = document.getElementById('chatContainer'); c.scrollTop = c.scrollHeight; }

const userInput = document.getElementById('userInput');
userInput.addEventListener('input', function() { this.style.height = 'auto'; this.style.height = Math.min(this.scrollHeight, 200) + 'px'; });
userInput.addEventListener('keydown', function(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
