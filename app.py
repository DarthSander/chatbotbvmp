# app.py – Geboorteplan-agent – Volledige en Geoptimaliseerde Flask-app

from __future__ import annotations
import os
import json
import uuid
import sqlite3
import time
from copy import deepcopy
from typing import List, Dict, Optional
from typing_extensions import TypedDict

from flask import (
    Flask, request, Response, jsonify, abort,
    send_file, send_from_directory, render_template
)
from flask_cors import CORS
from openai import OpenAI
from agents import Agent, Runner, function_tool

# ---------- strikt type voor thema’s en topics ----------
class NamedDescription(TypedDict):
    name: str
    description: str

# ---------- basisconfig ----------
client          = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID    = os.getenv("ASSISTANT_ID")
ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
    "https://chatbotbvmp.onrender.com"
]
DB_FILE         = "sessions.db"
MODEL_CHOICE    = "gpt-4.1"

# ---------- Flask-app setup ----------
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
    static_url_path=""
)
CORS(app, origins=ALLOWED_ORIGINS, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# ---------- standaardonderwerpen ----------
DEFAULT_TOPICS: Dict[str, List[NamedDescription]] = {
    "Ondersteuning": [
        {"name": "Wie wil je bij de bevalling?", "description": "Welke personen wil je er fysiek bij hebben?"},
        {"name": "Rol van je partner of ander persoon?", "description": "Specificeer taken of wensen voor je partner."},
        {"name": "Wil je een doula / kraamzorg?", "description": "Extra ondersteuning tijdens en na de bevalling."},
        {"name": "Wat verwacht je van het personeel?", "description": "Welke stijl van begeleiding past bij jou?"}
    ],
    "Bevalling & medisch beleid": [
        {"name": "Pijnstilling",    "description": "Medicamenteuze en niet-medicamenteuze opties."},
        {"name": "Interventies",    "description": "Bijv. inknippen, kunstverlossing, infuus."},
        {"name": "Noodsituaties",   "description": "Wat als het anders loopt dan gepland?"},
        {"name": "Placenta-keuzes", "description": "Placenta bewaren, laten staan, of doneren?"}
    ],
    "Sfeer en omgeving": [
        {"name": "Muziek & verlichting", "description": "Rustige muziek? Gedimd licht?"},
        {"name": "Privacy",              "description": "Wie mag binnenkomen en fotograferen?"},
        {"name": "Foto’s / video",       "description": "Wil je opnames laten maken?"},
        {"name": "Eigen spulletjes",     "description": "Bijv. eigen kussen, etherische olie."}
    ],
    "Voeding na de geboorte": [
        {"name": "Borstvoeding",         "description": "Ondersteuning, kolven, rooming-in."},
        {"name": "Flesvoeding",          "description": "Welke melk? Wie geeft de fles?"},
        {"name": "Combinatie-voeding",   "description": "Afwisselen borst en fles."},
        {"name": "Allergieën",           "description": "Rekening houden met familiaire allergieën."}
    ]
}

# ---------- SQLite-helper functions ----------
def init_db() -> None:
    with sqlite3.connect(DB_FILE) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                state TEXT NOT NULL
            )
        """)
init_db()

def load_state(sid: str) -> Optional[dict]:
    with sqlite3.connect(DB_FILE) as con:
        row = con.execute("SELECT state FROM sessions WHERE id=?", (sid,)).fetchone()
        return json.loads(row[0]) if row else None

def save_state(sid: str, st: dict) -> None:
    with sqlite3.connect(DB_FILE) as con:
        con.execute(
            "REPLACE INTO sessions (id, state) VALUES (?, ?)",
            (sid, json.dumps(st))
        )
        con.commit()

# ---------- In-memory sessies + persistence ----------
SESSION: Dict[str, dict] = {}

def get_session(sid: str) -> dict:
    if sid in SESSION:
        SESSION[sid].setdefault("id", sid)
        return SESSION[sid]

    if (db := load_state(sid)):
        st = db
        st.setdefault("history", [])
        st.setdefault("generated_topic_options", {})
        st.setdefault("last_interaction_timestamp", time.time())
        st["id"] = sid
        SESSION[sid] = st
        return st

    # Nieuwe sessie
    st = {
        "id": sid,
        "stage": "choose_theme",
        "themes": [],
        "topics": {},
        "qa": [],
        "history": [],
        "summary": "",
        "ui_theme_opts": [],
        "ui_topic_opts": [],
        "current_theme": None,
        "generated_topic_options": {},
        "last_interaction_timestamp": time.time()
    }
    SESSION[sid] = st
    return st

def persist(sid: str) -> None:
    save_state(sid, SESSION[sid])

# ---------- History-samenvatting (bij >40 messages) ----------
def summarize_chunk(chunk: List[dict]) -> str:
    if not chunk:
        return ""
    filtered = [m for m in chunk if isinstance(m, dict) and m.get('content')]
    text = "\n".join(f"{m['role']}: {m['content']}" for m in filtered)
    prompt = (
        "Vat dit deel van het gesprek over een geboorteplan samen in maximaal 300 tokens. "
        "Focus op de keuzes, wensen en open vragen."
    )
    r = client.chat.completions.create(
        model=MODEL_CHOICE,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": text}
        ],
        max_tokens=300
    )
    return r.choices[0].message.content.strip()

# ============================================================
# Tool-wrappers
# ============================================================
def _set_theme_options(session_id: str, options: List[str]) -> str:
    st = get_session(session_id)
    st["ui_theme_opts"] = options
    persist(session_id)
    return "ok"

def _set_topic_options(session_id: str, theme: str, options: List[NamedDescription]) -> str:
    st = get_session(session_id)
    st.setdefault("generated_topic_options", {})[theme] = options
    st["ui_topic_opts"] = options
    st["current_theme"] = theme
    persist(session_id)
    return "ok"

def _register_theme(session_id: str, theme: str, description: str = "") -> str:
    st = get_session(session_id)
    if len(st["themes"]) < 6 and theme not in [t["name"] for t in st["themes"]]:
        st["themes"].append({"name": theme, "description": description})
        st["stage"] = "choose_topic"
        st["ui_topic_opts"] = DEFAULT_TOPICS.get(theme, [])
        st["current_theme"] = theme
        persist(session_id)
    return "ok"

def _register_topic(session_id: str, theme: str, topic: str) -> str:
    st = get_session(session_id)
    lst = st["topics"].setdefault(theme, [])
    if len(lst) < 4 and topic not in lst:
        lst.append(topic)
        persist(session_id)
    return "ok"

def _complete_theme(session_id: str) -> str:
    st = get_session(session_id)
    all_ok = all(
        t["name"] in st["topics"] and st["topics"][t["name"]]
        for t in st["themes"]
    )
    st["stage"] = "qa" if all_ok else "choose_theme"
    st["ui_topic_opts"] = []
    st["current_theme"] = None
    persist(session_id)
    return "ok"

def _log_answer(session_id: str, theme: str, question: str, answer: str) -> str:
    st = get_session(session_id)
    found = False
    for qa in st["qa"]:
        if qa["theme"] == theme and qa["question"] == question:
            qa["answer"] = answer
            found = True
            break
    if not found:
        st["qa"].append({"theme": theme, "question": question, "answer": answer})
    persist(session_id)
    return "ok"

def _get_state(session_id: str) -> str:
    return json.dumps(get_session(session_id))

set_theme_options = function_tool(_set_theme_options)
set_topic_options = function_tool(_set_topic_options)
register_theme    = function_tool(_register_theme)
register_topic    = function_tool(_register_topic)
complete_theme    = function_tool(_complete_theme)
log_answer        = function_tool(_log_answer)
get_state_tool    = function_tool(_get_state)

# ============================================================
# Phase Handlers (Behavior-Tree Nodes)
# ============================================================
def handle_theme_selection(st: dict, msg: str) -> Optional[str]:
    if st["stage"] != "choose_theme":
        return None
    if not st["ui_theme_opts"]:
        set_theme_options(st["id"], list(DEFAULT_TOPICS.keys()))
        return None
    if msg in DEFAULT_TOPICS:
        register_theme(st["id"], msg, DEFAULT_TOPICS[msg][0]["description"])
        return f"Oké, thema '{msg}' is toegevoegd. Kies nu onderwerpen."
    return None

def handle_topic_selection(st: dict, msg: str) -> Optional[str]:
    if st["stage"] != "choose_topic":
        return None
    theme = st["current_theme"]
    if not st["ui_topic_opts"]:
        existing = st["generated_topic_options"].get(theme)
        if existing:
            set_topic_options(st["id"], theme, existing)
        else:
            set_topic_options(st["id"], theme, DEFAULT_TOPICS[theme])
        return None
    names = [t["name"] for t in st["ui_topic_opts"]]
    if msg in names:
        register_topic(st["id"], theme, msg)
        return f"Onderwerp '{msg}' toegevoegd aan thema '{theme}'."
    return None

def handle_complete_selection(st: dict, msg: str) -> Optional[str]:
    if msg.strip() == "user_finished_topic_selection":
        complete_theme(st["id"])
        return "Klaar met kiezen! We gaan nu verder met de vragenfase."
    return None

def handle_qa(st: dict, msg: str) -> Optional[str]:
    if st["stage"] != "qa":
        return None
    last = st["history"][-1]["content"] if st["history"] else ""
    if last.startswith("Vraag:"):
        theme = st["themes"][-1]["name"]
        log_answer(st["id"], theme, last, msg)
        return "Antwoord opgeslagen. Hier komt de volgende vraag…"
    # genereer volgende vraag (simpele placeholder)
    next_q = f"Vraag: wat is jouw wens voor {st['themes'][0]['name']}?"
    return next_q

def handle_proactive_help(st: dict, msg: str) -> Optional[str]:
    if time.time() - st.get("last_interaction_timestamp", 0) > 300:
        st["last_interaction_timestamp"] = time.time()
        return "Ik zie dat je even stilzit, kan ik ergens mee helpen?"
    return None

def handle_fallback(st: dict, msg: str) -> str:
    return "Hoe kan ik je verder helpen?"

# ============================================================
# Streaming-/chat-endpoint
# ============================================================
def stream_run(tid: str):
    with client.beta.threads.runs.stream(thread_id=tid, assistant_id=ASSISTANT_ID) as ev:
        for e in ev:
            if e.event == "thread.message.delta" and e.data.delta.content:
                yield e.data.delta.content[0].text.value

@app.post("/chat")
def chat():
    origin = request.headers.get("Origin")
    if origin and origin not in ALLOWED_ORIGINS:
        abort(403)
    data = request.get_json(force=True)
    msg = data.get("message", "")
    tid = data.get("thread_id") or client.beta.threads.create().id
    client.beta.threads.messages.create(thread_id=tid, role="user", content=msg)
    return Response(stream_run(tid), headers={"X-Thread-ID": tid}, mimetype="text/plain")

# ============================================================
# Synchronous /agent-endpoint
# ============================================================
@app.post("/agent")
def agent():
    origin = request.headers.get("Origin")
    if origin and origin not in ALLOWED_ORIGINS:
        abort(403)

    body = request.get_json(force=True)
    msg  = body.get("message", "")
    sid  = body.get("session_id") or str(uuid.uuid4())
    st   = get_session(sid)
    st["last_interaction_timestamp"] = time.time()

    # ─── Summarize Decorator ───────────────────────────────
    if len(st["history"]) > 40:
        st["summary"] += "\n" + summarize_chunk(st["history"][:-20])
        st["history"] = st["history"][-20:]
        persist(sid)

    # ─── Behavior-Tree Execution ───────────────────────────
    reply: Optional[str] = None
    for handler in (
        handle_theme_selection,
        handle_topic_selection,
        handle_complete_selection,
        handle_qa,
        handle_proactive_help
    ):
        reply = handler(st, msg)
        if reply is not None:
            break
    if reply is None:
        reply = handle_fallback(st, msg)

    # ─── Update history & persist ──────────────────────────
    st["history"].append({"role": "assistant", "content": reply})
    persist(sid)

    return jsonify({
        "assistant_reply": reply,
        "session_id": sid,
        "options": st["ui_topic_opts"] if st["stage"] == "choose_topic"
                   else st["ui_theme_opts"],
        "current_theme": st["current_theme"],
        **{k: v for k, v in st.items()
           if k not in ("ui_theme_opts", "ui_topic_opts")}
    })

# ============================================================
# Export endpoint
# ============================================================
@app.get("/export/<sid>")
def export_json(sid: str):
    st = load_state(sid)
    if not st:
        abort(404)
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"geboorteplan_{sid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))

# ============================================================
# Iframe route
# ============================================================
@app.route('/iframe')
def iframe_page():
    backend_url = os.getenv("RENDER_EXTERNAL_URL", "http://127.0.0.1:10000")
    return render_template('iframe_page.html', backend_url=backend_url)

# ============================================================
# SPA-fallback
# ============================================================
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path == "iframe":
        return iframe_page()
    full_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

# ============================================================
# Allow embedding
# ============================================================
@app.after_request
def allow_iframe(response):
    response.headers.pop("X-Frame-Options", None)
    return response

# ============================================================
# Run the app
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)