#!/usr/bin/env python3
# app.py – Geboorteplan-agent (volledige versie, Agents-SDK v2 ready)

from __future__ import annotations
import os, json, uuid, sqlite3, time, re, logging
from typing import List, Dict, Optional
from typing_extensions import TypedDict

from flask import Flask, request, jsonify, abort, send_file, send_from_directory, render_template
from flask_cors import CORS

from openai import OpenAI                    # >=1.14.0 (agents v2)
from agents import function_tool             # jouw eigen decorator

# ───────────────────────── logging ─────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("mae-backend")

# ───────────────────────── OpenAI config ───────────────────
client       = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID = os.getenv("ASSISTANT_ID", "asst_mBo58qbk0JCyptia1sa4nKkE")
MODEL_CHOICE = os.getenv("MODEL_CHOICE", "gpt-4o-mini")

# ───────────────────────── Flask/CORS ──────────────────────
ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
    "https://chatbotbvmp.onrender.com"
]
app = Flask(__name__, static_folder="static", template_folder="templates", static_url_path="")
CORS(app, origins=ALLOWED_ORIGINS, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# ───────────────────────── DB helpers ──────────────────────
DB_FILE = "sessions.db"
def init_db():
    with sqlite3.connect(DB_FILE) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            user_profile TEXT,
            thread_id TEXT
        )""")
init_db()

def load_state(sid: str) -> Optional[dict]:
    with sqlite3.connect(DB_FILE) as con:
        row = con.execute("SELECT state, user_profile, thread_id FROM sessions WHERE id=?", (sid,)).fetchone()
        if not row:
            return None
        state_json, profile_json, thread_id = row
        st = json.loads(state_json)
        if profile_json:
            st["user_profile"] = json.loads(profile_json)
        st["thread_id"] = thread_id
        return st

def save_state(sid: str, st: dict) -> None:
    with sqlite3.connect(DB_FILE) as con:
        st_copy = st.copy()
        profile = st_copy.pop("user_profile", None)
        thread  = st_copy.get("thread_id")
        con.execute(
            "REPLACE INTO sessions (id,state,user_profile,thread_id) VALUES (?,?,?,?)",
            (sid, json.dumps(st_copy), json.dumps(profile), thread)
        )
        con.commit()

# ───────────────────────── domein-logica (niets veranderd) ─────────────────
class NamedDescription(TypedDict):
    name: str
    description: str

DEFAULT_TOPICS: Dict[str, List[NamedDescription]] = {
    "Ondersteuning": [
        {"name": "Wie wil je bij de bevalling?",         "description": "Welke personen wil je er fysiek bij hebben?"},
        {"name": "Rol van je partner of ander persoon?", "description": "Specificeer taken of wensen voor je partner."},
        {"name": "Wil je een doula / kraamzorg?",        "description": "Extra ondersteuning tijdens en na de bevalling."},
        {"name": "Wat verwacht je van het personeel?",   "description": "Welke stijl van begeleiding past bij jou?"}
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
        {"name": "Borstvoeding",       "description": "Ondersteuning, kolven, rooming-in."},
        {"name": "Flesvoeding",        "description": "Welke melk? Wie geeft de fles?"},
        {"name": "Combinatie-voeding", "description": "Afwisselen borst en fles."},
        {"name": "Allergieën",         "description": "Rekening houden met familiaire allergieën."}
    ]
}

# ───────────────────────── in-memory sessies ─────────────────────────
SESSION: Dict[str, dict] = {}

def get_session(sid: str) -> dict:
    if sid in SESSION:
        return SESSION[sid]
    db = load_state(sid)
    if db:
        SESSION[sid] = db
        return db
    # nieuwe sessie
    SESSION[sid] = {
        "id": sid, "stage": "choose_theme",
        "themes": [], "topics": {}, "qa": [],
        "ui_theme_opts": [], "ui_topic_opts": [],
        "current_theme": None,
        "user_profile": None,
        "thread_id": None
    }
    return SESSION[sid]

def persist(sid: str) -> None:
    save_state(sid, SESSION[sid])

# ───────────────────────── Helper-functions (unchanged) ─────────────────────
def _set_theme_options(session_id: str, opts: List[str]) -> str:
    st = get_session(session_id)
    st["ui_theme_opts"] = opts
    persist(session_id)
    return "ok"

def _set_topic_options(session_id: str, theme: str, opts: List[NamedDescription]) -> str:
    st = get_session(session_id)
    st["ui_topic_opts"] = opts
    st["current_theme"] = theme
    persist(session_id)
    return "ok"

def _register_theme(session_id: str, theme: str, desc: str = "") -> str:
    st = get_session(session_id)
    if theme not in [t["name"] for t in st["themes"]]:
        st["themes"].append({"name": theme, "description": desc})
        st["stage"] = "choose_topic"
        st["current_theme"] = theme
        st["ui_topic_opts"] = DEFAULT_TOPICS.get(theme, [])
        persist(session_id)
    return "ok"

def _register_topic(session_id: str, theme: str, topic: str) -> str:
    st = get_session(session_id)
    st["topics"].setdefault(theme, [])
    if topic not in st["topics"][theme] and len(st["topics"][theme]) < 4:
        st["topics"][theme].append(topic)
        persist(session_id)
    return "ok"

def _complete_theme(session_id: str) -> str:
    st = get_session(session_id)
    if all(st["topics"].get(t["name"]) for t in st["themes"]):
        st["stage"] = "qa"
    else:
        st["stage"] = "choose_theme"
    st["current_theme"] = None
    st["ui_topic_opts"] = []
    persist(session_id)
    return "ok"

def _log_answer(session_id: str, theme: str, q: str, a: str) -> str:
    st = get_session(session_id)
    for qa in st["qa"]:
        if qa["theme"] == theme and qa["question"] == q:
            qa["answer"] = a
            break
    else:
        st["qa"].append({"theme": theme, "question": q, "answer": a})
    persist(session_id)
    return "ok"

# ───────────────────────── FunctionTool decorators ─────────────────────────
set_theme_options  = function_tool(_set_theme_options)
set_topic_options  = function_tool(_set_topic_options)
register_theme     = function_tool(_register_theme)
register_topic     = function_tool(_register_topic)
complete_theme     = function_tool(_complete_theme)
log_answer         = function_tool(_log_answer)

# ↓↓↓ verzamel de schemas direct NA de definitie – noodzakelijk voor de agent
tool_objs     = [set_theme_options, set_topic_options, register_theme,
                 register_topic, complete_theme, log_answer]
assistant_tools = [t.openai_schema for t in tool_objs]        # ← schema's
TOOL_IMPL       = {t.openai_schema["function"]["name"]: t for t in tool_objs}

# ───────────────────────── OpenAI-Assistant helpers ─────────────────────────
def create_or_get_thread(sess: dict) -> str:
    """Eén thread per sessie (blijft bewaard)."""
    if sess.get("thread_id"):
        return sess["thread_id"]
    thread = client.beta.threads.create()
    sess["thread_id"] = thread.id
    persist(sess["id"])
    return thread.id

def run_assistant(sid: str, thread_id: str, user_input: str) -> str:
    # 1. voeg user-bericht toe
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_input
    )

    # 2. start run
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
    )

    # 3. poll-lus
    while True:
        run = client.beta.threads.runs.retrieve(   # keyword-args!
            thread_id=thread_id,
            run_id=run.id
        )
        log.debug("run %s – status=%s", run.id, run.status)

        if run.status == "requires_action":
            handle_tool_calls(sid, thread_id, run)   # uitvoer hieronder
        elif run.status in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.8)

    if run.status != "completed":
        return "Er ging iets mis; probeer het nog eens."

    # 4. laatste assistant-bericht teruggeven
    msgs = client.beta.threads.messages.list(thread_id)
    last = next(m for m in reversed(msgs.data) if m.role == "assistant")
    return last.content[0].text.value

def handle_tool_calls(session_id: str, thread_id: str, run):
    calls = run.required_action.submit_tool_outputs.tool_calls
    outputs = []

    for call in calls:
        fn_name  = call.function.name
        payload  = json.loads(call.function.arguments or "{}")
        log.debug("Tool-call %s → %s(%s)", call.id, fn_name, payload)

        try:
            result = TOOL_IMPL[fn_name](**payload)
        except Exception as e:
            result = f"Error in tool {fn_name}: {e}"
            log.exception("Tool %s failed", fn_name)

        outputs.append({"tool_call_id": call.id, "output": result or "ok"})

    client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run.id,
        tool_outputs=outputs
    )

# ───────────────────────── Flask route /agent ─────────────────────────
def build_payload(st: dict, reply: str) -> dict:
    return {
        "assistant_reply": reply,
        "session_id": st["id"],
        "options": st["ui_topic_opts"] if st["stage"] == "choose_topic" else st["ui_theme_opts"],
        "current_theme": st["current_theme"],
        "themes": st["themes"],
        "topics": st["topics"],
        "qa": st["qa"],
        "stage": st["stage"]
    }

@app.post("/agent")
def agent_route():
    origin = request.headers.get("Origin")
    if origin and origin not in ALLOWED_ORIGINS:
        abort(403)

    body   = request.get_json(force=True)
    txt    = body.get("message", "") or ""
    sid    = body.get("session_id") or str(uuid.uuid4())
    sess   = get_session(sid)

    # ---------------------------------- run LLM/agent
    thread = create_or_get_thread(sess)
    reply  = run_assistant(sid, thread, txt)
    log.debug("IN  %s | %s", sid[-6:], txt)

    # ---------------------------------- front-end state blijft via tools!
    return jsonify(build_payload(sess, reply))

# ───────────────────────── Export / static files ─────────────────────────
@app.get("/export/<sid>")
def export_json(sid: str):
    st = load_state(sid)
    if not st:
        abort(404)
    path = os.path.join("/tmp", f"geboorteplan_{sid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path == "iframe":
        return render_template("iframe_page.html", backend_url=os.getenv("RENDER_EXTERNAL_URL", "http://127.0.0.1:10000"))
    full_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

# ───────────────────────── Local run ─────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)), debug=True)
