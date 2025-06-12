#!/usr/bin/env python3
# Mae – Geboorteplan-agent (function_tool-versie, GPT-4.1)

from __future__ import annotations
import os, json, uuid, sqlite3, time, logging
from typing import List, Dict, Optional
from typing_extensions import TypedDict
from flask import Flask, request, jsonify, abort, send_from_directory, send_file
from flask_cors import CORS
import openai
from agents import function_tool          # <-- weer in gebruik!

# ──────────── logging ────────────
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "DEBUG").upper(), logging.DEBUG),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("mae-backend")

# ──────────── Config ─────────────
openai.api_key  = os.getenv("OPENAI_API_KEY")
MODEL_NAME      = "gpt-4.1"
ASSISTANT_FILE  = "assistant_id.txt"
DB_FILE         = "sessions.db"
ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
    "https://chatbotbvmp.onrender.com"
]

# ──────────── Basisdata ──────────
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

# ──────────── Database ───────────
def init_db():
    with sqlite3.connect(DB_FILE) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS sessions(
                         id TEXT PRIMARY KEY,
                         thread_id TEXT,
                         state TEXT
                       )""")
init_db()

def load_session(sid: str) -> Optional[dict]:
    row = sqlite3.connect(DB_FILE).execute(
        "SELECT thread_id, state FROM sessions WHERE id=?", (sid,)
    ).fetchone()
    if not row: return None
    return {"id": sid, "thread_id": row[0], "state": json.loads(row[1] or "{}")}

def save_session(sid: str, thread: str, state: dict):
    with sqlite3.connect(DB_FILE) as con:
        con.execute("REPLACE INTO sessions(id, thread_id, state) VALUES(?,?,?)",
                    (sid, thread, json.dumps(state)))

# ──────────── In-memory state ─────
def blank_state(): return {
    "stage": "choose_theme",
    "themes": [], "topics": {}, "qa": [],
    "current_theme": None,
    "ui_theme_opts": [], "ui_topic_opts": []
}
ST: Dict[str, dict] = {}

# ──────────── Business-logic ──────
def _set_theme_options(sid: str, options: List[str]): ST[sid]["ui_theme_opts"] = options; return "ok"
def _set_topic_options(sid: str, theme: str, options: List[NamedDescription]):
    ST[sid].update(current_theme=theme, ui_topic_opts=options); return "ok"
def _register_theme(sid: str, theme: str, description: str=""):
    s=ST[sid]
    if len(s["themes"])<6 and theme not in [t["name"] for t in s["themes"]]:
        s["themes"].append({"name":theme,"description":description})
        s.update(stage="choose_topic", current_theme=theme, ui_topic_opts=DEFAULT_TOPICS.get(theme,[]))
    return "ok"
def _register_topic(sid: str, theme: str, topic: str):
    s=ST[sid]["topics"].setdefault(theme, [])
    if topic not in s and len(s)<4: s.append(topic)
    return "ok"
def _complete_theme(sid: str):
    s=ST[sid]; s.update(stage="choose_theme", current_theme=None, ui_topic_opts=[])
    if all(s["topics"].get(t["name"]) for t in s["themes"]): s["stage"]="qa"
    return "ok"
def _log_answer(sid: str, theme: str, question: str, answer: str):
    ST[sid]["qa"].append({"theme":theme,"question":question,"answer":answer}); return "ok"

# ──────────── function_tool wrappers ──────────
@function_tool
def set_theme_options(sid: str, options: List[str]) -> str:
    """Frontend: toon thema-chips"""
    return _set_theme_options(sid, options)

@function_tool
def set_topic_options(sid: str, theme: str, options: List[NamedDescription]) -> str:
    """Frontend: toon onderwerp-chips voor een thema"""
    return _set_topic_options(sid, theme, options)

@function_tool
def register_theme(sid: str, theme: str, description: str="") -> str:
    """Sla een gekozen thema op"""
    return _register_theme(sid, theme, description)

@function_tool
def register_topic(sid: str, theme: str, topic: str) -> str:
    """Sla een onderwerp onder een thema op"""
    return _register_topic(sid, theme, topic)

@function_tool
def complete_theme(sid: str) -> str:
    """Markeer huidig thema als afgerond"""
    return _complete_theme(sid)

@function_tool
def log_answer(sid: str, theme: str, question: str, answer: str) -> str:
    """Bewaar Q&A-antwoord"""
    return _log_answer(sid, theme, question, answer)

# Alle FunctionTool-objecten op één plek
tool_objs = [
    set_theme_options, set_topic_options, register_theme,
    register_topic, complete_theme, log_answer
]

# ── Robuuste schema-extractie
def get_schema(obj):
    # 1) nieuwe Agents-SDK: obj is dataclass FunctionTool
    if hasattr(obj, "name") and hasattr(obj, "params_json_schema"):
        return {
            "type":"function",
            "function":{
                "name": obj.name,
                "description": getattr(obj, "description", ""),
                "parameters": obj.params_json_schema
            }
        }
    # 2) oudere helper: dict-achtige
    if isinstance(obj, dict) and "function" in obj:
        return obj
    raise AttributeError("Onbekende FunctionTool-vorm")

assistant_tools = [get_schema(o) for o in tool_objs]

# Mapping: naam ➜ Python-functie (we kennen de originele)
TOOL_IMPL = {
    "set_theme_options":  _set_theme_options,
    "set_topic_options":  _set_topic_options,
    "register_theme":     _register_theme,
    "register_topic":     _register_topic,
    "complete_theme":     _complete_theme,
    "log_answer":         _log_answer
}

# ──────────── Assistant maken of laden ─────────
def ensure_assistant():
    if os.path.exists(ASSISTANT_FILE): return open(ASSISTANT_FILE).read().strip()
    prompt = (
        "Je bent Mae, digitale verloskundige.\n"
        "Vraag ALTIJD toestemming (‘Is het goed als ik…’) vóór je "
        "een tool oproept. Houd het menselijk en proactief."
    )
    a = openai.beta.assistants.create(name="Mae", instructions=prompt,
                                      model=MODEL_NAME, tools=assistant_tools)
    open(ASSISTANT_FILE,"w").write(a.id)
    log.info("Assistant aangemaakt %s", a.id)
    return a.id

ASSISTANT_ID = ensure_assistant()

# ──────────── Assistant-run helper ────────────
def run_assistant(sid:str, thread:str, user:str) -> str:
    openai.beta.threads.messages.create(thread_id=thread, role="user", content=user)
    run = openai.beta.threads.runs.create(thread_id=thread, assistant_id=ASSISTANT_ID)
    while True:
        run = openai.beta.threads.runs.retrieve(thread, run.id)
        if run.status == "requires_action":
            outs=[]
            for c in run.required_action.submit_tool_outputs.tool_calls:
                fn  = TOOL_IMPL[c.function.name]
                res = fn(sid, **json.loads(c.function.arguments or "{}"))
                outs.append({"tool_call_id": c.id, "output": res})
            openai.beta.threads.runs.submit_tool_outputs(thread, run.id, outputs=outs)
            continue
        if run.status in ("queued","in_progress"): time.sleep(0.4); continue
        if run.status != "completed":
            log.error("Run-fout: %s", run.last_error); return "⚠️ Er ging iets mis."
        break
    msg = openai.beta.threads.messages.list(thread, limit=1).data[0]
    return msg.content[0].text.value if msg.content else ""

# ──────────── Flask-app ───────────
app = Flask(__name__, static_folder="static", template_folder="templates", static_url_path="")
CORS(app, origins=ALLOWED_ORIGINS, allow_headers="*", methods=["GET","POST"])

@app.post("/agent")
def agent_route():
    d=request.get_json(force=True); txt=d.get("message",""); sid=d.get("session_id") or str(uuid.uuid4())
    sess=load_session(sid)
    if not sess:
        t=openai.beta.threads.create(); sess={"id":sid,"thread_id":t.id}; ST[sid]=blank_state()
    else: ST.setdefault(sid, blank_state())

    log.debug("IN  %s | %s", sid[-6:], txt)
    reply = run_assistant(sid, sess["thread_id"], txt)
    log.debug("OUT %s | %s", sid[-6:], reply[:80])
    save_session(sid, sess["thread_id"], ST[sid])

    return jsonify({
        "session_id": sid,
        "assistant_reply": reply,
        "stage": ST[sid]["stage"],
        "options": ST[sid]["ui_topic_opts"] if ST[sid]["stage"]=="choose_topic"
                                             else ST[sid]["ui_theme_opts"],
        "current_theme": ST[sid]["current_theme"],
        "themes": ST[sid]["themes"],
        "topics": ST[sid]["topics"],
        "qa": ST[sid]["qa"]
    })

# ──────────── Static & export ─────
@app.get("/export/<sid>")
def export_json(sid:str):
    if sid not in ST: abort(404)
    p=f"/tmp/geboorteplan_{sid}.json"
    with open(p,"w",encoding="utf-8") as f: json.dump(ST[sid], f, ensure_ascii=False, indent=2)
    return send_file(p, as_attachment=True, download_name=os.path.basename(p))

@app.route("/", defaults={"path":""})
@app.route("/<path:path>")
def index(path):
    root=app.static_folder
    if path and os.path.exists(os.path.join(root,path)):
        return send_from_directory(root, path)
    return send_from_directory(root, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","10000")), debug=True)
