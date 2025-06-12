#!/usr/bin/env python3
# Mae – Geboorteplan-agent (Assistant-gedreven, pro-actief met confirmaties)

from __future__ import annotations
import os, json, uuid, sqlite3, time, logging, re
from typing import List, Dict, Optional
from typing_extensions import TypedDict
from flask import (
    Flask, request, jsonify, abort, send_from_directory,
    send_file, render_template
)
from flask_cors import CORS
import openai
from agents import function_tool          # jouw util houdt params intact

# ───────────────────────────── Logging ─────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "DEBUG").upper(), logging.DEBUG),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("mae-backend")

# ───────────────────────────── Config ──────────────────────────────
openai.api_key   = os.getenv("OPENAI_API_KEY")
MODEL_NAME       = "gpt-4.1"                   # ⇦ gewenste model
ASSISTANT_FILE   = "assistant_id.txt"
DB_FILE          = "sessions.db"
ALLOWED_ORIGINS  = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
    "https://chatbotbvmp.onrender.com"
]

# ────────────────────── Basisdata (ongewijzigd) ────────────────────
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

# ────────────────────── SQLite sessies ─────────────────────────────
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
    if not row:
        return None
    return {"id": sid, "thread_id": row[0], "state": json.loads(row[1] or "{}")}

def save_session(sid: str, thread_id: str, state: dict):
    with sqlite3.connect(DB_FILE) as con:
        con.execute("REPLACE INTO sessions(id, thread_id, state) VALUES(?,?,?)",
                    (sid, thread_id, json.dumps(state)))

# ────────────────────── Business-logica (ongewijzigd) ──────────────
def blank_state() -> dict:
    return {
        "stage": "choose_theme",
        "themes": [], "topics": {}, "qa": [],
        "current_theme": None,
        "ui_theme_opts": [], "ui_topic_opts": []
    }

ST: Dict[str, dict] = {}

def _set_theme_options(sid: str, opts: List[str]) -> str:
    ST[sid]["ui_theme_opts"] = opts; return "ok"

def _set_topic_options(sid: str, theme: str, opts: List[NamedDescription]) -> str:
    ST[sid]["current_theme"] = theme
    ST[sid]["ui_topic_opts"] = opts; return "ok"

def _register_theme(sid: str, theme: str, desc: str = "") -> str:
    s = ST[sid]
    if len(s["themes"]) < 6 and theme not in [t["name"] for t in s["themes"]]:
        s["themes"].append({"name": theme, "description": desc})
        s["stage"] = "choose_topic"
        s["current_theme"] = theme
        s["ui_topic_opts"] = DEFAULT_TOPICS.get(theme, [])
    return "ok"

def _register_topic(sid: str, theme: str, topic: str) -> str:
    ST[sid]["topics"].setdefault(theme, [])
    if len(ST[sid]["topics"][theme]) < 4 and topic not in ST[sid]["topics"][theme]:
        ST[sid]["topics"][theme].append(topic)
    return "ok"

def _complete_theme(sid: str) -> str:
    s = ST[sid]
    s["stage"] = "choose_theme"
    s["current_theme"] = None
    s["ui_topic_opts"] = []
    if all(s["topics"].get(t["name"]) for t in s["themes"]):
        s["stage"] = "qa"
    return "ok"

def _log_answer(sid: str, theme: str, question: str, answer: str) -> str:
    ST[sid]["qa"].append({"theme": theme, "question": question, "answer": answer})
    return "ok"

# ─── toolschemas voor Assistant (functie-names exact!) ─────────────
tool_schemas = [
    function_tool(_set_theme_options),
    function_tool(_set_topic_options),
    function_tool(_register_theme),
    function_tool(_register_topic),
    function_tool(_complete_theme),
    function_tool(_log_answer)
]
TOOL_IMPL = {t["function"]["name"]: globals()[t["function"]["name"]] for t in tool_schemas}

# ────────────────────── Assistant setup ───────────────────────────
def ensure_assistant() -> str:
    if os.path.exists(ASSISTANT_FILE):
        return open(ASSISTANT_FILE).read().strip()

    prompt = (
        "Je bent Mae, digitale verloskundige.\n\n"
        "Proces:\n"
        "1. Toon een lijst standaardthema’s (tool: _set_theme_options). "
        "Vraag altijd of de gebruiker een eigen thema wil toevoegen.\n"
        "2. Wanneer een (nieuw) thema gekozen is, bevestig met een zin zoals "
        "“Oké, zullen we dat thema toevoegen?” – wacht op bevestiging.\n"
        "   Bij 'ja': roep _register_theme.\n"
        "3. Toon onderwerpenchips (_set_topic_options). "
        "Wanneer gebruiker onderwerp kiest, stap even terug en vraag: "
        "“Zal ik ‘<onderwerp>’ opslaan bij dit thema?” – bij bevestigen roep _register_topic.\n"
        "4. Als gebruiker zegt ‘klaar’ bij een thema, bevestig en roep _complete_theme.\n"
        "5. Zodra alle thema’s en onderwerpen klaar zijn (stage == qa), stel exact één "
        "vraag per onderwerp (description). Na ieder antwoord gebruik je _log_answer.\n\n"
        "Belangrijk: VÓÓR elke tool-call moet je de gebruiker om toestemming vragen "
        "met een duidelijke vraag in de chat. Pas als de gebruiker instemt (yes/ja), "
        "voer je de tool-call uit.\n"
        "Gebruik nooit eigen JSON in de chat. Gebruik uitsluitend de gedefinieerde tools."
    )

    assistant = openai.beta.assistants.create(
        name="Mae – Geboorteplan-coach",
        instructions=prompt,
        tools=[t["function"] for t in tool_schemas],
        model=MODEL_NAME
    )
    with open(ASSISTANT_FILE, "w") as f:
        f.write(assistant.id)
    log.info("Assistant aangemaakt: %s", assistant.id)
    return assistant.id

ASSISTANT_ID = ensure_assistant()

# ────────────────────── Assistant run helper ──────────────────────
def run_assistant(sid: str, thread_id: str, user_text: str) -> str:
    openai.beta.threads.messages.create(thread_id=thread_id, role="user", content=user_text)
    run = openai.beta.threads.runs.create(thread_id=thread_id, assistant_id=ASSISTANT_ID)
    while True:
        run = openai.beta.threads.runs.retrieve(thread_id, run.id)
        if run.status == "requires_action":
            outputs = []
            for call in run.required_action.submit_tool_outputs.tool_calls:
                fn_name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                log.debug("Tool-call van Assistant: %s %s", fn_name, args)
                result = TOOL_IMPL[fn_name](sid, **args)
                outputs.append({"tool_call_id": call.id, "output": result})
            openai.beta.threads.runs.submit_tool_outputs(thread_id, run.id, outputs=outputs)
            continue
        if run.status in ("queued", "in_progress"):
            time.sleep(0.4); continue
        if run.status != "completed":
            log.error("Run status %s, error %s", run.status, run.last_error)
            return "Er ging iets mis; probeer het later opnieuw."
        break
    msg = openai.beta.threads.messages.list(thread_id, limit=1).data[0]
    return msg.content[0].text.value if msg.content else ""

# ────────────────────── Flask server ──────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates", static_url_path="")
CORS(app, origins=ALLOWED_ORIGINS, allow_headers="*", methods=["GET","POST"])

@app.post("/agent")
def agent_route():
    data = request.get_json(force=True)
    user_text = data.get("message","")
    sid       = data.get("session_id") or str(uuid.uuid4())

    sess = load_session(sid)
    if not sess:
        thread = openai.beta.threads.create()
        sess = {"id": sid, "thread_id": thread.id, "state": {}}
        ST[sid] = blank_state()
    else:
        ST.setdefault(sid, blank_state())

    log.debug("IN  %s | %s", sid[-6:], user_text)
    assistant_reply = run_assistant(sid, sess["thread_id"], user_text)
    log.debug("OUT %s | %s", sid[-6:], assistant_reply[:80])

    save_session(sid, sess["thread_id"], ST[sid])

    return jsonify({
        "session_id": sid,
        "assistant_reply": assistant_reply,
        "stage": ST[sid]["stage"],
        "options": ST[sid]["ui_topic_opts"] if ST[sid]["stage"]=="choose_topic"
                                           else ST[sid]["ui_theme_opts"],
        "current_theme": ST[sid]["current_theme"],
        "themes": ST[sid]["themes"],
        "topics": ST[sid]["topics"],
        "qa": ST[sid]["qa"]
    })

# ────────────────────── Export + static ───────────────────────────
@app.get("/export/<sid>")
def export_json(sid: str):
    if sid not in ST: abort(404)
    path = f"/tmp/geboorteplan_{sid}.json"
    with open(path,"w",encoding="utf-8") as f:
        json.dump(ST[sid], f, ensure_ascii=False, indent=2)
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))

@app.route("/", defaults={"path":""})
@app.route("/<path:path>")
def static_files(path):
    root = app.static_folder
    if path and os.path.exists(os.path.join(root, path)):
        return send_from_directory(root, path)
    return send_from_directory(root, "index.html")

# ────────────────────── main ─────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), debug=True)
