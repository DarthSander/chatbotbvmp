# ============================================================
#  Geboorteplan-agent – volledige Flask-app
#  Compatibel met Agents-SDK 0.0.17 (strict-schema)
# ============================================================

from __future__ import annotations
import os, json, uuid, sqlite3
from copy import deepcopy
from typing import List, Dict, Optional
+from typing_extensions import TypedDict

from flask import Flask, request, Response, jsonify, abort, send_file
from flask_cors import CORS
from openai import OpenAI
from agents import Agent, Runner, function_tool

# ---------- strikt type voor objecten ----------
class NamedDescription(TypedDict):
    name: str
    description: str

# ---------- basisconfig ----------
client           = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID     = os.getenv("ASSISTANT_ID")                # gerelateerd aan je Assistants-dashboard
ALLOWED_ORIGINS  = ["https://bevalmeteenplan.nl", "https://www.bevalmeteenplan.nl"]
DB_FILE          = "sessions.db"

app = Flask(__name__)
CORS(app, origins=ALLOWED_ORIGINS, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# ---------- standaardonderwerpen ----------
DEFAULT_TOPICS: Dict[str, List[NamedDescription]] = {
    "Ondersteuning": [
        {"name": "Wie wil je bij de bevalling?",           "description": "Welke personen wil je er fysiek bij hebben?"},
        {"name": "Rol van je partner of ander persoon?",   "description": "Specificeer taken of wensen voor je partner."},
        {"name": "Wil je een doula / kraamzorg?",          "description": "Extra ondersteuning tijdens en na de bevalling."},
        {"name": "Wat verwacht je van het personeel?",     "description": "Welke stijl van begeleiding past bij jou?"}
    ],
    "Bevalling & medisch beleid": [
        {"name": "Pijnstilling",       "description": "Medicamenteuze en niet-medicamenteuze opties."},
        {"name": "Interventies",       "description": "Bijv. inknippen, kunstverlossing, infuus."},
        {"name": "Noodsituaties",      "description": "Wat als het anders loopt dan gepland?"},
        {"name": "Placenta-keuzes",    "description": "Placenta bewaren, laten staan, of doneren?"}
    ],
    "Sfeer en omgeving": [
        {"name": "Muziek & verlichting", "description": "Rustige muziek? Gedimd licht?"},
        {"name": "Privacy",             "description": "Wie mag binnenkomen en fotograferen?"},
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

# ---------- SQLite ----------
def init_db() -> None:
    with sqlite3.connect(DB_FILE) as con:
        con.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, state TEXT NOT NULL)")
init_db()

def load_state(sid: str) -> Optional[dict]:
    with sqlite3.connect(DB_FILE) as con:
        row = con.execute("SELECT state FROM sessions WHERE id=?", (sid,)).fetchone()
        return json.loads(row[0]) if row else None

def save_state(sid: str, st: dict) -> None:
    with sqlite3.connect(DB_FILE) as con:
        con.execute(
            "REPLACE INTO sessions (id, state) VALUES (?, ?)",
            (sid, json.dumps({k: v for k, v in st.items() if k != "history"}))
        )
        con.commit()

# ---------- sessiebeheer ----------
SESSION: Dict[str, dict] = {}
def get_session(sid: str) -> dict:
    if sid in SESSION:
        return SESSION[sid]
    if (db := load_state(sid)):
        SESSION[sid] = {**db, "history": []}
        return SESSION[sid]
    # nieuwe sessie
    SESSION[sid] = {
        "stage": "choose_theme",
        "themes": [],                 # [{'name','description'}]
        "topics": {},                 # {theme: [topic …]}
        "qa": [],                     # [{'theme','question','answer'}]
        "history": [],
        "summary": "",
        "ui_theme_opts": [],
        "ui_topic_opts": [],
        "current_theme": None
    }
    return SESSION[sid]

def persist(sid: str) -> None:
    save_state(sid, SESSION[sid])

# ---------- samenvatten ----------
def summarize_chunk(chunk: List[dict]) -> str:
    if not chunk:
        return ""
    txt = "\n".join(f"{m['role']}: {m['content']}" for m in chunk)
    r = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Vat samen in max 300 tokens."},
            {"role": "user",   "content": txt}
        ],
        max_tokens=300
    )
    return r.choices[0].message.content.strip()

# ============================================================
#  Tool-implementaties (allemaal strikt getype-de parameters)
# ============================================================

def _set_theme_options(session_id: str, options: List[NamedDescription]) -> str:
    st = get_session(session_id)
    st["ui_theme_opts"] = options
    persist(session_id)
    return "ok"

def _set_topic_options(session_id: str, theme: str, options: List[NamedDescription]) -> str:
    st = get_session(session_id)
    st["ui_topic_opts"] = options
    st["current_theme"] = theme
    persist(session_id)
    return "ok"

def _register_theme(session_id: str, theme: str, description: str = "") -> str:
    st = get_session(session_id)
    if len(st["themes"]) >= 6 or theme in [t["name"] for t in st["themes"]]:
        return "ok"
    st["themes"].append({"name": theme, "description": description})
    st["stage"] = "choose_topic"
    st["ui_topic_opts"] = DEFAULT_TOPICS.get(theme, [])
    st["current_theme"] = theme
    persist(session_id)
    return "ok"

def _register_topic(session_id: str, theme: str, topic: str) -> str:
    st = get_session(session_id)
    lst = st["topics"].setdefault(theme, [])
    if len(lst) >= 4 or topic in lst:
        return "ok"
    lst.append(topic)
    persist(session_id)
    return "ok"

def _complete_theme(session_id: str) -> str:
    st = get_session(session_id)
    st["stage"] = "qa" if len(st["themes"]) >= 6 else "choose_theme"
    st["ui_topic_opts"] = []
    st["current_theme"] = None
    persist(session_id)
    return "ok"

def _log_answer(session_id: str, theme: str, question: str, answer: str) -> str:
    get_session(session_id)["qa"].append({"theme": theme, "question": question, "answer": answer})
    persist(session_id)
    return "ok"

def _update_answer(session_id: str, question: str, new_answer: str) -> str:
    st = get_session(session_id)
    for qa in st["qa"]:
        if qa["question"] == question:
            qa["answer"] = new_answer
            break
    persist(session_id)
    return "ok"

def _get_state(session_id: str) -> str:
    """Voor debug / front-end polling"""
    return json.dumps(get_session(session_id))

# ---------- tool-wrappers ----------
set_theme_options = function_tool(_set_theme_options)
set_topic_options = function_tool(_set_topic_options)
register_theme    = function_tool(_register_theme)
register_topic    = function_tool(_register_topic)
complete_theme    = function_tool(_complete_theme)
log_answer        = function_tool(_log_answer)
update_answer     = function_tool(_update_answer)
get_state_tool    = function_tool(_get_state)

# ============================================================
#  Agent-template
# ============================================================
BASE_AGENT = Agent(
    name="Geboorteplan-agent",
    model="gpt-4.1",
    instructions=(
        "Je helpt ouders hun geboorteplan maken (je bent géén digitale verloskundige).\n\n"
        "• Gebruik `set_theme_options` (max 6 objecten `{name, description}`) om thema’s te tonen.\n"
        "• Na `register_theme` stuur je direct `set_topic_options` met ≥4 `{name, description}`.\n"
        "• UI roept `register_topic` per selectie (max 4) en daarna `complete_theme`.\n"
        "• Bij 6 thema’s ga je automatisch door naar QA en stel je vragen; sla antwoorden op met `log_answer`.\n"
        "• Alle antwoorden in het Nederlands. Gebruik bij élke tool het juiste `session_id`."
    ),
    tools=[
        set_theme_options, set_topic_options,
        register_theme, register_topic, complete_theme,
        log_answer, update_answer, get_state_tool
    ],
)

# ============================================================
#  Streaming-/chat-endpoint (optioneel)
# ============================================================
def stream_run(tid: str):
    with client.beta.threads.runs.stream(thread_id=tid, assistant_id=ASSISTANT_ID) as ev:
        for e in ev:
            if e.event == "thread.message.delta" and e.data.delta.content:
                yield e.data.delta.content[0].text.value

@app.post("/chat")
def chat():
    if (o := request.headers.get("Origin")) and o not in ALLOWED_ORIGINS:
        abort(403)
    d   = request.get_json(force=True)
    msg = d.get("message", "")
    tid = d.get("thread_id") or client.beta.threads.create().id
    client.beta.threads.messages.create(thread_id=tid, role="user", content=msg)
    return Response(stream_run(tid),
                    headers={"X-Thread-ID": tid},
                    mimetype="text/plain")

# ============================================================
#  Synchronous /agent-endpoint (belangrijk voor front-end)
# ============================================================
@app.post("/agent")
def agent():
    if (o := request.headers.get("Origin")) and o not in ALLOWED_ORIGINS:
        abort(403)

    body = request.get_json(force=True)
    msg  = body.get("message", "")
    sid  = body.get("session_id") or str(uuid.uuid4())
    st   = get_session(sid)

    # samenvatten bij lange history
    if len(st["history"]) > 40:
        st["summary"] = (st["summary"] + "\n" +
                         summarize_chunk(st["history"][:-20])).strip()
        st["history"] = st["history"][-20:]
        persist(sid)

    intro   = [{"role": "system", "content": "Samenvatting:\n" + st["summary"]}] if st["summary"] else []
    messages = intro + deepcopy(st["history"]) + [{"role": "user", "content": msg}]

    agent_inst = Agent(
        **{**BASE_AGENT.__dict__,
           "instructions": BASE_AGENT.instructions + f"\n\nGebruik session_id=\"{sid}\"."}
    )
    res = Runner().run_sync(agent_inst, messages)

    st["history"] = res.to_input_list()
    persist(sid)

    return jsonify({
        "assistant_reply": str(res.final_output),
        "session_id": sid,
        "options": st["ui_topic_opts"] if st["stage"] == "choose_topic" else st["ui_theme_opts"],
        "current_theme": st["current_theme"],
        # transparante extra-state behalve grote velden
        **{k: v for k, v in st.items() if k not in ("history", "ui_theme_opts", "ui_topic_opts")}
    })

# ---------- export (unchanged) ----------
@app.get("/export/<sid>")
def export_json(sid: str):
    st = load_state(sid)
    if not st:
        abort(404)
    path = f"/tmp/geboorteplan_{sid}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    return send_file(path, as_attachment=True, download_name=path.split("/")[-1])

# ---------- main ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
