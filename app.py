import os, json, uuid, sqlite3
from copy import deepcopy
from flask import Flask, request, Response, jsonify, abort, send_file
from flask_cors import CORS
from openai import OpenAI, RateLimitError
from agents import Agent, Runner, function_tool

# ─────────────────── Config ───────────────────────────────
openai_api_key = os.getenv("OPENAI_API_KEY")
client         = OpenAI(api_key=openai_api_key)
ASSISTANT_ID   = os.getenv("ASSISTANT_ID")           # alleen /chat-endpoint

ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
]

DB_FILE = "sessions.db"

app = Flask(__name__)
CORS(app, origins=ALLOWED_ORIGINS,
     allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# ────────── SQLite helpers (eerst definiëren!) ────────────
def init_db():
    """Maak de tabel als die er nog niet is (géén cursor-contextmanager)."""
    con = sqlite3.connect(DB_FILE)
    try:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id    TEXT PRIMARY KEY,
                state TEXT NOT NULL
            )
        """)
        con.commit()
    finally:
        cur.close()
        con.close()

def load_state(session_id: str) -> dict | None:
    with sqlite3.connect(DB_FILE) as con:
        cur = con.execute("SELECT state FROM sessions WHERE id=?", (session_id,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

def save_state(session_id: str, state: dict):
    blob = json.dumps({k: v for k, v in state.items() if k != "history"})
    with sqlite3.connect(DB_FILE) as con:
        con.execute("REPLACE INTO sessions(id, state) VALUES(?,?)",
                    (session_id, blob))
        con.commit()

init_db()

# ────────── In-memory cache + helpers ─────────────────────
SESSION: dict[str, dict] = {}      # session_id → state

def get_session(session_id: str) -> dict:
    if session_id in SESSION:
        return SESSION[session_id]

    if (db_state := load_state(session_id)):
        SESSION[session_id] = {**db_state, "history": []}
        return SESSION[session_id]

    # nieuwe sessie
    SESSION[session_id] = {
        "stage": "choose_theme",
        "themes": [],
        "topics": {},
        "qa": [],
        "history": []
    }
    return SESSION[session_id]

def persist(session_id: str):
    save_state(session_id, SESSION[session_id])

# ────────── Tool-implementaties (zonder decorator) ────────
def _register_theme(session_id, theme, description=""):
    st = get_session(session_id)
    if len(st["themes"]) < 6 and theme not in [t["name"] for t in st["themes"]]:
        st["themes"].append({"name": theme, "description": description})
        st["stage"] = "choose_topic"
    persist(session_id)
    return json.dumps(st)

def _deregister_theme(session_id, theme):
    st = get_session(session_id)
    st["themes"] = [t for t in st["themes"] if t["name"] != theme]
    st["topics"].pop(theme, None)
    st["qa"] = [qa for qa in st["qa"] if qa["theme"] != theme]
    st["stage"] = "choose_theme" if not st["themes"] else st["stage"]
    persist(session_id)
    return json.dumps(st)

def _register_topic(session_id, theme, topic, description=""):
    st = get_session(session_id)
    topics = st["topics"].setdefault(theme, [])
    if len(topics) < 4 and topic not in [t["name"] for t in topics]:
        topics.append({"name": topic, "description": description})
    if st["themes"] and all(len(st["topics"].get(t["name"], [])) >= 4
                            for t in st["themes"]):
        st["stage"] = "qa"
    persist(session_id)
    return json.dumps(st)

def _deregister_topic(session_id, theme, topic):
    st = get_session(session_id)
    if theme in st["topics"]:
        st["topics"][theme] = [t for t in st["topics"][theme] if t["name"] != topic]
        st["qa"] = [qa for qa in st["qa"]
                    if not (qa["theme"] == theme and qa["question"].startswith(topic))]
    st["stage"] = "choose_topic"
    persist(session_id)
    return json.dumps(st)

def _log_answer(session_id, theme, question, answer):
    st = get_session(session_id)
    st["qa"].append({"theme": theme, "question": question, "answer": answer})
    persist(session_id)
    return "ok"

def _update_answer(session_id, question, new_answer):
    st = get_session(session_id)
    for qa in st["qa"]:
        if qa["question"] == question:
            qa["answer"] = new_answer
            break
    persist(session_id)
    return "ok"

def _get_state(session_id):
    return json.dumps(get_session(session_id))

# ────────── Tool-wrappers ─────────────────────────────────
from agents import function_tool
register_theme   = function_tool(_register_theme)
deregister_theme = function_tool(_deregister_theme)
register_topic   = function_tool(_register_topic)
deregister_topic = function_tool(_deregister_topic)
log_answer       = function_tool(_log_answer)
update_answer    = function_tool(_update_answer)
get_state_tool   = function_tool(_get_state)

# ────────── Basismodel voor Agents-SDK ────────────────────
BASE_AGENT = Agent(
    name="Geboorteplan-agent",
    model="gpt-4.1",
    instructions=(
        "Je bent een digitale verloskundige.\n"
        "Fase 1: max 6 thema’s (`register_theme`).\n"
        "Fase 2: per thema max 4 onderwerpen (`register_topic`).\n"
        "Fase 3: stel vragen, log met `log_answer`.\n"
        "Sta wijzigingen toe met `deregister_*` en `update_answer`.\n"
        "Als alles besproken is, zet `stage` naar **review**\n"
        "Gebruik `get_state` om keuzes op te vragen.\n"
        "Antwoord altijd in het Nederlands."
    ),
    tools=[
        register_theme, deregister_theme,
        register_topic, deregister_topic,
        log_answer, update_answer, get_state_tool
    ],
)

# ────────── Assistants-v2 /chat (streaming) ───────────────
def stream_run(thread_id: str):
    with client.beta.threads.runs.stream(thread_id=thread_id,
                                         assistant_id=ASSISTANT_ID) as evs:
        for ev in evs:
            if ev.event == "thread.message.delta" and ev.data.delta.content:
                yield ev.data.delta.content[0].text.value
            elif ev.event == "error":
                raise RuntimeError(ev.data.message)

@app.post("/chat")
def chat():
    if (origin := request.headers.get("Origin")) and origin not in ALLOWED_ORIGINS:
        abort(403)

    data      = request.get_json(force=True)
    user_msg  = data.get("message", "")
    thread_id = data.get("thread_id") or client.beta.threads.create().id
    client.beta.threads.messages.create(thread_id=thread_id,
                                        role="user", content=user_msg)

    def gen():
        try:
            yield from stream_run(thread_id)
        except RateLimitError:
            yield "\n⚠️ Rate-limit; probeer later opnieuw."
        except Exception as exc:
            yield f"\n⚠️ Serverfout: {exc}"

    return Response(gen(), headers={"X-Thread-ID": thread_id},
                    mimetype="text/plain")

# ────────── Agents-SDK /agent (sync) ───────────────────────
@app.post("/agent")
def agent():
    if (origin := request.headers.get("Origin")) and origin not in ALLOWED_ORIGINS:
        abort(403)

    data       = request.get_json(force=True)
    user_msg   = data.get("message", "")
    session_id = data.get("session_id") or str(uuid.uuid4())

    st = get_session(session_id)

    agent_instance = Agent(
        **{**BASE_AGENT.__dict__,
           "instructions": BASE_AGENT.instructions +
               f"\n\nBelangrijk: gebruik altijd "
               f"`session_id=\"{session_id}\"` in elke tool-aanroep."}
    )

    input_items = deepcopy(st["history"])
    input_items.append({"role": "user", "content": user_msg})

    result = Runner().run_sync(agent_instance, input_items)
    st["history"] = result.to_input_list()
    persist(session_id)

    public_state = {k: v for k, v in st.items() if k != "history"}

    return jsonify({
        "assistant_reply": str(result.final_output),
        "session_id": session_id,
        **public_state,
    })

# ────────── JSON-export endpoint ───────────────────────────
@app.get("/export/<session_id>")
def export(session_id):
    st = load_state(session_id)
    if not st:
        abort(404)
    filename = f"geboorteplan_{session_id}.json"
    path     = f"/tmp/{filename}"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(st, fh, ensure_ascii=False, indent=2)
    return send_file(path, as_attachment=True,
                     download_name=filename, mimetype="application/json")

# ────────── Run local ──────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
