import os, json, uuid, sqlite3
from copy import deepcopy
from typing import List, Dict, Optional

from flask import Flask, request, Response, jsonify, abort, send_file
from flask_cors import CORS
from openai import OpenAI, RateLimitError
from agents import Agent, Runner, function_tool

# ───────────── Config ─────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client         = OpenAI(api_key=OPENAI_API_KEY)
ASSISTANT_ID   = os.getenv("ASSISTANT_ID")

ALLOWED_ORIGINS = ["https://bevalmeteenplan.nl", "https://www.bevalmeteenplan.nl"]
DB_FILE = "sessions.db"

app = Flask(__name__)
CORS(app, origins=ALLOWED_ORIGINS, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# ─────────── SQLite helpers ──────────
def init_db() -> None:
    with sqlite3.connect(DB_FILE) as con:
        con.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, state TEXT)")
init_db()

def load_state(sid: str) -> Optional[dict]:
    with sqlite3.connect(DB_FILE) as con:
        row = con.execute("SELECT state FROM sessions WHERE id=?", (sid,)).fetchone()
        return json.loads(row[0]) if row else None

def save_state(sid: str, state: dict) -> None:
    blob = json.dumps({k: v for k, v in state.items() if k != "history"})
    with sqlite3.connect(DB_FILE) as con:
        con.execute("REPLACE INTO sessions(id,state) VALUES(?,?)", (sid, blob))
        con.commit()

# ─────────── In-memory cache ─────────
SESSION: dict[str, dict] = {}

def get_session(sid: str) -> dict:
    if sid in SESSION:
        return SESSION[sid]
    if (db := load_state(sid)):
        SESSION[sid] = {**db, "history": []}
        return SESSION[sid]
    SESSION[sid] = {
        "stage": "choose_theme",    # of choose_topic | qa | review
        "themes": [],               # [{"name","description"} …]
        "topics": {},               # {theme: [{"name","description"} …]}
        "qa": [],                   # [{"theme","question","answer"} …]
        "history": [],
        "summary": "",
        # dynamische UI-velden (door agent gezet)
        "ui_options": [],
        "ui_current_theme": None
    }
    return SESSION[sid]

def persist(sid: str) -> None:
    save_state(sid, SESSION[sid])

# ─────────── Samenvatten (>40 turns) ──────────
def summarize_chunk(chunk: List[dict]) -> str:
    if not chunk:
        return ""
    txt = "\n".join(f"{m['role']}: {m['content']}" for m in chunk)
    r = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"Vat dit gesprek samen (max 300 tokens)."},
            {"role":"user","content":txt}
        ],
        max_tokens=300
    )
    return r.choices[0].message.content.strip()

# ─────────── Tool-implementaties ──────────
def _register_theme(session_id: str, theme: str, description: str = "") -> str:
    st = get_session(session_id)
    if len(st["themes"]) >= 6:
        return json.dumps({"error": "max themes"})
    if theme in [t["name"] for t in st["themes"]]:
        return json.dumps({"error": "duplicate theme"})
    st["themes"].append({"name": theme, "description": description})
    st["stage"] = "choose_topic"
    persist(session_id); return "ok"

def _deregister_theme(session_id: str, theme: str) -> str:
    st = get_session(session_id)
    st["themes"] = [t for t in st["themes"] if t["name"] != theme]
    st["topics"].pop(theme, None)
    st["qa"] = [qa for qa in st["qa"] if qa["theme"] != theme]
    st["stage"] = "choose_theme" if not st["themes"] else "choose_topic"
    persist(session_id); return "ok"

def _register_topic(session_id: str, theme: str,
                    topic: str, description: str = "") -> str:
    st = get_session(session_id)
    lst = st["topics"].setdefault(theme, [])
    if len(lst) >= 4:
        return json.dumps({"error": "max topics"})
    if topic in [t["name"] for t in lst]:
        return json.dumps({"error": "duplicate topic"})
    lst.append({"name": topic, "description": description})
    if all(len(st["topics"].get(t["name"], [])) >= 4 for t in st["themes"]):
        st["stage"] = "qa"
    persist(session_id); return "ok"

def _deregister_topic(session_id: str, theme: str, topic: str) -> str:
    st = get_session(session_id)
    if theme in st["topics"]:
        st["topics"][theme] = [t for t in st["topics"][theme] if t["name"] != topic]
        st["qa"] = [qa for qa in st["qa"]
                    if not (qa["theme"] == theme and qa["question"].startswith(topic))]
    st["stage"] = "choose_topic"
    persist(session_id); return "ok"

def _log_answer(session_id: str, theme: str,
                question: str, answer: str) -> str:
    get_session(session_id)["qa"].append({"theme": theme, "question": question, "answer": answer})
    persist(session_id); return "ok"

def _update_answer(session_id: str, question: str, new_answer: str) -> str:
    st = get_session(session_id)
    for qa in st["qa"]:
        if qa["question"] == question:
            qa["answer"] = new_answer; break
    persist(session_id); return "ok"

def _set_ui_options(session_id: str,
                    options: List[str],
                    current_theme: Optional[str] = None) -> str:
    """Laat de agent expliciet bepalen welke chips in de UI komen te staan."""
    st = get_session(session_id)
    st["ui_options"] = options
    st["ui_current_theme"] = {"name": current_theme} if current_theme else None
    persist(session_id); return "ok"

def _get_state(session_id: str) -> str:
    return json.dumps(get_session(session_id))

# ─────────── Tool-wrappers ──────────
register_theme   = function_tool(_register_theme)
deregister_theme = function_tool(_deregister_theme)
register_topic   = function_tool(_register_topic)
deregister_topic = function_tool(_deregister_topic)
log_answer       = function_tool(_log_answer)
update_answer    = function_tool(_update_answer)
set_ui_options   = function_tool(_set_ui_options)
get_state_tool   = function_tool(_get_state)

# ─────────── Agent template ──────────
BASE_AGENT = Agent(
    name="Geboorteplan-agent",
    model="gpt-4.1",
    instructions=(
      "Je helpt gebruikers een geboorteplan maken.\n\n"
      "Workflow:\n"
      "1. Fase 'choose_theme': vraag de gebruiker thema’s (max 6). "
      "Bepaal vervolgens via `set_ui_options` welke chips (thema-labels) de UI moet tonen.\n"
      "2. Bij selectie gebruik je `register_theme`. Als de gebruiker iets anders intypt, kun je dat direct registreren.\n"
      "3. Fase 'choose_topic': per gekozen thema bied je max 4 onderwerpen aan. "
      "Gebruik opnieuw `set_ui_options` om chips te tonen en `register_topic` als er geklikt wordt.\n"
      "4. Fase 'qa': stel vragen, log met `log_answer`.\n"
      "5. Laat wijzigen via `deregister_*` of `update_answer`.\n"
      "6. Is alles compleet? zet `stage` op **review** zodat de UI een popup toont.\n\n"
      "Gebruik `get_state` om de huidige stand op te vragen. "
      "Gebruik ALTIJD `session_id` in elke tool-aanroep.\n"
      "Antwoord consequent in het Nederlands."
    ),
    tools=[
        register_theme, deregister_theme,
        register_topic, deregister_topic,
        log_answer, update_answer,
        set_ui_options, get_state_tool
    ],
)

# ─────────── /chat (stream) ──────────
def stream_run(tid: str):
    with client.beta.threads.runs.stream(thread_id=tid, assistant_id=ASSISTANT_ID) as ev:
        for e in ev:
            if e.event=="thread.message.delta" and e.data.delta.content:
                yield e.data.delta.content[0].text.value

@app.post("/chat")
def chat():
    if (o:=request.headers.get("Origin")) and o not in ALLOWED_ORIGINS: abort(403)
    d=request.get_json(force=True)
    msg=d.get("message","")
    tid=d.get("thread_id") or client.beta.threads.create().id
    client.beta.threads.messages.create(thread_id=tid, role="user", content=msg)
    return Response(stream_run(tid), headers={"X-Thread-ID":tid}, mimetype="text/plain")

# ─────────── /agent (sync) ──────────
@app.post("/agent")
def agent():
    if (o:=request.headers.get("Origin")) and o not in ALLOWED_ORIGINS: abort(403)
    data=request.get_json(force=True)
    msg=data.get("message",""); sid=data.get("session_id") or str(uuid.uuid4())
    st=get_session(sid)

    # samenvatten
    if len(st["history"])>40:
        st["summary"]=(st["summary"]+"\n"+summarize_chunk(st["history"][:-20])).strip()
        st["history"]=st["history"][-20:]; persist(sid)

    intro=[{"role":"system","content":"Samenvatting:\n"+st["summary"]}] if st["summary"] else []
    items=intro+deepcopy(st["history"])+[{"role":"user","content":msg}]

    agent_inst=Agent(**{**BASE_AGENT.__dict__,
                        "instructions":BASE_AGENT.instructions+
                          f"\n\nGebruik session_id=\"{sid}\" in alle tools."})

    res=Runner().run_sync(agent_inst, items)
    st["history"]=res.to_input_list(); persist(sid)

    pub={k:v for k,v in st.items() if k!="history"}
    return jsonify({
        "assistant_reply": str(res.final_output),
        "session_id": sid,
        "options": st["ui_options"],
        "current_theme": st["ui_current_theme"],
        **pub
    })

# ─────────── export JSON ──────────
@app.get("/export/<sid>")
def export_json(sid: str):
    st=load_state(sid)
    if not st: abort(404)
    p=f"/tmp/geboorteplan_{sid}.json"
    with open(p,"w",encoding="utf-8") as f:
        json.dump(st,f,ensure_ascii=False,indent=2)
    return send_file(p,as_attachment=True,
                     download_name=p.split('/')[-1],
                     mimetype="application/json")

if __name__=="__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
