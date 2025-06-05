import os, json, uuid, sqlite3
from copy import deepcopy
from typing import List, Dict, Optional

from flask import Flask, request, Response, jsonify, abort, send_file
from flask_cors import CORS
from openai import OpenAI, RateLimitError
from agents import Agent, Runner, function_tool

# ───────── Config ─────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
ALLOWED_ORIGINS = ["https://bevalmeteenplan.nl", "https://www.bevalmeteenplan.nl"]
DB_FILE = "sessions.db"

app = Flask(__name__)
CORS(app, origins=ALLOWED_ORIGINS, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# ───────── Basis-onderwerpen per thema ─────────
DEFAULT_TOPICS: Dict[str, List[str]] = {
    "Ondersteuning": [
        "Wie wil je bij de bevalling?",
        "Rol van je partner of ander persoon?",
        "Wil je een doula / kraamzorg / verpleegkundige?",
        "Wat verwacht je van het personeel?"
    ],
    "Bevalling & medisch beleid": [
        "Pijnstilling",
        "Interventies",
        "Noodsituaties",
        "Placenta-keuzes"
    ],
    "Sfeer en omgeving": [
        "Muziek & verlichting",
        "Privacy",
        "Foto’s / video",
        "Eigen spulletjes"
    ],
    "Voeding na de geboorte": [
        "Borstvoeding",
        "Flesvoeding",
        "Combinatie-voeding",
        "Allergieën"
    ]
}

# ───────── SQLite ─────────
def init_db() -> None:
    with sqlite3.connect(DB_FILE) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS sessions "
            "(id TEXT PRIMARY KEY, state TEXT NOT NULL)"
        )
init_db()

def load_state(sid: str) -> Optional[dict]:
    with sqlite3.connect(DB_FILE) as con:
        row = con.execute("SELECT state FROM sessions WHERE id=?", (sid,)).fetchone()
        return json.loads(row[0]) if row else None

def save_state(sid: str, st: dict) -> None:
    blob = json.dumps({k: v for k, v in st.items() if k != "history"})
    with sqlite3.connect(DB_FILE) as con:
        con.execute("REPLACE INTO sessions(id,state) VALUES(?,?)", (sid, blob))
        con.commit()

# ───────── Sessie-cache ─────────
SESSION: dict[str, dict] = {}
def get_session(sid: str) -> dict:
    if sid in SESSION: return SESSION[sid]
    if (db := load_state(sid)):
        SESSION[sid] = {**db, "history": []}; return SESSION[sid]
    SESSION[sid] = {
        "stage": "choose_theme",      # choose_topic → qa → review
        "themes": [],                 # [{"name","description"}]
        "topics": {},                 # {theme: [topic …]}
        "qa": [],                     # [{"theme","question","answer"}]
        "history": [], "summary": "",
        "ui_theme_opts": [],          # chips
        "ui_topic_opts": [],          # chips
        "current_theme": None
    }
    return SESSION[sid]
def persist(sid: str) -> None: save_state(sid, SESSION[sid])

# ───────── Samenvatten (>40 turns) ─────────
def summarize_chunk(chunk: List[dict]) -> str:
    if not chunk: return ""
    txt = "\n".join(f"{m['role']}: {m['content']}" for m in chunk)
    r = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
          {"role":"system","content":"Vat samen in max 300 tokens."},
          {"role":"user","content":txt}
        ],
        max_tokens=300
    )
    return r.choices[0].message.content.strip()

# ───────── Tools  ─────────
def _set_theme_options(session_id: str, options: List[str]) -> str:
    st = get_session(session_id)
    st["ui_theme_opts"] = options
    persist(session_id); return "ok"

def _set_topic_options(session_id: str, theme: str, options: List[str]) -> str:
    st = get_session(session_id)
    st["ui_topic_opts"] = options
    st["current_theme"] = theme
    persist(session_id); return "ok"

def _register_theme(session_id: str, theme: str, description: str = "") -> str:
    st = get_session(session_id)
    if len(st["themes"]) >= 6 or theme in [t["name"] for t in st["themes"]]:
        return "ok"
    st["themes"].append({"name": theme, "description": description})
    st["stage"] = "choose_topic"
    # zet default topic-chips
    topics = DEFAULT_TOPICS.get(theme, [])
    st["ui_topic_opts"] = topics
    st["current_theme"] = theme
    persist(session_id); return "ok"

def _register_topic(session_id: str, theme: str, topic: str) -> str:
    st = get_session(session_id)
    lst = st["topics"].setdefault(theme, [])
    if len(lst) >= 4 or topic in lst:
        return "ok"
    lst.append(topic)
    # als 4 topics bereikt → naar QA-fase
    if len(lst) == 4:
        st["stage"] = "qa"
        st["ui_topic_opts"] = []
    persist(session_id); return "ok"

def _log_answer(session_id: str, theme: str,
                question: str, answer: str) -> str:
    get_session(session_id)["qa"].append({"theme": theme, "question": question, "answer": answer})
    persist(session_id); return "ok"

def _update_answer(session_id: str, question: str,
                   new_answer: str) -> str:
    st = get_session(session_id)
    for qa in st["qa"]:
        if qa["question"] == question:
            qa["answer"] = new_answer; break
    persist(session_id); return "ok"

def _get_state(session_id: str) -> str:
    return json.dumps(get_session(session_id))

# wrapper
set_theme_options = function_tool(_set_theme_options)
set_topic_options = function_tool(_set_topic_options)
register_theme    = function_tool(_register_theme)
register_topic    = function_tool(_register_topic)
log_answer        = function_tool(_log_answer)
update_answer     = function_tool(_update_answer)
get_state_tool    = function_tool(_get_state)

# ───────── Agent template ─────────
BASE_AGENT = Agent(
    name="Geboorteplan-agent",
    model="gpt-4.1",
    instructions=(
      "Je begeleidt stap-voor-stap een geboorteplan.\n"
      "Stap 1 - thema’s: roep eerst 'set_theme_options' aan met een lijst "
      "van relevante thema’s (max 6). De UI maakt daar chips van.\n"
      "Na 'register_theme' voor het gekozen thema, roep direct "
      "'set_topic_options' aan met maximaal 4 onderwerpen voor dít thema. "
      "Zodra de gebruiker klikt, gebruik je 'register_topic'.\n"
      "Als 4 topics voor dit thema gekozen zijn, stel vragen en log met 'log_answer'. "
      "Ga daarna naar het volgende thema totdat alles behandeld is.\n"
      "Gebruik altijd session_id in tools. Antwoord in NL."
    ),
    tools=[
        set_theme_options, set_topic_options,
        register_theme, register_topic,
        log_answer, update_answer,
        get_state_tool
    ],
)

# ───────── stream /chat ─────────
def stream_run(tid: str):
    with client.beta.threads.runs.stream(thread_id=tid, assistant_id=ASSISTANT_ID) as ev:
        for e in ev:
            if e.event=="thread.message.delta" and e.data.delta.content:
                yield e.data.delta.content[0].text.value

@app.post("/chat")
def chat():
    if (o:=request.headers.get("Origin")) and o not in ALLOWED_ORIGINS: abort(403)
    d=request.get_json(force=True); msg=d.get("message","")
    tid=d.get("thread_id") or client.beta.threads.create().id
    client.beta.threads.messages.create(thread_id=tid,role="user",content=msg)
    return Response(stream_run(tid),headers={"X-Thread-ID":tid},mimetype="text/plain")

# ───────── main /agent ─────────
@app.post("/agent")
def agent():
    if (o:=request.headers.get("Origin")) and o not in ALLOWED_ORIGINS: abort(403)
    d=request.get_json(force=True)
    msg=d.get("message",""); sid=d.get("session_id") or str(uuid.uuid4())
    st=get_session(sid)

    if len(st["history"])>40:
        st["summary"]=(st["summary"]+"\n"+summarize_chunk(st["history"][:-20])).strip()
        st["history"]=st["history"][-20:]; persist(sid)

    intro=[{"role":"system","content":"Samenvatting:\n"+st["summary"]}] if st["summary"] else []
    items=intro+deepcopy(st["history"])+[{"role":"user","content":msg}]

    agent_inst=Agent(**{**BASE_AGENT.__dict__,
                        "instructions":BASE_AGENT.instructions+
                          f"\n\nGebruik session_id=\"{sid}\"."})

    res=Runner().run_sync(agent_inst, items)
    st["history"]=res.to_input_list(); persist(sid)

    return jsonify({
        "assistant_reply": str(res.final_output),
        "session_id": sid,
        "options": st["ui_topic_opts"] if st["stage"]=="choose_topic" else st["ui_theme_opts"],
        "current_theme": st["current_theme"],
        **{k:v for k,v in st.items() if k not in ("history","ui_theme_opts","ui_topic_opts")}
    })

# ───────── download JSON ─────────
@app.get("/export/<sid>")
def export_json(sid: str):
    st=load_state(sid);  abort(404) if not st else None
    p=f"/tmp/geboorteplan_{sid}.json"
    with open(p,"w",encoding="utf-8") as f: json.dump(st,f,ensure_ascii=False,indent=2)
    return send_file(p,as_attachment=True,download_name=p.split('/')[-1])

if __name__=="__main__":
    app.run(host="0.0.0.0",port=10000,debug=True)
