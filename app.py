import os, json, uuid, sqlite3
from copy import deepcopy
from typing import List, Dict

from flask import Flask, request, Response, jsonify, abort, send_file
from flask_cors import CORS
from openai import OpenAI, RateLimitError
from agents import Agent, Runner, function_tool

# ───────── Config ─────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client         = OpenAI(api_key=OPENAI_API_KEY)
ASSISTANT_ID   = os.getenv("ASSISTANT_ID")

ALLOWED_ORIGINS = ["https://bevalmeteenplan.nl", "https://www.bevalmeteenplan.nl"]
DB_FILE = "sessions.db"

MASTER_THEMES: List[str] = [
    "Zwangerschap", "Bevalling", "Kraamtijd",
    "Voeding", "Voorbereiding", "Emoties"
]
MASTER_TOPICS: Dict[str, List[str]] = {
    "Zwangerschap": ["Gezondheid", "Sport", "Werk", "Klachten"],
    "Bevalling":    ["Pijnstilling", "Geboorteplek", "Partnerrol", "Complicaties"],
    "Kraamtijd":    ["Rust & herstel", "Hulp", "Bezoek", "Emoties"],
    "Voeding":      ["Borstvoeding", "Flesvoeding", "Dieet", "Allergieën"],
    "Voorbereiding":["Tas inpakken", "Geboorteplan", "Ademhaling", "Reistijd"],
    "Emoties":      ["Angst", "Blijdschap", "Stress", "Mindset"]
}

app = Flask(__name__)
CORS(app, origins=ALLOWED_ORIGINS, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# ───────── SQLite ─────────
def init_db() -> None:
    with sqlite3.connect(DB_FILE) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS sessions "
            "(id TEXT PRIMARY KEY, state TEXT NOT NULL)"
        )
init_db()

def load_state(sid: str) -> dict | None:
    with sqlite3.connect(DB_FILE) as con:
        row = con.execute("SELECT state FROM sessions WHERE id=?", (sid,)).fetchone()
        return json.loads(row[0]) if row else None

def save_state(sid: str, st: dict) -> None:
    blob = json.dumps({k: v for k, v in st.items() if k != "history"})
    with sqlite3.connect(DB_FILE) as con:
        con.execute("REPLACE INTO sessions(id,state) VALUES(?,?)", (sid, blob))
        con.commit()

# ───────── sessie-cache ─────────
SESSION: dict[str, dict] = {}
def get_session(sid: str) -> dict:
    if sid in SESSION: return SESSION[sid]
    if (db := load_state(sid)):
        SESSION[sid] = {**db, "history": []}; return SESSION[sid]
    SESSION[sid] = {
        "stage":"choose_theme","themes":[],
        "topics":{},"qa":[],
        "history":[], "summary":""
    }
    return SESSION[sid]
def persist(sid: str) -> None: save_state(sid, SESSION[sid])

# ───────── samenvatten ─────────
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

# ───────── tool-functies (met type-hints!) ─────────
def _register_theme(session_id: str, theme: str, description: str = "") -> str:
    st = get_session(session_id)
    if len(st["themes"]) < 6 and theme not in [t["name"] for t in st["themes"]]:
        st["themes"].append({"name": theme, "description": description})
        st["stage"] = "choose_topic"
    persist(session_id); return json.dumps(st)

def _deregister_theme(session_id: str, theme: str) -> str:
    st = get_session(session_id)
    st["themes"] = [t for t in st["themes"] if t["name"] != theme]
    st["topics"].pop(theme, None)
    st["qa"] = [q for q in st["qa"] if q["theme"] != theme]
    st["stage"] = "choose_theme" if not st["themes"] else "choose_topic"
    persist(session_id); return json.dumps(st)

def _register_topic(session_id: str, theme: str,
                    topic: str, description: str = "") -> str:
    st = get_session(session_id)
    lst = st["topics"].setdefault(theme, [])
    if len(lst) < 4 and topic not in [t["name"] for t in lst]:
        lst.append({"name": topic, "description": description})
    if all(len(st["topics"].get(t["name"], [])) >= 4 for t in st["themes"]):
        st["stage"] = "qa"
    persist(session_id); return json.dumps(st)

def _deregister_topic(session_id: str, theme: str, topic: str) -> str:
    st = get_session(session_id)
    if theme in st["topics"]:
        st["topics"][theme] = [t for t in st["topics"][theme] if t["name"] != topic]
        st["qa"] = [q for q in st["qa"]
                    if not (q["theme"] == theme and q["question"].startswith(topic))]
    st["stage"] = "choose_topic"; persist(session_id); return json.dumps(st)

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

# wrappers
register_theme   = function_tool(_register_theme)
deregister_theme = function_tool(_deregister_theme)
register_topic   = function_tool(_register_topic)
deregister_topic = function_tool(_deregister_topic)
log_answer       = function_tool(_log_answer)
update_answer    = function_tool(_update_answer)
get_state_tool   = function_tool(_get_state)

# ───────── Agent template ─────────
BASE_AGENT = Agent(
    name="Geboorteplan-agent",
    model="gpt-4.1",
    instructions=(
      "Je helpt een geboorteplan maken.\n"
      "Fase1: max 6 thema’s -> register_theme\n"
      "Fase2: max 4 topics per thema -> register_topic\n"
      "Fase3: vragen stellen -> log_answer\n"
      "Wijzigingen -> deregister_* / update_answer\n"
      "Einde -> stage=review\n"
      "Antwoord in het Nederlands."
    ),
    tools=[register_theme,deregister_theme,register_topic,deregister_topic,
           log_answer,update_answer,get_state_tool]
)

# ───────── stream /chat ─────────
def stream_run(tid: str):
    with client.beta.threads.runs.stream(thread_id=tid, assistant_id=ASSISTANT_ID) as evs:
        for ev in evs:
            if ev.event=="thread.message.delta" and ev.data.delta.content:
                yield ev.data.delta.content[0].text.value

@app.post("/chat")
def chat():
    if (o:=request.headers.get("Origin")) and o not in ALLOWED_ORIGINS: abort(403)
    d=request.get_json(force=True); msg=d.get("message","")
    tid=d.get("thread_id") or client.beta.threads.create().id
    client.beta.threads.messages.create(thread_id=tid,role="user",content=msg)
    return Response(stream_run(tid),headers={"X-Thread-ID":tid},mimetype="text/plain")

# ───────── UI-opties helper ─────────
def ui_options(st: dict) -> tuple[list, dict|None]:
    if st["stage"]=="choose_theme":
        chosen=[t["name"] for t in st["themes"]]
        return [t for t in MASTER_THEMES if t not in chosen], None
    if st["stage"]=="choose_topic":
        cur=st["themes"][-1] if st["themes"] else None
        if not cur: return [],None
        done=[t["name"] for t in st["topics"].get(cur["name"],[])]
        return [t for t in MASTER_TOPICS.get(cur["name"],[]) if t not in done], cur
    return [], None

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

    pre=[{"role":"system","content":"Samenvatting:\n"+st["summary"]}] if st["summary"] else []
    items=pre+deepcopy(st["history"])+[{"role":"user","content":msg}]

    agent_inst=Agent(**{**BASE_AGENT.__dict__,
                        "instructions":BASE_AGENT.instructions+
                          f"\n\nGebruik session_id=\"{sid}\" in tools."})

    res=Runner().run_sync(agent_inst,items)
    st["history"]=res.to_input_list(); persist(sid)

    opts,cur=ui_options(st)
    pub={k:v for k,v in st.items() if k!="history"}|{"options":opts,"current_theme":cur}

    return jsonify({"assistant_reply":str(res.final_output),
                    "session_id":sid, **pub})

# ───────── download JSON ─────────
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
    app.run(host="0.0.0.0",port=10000,debug=True)
