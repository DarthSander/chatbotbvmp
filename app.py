"""
app.py – Flask 2.3 / OpenAI SDK ≥1.26 + OpenAI Agents SDK (PyPI: openai-agents==0.0.17)

– /chat  : ongewijzigde Assistants-v2 streaming (met ASSISTANT_ID)
– /agent : nieuwe Agents-SDK route (GPT-4.1)  + tools register_theme / register_topic / log_answer
"""
import os, json, uuid
from flask import Flask, request, Response, jsonify, abort
from flask_cors import CORS, cross_origin
import openai
from openai import OpenAI, RateLimitError                        # SDK ≥1.26  :contentReference[oaicite:1]{index=1}

# ---------- basisconfig ------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
client         = OpenAI()                                        # gedeeld voor Assistant & Agents
ASSISTANT_ID   = os.getenv("ASSISTANT_ID")                       # jouw bestaande v2-assistant

ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
]

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/chat":  {"origins": ALLOWED_ORIGINS,
                    "methods": ["POST","OPTIONS"],
                    "allow_headers": ["Content-Type"]},
        r"/agent": {"origins": ALLOWED_ORIGINS,
                    "methods": ["POST","OPTIONS"],
                    "allow_headers": ["Content-Type"]},
    },
    supports_credentials=False
)

# ======================================================================
# 2. /chat – bestaande streaming‐endpoint (ongewijzigd)
# ======================================================================
def stream_run(thread_id: str):
    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
    ) as events:
        for ev in events:
            if ev.event == "thread.message.delta" and ev.data.delta.content:
                yield ev.data.delta.content[0].text.value
            elif ev.event == "error":
                raise RuntimeError(ev.data.message)

@app.post("/chat")
@cross_origin(origins=ALLOWED_ORIGINS)
def chat():
    origin = request.headers.get("Origin")
    if origin and origin not in ALLOWED_ORIGINS:
        abort(403)

    data       = request.get_json(force=True)
    user_msg   = data.get("message", "")
    thread_id  = data.get("thread_id") or client.beta.threads.create().id

    client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=user_msg
    )

    def generator():
        try:
            for chunk in stream_run(thread_id):
                yield chunk
        except RateLimitError:
            yield "\n⚠️ Rate-limit; probeer het later opnieuw."
        except Exception as exc:
            yield f"\n⚠️ Serverfout: {exc}"

    return Response(generator(),
                    headers={"X-Thread-ID": thread_id},
                    mimetype="text/plain")

# ======================================================================
# 3. /agent – nieuwe Agents-SDK route
# ======================================================================
from agents import Agent, Runner, function_tool                 # correct – zie GitHub-repo :contentReference[oaicite:2]{index=2}

# ---------- sessiegeheugen in RAM -------------------------------------
SESSION: dict[str, dict] = {}    # {session_id: {stage, themes, topics, qa}}

# ---------- tools ------------------------------------------------------
@function_tool
def register_theme(session_id: str, theme: str, description: str="") -> str:
    """
    Registreer een thema (max 6).
    """
    st = SESSION.setdefault(session_id,
        {"stage": "choose_theme", "themes": [], "topics": {}, "qa": []})
    if len(st["themes"]) < 6 \
       and theme not in [t["name"] for t in st["themes"]]:
        st["themes"].append({"name": theme, "description": description})
        if len(st["themes"]) == 6:
            st["stage"] = "choose_topic"
    return json.dumps(st)

@function_tool
def register_topic(session_id: str, theme: str, topic: str, description: str="") -> str:
    """
    Registreer onderwerp (max 4 per thema).
    """
    st = SESSION.setdefault(session_id,
        {"stage": "choose_theme", "themes": [], "topics": {}, "qa": []})
    topics = st["topics"].setdefault(theme, [])
    if len(topics) < 4 \
       and topic not in [t["name"] for t in topics]:
        topics.append({"name": topic, "description": description})

    all_done = st["themes"] and all(
        len(st["topics"].get(t["name"], [])) >= 4 for t in st["themes"])
    if all_done:
        st["stage"] = "qa"
    return json.dumps(st)

@function_tool
def log_answer(session_id: str, theme: str, question: str, answer: str) -> str:
    """Log Q-A in sessiegeheugen."""
    st = SESSION.setdefault(session_id,
        {"stage": "choose_theme", "themes": [], "topics": {}, "qa": []})
    st["qa"].append({"theme": theme, "question": question, "answer": answer})
    return "ok"

@function_tool
def get_state(session_id: str) -> str:
    """Return volledige JSON-state."""
    return json.dumps(SESSION.get(session_id, {}))

# ---------- agentdefinitie --------------------------------------------
chat_agent = Agent(
    model="gpt-4.1",                                              # nieuwste modelnaam 
    role=(
        "Je bent een digitale verloskundige. "
        "Fase 1 → laat gebruiker maximaal 6 thema’s kiezen, registreer met register_theme. "
        "Geef per thema een beschrijving ≤30 woorden voor hover-popup. "
        "Fase 2 → per gekozen thema maximaal 4 onderwerpen, registreer met register_topic. "
        "Fase 3 → stel concrete vragen over elk onderwerp, log via log_answer. "
        "Gebruik get_state om te controleren wat nog ontbreekt."
    ),
    tools=[register_theme, register_topic, log_answer, get_state]
)

# ---------- endpoint ---------------------------------------------------
@app.post("/agent")
@cross_origin(origins=ALLOWED_ORIGINS)
def agent():
    origin = request.headers.get("Origin")
    if origin and origin not in ALLOWED_ORIGINS:
        abort(403)

    payload    = request.get_json(force=True)
    user_msg   = payload.get("message", "")
    session_id = payload.get("session_id") or str(uuid.uuid4())

    runner = Runner(chat_agent, memory=session_id)
    result = runner.run_sync(user_msg, step_id=session_id)

    state = json.loads(get_state(session_id))
    return jsonify({
        "assistant_reply": result.output,
        "session_id": session_id,
        **state       # stage, themes, topics, qa
    })

# ======================================================================
# 4. Local run  (Render gebruikt gunicorn)
# ======================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
