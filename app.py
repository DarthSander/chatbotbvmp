# app.py – Flask 2.3+  /  OpenAI SDK ≥ 1.25  /  Agents SDK 0.2+
import os, json, uuid
from flask import Flask, request, Response, jsonify, abort
from flask_cors import CORS, cross_origin
import openai
from openai import OpenAI, RateLimitError

# ---------- bestaande Assistants v2 config ----------
openai.api_key = os.getenv("OPENAI_API_KEY")
client         = OpenAI()            # gebruikt zowel Assistants v2 als Agents
ASSISTANT_ID   = os.getenv("ASSISTANT_ID")

ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl"
]

app = Flask(__name__)
CORS(app,
     resources={r"/chat": {"origins": ALLOWED_ORIGINS,
                           "methods": ["POST","OPTIONS"],
                           "allow_headers": ["Content-Type"]},
                r"/agent": {"origins": ALLOWED_ORIGINS}},
     supports_credentials=False)

# ---------- bestaande /chat (streaming) onveranderd ----------
def stream_run(thread_id:str):
    with client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID):
        …

@app.post("/chat")
@cross_origin(origins=ALLOWED_ORIGINS)
def chat():
    …  # ongewijzigde logica

# ======================================================================
# Nieuwe : /agent  – OpenAI Agents SDK  (gpt-4.1)                       |
# ======================================================================
from agents import Agent, Runner, function_tool    # pip install openai-agents-python

# ---- 1. very lightweight session state (in-memory) -------------------
SESSION: dict[str, dict] = {}      # {session_id: {...}}

# ---- 2. tools --------------------------------------------------------
@function_tool
def register_theme(session_id:str, theme:str, description:str="") -> str:
    """
    Slaat een nieuw thema + beschrijving op (max 6).
    """
    st = SESSION.setdefault(session_id,
            {"stage":"choose_theme", "themes":[], "topics":{}, "qa":[]})
    if len(st["themes"]) < 6 and theme not in [t["name"] for t in st["themes"]]:
        st["themes"].append({"name":theme, "description":description})
        if len(st["themes"]) == 6:
            st["stage"] = "choose_topic"
    return json.dumps(st)

@function_tool
def register_topic(session_id:str, theme:str, topic:str, description:str="") -> str:
    """
    Voegt een onderwerp binnen het thema toe (max 4 p/ thema).
    """
    st = SESSION.setdefault(session_id,
            {"stage":"choose_theme", "themes":[], "topics":{}, "qa":[]})
    topics = st["topics"].setdefault(theme, [])
    if len(topics) < 4 and topic not in [t["name"] for t in topics]:
        topics.append({"name":topic, "description":description})
    # check of alle thema’s 4 topics hebben
    complete = st["themes"] and all(len(st["topics"].get(t["name"],[]))>=4
                                    for t in st["themes"])
    if complete:
        st["stage"] = "qa"
    return json.dumps(st)

@function_tool
def log_answer(session_id:str, theme:str, question:str, answer:str) -> str:
    """
    Logt het gegeven antwoord (wordt rechts getoond).
    """
    st = SESSION.setdefault(session_id,
            {"stage":"choose_theme", "themes":[], "topics":{}, "qa":[]})
    st["qa"].append({"theme":theme, "question":question, "answer":answer})
    return "ok"

@function_tool
def get_state(session_id:str) -> str:
    """Geeft complete state als JSON-string terug."""
    return json.dumps(SESSION.get(session_id, {}))

# ---- 3. agent-definitie (gpt-4.1) ------------------------------------
chat_agent = Agent(
    model="gpt-4.1",                                     # nieuw model
    role=(
      "Je bent een digitale verloskundige die de gebruiker helpt een geboorteplan te maken. "
      "Fase 1: laat de gebruiker maximaal 6 thema’s kiezen. "
      "Roep voor ieder gekozen thema register_theme(session_id, theme, description) aan. "
      "Gebruik een korte beschrijving (max 30 woorden) voor de popup. "
      "Fase 2: per thema maximaal 4 relevante onderwerpen; roep register_topic. "
      "Als alle thema’s × 4 onderwerpen gekozen zijn, ga je naar QA-fase. "
      "Stel dan gerichte vragen over elk onderwerp en roep na ieder antwoord "
      "log_answer(session_id, theme, vraag, antwoord). "
      "Gebruik get_state wanneer dat nodig is om te beslissen wat nog ontbreekt."
    ),
    tools=[register_theme, register_topic, log_answer, get_state],
)

# ---- 4. /agent-endpoint ----------------------------------------------
@app.post("/agent")
def agent():
    if (origin := request.headers.get("Origin")) and origin not in ALLOWED_ORIGINS:
        abort(403)

    data       = request.get_json(force=True)
    user_msg   = data.get("message","")
    session_id = data.get("session_id") or str(uuid.uuid4())

    # run agent synchronously (eenvoudig voor HTTP)
    runner = Runner(chat_agent, memory=session_id)
    result = runner.run_sync(user_msg, step_id=session_id)

    # huidige state ophalen
    state = json.loads(get_state(session_id))

    # stel UI-payload samen
    ui = {"stage": state["stage"],
          "themes": state.get("themes", []),
          "qa_log": state.get("qa", [])}

    if state["stage"] == "choose_theme":
        ui["options"] = [  # alles wat nog niet gekozen is
            "Thema-idee “{}”".format(i+1) for i in range(10)
        ]  # agent mag eigen voorstellen doen in de chat-reply
    elif state["stage"] == "choose_topic":
        # neem eerste thema dat nog topics mist
        for th in state["themes"]:
            if len(state["topics"].get(th["name"],[])) < 4:
                ui["current_theme"] = th
                ui["options"] = ["(agent stelt opties in de reply)"]
                break

    return jsonify({
        "assistant_reply": result.output,
        "session_id": session_id,
        **ui
    })

# ---------- Render / Gunicorn ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
