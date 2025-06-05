import os, json, uuid
from copy import deepcopy
from flask import Flask, request, Response, jsonify, abort
from flask_cors import CORS

import openai
from openai import OpenAI, RateLimitError

from agents import Agent, Runner, function_tool

# ─── Basisconfig ────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
client         = OpenAI()
ASSISTANT_ID   = os.getenv("ASSISTANT_ID")        # alleen nog gebruikt door /chat

ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
]

app = Flask(__name__)
CORS(app, origins=ALLOWED_ORIGINS,
          allow_headers="*",
          methods=["GET", "POST", "OPTIONS"])

# ─── Streaming-endpoint (onveranderd) ───────────────────────
def stream_run(thread_id: str):
    with client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID) as events:
        for ev in events:
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

    client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=user_msg)

    def generator():
        try:
            for chunk in stream_run(thread_id):
                yield chunk
        except RateLimitError:
            yield "\n⚠️ Rate-limit; probeer later opnieuw."
        except Exception as exc:
            yield f"\n⚠️ Serverfout: {exc}"

    return Response(generator(),
                    headers={"X-Thread-ID": thread_id},
                    mimetype="text/plain")

# ─── In-memory sessie-opslag ────────────────────────────────
SESSION: dict[str, dict] = {}      # <session_id> → state‐dict

# ─── Tool-implementaties (zonder decorator) ────────────────
def _register_theme(session_id: str, theme: str, description: str = "") -> str:
    st = SESSION.setdefault(session_id, {
        "stage": "choose_theme",
        "themes": [],
        "topics": {},
        "qa": [],
        "history": [],
    })
    if len(st["themes"]) < 6 and theme not in [t["name"] for t in st["themes"]]:
        st["themes"].append({"name": theme, "description": description})
        if len(st["themes"]) == 6:
            st["stage"] = "choose_topic"
    return json.dumps(st)

def _register_topic(session_id: str, theme: str, topic: str,
                    description: str = "") -> str:
    st = SESSION.setdefault(session_id, {
        "stage": "choose_theme",
        "themes": [],
        "topics": {},
        "qa": [],
        "history": [],
    })
    topics = st["topics"].setdefault(theme, [])
    if len(topics) < 4 and topic not in [t["name"] for t in topics]:
        topics.append({"name": topic, "description": description})
    # Ga naar QA-fase wanneer elk gekozen thema ≥4 topics heeft
    if st["themes"] and all(len(st["topics"].get(t["name"], [])) >= 4
                            for t in st["themes"]):
        st["stage"] = "qa"
    return json.dumps(st)

def _log_answer(session_id: str, theme: str,
                question: str, answer: str) -> str:
    st = SESSION.setdefault(session_id, {
        "stage": "choose_theme",
        "themes": [],
        "topics": {},
        "qa": [],
        "history": [],
    })
    st["qa"].append({"theme": theme, "question": question, "answer": answer})
    return "ok"

def _get_state(session_id: str) -> str:
    """Niet door Flask aanroepen, alleen voor het LLM als tool."""
    return json.dumps(SESSION.get(session_id, {}))

# ─── Tools voor de Agent (met decorator) ────────────────────
register_theme   = function_tool(_register_theme)
register_topic   = function_tool(_register_topic)
log_answer       = function_tool(_log_answer)
get_state_tool   = function_tool(_get_state)

# ─── Basis-agent (her-gebruikt bij iedere sessie) ───────────
BASE_AGENT = Agent(
    name="Geboorteplan-agent",
    model="gpt-4.1",
    instructions=(
        "Je bent een digitale verloskundige.\n"
        "Fase 1: laat gebruiker maximaal 6 thema’s kiezen; roep hiervoor "
        "`register_theme`.\n"
        "Fase 2: per thema maximaal 4 onderwerpen; gebruik `register_topic`.\n"
        "Fase 3: stel gerichte vragen en log antwoorden via `log_answer`.\n"
        "Gebruik `get_state` om te zien wat er al gekozen is.\n"
        "Antwoord altijd in het Nederlands."
    ),
    tools=[register_theme, register_topic, log_answer, get_state_tool],
)

# ─── /agent – Agents-SDK endpoint ───────────────────────────
@app.post("/agent")
def agent():
    if (origin := request.headers.get("Origin")) and origin not in ALLOWED_ORIGINS:
        abort(403)

    data       = request.get_json(force=True)
    user_msg   = data.get("message", "")
    session_id = data.get("session_id") or str(uuid.uuid4())

    # Zorg dat er een staat-dict bestaat
    st = SESSION.setdefault(session_id, {
        "stage": "choose_theme",
        "themes": [],
        "topics": {},
        "qa": [],
        "history": [],
    })

    # Dynamische instructie zodat het LLM weet welk session_id te gebruiken
    extra_instr = (
        f"\n\nBelangrijk: gebruik in elke tool-aanroep precies "
        f"het argument `session_id=\"{session_id}\"`."
    )
    agent_instance = Agent(
        name       = BASE_AGENT.name,
        model      = BASE_AGENT.model,
        instructions = BASE_AGENT.instructions + extra_instr,
        tools      = BASE_AGENT.tools,
    )

    # Bouw input-lijst op basis van opgeslagen geschiedenis
    history = deepcopy(st["history"])
    history.append(user_msg)

    # Voer de agent uit (synchroon voor eenvoud; streamen kan met run_streamed)
    result = Runner().run_sync(agent_instance, history)
    assistant_reply = str(result.output)

    # Conversatiegeschiedenis opslaan
    st["history"] = result.to_input_list()

    # JSON-payload terugsturen
    public_state = {k: v for k, v in st.items() if k != "history"}
    return jsonify({
        "assistant_reply": assistant_reply,
        "session_id": session_id,
        **public_state,
    })

# ─── Local run (Render gebruikt gunicorn) ───────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
