# ─── complete app.py ─────────────────────────────────────────
import os, json, uuid
from flask import Flask, request, Response, jsonify, abort
from flask_cors import CORS
import openai
from openai import OpenAI, RateLimitError

# ─── basisconfig ─────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
client         = OpenAI()                                       
ASSISTANT_ID   = os.getenv("ASSISTANT_ID")

ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
]

app = Flask(__name__)
CORS(
    app,
    origins=[
        "https://bevalmeteenplan.nl",
        "https://www.bevalmeteenplan.nl",
    ],
    allow_headers="*",
    methods=["GET", "POST", "OPTIONS"]
)

# ─── /chat – onveranderd (Assistants v2 + stream) ────────────
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

# ─── /agent – nieuwe route via Agents-SDK ────────────────────
from agents import Agent, Runner, function_tool

SESSION: dict[str, dict] = {}

@function_tool
def register_theme(session_id: str, theme: str, description: str = "") -> str:
    st = SESSION.setdefault(session_id,
        {"stage": "choose_theme", "themes": [], "topics": {}, "qa": []})
    if len(st["themes"]) < 6 and theme not in [t["name"] for t in st["themes"]]:
        st["themes"].append({"name": theme, "description": description})
        if len(st["themes"]) == 6:
            st["stage"] = "choose_topic"
    return json.dumps(st)

@function_tool
def register_topic(session_id: str, theme: str, topic: str, description: str = "") -> str:
    st = SESSION.setdefault(session_id,
        {"stage": "choose_theme", "themes": [], "topics": {}, "qa": []})
    topics = st["topics"].setdefault(theme, [])
    if len(topics) < 4 and topic not in [t["name"] for t in topics]:
        topics.append({"name": topic, "description": description})
    if st["themes"] and all(len(st["topics"].get(t["name"], [])) >= 4 for t in st["themes"]):
        st["stage"] = "qa"
    return json.dumps(st)

@function_tool
def log_answer(session_id: str, theme: str, question: str, answer: str) -> str:
    st = SESSION.setdefault(session_id,
        {"stage": "choose_theme", "themes": [], "topics": {}, "qa": []})
    st["qa"].append({"theme": theme, "question": question, "answer": answer})
    return "ok"

@function_tool
def get_state(session_id: str) -> str:
    return json.dumps(SESSION.get(session_id, {}))

chat_agent = Agent(
    name="Geboorteplan-agent",
    model="gpt-4.1",
    instructions=(
        "Je bent een digitale verloskundige. "
        "Fase 1: laat gebruiker max. 6 thema’s kiezen; roep register_theme. "
        "Fase 2: per thema max. 4 onderwerpen; roep register_topic. "
        "Fase 3: stel vragen en log via log_answer. "
        "Gebruik get_state om te controleren wat gekozen is."
    ),
    tools=[register_theme, register_topic, log_answer, get_state],
)

@app.post("/agent")
def agent():
    if (origin := request.headers.get("Origin")) and origin not in ALLOWED_ORIGINS:
        abort(403)

    data       = request.get_json(force=True)
    user_msg   = data.get("message", "")
    session_id = data.get("session_id") or str(uuid.uuid4())

    # RUNNER-aanroep voor openai-agents 0.0.17  (zonder keywords)
    result = Runner().run_sync(
        chat_agent,   # positioneel: agent
        user_msg      # positioneel: user message
        # géén step_id en géén memory
    )

    state = json.loads(get_state(session_id))
    return jsonify({
        "assistant_reply": result.output,
        "session_id": session_id,
        **state
    })

# ─── Local run (Render gebruikt gunicorn) ────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
