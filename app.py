import os, json, uuid
from copy import deepcopy

from flask import Flask, request, Response, jsonify, abort
from flask_cors import CORS
from openai import OpenAI, RateLimitError
from agents import Agent, Runner, function_tool

# ─── Config ──────────────────────────────────────────────────
openai_api_key = os.getenv("OPENAI_API_KEY")
client         = OpenAI(api_key=openai_api_key)
ASSISTANT_ID   = os.getenv("ASSISTANT_ID")           # alleen /chat

ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
]

app = Flask(__name__)
CORS(app, origins=ALLOWED_ORIGINS, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# ─── /chat (endpoint met Assistants v2, ongewijzigd) ─────────
def stream_run(thread_id: str):
    with client.beta.threads.runs.stream(thread_id=thread_id,
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

    client.beta.threads.messages.create(thread_id=thread_id, role="user", content=user_msg)

    def gen():
        try:
            yield from stream_run(thread_id)
        except RateLimitError:
            yield "\n⚠️ Rate-limit; probeer later opnieuw."
        except Exception as exc:
            yield f"\n⚠️ Serverfout: {exc}"

    return Response(gen(), headers={"X-Thread-ID": thread_id}, mimetype="text/plain")

# ─── In-memory sessies ───────────────────────────────────────
SESSION: dict[str, dict] = {}   # session_id → state-dict

# ─── Tool-implementaties (zonder decorator) ─────────────────
def _register_theme(session_id: str, theme: str, description: str="") -> str:
    st = SESSION.setdefault(session_id, {"stage":"choose_theme","themes":[],
                                         "topics":{}, "qa":[], "history":[]})
    if len(st["themes"]) < 6 and theme not in [t["name"] for t in st["themes"]]:
        st["themes"].append({"name": theme, "description": description})
        if len(st["themes"]) == 6:
            st["stage"] = "choose_topic"
    return json.dumps(st)

def _register_topic(session_id: str, theme: str, topic: str, description: str="") -> str:
    st = SESSION.setdefault(session_id, {"stage":"choose_theme","themes":[],
                                         "topics":{}, "qa":[], "history":[]})
    topics = st["topics"].setdefault(theme, [])
    if len(topics) < 4 and topic not in [t["name"] for t in topics]:
        topics.append({"name": topic, "description": description})
    if st["themes"] and all(len(st["topics"].get(t["name"], [])) >= 4 for t in st["themes"]):
        st["stage"] = "qa"
    return json.dumps(st)

def _log_answer(session_id: str, theme: str, question: str, answer: str) -> str:
    st = SESSION.setdefault(session_id, {"stage":"choose_theme","themes":[],
                                         "topics":{}, "qa":[], "history":[]})
    st["qa"].append({"theme": theme, "question": question, "answer": answer})
    return "ok"

def _get_state(session_id: str) -> str:
    """Tool-variant voor de LLM; niet rechtstreeks aanroepen in Flask-code."""
    return json.dumps(SESSION.get(session_id, {}))

# ─── Tool-wrappers voor de agent ─────────────────────────────
register_theme = function_tool(_register_theme)
register_topic = function_tool(_register_topic)
log_answer     = function_tool(_log_answer)
get_state_tool = function_tool(_get_state)

BASE_AGENT = Agent(
    name="Geboorteplan-agent",
    model="gpt-4.1",
    instructions=(
        "Je bent een digitale verloskundige.\n"
        "Fase 1: vraag max. 6 thema’s en roep `register_theme`.\n"
        "Fase 2: per thema max. 4 onderwerpen via `register_topic`.\n"
        "Fase 3: stel vragen en registreer antwoorden met `log_answer`.\n"
        "Gebruik `get_state` om de huidige keuzes te zien.\n"
        "Antwoord altijd in het Nederlands."
    ),
    tools=[register_theme, register_topic, log_answer, get_state_tool],
)

# ─── /agent (endpoint Agents-SDK) ────────────────────────────
@app.post("/agent")
def agent():
    if (origin := request.headers.get("Origin")) and origin not in ALLOWED_ORIGINS:
        abort(403)

    data       = request.get_json(force=True)
    user_msg   = data.get("message", "")
    session_id = data.get("session_id") or str(uuid.uuid4())

    st = SESSION.setdefault(session_id, {"stage":"choose_theme","themes":[],
                                         "topics":{}, "qa":[], "history":[]})

    # voeg extra instructie toe zodat de LLM zijn eigen session_id kent
    agent_instance = Agent(
        **{**BASE_AGENT.__dict__,
           "instructions": BASE_AGENT.instructions +
                           f"\n\nBelangrijk: gebruik altijd `session_id=\"{session_id}\"` "
                           f"in elke tool-aanroep."}
    )

    # ---------- HIER WAS DE BUG ----------
    input_items = deepcopy(st["history"])             # eerdere InputItems
    input_items.append({"role": "user", "content": user_msg})  # ✨ als object

    # run agent
    result  = Runner().run_sync(agent_instance, input_items)
    st["history"] = result.to_input_list()

    return jsonify({
        "assistant_reply": str(result.final_output),  # ✅ werkt in 0.0.17
        "session_id": session_id,
        **public_state,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
