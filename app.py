# app.py  –  Flask 2.3+ / OpenAI SDK ≥ 1.26  /  OpenAI-Agents ≥ 0.2
import os, json, uuid
from flask import Flask, request, Response, jsonify, abort
from flask_cors import CORS, cross_origin                       # CORS helper :contentReference[oaicite:2]{index=2}
import openai
from openai import OpenAI, RateLimitError                        # SDK ≥ 1.26 :contentReference[oaicite:3]{index=3}

# ───────────────────────────────────────────────────────────────
# 1. Basisconfig
# ───────────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
client         = OpenAI()                                        # één client voor alles
ASSISTANT_ID   = os.getenv("ASSISTANT_ID")                       # bestaande Assistant-v2

ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
]

app = Flask(__name__)

# één CORS-blok voor zowel /chat als /agent
CORS(
    app,
    resources={
        r"/chat":  {"origins": ALLOWED_ORIGINS,
                    "methods": ["POST", "OPTIONS"],
                    "allow_headers": ["Content-Type"]},
        r"/agent": {"origins": ALLOWED_ORIGINS,
                    "methods": ["POST", "OPTIONS"],
                    "allow_headers": ["Content-Type"]},
    },
    supports_credentials=False
)

# ───────────────────────────────────────────────────────────────
# 2. /chat – bestaande Assistant-v2 stream (ongewijzigd)
# ───────────────────────────────────────────────────────────────
def stream_run(thread_id: str):
    """
    Generator die tekst-chunks uit OpenAI Assistants-v2 streamt
    """                                                     # streaming-pattern uit Flask-docs :contentReference[oaicite:4]{index=4}
    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
    ) as events:
        for ev in events:
            if ev.event == "thread.message.delta" and ev.data.delta.content:
                yield ev.data.delta.content[0].text.value
            elif ev.event == "error":
                raise RuntimeError(ev.data.message)         # fallback

@app.post("/chat")
@cross_origin(origins=ALLOWED_ORIGINS)
def chat():
    origin = request.headers.get("Origin")
    if origin and origin not in ALLOWED_ORIGINS:
        abort(403)

    data      = request.get_json(force=True)
    user_msg  = data.get("message", "")
    thread_id = data.get("thread_id") or client.beta.threads.create().id

    # 1) voeg user-bericht toe
    client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=user_msg
    )

    # 2) stream assistant-antwoord door
    def generator():
        try:
            for chunk in stream_run(thread_id):
                yield chunk
        except RateLimitError:                              # rate-limit handler :contentReference[oaicite:5]{index=5}
            yield "\n⚠️ Rate-limit; probeer het later opnieuw."
        except Exception as exc:
            yield f"\n⚠️ Serverfout: {exc}"

    headers = {"X-Thread-ID": thread_id}
    return Response(generator(), headers=headers,
                    mimetype="text/plain")

# ───────────────────────────────────────────────────────────────
# 3. /agent – nieuwe route met OpenAI Agents-SDK
# ───────────────────────────────────────────────────────────────
from openai_agents import Agent, Runner, function_tool          # correcte import :contentReference[oaicite:6]{index=6}

# In-memory sessiestatus  {session_id: {...}}
SESSION: dict[str, dict] = {}

# ------------------------------- Tools ------------------------
@function_tool
def register_theme(session_id: str, theme: str,
                   description: str = "") -> str:
    """
    Registreer een gekozen thema (max 6)
    """
    st = SESSION.setdefault(session_id,
            {"stage": "choose_theme",
             "themes": [],                # [{name,description}]
             "topics": {},                # {theme: [{name,description}]}
             "qa": []})                   # [{theme, question, answer}]
    if len(st["themes"]) < 6 \
       and theme not in [t["name"] for t in st["themes"]]:
        st["themes"].append(
            {"name": theme, "description": description})
        if len(st["themes"]) == 6:
            st["stage"] = "choose_topic"
    return json.dumps(st)

@function_tool
def register_topic(session_id: str, theme: str,
                   topic: str, description: str = "") -> str:
    """
    Registreer een onderwerp binnen een thema (max 4)
    """
    st = SESSION.setdefault(session_id,
            {"stage": "choose_theme",
             "themes": [], "topics": {}, "qa": []})
    topics = st["topics"].setdefault(theme, [])
    if len(topics) < 4 \
       and topic not in [t["name"] for t in topics]:
        topics.append({"name": topic, "description": description})

    complete = st["themes"] and all(
        len(st["topics"].get(t["name"], [])) >= 4
        for t in st["themes"])
    if complete:
        st["stage"] = "qa"
    return json.dumps(st)

@function_tool
def log_answer(session_id: str, theme: str,
               question: str, answer: str) -> str:
    """Bewaar Q-A-paar voor de rechterkolom"""
    st = SESSION.setdefault(session_id,
            {"stage": "choose_theme",
             "themes": [], "topics": {}, "qa": []})
    st["qa"].append({
        "theme": theme, "question": question, "answer": answer})
    return "ok"

@function_tool
def get_state(session_id: str) -> str:
    """Geef de volledige sessiestatus terug"""
    return json.dumps(SESSION.get(session_id, {}))

# ------------------------------- Agent ------------------------
chat_agent = Agent(
    model="gpt-4.1",                                         # nieuwste model :contentReference[oaicite:7]{index=7}
    role=(
        "Je bent een digitale verloskundige. "
        "FASE 1 — laat gebruiker maximaal 6 thema’s kiezen; "
        "roep bij elk register_theme. "
        "Beschrijf elk thema in ≤ 30 woorden voor hover-popup. "
        "FASE 2 — per thema ≤ 4 onderwerpen; roep register_topic. "
        "FASE 3 — stel vragen over elk onderwerp; "
        "log antwoorden via log_answer. "
        "Gebruik get_state om te zien wat al gekozen is."
    ),
    tools=[register_theme, register_topic,
           log_answer, get_state]
)

# ---------------------------- /agent --------------------------
@app.post("/agent")
@cross_origin(origins=ALLOWED_ORIGINS)
def agent():
    origin = request.headers.get("Origin")
    if origin and origin not in ALLOWED_ORIGINS:
        abort(403)

    payload    = request.get_json(force=True)
    user_msg   = payload.get("message", "")
    session_id = payload.get("session_id") or str(uuid.uuid4())  # random UUID :contentReference[oaicite:8]{index=8}

    # run synchronously (makkelijk voor HTTP)
    runner = Runner(chat_agent, memory=session_id)
    result = runner.run_sync(user_msg, step_id=session_id)

    state = json.loads(get_state(session_id))
    return jsonify({
        "assistant_reply": result.output,
        "session_id": session_id,
        **state    # stage, themes, topics, qa
    })

# ───────────────────────────────────────────────────────────────
# 4. Local run (Render gebruikt Gunicorn)
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # localhost-debug; Render gebruikt $PORT met gunicorn
    app.run(host="0.0.0.0", port=10000, debug=True)
