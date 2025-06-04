# app.py – Flask 2.3+ / OpenAI SDK ≥ 1.25
import os, json
from flask import Flask, request, Response, abort
from flask_cors import CORS, cross_origin
import openai
from openai import OpenAI, RateLimitError

# ── 1. Basisconfig ───────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
client         = OpenAI()                      # SDK 1.25+ → runs.stream
ASSISTANT_ID   = os.getenv("ASSISTANT_ID")

ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl"           # (optioneel) www-variant
]

app = Flask(__name__)

# CORS voor alleen /chat: toegestaan origins + methoden + headers
CORS(
    app,
    resources={r"/chat": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
    }},
    supports_credentials=False     # zet True als je cookies/Authorization header nodig hebt
)

# ── 2. Streaming helper ──────────────────────────────────────
def stream_run(thread_id: str):
    """
    Generator die tekstchunks uit OpenAI Assistants-stream oplevert
    """
    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
        # stream_mode="text"  # → iets snellere SSE-events
    ) as events:
        for ev in events:
            if ev.event == "thread.message.delta" and ev.data.delta.content:
                yield ev.data.delta.content[0].text.value
            elif ev.event == "error":
                raise RuntimeError(ev.data.message)

# ── 3. Route /chat ───────────────────────────────────────────
@app.post("/chat")
@cross_origin(origins=ALLOWED_ORIGINS)         # fallback voor zekerheid
def chat():
    origin = request.headers.get("Origin")     # None bij cURL
    if origin and origin not in ALLOWED_ORIGINS:
        abort(403)                             # expliciet afwijzen

    data       = request.get_json(force=True)
    user_msg   = data.get("message", "")
    thread_id  = data.get("thread_id") or client.beta.threads.create().id

    # 3a. Voeg user-bericht toe aan de thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_msg
    )

    # 3b. Stream assistent-antwoord naar de browser
    def generator():
        try:
            for chunk in stream_run(thread_id):
                yield chunk
        except RateLimitError:
            yield "\n⚠️ Rate-limit; probeer het zo dadelijk opnieuw."
        except Exception as exc:
            yield f"\n⚠️ Serverfout: {exc}"

    headers = {"X-Thread-ID": thread_id}
    return Response(generator(), headers=headers, mimetype="text/plain")

# ── 4. Local run (Render gebruikt Gunicorn) ─────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
