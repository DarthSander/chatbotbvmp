# app.py – Flask 2.3+
import os, time, json
from flask import Flask, request, Response, abort
from flask_cors import CORS
import openai
from openai import OpenAI, RateLimitError

# ╭─ 1. Basisconfig ───────────────────────────────────────────╮
openai.api_key = os.getenv("OPENAI_API_KEY")
client         = OpenAI()                                     # ↩︎ sdk 1.25+
ASSISTANT_ID   = os.getenv("ASSISTANT_ID")
ALLOWED_FRONT  = os.getenv("FRONTEND_ORIGIN", "https://jouwsite.nl")

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": ALLOWED_FRONT}})

# ╭─ 2. Helpers ───────────────────────────────────────────────╮
def stream_run(thread_id: str):
    "Yield text chunks while the run is executing."
    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
        # optioneel: stream_mode="text"  (snellere plain-text events)
    ) as events:
        for event in events:
            if (event.event == "thread.message.delta"
                    and event.data.delta.content):
                yield event.data.delta.content[0].text.value
            elif event.event == "error":
                raise RuntimeError(event.data.message)

# ╭─ 3. Route ─────────────────────────────────────────────────╮
@app.post("/chat")
def chat():
    if request.origin != ALLOWED_FRONT:
        abort(403)

    data = request.get_json(force=True)
    user_msg   = data.get("message", "")
    thread_id  = data.get("thread_id") or client.beta.threads.create().id

    # (a) voeg gebruikers-bericht toe
    client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=user_msg
    )

    # (b) stuur streaming-response naar frontend
    def generator():
        try:
            for chunk in stream_run(thread_id):
                yield chunk
        except RateLimitError:
            yield "\n⚠️ Rate-limit; probeer het zo dadelijk opnieuw."
        except Exception as e:
            yield f"\n⚠️ Serverfout: {e}"
    # Gebruik ‘text/plain’ zodat JS via ReadableStream kan lezen
    headers = {"X-Thread-ID": thread_id}
    return Response(generator(), headers=headers, mimetype="text/plain")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)