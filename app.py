import os
import time
import requests
from flask import Flask, request, Response
from flask_cors import CORS

# ── Secrets uit environment ───────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   # Render → Environment
ASSISTANT_ID   = os.getenv("ASSISTANT_ID")     # Render → Environment
OPENAI_BASE    = "https://api.openai.com/v1"

# ── Flask-app + CORS ──────────────────────────────────────────────────────────
app = Flask(__name__)

# Staat alle origins toe; wil je beperken, vervang "*" door je domein.
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# ── Helpers voor OpenAI Assistants v2 ─────────────────────────────────────────
def openai_headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type":  "application/json"
    }

def create_thread() -> str:
    resp = requests.post(f"{OPENAI_BASE}/threads", headers=openai_headers(), json={})
    return resp.json().get("id", "")

def post_message(thread_id: str, user_input: str) -> None:
    requests.post(
        f"{OPENAI_BASE}/threads/{thread_id}/messages",
        headers=openai_headers(),
        json={"role": "user", "content": user_input}
    )

def run_assistant(thread_id: str) -> str:
    resp = requests.post(
        f"{OPENAI_BASE}/threads/{thread_id}/runs",
        headers=openai_headers(),
        json={"assistant_id": ASSISTANT_ID}
    )
    return resp.json().get("id", "")

def get_run_status(thread_id: str, run_id: str) -> str:
    resp = requests.get(
        f"{OPENAI_BASE}/threads/{thread_id}/runs/{run_id}",
        headers=openai_headers()
    )
    return resp.json().get("status", "")

def get_last_assistant_message(thread_id: str) -> str:
    resp = requests.get(
        f"{OPENAI_BASE}/threads/{thread_id}/messages",
        headers=openai_headers()
    )
    for msg in reversed(resp.json().get("data", [])):
        if msg.get("role") == "assistant":
            return msg["content"][0]["text"]["value"]
    return "Geen antwoord ontvangen."

# ── /chat-endpoint met streaming ──────────────────────────────────────────────
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    # Pre-flight (OPTIONS) direct OK
    if request.method == "OPTIONS":
        return Response(status=200)

    user_input = request.get_json().get("message", "")
    thread_id  = create_thread()
    post_message(thread_id, user_input)
    run_id     = run_assistant(thread_id)

    def stream():
        status = ""
        while status != "completed":
            status = get_run_status(thread_id, run_id)
            time.sleep(1)
            yield "."                       # wachtdotjes
        answer = get_last_assistant_message(thread_id)
        for ch in answer:
            yield ch
            time.sleep(0.012)               # geleidelijke reveal

    return Response(stream(), content_type="text/plain")

# ── Alleen lokaal testen ──────────────────────────────────────────────────────
if __name__ == "__main__":                   # Render gebruikt gunicorn
    app.run(debug=True, host="0.0.0.0", port=10000)
