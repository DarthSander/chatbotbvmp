import os
import time
import requests
from flask import Flask, request, Response
from flask_cors import CORS

# ── Config uit environment ────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   # in Render → Environment
ASSISTANT_ID   = os.getenv("ASSISTANT_ID")     # in Render → Environment
OPENAI_BASE    = "https://api.openai.com/v1"

# ── Flask-app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # staat alle origins toe; verfijn later zo nodig

# ── Helpers voor OpenAI Assistants v2 ─────────────────────────────────────────
def openai_headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

def create_thread() -> str:
    r = requests.post(f"{OPENAI_BASE}/threads", headers=openai_headers(), json={})
    return r.json().get("id", "")

def post_message(thread_id: str, user_input: str) -> None:
    requests.post(
        f"{OPENAI_BASE}/threads/{thread_id}/messages",
        headers=openai_headers(),
        json={"role": "user", "content": user_input}
    )

def run_assistant(thread_id: str) -> str:
    r = requests.post(
        f"{OPENAI_BASE}/threads/{thread_id}/runs",
        headers=openai_headers(),
        json={"assistant_id": ASSISTANT_ID}
    )
    return r.json().get("id", "")

def get_run_status(thread_id: str, run_id: str) -> dict:
    r = requests.get(
        f"{OPENAI_BASE}/threads/{thread_id}/runs/{run_id}",
        headers=openai_headers()
    )
    return r.json()

def get_last_assistant_message(thread_id: str) -> str:
    r = requests.get(
        f"{OPENAI_BASE}/threads/{thread_id}/messages",
        headers=openai_headers()
    )
    for msg in reversed(r.json().get("data", [])):
        if msg.get("role") == "assistant":
            return msg["content"][0]["text"]["value"]
    return "Geen antwoord ontvangen."

# ── /chat endpoint met streaming ──────────────────────────────────────────────
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    # Pre-flight (OPTIONS) direct afhandelen
    if request.method == "OPTIONS":
        return Response(status=200)

    # Gebruikersinput
    user_input = request.get_json().get("message", "")

    # Thread & run starten
    thread_id = create_thread()
    post_message(thread_id, user_input)
    run_id    = run_assistant(thread_id)

    # Streaminggenerator
    def stream():
        status = ""
        while status != "completed":
            status = get_run_status(thread_id, run_id).get("status", "")
            time.sleep(1)          # kleine pauze tijdens wachten
            yield "."              # placeholder voor "aan het nadenken"
        answer = get_last_assistant_message(thread_id)
        for ch in answer:
            yield ch
            time.sleep(0.012)      # geleidelijke opbouw

    return Response(stream(), content_type="text/plain")

# ── Alleen lokaal testen ──────────────────────────────────────────────────────
if __name__ == "__main__":          # Render gebruikt gunicorn → deze regel overslaan
    app.run(debug=True, host="0.0.0.0", port=10000)
