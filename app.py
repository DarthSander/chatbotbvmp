import os
import time
import requests
from flask import Flask, request, Response
from flask_cors import CORS, cross_origin

# ── Secrets uit environment ───────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   # Render → Environment
ASSISTANT_ID   = os.getenv("ASSISTANT_ID")     # Render → Environment
OPENAI_BASE    = "https://api.openai.com/v1"

# ── Flask + CORS ──────────────────────────────────────────────────────────────
app = Flask(__name__)

# CORS: laat alle origins toe (verfijn later tot jouw domein)
CORS(app, resources={r"/chat": {"origins": "*"}}, supports_credentials=False)

# ── OpenAI-helpers ────────────────────────────────────────────────────────────
def openai_headers():
    return {"Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type":  "application/json"}

def create_thread():
    return requests.post(f"{OPENAI_BASE}/threads",
                         headers=openai_headers(), json={}).json()["id"]

def post_message(tid, txt):
    requests.post(f"{OPENAI_BASE}/threads/{tid}/messages",
                  headers=openai_headers(),
                  json={"role": "user", "content": txt})

def run_assistant(tid):
    return requests.post(f"{OPENAI_BASE}/threads/{tid}/runs",
                         headers=openai_headers(),
                         json={"assistant_id": ASSISTANT_ID}).json()["id"]

def run_status(tid, rid):
    return requests.get(f"{OPENAI_BASE}/threads/{tid}/runs/{rid}",
                        headers=openai_headers()).json()["status"]

def last_answer(tid):
    msgs = requests.get(f"{OPENAI_BASE}/threads/{tid}/messages",
                        headers=openai_headers()).json()["data"]
    for m in reversed(msgs):
        if m["role"] == "assistant":
            return m["content"][0]["text"]["value"]
    return "Geen antwoord ontvangen."

# ── /chat-endpoint ────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
@cross_origin()                               # ← CORS-headers op POST + OPTIONS
def chat():
    user_input = request.get_json().get("message", "")
    tid = create_thread()
    post_message(tid, user_input)
    rid = run_assistant(tid)

    def stream():
        while run_status(tid, rid) != "completed":
            time.sleep(1)
            yield "."                         # wachtdotjes
        for ch in last_answer(tid):
            yield ch
            time.sleep(0.012)                # geleidelijke reveal

    return Response(stream(), content_type="text/plain")

# ── Lokaal testen ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
