# app.py
import os
import time
import requests
from flask import Flask, request, Response
from flask_cors import CORS

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID   = os.getenv("ASSISTANT_ID")

app = Flask(__name__)
CORS(app)  # <— hiermee staat het alle origins toe

OPENAI_BASE = "https://api.openai.com/v1"

def openai_headers():
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

def create_thread():
    r = requests.post(f"{OPENAI_BASE}/threads", headers=openai_headers(), json={})
    return r.json().get("id", "")

def post_message(thread_id, user_input):
    requests.post(
        f"{OPENAI_BASE}/threads/{thread_id}/messages",
        headers=openai_headers(),
        json={"role": "user", "content": user_input}
    )

def run_assistant(thread_id):
    r = requests.post(
        f"{OPENAI_BASE}/threads/{thread_id}/runs",
        headers=openai_headers(),
        json={"assistant_id": ASSISTANT_ID}
    )
    return r.json().get("id", "")

def get_run_status(thread_id, run_id):
    r = requests.get(
        f"{OPENAI_BASE}/threads/{thread_id}/runs/{run_id}",
        headers=openai_headers()
    )
    return r.json()

def get_last_assistant_message(thread_id):
    r = requests.get(
        f"{OPENAI_BASE}/threads/{thread_id}/messages",
        headers=openai_headers()
    )
    msgs = r.json().get("data", [])
    for msg in reversed(msgs):
        if msg.get("role") == "assistant":
            return msg["content"][0]["text"]["value"]
    return "Geen antwoord ontvangen."

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    # Let op: flask-cors zorgt al voor CORS-headers, 
    # maar we vangen hier expliciet de OPTIONS preflight af:
    if request.method == "OPTIONS":
        return Response(status=200)

    data = request.get_json()
    user_input = data.get("message", "")
    thread_id = create_thread()
    post_message(thread_id, user_input)
    run_id = run_assistant(thread_id)

    def stream():
        status = ""
        while status != "completed":
            status_data = get_run_status(thread_id, run_id)
            status = status_data.get("status", "")
            time.sleep(1)
            yield "."  # korte indicator tijdens “queued”
        answer = get_last_assistant_message(thread_id)
        for ch in answer:
            yield ch
            time.sleep(0.012)

    return Response(stream(), content_type="text/plain")

# In productie laat je de volgende 2 regels weg / commentaar, want Render draait met gunicorn
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
