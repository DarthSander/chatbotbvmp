# app.py
import os
from flask import Flask, request, Response
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Zet straks in Render als environment variable
ASSISTANT_ID = os.getenv("ASSISTANT_ID")      # Zet straks in Render als environment variable

app = Flask(__name__)

OPENAI_BASE = "https://api.openai.com/v1"

def openai_headers():
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

def create_thread():
    r = requests.post(f"{OPENAI_BASE}/threads", headers=openai_headers(), json={})
    return r.json()["id"]

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
    return r.json()["id"]

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
    msgs = r.json()["data"]
    for msg in reversed(msgs):
        if msg["role"] == "assistant":
            return msg["content"][0]["text"]["value"]
    return "Geen antwoord ontvangen."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message")
    thread_id = create_thread()
    post_message(thread_id, user_input)
    run_id = run_assistant(thread_id)

    # Poll run status (simulate streaming)
    def stream():
        import time
        dots = 0
        status = "queued"
        while status != "completed":
            status_data = get_run_status(thread_id, run_id)
            status = status_data["status"]
            time.sleep(1)
            # Simuleer streaming door ... te sturen
            yield "." * (dots % 3 + 1)
            dots += 1
        # Antwoord ophalen & streamen
        answer = get_last_assistant_message(thread_id)
        for c in answer:
            yield c
            time.sleep(0.012)
    return Response(stream(), content_type='text/plain')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
