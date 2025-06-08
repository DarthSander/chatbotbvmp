# app.py

# ============================================================
# Geboorteplan-agent – VOLLEDIG BIJGEWERKTE FLASK-APP
# Versie met herstelde /iframe route
# ============================================================

from __future__ import annotations
import os, json, uuid, sqlite3
from copy import deepcopy
from typing import List, Dict, Optional
from typing_extensions import TypedDict

from flask import (
    Flask, request, Response, jsonify, abort,
    send_file, send_from_directory, render_template  # <-- 'render_template' HIER TOEGEVOEGD
)
from flask_cors import CORS
from openai import OpenAI
from agents import Agent, Runner, function_tool

# ... (de rest van de configuratie blijft hetzelfde) ...
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
    "https://chatbotbvmp.onrender.com",
    "https://www.sandervandemark.nl" # Uit oude code overgenomen
]
DB_FILE = "sessions.db"
MODEL_CHOICE = "gpt-4.1"

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates", # Zorg dat de templates map herkend wordt
    static_url_path=""
)
CORS(app, origins=ALLOWED_ORIGINS, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# ... (Alle functies zoals DEFAULT_TOPICS, init_db, get_session, etc. blijven hier ongewijzigd) ...
# ... ik sla ze hier over voor de beknoptheid, maar ze moeten in uw bestand blijven staan ...

# (PLAATS HIER ALLE FUNCTIES VANAF DEFAULT_TOPICS TOT AAN DE AGENT ENDPOINT)
# ...
# ...
# ...

# ============================================================
# De endpoints
# ============================================================

# (HIER KOMEN DE /chat en /agent endpoints zoals ze al waren)
# ...
# ...

# ---------- export endpoint ----------
@app.get("/export/<sid>")
def export_json(sid: str):
    st = load_state(sid)
    if not st:
        abort(404)
    temp_dir = os.environ.get("TMPDIR", "/tmp")
    path = os.path.join(temp_dir, f"geboorteplan_{sid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))

# ============================================================
# HERSTELDE IFRAME ROUTE
# ============================================================
@app.route('/iframe')
def iframe_page():
    # Geef de Render backend URL door aan het template, net als in de oude code
    backend_url = os.getenv("RENDER_EXTERNAL_URL", "http://127.0.0.1:10000")
    return render_template('iframe_page.html', backend_url=backend_url)

# ============================================================
# SPA-fallback: serveer frontend-bestanden uit /static
# Deze blijft bestaan voor de hoofdpagina en andere routes
# ============================================================
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    # Voorkom dat deze route de /iframe route overschrijft
    if path == "iframe":
        return iframe_page()
    
    full_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

# ============================================================
# Remove X-Frame-Options zodat embedding mogelijk is
# ============================================================
@app.after_request
def allow_iframe(response):
    response.headers.pop("X-Frame-Options", None)
    return response

# ============================================================
# Run de app
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)

