# app.py – Geboorteplan-agent – Volledige, gecorrigeerde en complete versie

from __future__ import annotations
import os
import json
import uuid
import sqlite3
import time
from copy import deepcopy
from typing import List, Dict, Optional
from typing_extensions import TypedDict

from flask import (
    Flask, request, Response, jsonify, abort,
    send_file, send_from_directory, render_template
)
from flask_cors import CORS
from openai import OpenAI
from agents import function_tool

# ---------- Basisconfiguratie ----------
client          = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID    = os.getenv("ASSISTANT_ID")
ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
    "https://chatbotbvmp.onrender.com"
]
DB_FILE         = "sessions.db"
MODEL_CHOICE    = "gpt-4.1"

# ---------- Flask-app setup ----------
app = Flask(__name__, static_folder="static", template_folder="templates", static_url_path="")
CORS(app, origins=ALLOWED_ORIGINS, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# ---------- Strikte types ----------
class NamedDescription(TypedDict):
    name: str
    description: str

# ---------- Standaardonderwerpen ----------
DEFAULT_TOPICS: Dict[str, List[NamedDescription]] = {
    "Ondersteuning": [
        {"name": "Wie wil je bij de bevalling?", "description": "Welke personen wil je er fysiek bij hebben?"},
        {"name": "Rol van je partner of ander persoon?", "description": "Specificeer taken of wensen voor je partner."},
        {"name": "Wil je een doula / kraamzorg?", "description": "Extra ondersteuning tijdens en na de bevalling."},
        {"name": "Wat verwacht je van het personeel?", "description": "Welke stijl van begeleiding past bij jou?"}
    ],
    "Bevalling & medisch beleid": [
        {"name": "Pijnstilling", "description": "Medicamenteuze en niet-medicamenteuze opties."},
        {"name": "Interventies", "description": "Bijv. inknippen, kunstverlossing, infuus."},
        {"name": "Noodsituaties", "description": "Wat als het anders loopt dan gepland?"},
        {"name": "Placenta-keuzes", "description": "Placenta bewaren, laten staan, of doneren?"}
    ],
    "Sfeer en omgeving": [
        {"name": "Muziek & verlichting", "description": "Rustige muziek? Gedimd licht?"},
        {"name": "Privacy", "description": "Wie mag binnenkomen en fotograferen?"},
        {"name": "Foto’s / video", "description": "Wil je opnames laten maken?"},
        {"name": "Eigen spulletjes", "description": "Bijv. eigen kussen, etherische olie."}
    ],
    "Voeding na de geboorte": [
        {"name": "Borstvoeding", "description": "Ondersteuning, kolven, rooming-in."},
        {"name": "Flesvoeding", "description": "Welke melk? Wie geeft de fles?"},
        {"name": "Combinatie-voeding", "description": "Afwisselen borst en fles."},
        {"name": "Allergieën", "description": "Rekening houden met familiaire allergieën."}
    ]
}

# ---------- SQLite-helper functions ----------
def init_db() -> None:
    # BELANGRIJK: Als je deze app draait met een oude 'sessions.db', verwijder die file dan eerst.
    # Deze functie maakt een nieuwe db aan, maar past een bestaande NIET aan.
    with sqlite3.connect(DB_FILE) as con:
        con.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, state TEXT NOT NULL, user_profile TEXT)")
init_db()

def load_state(sid: str) -> Optional[dict]:
    with sqlite3.connect(DB_FILE) as con:
        row = con.execute("SELECT state, user_profile FROM sessions WHERE id=?", (sid,)).fetchone()
        if not row:
            return None
        state_json, user_profile_json = row
        st = json.loads(state_json)
        if user_profile_json:
            st["user_profile"] = json.loads(user_profile_json)
        return st

def save_state(sid: str, st: dict) -> None:
    with sqlite3.connect(DB_FILE) as con:
        st_to_save = st.copy()
        user_profile = st_to_save.pop("user_profile", None)
        con.execute("REPLACE INTO sessions (id, state, user_profile) VALUES (?, ?, ?)", (sid, json.dumps(st_to_save), json.dumps(user_profile)))
        con.commit()

# ---------- In-memory sessies + persistence ----------
SESSION: Dict[str, dict] = {}

def get_session(sid: str) -> dict:
    if sid in SESSION:
        return SESSION[sid]

    # CORRECTIE: De 'walrus operator' (:=) is hier verwijderd voor maximale compatibiliteit.
    db_state = load_state(sid)
    if db_state:
        st = db_state
        st.setdefault("id", sid)
        st.setdefault("history", [])
        st.setdefault("last_interaction_timestamp", time.time())
        SESSION[sid] = st
        return st

    # Nieuwe sessie
    st = {
        "id": sid, "stage": "choose_theme", "themes": [], "topics": {}, "qa": [], "history": [],
        "summary": "", "ui_theme_opts": [], "ui_topic_opts": [], "current_theme": None,
        "generated_topic_options": {}, "user_profile": None, "last_interaction_timestamp": time.time()
    }
    SESSION[sid] = st
    return st

def persist(sid: str) -> None:
    if sid in SESSION:
        save_state(sid, SESSION[sid])

# ---------- History-samenvatting ----------
def summarize_chunk(chunk: List[dict]) -> str:
    if not chunk:
        return ""
    filtered = [m for m in chunk if isinstance(m, dict) and m.get('content')]
    text = "\n".join(f"{m['role']}: {m['content']}" for m in filtered)
    prompt = "Vat dit deel van het gesprek over een geboorteplan samen in maximaal 300 tokens. Focus op de keuzes, wensen en open vragen."
    r = client.chat.completions.create(model=MODEL_CHOICE, messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}], max_tokens=300)
    return r.choices[0].message.content.strip()

# ============================================================
# Tool-wrappers
# ============================================================
def _set_theme_options(session_id: str, options: List[str]) -> str:
    st = get_session(session_id)
    st["ui_theme_opts"] = options
    persist(session_id)
    return "ok"

def _set_topic_options(session_id: str, theme: str, options: List[NamedDescription]) -> str:
    st = get_session(session_id)
    st.setdefault("generated_topic_options", {})[theme] = options
    st["ui_topic_opts"] = options
    st["current_theme"] = theme
    persist(session_id)
    return "ok"

def _register_theme(session_id: str, theme: str, description: str = "") -> str:
    st = get_session(session_id)
    if len(st["themes"]) < 6 and theme not in [t["name"] for t in st["themes"]]:
        st["themes"].append({"name": theme, "description": description})
        st["stage"] = "choose_topic"
        st["ui_topic_opts"] = DEFAULT_TOPICS.get(theme, [])
        st["current_theme"] = theme
        persist(session_id)
    return "ok"

def _register_topic(session_id: str, theme: str, topic: str) -> str:
    st = get_session(session_id)
    lst = st["topics"].setdefault(theme, [])
    if len(lst) < 4 and topic not in lst:
        lst.append(topic)
        persist(session_id)
    return "ok"

def _complete_theme(session_id: str) -> str:
    st = get_session(session_id)
    all_ok = all(t["name"] in st["topics"] and st["topics"][t["name"]] for t in st["themes"])
    st["stage"] = "qa" if all_ok else "choose_theme"
    st["ui_topic_opts"] = []
    st["current_theme"] = None
    persist(session_id)
    return "ok"

def _log_answer(session_id: str, theme: str, question: str, answer: str) -> str:
    st = get_session(session_id)
    found = False
    for qa in st["qa"]:
        if qa["theme"] == theme and qa["question"] == question:
            qa["answer"] = answer
            found = True
            break
    if not found:
        st["qa"].append({"theme": theme, "question": question, "answer": answer})
    persist(session_id)
    return "ok"

def _get_state(session_id: str) -> str:
    return json.dumps(get_session(session_id))

set_theme_options = function_tool(_set_theme_options)
set_topic_options = function_tool(_set_topic_options)
register_theme    = function_tool(_register_theme)
register_topic    = function_tool(_register_topic)
complete_theme    = function_tool(_complete_theme)
log_answer        = function_tool(_log_answer)
get_state_tool    = function_tool(_get_state)

# ============================================================
# NIEUWE Functies (Sentiment, Recovery, etc.)
# ============================================================
@function_tool
def detect_sentiment(session_id: str, message: str) -> str:
    """
    Analyseer 'message' op sentiment. Crasht niet bij een API-fout, maar geeft 'neutral' terug.
    """
    # Geen sentiment nodig voor lege berichten
    if not message.strip():
        return "neutral"

    prompt = f"Beoordeel de toon van het volgende bericht als 'positive', 'neutral' of 'negative'. Antwoord alleen met het woord.\nBericht: '{message}'"
    
    try:
        # Controleer of de API key überhaupt aanwezig is
        if not client.api_key:
            print("Waarschuwing: OPENAI_API_KEY niet gevonden. Sentiment-detectie overgeslagen.")
            return "neutral"

        resp = client.chat.completions.create(
            model=MODEL_CHOICE,
            messages=[{"role":"system", "content": prompt}],
            max_tokens=5, # Iets meer ruimte voor zekerheid
            temperature=0
        )
        sentiment = resp.choices[0].message.content.strip().lower()
        
        if sentiment in ['positive', 'neutral', 'negative']:
            return sentiment
        else:
            # Fallback als de LLM iets onverwachts retourneert
            return "neutral"
            
    except Exception as e:
        # Vang ALLE mogelijke fouten af (authenticatie, netwerk, etc.)
        print(f"Fout tijdens sentiment-detectie: {e}. Keer terug naar 'neutral'.")
        return "neutral"

def handle_session_recovery(sid: str) -> dict:
    """Zorgt ervoor dat we een geldige sessie hebben. De logica zit in get_session."""
    return get_session(sid)

def handle_error(st: dict, msg: str) -> Optional[str]:
    """Controleert op ongeldige input (bv. alleen spaties)."""
    if not msg.strip() and len(msg) > 0:
        return "Sorry, er ging iets mis. Kun je je vraag herhalen?"
    return None

def validate_answer(st: dict, msg: str) -> Optional[str]:
    """Valideert of een antwoord in de Q&A fase niet leeg is."""
    is_qa_stage = st.get("stage") == "qa"
    last_q_asked = st.get("history") and st["history"][-1]["content"].startswith("Vraag:")
    if is_qa_stage and last_q_asked and not msg.strip():
        return "Ik heb je antwoord niet goed begrepen, kun je het anders verwoorden?"
    return None

def update_user_model(st: dict) -> None:
    """Werkt het gebruikersprofiel bij met thema- en topickeuzes."""
    st["user_profile"] = {"themes": st.get("themes", []), "topics": st.get("topics", {})}
    persist(st["id"])
    print(f"User model updated for session {st['id']}")

# ============================================================
# Phase Handlers (Gedragslogica van de agent)
# ============================================================
def handle_theme_selection(st: dict, msg: str) -> Optional[str]:
    if st["stage"] != "choose_theme":
        return None
    if not st["ui_theme_opts"]:
        _set_theme_options(st["id"], list(DEFAULT_TOPICS.keys()))
        return "Laten we beginnen met het kiezen van de thema's voor jouw geboorteplan. Welke onderwerpen spreken je aan?"
    if msg in DEFAULT_TOPICS:
        _register_theme(st["id"], msg, "")
        return f"Oké, thema '{msg}' is toegevoegd. Je kunt nu onderwerpen voor dit thema kiezen, of een volgend thema selecteren."
    return None

def handle_topic_selection(st: dict, msg: str) -> Optional[str]:
    if st["stage"] != "choose_topic":
        return None
    theme = st["current_theme"]
    if not theme: # Veiligheidscheck
        st["stage"] = "choose_theme"
        return "Er was iets misgegaan. Kies eerst opnieuw een thema."
    if not st["ui_topic_opts"]:
        existing = st["generated_topic_options"].get(theme)
        options = existing if existing else DEFAULT_TOPICS.get(theme, [])
        _set_topic_options(st["id"], theme, options)
        return f"Kies nu de onderwerpen die je wilt bespreken voor het thema '{theme}'."
    names = [t["name"] for t in st.get("ui_topic_opts", [])]
    if msg in names:
        _register_topic(st["id"], theme, msg)
        return f"Onderwerp '{msg}' toegevoegd aan thema '{theme}'. Je kunt meer onderwerpen kiezen of aangeven dat je klaar bent met dit thema."
    return None

def handle_complete_selection(st: dict, msg: str) -> Optional[str]:
    # Deze handler wordt geactiveerd als de gebruiker klaar is met een selectie
    if "klaar" in msg.lower() or "verder" in msg.lower() or "volgende" in msg.lower():
        if st["stage"] == "choose_topic":
            st["stage"] = "choose_theme"
            st["ui_topic_opts"] = []
            st["current_theme"] = None
            persist(st["id"])
            return "Oké, we gaan terug naar de thema's. Kies een nieuw thema, of zeg dat je klaar bent met alle keuzes om naar de vragen te gaan."
        elif st["stage"] == "choose_theme":
            _complete_theme(st["id"])
            if st["stage"] == "qa":
                # Start de eerste vraag direct
                return handle_qa(st, "")
            else:
                return "Je moet voor elk gekozen thema minstens één onderwerp selecteren. Kies een thema om onderwerpen toe te voegen."
    return None

def handle_qa(st: dict, msg: str) -> Optional[str]:
    if st["stage"] != "qa":
        return None
    
    # Log het antwoord op de vorige vraag, als die er was
    if msg.strip() and st["history"] and st["history"][-1]["role"] == "assistant" and st["history"][-1]["content"].startswith("Vraag:"):
        current_qa_topic = st.get("current_qa_topic", {})
        if current_qa_topic:
            _log_answer(st["id"], current_qa_topic.get("theme"), current_qa_topic.get("question"), msg)

    # Vind de volgende onbeantwoorde vraag
    answered_qs = {(qa["theme"], qa["question"]) for qa in st.get("qa", [])}
    next_question_found = None
    for theme_info in st.get("themes", []):
        theme_name = theme_info["name"]
        for topic_name in st.get("topics", {}).get(theme_name, []):
            topic_obj = next((t for t in DEFAULT_TOPICS.get(theme_name, []) if t["name"] == topic_name), None)
            if topic_obj and (theme_name, topic_obj["description"]) not in answered_qs:
                next_question_found = {"theme": theme_name, "question": topic_obj["description"]}
                break
        if next_question_found:
            break

    if next_question_found:
        st["current_qa_topic"] = next_question_found
        persist(st["id"])
        return f"Vraag: {next_question_found['question']} (onderwerp: {next_question_found['theme']})"
    else:
        st["stage"] = "completed"
        update_user_model(st) # Werk profiel bij aan het einde
        persist(st["id"])
        return "Je hebt alle vragen beantwoord! Je kunt je geboorteplan nu exporteren."

def handle_proactive_help(st: dict, msg: str) -> Optional[str]:
    if time.time() - st.get("last_interaction_timestamp", 0) > 300: # 5 minuten
        st["last_interaction_timestamp"] = time.time()
        return "Ik zie dat je even stilzit. Kan ik ergens mee helpen?"
    return None

def handle_fallback(st: dict, msg: str) -> str:
    return "Ik begrijp je verzoek niet helemaal in deze context. Kun je het anders proberen? Je kunt bijvoorbeeld een thema kiezen, of 'klaar' zeggen als je een selectie wilt afronden."

# ============================================================
# Streaming-/chat-endpoint (optioneel, voor andere interfaces)
# ============================================================
@app.post("/chat")
def chat():
    origin = request.headers.get("Origin")
    if origin and origin not in ALLOWED_ORIGINS:
        abort(403)
    data = request.get_json(force=True)
    msg = data.get("message", "")
    tid = data.get("thread_id") or client.beta.threads.create().id
    client.beta.threads.messages.create(thread_id=tid, role="user", content=msg)
    def stream_run(tid: str):
        with client.beta.threads.runs.stream(thread_id=tid, assistant_id=ASSISTANT_ID) as ev:
            for e in ev:
                if e.event == "thread.message.delta" and e.data.delta.content:
                    yield e.data.delta.content[0].text.value
    return Response(stream_run(tid), headers={"X-Thread-ID": tid}, mimetype="text/plain")

# ============================================================
# Synchronous /agent-endpoint (Hoofdlogica)
# ============================================================
def _create_json_response(st: dict, assistant_reply: str):
    """Maakt een JSON-response op een manier die compatibel is met alle Python-versies."""
    response_data = {
        "assistant_reply": assistant_reply,
        "session_id": st["id"],
        "options": st.get("ui_topic_opts") if st.get("stage") == "choose_topic" else st.get("ui_theme_opts"),
        "current_theme": st.get("current_theme"),
    }
    filtered_state = {k: v for k, v in st.items() if k not in ("ui_theme_opts", "ui_topic_opts")}
    response_data.update(filtered_state)
    return jsonify(response_data)

@app.post("/agent")
def agent():
    origin = request.headers.get("Origin")
    if origin and origin not in ALLOWED_ORIGINS:
        abort(403)
    body = request.get_json(force=True)
    msg = body.get("message", "")
    sid = body.get("session_id") or str(uuid.uuid4())
    
    st = handle_session_recovery(sid)
    st["last_interaction_timestamp"] = time.time()
    st["history"].append({"role": "user", "content": msg})

    if len(st["history"]) > 40:
        st["summary"] += "\n" + summarize_chunk(st["history"][:-20])
        st["history"] = st["history"][-20:]

    sentiment = detect_sentiment(st["id"], msg)
    if sentiment == "negative":
        reply = "Ik merk dat dit misschien een gevoelig of frustrerend punt voor je is. Laten we hier even bij stilstaan. Kan ik iets voor je verduidelijken of op een andere manier helpen?"
        st["history"].append({"role": "assistant", "content": reply})
        persist(sid)
        return _create_json_response(st, reply)

    for check in [handle_error, validate_answer]:
        response = check(st, msg)
        if response:
            st["history"].append({"role": "assistant", "content": response})
            persist(sid)
            return _create_json_response(st, response)

    reply: Optional[str] = None
    for handler in (handle_theme_selection, handle_topic_selection, handle_complete_selection, handle_qa, handle_proactive_help):
        reply = handler(st, msg)
        if reply is not None:
            break
    if reply is None:
        reply = handle_fallback(st, msg)
    
    st["history"].append({"role": "assistant", "content": reply})
    persist(sid)
    return _create_json_response(st, reply)

# ============================================================
# Andere endpoints (Export, Frontend Serving)
# ============================================================
@app.get("/export/<sid>")
def export_json(sid: str):
    st = load_state(sid)
    if not st:
        abort(404)
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"geboorteplan_{sid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))

@app.route('/iframe')
def iframe_page():
    return render_template('iframe_page.html', backend_url=os.getenv("RENDER_EXTERNAL_URL", "http://127.0.0.1:10000"))

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path == "iframe":
        return iframe_page()
    full_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

@app.after_request
def allow_iframe(response):
    response.headers.pop("X-Frame-Options", None)
    return response

# ============================================================
# Run the app
# ============================================================
if __name__ == "__main__":
    # debug=True is nuttig voor ontwikkeling om gedetailleerde fouten te zien.
    # Voor live-productie kun je dit op False zetten.
    app.run(host="0.0.0.0", port=10000, debug=True)
