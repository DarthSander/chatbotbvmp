# app.py – Geboorteplan-agent – Volledige en Geoptimaliseerde Flask-app

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
from agents import Agent, Runner, function_tool

# ---------- strikt type voor thema’s en topics ----------
class NamedDescription(TypedDict):
    name: str
    description: str

# ---------- basisconfig ----------
client          = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID    = os.getenv("ASSISTANT_ID")
ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
    "https://chatbotbvmp.onrender.com"
]
DB_FILE         = "sessions.db"
MODEL_CHOICE    = "gpt-4.1" # Gebruikt voor sentiment-detectie en samenvatting

# ---------- Flask-app setup ----------
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
    static_url_path=""
)
CORS(app, origins=ALLOWED_ORIGINS, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# ---------- standaardonderwerpen ----------
DEFAULT_TOPICS: Dict[str, List[NamedDescription]] = {
    "Ondersteuning": [
        {"name": "Wie wil je bij de bevalling?", "description": "Welke personen wil je er fysiek bij hebben?"},
        {"name": "Rol van je partner of ander persoon?", "description": "Specificeer taken of wensen voor je partner."},
        {"name": "Wil je een doula / kraamzorg?", "description": "Extra ondersteuning tijdens en na de bevalling."},
        {"name": "Wat verwacht je van het personeel?", "description": "Welke stijl van begeleiding past bij jou?"}
    ],
    "Bevalling & medisch beleid": [
        {"name": "Pijnstilling",    "description": "Medicamenteuze en niet-medicamenteuze opties."},
        {"name": "Interventies",    "description": "Bijv. inknippen, kunstverlossing, infuus."},
        {"name": "Noodsituaties",   "description": "Wat als het anders loopt dan gepland?"},
        {"name": "Placenta-keuzes", "description": "Placenta bewaren, laten staan, of doneren?"}
    ],
    "Sfeer en omgeving": [
        {"name": "Muziek & verlichting", "description": "Rustige muziek? Gedimd licht?"},
        {"name": "Privacy",              "description": "Wie mag binnenkomen en fotograferen?"},
        {"name": "Foto’s / video",       "description": "Wil je opnames laten maken?"},
        {"name": "Eigen spulletjes",     "description": "Bijv. eigen kussen, etherische olie."}
    ],
    "Voeding na de geboorte": [
        {"name": "Borstvoeding",         "description": "Ondersteuning, kolven, rooming-in."},
        {"name": "Flesvoeding",          "description": "Welke melk? Wie geeft de fles?"},
        {"name": "Combinatie-voeding",   "description": "Afwisselen borst en fles."},
        {"name": "Allergieën",           "description": "Rekening houden met familiaire allergieën."}
    ]
}

# ---------- SQLite-helper functions (aangepast voor user_profile) ----------
def init_db() -> None:
    with sqlite3.connect(DB_FILE) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                user_profile TEXT
            )
        """)
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
        # Maak een kopie om de originele dict niet te wijzigen
        st_to_save = st.copy()
        user_profile = st_to_save.pop("user_profile", None)
        
        con.execute(
            "REPLACE INTO sessions (id, state, user_profile) VALUES (?, ?, ?)",
            (sid, json.dumps(st_to_save), json.dumps(user_profile))
        )
        con.commit()

# ---------- In-memory sessies + persistence ----------
SESSION: Dict[str, dict] = {}

def get_session(sid: str) -> dict:
    if sid in SESSION:
        SESSION[sid].setdefault("id", sid)
        return SESSION[sid]

    if (db_state := load_state(sid)):
        st = db_state
        st.setdefault("history", [])
        st.setdefault("generated_topic_options", {})
        st.setdefault("last_interaction_timestamp", time.time())
        st["id"] = sid
        SESSION[sid] = st
        return st

    # Nieuwe sessie
    st = {
        "id": sid,
        "stage": "choose_theme",
        "themes": [],
        "topics": {},
        "qa": [],
        "history": [],
        "summary": "",
        "ui_theme_opts": [],
        "ui_topic_opts": [],
        "current_theme": None,
        "generated_topic_options": {},
        "user_profile": None, # Nieuw veld
        "last_interaction_timestamp": time.time()
    }
    SESSION[sid] = st
    return st

def persist(sid: str) -> None:
    if sid in SESSION:
        save_state(sid, SESSION[sid])

# ---------- History-samenvatting (bij >40 messages) ----------
def summarize_chunk(chunk: List[dict]) -> str:
    if not chunk:
        return ""
    filtered = [m for m in chunk if isinstance(m, dict) and m.get('content')]
    text = "\n".join(f"{m['role']}: {m['content']}" for m in filtered)
    prompt = (
        "Vat dit deel van het gesprek over een geboorteplan samen in maximaal 300 tokens. "
        "Focus op de keuzes, wensen en open vragen."
    )
    r = client.chat.completions.create(
        model=MODEL_CHOICE,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": text}
        ],
        max_tokens=300
    )
    return r.choices[0].message.content.strip()

# ============================================================
# Tool-wrappers (inclusief nieuwe sentiment tool)
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
    all_ok = all(
        t["name"] in st["topics"] and st["topics"][t["name"]]
        for t in st["themes"]
    )
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

@function_tool
def detect_sentiment(session_id: str, message: str) -> str:
    """
    Analyseer 'message' op sentiment ('positive', 'neutral' of 'negative').
    """
    if not message.strip():
        return "neutral"
    prompt = (
        "Beoordeel de toon van het volgende bericht als 'positive', 'neutral' of 'negative'. Antwoord alleen met het woord.\n"
        f"Bericht: '{message}'"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_CHOICE,
            messages=[{"role":"system", "content": prompt}],
            max_tokens=3, # Ruimte voor de woorden
            temperature=0
        )
        sentiment = resp.choices[0].message.content.strip().lower()
        if sentiment in ['positive', 'neutral', 'negative']:
            return sentiment
    except Exception as e:
        print(f"Error in sentiment detection: {e}")
    return "neutral" # Fallback

# Tool objecten
set_theme_options = function_tool(_set_theme_options)
set_topic_options = function_tool(_set_topic_options)
register_theme    = function_tool(_register_theme)
register_topic    = function_tool(_register_topic)
complete_theme    = function_tool(_complete_theme)
log_answer        = function_tool(_log_answer)
get_state_tool    = function_tool(_get_state)
# De @function_tool decorator heeft 'detect_sentiment' al gewrapt. Hier hernoemen we het voor de duidelijkheid.
detect_sentiment_tool = detect_sentiment 

# ============================================================
# Nieuwe Nodes: Herstel, Validatie en Profiel Update
# ============================================================

def handle_session_recovery(sid: str) -> dict:
    """Herstelt een sessie als deze corrupt is."""
    st = get_session(sid)
    # Simuleer een controle op corruptie (bv. ontbrekend essentieel veld)
    if not st.get("stage"):
        # Herstel: reset naar basisinstellingen
        st = {
            "id": sid, "stage": "choose_theme", "themes": [], "topics": {},
            "qa": [], "history": [], "summary": "", "ui_theme_opts": [],
            "ui_topic_opts": [], "current_theme": None,
            "generated_topic_options": {}, "user_profile": None,
            "last_interaction_timestamp": time.time()
        }
        SESSION[sid] = st
        persist(sid)
    return st

def handle_error(st: dict, msg: str) -> Optional[str]:
    """Detecteert een fout of corrupt bericht en geeft een foutmelding."""
    try:
        if not msg.strip() and len(msg) > 0: # Check op alleen whitespace
            raise ValueError("Leeg bericht gedetecteerd")
        return None
    except Exception as e:
        print(f"Error handled: {e}")
        # Verwijder het corrupte bericht uit de geschiedenis indien nodig
        st["history"] = [entry for entry in st["history"] if entry.get("content") != msg]
        persist(st["id"])
        return "Sorry, er ging iets mis. Kun je je vraag herhalen?"

def validate_answer(st: dict, msg: str) -> Optional[str]:
    """Valideert of het antwoord in de Q&A fase niet leeg is."""
    is_qa_stage = st.get("stage") == "qa"
    last_q_asked = st.get("history") and st["history"][-1]["content"].startswith("Vraag:")
    
    if is_qa_stage and last_q_asked and not msg.strip():
        return "Ik heb je antwoord niet goed begrepen, kun je het anders verwoorden?"
    return None

def update_user_model(st: dict) -> None:
    """Werkt het gebruikersprofiel bij met thema- en topickeuzes."""
    user_profile = {
        "themes": st.get("themes", []),
        "topics": st.get("topics", {})
    }
    st["user_profile"] = user_profile
    persist(st["id"])
    print(f"User model updated for session {st['id']}")

# ============================================================
# Phase Handlers (Behavior-Tree Nodes)
# ============================================================
def handle_theme_selection(st: dict, msg: str) -> Optional[str]:
    if st["stage"] != "choose_theme":
        return None
    if not st["ui_theme_opts"]:
        _set_theme_options(st["id"], list(DEFAULT_TOPICS.keys()))
        return "Laten we beginnen met het kiezen van de thema's voor jouw geboorteplan. Welke onderwerpen spreken je aan?"
    if msg in DEFAULT_TOPICS:
        _register_theme(st["id"], msg, "") # Desc is niet meer nodig hier
        return f"Oké, thema '{msg}' is toegevoegd. Je kunt nu onderwerpen voor dit thema kiezen, of een volgend thema selecteren."
    return None

def handle_topic_selection(st: dict, msg: str) -> Optional[str]:
    if st["stage"] != "choose_topic":
        return None
    theme = st["current_theme"]
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
    if "klaar" in msg.lower() or "verder" in msg.lower() or "volgende" in msg.lower():
        if st["stage"] == "choose_topic":
             # Terug naar thema keuze
            st["stage"] = "choose_theme"
            st["ui_topic_opts"] = []
            st["current_theme"] = None
            persist(st["id"])
            return "Oké, we gaan terug naar de thema's. Kies een nieuw thema, of zeg dat je klaar bent met kiezen om naar de vragen te gaan."
        elif st["stage"] == "choose_theme":
            # Afronden van alle keuzes
             _complete_theme(st["id"])
             if st["stage"] == "qa":
                return "Top! Alle keuzes zijn gemaakt. We gaan nu dieper in op de onderwerpen met een paar vragen."
             else:
                return "Je moet voor elk gekozen thema minstens één onderwerp selecteren. Kies een thema om onderwerpen toe te voegen."
    return None

def handle_qa(st: dict, msg: str) -> Optional[str]:
    if st["stage"] != "qa":
        return None
    
    # Antwoord op vorige vraag loggen
    if st["history"] and st["history"][-1]["role"] == "assistant" and st["history"][-1]["content"].startswith("Vraag:"):
        question_text = st["history"][-1]["content"]
        # Vind de context (thema, topic) van de vraag
        # Voor nu, simpele aanname: link aan laatste onderwerp.
        # Een robuuster systeem zou de vraag-context opslaan.
        current_qa_topic = st.get("current_qa_topic", {})
        _log_answer(st["id"], current_qa_topic.get("theme"), current_qa_topic.get("question"), msg)

    # Vind de volgende onbeantwoorde vraag
    answered_qs = {(qa["theme"], qa["question"]) for qa in st.get("qa", [])}
    next_question_found = None
    for theme_info in st["themes"]:
        theme_name = theme_info["name"]
        for topic_name in st["topics"].get(theme_name, []):
            # Zoek het topic object voor de beschrijving
            topic_obj = next((t for t in DEFAULT_TOPICS.get(theme_name, []) if t["name"] == topic_name), None)
            if topic_obj and (theme_name, topic_obj["description"]) not in answered_qs:
                next_question_found = {
                    "theme": theme_name,
                    "question": topic_obj["description"] # Gebruik de beschrijving als vraag
                }
                break
        if next_question_found:
            break

    if next_question_found:
        st["current_qa_topic"] = next_question_found
        persist(st["id"])
        return f"Vraag: {next_question_found['question']} (onderwerp: {next_question_found['theme']})"
    else:
        st["stage"] = "completed"
        persist(st["id"])
        return "Je hebt alle vragen beantwoord! Je kunt je geboorteplan nu exporteren."

def handle_proactive_help(st: dict, msg: str) -> Optional[str]:
    if time.time() - st.get("last_interaction_timestamp", 0) > 300: # 5 minuten
        st["last_interaction_timestamp"] = time.time()
        return "Ik zie dat je even stilzit, kan ik ergens mee helpen?"
    return None

def handle_fallback(st: dict, msg: str) -> str:
    # Een simpele fallback die de state niet wijzigt.
    return "Ik begrijp je verzoek niet helemaal in deze context. Kun je het anders proberen? Je kunt bijvoorbeeld een thema kiezen, of 'klaar' zeggen."

# ============================================================
# Streaming-/chat-endpoint (onveranderd)
# ============================================================
def stream_run(tid: str):
    with client.beta.threads.runs.stream(thread_id=tid, assistant_id=ASSISTANT_ID) as ev:
        for e in ev:
            if e.event == "thread.message.delta" and e.data.delta.content:
                yield e.data.delta.content[0].text.value

@app.post("/chat")
def chat():
    origin = request.headers.get("Origin")
    if origin and origin not in ALLOWED_ORIGINS:
        abort(403)
    data = request.get_json(force=True)
    msg = data.get("message", "")
    tid = data.get("thread_id") or client.beta.threads.create().id
    client.beta.threads.messages.create(thread_id=tid, role="user", content=msg)
    return Response(stream_run(tid), headers={"X-Thread-ID": tid}, mimetype="text/plain")

# ============================================================
# Synchronous /agent-endpoint (volledig vernieuwd)
# ============================================================
@app.post("/agent")
def agent():
    origin = request.headers.get("Origin")
    if origin and origin not in ALLOWED_ORIGINS:
        abort(403)

    body = request.get_json(force=True)
    msg  = body.get("message", "")
    sid  = body.get("session_id") or str(uuid.uuid4())

    # 1. Sessierecovery controleren
    st = handle_session_recovery(sid)
    st["last_interaction_timestamp"] = time.time()
    st["history"].append({"role": "user", "content": msg})

    # Samenvatting bij lange geschiedenis
    if len(st["history"]) > 40:
        st["summary"] += "\n" + summarize_chunk(st["history"][:-20])
        st["history"] = st["history"][-20:]

    # 2. Sentiment pre-check
    sentiment = detect_sentiment_tool(st["id"], msg)
    if sentiment == "negative":
        reply = "Ik merk dat dit misschien een gevoelig of frustrerend punt voor je is. Laten we hier even bij stilstaan. Kan ik iets voor je verduidelijken of op een andere manier helpen?"
        st["history"].append({"role": "assistant", "content": reply})
        persist(sid)
        return jsonify({
            "assistant_reply": reply, "session_id": sid,
            **{k: v for k, v in st.items() if k not in ("ui_theme_opts", "ui_topic_opts")}
        })

    # 3. Error-handling check
    error_response = handle_error(st, msg)
    if error_response:
        st["history"].append({"role": "assistant", "content": error_response})
        persist(sid) # Persist is al in handle_error, maar voor de zekerheid
        return jsonify({
            "assistant_reply": error_response, "session_id": sid,
            **{k: v for k, v in st.items() if k not in ("ui_theme_opts", "ui_topic_opts")}
        })

    # 4. Validatie van Q&A antwoord
    validation_response = validate_answer(st, msg)
    if validation_response:
        st["history"].append({"role": "assistant", "content": validation_response})
        persist(sid)
        return jsonify({
            "assistant_reply": validation_response, "session_id": sid,
            **{k: v for k, v in st.items() if k not in (