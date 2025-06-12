#!/usr/bin/env python3
# app.py – Geboorteplan-agent (volledige versie met uitgebreide DEBUG-logs)

from __future__ import annotations
import os, json, uuid, sqlite3, time, re, logging
from typing import List, Dict, Optional
from typing_extensions import TypedDict
from flask import (
    Flask, request, jsonify, abort, send_file,
    send_from_directory, render_template
)
from flask_cors import CORS
from openai import OpenAI
from agents import function_tool

# ────────────────────────────────── logging ───────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("beval-agent")

# ────────────────────────────────── Config ────────────────────────────────────
client          = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID    = os.getenv("ASSISTANT_ID")
ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
    "https://chatbotbvmp.onrender.com"
]
DB_FILE       = "sessions.db"
MODEL_CHOICE  = "gpt-4.1"

app = Flask(__name__, static_folder="static",
            template_folder="templates", static_url_path="")
CORS(app, origins=ALLOWED_ORIGINS,
     allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# ───────────────────────────── Typing helpers ────────────────────────────────
class NamedDescription(TypedDict):
    name: str
    description: str

# ───────────────────────────── Standaard onderwerpen ─────────────────────────
DEFAULT_TOPICS: Dict[str, List[NamedDescription]] = {
    "Ondersteuning": [
        {"name": "Wie wil je bij de bevalling?",         "description": "Welke personen wil je er fysiek bij hebben?"},
        {"name": "Rol van je partner of ander persoon?",  "description": "Specificeer taken of wensen voor je partner."},
        {"name": "Wil je een doula / kraamzorg?",         "description": "Extra ondersteuning tijdens en na de bevalling."},
        {"name": "Wat verwacht je van het personeel?",    "description": "Welke stijl van begeleiding past bij jou?"}
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
        {"name": "Borstvoeding",       "description": "Ondersteuning, kolven, rooming-in."},
        {"name": "Flesvoeding",        "description": "Welke melk? Wie geeft de fles?"},
        {"name": "Combinatie-voeding", "description": "Afwisselen borst en fles."},
        {"name": "Allergieën",         "description": "Rekening houden met familiaire allergieën."}
    ]
}

# ───────────────────────────── SQLite helper ────────────────────────────────
def init_db() -> None:
    with sqlite3.connect(DB_FILE) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS sessions "
            "(id TEXT PRIMARY KEY, state TEXT NOT NULL, user_profile TEXT)"
        )
    log.debug("SQLite-db klaar: %s", DB_FILE)

init_db()

def load_state(sid: str) -> Optional[dict]:
    with sqlite3.connect(DB_FILE) as con:
        row = con.execute(
            "SELECT state, user_profile FROM sessions WHERE id=?", (sid,)
        ).fetchone()
        if not row:
            log.debug("DB: geen record voor sessie %s", sid)
            return None
        state_json, profile_json = row
        state = json.loads(state_json)
        if profile_json:
            state["user_profile"] = json.loads(profile_json)
        log.debug("DB: sessie %s geladen", sid)
        return state

def save_state(sid: str, st: dict) -> None:
    with sqlite3.connect(DB_FILE) as con:
        st_copy = st.copy()
        profile = st_copy.pop("user_profile", None)
        con.execute(
            "REPLACE INTO sessions (id,state,user_profile) VALUES (?,?,?)",
            (sid, json.dumps(st_copy), json.dumps(profile))
        )
        con.commit()
    log.debug("DB: sessie %s opgeslagen", sid)

# ───────────────────────────── Session management ───────────────────────────
SESSION: Dict[str, dict] = {}

def get_session(sid: str) -> dict:
    if sid in SESSION:
        return SESSION[sid]
    db = load_state(sid)
    if db:
        db.setdefault("id", sid)
        db.setdefault("history", [])
        db.setdefault("last_interaction_timestamp", time.time())
        SESSION[sid] = db
        return db
    st = {
        "id": sid, "stage": "choose_theme",
        "themes": [], "topics": {}, "qa": [], "history": [],
        "summary": "", "ui_theme_opts": [], "ui_topic_opts": [],
        "current_theme": None, "generated_topic_options": {},
        "user_profile": None, "last_interaction_timestamp": time.time()
    }
    SESSION[sid] = st
    log.debug("MEM: nieuwe sessie %s", sid)
    return st

def persist(sid: str) -> None:
    if sid in SESSION:
        save_state(sid, SESSION[sid])

# ───────────────────────────── Samenvatter ──────────────────────────────────
def summarize_chunk(chunk: List[dict]) -> str:
    if not chunk:
        return ""
    text = "\n".join(f"{m['role']}: {m['content']}" for m in chunk if m.get("content"))
    prompt = ("Vat dit deel van het gesprek over een geboorteplan samen "
              "in maximaal 300 tokens. Focus op de keuzes, wensen en open vragen.")
    resp = client.chat.completions.create(
        model=MODEL_CHOICE,
        messages=[{"role": "system", "content": prompt},
                  {"role": "user",   "content": text}],
        max_tokens=300
    )
    summary = resp.choices[0].message.content.strip()
    log.debug("Samenvatting (%d regels) gemaakt", len(summary.split()))
    return summary

# ───────────────────────────── Tool-wrappers ────────────────────────────────
def _set_theme_options(sid: str, opts: List[str]) -> str:
    st = get_session(sid)
    st["ui_theme_opts"] = opts
    persist(sid)
    log.debug("Tool: ui_theme_opts voor %s → %s", sid, opts)
    return "ok"

def _set_topic_options(sid: str, theme: str, opts: List[NamedDescription]) -> str:
    st = get_session(sid)
    st.setdefault("generated_topic_options", {})[theme] = opts
    st["ui_topic_opts"] = opts
    st["current_theme"] = theme
    persist(sid)
    log.debug("Tool: ui_topic_opts voor %s (%s)", sid, theme)
    return "ok"

def _register_theme(sid: str, theme: str, desc: str = "") -> str:
    st = get_session(sid)
    if len(st["themes"]) < 6 and theme not in [t["name"] for t in st["themes"]]:
        st["themes"].append({"name": theme, "description": desc})
        st["stage"] = "choose_topic"
        st["ui_topic_opts"] = DEFAULT_TOPICS.get(theme, [])
        st["current_theme"] = theme
        persist(sid)
        log.debug("Tool: thema '%s' toegevoegd (%s)", theme, sid)
    return "ok"

def _register_topic(sid: str, theme: str, topic: str) -> str:
    st = get_session(sid)
    topics = st["topics"].setdefault(theme, [])
    if len(topics) < 4 and topic not in topics:
        topics.append(topic)
        persist(sid)
        log.debug("Tool: topic '%s'→'%s' toegevoegd", theme, topic)
    return "ok"

def _complete_theme(sid: str) -> str:
    st = get_session(sid)
    all_ok = all(t["name"] in st["topics"] and st["topics"][t["name"]] for t in st["themes"])
    st["stage"] = "qa" if all_ok else "choose_theme"
    st["ui_topic_opts"] = []
    st["current_theme"] = None
    persist(sid)
    log.debug("Tool: complete_theme (%s) → stage=%s", sid, st["stage"])
    return "ok"

def _log_answer(sid: str, theme: str, q: str, a: str) -> str:
    st = get_session(sid)
    for qa in st["qa"]:
        if qa["theme"] == theme and qa["question"] == q:
            qa["answer"] = a
            break
    else:
        st["qa"].append({"theme": theme, "question": q, "answer": a})
    persist(sid)
    log.debug("Tool: answer opgeslagen (%s/%s) voor %s", theme, q, sid)
    return "ok"

# Decorators (function calling)
set_theme_options  = function_tool(_set_theme_options)
set_topic_options  = function_tool(_set_topic_options)
register_theme     = function_tool(_register_theme)
register_topic     = function_tool(_register_topic)
complete_theme     = function_tool(_complete_theme)
log_answer         = function_tool(_log_answer)

# ───────────────────────────── Utils: sentiment & validaties ───────────────
def detect_sentiment(sid: str, text: str) -> str:
    if not text.strip():
        return "neutral"
    prompt = ("Beoordeel de toon van het volgende bericht als 'positive', 'neutral' of 'negative'. "
              "Antwoord alleen met het woord.\nBericht: '%s'" % text.replace("'", ""))
    try:
        resp = client.chat.completions.create(
            model=MODEL_CHOICE,
            messages=[{"role":"system", "content": prompt}],
            max_tokens=5, temperature=0
        )
        label = resp.choices[0].message.content.strip().lower()
        return label if label in ("positive", "neutral", "negative") else "neutral"
    except Exception as e:
        log.warning("Sentiment fail: %s", e)
        return "neutral"

def handle_error(st: dict, msg: str) -> Optional[str]:
    if not msg.strip() and len(msg) > 0:
        return "Sorry, er ging iets mis. Kun je je vraag herhalen?"
    return None

def validate_answer(st: dict, msg: str) -> Optional[str]:
    if (st.get("stage") == "qa"
            and st["history"][-1]["content"].startswith("Vraag:")
            and not msg.strip()):
        return "Ik heb je antwoord niet goed begrepen, kun je het anders verwoorden?"
    return None

def update_user_model(st: dict) -> None:
    st["user_profile"] = {"themes": st["themes"], "topics": st["topics"]}
    persist(st["id"])

# ───────────────────────────── Handlers per fase ────────────────────────────
def handle_theme_selection(st: dict, txt: str) -> Optional[str]:
    if st["stage"] != "choose_theme":
        return None
    if not st["ui_theme_opts"]:
        _set_theme_options(st["id"], list(DEFAULT_TOPICS.keys()))
        return ("We beginnen met het kiezen van thema's. "
                "Welke spreken je aan?")
    chosen = next((t for t in DEFAULT_TOPICS if t.lower() in txt.lower()), None)
    if not chosen:
        return None
    if any(t["name"] == chosen for t in st["themes"]):
        st["stage"] = "choose_topic"
        st["current_theme"] = chosen
        st["ui_topic_opts"] = (
            st.get("generated_topic_options", {}).get(chosen)
            or DEFAULT_TOPICS[chosen]
        )
        persist(st["id"])
        return (f"We passen '{chosen}' aan. Kies je onderwerpen "
                "(of verwijder bestaande).")
    if len(st["themes"]) >= 6:
        return ("Je hebt al 6 thema's. Verwijder eerst een thema "
                "voor je er één toevoegt.")
    _register_theme(st["id"], chosen)
    return (f"Thema '{chosen}' toegevoegd. Kies nu max. 4 onderwerpen.")

def handle_topic_selection(st: dict, txt: str) -> Optional[str]:
    if st["stage"] != "choose_topic":
        return None
    theme = st.get("current_theme")
    if not theme:
        st["stage"] = "choose_theme"
        return "Kies eerst een thema."
    if not st["ui_topic_opts"]:
        _set_topic_options(
            st["id"], theme,
            st.get("generated_topic_options", {}).get(theme)
            or DEFAULT_TOPICS[theme]
        )
        return f"Kies de onderwerpen voor '{theme}'."
    names = [o["name"] for o in st["ui_topic_opts"]]
    if txt in names:
        cur = st["topics"].get(theme, [])
        if txt in cur:
            return f"'{txt}' staat al in je lijst."
        if len(cur) >= 4:
            return "Je hebt al 4 onderwerpen voor dit thema."
        _register_topic(st["id"], theme, txt)
        left = 4 - len(st["topics"][theme])
        return (f"'{txt}' toegevoegd. "
                f"Je kunt nog {left} onderwerp{'en' if left!=1 else ''} kiezen "
                "of 'klaar' zeggen.")
    return None

def handle_complete_selection(st: dict, txt: str) -> Optional[str]:
    if txt.lower() not in ("klaar", "verder", "volgende"):
        return None
    if st["stage"] == "choose_topic":
        st["stage"] = "choose_theme"
        st["current_theme"] = None
        st["ui_topic_opts"] = []
        persist(st["id"])
        return ("Prima, terug naar de themalijst. Kies een nieuw thema "
                "of zeg 'klaar' om door te gaan.")
    if st["stage"] == "choose_theme":
        _complete_theme(st["id"])
        if st["stage"] == "qa":
            return handle_qa(st, "")
        return ("Sommige thema's missen nog onderwerpen. "
                "Maak die eerst af.")
    return None

def handle_qa(st: dict, txt: str) -> Optional[str]:
    if st["stage"] != "qa":
        return None
    # evt. antwoord opslaan
    if (txt.strip() and st["history"]
            and st["history"][-1]["role"] == "assistant"
            and st["history"][-1]["content"].startswith("Vraag:")):
        qa_t = st.get("current_qa_topic")
        if qa_t:
            _log_answer(st["id"], qa_t["theme"], qa_t["question"], txt)
    answered = {(q["theme"], q["question"]) for q in st["qa"]}
    nxt = None
    for th in st["themes"]:
        th_name = th["name"]
        for tp in st["topics"].get(th_name, []):
            desc = next(d for d in DEFAULT_TOPICS[th_name] if d["name"] == tp)["description"]
            if (th_name, desc) not in answered:
                nxt = {"theme": th_name, "question": desc}
                break
        if nxt: break
    if nxt:
        st["current_qa_topic"] = nxt
        persist(st["id"])
        return f"Vraag: {nxt['question']} (onderwerp: {nxt['theme']})"
    st["stage"] = "completed"
    update_user_model(st)
    persist(st["id"])
    return "Alle vragen zijn beantwoord! Je kunt nu je geboorteplan exporteren."

def handle_proactive_help(st: dict, _txt: str) -> Optional[str]:
    if time.time() - st["last_interaction_timestamp"] > 300:
        st["last_interaction_timestamp"] = time.time()
        return "Ik zie dat je even pauzeert. Kan ik je ergens mee helpen?"
    return None

# Verwijder‐handlers (via tekst):
def handle_topic_removal(st: dict, txt: str) -> Optional[str]:
    m = re.match(r"(?i)verwijder\s+onderwerp\s+(.+?)\s+uit\s+thema\s+(.+)", txt)
    if not m:
        return None
    topic, theme = m.group(1).strip("\"' "), m.group(2).strip("\"' ")
    if theme not in st["topics"] or topic not in st["topics"][theme]:
        return None
    st["topics"][theme].remove(topic)
    st["stage"] = "choose_topic"
    st["current_theme"] = theme
    st["ui_topic_opts"] = DEFAULT_TOPICS[theme]
    persist(st["id"])
    return f"Onderwerp '{topic}' verwijderd uit '{theme}'."

def handle_theme_removal(st: dict, txt: str) -> Optional[str]:
    m = re.match(r"(?i)verwijder\s+thema\s+(.+)", txt)
    if not m:
        return None
    theme = m.group(1).strip("\"' ")
    if not any(t["name"].lower() == theme.lower() for t in st["themes"]):
        return None
    st["themes"] = [t for t in st["themes"] if t["name"].lower() != theme.lower()]
    st["topics"].pop(theme, None)
    if st["current_theme"] == theme:
        st["stage"] = "choose_theme"
        st["current_theme"] = None
        st["ui_topic_opts"] = []
    persist(st["id"])
    return f"Thema '{theme}' is verwijderd."

def handle_fallback(st: dict, _txt: str) -> str:
    return ("Ik begrijp je verzoek niet helemaal. "
            "Kun je het anders formuleren of een keuze maken op het scherm?")

# ───────────────────────────── /agent endpoint ─────────────────────────────
@app.post("/agent")
def agent():
    origin = request.headers.get("Origin")
    if origin and origin not in ALLOWED_ORIGINS:
        abort(403)

    body = request.get_json(force=True)
    msg  = body.get("message", "") or ""
    sid  = body.get("session_id") or str(uuid.uuid4())

    st = get_session(sid)
    st["last_interaction_timestamp"] = time.time()
    st["history"].append({"role": "user", "content": msg})
    log.debug("IN  %s | %s | stage=%s", sid[-6:], msg, st["stage"])

    # lange histories samenvatten
    if len(st["history"]) > 40:
        st["summary"] += "\n" + summarize_chunk(st["history"][:-20])
        st["history"] = st["history"][-20:]

    # snelle validations
    for fn in (handle_error, validate_answer):
        out = fn(st, msg)
        if out:
            reply = out
            break
    else:
        # fasehandlers
        reply = None
        for fn in (handle_topic_removal, handle_theme_removal,
                   handle_theme_selection, handle_topic_selection,
                   handle_complete_selection, handle_qa, handle_proactive_help):
            reply = fn(st, msg)
            if reply:
                break
        if reply is None:
            reply = handle_fallback(st, msg)

    st["history"].append({"role": "assistant", "content": reply})
    persist(sid)
    log.debug("OUT %s | %s | new-stage=%s", sid[-6:], reply[:60], st["stage"])

    return jsonify(build_payload(st, reply))

# payload helper
def build_payload(st: dict, reply: str) -> dict:
    return {
        "assistant_reply": reply,
        "session_id": st["id"],
        "options": st["ui_topic_opts"] if st["stage"] == "choose_topic" else st["ui_theme_opts"],
        "current_theme": st.get("current_theme"),
        "themes": st["themes"],
        "topics": st["topics"],
        "qa": st["qa"],
        "stage": st["stage"]
    }

# ───────────────────────────── Export & static ─────────────────────────────
@app.get("/export/<sid>")
def export_json(sid: str):
    st = load_state(sid)
    if not st:
        abort(404)
    path = os.path.join("/tmp", f"geboorteplan_{sid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path == "iframe":
        return render_template("iframe_page.html",
                               backend_url=os.getenv("RENDER_EXTERNAL_URL", "http://127.0.0.1:10000"))
    full = os.path.join(app.static_folder, path)
    if path and os.path.exists(full):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

# ───────────────────────────── main ─────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
