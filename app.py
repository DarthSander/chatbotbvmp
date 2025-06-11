from __future__ import annotations
import os, json, uuid, sqlite3, time, re
from copy import deepcopy
from typing import List, Dict, Optional
from typing_extensions import TypedDict

from flask import Flask, request, Response, jsonify, abort, send_file, send_from_directory, render_template
from flask_cors import CORS
from openai import OpenAI
from agents import function_tool

# Basisconfiguratie
client        = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID  = os.getenv("ASSISTANT_ID")
ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
    "https://chatbotbvmp.onrender.com"
]
DB_FILE       = "sessions.db"
MODEL_CHOICE  = "gpt-4.1"

# Flask-app setup
app = Flask(__name__, static_folder="static", template_folder="templates", static_url_path="")
CORS(app, origins=ALLOWED_ORIGINS, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# Strikte types
class NamedDescription(TypedDict):
    name: str
    description: str

# Standaardonderwerpen per thema (4 onderwerpen per thema maximaal)
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

# SQLite-helper functies
def init_db() -> None:
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
        con.execute("REPLACE INTO sessions (id, state, user_profile) VALUES (?, ?, ?)",
                    (sid, json.dumps(st_to_save), json.dumps(user_profile)))
        con.commit()

# In-memory sessions + persist
SESSION: Dict[str, dict] = {}

def get_session(sid: str) -> dict:
    if sid in SESSION:
        return SESSION[sid]
    db_state = load_state(sid)
    if db_state:
        st = db_state
        st.setdefault("id", sid)
        st.setdefault("history", [])
        st.setdefault("last_interaction_timestamp", time.time())
        SESSION[sid] = st
        return st
    # Nieuwesessie initialisatie
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

# Samenvatting van chatgeschiedenis (voor lange sessies, GPT)
def summarize_chunk(chunk: List[dict]) -> str:
    if not chunk:
        return ""
    filtered = [m for m in chunk if isinstance(m, dict) and m.get('content')]
    text = "\n".join(f"{m['role']}: {m['content']}" for m in filtered)
    prompt = "Vat dit deel van het gesprek over een geboorteplan samen in maximaal 300 tokens. Focus op de keuzes, wensen en open vragen."
    resp = client.chat.completions.create(model=MODEL_CHOICE, messages=[{"role":"system", "content": prompt}, {"role":"user", "content": text}], max_tokens=300)
    return resp.choices[0].message.content.strip()

# Tool-wrappers koppelen aan functies (voor agent-function calling indien gebruikt)
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
    # Controleer of elk gekozen thema minstens één onderwerp heeft
    all_ok = all(t["name"] in st["topics"] and st["topics"][t["name"]] for t in st["themes"])
    st["stage"] = "qa" if all_ok else "choose_theme"
    st["ui_topic_opts"] = []
    st["current_theme"] = None
    persist(session_id)
    return "ok"

def _log_answer(session_id: str, theme: str, question: str, answer: str) -> str:
    st = get_session(session_id)
    found = False
    for qa in st.get("qa", []):
        if qa["theme"] == theme and qa["question"] == question:
            qa["answer"] = answer
            found = True
            break
    if not found:
        st["qa"].append({"theme": theme, "question": question, "answer": answer})
    persist(session_id)
    return "ok"

def _get_state(session_id: str) -> str:
    # Voor eventueel opvragen van complete sessiestatus (JSON)
    return json.dumps(get_session(session_id))

def detect_sentiment(session_id: str, message: str) -> str:
    # Simpele sentimentanalyse via OpenAI
    if not message.strip():
        return "neutral"
    prompt = f"Beoordeel de toon van het volgende bericht als 'positive', 'neutral' of 'negative'. Antwoord alleen met het woord.\nBericht: '{message}'"
    try:
        if not client.api_key:
            # Geen API key beschikbaar
            return "neutral"
        resp = client.chat.completions.create(model=MODEL_CHOICE, messages=[{"role":"system", "content": prompt}], max_tokens=5, temperature=0)
        sentiment = resp.choices[0].message.content.strip().lower()
        if sentiment in ['positive', 'neutral', 'negative']:
            return sentiment
        return "neutral"
    except Exception:
        return "neutral"

# Tools beschikbaar maken voor agent (indien LLM function calling wordt gebruikt)
set_theme_options       = function_tool(_set_theme_options)
set_topic_options       = function_tool(_set_topic_options)
register_theme          = function_tool(_register_theme)
register_topic          = function_tool(_register_topic)
complete_theme          = function_tool(_complete_theme)
log_answer              = function_tool(_log_answer)
get_state_tool          = function_tool(_get_state)
detect_sentiment_tool   = function_tool(detect_sentiment)

# Nieuwe functies: sessie recovery & eenvoudige validatie
def handle_session_recovery(sid: str) -> dict:
    return get_session(sid)

def handle_error(st: dict, msg: str) -> Optional[str]:
    # Bijvoorbeeld: een technisch probleem of leeg bericht
    if not msg.strip() and len(msg) > 0:
        return "Sorry, er ging iets mis. Kun je je vraag herhalen?"
    return None

def validate_answer(st: dict, msg: str) -> Optional[str]:
    # Controleer of in Q&A-fase een antwoord leeg is gebleven
    is_qa_stage = st.get("stage") == "qa"
    last_q_asked = st.get("history") and st["history"][-1]["content"].startswith("Vraag:")
    if is_qa_stage and last_q_asked and not msg.strip():
        return "Ik heb je antwoord niet goed begrepen, kun je het anders verwoorden?"
    return None

def update_user_model(st: dict) -> None:
    # Werk optioneel een gebruikersprofiel bij met gemaakte keuzes
    st["user_profile"] = {"themes": st.get("themes", []), "topics": st.get("topics", {})}
    persist(st["id"])

def handle_theme_selection(st: dict, msg: str) -> Optional[str]:
    """
    Fase-handler voor het kiezen óf bewerken van thema’s.

    - Init: geeft de lijst thema-opties terug.
    - Nieuw thema: registreert als er minder dan 6 zijn.
    - Bestaand thema: zet fase op choose_topic en vult ui_topic_opts,
      zodat de frontend meteen weet welke onderwerpen er zijn.
    - Limiet bereikt: geeft uitleg.
    """
    if st["stage"] != "choose_theme":
        return None

    # ── Eerste keer: stuur lijst met thema’s ───────────────────────
    if not st.get("ui_theme_opts"):
        _set_theme_options(st["id"], list(DEFAULT_TOPICS.keys()))
        return (
            "Laten we beginnen met het kiezen van de thema's voor jouw geboorteplan. "
            "Welke onderwerpen spreken je aan?"
        )

    # ── Kijk of de gebruiker een beschikbaar thema noemt ───────────
    chosen_theme = next(
        (t for t in DEFAULT_TOPICS if t.lower() in msg.lower()), None
    )
    if not chosen_theme:
        return None

    # ── Thema is al gekozen → open opnieuw om te bewerken ──────────
    if any(t["name"] == chosen_theme for t in st["themes"]):
        st["stage"] = "choose_topic"
        st["current_theme"] = chosen_theme
        options = (
            st.get("generated_topic_options", {}).get(chosen_theme)
            or DEFAULT_TOPICS.get(chosen_theme, [])
        )
        st["ui_topic_opts"] = options      # <-- belangrijke toevoeging
        persist(st["id"])
        return (
            f"We gaan de onderwerpen voor '{chosen_theme}' aanpassen. "
            f"Kies of wijzig de onderwerpen voor dit thema."
        )

    # ── Nieuw thema mits limiet (6) niet bereikt ──────────────────
    if len(st["themes"]) >= 6:
        return (
            "Je hebt al 6 thema’s gekozen; je kunt geen nieuwe thema’s toevoegen "
            "tenzij je eerst een ander thema verwijdert."
        )

    _register_theme(st["id"], chosen_theme)
    return (
        f"Oké, thema '{chosen_theme}' is toegevoegd. "
        f"Je kunt nu onderwerpen voor dit thema kiezen, of een volgend thema selecteren."
    )


def handle_topic_selection(st: dict, msg: str) -> Optional[str]:
    if st["stage"] != "choose_topic":
        return None
    theme = st.get("current_theme")
    if not theme:
        # Fout in volgorde – terug naar themakeuze
        st["stage"] = "choose_theme"
        return "Er ging iets mis. Kies eerst een thema om onderwerpen aan toe te voegen."
    if not st.get("ui_topic_opts"):
        # Stel de onderwerpenkeuzes in (standaard of gegenereerd)
        options = st.get("generated_topic_options", {}).get(theme) or DEFAULT_TOPICS.get(theme, [])
        _set_topic_options(st["id"], theme, options)
        return f"Kies nu de onderwerpen die je wilt bespreken voor het thema '{theme}'."
    # Verwerkt een onderwerpkeuze (msg is exact de naam van een onderwerpoptie)
    names = [t["name"] for t in st.get("ui_topic_opts", [])]
    if msg in names:
        # Controleer huidige selectie voor dit thema
        current_topics = st["topics"].get(theme, [])
        if msg in current_topics:
            return f"Onderwerp '{msg}' heb je al gekozen voor thema '{theme}'."
        if len(current_topics) >= 4:
            return f"Je hebt al 4 onderwerpen gekozen voor thema '{theme}'. Verwijder er eerst één om een nieuw onderwerp toe te voegen."
        # Voeg het onderwerp toe
        _register_topic(st["id"], theme, msg)
        current_topics = st["topics"].get(theme, [])  # update lijst na toevoeging
        if len(current_topics) == 4:
            return (f"Onderwerp '{msg}' toegevoegd aan thema '{theme}'. "
                    f"Je hebt nu 4 onderwerpen gekozen voor dit thema, dat is het maximum. "
                    f"Als je een ander onderwerp wilt toevoegen, zul je eerst een onderwerp moeten verwijderen.")
        else:
            return f"Onderwerp '{msg}' toegevoegd aan thema '{theme}'. Je kunt meer onderwerpen kiezen of aangeven dat je klaar bent met dit thema."
    return None

def handle_complete_selection(st: dict, msg: str) -> Optional[str]:
    low_msg = msg.lower()
    if any(x in low_msg for x in ["klaar", "verder", "volgende"]):
        if st["stage"] == "choose_topic":
            # Gebruiker is klaar met huidig thema
            st["stage"] = "choose_theme"
            st["ui_topic_opts"] = []
            st["current_theme"] = None
            persist(st["id"])
            return "Oké, we gaan terug naar de thema's. Kies een nieuw thema, of zeg dat je klaar bent met alle keuzes om naar de vragen te gaan."
        elif st["stage"] == "choose_theme":
            _complete_theme(st["id"])
            if st["stage"] == "qa":
                # Alle thema's hebben minstens één onderwerp, start Q&A
                return handle_qa(st, "")
            else:
                # Nog niet alle gekozen thema's hebben onderwerpen
                return "Je moet voor elk gekozen thema minstens één onderwerp selecteren. Kies een thema om onderwerpen toe te voegen."
    return None

def handle_qa(st: dict, msg: str) -> Optional[str]:
    if st["stage"] != "qa":
        return None
    # Als de laatste agent-uiting een vraag was en de gebruiker nu antwoordt:
    if msg.strip() and st["history"] and st["history"][-1]["role"] == "assistant" and st["history"][-1]["content"].startswith("Vraag:"):
        current_qa = st.get("current_qa_topic", {})
        if current_qa:
            _log_answer(st["id"], current_qa.get("theme"), current_qa.get("question"), msg)
    # Zoek de volgende onbeantwoorde vraag op basis van gekozen onderwerpen
    answered_qs = {(qa["theme"], qa["question"]) for qa in st.get("qa", [])}
    next_q = None
    for theme_info in st.get("themes", []):
        theme_name = theme_info["name"]
        for topic_name in st.get("topics", {}).get(theme_name, []):
            # Zoek de vraag (description) horend bij dit topic
            topic_obj = next((t for t in DEFAULT_TOPICS.get(theme_name, []) if t["name"] == topic_name), None)
            if topic_obj and (theme_name, topic_obj["description"]) not in answered_qs:
                next_q = {"theme": theme_name, "question": topic_obj["description"]}
                break
        if next_q:
            break
    if next_q:
        st["current_qa_topic"] = next_q
        persist(st["id"])
        # Vraag stellen incl. het thema als referentie
        return f"Vraag: {next_q['question']} (onderwerp: {next_q['theme']})"
    else:
        # Alle vragen beantwoord, markeer sessie als afgerond
        st["stage"] = "completed"
        update_user_model(st)
        persist(st["id"])
        return "Je hebt alle vragen beantwoord! Je kunt je geboorteplan nu exporteren."

def handle_proactive_help(st: dict, msg: str) -> Optional[str]:
    # Bied hulp aan als de gebruiker lang niets heeft gedaan
    if time.time() - st.get("last_interaction_timestamp", 0) > 300:
        st["last_interaction_timestamp"] = time.time()
        return "Ik zie dat je even pauzeert. Kan ik je ergens mee helpen?"
    return None

# Nieuw: Handlers voor verwijder-commando's via de chat
def handle_topic_removal(st: dict, msg: str) -> Optional[str]:
    # Herken "verwijder onderwerp X uit thema Y"
    match = re.match(r"(?i)verwijder\s+onderwerp\s+(.+?)\s+uit\s+thema\s+(.+)", msg.strip())
    if not match:
        return None
    topic = match.group(1).strip().strip("'\"")
    theme = match.group(2).strip().strip("'\"")
    if theme in [t["name"] for t in st.get("themes", [])]:
        # Thema is gekozen; voer verwijdering uit als onderwerp bestaat
        if topic in st.get("topics", {}).get(theme, []):
            st["topics"][theme].remove(topic)
            # Als huidig bewerkte thema None is (we waren in choose_theme), zet naar het theme om te bewerken
            if st["stage"] == "choose_theme" or st.get("current_theme") != theme:
                st["stage"] = "choose_topic"
                st["current_theme"] = theme
                options = st.get("generated_topic_options", {}).get(theme) or DEFAULT_TOPICS.get(theme, [])
                st["ui_topic_opts"] = options
            persist(st["id"])
            return f"Onderwerp '{topic}' is verwijderd uit thema '{theme}'. Je kunt nu eventueel een ander onderwerp voor dit thema kiezen."
        else:
            return f"Ik kon het onderwerp '{topic}' niet vinden in thema '{theme}'."
    return f"Thema '{theme}' maakt geen deel uit van je selectie."

def handle_theme_removal(st: dict, msg: str) -> Optional[str]:
    # Herken "verwijder thema X"
    match = re.match(r"(?i)verwijder\s+thema\s+(.+)", msg.strip())
    if not match:
        return None
    theme = match.group(1).strip().strip("'\"")
    # Zoek het thema object in de lijst
    themes_list = st.get("themes", [])
    theme_obj = next((t for t in themes_list if t["name"].lower() == theme.lower()), None)
    if theme_obj:
        # Verwijder het thema en bijbehorende onderwerpen
        themes_list.remove(theme_obj)
        st.get("topics", {}).pop(theme_obj["name"], None)
        if st.get("current_theme") == theme_obj["name"]:
            # Als we midden in dit thema zaten, ga terug naar themakeuze
            st["current_theme"] = None
            if st["stage"] == "choose_topic":
                st["stage"] = "choose_theme"
            st["ui_topic_opts"] = []
        persist(st["id"])
        return f"Thema '{theme_obj['name']}' is verwijderd. Je kunt eventueel een ander thema toevoegen ter vervanging."
    return f"Thema '{theme}' zit niet in je huidige keuzes."

# Hoofdlogica: agent endpoint
@app.post("/agent")
def agent_endpoint():
    origin = request.headers.get("Origin")
    if origin and origin not in ALLOWED_ORIGINS:
        abort(403)
    body = request.get_json(force=True)
    msg = body.get("message", "") or ""
    sid = body.get("session_id") or str(uuid.uuid4())
    st = handle_session_recovery(sid)
    # Tijdstempel laatste interactie bijwerken
    st["last_interaction_timestamp"] = time.time()
    st["history"].append({"role": "user", "content": msg})
    # Allereerst: eenvoudige checks (sentiment, errors)
    sentiment = detect_sentiment(st["id"], msg)
    if sentiment == "negative":
        reply = "Ik merk dat dit een gevoelig onderwerp is. Kan ik iets voor je verduidelijken of anders helpen?"
        st["history"].append({"role": "assistant", "content": reply})
        persist(sid)
        return jsonify(_build_response_payload(st, reply))
    for check in (handle_error, validate_answer):
        response = check(st, msg)
        if response:
            st["history"].append({"role": "assistant", "content": response})
            persist(sid)
            return jsonify(_build_response_payload(st, response))
    # Schakel door naar specifieke fasehandlers
    reply = None
    for handler in (handle_topic_removal, handle_theme_removal, handle_theme_selection,
                    handle_topic_selection, handle_complete_selection, handle_qa, handle_proactive_help):
        reply = handler(st, msg)
        if reply is not None:
            break
    if reply is None:
        # Geen specifieke handler kon het aan – generieke fallback
        reply = "Ik begrijp je verzoek niet helemaal. Kun je het anders verwoorden of een keuze maken uit de opties?"
    # Sla agentantwoord op en geef JSON-response
    st["history"].append({"role": "assistant", "content": reply})
    persist(sid)
    return jsonify(_build_response_payload(st, reply))

def _build_response_payload(st: dict, assistant_reply: str) -> dict:
    """Hulpfunctie om de response payload te vormen op basis van de sessiestatus."""
    data = {
        "assistant_reply": assistant_reply,
        "session_id": st["id"],
        "options": st.get("ui_topic_opts") if st.get("stage") == "choose_topic" else st.get("ui_theme_opts"),
        "current_theme": st.get("current_theme"),
    }
    # Voeg selectie en QA-log toe voor front-end (exclusief UI-only keys)
    filtered_state = {k: v for k, v in st.items() if k not in ("ui_theme_opts", "ui_topic_opts")}
    data.update(filtered_state)
    return data

# Overige endpoints (exporteren en statische frontend)
@app.get("/export/<sid>")
def export_json(sid: str):
    st = load_state(sid)
    if not st:
        abort(404)
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"geboorteplan_{sid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path == "iframe":
        return render_template('iframe_page.html', backend_url=os.getenv("RENDER_EXTERNAL_URL", "http://127.0.0.1:10000"))
    full_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")
