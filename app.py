#!/usr/bin/env python3
# app.py – Geboorteplan-agent – Versie met Function Calling
# Deze versie gebruikt de OpenAI Chat Completions API met Tool Calling voor maximale controle.
# De AI stuurt het gesprek aan, maar de backend beheert de state en de agent-loop.
# Dit bestand is volledig op zichzelf staand, zonder externe 'agents.py' afhankelijkheid.

from __future__ import annotations
import os, json, uuid, sqlite3, time, logging, inspect

from typing import List, Dict, Optional, Any
from typing_extensions import TypedDict

from flask import Flask, request, jsonify, abort, send_file, send_from_directory, render_template
from flask_cors import CORS
from openai import OpenAI

# ───────────────────────── Logging & Basisconfiguratie ─────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("mae-backend")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_CHOICE = os.getenv("MODEL_CHOICE", "gpt-4o-mini")

ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
    "https://chatbotbvmp.onrender.com",
]
DB_FILE = "sessions.db"

# ───────────────────────── Flask App Setup ─────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates", static_url_path="")
CORS(app, origins=ALLOWED_ORIGINS, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# ───────────────────────── Tool Decorator & Schema Generator (NIEUW) ─────────────────────────
def function_tool(func: Any) -> Any:
    """Decorator die een functie omzet in een OpenAI-compatibele tool met een JSON-schema."""
    sig = inspect.signature(func)
    parameters = {"type": "object", "properties": {}, "required": []}
    
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    for name, param in sig.parameters.items():
        if name == "session_id": continue # Deze wordt automatisch geïnjecteerd

        param_type = type_mapping.get(param.annotation, "string")
        parameters["properties"][name] = {"type": param_type}
        if param.default is inspect.Parameter.empty:
            parameters["required"].append(name)

    openai_schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": parameters,
        },
    }
    
    # Koppel het schema aan de functie zelf
    func.openai_schema = openai_schema
    return func

def get_schema(ft: Any) -> dict:
    """Haalt het OpenAI-schema uit een functie-object dat is gedecoreerd met @function_tool."""
    return getattr(ft, "openai_schema", {})

# ───────────────────────── Strikte Types & Constanten ─────────────────────────
class NamedDescription(TypedDict):
    name: str
    description: str

DEFAULT_TOPICS: Dict[str, List[NamedDescription]] = {
    "Ondersteuning": [
        {"name": "Wie wil je bij de bevalling?", "description": "Welke personen wil je er fysiek bij hebben?"},
        {"name": "Rol van je partner of ander persoon?", "description": "Specificeer taken of wensen voor je partner."},
        {"name": "Wil je een doula / kraamzorg?", "description": "Extra ondersteuning tijdens en na de bevalling."},
        {"name": "Wat verwacht je van het personeel?", "description": "Welke stijl van begeleiding past bij jou?"},
    ],
    "Bevalling & medisch beleid": [
        {"name": "Pijnstilling", "description": "Medicamenteuze en niet-medicamenteuze opties."},
        {"name": "Interventies", "description": "Bijv. inknippen, kunstverlossing, infuus."},
        {"name": "Noodsituaties", "description": "Wat als het anders loopt dan gepland?"},
        {"name": "Placenta-keuzes", "description": "Placenta bewaren, laten staan, of doneren?"},
    ],
    "Sfeer en omgeving": [
        {"name": "Muziek & verlichting", "description": "Rustige muziek? Gedimd licht?"},
        {"name": "Privacy", "description": "Wie mag binnenkomen en fotograferen?"},
        {"name": "Foto’s / video", "description": "Wil je opnames laten maken?"},
        {"name": "Eigen spulletjes", "description": "Bijv. eigen kussen, etherische olie."},
    ],
    "Voeding na de geboorte": [
        {"name": "Borstvoeding", "description": "Ondersteuning, kolven, rooming-in."},
        {"name": "Flesvoeding", "description": "Welke melk? Wie geeft de fles?"},
        {"name": "Combinatie-voeding", "description": "Afwisselen borst en fles."},
        {"name": "Allergieën", "description": "Rekening houden met familiaire allergieën."},
    ],
}

# ───────────────────────── De "Grondwet" van de Agent (System Prompt) ─────────────────────────
SYSTEM_PROMPT = """
# ROL EN DOEL
Jij bent "Mae", een gespecialiseerde en empathische assistent. Jouw enige doel is om gebruikers te helpen bij het stap-voor-stap invullen van een geboorteplan. Je volgt ALTIJD een strikt, vaststaand proces en wijkt hier nooit van af. Je bent geen algemene chatbot.

# HET PROCES (STATE MACHINE)
Je begeleidt de gebruiker door de volgende fases, die worden bijgehouden via een 'stage'-variabele in de sessie. Je MOET je tools gebruiken om de 'stage' te veranderen en de gebruiker door het proces te leiden.

- **Fase 1: Thema's Kiezen (stage: 'choose_theme')**
  - Je startpunt. Bied de gebruiker de hoofdthema's aan. Als de gebruiker een thema kiest, roep je `register_theme` aan. Dit verandert de stage automatisch naar 'choose_topic'.
  - De gebruiker kan meerdere thema's kiezen.

- **Fase 2: Onderwerpen Kiezen (stage: 'choose_topic')**
  - Voor het HUIDIGE thema (`current_theme`), bied je de beschikbare onderwerpen aan. Als de gebruiker een onderwerp kiest, roep je `register_topic` aan.
  - Als de gebruiker aangeeft "klaar te zijn met dit thema" of "terug te willen naar thema's", zet je de stage terug naar 'choose_theme' zodat een nieuw thema gekozen kan worden. Gebruik hiervoor `complete_topic_selection`.

- **Fase 3: Afronden Keuzes & Start Vragen (stage overgang)**
  - Wanneer de gebruiker in de 'choose_theme' fase aangeeft helemaal klaar te zijn, roep je de `complete_all_selections` tool aan. Deze tool controleert of alles is ingevuld en zet de stage naar 'qa'.

- **Fase 4: Vragen Beantwoorden (stage: 'qa')**
  - Nadat `complete_all_selections` is aangeroepen, is de volgende stap het stellen van vragen. Vraag de gebruiker of ze klaar is voor de vragen.
  - Als ze ja zegt, roep de `get_next_question` tool aan om de EERSTE vraag op te halen.
  - Stel de vraag aan de gebruiker. Voor elk antwoord dat de gebruiker geeft, roep je de `log_answer_and_get_next` tool aan. Deze tool slaat het antwoord op en geeft direct de volgende vraag terug.
  - Ga door met het aanroepen van `log_answer_and_get_next` totdat deze aangeeft dat er geen vragen meer zijn.

- **Fase 5: Voltooid (stage: 'completed')**
  - Als de `get_next_question` of `log_answer_and_get_next` tool aangeeft dat alle vragen zijn beantwoord, feliciteer je de gebruiker en vertel je dat het geboorteplan kan worden geëxporteerd.

# REGELS
- Gebruik ALTIJD de tools om de state (zoals 'stage' of 'qa' lijsten) aan te passen.
- Wees kort en doelgericht. Begeleid de gebruiker naar de volgende stap.
- Als je een vraag niet begrijpt, vraag om verduidelijking in de context van de huidige fase.
- Geef nooit medisch advies.
- Start het gesprek ALTIJD met het aanbieden van de thema's.
"""

# ───────────────────────── Database & Sessiebeheer ─────────────────────────
def init_db():
    with sqlite3.connect(DB_FILE) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                state TEXT NOT NULL
            )"""
        )
init_db()

def load_state(sid: str) -> Optional[dict]:
    with sqlite3.connect(DB_FILE) as con:
        row = con.execute("SELECT state FROM sessions WHERE id=?", (sid,)).fetchone()
        if not row: return None
        log.debug("Sessie %s geladen uit DB.", sid[-6:])
        return json.loads(row[0])

def save_state(sid: str, st: dict) -> None:
    with sqlite3.connect(DB_FILE) as con:
        con.execute("REPLACE INTO sessions (id, state) VALUES (?, ?)", (sid, json.dumps(st)))
        con.commit()
        log.debug("Sessie %s opgeslagen in DB.", sid[-6:])

SESSION: Dict[str, dict] = {}
def get_session(sid: str) -> dict:
    if sid in SESSION: return SESSION[sid]
    
    db_state = load_state(sid)
    if db_state:
        SESSION[sid] = db_state
        return db_state

    log.info("Nieuwe sessie %s wordt aangemaakt.", sid[-6:])
    st = {
        "id": sid, "stage": "choose_theme", "themes": [], "topics": {}, "qa": [],
        "history": [{"role": "system", "content": SYSTEM_PROMPT}],
        "ui_theme_opts": list(DEFAULT_TOPICS.keys()),
        "ui_topic_opts": [],
        "current_theme": None,
        "current_qa_topic": None,
    }
    SESSION[sid] = st
    return st

def persist(sid: str):
    if sid in SESSION: save_state(sid, SESSION[sid])

# ───────────────────────── Tools met Ingebouwde Guardrails ─────────────────────────
@function_tool
def register_theme(session_id: str, theme: str) -> str:
    """Registreert een door de gebruiker gekozen thema en zet de state naar 'choose_topic'."""
    st = get_session(session_id)
    if st["stage"] != "choose_theme":
        return f"Error: Kan geen thema registreren. Huidige fase is '{st['stage']}', moet 'choose_theme' zijn."
    
    if theme not in [t["name"] for t in st["themes"]]:
        st["themes"].append({"name": theme, "description": ""})
        st["stage"] = "choose_topic"
        st["current_theme"] = theme
        st["ui_topic_opts"] = DEFAULT_TOPICS.get(theme, [])
        persist(session_id)
        return f"Ok, thema '{theme}' is toegevoegd. De fase is nu 'choose_topic'. De gebruiker kan nu onderwerpen voor dit thema kiezen."
    return "Thema was al gekozen."

@function_tool
def register_topic(session_id: str, topic: str) -> str:
    """Registreert een specifiek onderwerp voor het HUIDIGE thema."""
    st = get_session(session_id)
    theme = st.get("current_theme")
    if st["stage"] != "choose_topic" or not theme:
        return f"Error: Kan geen onderwerp registreren. Huidige fase is '{st['stage']}' (geen huidig thema) en moet 'choose_topic' zijn."

    lst = st["topics"].setdefault(theme, [])
    if topic not in lst:
        lst.append(topic)
        persist(session_id)
        return f"Ok, onderwerp '{topic}' toegevoegd aan thema '{theme}'."
    return "Onderwerp was al gekozen."

@function_tool
def complete_topic_selection(session_id: str) -> str:
    """Finaliseert de onderwerpkeuze voor het HUIDIGE thema en keert terug naar de themakeuze."""
    st = get_session(session_id)
    if st["stage"] != "choose_topic":
        return f"Error: Kan onderwerpselectie niet afronden. Fase is '{st['stage']}', moet 'choose_topic' zijn."
    
    st["stage"] = "choose_theme"
    st["current_theme"] = None
    st["ui_topic_opts"] = []
    persist(session_id)
    return "Ok, teruggekeerd naar themakeuze. Vraag de gebruiker een nieuw thema te kiezen of de selectie af te ronden."

@function_tool
def complete_all_selections(session_id: str) -> str:
    """Finaliseert ALLE keuzes. Controleert of alles is ingevuld en zet de state naar 'qa'."""
    st = get_session(session_id)
    if not st["themes"]:
        return "Error: Gebruiker moet minimaal één thema kiezen."
    if not all(t["name"] in st["topics"] and st["topics"][t["name"]] for t in st["themes"]):
        return "Error: Voor elk gekozen thema moet minstens één onderwerp geselecteerd zijn."

    st["stage"] = "qa"
    persist(session_id)
    return "Ok, alle keuzes zijn afgerond. De fase is nu 'qa'. Vraag de gebruiker of ze klaar is om de vragen te beantwoorden."

def _find_next_question(st: dict) -> Optional[dict]:
    answered_qs = {(qa["theme"], qa["question"]) for qa in st.get("qa", [])}
    for theme_info in st.get("themes", []):
        theme_name = theme_info["name"]
        for topic_name in st.get("topics", {}).get(theme_name, []):
            all_topics_for_theme = DEFAULT_TOPICS.get(theme_name, [])
            topic_obj = next((t for t in all_topics_for_theme if t["name"] == topic_name), None)
            if topic_obj and (theme_name, topic_obj["description"]) not in answered_qs:
                return {"theme": theme_name, "question": topic_obj["description"], "topic": topic_name}
    return None

@function_tool
def get_next_question(session_id: str) -> str:
    """Haalt de EERSTE vraag op voor de gebruiker om te beantwoorden."""
    st = get_session(session_id)
    if st["stage"] != "qa":
        return f"Error: Kan geen vraag ophalen. Huidige fase is '{st['stage']}', moet 'qa' zijn."
    
    next_q = _find_next_question(st)
    if next_q:
        st["current_qa_topic"] = next_q
        persist(session_id)
        return f"Vraag voor onderwerp '{next_q['topic']}': {next_q['question']}"
    
    st["stage"] = "completed"
    persist(session_id)
    return "Alle vragen zijn beantwoord. Er zijn geen vragen meer."

@function_tool
def log_answer_and_get_next(session_id: str, answer: str) -> str:
    """Slaat het antwoord op de HUIDIGE vraag op en haalt de VOLGENDE vraag op."""
    st = get_session(session_id)
    if st["stage"] != "qa":
        return f"Error: Kan geen antwoord opslaan. Huidige fase is '{st['stage']}', moet 'qa' zijn."

    current_q = st.get("current_qa_topic")
    if not current_q:
        return "Error: Er is geen huidige vraag om te beantwoorden. Roep eerst 'get_next_question' aan."

    # Log het antwoord
    st["qa"].append({"theme": current_q["theme"], "question": current_q["question"], "answer": answer})
    
    # Zoek de volgende vraag
    next_q = _find_next_question(st)
    if next_q:
        st["current_qa_topic"] = next_q
        persist(session_id)
        return f"Antwoord opgeslagen. Volgende vraag voor onderwerp '{next_q['topic']}': {next_q['question']}"
    
    st["stage"] = "completed"
    st["current_qa_topic"] = None
    persist(session_id)
    return "Antwoord opgeslagen. Alle vragen zijn nu beantwoord."

# ───────────────────────── Tool Setup voor de API ─────────────────────────
tool_funcs = [
    register_theme, register_topic, complete_topic_selection,
    complete_all_selections, get_next_question, log_answer_and_get_next
]
tools_schema = [get_schema(t) for t in tool_funcs]
TOOL_IMPLEMENTATIONS = {t.openai_schema['function']['name']: t for t in tool_funcs}


# ───────────────────────── De Nieuwe Agent Endpoint ─────────────────────────
def build_payload(st: dict, reply: str) -> dict:
    """Stelt de JSON-respons samen voor de frontend."""
    return {
        "assistant_reply": reply,
        "session_id": st["id"],
        "stage": st["stage"],
        "options": st["ui_topic_opts"] if st["stage"] == "choose_topic" else st["ui_theme_opts"],
        "current_theme": st.get("current_theme"),
        "themes": st["themes"],
        "topics": st["topics"],
        "qa": st["qa"],
    }

@app.post("/agent")
def agent_route():
    body = request.get_json(force=True) or {}
    msg  = body.get("message", "")
    sid  = body.get("session_id") or str(uuid.uuid4())
    st   = get_session(sid)

    st["history"].append({"role": "user", "content": msg})
    log.info("Sessie %s, User: \"%s\"", sid[-6:], msg)

    MAX_TURNS = 5
    for i in range(MAX_TURNS):
        log.debug("Agent loop turn %d. History size: %d", i + 1, len(st["history"]))
        
        response = client.chat.completions.create(
            model=MODEL_CHOICE,
            messages=st["history"],
            tools=tools_schema,
            tool_choice="auto"
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            reply = response_message.content
            log.info("Sessie %s, Assistant: \"%s\"", sid[-6:], reply)
            st["history"].append({"role": "assistant", "content": reply})
            persist(sid)
            return jsonify(build_payload(st, reply))

        st["history"].append(response_message)
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            log.info("Sessie %s, Tool Call: %s", sid[-6:], function_name)
            function_to_call = TOOL_IMPLEMENTATIONS.get(function_name)
            
            result = f"Error: Tool '{function_name}' niet gevonden."
            if function_to_call:
                try:
                    args = json.loads(tool_call.function.arguments)
                    args["session_id"] = sid # Injecteer session_id
                    result = function_to_call(**args)
                except Exception as e:
                    log.exception("Fout tijdens uitvoeren van tool %s", function_name)
                    result = f"Error: {e}"
            
            st["history"].append(
                {"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": result or "ok"}
            )
            log.debug("Sessie %s, Tool Result: %s", sid[-6:], result)

    log.error("Agent overschreed maximum aantal turns (%d).", MAX_TURNS)
    return jsonify({"assistant_reply": "Er ging iets mis, probeer het opnieuw."}), 500

# ───────────────────────── Overige Endpoints (Export, Frontend) ─────────────────────────
@app.get("/export/<sid>")
def export_json(sid: str):
    st = load_state(sid)
    if not st: abort(404)
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

# ───────────────────────── Entrypoint ─────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)), debug=True)
