#!/usr/bin/env python3
# app.py – Geboorteplan-agent – Versie met Slimme Suggesties & Bevestigingen
# Deze versie lost het probleem van dubbele bevestiging op en ondersteunt de nieuwe frontend met dynamische chat-chips.

from __future__ import annotations
import os, json, uuid, sqlite3, time, logging, inspect, re

from typing import List, Dict, Optional, Any, Literal
from typing_extensions import TypedDict
from enum import Enum

from flask import Flask, request, jsonify, abort, send_file, send_from_directory, render_template
from flask_cors import CORS
from openai import OpenAI

# ───────────────────────── Logging & Basisconfiguratie ─────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s [%(levelname)s] %(name)s – %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mae-backend")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_CHOICE = os.getenv("MODEL_CHOICE", "gpt-4o")

ALLOWED_ORIGINS = ["https://bevalmeteenplan.nl", "https://www.bevalmeteenplan.nl", "https://chatbotbvmp.onrender.com"]
DB_FILE = "sessions.db"

# ───────────────────────── Flask App Setup ─────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates", static_url_path="")
CORS(app, origins=ALLOWED_ORIGINS, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# ───────────────────────── Tool Decorator & Schema Generator ─────────────────────────
def function_tool(func: Any) -> Any:
    sig = inspect.signature(func)
    parameters = {"type": "object", "properties": {}, "required": []}
    type_mapping = {str: "string", int: "integer", float: "number", bool: "boolean"}
    for name, param in sig.parameters.items():
        if name == "session_id": continue
        param_type = type_mapping.get(param.annotation, "string")
        if hasattr(param.annotation, '__origin__') and param.annotation.__origin__ is Literal:
            parameters["properties"][name] = {"type": "string", "enum": list(param.annotation.__args__)}
        else:
            parameters["properties"][name] = {"type": param_type}
        if param.default is inspect.Parameter.empty:
            parameters["required"].append(name)
    func.openai_schema = {"type": "function", "function": {"name": func.__name__, "description": func.__doc__ or "", "parameters": parameters}}
    return func

def get_schema(ft: Any) -> dict: return getattr(ft, "openai_schema", {})

# ───────────────────────── Domein & State Management ─────────────────────────
class Stage(str, Enum):
    THEME_SELECTION = "THEME_SELECTION"
    TOPIC_SELECTION = "TOPIC_SELECTION"
    QA_SESSION = "QA_SESSION"
    COMPLETED = "COMPLETED"

DEFAULT_THEMES = [
    {"name": "Ondersteuning", "description": "Wie je erbij wilt en wat hun rol is."},
    {"name": "Bevalling & medisch beleid", "description": "Wensen rondom pijnstilling en interventies."},
    {"name": "Sfeer en omgeving", "description": "Voorkeuren voor licht, geluid en privacy."},
    {"name": "Voeding na de geboorte", "description": "Keuzes rondom borst- of flesvoeding."},
    {"name": "Kraamtijd", "description": "Wensen voor de eerste dagen na de bevalling."},
    {"name": "Communicatie", "description": "Hoe je wilt dat zorgverleners met je communiceren."},
    {"name": "Speciale wensen", "description": "Overige punten die voor jou belangrijk zijn."},
    {"name": "Fotografie en video", "description": "Regels rondom het maken van opnames."},
    {"name": "Comfortmaatregelen", "description": "Niet-medische manieren om met pijn om te gaan, zoals een bad of massage."},
    {"name": "Partnerbetrokkenheid", "description": "Specifieke taken en wensen voor je partner."}
]
DEFAULT_TOPICS_PER_THEME = {
    "Ondersteuning": ["Aanwezigheid partner", "Rol van de partner", "Aanwezigheid doula", "Communicatie met zorgverleners"],
    "Bevalling & medisch beleid": ["Pijnbestrijding opties", "Houding tijdens bevallen", "Medische interventies", "Keizersnede voorkeuren"],
    "Sfeer en omgeving": ["Muziek en geluid", "Licht en temperatuur", "Gebruik van water (douche/bad)", "Privacy wensen"],
    "Voeding na de geboorte": ["Borstvoeding of flesvoeding", "Voedingshoudingen", "Kolven", "Hulp bij voeding"],
}

SYSTEM_PROMPT = """
# ROL EN DOEL
Jij bent "Mae", een gespecialiseerde, proactieve en empathische assistent. Jouw hoofddoel is om gebruikers te helpen een persoonlijk en compleet geboorteplan te creëren. Je bent een flexibele partner.

# HET PROCES: EEN GIDS, GEEN GEVANGENIS
Het standaardproces heeft de volgende fases (stages), die je helpen de gebruiker te begeleiden:
1.  **THEME_SELECTION**: De gebruiker kiest de hoofdthema's (max 6). Je biedt de standaardlijst aan, maar de gebruiker mag ook zelf thema's verzinnen. Gebruik `offer_choices` om de lijst te tonen. Wacht tot de gebruiker de keuzes bevestigt.
2.  **TOPIC_SELECTION**: Voor een gekozen thema, kiest de gebruiker onderwerpen (max 4 per thema). Ook hier mag de gebruiker zelf onderwerpen toevoegen.
3.  **QA_SESSION**: Pas als de gebruiker expliciet aangeeft klaar te zijn met alle keuzes, roep je `start_qa_session` aan. Dit genereert 4 vragen per gekozen onderwerp. Vervolgens stel je de vragen één voor één met `get_next_question` en sla je antwoorden op met `log_answer`.
4.  **COMPLETED**: Alle vragen zijn beantwoord.

**JOUW FLEXIBILITEIT**
De gebruiker is de baas. Als de gebruiker in de QA_SESSION een thema wil wijzigen, dan doe je dat. Je taak is om de intentie te begrijpen en de juiste tool te gebruiken. Gebruik de `get_plan_status` tool om jezelf te oriënteren over de huidige staat van het plan en de procesfase. Als je gevraagd wordt om onderwerpen te bedenken, geef dan een lijst in je tekst-antwoord. De frontend zal hier knoppen van maken.

# GOUDEN REGEL & DE UITZONDERING
- **REGEL:** Voordat je een tool gebruikt die data wijzigt (`add_item`, `remove_item`, `update_answer`, `change_stage`), MOET je de gebruiker **eerst om een duidelijke bevestiging vragen**.
- **UITZONDERING:** Als een bericht van de gebruiker **al een expliciete bevestiging IS** (bijvoorbeeld: "Bevestig mijn themakeuze...", "Ja, dat is goed", "Voeg maar toe"), dan mag je de bijbehorende tools **direct aanroepen** zonder opnieuw te vragen. Je herkent dit aan de context en de woordkeuze.
"""

def init_db():
    with sqlite3.connect(DB_FILE) as con: con.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, state TEXT NOT NULL)")
init_db()
def load_state(sid: str) -> Optional[dict]:
    with sqlite3.connect(DB_FILE) as con:
        row = con.execute("SELECT state FROM sessions WHERE id=?", (sid,)).fetchone()
        if not row: return None
        return json.loads(row[0])
def save_state(sid: str, st: dict):
    with sqlite3.connect(DB_FILE) as con: con.execute("REPLACE INTO sessions (id, state) VALUES (?, ?)", (sid, json.dumps(st)))
SESSION: Dict[str, dict] = {}
def get_session(sid: str) -> dict:
    if sid in SESSION: return SESSION[sid]
    db_state = load_state(sid)
    if db_state:
        SESSION[sid] = db_state
        return db_state
    return _create_new_session(sid)
def _create_new_session(sid: str) -> dict:
    log.info("Nieuwe sessie %s wordt aangemaakt.", sid[-6:])
    st = {
        "id": sid, "history": [{"role": "system", "content": SYSTEM_PROMPT}],
        "stage": Stage.THEME_SELECTION.value,
        "plan": {"themes": [], "topics": {}, "qa_items": []},
        "qa_queue": [], "current_question": None
    }
    SESSION[sid] = st
    return st
def persist(sid: str):
    if sid in SESSION:
        log.debug("Sessie %s opgeslagen.", sid[-6:])
        save_state(sid, SESSION[sid])

# ───────────────────────── Volledig Uitgewerkte Tools ─────────────────────────
@function_tool
def get_plan_status(session_id: str) -> str:
    st = get_session(session_id)
    return json.dumps({"stage": st['stage'], "plan": st['plan']})

@function_tool
def offer_choices(session_id: str, choice_type: Literal['themes', 'topics'], theme_context: Optional[str] = None) -> str:
    if choice_type == 'themes':
        return f"De 10 standaard thema's zijn: {', '.join([t['name'] for t in DEFAULT_THEMES])}."
    if choice_type == 'topics':
        if not theme_context: return "Error: Ik moet weten voor welk thema ik onderwerpen moet aanbieden."
        topics = DEFAULT_TOPICS_PER_THEME.get(theme_context, [])
        if not topics: return f"Ik heb geen standaard onderwerpen voor het thema '{theme_context}'. Je kunt zelf onderwerpen voorstellen."
        return f"Standaard onderwerpen voor '{theme_context}': {', '.join(topics)}."
    return "Error: ongeldig choice_type."

@function_tool
def add_item(session_id: str, item_type: Literal['theme', 'topic'], name: str, theme_context: Optional[str] = None, is_custom: bool = False) -> str:
    st = get_session(session_id)
    plan = st["plan"]
    if item_type == 'theme':
        if len(plan["themes"]) >= 6: return "Error: Je kunt maximaal 6 thema's kiezen."
        if name not in [t["name"] for t in plan["themes"]]:
            plan["themes"].append({"name": name, "is_custom": is_custom})
            return f"Thema '{name}' toegevoegd."
        return f"Thema '{name}' was al gekozen."
    if item_type == 'topic':
        if not theme_context: return "Error: Ik moet weten aan welk thema ik dit onderwerp moet toevoegen."
        if theme_context not in [t["name"] for t in plan["themes"]]: return f"Error: Thema '{theme_context}' is nog niet gekozen."
        topics = plan["topics"].setdefault(theme_context, [])
        if len(topics) >= 4: return f"Error: Je kunt maximaal 4 onderwerpen per thema kiezen."
        if name not in topics:
            topics.append(name)
            return f"Onderwerp '{name}' toegevoegd aan thema '{theme_context}'."
        return f"Onderwerp '{name}' was al gekozen."
    return "Error: Ongeldig item_type."

@function_tool
def remove_item(session_id: str, item_type: Literal['theme', 'topic'], name: str, theme_context: Optional[str] = None) -> str:
    st = get_session(session_id)
    plan = st["plan"]
    name_lower = name.lower()
    if item_type == 'theme':
        theme_to_remove = next((t for t in plan["themes"] if t["name"].lower() == name_lower), None)
        if theme_to_remove:
            plan["themes"].remove(theme_to_remove)
            if theme_to_remove["name"] in plan["topics"]: del plan["topics"][theme_to_remove["name"]]
            return f"Thema '{name}' en bijbehorende onderwerpen zijn verwijderd."
        return f"Error: Thema '{name}' niet gevonden."
    if item_type == 'topic':
        if not theme_context: return "Error: 'theme_context' is verplicht bij verwijderen van onderwerp."
        if theme_context in plan["topics"] and name in plan["topics"][theme_context]:
            plan["topics"][theme_context].remove(name)
            return f"Onderwerp '{name}' is verwijderd uit thema '{theme_context}'."
        return f"Error: Onderwerp '{name}' niet gevonden onder thema '{theme_context}'."
    return "Error: Ongeldig item_type."

@function_tool
def change_stage(session_id: str, new_stage: Literal['THEME_SELECTION', 'TOPIC_SELECTION', 'QA_SESSION', 'COMPLETED']) -> str:
    st = get_session(session_id)
    st['stage'] = new_stage
    return f"Oké, de procesfase is nu veranderd naar {new_stage}."

# ... (andere tools zoals start_qa_session, etc. blijven hetzelfde) ...
@function_tool
def start_qa_session(session_id: str) -> str:
    st = get_session(session_id)
    if st['stage'] == Stage.QA_SESSION.value: return "We zijn al in de vragenronde."
    plan = st["plan"]
    st["qa_queue"] = []
    for theme_name, topics in plan["topics"].items():
        for topic in topics:
            prompt = f"Genereer 4 korte, open vragen voor een zwangere vrouw over het onderwerp '{topic}' binnen het thema '{theme_name}' voor haar geboorteplan. Geef alleen een JSON-object terug met een 'questions' key die een lijst van strings bevat. Voorbeeld: {{\"questions\": [\"vraag 1\", \"vraag 2\"]}}"
            try:
                response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": prompt}], response_format={"type": "json_object"})
                questions = json.loads(response.choices[0].message.content).get("questions", [])
                for q_text in questions:
                    st["qa_queue"].append({"theme": theme_name, "topic": topic, "question": q_text})
            except Exception:
                st["qa_queue"].append({"theme": theme_name, "topic": topic, "question": f"Wat zijn je wensen omtrent {topic}?"})
    if not st["qa_queue"]: return "Er zijn geen onderwerpen gekozen om vragen over te stellen. Kies eerst onderwerpen."
    st['stage'] = Stage.QA_SESSION.value
    return f"Oké, de vragenlijst is gemaakt. We gaan nu beginnen met de vragen. Roep 'get_next_question' aan om de eerste vraag te stellen."
@function_tool
def get_next_question(session_id: str) -> str:
    st = get_session(session_id)
    if st['stage'] != Stage.QA_SESSION.value: return "Error: We zijn niet in de vragenronde. Start deze eerst met 'start_qa_session'."
    if not st["qa_queue"]:
        st['stage'] = Stage.COMPLETED.value
        st['current_question'] = None
        return "Alle vragen zijn beantwoord! Het geboorteplan is compleet."
    st["current_question"] = st["qa_queue"].pop(0)
    cq = st["current_question"]
    return f"Vraag over '{cq['topic']}': {cq['question']}"
@function_tool
def log_answer(session_id: str, answer: str) -> str:
    st = get_session(session_id)
    cq = st.get("current_question")
    if not cq: return "Error: Er is geen actieve vraag om te beantwoorden. Gebruik eerst 'get_next_question'."
    st["plan"]["qa_items"].append({"question": cq["question"], "answer": answer, "theme": cq["theme"], "topic": cq["topic"]})
    st["current_question"] = None
    return "Antwoord opgeslagen. Roep 'get_next_question' aan voor de volgende vraag."
@function_tool
def update_answer(session_id: str, question_text: str, new_answer: str) -> str:
    st = get_session(session_id)
    for qa_item in st["plan"]["qa_items"]:
        if qa_item["question"] == question_text:
            qa_item["answer"] = new_answer
            return f"Het antwoord op de vraag '{question_text[:30]}...' is bijgewerkt."
    return "Error: De betreffende vraag is niet gevonden."


tool_funcs = [get_plan_status, offer_choices, add_item, remove_item, change_stage, start_qa_session, get_next_question, log_answer, update_answer]
tools_schema = [get_schema(t) for t in tool_funcs]
TOOL_IMPLEMENTATIONS = {t.openai_schema['function']['name']: t for t in tool_funcs}

# ───────────────────────── Hoofd-endpoint & Agent Loop ─────────────────────────
def build_payload(st: dict, reply: str) -> dict:
    payload = {
        "assistant_reply": reply, "session_id": st["id"], "stage": st["stage"],
        "plan": st["plan"], "suggested_replies": []
    }
    # Heuristics om suggesties te extraheren voor de frontend
    confirmation_triggers = ["akkoord?", "is dat correct?", "wil je dat ik", "zal ik"]
    if any(trigger in reply.lower() for trigger in confirmation_triggers):
        payload["suggested_replies"] = ["Ja", "Nee"]
    else:
        # Zoek naar lijsten (bullet points of genummerd)
        list_items = re.findall(r'^\s*[\*\-•\d]\.?\s+(.*)', reply, re.MULTILINE)
        if list_items:
            payload["suggested_replies"] = [item.strip() for item in list_items]
            
    return payload

@app.post("/agent")
def agent_route():
    body = request.get_json(force=True) or {}
    msg  = body.get("message", "").strip()
    sid  = body.get("session_id") or str(uuid.uuid4())
    if not msg: return jsonify({"assistant_reply": "Ik heb geen bericht ontvangen."})

    st = get_session(sid)
    st["history"].append({"role": "user", "content": msg})
    log.info("Sessie %s, Stage: %s, User: \"%s\"", sid[-6:], st['stage'], msg)

    final_reply = run_main_agent_loop(sid)

    st["history"].append({"role": "assistant", "content": final_reply})
    persist(sid)
    log.info("Sessie %s, Assistant: \"%s\"", sid[-6:], final_reply)
    
    return jsonify(build_payload(st, final_reply))

def run_main_agent_loop(session_id: str) -> str:
    st = get_session(session_id)
    MAX_TURNS = 5
    for i in range(MAX_TURNS):
        log.debug("Agent loop turn %d voor sessie %s.", i + 1, session_id[-6:])
        
        try:
            response = client.chat.completions.create(model=MODEL_CHOICE, messages=st["history"], tools=tools_schema, tool_choice="auto")
        except Exception as e:
            log.error("Fout bij aanroepen OpenAI API: %s", e)
            return "Sorry, er is momenteel een probleem met de verbinding. Probeer het later opnieuw."

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        st["history"].append(response_message.model_dump(exclude_unset=True))

        if not tool_calls:
            return response_message.content or "Ik weet niet zeker hoe ik moet reageren, kun je dat anders formuleren?"

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            log.info("Sessie %s, Tool Call: %s", session_id[-6:], function_name)
            function_to_call = TOOL_IMPLEMENTATIONS.get(function_name)
            result = f"Error: Tool '{function_name}' niet gevonden."
            if function_to_call:
                try:
                    args = json.loads(tool_call.function.arguments)
                    result = function_to_call(session_id=session_id, **args)
                    persist(session_id)
                except Exception as e:
                    log.exception("Fout tijdens uitvoeren van tool %s", function_name)
                    result = f"Error: {e}"
            st["history"].append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": str(result) or "ok"})
            log.debug("Sessie %s, Tool Result: %s", session_id[-6:], result)

    log.error("Agent overschreed maximum aantal turns (%d) voor sessie %s.", MAX_TURNS, session_id[-6:])
    return "Het lijkt erop dat er iets vastloopt. Probeer je verzoek anders te formuleren."

# ───────────────────────── Overige Endpoints ─────────────────────────
@app.get("/export/<sid>")
def export_json(sid: str):
    st = load_state(sid)
    if not st: abort(404)
    plan_data = st.get("plan", {})
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"geboorteplan_{sid}.json")
    with open(path, "w", encoding="utf-8") as f: json.dump(plan_data, f, ensure_ascii=False, indent=2)
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path == "iframe": return render_template('iframe_page.html', backend_url=os.getenv("RENDER_EXTERNAL_URL", "http://127.0.0.1:10000"))
    full_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(full_path): return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)), debug=True)
