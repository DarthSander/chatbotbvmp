# app.py

# ============================================================
# Geboorteplan-agent – VOLLEDIG BIJGEWERKTE FLASK-APP
# Versie met geïntegreerde verbeteringen voor robuustheid en betere AI-interactie
# Compatibel met Agents-SDK 0.0.17 (strict-schema)
# Aangepast naar gpt-4.1
# ============================================================

from __future__ import annotations
import os, json, uuid, sqlite3
from copy import deepcopy
from typing import List, Dict, Optional
from typing_extensions import TypedDict

from flask import (
    Flask, request, Response, jsonify, abort,
    send_file, send_from_directory
)
from flask_cors import CORS
from openai import OpenAI
from agents import Agent, Runner, function_tool

# ---------- strikt type voor thema’s en topics ----------
class NamedDescription(TypedDict):
    name: str
    description: str

# ---------- basisconfig ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID = os.getenv("ASSISTANT_ID")  # van je Assistants-dashboard
ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
    "https://chatbotbvmp.onrender.com"
]
DB_FILE = "sessions.db"
# Model geüpdatet naar gpt-4.1 zoals gevraagd
MODEL_CHOICE = "gpt-4.1"

# Maak de Flask-app zó dat alles in /static op “/…” wordt geserveerd
app = Flask(
    __name__,
    static_folder="static",    # map met index.html, css/, js/
    static_url_path=""         # mapt “/foo.js” → “static/foo.js”
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

# ---------- SQLite-helper functions ----------
def init_db() -> None:
    with sqlite3.connect(DB_FILE) as con:
        con.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, state TEXT NOT NULL)")
init_db()

def load_state(sid: str) -> Optional[dict]:
    with sqlite3.connect(DB_FILE) as con:
        row = con.execute("SELECT state FROM sessions WHERE id=?", (sid,)).fetchone()
        return json.loads(row[0]) if row else None

def save_state(sid: str, st: dict) -> None:
    with sqlite3.connect(DB_FILE) as con:
        con.execute(
            "REPLACE INTO sessions (id, state) VALUES (?, ?)",
            (sid, json.dumps(st))
        )
        con.commit()

# ---------- In-memory sessies + persistence ----------
SESSION: Dict[str, dict] = {}
def get_session(sid: str) -> dict:
    if sid in SESSION:
        return SESSION[sid]
    if (db := load_state(sid)):
        loaded_history = db.get("history", []) 
        SESSION[sid] = {**db, "history": loaded_history}
        # Zorg ervoor dat de nieuwe sleutel ook bestaat voor oudere sessies
        SESSION[sid].setdefault("generated_topic_options", {})
        return SESSION[sid]
    # Nieuwe sessie
    SESSION[sid] = {
        "stage": "choose_theme",
        "themes": [],
        "topics": {},
        "qa": [],
        "history": [],
        "summary": "",
        "ui_theme_opts": [],
        "ui_topic_opts": [],
        "current_theme": None,
        "generated_topic_options": {}
    }
    return SESSION[sid]

def persist(sid: str) -> None:
    save_state(sid, SESSION[sid])

# ---------- History-samenvatting (bij >40 messages) ----------
def summarize_chunk(chunk: List[dict]) -> str:
    if not chunk:
        return ""
    filtered_chunk = [m for m in chunk if isinstance(m, dict) and 'role' in m and 'content' in m and isinstance(m['content'], str)]
    
    if not filtered_chunk:
        return ""

    txt = "\n".join(f"{m['role']}: {m['content']}" for m in filtered_chunk)
    
    summary_prompt = (
        "Vat dit deel van het gesprek over een geboorteplan samen in maximaal 300 tokens. "
        "Focus op de specifieke keuzes, wensen en beslissingen die door de gebruiker zijn genoemd. "
        "Noteer ook eventuele twijfels of openstaande vragen. De samenvatting wordt gebruikt als context voor het vervolg van het gesprek."
    )

    r = client.chat.completions.create(
        model=MODEL_CHOICE,
        messages=[
            {"role": "system", "content": summary_prompt},
            {"role": "user",   "content": txt}
        ],
        max_tokens=300
    )
    return r.choices[0].message.content.strip()

# ============================================================
# Tool-implementaties met DOCSTRINGS en FOUTAFHANDELING
# ============================================================
def _set_theme_options(session_id: str, options: List[NamedDescription]) -> str:
    """
    Toont een lijst met themakeuzes aan de gebruiker in de UI.
    Gebruik deze functie aan het begin van het gesprek om de gebruiker de hoofdthema's te presenteren.
    
    :param session_id: De unieke ID van de gebruikerssessie.
    :param options: Een lijst van objecten met 'name' en 'description' voor elk thema.
    :return: Een bevestiging 'ok'.
    """
    try:
        st = get_session(session_id)
        st["ui_theme_opts"] = options
        persist(session_id)
        return "ok"
    except Exception as e:
        print(f"Error in _set_theme_options for session {session_id}: {e}")
        return f"Kon themakeuzes niet instellen vanwege een technische fout: {e}"

def _set_topic_options(session_id: str, theme: str, options: List[NamedDescription]) -> str:
    """
    Toont een lijst met onderwerpkeuzes (topics) binnen een specifiek thema aan de gebruiker.
    Gebruik deze functie nadat een gebruiker een thema heeft gekozen met 'register_theme'.
    Deze functie slaat de getoonde opties op voor consistentie.
    
    :param session_id: De unieke ID van de gebruikerssessie.
    :param theme: De naam van het bovenliggende thema.
    :param options: Een lijst van objecten met 'name' en 'description' voor elk onderwerp.
    :return: Een bevestiging 'ok'.
    """
    try:
        st = get_session(session_id)
        st.setdefault("generated_topic_options", {}).setdefault(theme, options)
        
        st["ui_topic_opts"] = options
        st["current_theme"] = theme
        persist(session_id)
        return "ok"
    except Exception as e:
        print(f"Error in _set_topic_options for session {session_id}: {e}")
        return f"Kon onderwerpkeuzes niet instellen vanwege een technische fout: {e}"

def _register_theme(session_id: str, theme: str, description: str = "") -> str:
    """
    Registreert een gekozen thema voor het geboorteplan.
    Gebruik deze functie wanneer de gebruiker een hoofdthema selecteert of bevestigt.
    Deze functie voegt het thema toe aan de sessie. Roep hierna 'set_topic_options' aan.
    
    :param session_id: De unieke ID van de gebruikerssessie.
    :param theme: De naam van het thema, bijvoorbeeld 'Ondersteuning'.
    :param description: Een optionele beschrijving van het thema.
    :return: Een bevestiging 'ok'.
    """
    try:
        st = get_session(session_id)
        if len(st["themes"]) >= 6 or theme in [t["name"] for t in st["themes"]]:
            return "ok"
        st["themes"].append({"name": theme, "description": description})
        st["stage"] = "choose_topic"
        st["ui_topic_opts"] = DEFAULT_TOPICS.get(theme, [])
        st["current_theme"] = theme
        persist(session_id)
        return "ok"
    except Exception as e:
        print(f"Error in _register_theme for session {session_id}: {e}")
        return f"Kon thema niet registreren vanwege een technische fout: {e}"

def _register_topic(session_id: str, theme: str, topic: str) -> str:
    """
    Registreert een gekozen onderwerp (topic) binnen een thema.
    Gebruik deze functie wanneer de gebruiker een specifiek onderwerp kiest dat hij wil bespreken.
    
    :param session_id: De unieke ID van de gebruikerssessie.
    :param theme: De naam van het thema waartoe dit onderwerp behoort.
    :param topic: De naam van het gekozen onderwerp.
    :return: Een bevestiging 'ok'.
    """
    try:
        st = get_session(session_id)
        lst = st["topics"].setdefault(theme, [])
        if len(lst) >= 4 or topic in lst:
            return "ok"
        lst.append(topic)
        persist(session_id)
        return "ok"
    except Exception as e:
        print(f"Error in _register_topic for session {session_id}: {e}")
        return f"Kon onderwerp niet registreren vanwege een technische fout: {e}"

def _complete_theme(session_id: str) -> str:
    """
    Rondt de fase van thema- en onderwerpkeuze af en start de Q&A-fase.
    Roep deze functie ALLEEN aan als de gebruiker expliciet aangeeft klaar te zijn,
    bijvoorbeeld door op een knop te klikken die de boodschap 'user_finished_topic_selection' stuurt.
    
    :param session_id: De unieke ID van de gebruikerssessie.
    :return: Een bevestiging 'ok'.
    """
    try:
        st = get_session(session_id)
        all_selected_themes_have_topics = True
        if not st["themes"]:
            all_selected_themes_have_topics = False
        else:
            for theme_obj in st["themes"]:
                theme_name = theme_obj["name"]
                if theme_name not in st["topics"] or not st["topics"][theme_name]:
                    all_selected_themes_have_topics = False
                    break
        
        if all_selected_themes_have_topics:
            st["stage"] = "qa"
        else:
            st["stage"] = "choose_theme" 
            
        st["ui_topic_opts"] = []
        st["current_theme"] = None
        persist(session_id)
        return "ok"
    except Exception as e:
        print(f"Error in _complete_theme for session {session_id}: {e}")
        return f"Kon themaselectie niet afronden vanwege een technische fout: {e}"

def _log_answer(session_id: str, theme: str, question: str, answer: str) -> str:
    """
    Slaat een antwoord van de gebruiker op een specifieke vraag op in de sessie.
    Gebruik deze functie tijdens de Q&A-fase om de wensen van de gebruiker vast te leggen.
    
    :param session_id: De unieke ID van de gebruikerssessie.
    :param theme: Het thema van de vraag.
    :param question: De vraag die gesteld is.
    :param answer: Het antwoord van de gebruiker.
    :return: Een bevestiging 'ok'.
    """
    try:
        st = get_session(session_id)
        for item in st["qa"]:
            if item["theme"] == theme and item["question"] == question:
                item["answer"] = answer
                persist(session_id)
                return "ok"
        st["qa"].append({"theme": theme, "question": question, "answer": answer})
        persist(session_id)
        return "ok"
    except Exception as e:
        print(f"Error in _log_answer for session {session_id}: {e}")
        return f"Kon antwoord niet opslaan vanwege een technische fout: {e}"

def _update_answer(session_id: str, question: str, new_answer: str) -> str:
    """
    Werkt een eerder gegeven antwoord op een vraag bij.
    Gebruik deze functie als de gebruiker een eerder antwoord wil wijzigen.
    
    :param session_id: De unieke ID van de gebruikerssessie.
    :param question: De oorspronkelijke vraag waarvan het antwoord moet worden bijgewerkt.
    :param new_answer: Het nieuwe antwoord van de gebruiker.
    :return: Een bevestiging 'ok'.
    """
    try:
        st = get_session(session_id)
        for qa in st["qa"]:
            if qa["question"] == question:
                qa["answer"] = new_answer
                break
        persist(session_id)
        return "ok"
    except Exception as e:
        print(f"Error in _update_answer for session {session_id}: {e}")
        return f"Kon antwoord niet bijwerken vanwege een technische fout: {e}"

def _get_state(session_id: str) -> str:
    """
    Haalt de volledige huidige staat van de sessie op.
    Gebruik deze functie aan het begin van een complexe beurt of bij twijfel
    om de context te begrijpen (huidige fase, gekozen thema's, etc.).
    
    :param session_id: De unieke ID van de gebruikerssessie.
    :return: Een JSON-string met de volledige sessiestatus.
    """
    try:
        return json.dumps(get_session(session_id))
    except Exception as e:
        print(f"Error in _get_state for session {session_id}: {e}")
        return f"Kon sessiestatus niet ophalen vanwege een technische fout: {e}"

# Wrappers voor Agents-SDK
set_theme_options = function_tool(_set_theme_options)
set_topic_options = function_tool(_set_topic_options)
register_theme    = function_tool(_register_theme)
register_topic    = function_tool(_register_topic)
complete_theme    = function_tool(_complete_theme)
log_answer        = function_tool(_log_answer)
update_answer     = function_tool(_update_answer)
get_state_tool    = function_tool(_get_state)

# ============================================================
# Agent-template met BIJGEWERKTE INSTRUCTIES
# ============================================================
BASE_AGENT = Agent(
    name="Geboorteplan-agent",
    model=MODEL_CHOICE,
    instructions=(
        "Je bent een vriendelijke, ondersteunende en neutrale coach die (aanstaande) ouders helpt hun geboorteplan te maken. "
        "Je toon is warm, bemoedigend en duidelijk. Gebruik 'je' en 'jullie' om de gebruiker aan te spreken.\n\n"
        
        "**BELANGRIJKE VEILIGHEIDSREGELS:**\n"
        "- Je bent GEEN digitale verloskundige. Geef NOOIT medisch advies.\n"
        "- Als een gebruiker een medische vraag stelt (bv. 'is een ruggenprik gevaarlijk?'), antwoord dan ALTIJD dat je hier geen advies over mag geven en verwijs de gebruiker door naar hun verloskundige of gynaecoloog.\n"
        "- Jouw rol is uitsluitend het inventariseren en documenteren van de wensen van de gebruiker.\n\n"

        "**ALGEMENE REGELS:**\n"
        "- Bij twijfel over de huidige status (welk thema, welke fase), gebruik EERST de `get_state_tool` om de context te controleren.\n"
        "- Geef na een keuze van de gebruiker een korte, bevestigende samenvatting. Bijv: 'Oké, thema 'Sfeer' is toegevoegd. Laten we nu de onderwerpen bekijken.'\n\n"

        "**VASTE WERKFLOW:**\n"
        "1. **Thema's Kiezen:** Start met het tonen van de standaard thema's via `set_theme_options`. De gebruiker kan maximaal 6 thema's kiezen. Gebruik `register_theme` voor elke keuze.\n"
        "2. **Onderwerpen Kiezen:** Direct na `register_theme` presenteer je de onderwerpen voor dat thema. **BELANGRIJK:** Voordat je nieuwe onderwerpen bedenkt, controleer met `get_state_tool` of er al onderwerpen zijn opgeslagen in de sessiestatus onder `generated_topic_options` voor het huidige thema. \n"
        "   - **ALS ER al opties zijn:** Roep `set_topic_options` aan met de *exacte* lijst uit `generated_topic_options[thema_naam]`.\n"
        "   - **ALS ER GEEN opties zijn:** Bedenk dan een relevante lijst met onderwerpen (max. 4-5) en roep `set_topic_options` aan. Deze zullen dan automatisch worden opgeslagen voor later.\n"
        "   - De gebruiker kiest onderwerpen, die je vastlegt met `register_topic`.\n"
        "3. **Afronding Keuzefase:** De gebruiker zal via de interface een knop 'Klaar met kiezen' hebben. Dit stuurt jou de boodschap: 'user_finished_topic_selection'. WANNEER je deze exacte boodschap ontvangt, roep je de functie `complete_theme` aan om naar de volgende fase te gaan.\n"
        "4. **Q&A Fase:** Nadat `complete_theme` succesvol was, begin je met het stellen van open vragen over de gekozen thema's en onderwerpen. Sla de antwoorden op met `log_answer`.\n"
        "5. **Alle antwoorden in het Nederlands.** Gebruik bij élke tool het juiste `session_id`."
    ),
    tools=[
        set_theme_options, set_topic_options,
        register_theme, register_topic, complete_theme,
        log_answer, update_answer, get_state_tool
    ],
)

# ============================================================
# Streaming-/chat-endpoint (optioneel)
# ============================================================
def stream_run(tid: str):
    with client.beta.threads.runs.stream(thread_id=tid, assistant_id=ASSISTANT_ID) as ev:
        for e in ev:
            if e.event == "thread.message.delta" and e.data.delta.content:
                yield e.data.delta.content[0].text.value

@app.post("/chat")
def chat():
    if (o := request.headers.get("Origin")) and o not in ALLOWED_ORIGINS:
        abort(403)
    d   = request.get_json(force=True)
    msg = d.get("message", "")
    tid = d.get("thread_id") or client.beta.threads.create().id
    client.beta.threads.messages.create(thread_id=tid, role="user", content=msg)
    return Response(stream_run(tid),
                    headers={"X-Thread-ID": tid},
                    mimetype="text/plain")

# ============================================================
# Synchronous /agent-endpoint
# ============================================================
@app.post("/agent")
def agent():
    if (o := request.headers.get("Origin")) and o not in ALLOWED_ORIGINS:
        abort(403)

    body = request.get_json(force=True)
    msg  = body.get("message", "")
    sid  = body.get("session_id") or str(uuid.uuid4())
    st   = get_session(sid)

    # samenvatten bij lange history
    if len(st["history"]) > 40:
        st["summary"] = (st["summary"] + "\n" +
                         summarize_chunk(st["history"][:-20])).strip()
        st["history"] = st["history"][-20:]

    intro  = ([{"role": "system", "content": "Samenvatting van vorig gesprek:\n" + st["summary"]}]
              if st["summary"] else [])
    messages = intro + deepcopy(st["history"]) + [{"role": "user", "content": msg}]

    agent_inst = Agent(
        **{**BASE_AGENT.__dict__,
           # --- BEGIN CORRECTIE ---
           "instructions": BASE_AGENT.instructions + f'\n\nGebruik session_id="{sid}".'}
           # --- EINDE CORRECTIE ---
    )
    res = Runner().run_sync(agent_inst, messages)

    st["history"] = res.to_input_list()
    persist(sid)

    return jsonify({
        "assistant_reply": str(res.final_output),
        "session_id": sid,
        "options": st["ui_topic_opts"] if st["stage"] == "choose_topic"
                   else st["ui_theme_opts"],
        "current_theme": st["current_theme"],
        **{k: v for k, v in st.items()
           if k not in ("ui_theme_opts", "ui_topic_opts")}
    })

# ---------- export endpoint ----------
@app.get("/export/<sid>")
def e