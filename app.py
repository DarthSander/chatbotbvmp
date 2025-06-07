# app.py

# ============================================================
# Geboorteplan-agent – volledige Flask-app met static hosting
# Compatibel met Agents-SDK 0.0.17 (strict-schema)
# ============================================================

from __future__ import annotations
import os, json, uuid, sqlite3
from copy import deepcopy
from typing import List, Dict, Optional, Any
from typing_extensions import TypedDict
import datetime # Importeer datetime voor timestamps

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
        # Nieuwe tabel voor individuele berichten
        con.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT, -- Opslaan als JSON string
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        """)
    print("Database tabellen gecontroleerd/aangemaakt.")
init_db()

def _log_message(session_id: str, message: Dict[str, Any]) -> None:
    """Slaat een individueel bericht op in de 'messages' tabel."""
    with sqlite3.connect(DB_FILE) as con:
        tool_calls_json = json.dumps(message.get("tool_calls")) if message.get("tool_calls") else None
        con.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls) VALUES (?, ?, ?, ?)",
            (session_id, message["role"], message.get("content"), tool_calls_json)
        )
        con.commit()

def load_state(sid: str) -> Optional[dict]:
    """Laadt de sessiestate en reconstrueert de history uit de messages tabel."""
    with sqlite3.connect(DB_FILE) as con:
        # Laad de basis sessiestate
        row_session = con.execute("SELECT state FROM sessions WHERE id=?", (sid,)).fetchone()
        if not row_session:
            return None
        
        state = json.loads(row_session[0])

        # Laad de berichten voor deze sessie
        cursor = con.execute(
            "SELECT role, content, tool_calls FROM messages WHERE session_id=? ORDER BY timestamp ASC",
            (sid,)
        )
        history = []
        for row_msg in cursor.fetchall():
            msg_dict: Dict[str, Any] = {"role": row_msg[0]}
            if row_msg[1]: # content
                msg_dict["content"] = row_msg[1]
            if row_msg[2]: # tool_calls
                msg_dict["tool_calls"] = json.loads(row_msg[2])
            history.append(msg_dict)
        
        state["history"] = history # Voeg de gereconstrueerde history toe
        return state

def save_state(sid: str, st: dict) -> None:
    """Slaat de sessiestate (zonder history) op in de sessions tabel."""
    with sqlite3.connect(DB_FILE) as con:
        # Maak een kopie en verwijder history, want die slaan we apart op
        state_to_save = {k: v for k, v in st.items() if k != "history"}
        con.execute(
            "REPLACE INTO sessions (id, state) VALUES (?, ?)",
            (sid, json.dumps(state_to_save))
        )
        con.commit()

# ---------- In-memory sessies + persistence ----------
SESSION: Dict[str, dict] = {}
def get_session(sid: str) -> dict:
    if sid in SESSION:
        return SESSION[sid]
    
    # Probeer te laden uit DB
    if (db_state := load_state(sid)):
        SESSION[sid] = db_state
        return SESSION[sid]
    
    # Nieuwe sessie als niet gevonden in DB
    SESSION[sid] = {
        "stage": "choose_theme",
        "themes": [],
        "topics": {},
        "qa": [],
        "history": [], # Initiële lege history voor nieuwe sessies
        "summary": "",
        "ui_theme_opts": [],
        "ui_topic_opts": [],
        "current_theme": None
    }
    # Initialiseer ook in de DB als een nieuwe sessie
    save_state(sid, SESSION[sid]) 
    return SESSION[sid]

def persist(sid: str) -> None:
    # `save_state` zorgt er nu voor dat history NIET wordt opgeslagen in de state kolom
    # en dat individuele berichten al gelogd zijn via _log_message
    save_state(sid, SESSION[sid])

# ---------- History-samenvatting (bij >40 messages) ----------
def summarize_chunk(chunk: List[dict]) -> str:
    if not chunk:
        return ""
    
    # Filter non-string content or messages without 'role'/'content' for robustness
    # Also exclude tool_calls if they don't have a content string for summarization
    filtered_chunk = []
    for m in chunk:
        if isinstance(m, dict) and 'role' in m:
            if m['role'] == 'tool_calls': # tool calls hebben geen 'content' in dit formaat
                continue
            if 'content' in m and isinstance(m['content'], str):
                filtered_chunk.append(m)
    
    if not filtered_chunk: # If no valid messages remain after filtering
        return ""

    txt = "\n".join(f"{m['role']}: {m['content']}" for m in filtered_chunk)
    
    # Voeg een timestamp toe om de context te geven voor de samenvatting
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    r = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Vat de volgende chatgeschiedenis samen in maximaal 300 tokens. Houd het beknopt en focus op belangrijke beslissingen en feiten. Huidige tijd: {timestamp}"},
            {"role": "user",   "content": txt}
        ],
        max_tokens=300
    )
    return r.choices[0].message.content.strip()

# ============================================================
# Tool-implementaties
# ============================================================
# Deze tool-implementaties blijven grotendeels hetzelfde,
# omdat ze de sessiestate direct muteren, die vervolgens wordt geperst.

def _set_theme_options(session_id: str, options: List[NamedDescription]) -> str:
    st = get_session(session_id)
    st["ui_theme_opts"] = options
    persist(session_id)
    return "ok"

def _set_topic_options(session_id: str, theme: str, options: List[NamedDescription]) -> str:
    st = get_session(session_id)
    st["ui_topic_opts"] = options
    st["current_theme"] = theme
    persist(session_id)
    return "ok"

def _register_theme(session_id: str, theme: str, description: str = "") -> str:
    st = get_session(session_id)
    if len(st["themes"]) >= 6 or theme in [t["name"] for t in st["themes"]]:
        return "ok"
    st["themes"].append({"name": theme, "description": description})
    st["stage"] = "choose_topic"
    st["ui_topic_opts"] = DEFAULT_TOPICS.get(theme, [])
    st["current_theme"] = theme
    persist(session_id)
    return "ok"

def _register_topic(session_id: str, theme: str, topic: str) -> str:
    st = get_session(session_id)
    lst = st["topics"].setdefault(theme, [])
    if len(lst) >= 4 or topic in lst:
        return "ok"
    lst.append(topic)
    persist(session_id)
    return "ok"

def _complete_theme(session_id: str) -> str:
    st = get_session(session_id)
    # Deze logica bepaalt wanneer overgegaan wordt naar QA of terug naar thema selectie
    # Je kunt hier complexere checks toevoegen, bijv. alle geselecteerde thema's hebben topics
    
    # Voor nu, als er al 6 thema's zijn gekozen, of als de gebruiker kiest om door te gaan,
    # dan naar QA. Anders terug naar themakeuze.
    # Dit is afhankelijk van de exacte UI flow en intentie.
    # Laten we aannemen dat complete_theme wordt aangeroepen wanneer de gebruiker klaar is met ONDERWERPEN voor een thema.
    
    # Als er al voldoende thema's zijn gekozen (bijv. 6), dan naar de QA fase.
    # Anders terug naar de themakeuze fase.
    if len(st["themes"]) >= 6: # Dit is een simpele trigger, kan complexer
        st["stage"] = "qa"
    else:
        st["stage"] = "choose_theme"
        
    st["ui_topic_opts"] = []
    st["current_theme"] = None
    persist(session_id)
    return "ok"

def _log_answer(session_id: str, theme: str, question: str, answer: str) -> str:
    st = get_session(session_id)
    # Controleer of deze vraag al bestaat binnen dit thema, zo ja, update het antwoord
    for item in st["qa"]:
        if item["theme"] == theme and item["question"] == question:
            item["answer"] = answer
            persist(session_id)
            return "ok"
    # Anders, voeg nieuwe QA toe
    st["qa"].append({
        "theme": theme,
        "question": question,
        "answer": answer
    })
    persist(session_id)
    return "ok"

def _update_answer(session_id: str, question: str, new_answer: str) -> str:
    st = get_session(session_id)
    for qa in st["qa"]:
        if qa["question"] == question:
            qa["answer"] = new_answer
            break
    persist(session_id)
    return "ok"

def _get_state(session_id: str) -> str:
    # Deze tool retourneert de JSON-representatie van de huidige sessiestate.
    # De history in deze state is de in-memory history.
    return json.dumps(get_session(session_id))

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
# Streaming-/chat-endpoint (optioneel)
# ============================================================
# Dit endpoint gebruikt OpenAI Assistants API threads, die hun eigen geschiedenis bijhouden.
# De history die hier wordt opgeslagen in je DB is gescheiden van de thread history.
# Als je ook de streaming endpoint wilt loggen in je DB, moet je hier vergelijkbare logica toevoegen.
# Voor nu focus ik op het /agent endpoint, omdat die je eigen Runner gebruikt.
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
    # We werken nu met de in-memory history (st["history"])
    # De summary wordt gebaseerd op deze history en opgeslagen in de session state.
    if len(st["history"]) > 40:
        st["summary"] = (st["summary"] + "\n" +
                         summarize_chunk(st["history"][:-20])).strip()
        # Behoud alleen de laatste 20 berichten in de in-memory history
        # Oudere berichten zijn immers al in de DB gelogd en worden samengevat.
        st["history"] = st["history"][-20:] 
    
    # Voeg de nieuwe user message toe aan de in-memory history en log deze direct
    user_message = {"role": "user", "content": msg}
    st["history"].append(user_message)
    _log_message(sid, user_message) # Log gebruikerbericht in de DB

    intro  = ([{"role": "system", "content": "Samenvatting:\n" + st["summary"]}]
              if st["summary"] else [])
    
    # Messages voor de agent zijn de intro + de diepe kopie van de huidige in-memory history
    # Dit zorgt ervoor dat de agent de volledige context heeft voor de huidige beurt.
    messages_for_agent = intro + deepcopy(st["history"])

    agent_inst = Agent(
        **{**BASE_AGENT.__dict__,
           "instructions": BASE_AGENT.instructions + f"\n\nGebruik session_id=\"{sid}\"."}
    )
    res = Runner().run_sync(agent_inst, messages_for_agent)

    # De output van res.to_input_list() bevat de complete conversatie inclusief de tool calls en antwoorden van de agent
    # We moeten nu de *nieuwe* berichten sinds de laatste user input loggen
    new_history = res.to_input_list()
    
    # Identificeer en log alleen de berichten die nieuw zijn sinds de user input
    # (d.w.z., de berichten van de agent en de tool calls/responses)
    # Dit kan complex zijn, afhankelijk van hoe `res.to_input_list()` omgaat met de initiële `messages_for_agent`.
    # Een simpele aanpak: neem alles na de user_message die we zojuist hebben toegevoegd.
    # Dit vereist wel dat messages_for_agent correct is.

    # Om het robuust te maken: vergelijk de lengte van de history voor en na de run.
    # Log de berichten die in `new_history` zitten en niet in `messages_for_agent` (minus de intro).
    
    # Beter: log alle berichten van de agent_run die NIET de initiële user_message zijn.
    # Loop door res.to_input_list() en log wat nieuw is.
    
    # Let op: res.to_input_list() geeft de HELE run terug, inclusief de user input die we al logden.
    # We moeten dus zorgen dat we alleen de *nieuwe* berichten loggen, die de agent heeft gegenereerd.
    # Het is veiliger om te itereren over de events/messages die de Runner zelf heeft gegenereerd.
    
    # De Runner returns een object met details over de run. De res.final_output is de laatste tekst.
    # Echter, res.to_input_list() is de eenvoudigste manier om de "chat" te krijgen.
    # We loggen de user_message al. Nu moeten we de rest van de agent's beurt loggen.
    
    # Dit is een heuristiek: we weten dat de laatste message in st["history"] de user_message is.
    # Alles daarna (in res.to_input_list()) komt van de agent.
    # We moeten ervoor zorgen dat we geen dubbele user_messages opslaan.
    
    # Laten we het zo doen: we nemen de `messages_for_agent` (die al de user_message bevat),
    # en we updaten `st["history"]` met `res.to_input_list()`.
    # De berichten die *nieuw* zijn in `res.to_input_list()` vergeleken met `messages_for_agent` zijn de agent's replies.
    
    logged_messages_count = len(messages_for_agent) # Aantal berichten dat al bestond of zojuist is toegevoegd (user)
    
    # Voeg alle berichten van de agent's antwoord toe aan de in-memory history en log ze.
    for i in range(logged_messages_count, len(new_history)):
        agent_message = new_history[i]
        st["history"].append(agent_message) # Voeg toe aan in-memory history
        _log_message(sid, agent_message)    # Log in de DB

    persist(sid) # Sla de bijgewerkte sessiestate (zonder history, want die staat in messages tabel) op

    return jsonify({
        "assistant_reply": str(res.final_output),
        "session_id": sid,
        "options": st["ui_topic_opts"] if st["stage"] == "choose_topic"
                   else st["ui_theme_opts"],
        "current_theme": st["current_theme"],
        **{k: v for k, v in st.items()
           if k not in ("history", "ui_theme_opts", "ui_topic_opts")} # history wordt nu apart afgehandeld
    })

# ---------- export endpoint ----------
@app.get("/export/<sid>")
def export_json(sid: str):
    # Bij export willen we de volledige state, inclusief de gereconstrueerde history
    st = load_state(sid) 
    if not st:
        abort(404)
    path = f"/tmp/geboorteplan_{sid}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))

# ============================================================
# SPA-fallback: serveer frontend-bestanden uit /static
# ============================================================
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
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

