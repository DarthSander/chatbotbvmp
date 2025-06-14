#!/usr/bin/env python3
# app.py – Geboorteplan-agent • Volledige versie 13-06-2025
# VERSIE MET UITGEBREIDE LOGGING

from __future__ import annotations
import os, json, uuid, sqlite3, time, logging, inspect, pathlib
import re, uuid   
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from flask import Flask, request, jsonify, abort, send_file, send_from_directory, render_template
from flask_cors import CORS
from openai import OpenAI

# ─────────────────────────── Basisconfiguratie ───────────────────────────
ROOT = pathlib.Path(__file__).parent
DB_FILE = ROOT / "sessions.db"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MODEL_CHOICE = os.getenv("MODEL_CHOICE", "gpt-4o")

# Uitgebreide logging configuratie
logging.basicConfig(level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("mae-backend")

log.info("Mae-backend applicatie start...")
log.info(f"Log level ingesteld op: {LOG_LEVEL}")
log.info(f"Gebruikt OpenAI model: {MODEL_CHOICE}")
log.info(f"Database bestandslocatie: {DB_FILE}")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
    "https://chatbotbvmp.onrender.com"
]

app = Flask(__name__, static_folder="static", template_folder="templates", static_url_path="")
CORS(app, origins=ALLOWED_ORIGINS, allow_headers="*", methods=["GET", "POST", "OPTIONS"])
log.info(f"CORS ingeschakeld voor origins: {ALLOWED_ORIGINS}")

# ─────────────────── Helper: decorator → function-schema ───────────────────
def function_tool(fn: Any) -> Any:
    sig = inspect.signature(fn)
    schema = {"type": "object", "properties": {}, "required": []}
    py2json = {str:"string", int:"integer", float:"number", bool:"boolean"}
    for name, param in sig.parameters.items():
        if name == "session_id": continue
        if getattr(param.annotation, "__origin__", None) is Literal:
            schema["properties"][name] = {"type":"string","enum":list(param.annotation.__args__)}
        else:
            schema["properties"][name] = {"type": py2json.get(param.annotation,"string")}
        if param.default is inspect.Parameter.empty:
            schema["required"].append(name)
    fn.openai_schema = {"type":"function","function":{
        "name": fn.__name__, "description": fn.__doc__ or "", "parameters": schema}}
    return fn
def get_schema(f): return f.openai_schema

# ─────────────────────────── Domein-state ────────────────────────────────
class Stage(str,Enum):
    THEME_SELECTION="THEME_SELECTION"; TOPIC_SELECTION="TOPIC_SELECTION"
    QA_SESSION="QA_SESSION"; COMPLETED="COMPLETED"

DEFAULT_THEMES=[{"name":"Ondersteuning","description":"Wie je erbij wilt en wat hun rol is."},
{"name":"Bevalling & medisch beleid","description":"Wensen rondom pijnstilling en interventies."},
{"name":"Sfeer en omgeving","description":"Voorkeuren voor licht, geluid en privacy."},
{"name":"Voeding na de geboorte","description":"Keuzes rondom borst- of flesvoeding."},
{"name":"Kraamtijd","description":"Wensen voor de eerste dagen na de bevalling."},
{"name":"Communicatie","description":"Hoe je wilt dat zorgverleners met je communiceren."},
{"name":"Speciale wensen","description":"Overige punten die voor jou belangrijk zijn."},
{"name":"Fotografie en video","description":"Regels rondom het maken van opnames."},
{"name":"Comfortmaatregelen","description":"Niet-medische manieren om pijn te verlichten."},
{"name":"Partnerbetrokkenheid","description":"Specifieke taken en wensen voor je partner."}]

DEFAULT_TOPICS_PER_THEME={
 "Ondersteuning":["Aanwezigheid partner","Rol van de partner","Aanwezigheid doula","Communicatie met zorgverleners"],
 "Bevalling & medisch beleid":["Pijnbestrijding opties","Houding tijdens bevallen","Medische interventies","Keizersnede voorkeuren"],
 "Sfeer en omgeving":["Muziek en geluid","Licht en temperatuur","Gebruik van water (douche/bad)","Privacy wensen"],
 "Voeding na de geboorte":["Borstvoeding of flesvoeding","Voedingshoudingen","Kolven","Hulp bij voeding"],
}

# ─────────────────────────── DB-initialisatie ────────────────────────────
if not DB_FILE.exists():
    log.warning(f"Databasebestand {DB_FILE} niet gevonden. Nieuwe database wordt aangemaakt.")
    with sqlite3.connect(DB_FILE) as con:
        con.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, state TEXT NOT NULL)")
        con.execute("CREATE TABLE summaries (id TEXT, ts REAL, summary TEXT)")
    log.info("Database en tabellen (sessions, summaries) succesvol aangemaakt.")

def load_state(sid:str)->Optional[Dict[str,Any]]:
    log.debug(f"Pogen state te laden voor session_id: {sid}")
    with sqlite3.connect(DB_FILE) as con:
        row = con.execute("SELECT state FROM sessions WHERE id=?",(sid,)).fetchone()
        if row:
            log.debug(f"State gevonden en geladen voor session_id: {sid}")
            return json.loads(row[0])
        else:
            log.debug(f"Geen state gevonden voor session_id: {sid}")
            return None
def save_state(sid:str, st:Dict[str,Any]):
    log.debug(f"State opslaan voor session_id: {sid}")
    with sqlite3.connect(DB_FILE) as con:
        con.execute("REPLACE INTO sessions (id,state) VALUES (?,?)",(sid,json.dumps(st)))
    log.debug(f"State succesvol opgeslagen voor session_id: {sid}")

# ─────────────────────────── Sessions ────────────────────────────────────
def get_session(sid:str)->Dict[str,Any]:
    st=load_state(sid)
    if st:
        log.info(f"Bestaande sessie {sid} geladen. Huidige fase: {st.get('stage', 'ONBEKEND')}")
        return st
    log.info(f"Nieuwe sessie gestart met id: {sid}")
    st={"id":sid,
        "history":[{"role":"system","content":SYSTEM_PROMPT}],
        "stage":Stage.THEME_SELECTION.value,
        "plan":{"themes":[], "topics":{}, "qa_items":[]},
        "qa_queue":[], "current_question":None}
    save_state(sid,st); return st

# ───────────────────── Guardrail-sentiment ───────────────────────────────
def detect_sentiment(text:str)->Literal["ok","needs_menu"]:
    vague=["weet niet","geen idee","idk"]
    result = "needs_menu" if len(text.split())<4 or any(v in text.lower() for v in vague) else "ok"
    log.debug(f"Sentimentanalyse voor tekst '{text[:50]}...': {result}")
    return result

# ─────────────────────────── Tools ───────────────────────────────────────
@function_tool
def get_plan_status(session_id:str)->str:
    """Geeft de huidige status van het geboorteplan terug."""
    log.info(f"Tool 'get_plan_status' aangeroepen voor sessie: {session_id}")
    st=get_session(session_id)
    status = json.dumps({"stage":st["stage"],"plan":st["plan"]})
    log.debug(f"Plan status voor {session_id}: {status}")
    return status

@function_tool
def offer_choices(session_id:str,choice_type:Literal['themes','topics'],theme_context:Optional[str]=None)->str:
    """Biedt een lijst van keuzes voor thema's of onderwerpen."""
    log.info(f"Tool 'offer_choices' aangeroepen voor sessie: {session_id}. Type: {choice_type}, Context: {theme_context}")
    if choice_type == 'themes':
        choices = ", ".join(t["name"] for t in DEFAULT_THEMES)
        log.debug("Beschikbare thema's worden aangeboden.")
        return choices

    if choice_type == 'topics':
        if not theme_context:
            log.warning(f"Aanroep 'offer_choices' voor topics zonder theme_context in sessie {session_id}.")
            return "Error: theme_context is verplicht voor het opvragen van topics."

        log.debug(f"Zoeken naar onderwerpen voor thema: {theme_context}")
        for key, topics in DEFAULT_TOPICS_PER_THEME.items():
            if key.lower() == theme_context.lower():
                log.debug(f"Onderwerpen gevonden voor '{theme_context}': {topics}")
                return ", ".join(topics)

        log.warning(f"Geen standaard onderwerpen gevonden voor thema '{theme_context}' in sessie {session_id}.")
        return "Geen standaard onderwerpen gevonden. Bedenk zelf 4 tot 5 relevante suggesties voor dit thema."

    log.error(f"Ongeldig choice_type '{choice_type}' in 'offer_choices' voor sessie {session_id}.")
    return "Error: Ongeldig choice_type."

@function_tool
def add_item(session_id:str,item_type:Literal['theme','topic'],name:str,
             theme_context:Optional[str]=None,is_custom:bool=False)->str:
    """Voegt een thema of onderwerp toe aan het plan."""
    log.info(f"Tool 'add_item' voor sessie {session_id}: Type={item_type}, Naam='{name}', Context='{theme_context}', Custom={is_custom}")
    st=get_session(session_id); plan=st["plan"]
    if item_type=='theme':
        if len(plan["themes"])>=6:
            log.warning(f"Poging om meer dan 6 thema's toe te voegen in sessie {session_id}.")
            return "Error:max 6 thema's."
        if name not in [t["name"] for t in plan["themes"]]:
            plan["themes"].append({"name":name,"is_custom":is_custom})
            st["stage"]=Stage.TOPIC_SELECTION.value
            log.info(f"Thema '{name}' toegevoegd aan plan voor sessie {session_id}. Nieuwe fase: {st['stage']}")
    elif item_type=='topic' and theme_context:
        plan["topics"].setdefault(theme_context,[])
        if len(plan["topics"][theme_context])<4 and name not in plan["topics"][theme_context]:
            plan["topics"][theme_context].append(name)
            log.info(f"Onderwerp '{name}' toegevoegd aan thema '{theme_context}' voor sessie {session_id}.")
    save_state(session_id,st); return "ok"

@function_tool
def remove_item(session_id:str,item_type:Literal['theme','topic'],name:str,
                theme_context:Optional[str]=None)->str:
    """Verwijdert een thema of onderwerp uit het plan."""
    log.info(f"Tool 'remove_item' voor sessie {session_id}: Type={item_type}, Naam='{name}', Context='{theme_context}'")
    st=get_session(session_id); plan=st["plan"]
    if item_type=='theme':
        plan["themes"]=[t for t in plan["themes"] if t["name"]!=name]
        plan["topics"].pop(name,None)
        log.info(f"Thema '{name}' en bijbehorende topics verwijderd voor sessie {session_id}.")
    elif item_type=='topic' and theme_context:
        if name in plan["topics"].get(theme_context,[]):
             plan["topics"].get(theme_context,[]).remove(name)
             log.info(f"Onderwerp '{name}' van thema '{theme_context}' verwijderd voor sessie {session_id}.")
    save_state(session_id,st); return "ok"

@function_tool
def update_item(session_id:str,item_type:Literal['theme','topic'],
                old_name:str,new_name:str,theme_context:Optional[str]=None)->str:
    """Werkt de naam van een thema of onderwerp bij."""
    log.info(f"Tool 'update_item' voor sessie {session_id}: Type={item_type}, Oud='{old_name}', Nieuw='{new_name}', Context='{theme_context}'")
    st=get_session(session_id); plan=st["plan"]
    if item_type=='theme':
        for t in plan["themes"]:
            if t["name"]==old_name: t["name"]=new_name
        if old_name in plan["topics"]:
            plan["topics"][new_name]=plan["topics"].pop(old_name)
        log.info(f"Thema hernoemd van '{old_name}' naar '{new_name}' in sessie {session_id}.")
    elif item_type=='topic' and theme_context:
        plan["topics"][theme_context]=[new_name if x==old_name else x for x in plan["topics"].get(theme_context,[])]
        for qa in plan["qa_items"]:
            if qa["theme"]==theme_context and qa["topic"]==old_name:
                qa["topic"]=new_name
        log.info(f"Onderwerp in thema '{theme_context}' hernoemd van '{old_name}' naar '{new_name}' in sessie {session_id}.")
    save_state(session_id,st); return "ok"

# --- QA-tools ---
@function_tool
def start_qa_session(session_id:str)->str:
    """Start de vragenronde (QA-sessie)."""
    st=get_session(session_id)
    log.info(f"Tool 'start_qa_session' aangeroepen voor sessie {session_id}.")
    if st["stage"]==Stage.QA_SESSION.value:
        log.warning(f"Poging om QA-sessie te starten die al bezig is in sessie {session_id}.")
        return "We zijn al in de vragenronde."
    st["qa_queue"]=[]
    for theme,topics in st["plan"]["topics"].items():
        for t in topics:
            st["qa_queue"].append({"theme":theme,"topic":t,"question":f"Wat zijn je wensen rondom {t}?"})
    st["stage"]=Stage.QA_SESSION.value
    log.info(f"QA-sessie gestart voor sessie {session_id}. {len(st['qa_queue'])} vragen in wachtrij. Nieuwe fase: {st['stage']}")
    save_state(session_id,st); return "ok"

@function_tool
def get_next_question(session_id:str)->str:
    """Haalt de volgende vraag op uit de wachtrij."""
    st=get_session(session_id)
    log.info(f"Tool 'get_next_question' aangeroepen voor sessie {session_id}.")
    if not st["qa_queue"]:
        st["stage"]=Stage.COMPLETED.value
        save_state(session_id,st)
        log.info(f"Geen vragen meer in de wachtrij voor sessie {session_id}. Fase naar COMPLETED gezet.")
        return "Alle vragen zijn beantwoord!"
    st["current_question"]=st["qa_queue"].pop(0)
    save_state(session_id,st)
    question_text = f"Vraag over '{st['current_question']['topic']}': {st['current_question']['question']}"
    log.info(f"Volgende vraag voor sessie {session_id}: {question_text}")
    return question_text

@function_tool
def log_answer(session_id:str,answer:str)->str:
    """Slaat het antwoord op een vraag op."""
    st=get_session(session_id); cq=st["current_question"]
    log.info(f"Tool 'log_answer' aangeroepen voor sessie {session_id} met antwoord: '{answer[:50]}...'")
    if not cq:
        log.error(f"Poging om antwoord op te slaan zonder actieve vraag in sessie {session_id}.")
        return "Error:Geen actieve vraag."
    st["plan"]["qa_items"].append({**cq,"answer":answer}); st["current_question"]=None
    log.info(f"Antwoord op vraag over '{cq['topic']}' opgeslagen voor sessie {session_id}.")
    save_state(session_id,st); return "Antwoord opgeslagen."

@function_tool
def update_question(session_id:str,old_question:str,new_question:str)->str:
    """Werkt een bestaande vraag bij."""
    log.info(f"Tool 'update_question' aangeroepen voor sessie {session_id}. Oud: '{old_question[:30]}...', Nieuw: '{new_question[:30]}...'")
    st=get_session(session_id)
    st["qa_queue"]=[{**q,"question":new_question} if q["question"]==old_question else q for q in st["qa_queue"]]
    for qa in st["plan"]["qa_items"]:
        if qa["question"]==old_question: qa["question"]=new_question
    if st["current_question"] and st["current_question"]["question"]==old_question:
        st["current_question"]["question"]=new_question
    save_state(session_id,st); return "ok"

@function_tool
def remove_question(session_id:str,question_text:str)->str:
    """Verwijdert een vraag."""
    log.info(f"Tool 'remove_question' aangeroepen voor sessie {session_id} voor vraag: '{question_text[:50]}...'")
    st=get_session(session_id)
    st["qa_queue"]=[q for q in st["qa_queue"] if q["question"]!=question_text]
    st["plan"]["qa_items"]=[q for q in st["plan"]["qa_items"] if q["question"]!=question_text]
    save_state(session_id,st); return "ok"

@function_tool
def update_answer(session_id:str,question_text:str,new_answer:str)->str:
    """Werkt een gegeven antwoord bij."""
    log.info(f"Tool 'update_answer' aangeroepen voor sessie {session_id} voor vraag '{question_text[:30]}...' met nieuw antwoord: '{new_answer[:30]}...'")
    st=get_session(session_id)
    for qa in st["plan"]["qa_items"]:
        if qa["question"]==question_text: qa["answer"]=new_answer
    save_state(session_id,st); return "ok"

# --- Proactieve-Gids tools ---
@function_tool
def find_web_resources(topic:str,depth:Literal['brief','diep']='brief')->str:
    """Zoekt naar webbronnen over een bepaald onderwerp."""
    log.info(f"Tool 'find_web_resources' aangeroepen. Topic: {topic}, Depth: {depth}")
    return json.dumps({"summary":f"Samenvatting over {topic}",
                       "links":[f"https://example.com/{topic}"]})

@function_tool
def vergelijk_opties(options:List[str])->str:
    """Vergelijkt verschillende opties."""
    log.info(f"Tool 'vergelijk_opties' aangeroepen. Opties: {options}")
    return f"Vergelijking: {', '.join(options)}"

@function_tool
def geef_denkvraag(theme:str)->str:
    """Geeft een reflectievraag over een thema."""
    log.info(f"Tool 'geef_denkvraag' aangeroepen. Thema: {theme}")
    return f"Hoe voel je je over {theme}?"

@function_tool
def find_external_organization(keyword:str)->str:
    """Zoekt naar externe organisaties."""
    log.info(f"Tool 'find_external_organization' aangeroepen. Keyword: {keyword}")
    return json.dumps({"organisaties":[f"{keyword} support groep"]})

@function_tool
def check_onbeantwoorde_punten(session_id:str)->str:
    """Controleert of er nog onbeantwoorde vragen zijn."""
    log.info(f"Tool 'check_onbeantwoorde_punten' aangeroepen voor sessie {session_id}.")
    st=get_session(session_id)
    return json.dumps({"missing":st["qa_queue"]})

@function_tool
def genereer_plan_tekst(session_id:str,format:Literal['markdown','plain']='markdown')->str:
    """Genereert de tekst van het geboorteplan."""
    log.info(f"Tool 'genereer_plan_tekst' aangeroepen voor sessie {session_id} in formaat: {format}")
    st=get_session(session_id)
    return "# Geboorteplan\n"+json.dumps(st["plan"],ensure_ascii=False,indent=2)

@function_tool
def present_tool_choices(choices:str)->str:
    """Geeft JSON-string terug zodat de frontend dit als quick-replies kan tonen."""
    log.info(f"Tool 'present_tool_choices' aangeroepen met keuzes: {choices}")
    return choices

@function_tool
def save_plan_summary(session_id:str)->str:
    """Slaat een samenvatting van het plan op in de database."""
    st=get_session(session_id)
    log.info(f"Tool 'save_plan_summary' aangeroepen voor sessie {session_id}.")
    prompt="Vat dit geboorteplan samen in 5 bulletpoints:"+json.dumps(st["plan"],ensure_ascii=False)
    try:
        resp=client.chat.completions.create(model="gpt-3.5-turbo",messages=[{"role":"user","content":prompt}])
        summary=resp.choices[0].message.content.strip()
        log.info(f"Samenvatting succesvol gegenereerd voor sessie {session_id}.")
    except Exception as e:
        log.error(f"Genereren van samenvatting mislukt voor sessie {session_id}: {e}", exc_info=True)
        summary=f"(samenvatting mislukt: {e})"
    with sqlite3.connect(DB_FILE) as con:
        con.execute("INSERT INTO summaries (id,ts,summary) VALUES (?,?,?)",(session_id,time.time(),summary))
    log.info(f"Samenvatting opgeslagen in database voor sessie {session_id}.")
    return "summary_saved"

# *** NIEUWE TOOL TOEGEVOEGD ***
@function_tool
def propose_quick_replies(session_id: str, replies: List[str]) -> str:
    """Stel een lijst van quick reply-knoppen voor die de frontend kan tonen. Gebruik dit voor ja/nee-vragen of om de gebruiker te helpen kiezen."""
    log.info(f"Tool 'propose_quick_replies' aangeroepen voor sessie {session_id} met replies: {replies}")
    # Deze tool is een signaal voor de 'build_payload' functie.
    # We geven de lijst van replies gewoon terug als bevestiging.
    return f"Quick replies voorgesteld: {', '.join(replies)}"

# *** SYSTEM_PROMPT VOLLEDIG BIJGEWERKT ***
SYSTEM_PROMPT = """
# MAE - GEBOORTEPLAN ASSISTENT

## IDENTITEIT
Je bent Mae, een warme, empathische en proactieve assistent die zwangere vrouwen helpt bij het maken van een persoonlijk geboorteplan. Je bent flexibel, luistert goed en past je aan de wensen van de gebruiker aan.

## WERKWIJZE: VIER FASEN
Gebruik `get_plan_status()` om te controleren in welke fase je bent.

### HOOFDPRINCIPE: PROACTIEVE NAVIGATIE
Wacht niet altijd op de gebruiker, maar stel proactief de volgende logische stap voor. Vraag bijvoorbeeld na het kiezen van onderwerpen: "Mooi, de basis staat. Ben je klaar om met de vragen te beginnen, of wil je eerst nog iets aanpassen?"

### 1. THEME_SELECTION
- **Doel**: Gebruiker kiest maximaal 6 hoofdthema's.
- **Actie**: Gebruik `offer_choices(choice_type='themes')` om standaardthema's te tonen.
- **Flexibiliteit**: Gebruiker mag eigen thema's toevoegen (is_custom=True).
- **Tools**: `add_item(item_type='theme')`, `remove_item(item_type='theme')`, `update_item(item_type='theme')`.

### 2. TOPIC_SELECTION
- **Doel**: Voor elk gekozen thema selecteert gebruiker maximaal 4 onderwerpen.
- **Actie**: Gebruik `offer_choices(choice_type='topics', theme_context='THEMA_NAAM')`.
- **Speciale Instructie**: Wanneer `offer_choices` de tekst "Geen standaard onderwerpen gevonden..." teruggeeft, betekent dit dat je proactief 3 tot 4 relevante onderwerpen voor dat thema moet bedenken en aan de gebruiker moet voorstellen.
- **Principe van Samenvatting**: Geef na het voltooien van de onderwerpen voor een thema een korte, bevestigende samenvatting. Voorbeeld: "Oké, voor het thema 'Sfeer en omgeving' hebben we nu de onderwerpen 'Muziek', 'Licht' en 'Privacy'. Klopt dat zo?"
- **Tools**: `add_item(item_type='topic')`, `remove_item(item_type='topic')`, `update_item(item_type='topic')`.

### 3. QA_SESSION
- **Start**: Roep `start_qa_session()` aan om de vragenronde te beginnen.
- **Proces**: Stel vragen één-voor-één met `get_next_question()` en sla antwoorden op met `log_answer()`.
- **Proactieve gids**: Bij korte/vage antwoorden, bied hulp aan via `present_tool_choices()`.
- **QA Management**: `update_question()`, `remove_question()`, `update_answer()`.

### 4. COMPLETED
- **Resultaat**: Alle vragen zijn beantwoord, het plan is compleet.
- **Actie**: `save_plan_summary()` wordt automatisch aangeroepen.
- **Export**: `genereer_plan_tekst()` voor een tekstversie.

## COMMUNICATIE PRINCIPES

### Quick Replies Sturen (Ja/Nee en andere keuzes)
- **REGEL**: Wanneer je een vraag stelt die een duidelijke keuze van de gebruiker vereist, roep je DIRECT na je vraag de tool `propose_quick_replies` aan om knoppen te genereren.
- **Voorbeeld 1 (Bevestiging)**:
  - Jouw tekst: "Zal ik het thema 'Pijnstilling' toevoegen?"
  - Jouw volgende actie: Roep `propose_quick_replies(replies=["Ja", "Nee"])` aan.
- **Voorbeeld 2 (Keuze bieden)**:
  - Jouw tekst: "Wil je meer informatie over dit onderwerp, of wil je doorgaan naar de volgende vraag?"
  - Jouw volgende actie: Roep `propose_quick_replies(replies=["Meer informatie", "Volgende vraag"])` aan.
- Dit vervangt de oude, onbetrouwbare detectie van sleutelwoorden.

## GESPREKSVOERING

### Toon & Stijl
- **Warm en ondersteunend**: "Wat fijn dat je bezig bent met je geboorteplan!".
- **Duidelijke instructies**: "Ik toon je nu de beschikbare thema's...".
- **Flexibel**: "We kunnen altijd terug om iets aan te passen.".
- **Proactief**: "Wil je dat ik wat meer uitleg geef over...?".
- **Toon begrip bij gevoelige onderwerpen**: Erken het persoonlijke karakter van een keuze. Zeg bijvoorbeeld: "Dat is een heel persoonlijke keuze, dank je wel voor het delen. Ik zorg ervoor dat dit duidelijk in je plan komt te staan."

### Foutafhandeling & Grensgevallen
- **Tools kunnen "Error" teruggeven**: Leg vriendelijk uit dat er iets niet lukt en probeer het op een andere manier.
- **Fallback bij falende tools**: Als een tool (zoals `offer_choices`) tweemaal faalt, stop dan met het aanroepen ervan. Schakel over op een open vraag. Voorbeeld: "Het lukt me even niet om de standaardopties op te halen. Kun je misschien zelf een paar onderwerpen voor dit thema noemen die voor jou belangrijk zijn?"
- **Omgaan met Off-Topic Vragen**: Als de gebruiker een vraag stelt die niets te maken heeft met het geboorteplan, geef dan geen antwoord op de vraag zelf. Erken de vraag kort en stuur het gesprek vriendelijk terug. Voorbeeld: "Dat is een interessante vraag, maar mijn expertise is echt gericht op het helpen samenstellen van jouw geboorteplan. Zullen we verdergaan met het volgende onderwerp?"
- **Validatie**: Valideer gebruikersinput voordat je tools aanroept (bijv. max 6 thema's, max 4 onderwerpen).

## FLEXIBILITEIT REGELS
- De gebruiker heeft altijd de controle en mag op elk moment terug naar een vorige fase of iets aan te passen.
- Gebruik de diverse `update_` en `remove_` tools om de gebruiker deze flexibiliteit te bieden.

Je bent een deskundige gids die het geboorteplan-proces soepel en ondersteunend begeleidt, waarbij de gebruiker altijd de controle behoudt.
"""

# ─────────────────────── Tool-registratie (BIJGEWERKT) ──────────────────
tool_funcs=[get_plan_status, offer_choices, add_item, remove_item, update_item,
            start_qa_session, get_next_question, log_answer,
            update_question, remove_question, update_answer,
            find_web_resources, vergelijk_opties, geef_denkvraag, find_external_organization,
            check_onbeantwoorde_punten, genereer_plan_tekst,
            present_tool_choices, save_plan_summary,
            propose_quick_replies] # <-- Nieuwe tool toegevoegd
tools_schema=[get_schema(t) for t in tool_funcs]
TOOL_MAP={t.openai_schema['function']['name']:t for t in tool_funcs}
log.info(f"{len(tool_funcs)} tools geregistreerd: {[name for name in TOOL_MAP.keys()]}")

# ───────────────────────── Agent-loop ────────────────────────────────────
def _auto_quick_replies(st: dict, assistant_msg) -> Optional[dict]:
    """
    Injecteer – indien nodig – een pseudo-assistant-bericht met een
    propose_quick_replies-tool-call.  Geeft het gegenereerde bericht-dict
    terug óf None wanneer er niets hoeft te gebeuren.
    """
    # fases waarin we géén knoppen willen (sidebar-flows)
    SKIP_STAGES = {
        Stage.THEME_SELECTION.value,
        Stage.TOPIC_SELECTION.value,
        Stage.COMPLETED.value,
    }

    # 1) al quick-replies aanwezig? 2) fase uitgesloten? 3) geen vraagteken → niets doen
    if (any(tc.function.name == "propose_quick_replies"
            for tc in (assistant_msg.tool_calls or []))
        or st["stage"] in SKIP_STAGES
        or not (assistant_msg.content or "").strip().endswith("?")):
        return None

    # heuristiek: “… of …?” ⇒ twee opties; anders standaard Ja/Nee
    m = re.search(r"\b(.+?)\s+of\s+(.+?)\?$", assistant_msg.content, re.I)
    replies = ([m.group(1).strip().capitalize(),
                m.group(2).strip().capitalize()] if m else ["Ja", "Nee"])

    # roep de échte tool aan (voor logging + consistentie)
    propose_quick_replies(session_id=st["id"], replies=replies)

    # pseudo-tool-call bericht (géén tool-result opnemen → laatste assistant-item blijft ‘assistant’)
    return {
        "role": "assistant",
        "tool_calls": [{
            "id": str(uuid.uuid4()),
            "function": {
                "name": "propose_quick_replies",
                "arguments": json.dumps({"replies": replies})
            }
        }]
    }


# ──────────────────────────────────────────────────────────────────────────
def run_main_agent_loop(sid: str) -> str:
    """
    Hoofdloop: stuurt de conversatie naar OpenAI, voert tool-calls uit,
    bewaart sessiestate en voegt – zo nodig – automatisch quick-replies toe.
    """
    st = get_session(sid)
    log.info(f"Start main agent loop voor sessie {sid}. History-len: {len(st['history'])}")

    # ---------- eenvoudige guardrail -------------------------------------------------
    if detect_sentiment(st["history"][-1]["content"]) == "needs_menu":
        menu = json.dumps({
            "choices": [
                {"label": "Meer info",
                 "tool":  "find_web_resources",
                 "args":  {"topic": "pijnstilling"}},
                {"label": "Vergelijk opties",
                 "tool":  "vergelijk_opties",
                 "args":  {"options": ["epiduraal", "badbevalling"]}},
                {"label": "Reflectie",
                 "tool":  "geef_denkvraag",
                 "args":  {"theme": "pijnstilling"}}
            ]
        })
        return present_tool_choices(session_id=sid, choices=menu)
    # -------------------------------------------------------------------------------

    for turn in range(5):                     # MAX_TURNS = 5
        log.info(f"Agent-beurt {turn + 1}/5 voor sessie {sid}")

        # 1) LLM-antwoord ophalen
        resp = client.chat.completions.create(
            model       = MODEL_CHOICE,
            messages    = st["history"],
            tools       = tools_schema,
            tool_choice = "auto"
        )
        msg = resp.choices[0].message

        # 2) voeg het assistant-bericht meteen toe
        st["history"].append(msg.model_dump(exclude_unset=True))

        # 3) voer eventuele tool-calls van het LLM-bericht uit
        tool_results: List[Dict[str, Any]] = []
        for call in (msg.tool_calls or []):
            fn = TOOL_MAP.get(call.function.name)
            try:
                args   = json.loads(call.function.arguments)
                result = fn(session_id=sid, **args) if fn else f"Error: tool '{call.function.name}' not found."
            except Exception as e:
                result = f"Error executing tool: {e}"
                log.error(result, exc_info=True)

            tool_results.append({
                "tool_call_id": call.id,
                "role"       : "tool",
                "name"       : call.function.name,
                "content"    : str(result)
            })

        # 4) history veilig mergen (state kan in tool-calls gewijzigd zijn)
        local_history = st["history"] + tool_results
        st            = get_session(sid)           # verse state met plan-updates
        st["history"] = local_history

        # 5) automatisch quick-replies toevoegen – ná de tool-results,
        #    zodat dit het laatste assistant-item wordt
        injected = _auto_quick_replies(st, msg)
        if injected:
            st["history"].append(injected)

        # 6) state opslaan (één keer per loop-iteratie)
        save_state(sid, st)

        # 7) SHORT-CIRCUIT: als (a) we zojuist iets injecteerden óf
        #    (b) het LLM zelf quick-replies heeft aangeroepen,
        #    dan is dit gebruikers-beurtje klaar.
        if (injected or
            (msg.tool_calls and msg.tool_calls[-1].function.name == "propose_quick_replies")):
            return msg.content or "(geen antwoord)"

        # 8) als er überhaupt géén tool-calls waren, zijn we ook klaar
        if not msg.tool_calls:
            return msg.content or "(geen antwoord)"

        # anders: ga door naar de volgende turn (max 5)

    log.warning(f"MAX_TURNS bereikt voor sessie {sid}")
    return "(max turns bereikt)"


# ───────────────────── build_payload (VOLLEDIG VERVANGEN) ────────────────
def build_payload(st:Dict[str,Any], reply:str)->Dict[str,Any]:
    """Stelt de JSON payload samen die naar de frontend wordt gestuurd."""
    log.info(f"Samenstellen van payload voor sessie {st.get('id', 'onbekend')}.")
    payload = {
        "assistant_reply": reply,
        "session_id": st["id"],
        "stage": st["stage"],
        "plan": st["plan"],
        "suggested_replies": []
    }

    if not st["history"]:
        log.warning(f"build_payload aangeroepen met lege history voor sessie {st.get('id')}")
        return payload

    last_message = st["history"][-1]
    if (last_message and
        last_message.get("role") == "assistant" and
        last_message.get("tool_calls")):

        log.debug(f"Laatste bericht in history bevat tool calls. Controleren op 'propose_quick_replies'.")
        for tool_call in last_message["tool_calls"]:
            if tool_call.get("function", {}).get("name") == "propose_quick_replies":
                try:
                    args = json.loads(tool_call["function"]["arguments"])
                    payload["suggested_replies"] = args.get("replies", [])
                    log.info(f"Quick replies gevonden en toegevoegd aan payload: {payload['suggested_replies']}")
                    break
                except (json.JSONDecodeError, TypeError) as e:
                    log.error(f"Fout bij parsen van 'propose_quick_replies' argumenten: {e}", exc_info=True)
                    pass

    log.debug(f"Finale payload voor sessie {st.get('id')}:\n{json.dumps(payload, indent=2, ensure_ascii=False)}")
    return payload

# ───────────────────────── Flask-routes ─────────────────────────────────
@app.post("/agent")
def agent_route():
    log.info(f"Binnenkomend verzoek op /agent van IP: {request.remote_addr}")
    try:
        body=request.get_json(force=True) or {}
        msg=body.get("message","").strip()
        sid=body.get("session_id") or str(uuid.uuid4())
        log.info(f"Sessie ID: {sid}, Bericht: '{msg}'")

        if not msg:
            log.warning("Leeg bericht ontvangen.")
            return jsonify({"assistant_reply":"(leeg bericht)","session_id":sid})

        st=get_session(sid)
        st["history"].append({"role":"user","content":msg})
        save_state(sid,st)
        log.debug(f"Gebruikersbericht toegevoegd aan history voor sessie {sid}.")

        reply=run_main_agent_loop(sid)
        log.info(f"Agent-loop voltooid voor sessie {sid}. Antwoord: '{reply[:100]}...'")

        final_st = get_session(sid)

        if final_st["stage"]==Stage.COMPLETED.value:
            log.info(f"Sessie {sid} is in COMPLETED fase. Samenvatting wordt opgeslagen.")
            save_plan_summary(session_id=sid)

        response_payload = build_payload(final_st,reply)
        return jsonify(response_payload)
    except Exception as e:
        log.critical(f"Onverwachte fout in /agent route: {e}", exc_info=True)
        return jsonify({"error": "Er is een interne serverfout opgetreden."}), 500

@app.get("/export/<sid>")
def export_json(sid):
    log.info(f"Exportverzoek voor sessie {sid} (JSON).")
    st=load_state(sid);
    if not st:
        log.warning(f"Exportverzoek voor onbekende sessie {sid}.")
        abort(404)
    path=ROOT/f"geboorteplan_{sid}.json"
    path.write_text(json.dumps(st["plan"],ensure_ascii=False,indent=2),"utf-8")
    log.info(f"Geboorteplan voor {sid} geëxporteerd naar {path}")
    return send_file(path,as_attachment=True,download_name=path.name)

# iframe- en statische bestanden (oorspronkelijk)
@app.route("/", defaults={"path":""})
@app.route("/<path:path>")
def serve_frontend(path):
    log.debug(f"Frontend verzoek voor pad: '{path}'")
    if path=="iframe":
        backend_url = os.getenv("RENDER_EXTERNAL_URL","http://127.0.0.1:10000")
        log.info(f"Iframe pagina wordt geserveerd met backend_url: {backend_url}")
        return render_template("iframe_page.html", backend_url=backend_url)

    full_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(full_path):
        log.debug(f"Statisch bestand '{path}' wordt geserveerd.")
        return send_from_directory(app.static_folder, path)

    log.debug("Geen specifiek pad gevonden, 'index.html' wordt geserveerd.")
    return send_from_directory(app.static_folder, "index.html")

if __name__=="__main__":
    port = int(os.getenv("PORT", 10000))
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    log.info(f"Flask applicatie wordt gestart op 0.0.0.0:{port} (Debug modus: {debug_mode})")
    app.run("0.0.0.0", port, debug=debug_mode)
