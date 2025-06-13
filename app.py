#!/usr/bin/env python3
# app.py – Geboorteplan-agent • Volledige versie 13-06-2025

from __future__ import annotations
import os, json, uuid, sqlite3, time, logging, inspect, pathlib
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

logging.basicConfig(level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger("mae-backend")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
    "https://chatbotbvmp.onrender.com"
]

app = Flask(__name__, static_folder="static", template_folder="templates", static_url_path="")
CORS(app, origins=ALLOWED_ORIGINS, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

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
    with sqlite3.connect(DB_FILE) as con:
        con.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, state TEXT NOT NULL)")
        con.execute("CREATE TABLE summaries (id TEXT, ts REAL, summary TEXT)")

def load_state(sid:str)->Optional[Dict[str,Any]]:
    with sqlite3.connect(DB_FILE) as con:
        row = con.execute("SELECT state FROM sessions WHERE id=?",(sid,)).fetchone()
        return json.loads(row[0]) if row else None
def save_state(sid:str, st:Dict[str,Any]):
    with sqlite3.connect(DB_FILE) as con:
        con.execute("REPLACE INTO sessions (id,state) VALUES (?,?)",(sid,json.dumps(st)))

# ─────────────────────────── Sessions ────────────────────────────────────
def get_session(sid:str)->Dict[str,Any]:
    st=load_state(sid)
    if st: return st
    st={"id":sid,
        "history":[{"role":"system","content":"SYSTEM_PROMPT"}],
        "stage":Stage.THEME_SELECTION.value,
        "plan":{"themes":[], "topics":{}, "qa_items":[]},
        "qa_queue":[], "current_question":None}
    save_state(sid,st); return st

# ───────────────────── Guardrail-sentiment ───────────────────────────────
def detect_sentiment(text:str)->Literal["ok","needs_menu"]:
    vague=["weet niet","geen idee","idk"]
    return "needs_menu" if len(text.split())<4 or any(v in text.lower() for v in vague) else "ok"

# ─────────────────────────── Tools ───────────────────────────────────────
@function_tool
def get_plan_status(session_id:str)->str:
    st=get_session(session_id); return json.dumps({"stage":st["stage"],"plan":st["plan"]})

@function_tool
def offer_choices(session_id:str,choice_type:Literal['themes','topics'],theme_context:Optional[str]=None)->str:
    if choice_type=='themes': return ", ".join(t["name"] for t in DEFAULT_THEMES)
    if choice_type=='topics': return ", ".join(DEFAULT_TOPICS_PER_THEME.get(theme_context,[])) if theme_context else "Error"
    return "Error"

@function_tool
def add_item(session_id:str,item_type:Literal['theme','topic'],name:str,
             theme_context:Optional[str]=None,is_custom:bool=False)->str:
    st=get_session(session_id); plan=st["plan"]
    if item_type=='theme':
        if len(plan["themes"])>=6: return "Error:max 6 thema's."
        if name not in [t["name"] for t in plan["themes"]]:
            plan["themes"].append({"name":name,"is_custom":is_custom}); st["stage"]=Stage.TOPIC_SELECTION.value
    elif item_type=='topic' and theme_context:
        plan["topics"].setdefault(theme_context,[])
        if len(plan["topics"][theme_context])<4 and name not in plan["topics"][theme_context]:
            plan["topics"][theme_context].append(name)
    save_state(session_id,st); return "ok"

@function_tool
def remove_item(session_id:str,item_type:Literal['theme','topic'],name:str,
                theme_context:Optional[str]=None)->str:
    st=get_session(session_id); plan=st["plan"]
    if item_type=='theme':
        plan["themes"]=[t for t in plan["themes"] if t["name"]!=name]; plan["topics"].pop(name,None)
    elif item_type=='topic' and theme_context:
        plan["topics"].get(theme_context,[]).remove(name)
    save_state(session_id,st); return "ok"

@function_tool
def update_item(session_id:str,item_type:Literal['theme','topic'],
                old_name:str,new_name:str,theme_context:Optional[str]=None)->str:
    st=get_session(session_id); plan=st["plan"]
    if item_type=='theme':
        for t in plan["themes"]:
            if t["name"]==old_name: t["name"]=new_name
        if old_name in plan["topics"]:
            plan["topics"][new_name]=plan["topics"].pop(old_name)
    elif item_type=='topic' and theme_context:
        plan["topics"][theme_context]=[new_name if x==old_name else x for x in plan["topics"].get(theme_context,[])]
        for qa in plan["plan"]["qa_items"]:
            if qa["theme"]==theme_context and qa["topic"]==old_name:
                qa["topic"]=new_name
    save_state(session_id,st); return "ok"

# ─── QA-tools ───
@function_tool
def start_qa_session(session_id:str)->str:
    st=get_session(session_id)
    if st["stage"]==Stage.QA_SESSION.value: return "We zijn al in de vragenronde."
    st["qa_queue"]=[]
    for theme,topics in st["plan"]["topics"].items():
        for t in topics:
            st["qa_queue"].append({"theme":theme,"topic":t,"question":f"Wat zijn je wensen rondom {t}?"})
    st["stage"]=Stage.QA_SESSION.value; save_state(session_id,st); return "ok"

@function_tool
def get_next_question(session_id:str)->str:
    st=get_session(session_id)
    if not st["qa_queue"]:
        st["stage"]=Stage.COMPLETED.value; save_state(session_id,st); return "Alle vragen zijn beantwoord!"
    st["current_question"]=st["qa_queue"].pop(0); save_state(session_id,st)
    return f"Vraag over '{st['current_question']['topic']}': {st['current_question']['question']}"

@function_tool
def log_answer(session_id:str,answer:str)->str:
    st=get_session(session_id); cq=st["current_question"]
    if not cq: return "Error:Geen actieve vraag."
    st["plan"]["qa_items"].append({**cq,"answer":answer}); st["current_question"]=None
    save_state(session_id,st); return "Antwoord opgeslagen."

@function_tool
def update_question(session_id:str,old_question:str,new_question:str)->str:
    st=get_session(session_id)
    st["qa_queue"]=[{**q,"question":new_question} if q["question"]==old_question else q for q in st["qa_queue"]]
    for qa in st["plan"]["qa_items"]:
        if qa["question"]==old_question: qa["question"]=new_question
    if st["current_question"] and st["current_question"]["question"]==old_question:
        st["current_question"]["question"]=new_question
    save_state(session_id,st); return "ok"

@function_tool
def remove_question(session_id:str,question_text:str)->str:
    st=get_session(session_id)
    st["qa_queue"]=[q for q in st["qa_queue"] if q["question"]!=question_text]
    st["plan"]["qa_items"]=[q for q in st["plan"]["qa_items"] if q["question"]!=question_text]
    save_state(session_id,st); return "ok"

@function_tool
def update_answer(session_id:str,question_text:str,new_answer:str)->str:
    st=get_session(session_id)
    for qa in st["plan"]["qa_items"]:
        if qa["question"]==question_text: qa["answer"]=new_answer
    save_state(session_id,st); return "ok"

# ─── Proactieve-Gids tools ───
@function_tool
def find_web_resources(topic:str,depth:Literal['brief','diep']='brief')->str:
    return json.dumps({"summary":f"Samenvatting over {topic}",
                       "links":[f"https://example.com/{topic}"]})

@function_tool
def vergelijk_opties(options:List[str])->str:
    return f"Vergelijking: {', '.join(options)}"

@function_tool
def geef_denkvraag(theme:str)->str:
    return f"Hoe voel je je over {theme}?"

@function_tool
def find_external_organization(keyword:str)->str:
    return json.dumps({"organisaties":[f"{keyword} support groep"]})

@function_tool
def check_onbeantwoorde_punten(session_id:str)->str:
    st=get_session(session_id); return json.dumps({"missing":st["qa_queue"]})

@function_tool
def genereer_plan_tekst(session_id:str,format:Literal['markdown','plain']='markdown')->str:
    st=get_session(session_id)
    return "# Geboorteplan\n"+json.dumps(st["plan"],ensure_ascii=False,indent=2)

@function_tool
def present_tool_choices(choices:str)->str:
    """Geeft JSON-string terug zodat de frontend dit als quick-replies kan tonen."""
    return choices

@function_tool
def save_plan_summary(session_id:str)->str:
    st=get_session(session_id)
    prompt="Vat dit geboorteplan samen in 5 bulletpoints:"+json.dumps(st["plan"],ensure_ascii=False)
    try:
        resp=client.chat.completions.create(model="gpt-3.5-turbo",messages=[{"role":"user","content":prompt}])
        summary=resp.choices[0].message.content.strip()
    except Exception as e:
        summary=f"(samenvatting mislukt: {e})"
    with sqlite3.connect(DB_FILE) as con:
        con.execute("INSERT INTO summaries (id,ts,summary) VALUES (?,?,?)",(session_id,time.time(),summary))
    return "summary_saved"

SYSTEM_PROMPT = """
# MAE - GEBOORTEPLAN ASSISTENT

## IDENTITEIT
Je bent Mae, een warme, empathische en proactieve assistent die zwangere vrouwen helpt bij het maken van een persoonlijk geboorteplan. Je bent flexibel, luistert goed en past je aan de wensen van de gebruiker aan.

## WERKWIJZE: VIER FASEN
Gebruik `get_plan_status()` om te controleren in welke fase je bent:

### 1. THEME_SELECTION
- **Doel**: Gebruiker kiest maximaal 6 hoofdthema's
- **Actie**: Gebruik `offer_choices(choice_type='themes')` om standaardthema's te tonen
- **Flexibiliteit**: Gebruiker mag eigen thema's toevoegen (is_custom=True)
- **Tools**: `add_item(item_type='theme')`, `remove_item(item_type='theme')`, `update_item(item_type='theme')`

### 2. TOPIC_SELECTION  
- **Doel**: Voor elk gekozen thema selecteert gebruiker maximaal 4 onderwerpen
- **Actie**: Gebruik `offer_choices(choice_type='topics', theme_context='THEMA_NAAM')`
- **Flexibiliteit**: Gebruiker mag eigen onderwerpen toevoegen
- **Tools**: `add_item(item_type='topic')`, `remove_item(item_type='topic')`, `update_item(item_type='topic')`

### 3. QA_SESSION
- **Start**: Roep `start_qa_session()` aan om vragenronde te beginnen
- **Proces**: 
  1. `get_next_question()` - Stel volgende vraag
  2. Wacht op antwoord van gebruiker
  3. `log_answer(answer='...')` - Sla antwoord op
  4. Herhaal tot alle vragen beantwoord zijn
- **Proactieve gids**: Bij korte/vage antwoorden, bied hulp aan via `present_tool_choices()`
- **QA Management**: `update_question()`, `remove_question()`, `update_answer()`

### 4. COMPLETED
- **Resultaat**: Alle vragen beantwoord, plan compleet
- **Actie**: `save_plan_summary()` wordt automatisch aangeroepen
- **Export**: `genereer_plan_tekst()` voor tekstversie

## COMMUNICATIE PRINCIPES

### Bevestiging Vereist
**REGEL**: Vraag altijd bevestiging voordat je wijzigingen doorvoert:
- "Zal ik het thema 'Pijnstilling' toevoegen?"
- "Wil je dat ik 'Epiduraal' als onderwerp toevoeg?"

**UITZONDERING**: Bij expliciete bevestigingen (zoals "Ja", "Doe maar", "Voeg toe", "Akkoord", "Prima", "Oké", "Ga door", "Klopt", etc.) voer direct uit zonder opnieuw te vragen.

### Proactieve Ondersteuning
Wanneer gebruiker kort/vaag antwoordt, bied contextspecifieke hulp:
```python
menu_options = {
    "choices": [
        {"label": "Meer informatie", "tool": "find_web_resources", "args": {"topic": "RELEVANT_TOPIC"}},
        {"label": "Vergelijk opties", "tool": "vergelijk_opties", "args": {"options": ["optie1", "optie2"]}},
        {"label": "Reflectievraag", "tool": "geef_denkvraag", "args": {"theme": "HUIDIG_THEMA"}},
        {"label": "Externe hulp", "tool": "find_external_organization", "args": {"keyword": "ZOEKTERM"}}
    ]
}
```

## FLEXIBILITEIT REGELS

### Gebruiker Heeft Controle
- Gebruiker mag altijd terug naar vorige fasen
- Thema's en onderwerpen mogen aangepast worden tijdens QA_SESSION
- Vragen mogen overgeslagen of aangepast worden
- Plan mag op elk moment geëxporteerd worden

### Aanpassingen Tijdens Sessie
- `update_item(item_type, old_name, new_name, theme_context=None)` - Wijzig thema/onderwerp namen
- `update_question(old_question, new_question)` - Pas vraagstelling aan
- `update_answer(question_text, new_answer)` - Wijzig eerder gegeven antwoord
- `remove_question(question_text)` - Verwijder vraag uit sessie
- `remove_item(item_type, name, theme_context=None)` - Verwijder thema/onderwerp

### Overzicht & Planning
- `check_onbeantwoorde_punten()` - Overzicht openstaande vragen
- `genereer_plan_tekst(format='markdown')` - Toon tussentijds overzicht
- `save_plan_summary()` - Sla samenvatting op (automatisch bij completion)

## ALLE BESCHIKBARE TOOLS (18 TOTAAL)

### Core Navigatie & Status
- `get_plan_status()` - Controleer huidige fase en planstatus
- `offer_choices(choice_type, theme_context=None)` - Toon beschikbare opties

### Plan Beheer
- `add_item(item_type, name, theme_context=None, is_custom=False)` - Voeg thema/topic toe
- `remove_item(item_type, name, theme_context=None)` - Verwijder thema/topic  
- `update_item(item_type, old_name, new_name, theme_context=None)` - Wijzig naam

### QA Sessie Management
- `start_qa_session()` - Begin vragenronde
- `get_next_question()` - Stel volgende vraag
- `log_answer(answer)` - Sla antwoord op
- `update_question(old_question, new_question)` - Wijzig vraag
- `remove_question(question_text)` - Verwijder vraag
- `update_answer(question_text, new_answer)` - Wijzig antwoord

### Proactieve Ondersteuning  
- `find_web_resources(topic, depth='brief')` - Zoek informatie over onderwerp
- `vergelijk_opties(options)` - Vergelijk verschillende opties
- `geef_denkvraag(theme)` - Stel reflectievraag over thema
- `find_external_organization(keyword)` - Zoek externe organisaties

### Overzicht & Export
- `check_onbeantwoorde_punten()` - Toon openstaande vragen
- `genereer_plan_tekst(format='markdown')` - Genereer plantekst
- `present_tool_choices(choices)` - Toon menu-opties aan gebruiker
- `save_plan_summary()` - Bewaar samenvatting (auto bij completion)

## GESPREKSVOERING

### Toon & Stijl
- **Warm en ondersteunend**: "Wat fijn dat je bezig bent met je geboorteplan!"
- **Duidelijke instructies**: "Ik toon je nu de beschikbare thema's..."
- **Flexibel**: "We kunnen altijd terug om iets aan te passen"
- **Proactief**: "Wil je dat ik wat meer uitleg geef over...?"

### Voorbeelden Formulering
- ✅ "Zal ik 'Pijnstilling' toevoegen aan je thema's?"
- ✅ "Wil je meer informatie over epiduraal versus natuurlijke pijnverlichting?"
- ✅ "Laten we eerst kijken naar de onderwerpen voor 'Ondersteuning'"
- ❌ "Voeg pijnstilling toe" (zonder bevestiging)

### Foutafhandeling
- Tools kunnen "Error" terugeven - leg uit wat er mis ging
- Maximaal 6 thema's, maximaal 4 onderwerpen per thema
- Valideer altijd invoer voordat je tools aanroept

## ADVANCED FEATURES

### Contextuele Intelligentie
- Herken wanneer gebruiker twijfelt → bied `geef_denkvraag()`
- Merk verwarring op → gebruik `find_web_resources()`
- Gebruiker wil vergelijken → roep `vergelijk_opties()` aan

### Sessie Management
- Gebruik altijd correcte `session_id` parameter
- `get_plan_status()` voor overzicht huidige staat
- Bewaar alle wijzigingen direct via tools

Je bent een deskundige gids die het geboorteplan-proces soepel en ondersteunend begeleidt, waarbij de gebruiker altijd de controle behoudt.
"""

# ─────────────────────── Tool-registratie ───────────────────────────────
tool_funcs=[get_plan_status, offer_choices, add_item, remove_item, update_item,
            start_qa_session, get_next_question, log_answer,
            update_question, remove_question, update_answer,
            find_web_resources, vergelijk_opties, geef_denkvraag, find_external_organization,
            check_onbeantwoorde_punten, genereer_plan_tekst,
            present_tool_choices, save_plan_summary]
tools_schema=[get_schema(t) for t in tool_funcs]
TOOL_MAP={t.openai_schema['function']['name']:t for t in tool_funcs}

# ───────────────────────── Agent-loop ────────────────────────────────────
def run_main_agent_loop(sid:str)->str:
    st=get_session(sid)

    # Guardrail vóór de OpenAI-call
    if detect_sentiment(st["history"][-1]["content"])=="needs_menu":
        menu=json.dumps({"choices":[
            {"label":"Meer info","tool":"find_web_resources","args":{"topic":"pijnstilling"}},
            {"label":"Vergelijk opties","tool":"vergelijk_opties","args":{"options":["epiduraal","badbevalling"]}},
            {"label":"Reflectie","tool":"geef_denkvraag","args":{"theme":"pijnstilling"}}
        ]})
        return present_tool_choices(session_id=sid,choices=menu)

    for _ in range(5):  # MAX_TURNS
        resp=client.chat.completions.create(model=MODEL_CHOICE,
                messages=st["history"], tools=tools_schema, tool_choice="auto")
        msg=resp.choices[0].message
        st["history"].append(msg.model_dump(exclude_unset=True))
        if not msg.tool_calls:
            return msg.content or "(geen antwoord)"
        for call in msg.tool_calls:
            fn=TOOL_MAP[call.function.name]
            result=fn(session_id=sid, **json.loads(call.function.arguments))
            st["history"].append({"tool_call_id":call.id,"role":"tool",
                                  "name":call.function.name,"content":str(result)})
            save_state(sid,st)
    return "(max turns bereikt)"

# ───────────────────── build_payload (origineel) ─────────────────────────
def build_payload(st:Dict[str,Any], reply:str)->Dict[str,Any]:
    payload={"assistant_reply":reply,
             "session_id":st["id"],"stage":st["stage"],"plan":st["plan"],
             "suggested_replies":[]}
    triggers=["akkoord?","is dat correct?","wil je dat ik","zal ik"]
    if any(t in reply.lower() for t in triggers):
        payload["suggested_replies"]=["Ja","Nee"]
    return payload

# ───────────────────────── Flask-routes ─────────────────────────────────
@app.post("/agent")
def agent_route():
    body=request.get_json(force=True) or {}
    msg=body.get("message","").strip()
    sid=body.get("session_id") or str(uuid.uuid4())
    if not msg:
        return jsonify({"assistant_reply":"(leeg bericht)","session_id":sid})
    st=get_session(sid); st["history"].append({"role":"user","content":msg}); save_state(sid,st)
    reply=run_main_agent_loop(sid)
    st["history"].append({"role":"assistant","content":reply}); save_state(sid,st)
    if st["stage"]==Stage.COMPLETED.value:
        save_plan_summary(session_id=sid)
    return jsonify(build_payload(st,reply))

@app.get("/export/<sid>")
def export_json(sid):
    st=load_state(sid); 
    if not st: abort(404)
    path=ROOT/f"geboorteplan_{sid}.json"
    path.write_text(json.dumps(st["plan"],ensure_ascii=False,indent=2),"utf-8")
    return send_file(path,as_attachment=True,download_name=path.name)

# iframe- en statische bestanden (oorspronkelijk)
@app.route("/", defaults={"path":""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path=="iframe":
        return render_template("iframe_page.html",
            backend_url=os.getenv("RENDER_EXTERNAL_URL","http://127.0.0.1:10000"))
    full=os.path.join(app.static_folder,path)
    if path and os.path.exists(full):
        return send_from_directory(app.static_folder,path)
    return send_from_directory(app.static_folder,"index.html")

if __name__=="__main__":
    app.run("0.0.0.0", int(os.getenv("PORT",10000)), debug=True)
