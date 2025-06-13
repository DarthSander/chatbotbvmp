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
    """Biedt een lijst van keuzes voor thema's of onderwerpen."""
    if choice_type == 'themes':
        return ", ".join(t["name"] for t in DEFAULT_THEMES)

    if choice_type == 'topics':
        if not theme_context:
            return "Error: theme_context is verplicht voor het opvragen van topics."
        
        # Case-insensitive lookup
        for key, topics in DEFAULT_TOPICS_PER_THEME.items():
            if key.lower() == theme_context.lower():
                return ", ".join(topics)
        
        # Geef geen lege string terug, maar een expliciete instructie voor de agent.
        return "Geen standaard onderwerpen gevonden. Bedenk zelf 4 tot 5 relevante suggesties voor dit thema."

    return "Error: Ongeldig choice_type."

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

# --- QA-tools ---
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

# --- Proactieve-Gids tools ---
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

# *** NIEUWE TOOL TOEGEVOEGD ***
@function_tool
def propose_quick_replies(session_id: str, replies: List[str]) -> str:
    """Stel een lijst van quick reply-knoppen voor die de frontend kan tonen. Gebruik dit voor ja/nee-vragen of om de gebruiker te helpen kiezen."""
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
- De gebruiker heeft altijd de controle en mag op elk moment terug naar een vorige fase of iets aanpassen.
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
        
        # Als de agent alleen tekst teruggeeft, is de beurt voorbij
        if not msg.tool_calls:
            return msg.content or "(geen antwoord)"

        # Als de agent een tool aanroept
        tool_results = []
        for call in msg.tool_calls:
            fn = TOOL_MAP.get(call.function.name)
            if not fn:
                result = f"Error: tool '{call.function.name}' not found."
            else:
                try:
                    args = json.loads(call.function.arguments)
                    result = fn(session_id=sid, **args)
                except Exception as e:
                    result = f"Error executing tool: {e}"
            
            tool_results.append({
                "tool_call_id": call.id,
                "role": "tool",
                "name": call.function.name,
                "content": str(result)
            })

        # Als de laatste tool-aanroep propose_quick_replies was, is de beurt voorbij voor de agent
        # De payload-functie handelt de rest af. We retourneren de vorige tekst van de agent.
        if msg.tool_calls[-1].function.name == 'propose_quick_replies':
            # Zoek de laatste 'assistant' boodschap die geen tool call was.
            for i in range(len(st["history"]) - 2, -1, -1):
                if st["history"][i].get("role") == "assistant" and st["history"][i].get("content"):
                    return st["history"][i]["content"]
            return "(actie wordt voorbereid)" # Fallback

        st["history"].extend(tool_results)
        save_state(sid, st)

    return "(max turns bereikt)"

# ───────────────────── build_payload (VOLLEDIG VERVANGEN) ────────────────
def build_payload(st:Dict[str,Any], reply:str)->Dict[str,Any]:
    """Stelt de JSON payload samen die naar de frontend wordt gestuurd."""
    payload = {
        "assistant_reply": reply,
        "session_id": st["id"],
        "stage": st["stage"],
        "plan": st["plan"],
        "suggested_replies": []
    }

    # Kijk naar de laatste berichten in de geschiedenis om de tool call te vinden
    if not st["history"]:
        return payload

    # De tool_calls zitten in het 'assistant' bericht
    last_message = st["history"][-1]
    if (last_message and
        last_message.get("role") == "assistant" and
        last_message.get("tool_calls")):
        
        # Zoek specifiek naar een aanroep van propose_quick_replies
        for tool_call in last_message["tool_calls"]:
            if tool_call.get("function", {}).get("name") == "propose_quick_replies":
                try:
                    # Haal de 'replies' argumenten op uit de tool call
                    args = json.loads(tool_call["function"]["arguments"])
                    payload["suggested_replies"] = args.get("replies", [])
                    # Stop met zoeken zodra we het gevonden hebben
                    break
                except (json.JSONDecodeError, TypeError):
                    # Argumenten waren geen valide JSON, negeer
                    pass
    
    return payload

# ───────────────────────── Flask-routes ─────────────────────────────────
@app.post("/agent")
def agent_route():
    body=request.get_json(force=True) or {}
    msg=body.get("message","").strip()
    sid=body.get("session_id") or str(uuid.uuid4())
    if not msg:
        return jsonify({"assistant_reply":"(leeg bericht)","session_id":sid})
    
    st=get_session(sid)
    # De System Prompt staat nu in de history, we hoeven hem niet opnieuw toe te voegen.
    st["history"].append({"role":"user","content":msg})
    save_state(sid,st)

    reply=run_main_agent_loop(sid)

    # De agent loop voegt nu zelf de history toe, dus we hoeven dat hier niet dubbel te doen.
    # We laden de laatste state om de payload te bouwen.
    final_st = get_session(sid)

    if final_st["stage"]==Stage.COMPLETED.value:
        save_plan_summary(session_id=sid)
    
    return jsonify(build_payload(final_st,reply))

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
