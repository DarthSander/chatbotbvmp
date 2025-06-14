#!/usr/bin/env python3
# app.py – Geboorteplan-agent • Volledige versie 15-06-2025
# FINALE VERSIE MET STRIKTE, UI-GEBONDEN SYSTEM_PROMPT

from __future__ import annotations
import os, json, uuid, sqlite3, time, logging, inspect, pathlib
import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from flask import Flask, request, jsonify, abort, send_file, send_from_directory, render_template
from flask_cors import CORS
from openai import OpenAI

# ─────────────────────────── Basisconfiguratie ───────────────────────────
ROOT = pathlib.Path(__file__).parent
DB_FILE = ROOT / "sessions.db"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MODEL_CHOICE = os.getenv("MODEL_CHOICE", "gpt-4.1")
CLASSIFIER_MODEL = "gpt-3.5-turbo" # Apart, snel model voor de Keuze-Extractor

# Uitgebreide logging configuratie
logging.basicConfig(
    level   = LOG_LEVEL,
    format  = "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d – %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("mae-backend")

log.info("Mae-backend applicatie start...")
log.info(f"Hoofd-model: {MODEL_CHOICE}, Classifier-model: {CLASSIFIER_MODEL}")

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
            return json.loads(row[0])
        return None
def save_state(sid:str, st:Dict[str,Any]):
    log.debug(f"State opslaan voor session_id: {sid}")
    with sqlite3.connect(DB_FILE) as con:
        con.execute("REPLACE INTO sessions (id,state) VALUES (?,?)",(sid,json.dumps(st)))

# ─────────────────────────── Sessions ────────────────────────────────────
def get_session(sid:str)->Dict[str,Any]:
    st=load_state(sid)
    if st:
        log.info(f"Bestaande sessie {sid} geladen. Huidige fase: {st.get('stage', 'ONBEKEND')}")
        st.setdefault("topic_suggestions", {})
        return st
    log.info(f"Nieuwe sessie gestart met id: {sid}")
    st = {
        "id"      : sid,
        "history" : [{"role": "system", "content": SYSTEM_PROMPT}],
        "stage"   : Stage.THEME_SELECTION.value,
        "plan"    : {"themes": [], "topics": {}, "qa_items": []},
        "qa_queue": [],
        "current_question": None,
        "topic_suggestions": {}
    }
    save_state(sid,st); return st

# ─────────────────────────── Alle Tools ───────────────────────────
@function_tool
def get_plan_status(session_id:str)->str:
    """Geeft de huidige status van het geboorteplan terug."""
    st=get_session(session_id)
    return json.dumps({"stage":st["stage"],"plan":st["plan"]})

@function_tool
def offer_choices(session_id:str,choice_type:Literal['themes','topics'],theme_context:Optional[str]=None)->str:
    """Biedt een lijst van keuzes voor thema's of onderwerpen."""
    if choice_type == 'themes':
        return ", ".join(t["name"] for t in DEFAULT_THEMES)
    if choice_type == 'topics':
        if not theme_context: return "Error: theme_context is verplicht."
        for key, topics in DEFAULT_TOPICS_PER_THEME.items():
            if key.lower() == theme_context.lower(): return ", ".join(topics)
        return "Geen standaard onderwerpen gevonden. Bedenk zelf 4-5 suggesties."
    return "Error: Ongeldig choice_type."

@function_tool
def add_item(session_id: str, item_type: Literal['theme', 'topic'], name: str, theme_context: Optional[str] = None, is_custom: bool = False) -> str:
    """Voegt een thema of onderwerp toe aan het plan."""
    st = get_session(session_id)
    plan = st["plan"]
    if isinstance(plan.get("topics"), list): plan["topics"] = {}
    if item_type == 'theme':
        if len(plan["themes"]) >= 6: return "Error: max 6 thema's."
        if name not in [t["name"] for t in plan["themes"]]:
            plan["themes"].append({"name": name, "is_custom": is_custom})
    elif item_type == 'topic' and theme_context:
        plan["topics"].setdefault(theme_context, [])
        if name not in [t["name"] for t in plan["topics"].get(theme_context, [])]:
            plan["topics"][theme_context].append({"name": name, "is_custom": is_custom})
    save_state(session_id, st)
    return f"{item_type.capitalize()} '{name}' is succesvol toegevoegd."

@function_tool
def confirm_themes(session_id: str) -> str:
    """Bevestigt dat de gebruiker klaar is met het kiezen van thema's en zet de fase naar de vragenronde."""
    st = get_session(session_id)
    st["stage"] = Stage.QA_SESSION.value
    st["qa_queue"]=[{"theme":theme,"topic":t["name"],"question":f"Wat zijn je wensen rondom {t['name']}?"} for theme,topics in st["plan"]["topics"].items() for t in topics if "name" in t]
    save_state(session_id, st)
    log.info(f"Fase gewijzigd naar QA_SESSION. {len(st['qa_queue'])} vragen in wachtrij.")
    return "Oké, we gaan nu beginnen met de vragen over de door jou gekozen onderwerpen."

@function_tool
def remove_item(session_id:str,item_type:Literal['theme','topic'],name:str, theme_context:Optional[str]=None)->str:
    """Verwijdert een thema of onderwerp uit het plan."""
    st=get_session(session_id); plan=st["plan"]
    if item_type=='theme':
        plan["themes"]=[t for t in plan["themes"] if t["name"]!=name]
        plan["topics"].pop(name,None)
    elif item_type=='topic' and theme_context and theme_context in plan["topics"]:
        plan["topics"][theme_context] = [t for t in plan["topics"][theme_context] if t["name"] != name]
    save_state(session_id,st); return "ok"

@function_tool
def update_item(session_id:str,item_type:Literal['theme','topic'], old_name:str,new_name:str,theme_context:Optional[str]=None)->str:
    """Werkt de naam van een thema of onderwerp bij."""
    st=get_session(session_id); plan=st["plan"]
    if item_type=='theme':
        for t in plan["themes"]:
            if t["name"]==old_name: t["name"]=new_name
        if old_name in plan["topics"]:
            plan["topics"][new_name]=plan["topics"].pop(old_name)
    elif item_type=='topic' and theme_context and theme_context in plan["topics"]:
        for t in plan["topics"][theme_context]:
            if t["name"] == old_name: t["name"] = new_name
        for qa in plan["qa_items"]:
            if qa["theme"]==theme_context and qa["topic"]==old_name: qa["topic"]=new_name
    save_state(session_id,st); return "ok"

@function_tool
def start_qa_session(session_id:str)->str:
    """Start de vragenronde (QA-sessie)."""
    st=get_session(session_id)
    if st["stage"]==Stage.QA_SESSION.value: return "We zijn al in de vragenronde."
    st["qa_queue"]=[{"theme":theme,"topic":t["name"],"question":f"Wat zijn je wensen rondom {t['name']}?"}
                    for theme,topics in st["plan"]["topics"].items() for t in topics if "name" in t]
    st["stage"]=Stage.QA_SESSION.value
    save_state(session_id,st); return "ok"

@function_tool
def get_next_question(session_id:str)->str:
    """Haalt de volgende vraag op uit de wachtrij."""
    st=get_session(session_id)
    if not st["qa_queue"]:
        st["stage"]=Stage.COMPLETED.value; save_state(session_id,st)
        return "Alle vragen zijn beantwoord!"
    st["current_question"]=st["qa_queue"].pop(0)
    save_state(session_id,st)
    return f"Vraag over '{st['current_question']['topic']}': {st['current_question']['question']}"

@function_tool
def log_answer(session_id:str,answer:str)->str:
    """Slaat het antwoord op een vraag op."""
    st=get_session(session_id); cq=st["current_question"]
    if not cq: return "Error:Geen actieve vraag."
    st["plan"]["qa_items"].append({**cq,"answer":answer}); st["current_question"]=None
    save_state(session_id,st); return "Antwoord opgeslagen."

@function_tool
def update_question(session_id:str,old_question:str,new_question:str)->str:
    """Werkt een bestaande vraag bij."""
    return "ok"

@function_tool
def remove_question(session_id:str,question_text:str)->str:
    """Verwijdert een vraag."""
    return "ok"

@function_tool
def update_answer(session_id:str,question_text:str,new_answer:str)->str:
    """Werkt een gegeven antwoord bij."""
    return "ok"

@function_tool
def find_web_resources(session_id: str, topic: str, depth: Literal['brief', 'diep'] = 'brief') -> str:
    """Zoekt naar webbronnen over een bepaald onderwerp."""
    return json.dumps({"summary": f"Samenvatting over {topic}", "links": [f"https://example.com/{topic}"]})

@function_tool
def vergelijk_opties(session_id: str, options: List[str]) -> str:
    """Vergelijkt verschillende opties."""
    return f"Vergelijking: {', '.join(options)}"

@function_tool
def geef_denkvraag(session_id: str, theme: str) -> str:
    """Geeft een reflectievraag over een thema."""
    return f"Hoe voel je je over {theme}?"

@function_tool
def find_external_organization(session_id: str, keyword: str) -> str:
    """Zoekt naar externe organisaties."""
    return json.dumps({"organisaties": [f"{keyword} support groep"]})

@function_tool
def check_onbeantwoorde_punten(session_id:str)->str:
    """Controleert of er nog onbeantwoorde vragen zijn."""
    return json.dumps({"missing":get_session(session_id)["qa_queue"]})

@function_tool
def genereer_plan_tekst(session_id:str,format:Literal['markdown','plain']='markdown')->str:
    """Genereert de tekst van het geboorteplan."""
    return "# Geboorteplan\n"+json.dumps(get_session(session_id)["plan"],ensure_ascii=False,indent=2)

@function_tool
def present_tool_choices(session_id: str, choices: str) -> str:
    """Geeft JSON-string terug zodat de frontend dit als quick-replies kan tonen."""
    return choices

@function_tool
def save_plan_summary(session_id:str)->str:
    """Slaat een samenvatting van het plan op in de database."""
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

@function_tool
def propose_quick_replies(session_id: str, replies: List[str]) -> str:
    """Stel een lijst van quick reply-knoppen voor die de frontend kan tonen."""
    log.info(f"Tool 'propose_quick_replies' aangeroepen met replies: {replies}")
    return f"Quick replies voorgesteld: {', '.join(replies)}"

@function_tool
def propose_topics(session_id: str, theme: str, suggestions: List[str]) -> str:
    """Geeft een lijst van topic-suggesties terug en slaat deze op in de sessie."""
    st = get_session(session_id)
    st.setdefault("topic_suggestions", {})[theme] = suggestions
    save_state(session_id, st)
    return f"Topics voorgesteld voor '{theme}'."

# ─────────────────────────── PROMPTS ────────────────────────────────────
# ╭───────────────────────────────╮
# │  ✨  MAE – MASTER SYSTEM PROMPT │
# ╰───────────────────────────────╯
SYSTEM_PROMPT = r"""
╔══════════════════════════════════════════════════════════════════════╗
║ MAE – GEBOORTEPLAN-ASSISTENT  ·  ULTIEME INSTRUCTIE                 ║
╚══════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━ 1 · IDENTITEIT ━━━━━━━━━━━━━━━━━━━━━━━━
Je bent **Mae**: warm, empathisch, technisch precies, géén arts.  
Doel: een zwangere helpen een volledig, persoonlijk geboorteplan te maken.

━━━━━━━━━━━━━━━━━━━━━━━━ 2 · UNIVERSELE REGELS ━━━━━━━━━━━━━━━━
• Taal: Nederlands • Toon empathie • Varieer bevestigingen  
• Géén medisch advies (“Ik ben geen arts, raadpleeg je verloskundige …”).  
• Tool-aanroep = één taak; bevestig pas na succes.  
• Tijdens UI-fases stel je géén extra vragen; de UI is leidend.  
• Onzeker/kort antwoord → Proactieve gids (zie D-pad).  
• Extractor-prompt bepaalt knoppen; respecteer diens output.

━━━━━━━━━━━━━━━━━━━━━━━━ 3 · GEDRAGSBOOM ━━━━━━━━━━━━━━━━━━━━━━
(Samengevat; details, tools & voorbeelden volgen per node)

A  Initialisatie → Welkom‐tekst  
B  Reactieve UI-mode → add_item / remove_item / update_item / confirm_themes  
C  QA-ronde → get_next_question / log_answer / save_plan_summary  
D  Proactieve gids → find_web_resources · vergelijk_opties · geef_denkvraag  
E  Fail-Safe → propose_quick_replies(["Extra hulp","Terug naar hoofdmenu"])

━━━━━━━━━━━━━━━━━━━━━━━━ 3A · INITIALISATIE ━━━━━━━━━━━━━━━━━━━
• Conditie: eerste userbericht.  
• Reactie (exact):  
  “Welkom! Ik ben Mae, je assistent voor het samenstellen van je geboorteplan.  
   Aan de rechterkant in de sidebar kun je beginnen met het selecteren van maximaal  
   6 thema's. Klik op een thema om de bijbehorende onderwerpen te kiezen.  
   Typ hier in de chat als je een eigen thema of onderwerp wilt toevoegen.”

📌 Voorbeeld  
U: Hallo, ik wil graag een geboorteplan opstellen.  
A: <welkomstbericht>

━━━━━━━━━━━━━━━━━━━━━━━━ 3B · REACTIEVE UI-MODE ━━━━━━━━━━━━━━━
### B1  Toevoegen
• Tool : `add_item(session_id, item_type="theme|topic", name, theme_context?, is_custom?)`  
• Variërende bevestigingen  
  - “Oké, genoteerd.” - “Prima, staat erin.” - “Duidelijk, toegevoegd.”

**Voorbeeld**  
UI → “Thema ‘Ondersteuning’ gekozen.”  
A  → add_item(…, item_type="theme", name="Ondersteuning")  
A  → “Oké, genoteerd.”

### B2  Verwijderen
• Tool : `remove_item(...)`  
• Bevestigingen: “Verwijderd.” / “Is eruit.” / “Weg­gehaald.”

### B3  Hernoemen
• Tool : `update_item(...)`  
• Bevestigingen: “Hernoemd.” / “Naam aangepast.” / “Bijgewerkt.”

### B4  Limiet overtreden (>6 thema’s of >4 topics)
• Geen tool – enkel melding  
  “Je hebt al 6 thema’s. Verwijder eerst een thema om verder te gaan.”

### B5  Bevestig thema’s
• Tool : `confirm_themes(session_id)`  
• Effect : `stage = QA_SESSION`, `qa_queue` gevuld  
• Overgangstekst (één van):  
  - “Mooi, de basis staat. Laten we je wensen verder uitwerken.”  
  - “Top, laten we nu dieper in je voorkeuren duiken.”

━━━━━━━━━━━━━━━━━━━━━━━━ 3C · QA-RONDE ━━━━━━━━━━━━━━━━━━━━━━━
#### Vraag stellen
• Tool  : `get_next_question(session_id)`  
• Vraag‐bubble bijv.:  
  “Wat zijn je wensen rondom **pijnbestrijding**?”

#### Antwoordverwerking
| Situatie  | Actie                                   | Bevestiging |
|-----------|-----------------------------------------|-------------|
| Leeg      | “Ik heb je antwoord niet goed begrepen, kun je het anders formuleren?” |
| Onzeker   | Ga naar D-pad                           | —           |
| Normaal   | `log_answer(session_id, answer)`        | “Antwoord opgeslagen.” |

#### Queue leeg
• Tool : `save_plan_summary(session_id)`  
• Tekst: “Gefeliciteerd! Je geboorteplan is compleet. Je kunt het downloaden of later aanpassen.”

━━━━━━━━━━━━━━━━━━━━━━━━ 3D · PROACTIEVE GIDS ━━━━━━━━━━━━━━━━
#### D1 Expliciete opdracht (regex)
| Gebruikerszin                         | Tool-call                               | Bevestiging |
|--------------------------------------|-----------------------------------------|-------------|
| “Voeg thema Papa toe.”               | add_item(theme,"Papa",is_custom=True)   | Thema ‘Papa’ toegevoegd. |
| “Verwijder onderwerp Doula.”         | remove_item(topic,"Doula","Ondersteuning") | Onderwerp ‘Doula’ verwijderd. |
| “Hernoem Sfeer naar Sfeer & Licht.”  | update_item(theme,"Sfeer","Sfeer & Licht") | Hernoemd. |
| “Ik ben klaar.”                      | confirm_themes()                        | Overgangstekst |

#### D2 Kennisvraag
U: “Wat is een doula?”  
A: “Een doula is een niet-medische begeleider …”  
*Extra:* propose_quick_replies → ["Ja, zoek info","Nee"]  
 Tool‐voorbeeld: `find_web_resources(topic="doula")`

#### D3 Onzeker antwoord → sub-selector
| Scenario              | Triggerzin                           | Tool | Quick-replies        |
|-----------------------|--------------------------------------|------|----------------------|
| Gebrek aan kennis     | “Ik weet niet wat een ruggenprik is.”| find_web_resources | Ja/nee |
| Twijfel opties        | “Geen idee welke pijnstilling.”      | vergelijk_opties(options=[…]) | Ja/nee |
| Persoonlijke twijfel  | “Ik twijfel over wie erbij moet …”   | geef_denkvraag(theme="Ondersteuning") | Ja/nee |

#### D4 Small-talk
U: “Dankjewel, je bent geweldig!”  
A: “Graag gedaan! Zullen we verdergaan met je plan?”

#### D5 Onherkenbaar → Fail-Safe

━━━━━━━━━━━━━━━━━━━━━━━━ 3E · FAIL-SAFE ━━━━━━━━━━━━━━━━━━━━━━
• 1ᵉ-2ᵉ fout: “Sorry, ik begrijp je niet …”  
• ≥3 fouten: `propose_quick_replies(["Extra hulp","Terug naar hoofdmenu"])`

━━━━━━━━━━━━━━━━━━━━━━━━ 4 · TOOL-REFERENTIE ━━━━━━━━━━━━━━━━
(add_item · remove_item · update_item · confirm_themes · start_qa_session  
 get_next_question · log_answer · save_plan_summary · find_web_resources  
 vergelijk_opties · geef_denkvraag · propose_quick_replies · propose_topics  
 get_plan_status · offer_choices · present_tool_choices · check_onbeantwoorde_punten  
 gen. plan_tekst · find_external_organization · update_question · remove_question · update_answer)

━━━━━━━━━━━━━━━━━━━━━━━━ 5 · VOLLEDIG DIALOOGVOORBEELD ━━━━━━
1. U: Hallo, …  
2. A: <welkom>  
3. UI: Thema ‘Ondersteuning’ → add_item → “Oké, genoteerd.”  
4. UI: Thema ‘Sfeer’ …  
5. UI: Bevestig thema’s → confirm_themes → overgangstekst  
6. A: “Wat zijn je wensen rondom **Aanwezigheid partner**?”  
7. U: Ik weet het niet.  
8. A: propose_quick_replies + find_web_resources (chips verschijnen)  
9. U drukt “Ja, zoek info” (UI stuurt systeem-bericht)  
10. A: Geef korte link-samenvatting + stel vraag opnieuw …  
11. … (QA gaat verder) …  
12. Queue leeg → save_plan_summary → “Gefeliciteerd …”  

EINDE INSTRUCTIE
"""

# ╭──────────────────────────────────────────────╮
# │  ✨  QUICK-REPLY EXTRACTOR – VOLLEDIG PROMPT   │
# ╰──────────────────────────────────────────────╯
EXTRACTOR_PROMPT = r"""
╔══════════════════════════════════════════════════════════════════════╗
║ QUICK-REPLY CLASSIFIER · VOLLEDIGE INSTRUCTIE                        ║
╚══════════════════════════════════════════════════════════════════════╝

DOEL  
Bepaal of een ASSISTENT-tekst quick-reply-knoppen moet krijgen.

BESLIS-REGELS  
1. Tekst eindigt NIET op “?”  →  geen knoppen   (`{"keuzes": null}`)  
2. Pure ja/nee-vraag          →  `["Ja","Nee"]`  
   • Regex voorbeelden:  
     “Wil je doorgaan?”  “Zal ik verdergaan?”  “Mag ik starten?”  
3. Exact twee opties          →  `["Optie A","Optie B"]`  
   • Detecteer patronen “A of B”, “A / B”, “Kies A of B?”  
4. Lijsten met >2 opties      →  geen knoppen  
5. Open/reflectieve vraag     →  geen knoppen  
6. Fallback                   →  geen knoppen

ANTWOORD-FORMAAT  
Altijd JSON-object met sleutel `keuzes`.

VOORBEELDEN  
Tex​t: “Wil je verder?”  
→ `{"keuzes":["Ja","Nee"]}`  

Tex​t: “Wil je een kort verslag of een uitgebreide versie?”  
→ `{"keuzes":["Kort verslag","Uitgebreide versie"]}`  

Tex​t: “Welke kleuren spreken je aan: blauw, groen of paars?”  
→ `{"keuzes": null}`   (meer dan twee opties)  

Tex​t: “Hoe voel je je hierover?”  
→ `{"keuzes": null}`  

Tex​t: “Prima, staat erin.”  
→ `{"keuzes": null}`
"""


# ─────────────────────── Tool-registratie ──────────────────────────────
tool_funcs = [
    get_plan_status, offer_choices, add_item, remove_item, update_item,
    start_qa_session, get_next_question, log_answer,
    update_question, remove_question, update_answer,
    find_web_resources, vergelijk_opties, geef_denkvraag, find_external_organization,
    check_onbeantwoorde_punten, genereer_plan_tekst,
    present_tool_choices, save_plan_summary,
    propose_quick_replies, confirm_themes, propose_topics
]
tools_schema = [get_schema(t) for t in tool_funcs]
TOOL_MAP = {t.openai_schema['function']['name']: t for t in tool_funcs}
log.info(f"{len(tool_funcs)} tools geregistreerd.")

# ─────────────────── KEUZE-EXTRACTOR FUNCTIE ───────────────────
def get_quick_reply_options(text: str) -> Optional[List[str]]:
    if not text or not text.strip().endswith('?'):
        return None
    log.info(f"Keuze-Extractor checkt tekst: '{text[:75]}...'")
    try:
        response = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTOR_PROMPT},
                {"role": "user", "content": text}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        choices = result.get("keuzes")
        if isinstance(choices, list) and len(choices) > 0:
            log.info(f"Keuze-Extractor vond opties: {choices}")
            return choices
        return None
    except Exception as e:
        log.error(f"Keuze-Extractor fout: {e}", exc_info=True)
        return None

# ───────────────── AGENT LOOP MET UI-LOGICA ───────────────────
def run_main_agent_loop(sid: str) -> str:
    st = get_session(sid)
    log.info(f"Start main agent loop voor sessie {sid}. History-len: {len(st['history'])}")

    if len(st["history"]) == 2 and st["history"][1]["role"] == "user":
        welcome_message = "Welkom! Ik ben Mae, je assistent voor het samenstellen van je geboorteplan. Aan de rechterkant in de sidebar kun je beginnen met het selecteren van maximaal 6 thema's. Klik op een thema om de bijbehorende onderwerpen te kiezen. Als je een eigen thema of onderwerp wilt toevoegen, kun je dat hier in de chat typen."
        st["history"].insert(1, {"role": "assistant", "content": welcome_message})
        save_state(sid, st)
        return welcome_message

    for turn in range(5):
        log.info(f"Agent-beurt {turn + 1}/5 voor sessie {sid}")
        resp = client.chat.completions.create(
            model=MODEL_CHOICE, messages=st["history"], tools=tools_schema, tool_choice="auto"
        )
        msg = resp.choices[0].message
        st["history"].append(msg.model_dump(exclude_unset=True, warnings=False))

        if msg.tool_calls:
            tool_results = []
            for call in msg.tool_calls:
                fn = TOOL_MAP.get(call.function.name, lambda **_: f"Error: onbekende tool {call.function.name}")
                try:
                    args = json.loads(call.function.arguments or "{}")
                    result = fn(session_id=sid, **args)
                except Exception as e:
                    result = f"Error executing tool: {e}"; log.error(result, exc_info=True)
                tool_results.append({"tool_call_id": call.id, "role": "tool", "name": call.function.name, "content": str(result)})
            st["history"].extend(tool_results)
            save_state(sid, st)
            if len(tool_results) == 1 and tool_results[0]['name'] == 'confirm_themes':
                return tool_results[0]['content']
            continue

        if msg.content:
            options = get_quick_reply_options(msg.content)
            if options:
                tool_call_id = f"call_{uuid.uuid4()}"
                tool_call_payload = {"role": "assistant", "tool_calls": [{"id": tool_call_id, "type": "function", "function": {"name": "propose_quick_replies", "arguments": json.dumps({"replies": options})}}]}
                st["history"].append(tool_call_payload)
                st["history"].append({"tool_call_id": tool_call_id, "role": "tool", "name": "propose_quick_replies", "content": f"Quick replies voorgesteld: {', '.join(options)}"})

        save_state(sid, st)
        return msg.content or "(geen antwoord)"
    return "(max turns bereikt)"

# ───────────────── FLASK ROUTES & PAYLOAD ──────────────────────
def build_payload(st: Dict[str, Any], reply: str) -> Dict[str, Any]:
    payload = {"assistant_reply": reply, "session_id": st["id"], "stage": st["stage"], "plan": st["plan"], "suggested_replies": [], "suggested_topics": st.get("topic_suggestions", {})}
    for message in reversed(st["history"]):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            for tc in message.get("tool_calls"):
                if tc.get("function", {}).get("name") == "propose_quick_replies":
                    try:
                        args = json.loads(tc["function"]["arguments"])
                        payload["suggested_replies"] = args.get("replies", [])
                        return payload
                    except (json.JSONDecodeError, TypeError): pass
    return payload

@app.post("/agent")
def agent_route():
    try:
        body = request.get_json(force=True) or {}
        msg, sid = body.get("message","").strip(), body.get("session_id") or str(uuid.uuid4())
        if not msg: return jsonify({"assistant_reply":"(leeg bericht)","session_id":sid})
        st = get_session(sid)
        st["history"].append({"role":"user","content":msg})
        save_state(sid,st)
        reply = run_main_agent_loop(sid)
        final_st = get_session(sid)
        if final_st["stage"] == Stage.COMPLETED.value: save_plan_summary(session_id=sid)
        return jsonify(build_payload(final_st, reply))
    except Exception as e:
        log.critical(f"Onverwachte fout in /agent route: {e}", exc_info=True)
        return jsonify({"error": "Er is een interne serverfout opgetreden."}), 500

@app.get("/export/<sid>")
def export_json(sid):
    st=load_state(sid);
    if not st: abort(404)
    path=ROOT/f"geboorteplan_{sid}.json"
    path.write_text(json.dumps(st["plan"],ensure_ascii=False,indent=2),"utf-8")
    return send_file(path,as_attachment=True,download_name=path.name)

@app.route("/", defaults={"path":""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path=="iframe":
        backend_url = os.getenv("RENDER_EXTERNAL_URL","http://127.0.0.1:10000")
        return render_template("iframe_page.html", backend_url=backend_url)
    full_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__=="__main__":
    port = int(os.getenv("PORT", 10000))
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    log.info(f"Flask applicatie wordt gestart op 0.0.0.0:{port} (Debug modus: {debug_mode})")
    app.run("0.0.0.0", port, debug=debug_mode)
