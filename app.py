#!/usr/bin/env python3
# app.py – Geboorteplan-assistent • Versie 12.0 (met Server-Side Sessions) 
# (Aangepast voor timer en betaling)

import re
import os
import json
import logging
import pathlib
from typing import Any, Dict, Optional, Generator, List
from datetime import date, timedelta, datetime  # Added datetime for timer
from functools import wraps

from flask import (
    Flask, request, jsonify, abort, Response, stream_with_context,
    render_template, redirect, url_for, session, flash
)
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_session import Session  # server-side sessions
from openai import OpenAI
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix

# Mollie API client (install via pip: mollie-api-python)
from mollie.api.client import Client as MollieClient  # Added Mollie client

# Lokale modules
from database import db, User, BirthPlan
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ── BASIS-CONFIG ────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent
load_dotenv(dotenv_path=ROOT / ".env")

# ── DATABASE-URI (eenvoudig & robuust) ─────────────────────────────────
db_uri = os.getenv("DATABASE_URL", "").strip()  # kan leeg zijn
if db_uri.startswith("postgres://"):  # Render legacy prefix
    db_uri = db_uri.replace("postgres://", "postgresql://", 1)
if not db_uri:  # niks gezet → SQLite
    db_uri = f"sqlite:///{ROOT / 'database.db'}"

# ── FLASK-APP ──────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder="static",
    static_url_path="/static",
    template_folder="templates",
)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# ── APP-CONFIG ─────────────────────────────────────────────────────────
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY", "vervang-dit-met-een-echt-geheim-voor-lokaal-testen"),
    SQLALCHEMY_DATABASE_URI=db_uri,
    SQLALCHEMY_TRACK_MODIFICATIONS=False,

    # Server-side sessions (Flask-Session + SQLAlchemy)
    SESSION_TYPE="sqlalchemy",
    SESSION_PERMANENT=True,
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
    SESSION_USE_SIGNER=True,
    SESSION_SQLALCHEMY_TABLE="sessions",
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_SAMESITE="None",
    SESSION_COOKIE_HTTPONLY=True,
)

# ── EXTENSIES KOPPELEN ────────────────────────────────────────────────
db.init_app(app)
app.config["SESSION_SQLALCHEMY"] = db
sess = Session(app)  # initialiseert session-manager
bcrypt = Bcrypt(app)

# ── CORS ───────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
    "https://chatbotbvmp.onrender.com",
]
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)

# ── PAD- & MODEL-CONFIGURATIE ──────────────────────────────────────────
PLAN_TEMPLATE_FILE = ROOT / "geboorteplan_template.json"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MODEL_CHOICE = os.getenv("MODEL_CHOICE", "gpt-4o-mini")
VALIDATOR_MODEL = os.getenv("VALIDATOR_MODEL", "gpt-4o")

# ── LOGGING ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d – %(message)s",
)
log = logging.getLogger("geboorteplan-assistent")

# ── OPENAI CLIENT ──────────────────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# RAG CONFIG en Setup... (geen wijzigingen hier)
KNOWLEDGE_BASE_FILE = ROOT / "kennisbank.md"
VECTOR_DB_PATH = ROOT / "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
vector_retriever = None
try:
    log.info("Laden van embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if not VECTOR_DB_PATH.exists():
        log.warning("Vector database niet gevonden. Bouwen van nieuwe database...")
        if KNOWLEDGE_BASE_FILE.exists():
            loader = TextLoader(str(KNOWLEDGE_BASE_FILE), encoding="utf-8")
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            vector_db = FAISS.from_documents(docs, embeddings)
            vector_db.save_local(str(VECTOR_DB_PATH))
            vector_retriever = vector_db.as_retriever(search_kwargs={"k": 2})
            log.info(f"Nieuwe vector database opgeslagen.")
    else:
        log.info("Laden van bestaande vector database...")
        vector_db = FAISS.load_local(str(VECTOR_DB_PATH), embeddings, allow_dangerous_deserialization=True)
        vector_retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    log.info("Vector database succesvol geladen.")
except Exception as e:
    log.error(f"Kritieke fout bij opzetten van RAG: {e}", exc_info=True)

# ── MOLLIE CLIENT INITIALISATIE ───────────────────────────────────────
mollie_client = MollieClient()
mollie_client.set_api_key(os.getenv("MOLLIE_API_KEY", ""))  # Zorg dat MOLLIE_API_KEY in .env staat

# --- DATABASE INITIALISATIE COMMANDO ---
@app.cli.command("init-db")
def init_db_command():
    """Maakt alle databasetabellen aan, inclusief de nieuwe 'sessions' tabel."""
    with app.app_context():
        db.create_all()
    print("Database tabellen succesvol aangemaakt/bijgewerkt.")

# Statische teksten, helpers, decorators, etc... (geen wijzigingen hier)
INTRO_MESSAGE = """Met het beantwoorden van de volgende vragenlijst proberen we 
jouw wensen en voorkeuren op te nemen in je persoonlijk bevalplan. ..."""
FINAL_MESSAGE = """Dit bevalplan beschrijft jouw ideale bevalling..."""

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            log.warning(f"Sessie 'user_id' niet gevonden voor pad {request.path}. Doorverwijzen naar login.")
            flash("Je moet ingelogd zijn om deze pagina te bekijken.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_plan_for_user(user_id: int) -> Optional[BirthPlan]:
    return BirthPlan.query.filter_by(user_id=user_id).first()

def get_or_create_plan_for_user(user_id: int) -> BirthPlan:
    plan = get_plan_for_user(user_id)
    if plan: 
        return plan
    log.info(f"Nieuw geboorteplan wordt aangemaakt voor gebruiker ID: {user_id}")
    with open(PLAN_TEMPLATE_FILE, "r", encoding="utf-8") as f:
        plan_template = json.load(f)
    new_plan = BirthPlan(user_id=user_id, plan=plan_template, history=[])
    db.session.add(new_plan)
    db.session.commit()
    return new_plan

def save_plan_state(plan: BirthPlan, st: Dict[str, Any]):
    plan.plan = st.get('plan')
    plan.history = st.get('history')
    db.session.commit()

def find_topic_by_id(plan: list, topic_id: str):
    for theme in plan:
        for topic in theme.get('topics', []):
            if topic['id'] == topic_id:
                return topic, theme
    return None, None

def is_plan_complete(plan: list) -> bool:
    for theme in plan:
        for topic in theme.get('topics', []):
            if not topic.get('answer'):
                return False
    return True

def stream_llm_response(messages: list) -> Generator[str, None, None]:
    try:
        stream = client.chat.completions.create(model=MODEL_CHOICE, messages=messages, stream=True)
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content: 
                yield f"data: {json.dumps({'content': content})}\n\n"
    except Exception as e:
        log.error(f"Streaming LLM fout: {e}")
        yield f"data: {json.dumps({'error': 'Sorry, er ging iets mis met de OpenAI API.'})}\n\n"

def is_mobile_device():
    """Detecteert of het request van een mobiel apparaat komt."""
    user_agent = request.headers.get('User-Agent', '').lower()
    mobile_pattern = re.compile(
        r'(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|'
        r'elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge|maemo|midp|'
        r'mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|'
        r'rim)|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|'
        r'vodafone|wap|windows ce|xda|xiino|ipad|playbook|silk',
        re.IGNORECASE | re.MULTILINE
    )
    return bool(mobile_pattern.search(user_agent))

def render_mobile_aware_template(desktop_template, **kwargs):
    """
    Rendert een mobiele template als de gebruiker mobiel is,
    anders de desktop template.
    """
    if request.args.get('mobile') == 'true':
        log.info("Mobiele weergave geforceerd via query parameter.")
        is_mobile = True
    else:
        is_mobile = is_mobile_device()

    if is_mobile:
        mobile_template_name = f"mobile_{desktop_template}"
        mobile_template_path = os.path.join(app.template_folder, mobile_template_name)
        if os.path.exists(mobile_template_path):
            log.info(f"Mobiel apparaat gedetecteerd. '{mobile_template_name}' wordt gerenderd.")
            return render_template(mobile_template_name, **kwargs)
        else:
            log.warning(f"Mobiel apparaat gedetecteerd, maar '{mobile_template_name}' niet gevonden. Fallback naar desktop.")
    # Fallback naar desktop versie
    return render_template(desktop_template, **kwargs)

@app.route('/')
def root():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # ...
        # Maak nieuwe gebruiker aan
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        due_date = date.fromisoformat(due_date_str)
        woman_dob = date.fromisoformat(woman_dob_str) if woman_dob_str else None
        new_user = User(
            email=email, username=username, password_hash=hashed_password, woman_name=woman_name,
            partner_name=partner_name, woman_dob=woman_dob, due_date=due_date, woman_phone=woman_phone,
            partner_phone=partner_phone, baby_name=baby_name, baby_name_secret=baby_name_secret,
            midwifery_practice=midwifery_practice, midwifery_phone=midwifery_phone,
            medical_complications=medical_complications, paid=False  # Initialize paid status
        )
        db.session.add(new_user)
        db.session.commit()
        get_or_create_plan_for_user(new_user.id)  # Maak ook direct een plan aan
        flash("Account succesvol aangemaakt! Je kunt nu inloggen.", "success")
        return redirect(url_for('login'))
    return render_mobile_aware_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # ... validatie ...
        if user and bcrypt.check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            log.info(f"Gebruiker {user.username} (ID: {user.id}) succesvol ingelogd.")
            return redirect(url_for('dashboard'))
        else:
            flash("Inloggen mislukt. Controleer je e-mailadres en wachtwoord.", "error")
            return redirect(url_for('login'))
    return render_mobile_aware_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.clear()
    flash("Je bent succesvol uitgelogd.", "success")
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    user = User.query.get_or_404(session['user_id'])
    # Toon dashboard en betalingskeuzes
    return render_mobile_aware_template('dashboard.html', user=user)

@app.route('/start_trial')
@login_required
def start_trial():
    # Start 5-minuten vrije timer
    session['trial_start'] = datetime.utcnow()
    return redirect(url_for('vragenlijst'))

@app.route('/start_payment')
@login_required
def start_payment():
    user = User.query.get_or_404(session['user_id'])
    # Stel betaling in via Mollie voor iDEAL
    payment = mollie_client.payments.create({
        "amount": {
            "currency": "EUR",
            "value": "9.99"  # Stel bedrag in, voorbeeld €9.99
        },
        "description": "Betaling voor geboorteplan downloaden",
        "redirectUrl": url_for('payment_return', _external=True),
        "webhookUrl": url_for('payment_webhook', _external=True),
        "method": ["ideal"],
        "metadata": {
            "user_id": user.id
        }
    })
    # Redirect de gebruiker naar de Mollie betaalpagina
    return redirect(payment.get("checkout_url", "/"))

@app.route('/payment_return')
@login_required
def payment_return():
    payment_id = request.args.get('id')
    if not payment_id:
        flash("Betaling geannuleerd of mislukt.", "error")
        return redirect(url_for('dashboard'))
    # Controleer de betaling via Mollie API
    try:
        payment = mollie_client.payments.get(payment_id)
    except Exception as e:
        log.error(f"Mollie ophalen mislukt: {e}")
        flash("Betaling kon niet worden gevalideerd.", "error")
        return redirect(url_for('dashboard'))
    if payment.is_paid():
        user = User.query.get_or_404(session['user_id'])
        user.paid = True
        db.session.commit()
        flash("Betaling succesvol! U kunt nu uw geboorteplan downloaden.", "success")
        return redirect(url_for('download_plan'))
    else:
        flash("Betaling is niet voltooid. Probeer opnieuw.", "error")
        return redirect(url_for('dashboard'))

@app.route('/payment_webhook', methods=['POST'])
def payment_webhook():
    # Mollie webhook wordt aangesproken na statusverandering
    data = request.form
    payment_id = data.get('id')
    if not payment_id:
        return ('', 400)
    try:
        payment = mollie_client.payments.get(payment_id)
        if payment.is_paid():
            user_id = payment.metadata.get('user_id')
            user = User.query.get(user_id)
            if user:
                user.paid = True
                db.session.commit()
    except Exception as e:
        log.error(f"Webhook verwerken mislukt: {e}")
    return ('', 200)

@app.route('/download_plan')
@login_required
def download_plan():
    user = User.query.get_or_404(session['user_id'])
    if not user.paid:
        flash("U moet betalen om uw geboorteplan te downloaden.", "error")
        return redirect(url_for('dashboard'))
    plan_obj = get_plan_for_user(user.id)
    if not plan_obj:
        flash("Geen geboorteplan gevonden.", "error")
        return redirect(url_for('dashboard'))
    # Exporteer het plan als JSON
    plan_data = plan_obj.plan
    response = jsonify(plan_data)
    response.headers['Content-Disposition'] = 'attachment; filename=geboorteplan.json'
    return response

@app.route('/vragenlijst')
@login_required
def vragenlijst():
    user = User.query.get_or_404(session['user_id'])
    # Controleer of gebruiker mag starten
    if not user.paid:
        trial_start = session.get('trial_start')
        if not trial_start:
            # Geen keuze gemaakt, terugsturen naar dashboard
            flash("Kies eerst of u de vragenlijst gratis wilt proberen of wilt betalen.", "error")
            return redirect(url_for('dashboard'))
        # Controleer timer
        elapsed = datetime.utcnow() - session.get('trial_start')
        if elapsed > timedelta(minutes=5):
            flash("De gratis tijd is verstreken. Betaal om door te gaan.", "error")
            return redirect(url_for('dashboard'))
    # Toon de vragenlijst (vragenlijst wordt geladen via JS)
    return render_mobile_aware_template('index.html')

# --- API ROUTE (uitgebreid met betalingstijd check) ---
@app.route("/agent", methods=['POST'])
@login_required
def agent_route():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    plan_obj = get_or_create_plan_for_user(user_id)
    st = {"id": plan_obj.user_id, "plan": plan_obj.plan, "history": plan_obj.history}
    body = request.get_json(force=True) or {}
    command = body.get("command")
    data = body.get("data", {})

    # Controleer betalingstijd voor gratis gebruikers
    if command in ["save_answer", "skip_question", "submit_clarification", "question_selected"]:
        if not user.paid:
            trial_start = session.get('trial_start')
            if trial_start:
                elapsed = datetime.utcnow() - session.get('trial_start')
                if elapsed > timedelta(minutes=5):
                    # Betaalplichtig geworden
                    return jsonify({"status": "payment_required", "message": "De gratis tijd is verstreken. Betaal om verder te gaan."})
            else:
                # Probeer vragen te beantwoorden zonder te starten
                abort(403, "Geen gratis sessie gestart.")

    if command == "initialize":
        return jsonify({"session_id": st["id"], "state": st, "welcome_message": INTRO_MESSAGE})

    elif command == "question_selected":
        topic_id = data.get("topic_id")
        topic, _ = find_topic_by_id(st["plan"], topic_id)
        if not topic: 
            abort(404, "Topic niet gevonden")
        explanation_message = f"**Toelichting bij \"{topic['question']}\"**:\n\n{topic['explanation']}"
        return jsonify({"session_id": st["id"], "state": st, "explanation": explanation_message, "topic_name": topic["name"]})

    elif command in ["save_answer", "submit_clarification"]:
        topic_id = data.get("topic_id")
        original_answer = data.get("answer") if command == "save_answer" else data.get("original_answer")
        clarification = data.get("clarification") if command == "submit_clarification" else None
        topic, theme = find_topic_by_id(st["plan"], topic_id)
        if not topic: 
            abort(404, "Topic niet gevonden")
        plan_summary = "\n".join(
            f"- Thema '{t['name']}', Vraag '{top['question']}', Antwoord: '{top.get('answer', '')}'"
            for t in st['plan'] for top in t['topics']
            if top.get('answer') and top['id'] != topic_id and top.get('answer') != '__SKIPPED__'
        ) or "Nog geen andere antwoorden gegeven."
        clarification_text = f"De gebruiker heeft dit verduidelijkt: '{clarification}'." if clarification else ""
        validation_prompt = f"""Je bent een kritische maar vriendelijke verloskundige adviseur. Analyseer het antwoord van een gebruiker.
CONTEXT: De gebruiker stelt een geboorteplan op. HUIDIGE THEMA: {theme['name']}. DE VRAAG WAS: "{topic['question']}". HET GEGEVEN ANTWOORD IS: "{original_answer}". {clarification_text}
SAMENVATTING VAN EERDERE KEUZES: {plan_summary}
TAAK: Controleer het antwoord op onzin en tegenstrijdigheden.
- Als het antwoord logisch en consistent is, antwoord dan met exact: OK
- Anders formuleer een korte, vriendelijke tegenvraag."""
        try:
            validation_response = client.chat.completions.create(model=VALIDATOR_MODEL, messages=[{"role": "user", "content": validation_prompt}], temperature=0.2)
            validation_result = validation_response.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"Validatie-call mislukt: {e}")
            validation_result = "OK"
        if validation_result == "OK":
            topic["answer"] = original_answer
            st["history"].append({"role": "system", "content": f"ANTWOORD OPGESLAGEN voor '{topic['name']}': '{original_answer}'"})
            save_plan_state(plan_obj, st)
            final_message = FINAL_MESSAGE if is_plan_complete(st["plan"]) else ""
            return jsonify({"session_id": st["id"], "state": st, "status": "ok", "final_message": final_message})
        else:
            st["history"].append({"role": "system", "content": f"ONGELDIG ANTWOORD voor '{topic['name']}': '{original_answer}'. Validatievraag wordt gesteld."})
            save_plan_state(plan_obj, st)
            return jsonify({"session_id": st["id"], "state": st, "status": "validation_failed", "feedback": validation_result, "original_answer": original_answer})

    elif command == "skip_question":
        topic_id = data.get("topic_id")
        topic, _ = find_topic_by_id(st["plan"], topic_id)
        if topic:
            topic["answer"] = "__SKIPPED__"
            st["history"].append({"role": "system", "content": f"VRAAG OVERGESLAGEN: '{topic['name']}'"})
            save_plan_state(plan_obj, st)
        final_message = FINAL_MESSAGE if is_plan_complete(st["plan"]) else ""
        return jsonify({"session_id": st["id"], "state": st, "status": "ok", "final_message": final_message})

    elif command in ["start_guided_dialogue", "user_message"]:
        user_message = data.get("message", "Help me met deze vraag.")
        st["history"].append({"role": "user", "content": user_message})
        topic, _ = find_topic_by_id(st['plan'], data.get('topic_id'))
        system_prompt = "Je bent Mae, een behulpzame assistent. Beantwoord de vraag van de gebruiker kort en bondig."
        if command == "user_message" and vector_retriever:
            context = "\n\n".join([doc.page_content for doc in vector_retriever.invoke(user_message)])
            system_prompt = f"""Je bent Mae, een behulpzame assistent gespecialiseerd in geboortezorg. Beantwoord de vraag van de gebruiker 
UITSLUITEND op basis van de volgende context. Als de informatie niet in de context staat, zeg dan dat je het niet weet. Wees beknopt.
CONTEXT:\n---\n{context}\n---"""
        elif command == "start_guided_dialogue" and topic:
            question_context = f"De gebruiker wil hulp bij de vraag: '{topic['question']}'. De officiële toelichting is: '{topic['explanation']}'."
            system_prompt = f"BELANGRIJK: Je bent nu in 'begeleide dialoog'-modus. {question_context} Je doel is de gebruiker te helpen een eigen antwoord te formuleren. Begin met een samenvatting van de toelichting en stel DAARNA een open, verkennende vraag om de dialoog te starten."
        messages_for_llm = [{"role": "system", "content": system_prompt}] + [msg for msg in st["history"][-7:] if msg.get("role") != "system"]
        def generate():
            full_response = ""
            for content_chunk in stream_llm_response(messages_for_llm):
                parsed_chunk = json.loads(content_chunk.replace("data: ", ""))
                if parsed_chunk.get('content'):
                    full_response += parsed_chunk['content']
                    yield f"data: {json.dumps({'content': full_response})}\n\n"
        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    abort(400, "Onbekende command.")

