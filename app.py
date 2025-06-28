#!/usr/bin/env python3
# app.py – Geboorteplan-assistent • Versie 10.2 (Met DB Initialisatie & CORS/Iframe) 28-06-2025

import os
import json
import logging
import pathlib
from typing import Any, Dict, Optional, Generator, List
from datetime import date
from functools import wraps

from flask import Flask, request, jsonify, abort, Response, stream_with_context, render_template, redirect, url_for, \
    session, flash
from flask_bcrypt import Bcrypt
from flask_cors import CORS  # TOEGEVOEGDE IMPORT
from openai import OpenAI
from dotenv import load_dotenv

# Lokale imports voor database en RAG
from database import db, User, BirthPlan
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATIE ---
ROOT = pathlib.Path(__file__).parent
dotenv_path = ROOT / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Flask App Initialisatie
app = Flask(__name__, static_folder="static", static_url_path="/static", template_folder="templates")
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "een-zeer-geheim-geheim-voor-ontwikkeling")
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{ROOT / 'database.db'}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- NIEUWE CORS CONFIGURATIE ---
ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
    "https://chatbotbvmp.onrender.com",
    # VERVANG DIT MET JE EIGEN HOSTINGER DOMEIN
    # "https://jouw-hostinger-website.com"
]
CORS(app, origins=ALLOWED_ORIGINS)
# --- EINDE NIEUWE SECTIE ---

# Extensies Initialiseren
db.init_app(app)
bcrypt = Bcrypt(app)

# Pad- en modelconfiguratie
PLAN_TEMPLATE_FILE = ROOT / "geboorteplan_template.json"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MODEL_CHOICE = os.getenv("MODEL_CHOICE", "gpt-4o-mini")
VALIDATOR_MODEL = os.getenv("VALIDATOR_MODEL", "gpt-4o")

# --- RAG CONFIGURATIE ---
KNOWLEDGE_BASE_FILE = ROOT / "kennisbank.md"
VECTOR_DB_PATH = ROOT / "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Logging setup
logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d – %(message)s")
log = logging.getLogger("geboorteplan-assistent")

# OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- RAG Setup: Laad of bouw de vector database ---
try:
    log.info("Laden van embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if not VECTOR_DB_PATH.exists():
        log.warning("Vector database niet gevonden. Bouwen van nieuwe database...")
        if not KNOWLEDGE_BASE_FILE.exists():
            raise FileNotFoundError(f"Kennisbank-bestand niet gevonden op: {KNOWLEDGE_BASE_FILE}")
        loader = TextLoader(str(KNOWLEDGE_BASE_FILE), encoding="utf-8")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        log.info(f"Kennisbank opgesplitst in {len(docs)} stukken.")
        vector_db = FAISS.from_documents(docs, embeddings)
        vector_db.save_local(str(VECTOR_DB_PATH))
        log.info(f"Nieuwe vector database opgeslagen in: {VECTOR_DB_PATH}")
    else:
        log.info("Laden van bestaande vector database...")
        vector_db = FAISS.load_local(str(VECTOR_DB_PATH), embeddings, allow_dangerous_deserialization=True)

    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    log.info("Vector database succesvol geladen.")

except Exception as e:
    log.error(f"Kritieke fout bij opzetten van RAG: {e}", exc_info=True)
    vector_retriever = None

# --- Statische Teksten ---
INTRO_MESSAGE = """Met het beantwoorden van de volgende vragenlijst proberen we jouw wensen en voorkeuren op te nemen in je persoonlijk bevalplan. Ook als je bij sommige vragen geen specifieke voorkeur hebt, is het waardevol om vooraf na te denken over verschillende situaties en wat deze bij jou oproepen. Door je bewust te zijn van je gedachten en gevoelens, kun je tijdens de bevalling beter aangeven wat je nodig hebt. Invulling geven aan jouw bevalplan is een manier om samen met je partner en zorgverleners het gesprek aan te gaan over wat voor jou belangrijk is. Dit draagt bij aan een gevoel van betrokkenheid en regie, ongeacht hoe je bevalling uiteindelijk verloopt.

In Nederland staat de veiligheid van moeder en kind natuurlijk altijd voorop. Het is goed om je te realiseren dat bij medische noodzaak protocollen kunnen afwijken van je wensen. Je verloskundige/gynaecoloog bespreekt dit altijd met je, tenzij het een acute noodsituatie betreft."""

FINAL_MESSAGE = """Dit bevalplan beschrijft jouw ideale bevalling. Niet alle bevallingen verlopen volgens plan. Denk vooraf ook na over situaties zoals een inleiding, langdurige bevalling of (spoed)keizersnede en bespreek je wensen hierover met je verloskundige. Ook in afwijkende situaties blijven veel wensen uit dit plan relevant. Deel het daarom altijd met je zorgteam.

Onthoud: een goede bevalervaring betekent dat jij je gehoord voelt, goed geïnformeerd bent en jij en je kindje gezond zijn."""


# --- DATABASE INITIALISATIE COMMANDO ---
@app.cli.command("init-db")
def init_db_command():
    """Maakt de databasetabellen aan."""
    with app.app_context():
        db.create_all()
    print("Database geïnitialiseerd en tabellen aangemaakt.")


# --- NIEUWE IFRAME SECURITY HEADER ---
@app.after_request
def add_security_headers(response):
    """Voegt de nodige Content-Security-Policy header toe om iframe embedding toe te staan."""
    # Construeer de 'frame-ancestors' waarde uit de ALLOWED_ORIGINS lijst
    frame_ancestors = " ".join(ALLOWED_ORIGINS)
    response.headers['Content-Security-Policy'] = f"frame-ancestors 'self' {frame_ancestors}"
    return response
# --- EINDE NIEUWE SECTIE ---


# --- AUTHENTICATIE DECORATOR ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Je moet ingelogd zijn om deze pagina te bekijken.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


# --- GEBRUIKER & GEBOORTEPLAN BEHEER ---
def get_plan_for_user(user_id: int) -> Optional[BirthPlan]:
    return BirthPlan.query.filter_by(user_id=user_id).first()


def get_or_create_plan_for_user(user_id: int) -> BirthPlan:
    plan = get_plan_for_user(user_id)
    if plan:
        return plan

    log.info(f"Nieuw geboorteplan wordt aangemaakt voor gebruiker ID: {user_id}")
    with open(PLAN_TEMPLATE_FILE, "r", encoding="utf-8") as f:
        plan_template = json.load(f)

    new_plan = BirthPlan(
        user_id=user_id,
        plan=plan_template,
        history=[]
    )
    db.session.add(new_plan)
    db.session.commit()
    return new_plan


def save_plan_state(plan: BirthPlan, st: Dict[str, Any]):
    plan.plan = st.get('plan')
    plan.history = st.get('history')
    db.session.commit()
    log.debug(f"Status opgeslagen voor gebruiker ID: {plan.user_id}")


# --- HELPER FUNCTIES ---
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


# --- PAGINA ROUTES ---
@app.route('/')
def root():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # --- Accountgegevens ophalen ---
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')

        # --- Persoonlijke gegevens ophalen ---
        woman_name = request.form.get('woman_name')
        partner_name = request.form.get('partner_name')
        woman_dob_str = request.form.get('woman_dob')
        due_date_str = request.form.get('due_date')
        woman_phone = request.form.get('woman_phone')
        partner_phone = request.form.get('partner_phone')

        # --- Baby & Zorg gegevens ophalen ---
        baby_name = request.form.get('baby_name')
        baby_name_secret = True if request.form.get('baby_name_secret') else False
        midwifery_practice = request.form.get('midwifery_practice')
        midwifery_phone = request.form.get('midwifery_phone')
        medical_complications = request.form.get('medical_complications')

        # --- Validatie ---
        if not all([email, username, password, woman_name, due_date_str]):
            flash("Vul alle verplichte velden (*) in.", "error")
            return redirect(url_for('register'))

        if User.query.filter((User.email == email) | (User.username == username)).first():
            flash("E-mailadres of gebruikersnaam is al in gebruik.", "error")
            return redirect(url_for('register'))

        # --- Gegevens verwerken en opslaan ---
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        due_date = date.fromisoformat(due_date_str)
        woman_dob = date.fromisoformat(woman_dob_str) if woman_dob_str else None

        new_user = User(
            email=email,
            username=username,
            password_hash=hashed_password,
            woman_name=woman_name,
            partner_name=partner_name,
            woman_dob=woman_dob,
            due_date=due_date,
            woman_phone=woman_phone,
            partner_phone=partner_phone,
            baby_name=baby_name,
            baby_name_secret=baby_name_secret,
            midwifery_practice=midwifery_practice,
            midwifery_phone=midwifery_phone,
            medical_complications=medical_complications
        )

        db.session.add(new_user)
        db.session.commit()

        get_or_create_plan_for_user(new_user.id)

        flash("Account succesvol aangemaakt! Je kunt nu inloggen.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            log.info(f"Gebruiker {user.username} (ID: {user.id}) succesvol ingelogd.")
            return redirect(url_for('dashboard'))
        else:
            flash("Inloggen mislukt. Controleer je e-mailadres en wachtwoord.", "error")
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    user_id = session.pop('user_id', None)
    if user_id:
        log.info(f"Gebruiker ID: {user_id} uitgelogd.")
    flash("Je bent succesvol uitgelogd.", "success")
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    user = User.query.get_or_404(session['user_id'])
    return render_template('dashboard.html', user=user)


@app.route('/vragenlijst')
@login_required
def vragenlijst():
    return render_template('index.html')


# --- API ROUTE (CHATBOT LOGICA) ---
@app.route("/agent", methods=['POST'])
@login_required
def agent_route():
    user_id = session.get('user_id')
    plan_obj = get_or_create_plan_for_user(user_id)

    st = {
        "id": plan_obj.user_id,
        "plan": plan_obj.plan,
        "history": plan_obj.history
    }

    body = request.get_json(force=True) or {}
    command = body.get("command")
    data = body.get("data", {})

    if command == "initialize":
        return jsonify({"session_id": st["id"], "state": st, "welcome_message": INTRO_MESSAGE})

    elif command == "question_selected":
        topic_id = data.get("topic_id")
        topic, _ = find_topic_by_id(st["plan"], topic_id)
        if not topic: abort(404, "Topic niet gevonden")
        explanation_message = f"**Toelichting bij \"{topic['question']}\"**:\n\n{topic['explanation']}"
        return jsonify(
            {"session_id": st["id"], "state": st, "explanation": explanation_message, "topic_name": topic["name"]})

    elif command == "save_answer" or command == "submit_clarification":
        topic_id = data.get("topic_id")
        original_answer = data.get("answer") if command == "save_answer" else data.get("original_answer")
        clarification = data.get("clarification") if command == "submit_clarification" else None
        topic, theme = find_topic_by_id(st["plan"], topic_id)
        if not topic: abort(404, "Topic niet gevonden")

        plan_summary = "\n".join(
            f"- Thema '{t['name']}', Vraag '{top['question']}', Antwoord: '{top.get('answer', '')}'"
            for t in st['plan'] for top in t['topics'] if
            top.get('answer') and top['id'] != topic_id and top.get('answer') != '__SKIPPED__'
        ) or "Nog geen andere antwoorden gegeven."

        clarification_text = f"De gebruiker heeft dit verduidelijkt: '{clarification}'." if clarification else ""
        validation_prompt = f"""Je bent een kritische maar vriendelijke verloskundige adviseur. Analyseer het antwoord van een gebruiker.
CONTEXT: De gebruiker stelt een geboorteplan op. HUIDIGE THEMA: {theme['name']}. DE VRAAG WAS: "{topic['question']}". HET GEGEVEN ANTWOORD IS: "{original_answer}". {clarification_text}
SAMENVATTING VAN EERDERE KEUZES: {plan_summary}
TAAK: Controleer het antwoord op onzin en tegenstrijdigheden.
- Als het antwoord (eventueel met de verduidelijking) logisch en consistent is, antwoord dan met exact en alleen: OK
- Als het antwoord onzinnig of tegenstrijdig blijft, formuleer dan een korte, vriendelijke, open wedervraag om de gebruiker te helpen. Wees direct."""

        try:
            validation_response = client.chat.completions.create(model=VALIDATOR_MODEL, messages=[
                {"role": "user", "content": validation_prompt}], temperature=0.2)
            validation_result = validation_response.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"Validatie-call mislukt: {e}")
            validation_result = "OK"

        if validation_result == "OK":
            topic["answer"] = original_answer
            st["history"].append(
                {"role": "system", "content": f"ANTWOORD OPGESLAGEN voor '{topic['name']}': '{original_answer}'"})
            save_plan_state(plan_obj, st)
            final_message = FINAL_MESSAGE if is_plan_complete(st["plan"]) else ""
            return jsonify({"session_id": st["id"], "state": st, "status": "ok", "final_message": final_message})
        else:
            st["history"].append({"role": "system",
                                  "content": f"ONGELDIG ANTWOORD voor '{topic['name']}': '{original_answer}'. Validatievraag wordt gesteld."})
            save_plan_state(plan_obj, st)
            return jsonify(
                {"session_id": st["id"], "state": st, "status": "validation_failed", "feedback": validation_result,
                 "original_answer": original_answer})

    elif command == "skip_question":
        topic_id = data.get("topic_id")
        topic, _ = find_topic_by_id(st["plan"], topic_id)
        if topic:
            topic["answer"] = "__SKIPPED__"
            st["history"].append({"role": "system", "content": f"VRAAG OVERGESLAGEN: '{topic['name']}'"})
            save_plan_state(plan_obj, st)
        final_message = FINAL_MESSAGE if is_plan_complete(st["plan"]) else ""
        return jsonify({"session_id": st["id"], "state": st, "status": "ok", "final_message": final_message})

    elif command == "start_guided_dialogue" or command == "user_message":
        user_message = data.get("message", "Help me met deze vraag.")
        st["history"].append({"role": "user", "content": user_message})
        topic, _ = find_topic_by_id(st['plan'], data.get('topic_id'))

        system_prompt = "Je bent Mae, een behulpzame assistent. Beantwoord de vraag van de gebruiker kort en bondig."

        if command == "user_message" and vector_retriever:
            log.info(f"RAG: Zoeken naar context voor vraag: '{user_message}'")
            retrieved_docs = vector_retriever.invoke(user_message)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            log.info(f"RAG: Gevonden context:\n{context}")

            system_prompt = f"""Je bent Mae, een behulpzame assistent gespecialiseerd in geboortezorg. Beantwoord de vraag van de gebruiker UITSLUITEND op basis van de volgende context. Als de informatie niet in de context staat, zeg dan dat je het niet weet. Wees beknopt.

CONTEXT:
---
{context}
---
"""
        elif command == "start_guided_dialogue" and topic:
            question_context = f"De gebruiker wil hulp bij de vraag: '{topic['question']}'. De officiële toelichting is: '{topic['explanation']}'."
            system_prompt = f"BELANGRIJK: Je bent nu in 'begeleide dialoog'-modus. {question_context} Je doel is de gebruiker te helpen een eigen antwoord te formuleren. Begin met een samenvatting van de toelichting en stel DAARNA een open, verkennende vraag om de dialoog te starten."

        messages_for_llm = [{"role": "system", "content": system_prompt}]
        messages_for_llm.extend([msg for msg in st["history"][-7:] if msg.get("role") != "system"])

        def generate():
            full_response = ""
            for content_chunk in stream_llm_response(messages_for_llm):
                parsed_chunk = json.loads(content_chunk.replace("data: ", ""))
                full_response += parsed_chunk.get('content', '')
                yield content_chunk
            st["history"].append({"role": "assistant", "content": full_response})
            save_plan_state(plan_obj, st)

        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    else:
        abort(400, "Onbekend commando")
