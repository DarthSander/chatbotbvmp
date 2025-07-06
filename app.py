#!/usr/bin/env python3
# app.py – Geboorteplan-assistent • Versie met E-mail, Authenticatie & Data Retentie

import re, os, json, logging, pathlib, click
from typing import Any, Dict, Optional, Generator, List
from datetime import date, timedelta, datetime
from functools import wraps

from flask import (
    Flask, request, jsonify, abort, Response, stream_with_context,
    render_template, redirect, url_for, session, flash
)
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_session import Session
from flask_mail import Mail, Message
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
from openai import OpenAI
from mollie.api.client import Client as MollieClient
from itsdangerous import URLSafeTimedSerializer

# ── lokale modules ────────────────────────────────
from database import db, User, BirthPlan
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ── basis-config ──────────────────────────────────
ROOT = pathlib.Path(__file__).parent
load_dotenv(ROOT / ".env")

# ── database URI ──────────────────────────────────
db_uri = os.getenv("DATABASE_URL", "").strip()
if db_uri.startswith("postgres://"):
    db_uri = db_uri.replace("postgres://", "postgresql://", 1)
if not db_uri:
    db_uri = f"sqlite:///{ROOT / 'database.db'}"

# ── Flask-app ─────────────────────────────────────
app = Flask(__name__,
            static_folder="static",
            static_url_path="/static",
            template_folder="templates")
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# ── app-config ────────────────────────────────────
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY", "vervang-dit-met-een-echt-geheim-voor-lokaal-testen"),
    SQLALCHEMY_DATABASE_URI=db_uri,
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SESSION_TYPE="sqlalchemy",
    SESSION_PERMANENT=True,
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
    SESSION_USE_SIGNER=True,
    SESSION_SQLALCHEMY_TABLE="sessions",
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_SAMESITE="None",
    SESSION_COOKIE_HTTPONLY=True,
)

# --- E-mailconfiguratie ---
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.zoho.eu')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'true').lower() in ['true', '1', 't']
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = ('Beval met een Plan', os.getenv('MAIL_USERNAME'))

# ── extensies ─────────────────────────────────────
db.init_app(app)
app.config["SESSION_SQLALCHEMY"] = db
sess = Session(app)
bcrypt = Bcrypt(app)
mail = Mail(app)

# --- Serializer voor veilige tokens ---
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# ── CORS ──────────────────────────────────────────
ALLOWED_ORIGINS = [
    "https://bevalmeteenplan.nl",
    "https://www.bevalmeteenplan.nl",
    "https://chatbotbvmp.onrender.com",
]
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)

# ── logging ───────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d – %(message)s")
log = logging.getLogger("geboorteplan-assistent")

# ── OpenAI & RAG set-up ───────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_CHOICE = os.getenv("MODEL_CHOICE", "gpt-4o-mini")
VALIDATOR_MODEL = os.getenv("VALIDATOR_MODEL", "gpt-4o")

PLAN_TEMPLATE_FILE = ROOT / "geboorteplan_template.json"
KNOWLEDGE_BASE_FILE = ROOT / "kennisbank.md"
VECTOR_DB_PATH = ROOT / "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

vector_retriever = None
try:
    log.info("Laden van embedding-model …")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if not VECTOR_DB_PATH.exists():
        log.warning("Vector-db niet gevonden – bouwen …")
        if KNOWLEDGE_BASE_FILE.exists():
            loader = TextLoader(str(KNOWLEDGE_BASE_FILE), encoding="utf-8")
            docs = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
            FAISS.from_documents(docs, embeddings).save_local(str(VECTOR_DB_PATH))
    vector_db = FAISS.load_local(str(VECTOR_DB_PATH), embeddings, allow_dangerous_deserialization=True)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    log.info("Vector-db geladen.")
except Exception:
    log.error("RAG set-up fout:", exc_info=True)

# ── Mollie client ─────────────────────────────────
MOLLIE_KEY = os.getenv("MOLLIE_API_KEY", "").strip()
mollie_client = MollieClient()
try:
    mollie_client.set_api_key(MOLLIE_KEY)
except Exception:
    log.warning("⚠️  Mollie API-key ontbreekt of ongeldig; betalingen uitgeschakeld.")

# ── statische teksten ─────────────────────────────
INTRO_MESSAGE = (
    "Met het beantwoorden van de volgende vragenlijst proberen we jouw wensen en voorkeuren op te nemen in je persoonlijk bevalplan. "
    "Ook als je bij sommige vragen geen specifieke voorkeur hebt, is het waardevol om vooraf na te denken over verschillende situaties en wat deze bij jou oproepen. "
    "Door je bewust te zijn van je gedachten en gevoelens, kun je tijdens de bevalling beter aangeven wat je nodig hebt. "
    "Invulling geven aan jouw bevalplan is een manier om samen met je partner en zorgverleners het gesprek aan te gaan over wat voor jou belangrijk is. "
    "Dit draagt bij aan een gevoel van betrokkenheid en regie, ongeacht hoe je bevalling uiteindelijk verloopt. "
    "In Nederland staat de veiligheid van moeder en kind natuurlijk altijd voorop. "
    "Het is goed om je te realiseren dat bij medische noodzaak kunnen protocollen afwijken van je wensen. "
    "Je verloskundige/gynaecoloog bespreekt dit altijd met je, tenzij het een acute noodsituatie betreft."
)
FINAL_MESSAGE = (
    "Dit bevalplan beschrijft jouw ideale bevalling. Niet alle bevallingen "
    "verlopen volgens plan …"
)


# ── helpers ───────────────────────────────────────
def login_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if "user_id" not in session:
            flash("Je moet ingelogd zijn om deze pagina te bekijken.", "error")
            return redirect(url_for("login"))
        return f(*a, **kw)

    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Je moet ingelogd zijn om deze pagina te bekijken.", "error")
            return redirect(url_for("login"))
        user = User.query.get(session["user_id"])
        if not user or not user.is_admin:
            flash("Je hebt geen toegang tot deze pagina.", "error")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)

    return wrapper


def send_email(to, subject, template):
    """Verstuurt een e-mail met de gegeven parameters."""
    try:
        msg = Message(
            subject,
            recipients=[to],
            html=template,
            sender=app.config['MAIL_DEFAULT_SENDER']
        )
        mail.send(msg)
    except Exception as e:
        log.error(f"Fout bij versturen van e-mail naar {to}: {e}")


def get_or_create_plan_for_user(uid: int) -> BirthPlan:
    plan = BirthPlan.query.filter_by(user_id=uid).first()
    if plan: return plan
    with open(PLAN_TEMPLATE_FILE, encoding="utf-8") as f:
        tpl = json.load(f)
    plan = BirthPlan(user_id=uid, plan=tpl, history=[])
    db.session.add(plan);
    db.session.commit()
    return plan


def save_plan_state(obj: BirthPlan, st: Dict[str, Any]):
    obj.plan, obj.history = st["plan"], st["history"];
    db.session.commit()


def find_topic_by_id(plan: list, tid: str):
    for th in plan:
        for tp in th["topics"]:
            if tp["id"] == tid: return tp, th
    return None, None


def is_plan_complete(plan: list) -> bool:
    return all(tp.get("answer") for th in plan for tp in th["topics"])


def stream_llm_response(msgs: list) -> Generator[str, None, None]:
    try:
        for ch in client.chat.completions.create(model=MODEL_CHOICE,
                                                 messages=msgs, stream=True):
            if (c := ch.choices[0].delta.content):
                yield f"data: {json.dumps({'content': c})}\n\n"
    except Exception:
        yield f"data: {json.dumps({'error': 'LLM-fout'})}\n\n"
        log.error("Streaming-LLM:", exc_info=True)


def render_mobile_aware_template(desktop, **kw):
    force_m = request.args.get("mobile") == "true"
    is_mob = force_m or "mobile" in request.user_agent.string.lower()
    if is_mob:
        mob = f"mobile_{desktop}"
        if (pathlib.Path(app.template_folder) / mob).exists():
            return render_template(mob, **kw)
    return render_template(desktop, **kw)


# ─── routes ──────────────────────────────────────────────────────────────────
@app.cli.command("init-db")
def init_db():
    with app.app_context(): db.create_all()
    print("✓ tabellen aangemaakt")


@app.route("/")
def root():
    if "user_id" in session:
        user = User.query.get(session["user_id"])
        if user:
            user.last_activity = datetime.utcnow()
            user.last_seen_page = 'dashboard'
            db.session.commit()
        if user and user.is_admin:
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/logout")
def logout():
    session.clear();
    flash("Je bent uitgelogd.", "success");
    return redirect(url_for("login"))


# ---------- registratie ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        f = request.form
        required = [f.get(x, "").strip() for x in ("email", "username", "password", "woman_name", "due_date")]
        if not all(required):
            flash("Vul alle verplichte velden in.", "error")
            return redirect(url_for("register"))
        if User.query.filter((User.email == required[0]) | (User.username == required[1])).first():
            flash("E-mail of gebruikersnaam al in gebruik.", "error")
            return redirect(url_for("register"))

        user = User(
            email=required[0], username=required[1],
            password_hash=bcrypt.generate_password_hash(required[2]).decode(),
            woman_name=required[3], due_date=date.fromisoformat(required[4]),
            partner_name=f.get("partner_name"), woman_phone=f.get("woman_phone"),
            partner_phone=f.get("partner_phone"), baby_name=f.get("baby_name"),
            baby_name_secret=bool(f.get("baby_name_secret")),
            midwifery_practice=f.get("midwifery_practice"),
            midwifery_phone=f.get("midwifery_phone"),
            medical_complications=f.get("medical_complications"),
            paid=True,
            is_confirmed=False
        )
        db.session.add(user);
        db.session.commit()
        get_or_create_plan_for_user(user.id)

        token = s.dumps(user.email, salt='email-confirm-salt')
        confirm_url = url_for('confirm_email', token=token, _external=True)
        html = render_template('email/activate_account.html', confirm_url=confirm_url)
        send_email(user.email, "Bevestig je account", html)

        flash("Account aangemaakt! Controleer je e-mail om je account te activeren.", "success")
        return redirect(url_for("login"))
    return render_mobile_aware_template("register.html")


# ---------- e-mailbevestiging ----------
@app.route('/confirm/<token>')
def confirm_email(token):
    try:
        email = s.loads(token, salt='email-confirm-salt', max_age=3600)
    except Exception:
        flash('De bevestigingslink is ongeldig of verlopen.', 'error')
        return redirect(url_for('login'))

    user = User.query.filter_by(email=email).first_or_404()

    if user.is_confirmed:
        flash('Account al bevestigd. Log alsjeblieft in.', 'success')
    else:
        user.is_confirmed = True
        db.session.commit()
        flash('Je account is succesvol bevestigd! Je kunt nu inloggen.', 'success')

    return redirect(url_for('login'))


# ---------- login ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email, pw = request.form.get("email"), request.form.get("password")
        user = User.query.filter_by(email=email).first()

        if not user or not bcrypt.check_password_hash(user.password_hash, pw):
            flash("Inloggen mislukt. Controleer je e-mail en wachtwoord.", "error")
            return redirect(url_for("login"))

        # AANGEPAST: Sla de e-mailbevestiging over als de gebruiker een admin is.
        # De foutmelding wordt nu alleen getoond als de gebruiker NIET bevestigd is
        # EN ook NIET een admin is.
        if not user.is_confirmed and not user.is_admin:
            flash("Je account is nog niet bevestigd. Controleer je e-mail.", "error")
            return redirect(url_for("login"))

        # Als de check is doorstaan, log de gebruiker in.
        session["user_id"] = user.id;
        user.last_activity = datetime.utcnow()
        db.session.commit()
        flash(f"Welkom terug, {user.woman_name}!", "success")

        # Stuur admin naar het admin dashboard, anders naar het gewone dashboard.
        if user.is_admin:
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("dashboard"))

    return render_mobile_aware_template("login.html")


# ---------- wachtwoord reset ----------
@app.route("/request_reset", methods=['GET', 'POST'])
def request_reset():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()

        if user:
            token = s.dumps(user.email, salt='password-reset-salt')
            reset_url = url_for('reset_with_token', token=token, _external=True)
            html = render_template('email/reset_password.html', reset_url=reset_url)
            send_email(user.email, "Wachtwoord Reset", html)

        flash('Als er een account met dit e-mailadres bestaat, is er een reset-link verstuurd.', 'info')
        return redirect(url_for('login'))

    return render_template('request_reset.html')


@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_with_token(token):
    try:
        email = s.loads(token, salt='password-reset-salt', max_age=3600)
    except Exception:
        flash('De wachtwoord-reset link is ongeldig of verlopen.', 'error')
        return redirect(url_for('request_reset'))

    if request.method == 'POST':
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first_or_404()

        user.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        db.session.commit()

        flash('Je wachtwoord is succesvol bijgewerkt! Je kunt nu inloggen.', 'success')
        return redirect(url_for('login'))

    return render_template('reset_with_token.html', token=token)


# ---------- dashboard ----------
@app.route("/dashboard")
@login_required
def dashboard():
    user = User.query.get(session["user_id"])
    user.last_activity = datetime.utcnow()
    user.last_seen_page = 'dashboard'
    db.session.commit()
    return render_mobile_aware_template("dashboard.html", user=user)


# ---------- trial & betaling ----------
@app.route("/start_trial")
@login_required
def start_trial():
    ts = datetime.utcnow()
    session["trial_start"] = ts
    resp = redirect(url_for("vragenlijst"))
    resp.set_cookie("trial_start_ts", ts.isoformat(timespec="seconds"), max_age=300, secure=True, samesite="Lax",
                    httponly=False)
    return resp


@app.route("/start_payment")
@login_required
def start_payment():
    if not MOLLIE_KEY:
        flash("Betalen is tijdelijk niet beschikbaar.", "error")
        return redirect(url_for("dashboard"))
    user = User.query.get(session["user_id"])
    payment = mollie_client.payments.create({
        "amount": {"currency": "EUR", "value": "9.99"},
        "description": "Betaling geboorteplan",
        "redirectUrl": url_for("payment_return", _external=True),
        "webhookUrl": url_for("payment_webhook", _external=True),
        "method": ["ideal"],
        "metadata": {"user_id": user.id}
    })
    resp = redirect(payment.checkout_url)
    resp.set_cookie("payment_ref", payment.id, max_age=600, secure=True, samesite="Lax", httponly=True)
    return resp


@app.route("/payment_return")
@login_required
def payment_return():
    payment_id = request.args.get("id")
    try:
        payment = mollie_client.payments.get(payment_id)
        if payment.is_paid():
            user = User.query.get(session["user_id"])
            user.paid = True
            db.session.commit()
            flash("Betaling gelukt!", "success")
            resp = redirect(url_for("vragenlijst"))
            resp.delete_cookie("payment_ref")
            return resp
    except Exception as e:
        log.error("Mollie return:", exc_info=True)
    flash("Betaling geannuleerd of mislukt.", "error")
    resp = redirect(url_for("dashboard"))
    resp.delete_cookie("payment_ref")
    return resp


@app.route("/payment_webhook", methods=["POST"])
def payment_webhook():
    payment_id = request.form.get("id")
    try:
        payment = mollie_client.payments.get(payment_id)
        if payment.is_paid():
            user = User.query.get(payment.metadata["user_id"])
            user.paid = True
            db.session.commit()
            resp = Response("", status=200)
            resp.delete_cookie("payment_ref")
            return resp
    except Exception:
        log.error("Webhook:", exc_info=True)
    return "", 200


# ---------- vragenlijst ----------
@app.route("/vragenlijst")
@login_required
def vragenlijst():
    user = User.query.get(session["user_id"])
    trial_start = session.get("trial_start")
    if not trial_start and (c := request.cookies.get("trial_start_ts")):
        try:
            trial_start = datetime.fromisoformat(c)
            session["trial_start"] = trial_start
        except ValueError:
            trial_start = None
    if not user.paid:
        if not trial_start or datetime.utcnow() - trial_start > timedelta(minutes=5):
            flash("De gratis proefperiode is voorbij. Betaal om onbeperkt toegang te krijgen.", "error")
            return redirect(url_for("dashboard"))
    user.last_activity = datetime.utcnow()
    user.last_seen_page = 'vragenlijst'
    db.session.commit()
    return render_mobile_aware_template("index.html")


@app.route('/edit_profile', methods=['POST'])
@login_required
def edit_profile():
    user = User.query.get(session["user_id"])
    if not user: abort(404)
    user.woman_name = request.form.get('woman_name', user.woman_name)
    user.partner_name = request.form.get('partner_name', user.partner_name)
    user.baby_name = request.form.get('baby_name', user.baby_name)
    user.baby_name_secret = 'baby_name_secret' in request.form
    if due_date_str := request.form.get('due_date'):
        user.due_date = date.fromisoformat(due_date_str)
    db.session.commit()
    flash('Je gegevens zijn succesvol bijgewerkt!', 'success')
    return redirect(url_for('dashboard'))


# ---------- download ----------
@app.route("/download_plan")
@login_required
def download_plan():
    user = User.query.get(session["user_id"])
    if not user.paid:
        flash("Betaal eerst om te kunnen downloaden.", "error")
        return redirect(url_for("dashboard"))
    plan_obj = get_or_create_plan_for_user(user.id)
    resp = jsonify(plan_obj.plan)
    resp.headers["Content-Disposition"] = "attachment; filename=geboorteplan.json"
    return resp


# ---------- Visueel Plan Routes ----------
@app.route("/select_template")
@login_required
def select_template():
    user = User.query.get(session["user_id"])
    if not user.paid:
        flash("Deze functie is alleen beschikbaar na betaling.", "error")
        return redirect(url_for("dashboard"))
    templates = []
    layouts_dir = ROOT / "static" / "layouts"
    if layouts_dir.is_dir():
        for layout_file in layouts_dir.glob("*.json"):
            try:
                with open(layout_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                template_id = layout_file.stem
                display_name = config.get("display_name", template_id.replace("_", " ").title())
                if thumbnail := config.get("image_filename"):
                    templates.append({"id": template_id, "name": display_name, "thumbnail_filename": thumbnail})
            except Exception as e:
                log.error(f"Fout bij het laden van template {layout_file.name}: {e}")
    return render_template("templates.html", templates=templates, user=user)


@app.route("/visual-plan/<template_id>")
@login_required
def visual_plan(template_id: str):
    user = User.query.get(session["user_id"])
    if not user.paid:
        flash("Deze functie is alleen beschikbaar na betaling.", "error")
        return redirect(url_for("dashboard"))
    layout_config_path = ROOT / "static" / "layouts" / f"{template_id}.json"
    if not layout_config_path.is_file():
        abort(404, "Layout configuratie niet gevonden")
    with open(layout_config_path, 'r', encoding='utf-8') as f:
        layout_config = json.load(f)
    if not (image_filename := layout_config.get("image_filename")) or not (ROOT / "static" / image_filename).is_file():
        abort(404, "Template afbeelding niet gevonden in configuratie")
    user_placements_for_template = []
    if 'user_data_placements' in layout_config:
        for placement in layout_config['user_data_placements']:
            key, label = placement.get('key'), placement.get('label', '')
            value = getattr(user, key, None)
            if value is not None and value != '':
                value_str = value.strftime('%d-%m-%Y') if isinstance(value, date) else str(value)
                user_placements_for_template.append({
                    "text": f"{label} {value_str}", "x": placement.get('x'), "y": placement.get('y'),
                    "font": placement.get('font')
                })
    plan_obj = get_or_create_plan_for_user(user.id)
    plan_obj.visual_template = template_id
    db.session.commit()
    return render_template(
        "visual_plan.html",
        plan_data_json=json.dumps(plan_obj.plan),
        layout_config_json=json.dumps(layout_config),
        user_placements_json=json.dumps(user_placements_for_template),
        image_filename=image_filename
    )


# ---------- Admin & Cookie Routes ----------
@app.route("/admin")
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    all_users = User.query.order_by(User.last_activity.desc()).all()
    online_threshold = datetime.utcnow() - timedelta(minutes=15)
    return render_template("admin_dashboard.html",
                           all_users=all_users,
                           online_threshold=online_threshold)


@app.route("/admin/user/<int:user_id>")
@admin_required
def admin_user_view(user_id):
    user_to_view = User.query.get_or_404(user_id)
    return render_template("admin_user_view.html", user=user_to_view)


@app.route("/cookie-settings")
def cookie_settings():
    return render_template("cookie_settings.html")


# ---------- /agent API ----------
@app.route("/agent", methods=["POST"])
@login_required
def agent_route():
    user = User.query.get(session["user_id"])
    user.last_activity = datetime.utcnow()
    plan_obj = get_or_create_plan_for_user(user.id)
    st = {"id": plan_obj.user_id, "plan": plan_obj.plan, "history": list(plan_obj.history)}
    body = request.get_json(force=True) or {}
    command, data = body.get("command"), body.get("data", {})
    if page_context := data.get("page_context"):
        user.last_seen_page = page_context
    db.session.commit()
    if command in {"save_answer", "skip_question", "submit_clarification", "question_selected"}:
        if not user.paid:
            ts = session.get("trial_start")
            if not ts or datetime.utcnow() - ts > timedelta(minutes=5):
                return jsonify({"status": "payment_required",
                                "message": "De gratis tijd is verstreken. Betaal om verder te gaan."})
    if command == "initialize":
        return jsonify({"session_id": st["id"], "state": st, "welcome_message": INTRO_MESSAGE})
    if command == "question_selected":
        tid = data.get("topic_id");
        tp, _ = find_topic_by_id(st["plan"], tid)
        if not tp: abort(404, "Topic niet gevonden")
        expl = f"**Toelichting bij \"{tp['question']}\"**:\n\n{tp['explanation']}"
        return jsonify({"session_id": st["id"], "state": st,
                        "explanation": expl, "topic_name": tp["name"]})
    if command in {"save_answer", "submit_clarification"}:
        tid = data.get("topic_id")
        original_answer = data.get("answer") if command == "save_answer" else data.get("original_answer")
        clar = data.get("clarification") if command == "submit_clarification" else None
        tp, th = find_topic_by_id(st["plan"], tid)
        if tp is None: abort(404, "Topic niet gevonden")
        user_input = clar if command == "submit_clarification" else original_answer
        if user_input: st["history"].append({"role": "user", "content": user_input})
        processed_answer = original_answer
        if command == "save_answer":
            try:
                processing_prompt = (
                    "Je bent een tekstverwerker voor een geboorteplan. Verwerk de volgende tekst van een gebruiker volgens deze regels:\n"
                    "1. Zorg dat de tekst maximaal 120 tekens lang is. Kort de tekst in of vat samen indien nodig, maar behoud de kernboodschap.\n"
                    "2. Identificeer in de resulterende tekst het allerbelangrijkste trefwoord of een korte woordgroep (2-3 woorden) en maak dit dikgedrukt met markdown (bijv. `**thuis bevallen**`).\n\n"
                    f"Originele tekst: \"{original_answer}\"\n\nVerwerkte tekst:"
                )
                response = client.chat.completions.create(
                    model=MODEL_CHOICE, messages=[{"role": "user", "content": processing_prompt}], temperature=0.4
                )
                processed_answer = response.choices[0].message.content.strip()
            except Exception as e:
                log.error(f"Fout bij verwerken van antwoord: {e}")
                processed_answer = original_answer[:120]
        plan_summary = "\n".join(
            f"- {t['name']} → '{top['question']}' = '{top.get('answer', '')}'"
            for t in st["plan"] for top in t["topics"]
            if top.get("answer") and top["id"] != tid and top["answer"] != "__SKIPPED__"
        ) or "Nog geen andere antwoorden gegeven."
        clar_txt = f"De gebruiker verduidelijkte: '{clar}'." if clar else ""
        v_prompt = (
            "Je bent een kritische maar vriendelijke verloskundige adviseur. "
            f"THEMA: {th['name']}. VRAAG: \"{tp['question']}\". ANTWOORD: \"{processed_answer}\". "
            f"{clar_txt}\nSAMENVATTING ANDERE KEUZES:\n{plan_summary}\n"
            "TAKEN:\n- Als het antwoord logisch en consistent is: antwoord exact 'OK'\n- Anders: stel een korte, vriendelijke tegenvraag."
        )
        try:
            resp = client.chat.completions.create(model=VALIDATOR_MODEL,
                                                  messages=[{"role": "user", "content": v_prompt}], temperature=0.2)
            valid = resp.choices[0].message.content.strip()
        except Exception:
            log.error("Validatie-call:", exc_info=True);
            valid = "OK"
        if valid == "OK":
            tp["answer"] = processed_answer
            st["history"].append(
                {"role": "system", "content": f"ANTWOORD OPGESLAGEN voor '{tp['name']}': '{processed_answer}'"})
            save_plan_state(plan_obj, st)
            final = FINAL_MESSAGE if is_plan_complete(st["plan"]) else ""
            return jsonify({"session_id": st["id"], "state": st, "status": "ok", "final_message": final})
        else:
            st["history"].append(
                {"role": "system", "content": f"ONGELDIG ANTWOORD voor '{tp['name']}': '{original_answer}'"})
            save_plan_state(plan_obj, st)
            return jsonify({"session_id": st["id"], "state": st, "status": "validation_failed", "feedback": valid,
                            "original_answer": original_answer})
    if command == "skip_question":
        tid = data.get("topic_id");
        tp, _ = find_topic_by_id(st["plan"], tid)
        if tp: tp["answer"] = "__SKIPPED__"
        st["history"].append({"role": "system", "content": f"VRAAG OVERGESLAGEN: '{tp['name']}'"})
        save_plan_state(plan_obj, st)
        final = FINAL_MESSAGE if is_plan_complete(st["plan"]) else ""
        return jsonify({"session_id": st["id"], "state": st, "status": "ok", "final_message": final})
    if command in {"start_guided_dialogue", "user_message"}:
        msg = data.get("message", "")
        if msg: st["history"].append({"role": "user", "content": msg})
        tp, _ = find_topic_by_id(st["plan"], data.get("topic_id"))
        if command == "start_guided_dialogue" and tp:
            context = f"De gebruiker wil hulp bij de vraag: '{tp['question']}'. Toelichting: '{tp['explanation']}'."
            system = f"BELANGRIJK: Je bent nu in 'begeleide dialoog'-modus. {context} Vat kort samen en stel daarna een open vraag."
        elif command == "user_message" and vector_retriever:
            ctx = "\n\n".join(d.page_content for d in vector_retriever.invoke(msg))
            system = "Je bent Mae, een behulpzame assistent in geboortezorg. Beantwoord uitsluitend op basis van de context. Als info ontbreekt, zeg dat eerlijk.\n---\n" + ctx + "\n---"
        else:
            system = "Je bent Mae, een behulpzame assistent."
        messages = [{"role": "system", "content": system}] + [m for m in st["history"][-7:] if m["role"] != "system"]

        def gen():
            full_content = ""
            for chunk in stream_llm_response(messages):
                yield chunk
                try:
                    json_str = chunk.split('data: ', 1)[1]
                    part = json.loads(json_str)
                    if 'content' in part:
                        full_content += part['content']
                except (IndexError, json.JSONDecodeError):
                    pass
            st["history"].append({"role": "assistant", "content": full_content})
            save_plan_state(plan_obj, st)

        return Response(stream_with_context(gen()), mimetype="text/event-stream")
    abort(400, "Onbekend command.")


# ---------- CLI Command voor Data Retentie ----------
@app.cli.command("anonymize-old-users")
def anonymize_old_users():
    """Vindt gebruikers wier uitgerekende datum > 1 maand geleden was en anonimiseert hen."""
    anonymization_date = datetime.utcnow().date() - timedelta(days=30)
    users_to_anonymize = User.query.filter(
        User.due_date < anonymization_date,
        User.is_anonymized == False
    ).all()
    log.info(f"Vond {len(users_to_anonymize)} gebruiker(s) om te anonimiseren.")
    for user in users_to_anonymize:
        try:
            log.info(f"Start anonimisatie voor user ID: {user.id}")
            if user.woman_dob:
                user.birth_year = user.woman_dob.year
            plan = user.birth_plan
            if plan:
                anonymized_history = [
                    {"role": "user", "content": msg["content"]}
                    for msg in plan.history if msg.get("role") == "user"
                ]
                plan.history = anonymized_history
            user.email = f"anonymized_{user.id}@example.com"
            user.username = f"anonymized_{user.id}"
            user.password_hash = None
            user.woman_name = "Geanonimiseerd"
            user.partner_name = None
            user.woman_dob = None
            user.midwifery_practice = None
            user.midwifery_phone = None
            user.woman_phone = None
            user.partner_phone = None
            user.medical_complications = None
            user.is_anonymized = True
            db.session.commit()
            log.info(f"User ID {user.id} succesvol geanonimiseerd.")
        except Exception as e:
            log.error(f"Fout bij anonimiseren van user ID {user.id}: {e}")
            db.session.rollback()
    print("✓ Anonimisatie taak voltooid.")


# ---------- CLI Command voor het aanmaken van een Admin ----------
@app.cli.command("create-admin")
@click.argument("email")
@click.argument("password")
@click.argument("name")
def create_admin(email, password, name):
    """Maakt een nieuwe admin-gebruiker aan."""
    if User.query.filter_by(email=email).first():
        print(f"Fout: Gebruiker met email {email} bestaat al.")
        return
    admin_user = User(
        email=email,
        username=email,
        password_hash=bcrypt.generate_password_hash(password).decode('utf-8'),
        woman_name=name,
        due_date=date.today(),
        is_admin=True,
        paid=True,
        is_confirmed=True  # Admins zijn direct bevestigd
    )
    db.session.add(admin_user)
    db.session.commit()
    print(f"✓ Admin-gebruiker '{name}' met email '{email}' succesvol aangemaakt.")
