#!/usr/bin/env python3
# app.py – Geboorteplan-assistent • Versie 12.0  (timer + Mollie-betaling)

import re, os, json, logging, pathlib
from typing import Any, Dict, Optional, Generator, List
from datetime import date, timedelta, datetime

from flask import (
    Flask, request, jsonify, abort, Response, stream_with_context,
    render_template, redirect, url_for, session, flash
)
from flask_bcrypt    import Bcrypt
from flask_cors      import CORS
from flask_session   import Session          # server-side sessions
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv          import load_dotenv
from openai          import OpenAI
from mollie.api.client import Client as MollieClient  # Mollie

# ── lokale modules ────────────────────────────────
from database import db, User, BirthPlan
from langchain_community.vectorstores   import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter            import CharacterTextSplitter
from langchain_huggingface              import HuggingFaceEmbeddings

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

    # Server-side sessions  (Flask-Session + SQLAlchemy)
    SESSION_TYPE="sqlalchemy",
    SESSION_PERMANENT=True,
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
    SESSION_USE_SIGNER=True,
    SESSION_SQLALCHEMY_TABLE="sessions",

    # Cookie-instellingen  – embed in iFrame ⇒ SameSite=None
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_SAMESITE="None",
    SESSION_COOKIE_HTTPONLY=True,
)

# ── extensies ─────────────────────────────────────
db.init_app(app)
app.config["SESSION_SQLALCHEMY"] = db
sess   = Session(app)
bcrypt = Bcrypt(app)

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
client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_CHOICE     = os.getenv("MODEL_CHOICE",     "gpt-4o-mini")
VALIDATOR_MODEL  = os.getenv("VALIDATOR_MODEL",  "gpt-4o")

PLAN_TEMPLATE_FILE = ROOT / "geboorteplan_template.json"
KNOWLEDGE_BASE_FILE= ROOT / "kennisbank.md"
VECTOR_DB_PATH     = ROOT / "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

vector_retriever = None
try:
    log.info("Laden van embedding-model …")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if not VECTOR_DB_PATH.exists():
        log.warning("Vector-db niet gevonden – bouwen …")
        if KNOWLEDGE_BASE_FILE.exists():
            loader = TextLoader(str(KNOWLEDGE_BASE_FILE), encoding="utf-8")
            docs   = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)\
                     .split_documents(loader.load())
            FAISS.from_documents(docs, embeddings)\
                 .save_local(str(VECTOR_DB_PATH))
    vector_db = FAISS.load_local(str(VECTOR_DB_PATH), embeddings,
                                 allow_dangerous_deserialization=True)
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
    "Met het beantwoorden van de volgende vragenlijst proberen we jouw "
    "wensen en voorkeuren op te nemen in je persoonlijk bevalplan. …"
)
FINAL_MESSAGE = (
    "Dit bevalplan beschrijft jouw ideale bevalling. Niet alle bevallingen "
    "verlopen volgens plan …"
)

# ── helpers ───────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*a, **kw):
        if "user_id" not in session:
            flash("Je moet ingelogd zijn om deze pagina te bekijken.", "error")
            return redirect(url_for("login"))
        return f(*a, **kw)
    return wrapper

def get_or_create_plan_for_user(uid: int) -> BirthPlan:
    plan = BirthPlan.query.filter_by(user_id=uid).first()
    if plan: return plan
    with open(PLAN_TEMPLATE_FILE, encoding="utf-8") as f:
        tpl = json.load(f)
    plan = BirthPlan(user_id=uid, plan=tpl, history=[])
    db.session.add(plan); db.session.commit()
    return plan

def save_plan_state(obj: BirthPlan, st: Dict[str, Any]):
    obj.plan, obj.history = st["plan"], st["history"]; db.session.commit()

def find_topic_by_id(plan: list, tid: str):
    for th in plan:
        for tp in th["topics"]:
            if tp["id"] == tid: return tp, th
    return None, None

def is_plan_complete(plan: list) -> bool:
    return all(tp.get("answer") for th in plan for tp in th["topics"])

def stream_llm_response(msgs:list) -> Generator[str,None,None]:
    try:
        for ch in client.chat.completions.create(model=MODEL_CHOICE,
                                                 messages=msgs, stream=True):
            if (c := ch.choices[0].delta.content):
                yield f"data: {json.dumps({'content': c})}\n\n"
    except Exception:
        yield f"data: {json.dumps({'error':'LLM-fout'})}\n\n"
        log.error("Streaming-LLM:", exc_info=True)

def render_mobile_aware_template(desktop, **kw):
    force_m = request.args.get("mobile") == "true"
    is_mob  = force_m or "mobile" in request.user_agent.string.lower()
    if is_mob:
        mob = f"mobile_{desktop}"
        if (pathlib.Path(app.template_folder) / mob).exists():
            return render_template(mob, **kw)
    return render_template(desktop, **kw)

# ───────── routes ─────────
@app.cli.command("init-db")
def init_db():
    with app.app_context(): db.create_all()
    print("✓ tabellen aangemaakt")

@app.route("/")
def root(): return redirect(url_for("dashboard") if "user_id" in session else "login")

@app.route("/logout")
def logout():
    session.clear(); flash("Je bent uitgelogd.","success"); return redirect(url_for("login"))

# ---------- registratie ----------
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        f = request.form
        required = [f.get(x,"").strip() for x in
                    ("email","username","password","woman_name","due_date")]
        if not all(required):
            flash("Vul alle verplichte velden in.","error")
            return redirect(url_for("register"))
        if User.query.filter((User.email==required[0])|(User.username==required[1])).first():
            flash("E-mail of gebruikersnaam al in gebruik.","error")
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
            paid=True
        )
        db.session.add(user); db.session.commit()
        get_or_create_plan_for_user(user.id)
        flash("Account aangemaakt! Log nu in.","success")
        return redirect(url_for("login"))
    return render_mobile_aware_template("register.html")

# ---------- login ----------
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        email, pw = request.form.get("email"), request.form.get("password")
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password_hash, pw):
            session["user_id"]=user.id; return redirect(url_for("dashboard"))
        flash("Inloggen mislukt.","error"); return redirect(url_for("login"))
    return render_mobile_aware_template("login.html")

# ---------- dashboard ----------
@app.route("/dashboard")
@login_required
def dashboard():
    return render_mobile_aware_template("dashboard.html",
                                        user=User.query.get(session["user_id"]))

# ---------- trial & betaling ----------
@app.route("/start_trial")
@login_required
def start_trial():
    session["trial_start"]=datetime.utcnow()
    return redirect(url_for("vragenlijst"))

@app.route("/start_payment")
@login_required
def start_payment():
    if not MOLLIE_KEY:
        flash("Betalen is tijdelijk niet beschikbaar.","error")
        return redirect(url_for("dashboard"))
    user = User.query.get(session["user_id"])
    payment = mollie_client.payments.create({
        "amount":{"currency":"EUR","value":"9.99"},
        "description":"Betaling geboorteplan",
        "redirectUrl": url_for("payment_return", _external=True),
        "webhookUrl":  url_for("payment_webhook", _external=True),
        "method":["ideal"],
        "metadata":{"user_id":user.id}
    })
    return redirect(payment.get("checkout_url", url_for("dashboard")))

@app.route("/payment_return")
@login_required
def payment_return():
    pid = request.args.get("id")
    try:
        if mollie_client.payments.get(pid).is_paid():
            u=User.query.get(session["user_id"]); u.paid=True; db.session.commit()
            flash("Betaling gelukt!","success")
            return redirect(url_for("vragenlijst"))
    except Exception:
        log.error("Mollie return:", exc_info=True)
    flash("Betaling geannuleerd of mislukt.","error")
    return redirect(url_for("dashboard"))

@app.route("/payment_webhook", methods=["POST"])
def payment_webhook():
    pid = request.form.get("id")
    try:
        p = mollie_client.payments.get(pid)
        if p.is_paid():
            u = User.query.get(p.metadata["user_id"]); u.paid=True; db.session.commit()
    except Exception: log.error("Webhook:", exc_info=True)
    return "", 200

# ---------- vragenlijst ----------
@app.route("/vragenlijst")
@login_required
def vragenlijst():
    u = User.query.get(session["user_id"])
    if not u.paid:
        ts = session.get("trial_start")
        if not ts:
            flash("Kies eerst gratis proef of betaal.","error")
            return redirect(url_for("dashboard"))
        if datetime.utcnow() - ts > timedelta(minutes=5):
            flash("Gratis tijd verstreken. Betaal om verder te gaan.","error")
            return redirect(url_for("dashboard"))
    return render_mobile_aware_template("index.html")

# ---------- download ----------
@app.route("/download_plan")
@login_required
def download_plan():
    u = User.query.get(session["user_id"])
    if not u.paid:
        flash("Betaal eerst om te kunnen downloaden.","error")
        return redirect(url_for("dashboard"))
    plan_obj = get_or_create_plan_for_user(u.id)
    resp = jsonify(plan_obj.plan)
    resp.headers["Content-Disposition"] = "attachment; filename=geboorteplan.json"
    return resp

# ---------- /agent API ----------
@app.route("/agent", methods=["POST"])
@login_required
def agent_route():
    user_id = session["user_id"]
    user    = User.query.get(user_id)
    plan_obj = get_or_create_plan_for_user(user_id)
    st = {"id": plan_obj.user_id, "plan": plan_obj.plan, "history": plan_obj.history}

    body = request.get_json(force=True) or {}
    command = body.get("command"); data = body.get("data", {})

    # ---- gratis tijd check voor muterende commands ----
    if command in {"save_answer","skip_question","submit_clarification","question_selected"}:
        if not user.paid:
            ts = session.get("trial_start")
            if not ts or datetime.utcnow() - ts > timedelta(minutes=5):
                return jsonify({"status":"payment_required",
                                "message":"De gratis tijd is verstreken. Betaal om verder te gaan."})

    # ---- initialize ----
    if command == "initialize":
        return jsonify({"session_id":st["id"],"state":st,
                        "welcome_message":INTRO_MESSAGE})

    # ---- question_selected ----
    if command == "question_selected":
        tid = data.get("topic_id"); tp, _ = find_topic_by_id(st["plan"], tid)
        if not tp: abort(404,"Topic niet gevonden")
        expl = f"**Toelichting bij \"{tp['question']}\"**:\n\n{tp['explanation']}"
        return jsonify({"session_id":st["id"],"state":st,
                        "explanation":expl,"topic_name":tp["name"]})

    # ---- save_answer / submit_clarification ----
    if command in {"save_answer","submit_clarification"}:
        tid = data.get("topic_id")
        orig = data.get("answer") if command=="save_answer" else data.get("original_answer")
        clar = data.get("clarification") if command=="submit_clarification" else None
        tp, th = find_topic_by_id(st["plan"], tid)
        if tp is None: abort(404,"Topic niet gevonden")

        plan_summary = "\n".join(
            f"- {t['name']} → '{top['question']}' = '{top.get('answer','')}'"
            for t in st["plan"] for top in t["topics"]
            if top.get("answer") and top["id"]!=tid and top["answer"]!="__SKIPPED__"
        ) or "Nog geen andere antwoorden gegeven."
        clar_txt = f"De gebruiker verduidelijkte: '{clar}'." if clar else ""
        v_prompt = (
            "Je bent een kritische maar vriendelijke verloskundige adviseur. "
            f"THEMA: {th['name']}. VRAAG: \"{tp['question']}\". ANTWOORD: \"{orig}\". "
            f"{clar_txt}\nSAMENVATTING ANDERE KEUZES:\n{plan_summary}\n"
            "TAKEN:\n"
            "- Als het antwoord logisch en consistent is: antwoord exact 'OK'\n"
            "- Anders: stel een korte, vriendelijke tegenvraag."
        )
        try:
            resp = client.chat.completions.create(model=VALIDATOR_MODEL,
                    messages=[{"role":"user","content":v_prompt}], temperature=0.2)
            valid = resp.choices[0].message.content.strip()
        except Exception:
            log.error("Validatie-call:", exc_info=True); valid="OK"

        if valid=="OK":
            tp["answer"] = orig
            st["history"].append({"role":"system",
                                  "content":f"ANTWOORD OPGESLAGEN voor '{tp['name']}': '{orig}'"})
            save_plan_state(plan_obj, st)
            final = FINAL_MESSAGE if is_plan_complete(st["plan"]) else ""
            return jsonify({"session_id":st["id"],"state":st,"status":"ok",
                            "final_message":final})
        else:
            st["history"].append({"role":"system",
                                  "content":f"ONGELDIG ANTWOORD voor '{tp['name']}': '{orig}'"})
            save_plan_state(plan_obj, st)
            return jsonify({"session_id":st["id"],"state":st,
                            "status":"validation_failed","feedback":valid,
                            "original_answer":orig})

    # ---- skip_question ----
    if command == "skip_question":
        tid = data.get("topic_id"); tp, _ = find_topic_by_id(st["plan"], tid)
        if tp: tp["answer"]="__SKIPPED__"
        st["history"].append({"role":"system","content":f"VRAAG OVERGESLAGEN: '{tp['name']}'"})
        save_plan_state(plan_obj, st)
        final = FINAL_MESSAGE if is_plan_complete(st["plan"]) else ""
        return jsonify({"session_id":st["id"],"state":st,"status":"ok",
                        "final_message":final})

    # ---- guided_dialogue / user_message ----
    if command in {"start_guided_dialogue","user_message"}:
        msg = data.get("message","")
        if msg: st["history"].append({"role":"user","content":msg})
        tp,_ = find_topic_by_id(st["plan"], data.get("topic_id"))
        if command=="start_guided_dialogue" and tp:
            context = (f"De gebruiker wil hulp bij de vraag: '{tp['question']}'. "
                       f"Toelichting: '{tp['explanation']}'.")
            system = ("BELANGRIJK: Je bent nu in 'begeleide dialoog'-modus. "
                      f"{context} Vat kort samen en stel daarna een open vraag.")
        elif command=="user_message" and vector_retriever:
            ctx = "\n\n".join(d.page_content for d in
                              vector_retriever.invoke(msg))
            system = ("Je bent Mae, een behulpzame assistent in geboortezorg. "
                      "Beantwoord uitsluitend op basis van de context. "
                      "Als info ontbreekt, zeg dat eerlijk.\n---\n"+ctx+"\n---")
        else:
            system = "Je bent Mae, een behulpzame assistent."

        messages = [{"role":"system","content":system}] + \
                   [m for m in st["history"][-7:] if m["role"]!="system"]
        def gen():
            full=""
            for chunk in stream_llm_response(messages):
                part=json.loads(chunk[6:])  # strip 'data: '
                if part.get("content"):
                    full = part["content"]
                    yield f"data:{json.dumps(part)}\n\n"
            st["history"].append({"role":"assistant","content":full})
        return Response(stream_with_context(gen()),
                        mimetype="text/event-stream")

    abort(400,"Onbekend command.")
