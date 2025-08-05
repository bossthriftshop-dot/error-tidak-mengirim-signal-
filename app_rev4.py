import sqlite3
import os
import time
import logging
import threading
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template, session, redirect, url_for, send_from_directory, flash, g
from flask_socketio import SocketIO
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import secrets
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField, HiddenField
from wtforms.validators import DataRequired, EqualTo, Regexp
from wtforms import BooleanField
from dateutil.relativedelta import relativedelta

app = Flask(__name__)
app.secret_key = secrets.token_hex(24)
app.config['SECRET_KEY'] = app.secret_key # Fix for 500 error on login form
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")
DATABASE_FILE = 'users.db'
UPLOAD_FOLDER = 'static/bukti_pembayaran'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

INTERNAL_SECRET_KEY = "c1b086d4-a681-48df-957f-6fcc35a82f6d"
last_signal_info = {}
signal_context_cache = {} # Cache untuk menyimpan konteks sinyal untuk pembelajaran
SYMBOL_ALIAS_MAP = {
    'XAUUSD': 'XAUUSD', 'XAUUSDc': 'XAUUSD', 'XAUUSDm': 'XAUUSD', 'GOLD': 'XAUUSD',
    'BTCUSD': 'BTCUSD', 'BTCUSDc': 'BTCUSD', 'BTCUSDm': 'BTCUSD',
}

feedback_file_lock = threading.Lock()
db_write_lock = threading.Lock()

open_positions_map = {}
open_positions_lock = threading.Lock()

def get_open_positions(api_key, symbol):
    key = f"{api_key}_{symbol}"
    with open_positions_lock:
        return open_positions_map.get(key, [])

# ====================================================================
# --- PERBAIKAN: Fungsi ini sekarang bisa "melupakan" posisi lama ---
# ====================================================================
def is_too_close_to_open_position(new_entry, open_positions, pip_threshold=100.0, max_age_hours=8):
    """
    Memeriksa kedekatan dengan posisi, tetapi hanya untuk posisi yang 'baru' (kurang dari max_age_hours).
    """
    now = datetime.now()
    recent_positions = []

    # Filter untuk hanya mengambil posisi yang masih relevan
    for pos in open_positions:
        try:
            # Mengonversi waktu simpan posisi dari string ke objek datetime
            pos_time = datetime.fromisoformat(pos['time'])
            # Jika posisi lebih baru dari batas waktu maksimal, anggap masih relevan
            if (now - pos_time) < timedelta(hours=max_age_hours):
                recent_positions.append(pos)
        except (ValueError, KeyError):
            # Jika ada error pada data waktu, anggap saja relevan untuk keamanan
            recent_positions.append(pos)

    # Lakukan pemeriksaan jarak hanya pada posisi yang relevan
    for pos in recent_positions:
        try:
            if abs(float(pos['entry']) - float(new_entry)) < pip_threshold:
                return True
        except (ValueError, TypeError):
            continue

    return False
# ====================================================================


# === FORM DAN DB ===
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Ulangi Password', validators=[
        DataRequired(),
        EqualTo('password', message='Password dan Ulangi Password harus sama')
    ])
    whatsapp_number = StringField('No. WhatsApp', validators=[
        DataRequired(),
        Regexp(r'^\+\d{8,16}$', message='Format Nomor WhatsApp tidak valid (contoh: +6281234567890)')
    ])
    agree_terms = BooleanField('Saya menyetujui Syarat & Ketentuan', validators=[DataRequired(message='Anda harus menyetujui Syarat & Ketentuan')])
    submit = SubmitField('Register')

class SubscribeForm(FlaskForm):
    duration = HiddenField('Duration', validators=[DataRequired()])
    proof_file = FileField('Unggah Bukti Pembayaran', validators=[DataRequired()])
    submit = SubmitField('Kirim Bukti Pembayaran')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_signal_id(api_key, order_type, timestamp):
    data = f"{api_key}:{order_type}:{timestamp}"
    return hashlib.md5(data.encode()).hexdigest()

def get_user_license_details(user_data):
    end_date_obj = datetime.strptime(user_data['end_date'], '%Y-%m-%d').date()
    today = datetime.now().date()
    status = user_data['status']

    if status == 'pending_activation':
        return "Menunggu Aktivasi", "bg-warning"
    elif today <= end_date_obj and status in ['active', 'trial']:
        return "Aktif", "bg-success"
    else:
        return "Kadaluarsa", "bg-danger"

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE_FILE, check_same_thread=False, timeout=30)
    g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db_data():
    with app.app_context():
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                api_key TEXT UNIQUE NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'trial',
                proof_filename TEXT DEFAULT NULL,
                duration_pending INTEGER DEFAULT NULL,
                whatsapp_number TEXT UNIQUE
            )
        ''')
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN whatsapp_number TEXT")
        except sqlite3.OperationalError:
            pass # Kolom sudah ada
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        cursor.execute("SELECT COUNT(*) FROM admins WHERE username = 'admin'")
        if cursor.fetchone()[0] == 0:
            default_admin_password = generate_password_hash('admin123')
            cursor.execute("INSERT INTO admins (username, password) VALUES (?, ?)", ('admin', default_admin_password))
        conn.commit()

def is_api_key_valid(api_key):
    conn = get_db()
    user = conn.execute("SELECT start_date, end_date, status FROM users WHERE api_key = ?", (api_key,)).fetchone()
    if user:
        license_end = datetime.strptime(user['end_date'], '%Y-%m-%d').date()
        today = datetime.now().date()
        return today <= license_end and user['status'] in ['trial', 'active']
    return False

@app.before_request
def require_login():
    public_routes = [
        'login_page', 'register_page', 'get_signal', 'admin_login', 'static',
        'status_page', 'receive_signal', 'feedback_trade', 'index', 'home_page', 'panduan_page'
    ]
    if request.path.startswith('/admin') and 'admin_id' in session:
        return
    if request.endpoint in public_routes or (request.endpoint and 'static' in request.endpoint):
        return
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

@app.route('/')
def index():
    return redirect(url_for('home_page'))

@app.route('/home')
def home_page():
    if 'user_id' in session:
        return redirect(url_for('dashboard_page'))
    return render_template('home.html', current_year=datetime.now().year)

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    form = RegisterForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        whatsapp_number = form.whatsapp_number.data
        conn = get_db()
        try:
            api_key = str(uuid.uuid4())
            today = datetime.now().date()
            end_date = today + relativedelta(days=7)
            conn.execute('''
                INSERT INTO users (username, password, api_key, start_date, end_date, status, whatsapp_number)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (username, generate_password_hash(password), api_key, today.isoformat(), end_date.isoformat(), 'trial', whatsapp_number))
            conn.commit()
            flash('Registrasi berhasil! Silakan login.', 'success')
            return redirect(url_for('login_page'))
        except Exception as e:
            flash(f'Gagal registrasi: {e}', 'danger')
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({"status": "error", "message": "Username dan password diperlukan"}), 400

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session.permanent = True # Sesi akan bertahan lebih lama
            return jsonify({"status": "success", "redirect_url": url_for('dashboard_page')})

        return jsonify({"status": "error", "message": "Username atau password salah"}), 401

    # Untuk request GET, tampilkan halaman login
    form = LoginForm()
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Anda telah logout.', 'info')
    return redirect(url_for('login_page'))

@app.route('/dashboard')
def dashboard_page():
    user_id = session.get('user_id')
    if not user_id: return redirect(url_for('login_page'))
    conn = get_db()
    user_data = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user_data: return redirect(url_for('logout'))
    status_text, badge_class = get_user_license_details(user_data)
    current_signal = last_signal_info.get(user_data['api_key'], last_signal_info.get(INTERNAL_SECRET_KEY))
    return render_template('index.html', user=user_data, license_status=status_text, badge_class=badge_class, last_signal=current_signal)

@app.route('/lisensi')
def lisensi_page():
    user_id = session.get('user_id')
    if not user_id: return redirect(url_for('login_page'))
    conn = get_db()
    user_data = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    status_text, badge_class = get_user_license_details(user_data)
    return render_template('lisensi.html', api_key=user_data['api_key'], start_date=user_data['start_date'], end_date=user_data['end_date'], license_status=status_text, badge_class=badge_class)

@app.route('/panduan')
def panduan_page():
    return render_template('panduan.html')

@app.route('/subscribe')
def subscribe_page():
    form = SubscribeForm()
    return render_template('subscribe.html', form=form)

@app.route('/upload_proof', methods=['POST'])
def upload_proof():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    form = SubscribeForm()
    if form.validate_on_submit():
        file = form.proof_file.data
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{session['user_id']}_{int(time.time())}_{file.filename}")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            conn = get_db()
            conn.execute("UPDATE users SET proof_filename = ?, duration_pending = ? WHERE id = ?", (filename, form.duration.data, session['user_id']))
            conn.commit()
            flash("Bukti pembayaran berhasil diupload! Silakan tunggu aktivasi.", "success")
            return redirect(url_for('dashboard_page'))
    flash("Format file tidak valid!", "danger")
    return redirect(url_for('subscribe_page'))

@app.route('/status')
def status_page():
    api_key_to_check = INTERNAL_SECRET_KEY
    if 'user_id' in session:
        conn = get_db()
        user_data = conn.execute("SELECT api_key FROM users WHERE id = ?", (session['user_id'],)).fetchone()
        if user_data:
            api_key_to_check = user_data['api_key']
    current_signal = last_signal_info.get(api_key_to_check, last_signal_info.get(INTERNAL_SECRET_KEY))
    return render_template('status.html', last_signal=current_signal)

@app.route('/api/get_signal', methods=['GET'])
def get_signal():
    api_key = request.args.get('key')
    if not api_key or not is_api_key_valid(api_key):
        return jsonify({"error": "Unauthorized. Invalid or expired API Key."}), 401

    symbol = request.args.get('symbol', 'XAUUSD').upper()
    mapped_symbol = SYMBOL_ALIAS_MAP.get(symbol, 'XAUUSD')
    signal_data_key = f"{api_key}_{mapped_symbol}"
    signal_data = last_signal_info.get(signal_data_key)

    if signal_data and (datetime.now() - datetime.strptime(signal_data['timestamp'], '%Y-%m-%d %H:%M:%S')).total_seconds() < 300:
        response_data = signal_data['signal_json']
        response_data.update({"signal_id": signal_data['signal_id'], "order_type": signal_data['order_type']})
        return jsonify(response_data)

    return jsonify({"order_type": "WAIT"})

@app.route('/api/internal/submit_signal', methods=['POST'])
def receive_signal():
    global last_signal_info, open_positions_map, signal_context_cache
    data = request.json
    if not data: return jsonify({"error": "No data received"}), 400

    api_key = data.get('api_key', INTERNAL_SECRET_KEY)
    symbol = data.get('symbol', 'XAUUSD').upper()
    mapped_symbol = SYMBOL_ALIAS_MAP.get(symbol, 'XAUUSD')
    signal_type = data.get('signal')
    order_type = data.get('order_type', signal_type)
    signal_json = data.get('signal_json', {})
    score = data.get('score')
    info = data.get('info')
    profile_name = data.get('profile_name') # Ambil nama profil

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    signal_id = generate_signal_id(api_key, order_type, timestamp)

    # Simpan konteks untuk pembelajaran nanti, sekarang termasuk profil
    signal_context_cache[signal_id] = {
        "score": score,
        "info": info,
        "timestamp": timestamp,
        "symbol": symbol,
        "order_type": order_type,
        "profile_name": profile_name
    }

    entry_price = None
    if signal_type == 'BUY':
        entry_price = signal_json.get('BuyEntry') or signal_json.get('BuyStop') or signal_json.get('Buylimit')
    elif signal_type == 'SELL':
        entry_price = signal_json.get('SellEntry') or signal_json.get('SellStop') or signal_json.get('Selllimit')

    if not entry_price:
        entry_price = None

    if entry_price is not None:
        pip_threshold = 100.0
        open_positions = get_open_positions(api_key, mapped_symbol)

        if is_too_close_to_open_position(entry_price, open_positions, pip_threshold):
            logging.warning(f"Sinyal ditolak oleh server: Entry {entry_price} terlalu dekat dengan posisi aktif (pips < {pip_threshold}), sinyal di-skip")
            return jsonify({"error": f"Entry {entry_price} terlalu dekat dengan posisi aktif, sinyal di-skip"}), 409

        key = f"{api_key}_{mapped_symbol}"
        pos = {"entry": float(entry_price), "type": signal_type, "time": datetime.now().isoformat()}
        with open_positions_lock:
            if key not in open_positions_map: open_positions_map[key] = []
            now = datetime.now()
            open_positions_map[key] = [p for p in open_positions_map.get(key, []) if (now - datetime.fromisoformat(p['time'])) < timedelta(hours=8)]
            open_positions_map[key].append(pos)

    signal_payload = {
        'signal_id': signal_id, # Gunakan signal_id yang sudah dibuat
        'order_type': order_type,
        'timestamp': timestamp,
        'signal_json': signal_json
    }

    conn = get_db()
    all_api_keys = [row['api_key'] for row in conn.execute("SELECT api_key FROM users WHERE status IN ('active','trial')").fetchall()]
    for user_api_key in all_api_keys:
        key_to_update = f"{user_api_key}_{mapped_symbol}"
        last_signal_info[key_to_update] = signal_payload.copy()

    logging.info(f"📢 Sinyal BROADCAST: Type={order_type}, Simbol={symbol}.")
    return jsonify({"message": "Signal received and broadcasted"}), 200

@app.route("/api/feedback_trade", methods=["POST"])
def feedback_trade():
    data = request.json
    if not data: return jsonify({"status": "error", "message": "No data received"}), 400

    # Ambil konteks dari cache
    signal_id = data.get('signal_id')
    context = signal_context_cache.pop(signal_id, {}) # Ambil dan hapus dari cache

    # Gabungkan data feedback dengan konteks
    full_feedback = {**context, **data}

    feedback_path = "trade_feedback.json"
    with feedback_file_lock:
        try:
            with open(feedback_path, "r+") as f:
                current_data = json.load(f)
                current_data.append(full_feedback)
                f.seek(0)
                json.dump(current_data, f, indent=2)
        except (FileNotFoundError, json.JSONDecodeError):
            with open(feedback_path, "w") as f:
                json.dump([full_feedback], f, indent=2)
    return jsonify({"status": "success"}), 200

# === ADMIN ENDPOINTS ===
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        conn = get_db()
        admin = conn.execute("SELECT * FROM admins WHERE username = ?", (username,)).fetchone()
        if admin and check_password_hash(admin['password'], password):
            session['admin_id'] = admin['id']
            return redirect(url_for('admin_dashboard'))
        flash('Username atau password admin salah', 'danger')
    return render_template('admin_login.html', form=form)

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_id', None)
    return redirect(url_for('admin_login'))

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'admin_id' not in session: return redirect(url_for('admin_login'))
    conn = get_db()
    users_raw = conn.execute("SELECT * FROM users ORDER BY id DESC").fetchall()
    users_data = []
    for user in users_raw:
        user_dict = dict(user)
        user_dict['status_text'], user_dict['badge_class'] = get_user_license_details(user_dict)
        if user_dict['proof_filename']:
            user_dict['proof_url'] = url_for('static', filename=f"bukti_pembayaran/{user_dict['proof_filename']}")
        users_data.append(user_dict)
    return render_template('admin_dashboard.html', users=users_data)

@app.route('/admin/activate_license/<int:user_id>', methods=['POST'])
def admin_activate_license(user_id):
    if 'admin_id' not in session: return jsonify({"error": "Unauthorized"}), 401
    conn = get_db()
    user_info = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user_info: return jsonify({"error": "Pengguna tidak ditemukan."}), 404

    duration_months = int(request.form.get('duration_months', user_info['duration_pending']))
    if not duration_months: return jsonify({"error": "Durasi tidak valid."}), 400

    current_end_date = datetime.strptime(user_info['end_date'], '%Y-%m-%d').date()
    base_date = max(current_end_date, datetime.now().date())
    new_end_date = base_date + relativedelta(months=duration_months)
    new_start_date = user_info['start_date'] if current_end_date > datetime.now().date() else datetime.now().date().isoformat()

    conn.execute("""
        UPDATE users SET start_date = ?, end_date = ?, status = 'active',
        proof_filename = NULL, duration_pending = NULL WHERE id = ?
    """, (new_start_date, new_end_date.isoformat(), user_id))
    conn.commit()
    return jsonify({"message": "Lisensi berhasil diaktifkan."})

@app.route('/download/ea')
def download_ea():
    if 'user_id' not in session: return redirect(url_for('login_page'))
    conn = get_db()
    user_info = conn.execute("SELECT status, end_date, api_key FROM users WHERE id = ?", (session['user_id'],)).fetchone()
    if not user_info or not is_api_key_valid(user_info['api_key']):
        flash("Lisensi Anda tidak aktif. Silakan perpanjang lisensi untuk men-download.", 'warning')
        return redirect(url_for('dashboard_page'))

    ea_directory = os.path.join(app.root_path, 'static')
    ea_filename = 'Esteh AI Update.zip'
    return send_from_directory(directory=ea_directory, path=ea_filename, as_attachment=True)

# === SOCKET.IO EVENTS ===
@socketio.on('connect')
def handle_connect():
    """Event handler for new client connections."""
    logging.info(f"Browser client terhubung: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Event handler for client disconnections."""
    logging.info(f"Browser client terputus: {request.sid}")

@socketio.on('submit_log')
def handle_submit_log(data):
    """
    Event handler for receiving logs from the bot script client.
    It then broadcasts the log to all connected browser clients.
    """
    # 'new_log' is the event that the browser clients will listen for.
    socketio.emit('new_log', {'message': data.get('message', '')})

# ========== INIT & RUN ==========
if __name__ == '__main__':
    os.makedirs(os.path.join(app.root_path, 'static'), exist_ok=True)
    os.makedirs(os.path.join(app.root_path, app.config['UPLOAD_FOLDER']), exist_ok=True)
    with app.app_context():
        init_db_data()
    # app.run(host='0.0.0.0', port=5000, debug=False) # Old way
    socketio.run(app, host='0.0.0.0', port=5000, debug=False) # New way with SocketIO
