import os
import time
import socket
import psutil
import joblib
import numpy as np
import pytz
import threading
import mysql.connector
import winsound
from datetime import datetime
from plyer import notification
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from getpass import getpass

# ======================================================
# 🌍 ENVIRONMENT SETUP
# ======================================================
load_dotenv()
IST = pytz.timezone("Asia/Kolkata")

# Ask IP dynamically if not found in .env
DB_HOST = os.getenv("DB_HOST")
if not DB_HOST:
    DB_HOST = input("Enter database host IP (e.g., 10.132.93.84): ").strip()

DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "StrongPassword123")
DB_NAME = os.getenv("DB_NAME", "CPUMETRIC")

# Retry connection setup
print(f"🔗 Connecting to MySQL host {DB_HOST}...")
while True:
    try:
        conn_test = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
        )
        conn_test.close()
        print("✅ Database connection successful.")
        break
    except Exception as e:
        print(f"⚠ Database not reachable ({e}). Retrying in 5s...")
        time.sleep(5)

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# ======================================================
# 🔹 DATABASE MODELS
# ======================================================
class Admin(db.Model):
    __tablename__ = "admin"
    admin_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    password_hash = db.Column(db.String(255), nullable=False)


class SystemInfo(db.Model):
    __tablename__ = "system_info"
    system_id = db.Column(db.Integer, primary_key=True)
    system_name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(150))
    ip_address = db.Column(db.String(50))
    admin_id = db.Column(db.Integer, db.ForeignKey("admin.admin_id"))


class SystemMetrics(db.Model):
    __tablename__ = "system_metrics"
    metric_id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    system_id = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(IST))
    CPU_Usage = db.Column(db.Float)
    Memory_Usage = db.Column(db.Float)
    Disk_IO = db.Column(db.Float)
    Network_Latency = db.Column(db.Float)
    Error_Rate = db.Column(db.Float)


class PredictionLog(db.Model):
    __tablename__ = "prediction_log"
    prediction_id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    system_id = db.Column(db.Integer)
    downtime_risk = db.Column(db.Boolean)
    probability = db.Column(db.Float)
    estimated_time_to_downtime = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(IST))


class Notification(db.Model):
    __tablename__ = "notification"
    notification_id = db.Column(db.Integer, primary_key=True)
    admin_id = db.Column(db.Integer)
    system_id = db.Column(db.Integer)
    message = db.Column(db.String(255))
    risk_level = db.Column(db.String(50))
    status = db.Column(db.String(50))
    sent_time = db.Column(db.DateTime)


# ======================================================
# 🔐 LOGIN / REGISTRATION
# ======================================================
def ensure_admin():
    """Check or create admin safely."""
    with app.app_context():
        choice = input("\nDo you already have an admin account? (y/n): ").strip().lower()

        if choice == "y":
            email = input("Enter your email: ").strip()
            password = getpass("Enter your password: ").strip()
            admin = Admin.query.filter_by(email=email, password_hash=password).first()
            if admin:
                print(f"✅ Welcome back, {admin.name}!")
                return admin
            else:
                print("❌ Invalid credentials. Try again or register.")
                return ensure_admin()
        else:
            print("\n📝 Registering new admin.")
            name = input("Enter your full name: ").strip()
            email = input("Enter your email: ").strip()
            phone = input("Enter contact number: ").strip()
            password = getpass("Set your password: ").strip()

            # Prevent duplicate
            existing = Admin.query.filter_by(email=email).first()
            if existing:
                print("⚠ Admin already exists, logging you in.")
                return existing

            admin = Admin(name=name, email=email, phone=phone, password_hash=password)
            db.session.add(admin)
            db.session.commit()
            print("✅ Registration successful.")
            return admin


# ======================================================
# 🔹 METRIC COLLECTION
# ======================================================
def collect_metrics():
    return {
        "CPU_Usage": psutil.cpu_percent(interval=1),
        "Memory_Usage": psutil.virtual_memory().percent,
        "Disk_IO": psutil.disk_usage("/").percent,
        "Network_Latency": np.random.uniform(1, 10),
        "Error_Rate": np.random.uniform(0, 0.05),
        "timestamp": datetime.now(IST)
    }


# ======================================================
# 🔹 PREDICTION & LOGGING
# ======================================================
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

def make_prediction(admin, metrics):
    with app.app_context():
        ip_addr = socket.gethostbyname(socket.gethostname())
        system = SystemInfo.query.filter_by(ip_address=ip_addr).first()
        if not system:
            system = SystemInfo(
                system_name=socket.gethostname(),
                location="Client System",
                ip_address=ip_addr,
                admin_id=admin.admin_id
            )
            db.session.add(system)
            db.session.commit()

        print("\n📊 Collected Metrics:")
        for k, v in metrics.items():
            if k != "timestamp":
                print(f"  • {k:<16}: {v}")

        if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
            prediction_value = np.random.choice([0, 1], p=[0.8, 0.2])
            probability_value = np.random.uniform(60, 95)
        else:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            features = [metrics[k] for k in ["CPU_Usage", "Memory_Usage", "Disk_IO", "Network_Latency", "Error_Rate"]]
            while len(features) < 20:
                features.append(0.0)
            scaled = scaler.transform([features])
            prediction_value = model.predict(scaled)[0]
            probability_value = model.predict_proba(scaled)[0][1] * 100

        db.session.add(SystemMetrics(system_id=system.system_id, **metrics))
        db.session.add(PredictionLog(
            system_id=system.system_id,
            downtime_risk=int(prediction_value),
            probability=float(probability_value),
            estimated_time_to_downtime=15 if prediction_value else None,
            created_at=datetime.now(IST)
        ))
        db.session.commit()

        if prediction_value == 1:
            print(f"⚠ High Downtime Risk Detected ({probability_value:.2f}%)")
        print(f"🧠 Prediction: {'⚠ HIGH RISK' if prediction_value else '✅ NORMAL'} | {probability_value:.2f}%\n")


# ======================================================
# 🔔 WATCH SYSTEM ALERTS
# ======================================================
def watch_notifications_for_system():
    print("👀 Watching for system-specific notifications...")

    conn = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
    cur = conn.cursor(dictionary=True)

    ip = socket.gethostbyname(socket.gethostname())
    cur.execute("SELECT system_id FROM system_info WHERE ip_address=%s", (ip,))
    system = cur.fetchone()

    while not system:
        print("⚠ Waiting for system to register...")
        time.sleep(5)
        cur.execute("SELECT system_id FROM system_info WHERE ip_address=%s", (ip,))
        system = cur.fetchone()

    system_id = system["system_id"]
    print(f"📡 Listening for alerts for system_id={system_id}")

    cur.execute("SELECT MAX(notification_id) AS last_id FROM notification;")
    res = cur.fetchone()
    last_seen_id = res["last_id"] if res and res["last_id"] else 0

    while True:
        cur.execute(
            "SELECT * FROM notification WHERE notification_id > %s AND system_id = %s ORDER BY notification_id ASC",
            (last_seen_id, system_id)
        )
        new_notifs = cur.fetchall()
        for n in new_notifs:
            last_seen_id = n["notification_id"]
            if n["status"].lower() == "unread":
                print(f"\n🚨 Alert: {n['message']}")
                winsound.Beep(1500, 700)
                notification.notify(
                    title=f"⚠ {n['risk_level']} Alert (System {system_id})",
                    message=n["message"],
                    timeout=8
                )
                mark = conn.cursor()
                mark.execute("UPDATE notification SET status='Read' WHERE notification_id=%s", (n["notification_id"],))
                conn.commit()
                mark.close()
        time.sleep(3)


# ======================================================
# 🚀 MAIN LOOP
# ======================================================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        admin = ensure_admin()

    print("✅ Tables Verified. Starting Agent...")
    threading.Thread(target=watch_notifications_for_system, daemon=True).start()

    while True:
        metrics = collect_metrics()
        make_prediction(admin, metrics)
        time.sleep(60)
