from flask import (
    Flask,
    render_template,
    Response,
    send_file,
    request,
    jsonify
)
 
from ultralytics import YOLO
 
import cv2
import datetime
import os
import sqlite3
import time
import threading
import requests
import queue
 
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    HRFlowable
)
 
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
 
# =========================================================
# FLASK
# =========================================================
app = Flask(__name__)
 
# =========================================================
# SYSTEM START TIME
# =========================================================
START_TIME = datetime.datetime.now()
 
# =========================================================
# DATABASE
# =========================================================
def init_db():
    conn = sqlite3.connect("violations.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT,
            time        TEXT,
            detail      TEXT,
            image       TEXT
        )
    """)
    conn.commit()
    conn.close()
 
init_db()
 
# =========================================================
# LOAD MODELS  (hanya sekali di startup)
# =========================================================
# Model PPE  → mendeteksi no-helmet dan no-vest
model_ppe   = YOLO("model/best-ppe.pt")
 
# Model Boots → mendeteksi no-boots
model_boots = YOLO("model/best-boots5.pt")
 
print("PPE CLASSES   :", model_ppe.names)
print("BOOTS CLASSES :", model_boots.names)
 
# =========================================================
# FOLDER SETUP
# =========================================================
os.makedirs("snapshots", exist_ok=True)
os.makedirs("logs",      exist_ok=True)
 
# =========================================================
# TELEGRAM CONFIG
# =========================================================
BOT_TOKEN = "8794935914:AAF_Xy8jHZnDCbxAOZyGeo5jz-AozOW3gtY"
CHAT_ID   = "835216707"
 
# =========================================================
# SNAPSHOT / NOTIFIKASI COOLDOWN
# Sesuai Bab III §3.5.1: cooldown 5 detik per jenis pelanggaran
# =========================================================
SNAPSHOT_COOLDOWN = 5          # detik antar snapshot per jenis pelanggaran
snapshot_cache    = {}         # {violation_key: last_timestamp}
snapshot_lock     = threading.Lock()
 
# =========================================================
# TELEGRAM ALERT  (async – tidak memblokir pipeline deteksi)
# =========================================================
_telegram_queue = queue.Queue()
 
def _telegram_worker():
    """Thread background khusus mengirim notifikasi Telegram."""
    while True:
        item = _telegram_queue.get()
        if item is None:
            break
        image_path, status = item
        try:
            url    = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            now    = datetime.datetime.now()
            caption = (
                "🚨 *PPE-CAM ALERT*\n\n"
                f"📅 Tanggal : {now.strftime('%d/%m/%Y')}\n"
                f"⏰ Waktu   : {now.strftime('%H:%M:%S')}\n\n"
                f"⚠ Pelanggaran:\n{status}\n\n"
                "📍 Smart Construction Monitoring"
            )
            with open(image_path, "rb") as photo:
                requests.post(
                    url,
                    data={"chat_id": CHAT_ID, "caption": caption, "parse_mode": "Markdown"},
                    files={"photo": photo},
                    timeout=10
                )
            print("✅ Telegram alert sent:", status)
        except Exception as e:
            print("❌ Telegram Error:", e)
        finally:
            _telegram_queue.task_done()
 
_tg_thread = threading.Thread(target=_telegram_worker, daemon=True)
_tg_thread.start()
 
 
def send_telegram_alert(image_path, status):
    """Antri pengiriman Telegram; tidak memblokir thread utama."""
    _telegram_queue.put((image_path, status))
 
 
# =========================================================
# SAVE SNAPSHOT  (thread-safe)
# =========================================================
def save_snapshot(frame, violation_key, status_label):
    """
    Simpan snapshot + catat ke DB + kirim Telegram
    hanya jika cooldown sudah terlewati.
 
    violation_key : string unik per jenis pelanggaran,
                    dipakai sebagai kunci cache.
    status_label  : teks pelanggaran yang ditampilkan di Telegram.
    """
    now = time.time()
    with snapshot_lock:
        last = snapshot_cache.get(violation_key, 0)
        if now - last <= SNAPSHOT_COOLDOWN:
            return                       # masih dalam cooldown – lewati
        snapshot_cache[violation_key] = now
 
    ts       = datetime.datetime.now()
    filename = ts.strftime(f"snapshots/{violation_key}_%Y%m%d_%H%M%S.jpg")
 
    # Simpan gambar
    cv2.imwrite(filename, frame)
    print(f"📸 Snapshot saved: {filename}")
 
    # Kirim Telegram (async)
    send_telegram_alert(filename, status_label)
 
    # Simpan ke SQLite
    try:
        conn   = sqlite3.connect("violations.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO violations (date, time, detail, image) VALUES (?, ?, ?, ?)",
            (ts.strftime("%d/%m/%Y"), ts.strftime("%H:%M:%S"), status_label, filename)
        )
        conn.commit()
    except Exception as e:
        print("❌ DB Error:", e)
    finally:
        conn.close()
 
 
# =========================================================
# WEBCAM – konfigurasi optimal untuk FPS
# =========================================================
# Gunakan indeks 0 (kamera internal) atau 1 (kamera eksternal WEB-01).
# Ubah ke 1 jika webcam eksternal tidak terdeteksi di indeks 0.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # CAP_DSHOW mempercepat open di Windows
 
# Resolusi 1280×720 sesuai spesifikasi webcam WEB-01 HD720
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
 
# Buffer kecil → mengurangi lag (ambil frame terbaru, bukan antrian lama)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
 
# Paksa 30 FPS dari kamera
cap.set(cv2.CAP_PROP_FPS, 30)
 
# Codec MJPG lebih cepat dibanding YUY2 untuk webcam USB
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
 
 
# =========================================================
# FRAME BUFFER – decoupled capture vs. inference
# Kamera terus membaca ke buffer; inference membaca buffer terbaru.
# Ini mencegah antrian frame lama memperlambat tampilan.
# =========================================================
_frame_buffer   = None
_frame_lock     = threading.Lock()
_capture_active = True
 
def _capture_loop():
    """Thread terpisah: terus-menerus baca frame dari webcam."""
    global _frame_buffer, _capture_active
    while _capture_active:
        ret, frame = cap.read()
        if ret:
            with _frame_lock:
                _frame_buffer = frame
 
_capture_thread = threading.Thread(target=_capture_loop, daemon=True)
_capture_thread.start()
 
 
# =========================================================
# HELPER: pre-compute ID kelas dari nama model
# =========================================================
def _find_class_id(names_dict, *keywords):
    """Kembalikan ID kelas pertama yang namanya mengandung salah satu keyword."""
    for k, v in names_dict.items():
        norm = v.strip().lower()
        if any(kw in norm for kw in keywords):
            return k
    return None
 
_ID_NO_HELMET = _find_class_id(model_ppe.names,   "no-helmet", "no helmet")
_ID_NO_VEST   = _find_class_id(model_ppe.names,   "no-vest",   "no vest")
_ID_NO_BOOT   = _find_class_id(model_boots.names, "no boot",   "no-boot", "no_boot")
 
print(f"Class IDs → no-helmet:{_ID_NO_HELMET} | no-vest:{_ID_NO_VEST} | no-boot:{_ID_NO_BOOT}")
 
 
# =========================================================
# VIDEO STREAM GENERATOR
# =========================================================
def generate_frames():
    """
    Generator MJPEG untuk endpoint /video.
 
    Optimasi FPS:
    1. Frame diambil dari _frame_buffer (capture thread terpisah).
    2. Model dijalankan dengan half=True (FP16) bila GPU tersedia,
       dan verbose=False untuk menekan overhead logging.
    3. Resize ke 416 px (bukan 640) untuk inferensi lebih cepat;
       hasilnya tetap divisualisasikan pada frame asli 720p.
    4. Warna dikodekan JPEG dengan kualitas 80 untuk memperkecil payload.
    """
    INFER_SIZE = 416   # ukuran input model – turunkan ke 320 bila FPS masih kurang
 
    while True:
        # Ambil frame terbaru dari buffer
        with _frame_lock:
            if _frame_buffer is None:
                time.sleep(0.01)
                continue
            frame = _frame_buffer.copy()
 
        ppe_cls_ids   = set()
        boots_cls_ids = set()
 
        # ── PPE INFERENCE (no-helmet & no-vest) ──────────────────────
        results_ppe = model_ppe.predict(
            frame,
            conf=0.25,           # threshold confidence (Bab III §3.3.3)
            iou=0.45,            # threshold NMS (Bab III §3.3.3)
            imgsz=INFER_SIZE,
            verbose=False,
            half=False           # ganti True jika GPU CUDA tersedia
        )[0]
 
        for box in results_ppe.boxes:
            cls   = int(box.cls[0])
            label = model_ppe.names[cls]
            ppe_cls_ids.add(cls)
 
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
 
            # Warna per kelas (Bab IV §4.1.2)
            if cls == _ID_NO_HELMET:
                color = (0, 0, 255)    # merah  → no-helmet
            elif cls == _ID_NO_VEST:
                color = (0, 165, 255)  # oranye → no-vest
            else:
                color = (0, 255, 0)    # hijau  → helmet/vest detected
 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label.upper()} {conf:.2f}",
                (x1, max(y1 - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )
 
        # ── BOOTS INFERENCE (no-boots) ────────────────────────────────
        results_boots = model_boots.predict(
            frame,
            conf=0.25,
            iou=0.45,
            imgsz=INFER_SIZE,
            verbose=False,
            half=False
        )[0]
 
        for box in results_boots.boxes:
            cls       = int(box.cls[0])
            label_raw = model_boots.names[cls]
            label_norm = label_raw.strip().lower().replace("_", " ").replace("-", " ")

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if "no boot" in label_norm:
                boots_cls_ids.add(cls)
                color = (255, 0, 255)   # ungu → no-boots
                text = f"NO-BOOTS {conf:.2f}"
            else:
                color = (0, 255, 0)     # hijau → boots (lengkap)
                text = f"BOOTS {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )
 
        # ── LOGIKA PELANGGARAN (Bab III §3.4.3) ──────────────────────
        violations = []
 
        if _ID_NO_HELMET is not None and _ID_NO_HELMET in ppe_cls_ids:
            violations.append("NO_HELMET")
 
        if _ID_NO_VEST is not None and _ID_NO_VEST in ppe_cls_ids:
            violations.append("NO_VEST")
 
        if _ID_NO_BOOT is not None and _ID_NO_BOOT in boots_cls_ids:
            violations.append("NO_BOOTS")
 
        # ── STATUS & SNAPSHOT ─────────────────────────────────────────
        if violations:
            status_text  = " | ".join(violations)
            status_color = (0, 0, 255)
 
            violation_key = "_".join(sorted(violations))
            save_snapshot(frame.copy(), violation_key, status_text)
        else:
            status_text  = "APD LENGKAP"
            status_color = (0, 200, 0)
 
        # ── OVERLAY TEKS ──────────────────────────────────────────────
        # Status APD
        cv2.putText(
            frame, status_text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3
        )
 
        # Timestamp real-time
        ts_str = datetime.datetime.now().strftime("%A, %d-%m-%Y  %H:%M:%S")
        cv2.putText(
            frame, ts_str,
            (20, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
 
        # ── ENCODE & YIELD ────────────────────────────────────────────
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
 
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )
 
 
# =========================================================
# ROUTES
# =========================================================
 
@app.route("/")
def dashboard():
    return render_template("index.html")
 
 
@app.route("/monitoring")
def monitoring():
    return render_template("monitoring.html")
 
 
@app.route("/violations")
def violations_page():
    files = sorted(os.listdir("snapshots"), reverse=True)
    return render_template("violations.html", files=files)
 
 
@app.route("/reports")
def reports_page():
    filter_type = request.args.get("filter")
    start_date  = request.args.get("start")
    end_date    = request.args.get("end")
    page        = int(request.args.get("page", 1))
    limit       = 50
    offset      = (page - 1) * limit
 
    conn   = sqlite3.connect("violations.db")
    cursor = conn.cursor()
 
    query  = "SELECT * FROM violations"
    params = []
 
    today = datetime.datetime.now().strftime("%d/%m/%Y")
 
    if filter_type == "today":
        query += " WHERE date = ?"
        params.append(today)
 
    elif filter_type == "week":
        now      = datetime.datetime.now()
        week_ago = now - datetime.timedelta(days=7)
        query += """
            WHERE
            substr(date,7,4)||'-'||substr(date,4,2)||'-'||substr(date,1,2)
            BETWEEN ? AND ?
        """
        params += [week_ago.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")]
 
    elif start_date and end_date:
        query += """
            WHERE
            substr(date,7,4)||'-'||substr(date,4,2)||'-'||substr(date,1,2)
            BETWEEN ? AND ?
        """
        params += [start_date, end_date]
 
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params += [limit, offset]
 
    cursor.execute(query, params)
    data = cursor.fetchall()
    conn.close()
 
    return render_template("reports.html", data=data, page=page)
 
 
@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
 
 
@app.route("/snapshots/<filename>")
def snapshot_file(filename):
    return send_file(os.path.join("snapshots", filename))
 
 
# =========================================================
# STATS API  (Bab IV §4.1.2 – dashboard statistik)
# =========================================================
@app.route("/stats")
def stats():
    conn   = sqlite3.connect("violations.db")
    cursor = conn.cursor()
    today  = datetime.datetime.now().strftime("%d/%m/%Y")
 
    cursor.execute("SELECT COUNT(*) FROM violations WHERE date = ?", (today,))
    total_today = cursor.fetchone()[0]
 
    cursor.execute("SELECT COUNT(*) FROM violations WHERE detail LIKE '%HELMET%'")
    total_helmet = cursor.fetchone()[0]
 
    cursor.execute("SELECT COUNT(*) FROM violations WHERE detail LIKE '%VEST%'")
    total_vest = cursor.fetchone()[0]
 
    cursor.execute("SELECT COUNT(*) FROM violations WHERE detail LIKE '%BOOTS%'")
    total_boots = cursor.fetchone()[0]
 
    conn.close()
 
    total_snapshots  = len(os.listdir("snapshots"))
    uptime_sec       = int((datetime.datetime.now() - START_TIME).total_seconds())
    h, rem           = divmod(uptime_sec, 3600)
    m, s             = divmod(rem, 60)
 
    return jsonify({
        "total_today":     total_today,
        "total_helmet":    total_helmet,
        "total_vest":      total_vest,
        "total_boots":     total_boots,
        "total_snapshots": total_snapshots,
        "uptime":          f"{h:02}:{m:02}:{s:02}",
        "clock":           datetime.datetime.now().strftime("%A, %d-%m-%Y %H:%M:%S")
    })
 
 
# =========================================================
# PDF REPORT  (Bab III §3.6 – ReportLab)
# =========================================================
@app.route("/report")
def report():
    """
    Hasilkan laporan PDF berisi:
    1. Judul laporan
    2. Tabel informasi tiap pelanggaran (tanggal, waktu, jenis)
    3. Dokumentasi visual (snapshot)
    4. Pemisah antar kejadian
 
    Sesuai struktur laporan Bab III §3.6.1 dan hasil implementasi
    yang diuraikan di Bab IV §4.1.4.
    """
    # Ambil parameter filter dari query string
    filter_type = request.args.get("filter")
    start_date  = request.args.get("start")
    end_date    = request.args.get("end")
    
    pdf_path = "logs/report.pdf"
 
    doc    = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm,   bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
 
    # Style khusus
    title_style  = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=16, spaceAfter=6
    )
    normal_style = styles["Normal"]
    bold_style   = ParagraphStyle(
        "Bold", parent=normal_style,
        fontName="Helvetica-Bold"
    )
 
    elements = []
 
    # ── JUDUL ──────────────────────────────────────────────────────
    now      = datetime.datetime.now()
    elements.append(Paragraph("LAPORAN PELANGGARAN APD", title_style))
    elements.append(Paragraph(
        f"Dicetak: {now.strftime('%d/%m/%Y %H:%M:%S')}",
        normal_style
    ))
    elements.append(Spacer(1, 0.5*cm))
 
    # ── RINGKASAN EKSEKUTIF ────────────────────────────────────────
    conn   = sqlite3.connect("violations.db")
    cursor = conn.cursor()
 
    today = datetime.datetime.now().strftime("%d/%m/%Y")
    
    # Prepare WHERE clause dan parameters untuk filter
    where_clause = ""
    where_params = []
    
    if filter_type == "today":
        where_clause = " WHERE date = ?"
        where_params = [today]
    elif filter_type == "week":
        date_obj = datetime.datetime.now()
        week_ago = date_obj - datetime.timedelta(days=7)
        where_clause = " WHERE substr(date,7,4)||'-'||substr(date,4,2)||'-'||substr(date,1,2) BETWEEN ? AND ?"
        where_params = [week_ago.strftime("%Y-%m-%d"), date_obj.strftime("%Y-%m-%d")]
    elif start_date and end_date:
        where_clause = " WHERE substr(date,7,4)||'-'||substr(date,4,2)||'-'||substr(date,1,2) BETWEEN ? AND ?"
        where_params = [start_date, end_date]

    # Count total violations dengan filter
    cursor.execute(f"SELECT COUNT(*) FROM violations{where_clause}", where_params)
    total_all = cursor.fetchone()[0]

    # Count per jenis pelanggaran dengan filter
    cursor.execute(f"SELECT COUNT(*) FROM violations{where_clause} AND detail LIKE '%HELMET%'", where_params)
    total_h = cursor.fetchone()[0]

    cursor.execute(f"SELECT COUNT(*) FROM violations{where_clause} AND detail LIKE '%VEST%'", where_params)
    total_v = cursor.fetchone()[0]

    cursor.execute(f"SELECT COUNT(*) FROM violations{where_clause} AND detail LIKE '%BOOTS%'", where_params)
    total_b = cursor.fetchone()[0]

    # Ambil data pelanggaran dengan filter tanggal (limit 50)
    data_query = f"SELECT date, time, detail, image FROM violations{where_clause} ORDER BY id DESC LIMIT ?"
    cursor.execute(data_query, where_params + [50])
    report_rows = cursor.fetchall()
    conn.close()
 
    summary_data = [
        ["Keterangan",           "Jumlah"],
        ["Total Pelanggaran",    str(total_all)],
        ["Pelanggaran No-Helmet",str(total_h)],
        ["Pelanggaran No-Vest",  str(total_v)],
        ["Pelanggaran No-Boots", str(total_b)],
    ]
    summary_table = Table(summary_data, colWidths=[10*cm, 5*cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#2E75B6")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("BACKGROUND",   (0, 1), (0, -1),  colors.lightgrey),
        ("GRID",         (0, 0), (-1, -1), 1, colors.black),
        ("FONTSIZE",     (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("ALIGN",        (1, 0), (1, -1),  "CENTER"),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.8*cm))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2E75B6")))
    elements.append(Spacer(1, 0.5*cm))
 
    # ── DETAIL TIAP PELANGGARAN ────────────────────────────────────
    if not report_rows:
        elements.append(Paragraph("Tidak ada data pelanggaran.", normal_style))
    else:
        for idx, (date_val, time_val, detail, img_path) in enumerate(report_rows, start=1):
            violation_detail = []
            if "HELMET" in detail: violation_detail.append("No Helmet")
            if "VEST"   in detail: violation_detail.append("No Vest")
            if "BOOTS"  in detail: violation_detail.append("No Boots")
            detail_text = ", ".join(violation_detail) if violation_detail else detail or "-"
 
            fmt_date = date_val or "-"
            fmt_time = time_val or "-"
 
            elements.append(Paragraph(f"<b>Pelanggaran #{idx}</b>", bold_style))
            elements.append(Spacer(1, 0.2*cm))
 
            info_data = [
                ["Tanggal",            fmt_date],
                ["Waktu",              fmt_time],
                ["Detail Pelanggaran", detail_text],
            ]
            info_table = Table(info_data, colWidths=[5*cm, 11*cm])
            info_table.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (0, -1), colors.lightgrey),
                ("TEXTCOLOR",    (0, 0), (-1,-1), colors.black),
                ("GRID",         (0, 0), (-1,-1), 1, colors.black),
                ("FONTNAME",     (0, 0), (-1,-1), "Helvetica"),
                ("FONTSIZE",     (0, 0), (-1,-1), 11),
                ("BOTTOMPADDING",(0, 0), (-1,-1), 6),
                ("TOPPADDING",   (0, 0), (-1,-1), 6),
            ]))
            elements.append(info_table)
            elements.append(Spacer(1, 0.3*cm))
 
            # Gambar snapshot
            if os.path.exists(img_path):
                elements.append(Paragraph("<b>Dokumentasi Visual:</b>", bold_style))
                elements.append(Spacer(1, 0.2*cm))
                img = Image(img_path, width=14*cm, height=8.5*cm)
                elements.append(img)
 
            elements.append(Spacer(1, 0.6*cm))
            elements.append(HRFlowable(width="100%"))
            elements.append(Spacer(1, 0.4*cm))
 
    doc.build(elements)
    return send_file(pdf_path, as_attachment=True)
 
 
# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        # Hentikan thread capture saat aplikasi ditutup
        _capture_active = False
        _telegram_queue.put(None)