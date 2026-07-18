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
 
# Model Person Standar untuk Object Tracking
model_person = YOLO("model/yolo11n.pt")
 
print("PPE CLASSES   :", model_ppe.names)
print("BOOTS CLASSES :", model_boots.names)
print("PERSON CLASSES:", model_person.names)
 
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
# STATE TRACKING FOR PERSON ID ALERTS
# =========================================================
# Format: {person_id: set(violation_types_notified)}
# violation_type: 'NO_HELMET', 'NO_VEST', 'NO_BOOTS'
tracked_alerts = {}
person_last_seen = {}
tracking_lock = threading.Lock()
TRACKING_TIMEOUT = 3.0  # detik toleransi sebelum ID dianggap keluar area
 
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
                "📍 CAM 1 - LOKASI PROYEK A"
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
    Simpan snapshot + catat ke DB + kirim Telegram.
    Cooldown/Throttling dikontrol secara eksternal oleh pemanggil.
    """
    ts       = datetime.datetime.now()
    # Menambahkan microsecond (%f) agar nama file unik jika terjadi beberapa deteksi dalam detik yang sama
    filename = ts.strftime(f"snapshots/{violation_key}_%Y%m%d_%H%M%S_%f.jpg")
 
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
 
print(f"Class IDs -> no-helmet:{_ID_NO_HELMET} | no-vest:{_ID_NO_VEST} | no-boot:{_ID_NO_BOOT}")


# =========================================================
# HELPER: Map violation box to tracked person ID (Spatial Association)
# =========================================================
def get_best_matching_person(violation_bbox, tracked_persons):
    """
    Mencari ID orang yang paling cocok dengan kotak pelanggaran.
    violation_bbox: (vx1, vy1, vx2, vy2)
    tracked_persons: {track_id: (px1, py1, px2, py2)}
    """
    vx1, vy1, vx2, vy2 = violation_bbox
    v_area = (vx2 - vx1) * (vy2 - vy1)
    if v_area <= 0:
        return None
        
    best_id = None
    best_iov = 0.0
    
    for tid, pbox in tracked_persons.items():
        px1, py1, px2, py2 = pbox
        # Hitung area irisan (intersection)
        ix1 = max(vx1, px1)
        iy1 = max(vy1, py1)
        ix2 = min(vx2, px2)
        iy2 = min(vy2, py2)
        
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection_area = iw * ih
        
        iov = intersection_area / v_area
        # Asosiasikan jika setidaknya 40% area pelanggaran masuk di dalam kotak orang
        if iov > best_iov and iov >= 0.4:
            best_iov = iov
            best_id = tid
            
    return best_id


# =========================================================
# HELPER: Check if boot detection box is near person feet (Spatial Filtering)
# =========================================================
def is_box_near_person_feet(boot_bbox, tracked_persons):
    """
    Memeriksa apakah kotak sepatu berada di area kaki dari salah satu orang yang terdeteksi.
    boot_bbox: (bx1, by1, bx2, by2)
    tracked_persons: {track_id: (px1, py1, px2, py2)}
    """
    bx1, by1, bx2, by2 = boot_bbox
    b_center_x = (bx1 + bx2) / 2
    b_center_y = (by1 + by2) / 2
    
    # Jika tidak ada orang yang terdeteksi, kita anggap semua deteksi sepatu di luar area manusia adalah false positive
    if not tracked_persons:
        return False
        
    for tid, pbox in tracked_persons.items():
        px1, py1, px2, py2 = pbox
        p_height = py2 - py1
        
        # Area kaki didefinisikan sebagai bagian bawah (50% ke bawah) dari tubuh orang
        feet_top = py1 + 0.5 * p_height
        # Toleransi 15% di bawah batas kotak orang
        feet_bottom = py2 + 0.15 * p_height
        
        # Toleransi lebar sedikit melebar ke kiri/kanan (15% lebar orang)
        p_width = px2 - px1
        feet_left = px1 - 0.15 * p_width
        feet_right = px2 + 0.15 * p_width
        
        # Cek apakah koordinat tengah kotak sepatu berada dalam batas area kaki orang ini
        if feet_left <= b_center_x <= feet_right and feet_top <= b_center_y <= feet_bottom:
            return True
            
    return False


# =========================================================
# VIDEO STREAM GENERATOR
# =========================================================
def generate_frames():
    """
    Generator MJPEG untuk endpoint /video.
    Optimasi FPS & Integrasi Tracking:
    1. Lacak 'person' menggunakan model_person (yolo11n.pt) dengan ByteTrack.
    2. Deteksi pelanggaran kustom menggunakan model_ppe dan model_boots.
    3. Asosiasikan pelanggaran ke Person ID dan kirim notifikasi unik sekali saja.
    """
    INFER_SIZE = 640   # Menggunakan 640 untuk akurasi pelacakan person yang optimal

    while True:
        # Ambil frame terbaru dari buffer
        with _frame_lock:
            if _frame_buffer is None:
                time.sleep(0.01)
                continue
            frame = _frame_buffer.copy()

        now_time = time.time()

        # ── 1. PERSON TRACKING (YOLO11 Standard) ──────────────────────
        results_person = model_person.track(
            frame,
            classes=[0],  # filter hanya person
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,    # Diturunkan ke 0.25 agar deteksi orang lebih sensitif
            iou=0.45,
            verbose=False,
            imgsz=INFER_SIZE
        )[0]

        # Ekstrak data orang yang terlacak: {track_id: [x1, y1, x2, y2]}
        tracked_persons = {}
        if results_person.boxes is not None and results_person.boxes.id is not None:
            track_ids = results_person.boxes.id.int().tolist()
            bboxes = results_person.boxes.xyxy.int().tolist()
            for tid, bbox in zip(track_ids, bboxes):
                tracked_persons[tid] = bbox
                with tracking_lock:
                    person_last_seen[tid] = now_time

        # Bersihkan ID orang yang sudah keluar area (timeout)
        with tracking_lock:
            expired_ids = [tid for tid, last_seen in person_last_seen.items() if now_time - last_seen > TRACKING_TIMEOUT]
            for tid in expired_ids:
                person_last_seen.pop(tid, None)
                tracked_alerts.pop(tid, None)
                print(f"🔄 Tracking ID #{tid} di-reset karena tidak terdeteksi selama {TRACKING_TIMEOUT} detik.")

        # ── 2. VIOLATIONS DETECTION (Custom Models) ───────────────────
        results_ppe = model_ppe.predict(
            frame,
            conf=0.25,
            iou=0.45,
            imgsz=INFER_SIZE,
            verbose=False
        )[0]

        results_boots = model_boots.predict(
            frame,
            conf=0.25,
            iou=0.45,
            imgsz=INFER_SIZE,
            verbose=False
        )[0]

        # Kumpulkan seluruh pelanggaran frame ini: (tipe, conf, [vx1, vy1, vx2, vy2])
        frame_violations = []

        # Deteksi PPE (no-helmet & no-vest)
        for box in results_ppe.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = [x1, y1, x2, y2]
            if cls == _ID_NO_HELMET:
                frame_violations.append(("NO_HELMET", conf, bbox))
            elif cls == _ID_NO_VEST:
                frame_violations.append(("NO_VEST", conf, bbox))

        # Deteksi sepatu (no-boots & boots patuh)
        for box in results_boots.boxes:
            cls       = int(box.cls[0])
            label_raw = model_boots.names[cls]
            label_norm = label_raw.strip().lower().replace("_", " ").replace("-", " ")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = [x1, y1, x2, y2]
            conf = float(box.conf[0])
            
            # Filter: Lewati jika deteksi sepatu tidak berada di dekat kaki orang mana pun
            if not is_box_near_person_feet(bbox, tracked_persons):
                continue
                
            if "no boot" in label_norm:
                frame_violations.append(("NO_BOOTS", conf, bbox))
            else:
                # Gambarkan sepatu patuh (BOOTS hijau) langsung di frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"BOOTS {conf:.2f}",
                    (x1, max(y1 - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                )

        # ── 3. ASOSIASI SPASIAL & PEMICU ALARM ────────────────────────
        # Format: (tipe, conf, bbox, person_id)
        drawn_violations = []
        alert_texts = []
        violations_to_save = set()

        for vtype, vconf, vbbox in frame_violations:
            person_id = get_best_matching_person(vbbox, tracked_persons)
            drawn_violations.append((vtype, vconf, vbbox, person_id))

            if person_id is not None:
                # Cek apakah ID orang ini sudah dikirimi notifikasi untuk jenis pelanggaran ini
                should_alert = False
                with tracking_lock:
                    if person_id not in tracked_alerts:
                        tracked_alerts[person_id] = set()
                    if vtype not in tracked_alerts[person_id]:
                        tracked_alerts[person_id].add(vtype)
                        should_alert = True
                
                if should_alert:
                    alert_texts.append(f"Orang #{person_id}: {vtype}")
                    violations_to_save.add(vtype)
            else:
                # Cadangan: Jika deteksi orang gagal, gunakan cooldown global (cooldown per pelanggaran)
                violation_key = f"global_{vtype}"
                should_alert = False
                with snapshot_lock:
                    last = snapshot_cache.get(violation_key, 0)
                    if now_time - last > SNAPSHOT_COOLDOWN:
                        snapshot_cache[violation_key] = now_time
                        should_alert = True
                if should_alert:
                    alert_texts.append(f"Pelanggaran: {vtype}")
                    violations_to_save.add(vtype)

        # ── 4. GAMBAR BOUNDING BOXES ──────────────────────────────────
        # Gambar kotak ungu untuk Person yang terdeteksi & dilacak
        for tid, pbox in tracked_persons.items():
            px1, py1, px2, py2 = pbox
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 255), 2)
            cv2.putText(
                frame,
                f"PERSON #{tid}",
                (px1, max(py1 - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2
            )

        # Gambar kotak pelanggaran
        for vtype, vconf, vbbox, person_id in drawn_violations:
            vx1, vy1, vx2, vy2 = vbbox

            # Warna per kelas
            if vtype == "NO_HELMET":
                color = (0, 0, 255)    # merah  → no-helmet
                label_text = f"NO-HELMET {vconf:.2f}"
            elif vtype == "NO_VEST":
                color = (0, 165, 255)  # oranye → no-vest
                label_text = f"NO-VEST {vconf:.2f}"
            elif vtype == "NO_BOOTS":
                color = (255, 0, 255)  # ungu → no-boots
                label_text = f"NO-BOOTS {vconf:.2f}"
            else:
                continue

            # Tambahkan info ID orang jika ada asosiasinya
            if person_id is not None:
                label_text += f" (ID:#{person_id})"

            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), color, 2)
            cv2.putText(
                frame,
                label_text,
                (vx1, max(vy1 - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

        # ── 5. STATUS TEKS OVERLAY ────────────────────────────────────
        active_violations = sorted(list(set(vtype for vtype, _, _, _ in drawn_violations)))
        if active_violations:
            status_text  = " | ".join(active_violations)
            status_color = (0, 0, 255)
        else:
            status_text  = "APD LENGKAP"
            status_color = (0, 200, 0)

        # Status APD di kiri atas
        cv2.putText(
            frame, status_text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3
        )

        # Timestamp real-time di kanan bawah
        ts_str = datetime.datetime.now().strftime("%A, %d-%m-%Y  %H:%M:%S")
        cv2.putText(
            frame, ts_str,
            (20, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
 
        # ── 6. SNAPSHOT & TELEGRAM TRIGGER (Setelah Gambar Terlukis) ──
        if alert_texts:
            status_text_alert = " | ".join(alert_texts)
            violation_key = "_".join(sorted(list(violations_to_save)))
            # Mengambil salinan frame yang sudah digambari bounding box & teks overlay
            save_snapshot(frame.copy(), violation_key, status_text_alert)

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
    snapshots_dir = "snapshots"
    if not os.path.exists(snapshots_dir):
        files = []
    else:
        # Saring file biasa dan urutkan berdasarkan waktu modifikasi terbaru di atas
        files = [f for f in os.listdir(snapshots_dir) if os.path.isfile(os.path.join(snapshots_dir, f))]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(snapshots_dir, x)), reverse=True)
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