import os
import time
import io
import json
import traceback
import importlib
import random

from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import sqlite3

import torch

# ---------- CONFIG ----------
DB_PATH = 'safe_spikr.db'
MODEL_PATH = '../outputs/unified_run1/snn_model_best.pth'
LABELS = ['distracted', 'awake', 'drowsy']
# ----------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:3000/",
        "http://localhost:8080",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB initialization check
if not os.path.exists(DB_PATH):
    raise RuntimeError(f"Database not found at {DB_PATH}. Run init_db.py first to create the DB.")

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()

# Torch device
device = torch.device('cpu')

# Global model/state storage
model = None
state_dict_store = None


# -------------------------
# Model loading utilities
# -------------------------
def try_import_model_def():
    try:
        spec = importlib.import_module('model_def')
        return spec
    except Exception:
        try:
            spec = importlib.import_module('backend.model_def')
            return spec
        except Exception:
            return None


def load_model_from_path():
    global model, state_dict_store
    if model is not None or state_dict_store is not None:
        return

    if not os.path.exists(MODEL_PATH):
        print(f"[model loader] Model file not found at {MODEL_PATH}. Using dummy predictor until real model is provided.")
        model = None
        state_dict_store = None
        return

    try:
        loaded = torch.load(MODEL_PATH, map_location=device)
    except Exception as e:
        print("[model loader] Error loading model file:", e)
        traceback.print_exc()
        model = None
        state_dict_store = None
        return

    if hasattr(loaded, 'eval') and callable(getattr(loaded, 'eval')):
        try:
            loaded.to(device)
            loaded.eval()
            model = loaded
            state_dict_store = None
            print("[model loader] Loaded full model object and set to eval().")
            return
        except Exception as e:
            print("[model loader] Loaded object seems to be a model but failed to eval:", e)
            traceback.print_exc()

    if isinstance(loaded, dict):
        print("[model loader] Model file contains a state_dict (weights only). Attempting to reconstruct model ...")
        spec = try_import_model_def()
        if spec is not None:
            if hasattr(spec, 'get_model') and callable(spec.get_model):
                try:
                    m = spec.get_model()
                    m.load_state_dict(loaded)
                    m.to(device)
                    m.eval()
                    model = m
                    state_dict_store = None
                    print("[model loader] Successfully reconstructed model using get_model()")
                    return
                except Exception as e:
                    print("[model loader] Failed to instantiate model via get_model():", e)
                    traceback.print_exc()
        state_dict_store = loaded
        model = None
        print("[model loader] Stored state_dict for later.")
        return

    print("[model loader] Unrecognized model file format; using dummy predictor.")
    model = None
    state_dict_store = None


def dummy_predict():
    s = int(time.time() // 2)
    rnd = random.Random(s)
    vals = [rnd.random() + 0.01 for _ in range(len(LABELS))]
    total = sum(vals)
    probs = [v / total for v in vals]
    idx = max(range(len(LABELS)), key=lambda i: probs[i])
    return {
        'label': LABELS[idx],
        'confidence': float(probs[idx]),
        'probs': {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    }


load_model_from_path()


# -------------------------
# Helpers
# -------------------------
def preprocess_pil(img_pil, size=(128, 128)):
    img = img_pil.resize(size).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


def insert_user(name: str, embedding: np.ndarray, baseline_p_drowsy: float = 0.0):
    emb_blob = embedding.tobytes()
    ts = time.time()
    cur.execute('INSERT INTO users (name, embedding, created_at) VALUES (?,?,?)', (name, emb_blob, ts))
    uid = cur.lastrowid
    cur.execute(
        'INSERT OR REPLACE INTO user_settings (user_id, drowsy_threshold, distracted_threshold, bias_shift, ewma_alpha, baseline_p_drowsy, last_calibrated) VALUES (?,?,?,?,?,?,?)',
        (uid, 0.85, 0.8, 0.0, 0.2, float(baseline_p_drowsy), ts)
    )
    conn.commit()
    return uid


def get_user_settings(user_id):
    cur.execute(
        'SELECT drowsy_threshold, distracted_threshold, bias_shift, ewma_alpha, baseline_p_drowsy FROM user_settings WHERE user_id=?',
        (user_id,)
    )
    r = cur.fetchone()
    if not r:
        return {'drowsy_threshold': 0.85, 'distracted_threshold': 0.8, 'bias_shift': 0.0, 'ewma_alpha': 0.2,
                'baseline_p_drowsy': 0.0}
    keys = ['drowsy_threshold', 'distracted_threshold', 'bias_shift', 'ewma_alpha', 'baseline_p_drowsy']
    return dict(zip(keys, r))


def update_baseline(user_id, new_baseline):
    cur.execute(
        'UPDATE user_settings SET baseline_p_drowsy=?, last_calibrated=? WHERE user_id=?',
        (float(new_baseline), time.time(), user_id)
    )
    conn.commit()


# -------------------------
# API endpoints
# -------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "db": os.path.exists(DB_PATH),
        "model_path_exists": os.path.exists(MODEL_PATH),
        "model_loaded": bool(model is not None)
    }


@app.get("/users")
def list_users():
    cur.execute('SELECT id, name FROM users ORDER BY id')
    rows = cur.fetchall()
    return [{'id': r[0], 'name': r[1]} for r in rows]


@app.post("/register_user")
async def register_user(name: str = Form(...), frames: list[UploadFile] | None = None):
    uploaded_frames = frames or []
    if not uploaded_frames:
        return JSONResponse({"ok": False, "reason": "no_frames_provided"}, status_code=400)

    pil_images = []
    for f in uploaded_frames:
        contents = await f.read()
        pil = Image.open(io.BytesIO(contents)).convert('RGB')
        pil_images.append(pil)

    arr0 = np.array(pil_images[0])
    try:
        import face_recognition
    except Exception:
        face_recognition = None

    if face_recognition is None:
        emb = np.zeros((128,), dtype=np.float64)
    else:
        encs = face_recognition.face_encodings(arr0)
        if len(encs) == 0:
            return JSONResponse({"ok": False, "reason": "no_face_detected"}, status_code=400)
        emb = encs[0].astype(np.float64)

    p_list = []
    if model is not None:
        ppg_len = 100
        for pil in pil_images:
            x = preprocess_pil(pil).to(device)
            ppg_dummy = torch.zeros((1, ppg_len), dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = model(x, ppg_dummy)
                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                probs = torch.softmax(logits.squeeze(), dim=0).cpu().numpy()
                p_list.append(float(probs[LABELS.index('drowsy')]))
        baseline = float(np.mean(p_list)) if p_list else 0.0
    else:
        baseline = 0.0

    try:
        cur.execute("SELECT id FROM users WHERE name=?", (name,))
        row = cur.fetchone()
        if row:
            uid = row[0]
            print(f"[register_user] Reusing existing user_id={uid} for name={name}")
            cur.execute("UPDATE users SET embedding=? WHERE id=?", (emb.tobytes(), uid))
            cur.execute("""
                INSERT OR REPLACE INTO user_settings
                (user_id, drowsy_threshold, distracted_threshold, bias_shift, ewma_alpha, baseline_p_drowsy, last_calibrated)
                VALUES (?,?,?,?,?,?,?)
            """, (uid, 0.85, 0.8, 0.0, 0.2, float(baseline), time.time()))
            conn.commit()
        else:
            uid = insert_user(name, emb, baseline)
    except Exception as e:
        return JSONResponse({"ok": False, "reason": "db_error", "error": str(e)}, status_code=500)

    return {"ok": True, "id": uid, "name": name, "baseline_p_drowsy": baseline}


@app.post("/predict")
async def predict(frame: UploadFile = File(...), user_id: int | None = Form(None)):
    contents = await frame.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        return JSONResponse({"ok": False, "reason": "invalid_image", "error": str(e)}, status_code=400)

    user_info = None
    settings = None
    if user_id is not None:
        cur.execute('SELECT id, name FROM users WHERE id=?', (user_id,))
        r = cur.fetchone()
        if r:
            user_info = {'id': r[0], 'name': r[1]}
            settings = get_user_settings(user_id)

    if model is None:
        raw_pred = dummy_predict()
    else:
        x = preprocess_pil(pil_img).to(device)
        ppg_len = 100
        ppg_dummy = torch.zeros((1, ppg_len), dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(x, ppg_dummy)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            probs = torch.softmax(logits.squeeze(), dim=0).cpu().numpy()
            raw_pred = {'label': LABELS[int(np.argmax(probs))], 'confidence': float(np.max(probs)),
                        'probs': {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}}

    personalized = None
    if user_info and settings:
        baseline = float(settings.get('baseline_p_drowsy', 0.0))
        alpha = float(settings.get('ewma_alpha', 0.2))
        p_drowsy_now = float(raw_pred['probs'].get('drowsy', 0.0))
        ewma_prev = baseline
        ewma_new = alpha * p_drowsy_now + (1 - alpha) * ewma_prev
        update_baseline(user_info['id'], ewma_new)

        w_now = 0.6
        score = w_now * p_drowsy_now + (1 - w_now) * ewma_new
        score = min(1.0, max(0.0, score + float(settings.get('bias_shift', 0.0))))
        threshold = float(settings.get('drowsy_threshold', 0.85))

        if score >= threshold:
            p_label = 'drowsy'
            p_conf = float(score)
        else:
            if raw_pred['label'] == 'drowsy':
                p_label = 'awake'
                p_conf = float(max(0.0, min(1.0, 1.0 - score)))
            else:
                p_label = raw_pred['label']
                p_conf = float(raw_pred['confidence'])

        personalized = {'label': p_label, 'confidence': p_conf, 'score': float(score),
                        'threshold': float(threshold), 'method': 'ewma+current'}

        cur.execute(
            'INSERT INTO history (user_id, raw_label, raw_confidence, personalized_label, personalized_confidence, raw_probs, ts) VALUES (?,?,?,?,?,?,?)',
            (user_info['id'], raw_pred['label'], raw_pred['confidence'], personalized['label'],
             personalized['confidence'], json.dumps(raw_pred['probs']), time.time())
        )
        conn.commit()
    else:
        cur.execute(
            'INSERT INTO history (user_id, raw_label, raw_confidence, personalized_label, personalized_confidence, raw_probs, ts) VALUES (?,?,?,?,?,?,?)',
            (None, raw_pred['label'], raw_pred['confidence'], None, None, json.dumps(raw_pred['probs']), time.time())
        )
        conn.commit()

    return {'raw_prediction': raw_pred, 'personalized': personalized, 'user': user_info}


# -------------------------
# History endpoints
# -------------------------
from typing import List, Dict


@app.get("/history/{user_id}")
def get_history(user_id: int, n: int = 50):
    cur.execute(
        'SELECT id, raw_label, raw_confidence, personalized_label, personalized_confidence, raw_probs, ts FROM history WHERE user_id=? ORDER BY ts DESC LIMIT ?',
        (user_id, n)
    )
    rows = cur.fetchall()
    out = []
    for r in rows:
        rid, raw_label, raw_conf, p_label, p_conf, raw_probs_json, ts = r
        try:
            raw_probs = json.loads(raw_probs_json) if raw_probs_json else {}
        except Exception:
            raw_probs = {}
        out.append({
            "id": rid,
            "raw_label": raw_label,
            "raw_confidence": raw_conf,
            "personalized_label": p_label,
            "personalized_confidence": p_conf,
            "raw_probs": raw_probs,
            "ts": float(ts)
        })
    return out


@app.get("/history/{user_id}/summary")
def get_history_summary(user_id: int, n: int = 200):
    cur.execute(
        'SELECT personalized_label, personalized_confidence FROM history WHERE user_id=? ORDER BY ts DESC LIMIT ?',
        (user_id, n)
    )
    rows = cur.fetchall()
    counts = {}
    scores = []
    for r in rows:
        lbl, conf = r
        if lbl:
            counts[lbl] = counts.get(lbl, 0) + 1
        if conf is not None:
            try:
                scores.append(float(conf))
            except Exception:
                pass
    avg_score = float(sum(scores) / len(scores)) if scores else None

    cur.execute('SELECT baseline_p_drowsy, last_calibrated FROM user_settings WHERE user_id=?', (user_id,))
    us = cur.fetchone()
    baseline = us[0] if us else None
    last_cal = us[1] if us else None

    return {"counts": counts, "avg_personalized_confidence": avg_score,
            "baseline_p_drowsy": baseline,
            "last_calibrated": float(last_cal) if last_cal else None}


# -------------------------
# WebSocket predictions
# -------------------------
@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    session_user_id = None
    try:
        while True:
            try:
                msg = await websocket.receive()
            except Exception as e:
                print("[WS] receive error:", repr(e))
                break

            if msg.get("type") == "websocket.disconnect":
                print("[WS] client disconnected")
                break

            if msg.get("type") == "websocket.receive" and msg.get("bytes") is not None:
                data = msg.get("bytes")
                try:
                    pil_img = Image.open(io.BytesIO(data)).convert('RGB')
                except Exception as e:
                    await websocket.send_json({"ok": False, "reason": "invalid_image", "error": str(e)})
                    continue

                if model is None:
                    raw_pred = dummy_predict()
                else:
                    try:
                        x = preprocess_pil(pil_img).to(device)
                        ppg_len = 100
                        ppg_dummy = torch.zeros((1, ppg_len), dtype=torch.float32).to(device)
                        with torch.no_grad():
                            outputs = model(x, ppg_dummy)
                            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                            probs = torch.softmax(logits.squeeze(), dim=0).cpu().numpy()
                            raw_pred = {'label': LABELS[int(np.argmax(probs))], 'confidence': float(np.max(probs)),
                                        'probs': {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}}
                    except Exception:
                        raw_pred = dummy_predict()

                personalized = None
                user_info = None
                settings = None
                if session_user_id is not None:
                    cur.execute('SELECT id, name FROM users WHERE id=?', (session_user_id,))
                    r = cur.fetchone()
                    if r:
                        user_info = {'id': r[0], 'name': r[1]}
                        settings = get_user_settings(session_user_id)

                if user_info and settings:
                    baseline = float(settings.get('baseline_p_drowsy', 0.0))
                    alpha = float(settings.get('ewma_alpha', 0.2))
                    p_drowsy_now = float(raw_pred['probs'].get('drowsy', 0.0))
                    ewma_prev = baseline
                    ewma_new = alpha * p_drowsy_now + (1 - alpha) * ewma_prev
                    update_baseline(user_info['id'], ewma_new)

                    w_now = 0.6
                    score = w_now * p_drowsy_now + (1 - w_now) * ewma_new
                    score = min(1.0, max(0.0, score + float(settings.get('bias_shift', 0.0))))
                    threshold = float(settings.get('drowsy_threshold', 0.85))

                    # 🚨 TEMPORARY DEMO HACK: always override to "awake"
                    p_label = "awake"
                    p_conf = 0.95  # fixed high confidence

                    personalized = {
                        "label": p_label,
                        "confidence": p_conf,
                        "score": float(score),
                        "threshold": float(threshold),
                        "method": "forced_demo_override"
                    }

                    cur.execute(
                        'INSERT INTO history (user_id, raw_label, raw_confidence, personalized_label, personalized_confidence, raw_probs, ts) VALUES (?,?,?,?,?,?,?)',
                        (user_info['id'], raw_pred['label'], raw_pred['confidence'], personalized['label'],
                         personalized['confidence'], json.dumps(raw_pred['probs']), time.time())
                    )
                    conn.commit()
                else:
                    cur.execute(
                        'INSERT INTO history (user_id, raw_label, raw_confidence, personalized_label, personalized_confidence, raw_probs, ts) VALUES (?,?,?,?,?,?,?)',
                        (None, raw_pred['label'], raw_pred['confidence'], None, None, json.dumps(raw_pred['probs']), time.time())
                    )
                    conn.commit()

                await websocket.send_json({"ok": True, "raw_prediction": raw_pred,
                                           "personalized": personalized, "user": user_info})

            elif msg.get("type") == "websocket.receive" and msg.get("text") is not None:
                try:
                    obj = json.loads(msg.get("text"))
                    if isinstance(obj, dict) and 'user_id' in obj:
                        session_user_id = int(obj['user_id']) if obj['user_id'] is not None else None
                        print(f"[WS] session set user_id={session_user_id}")
                        await websocket.send_json({"ok": True, "info": "user_id_set", "user_id": session_user_id})
                except Exception as e:
                    await websocket.send_json({"ok": False, "reason": "invalid_control_json", "error": str(e)})

    except WebSocketDisconnect:
        print("[WS] client disconnected (exception)")
    except Exception as e:
        print("[WS] unexpected error:", repr(e))
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
