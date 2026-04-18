from __future__ import annotations

import io
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import qrcode
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import PyMongoError


class GenerateQRCodeRequest(BaseModel):
    data: str = Field(min_length=1, description="Text or URL to encode")


class AssureTagRequest(BaseModel):
    data: str = Field(min_length=1, description="Text to encode in outer QR")


class AssureTraceRequest(BaseModel):
    data: str = Field(min_length=1, description="Product or asset identifier")


class InMemoryRequestStore:
    def __init__(self) -> None:
        self.generate_requests: List[Dict[str, Any]] = []
        self.scan_requests: List[Dict[str, Any]] = []

    def insert_generate_request(self, document: Dict[str, Any]) -> None:
        self.generate_requests.append(document)

    def insert_scan_request(self, document: Dict[str, Any]) -> None:
        self.scan_requests.append(document)

    def find_by_inner_serial(self, serial: str, qr_type: str) -> Optional[Dict[str, Any]]:
        for doc in self.generate_requests:
            if doc.get("inner_serial") == serial and doc.get("type") == qr_type:
                return doc
        return None


class MongoRequestStore:
    def __init__(self, uri: str, db_name: str) -> None:
        self.client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        self.client.admin.command("ping")
        self.db = self.client[db_name]
        self.generate_collection = self.db["generate_qr_code_requests"]
        self.scan_collection = self.db["scan_qr_code_requests"]

    def insert_generate_request(self, document: Dict[str, Any]) -> None:
        self.generate_collection.insert_one(document)

    def insert_scan_request(self, document: Dict[str, Any]) -> None:
        self.scan_collection.insert_one(document)

    def find_by_inner_serial(self, serial: str, qr_type: str) -> Optional[Dict[str, Any]]:
        return self.generate_collection.find_one(
            {"inner_serial": serial, "type": qr_type}, {"_id": 0}
        )


_INDIA_CITIES = [
    {"city": "Mumbai", "state": "Maharashtra", "lat": 19.0760, "lng": 72.8777},
    {"city": "Delhi", "state": "Delhi", "lat": 28.6139, "lng": 77.2090},
    {"city": "Bangalore", "state": "Karnataka", "lat": 12.9716, "lng": 77.5946},
    {"city": "Chennai", "state": "Tamil Nadu", "lat": 13.0827, "lng": 80.2707},
    {"city": "Hyderabad", "state": "Telangana", "lat": 17.3850, "lng": 78.4867},
    {"city": "Kolkata", "state": "West Bengal", "lat": 22.5726, "lng": 88.3639},
    {"city": "Pune", "state": "Maharashtra", "lat": 18.5204, "lng": 73.8567},
]


def _generate_nested_qr(outer_text: str, inner_text: str) -> bytes:
    from PIL import Image as PILImage

    qr_outer = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=10, border=4
    )
    qr_outer.add_data(outer_text)
    qr_outer.make(fit=True)

    qr_inner = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=4, border=2
    )
    qr_inner.add_data(inner_text)
    qr_inner.make(fit=True)

    outer_buf = io.BytesIO()
    qr_outer.make_image(fill_color="black", back_color="white").save(outer_buf, format="PNG")
    outer_buf.seek(0)
    outer_img = PILImage.open(outer_buf).convert("RGB")

    inner_buf = io.BytesIO()
    qr_inner.make_image(fill_color="black", back_color="white").save(inner_buf, format="PNG")
    inner_buf.seek(0)
    inner_img = PILImage.open(inner_buf).convert("RGB")

    outer_w, outer_h = outer_img.size
    inner_size = int(outer_w * 0.22)
    inner_resized = inner_img.resize((inner_size, inner_size), PILImage.NEAREST)

    pad = 4
    padded_size = inner_size + pad * 2
    padded = PILImage.new("RGB", (padded_size, padded_size), "white")
    padded.paste(inner_resized, (pad, pad))

    x = (outer_w - padded_size) // 2
    y = (outer_h - padded_size) // 2
    outer_img.paste(padded, (x, y))

    result_buf = io.BytesIO()
    outer_img.save(result_buf, format="PNG")
    return result_buf.getvalue()


def _decode_small_qr(crop: np.ndarray) -> Optional[str]:
    """Try multiple preprocessing strategies to decode a small/dense QR."""
    detector = cv2.QRCodeDetector()
    cw, ch = crop.shape[1], crop.shape[0]
    target = 600
    scale = max(2, target // max(cw, ch, 1))
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    variants: list = [big]

    # adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    variants.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    _, otsu = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))

    # sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(big, -1, kernel)
    variants.append(sharp)

    # bilateral denoise then threshold
    denoised = cv2.bilateralFilter(big, 9, 75, 75)
    d_gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    _, d_thresh = cv2.threshold(d_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(d_thresh, cv2.COLOR_GRAY2BGR))

    for v in variants:
        text, _, _ = detector.detectAndDecode(v)
        if text:
            return text
    return None


def _scan_dual_qr(file_bytes: bytes) -> tuple:
    """Return (outer_text, inner_text) from an image; either may be None."""
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image file")

    # Upscale whole image 2x for better detection of both codes
    h0, w0 = image.shape[:2]
    image = cv2.resize(image, (w0 * 2, h0 * 2), interpolation=cv2.INTER_CUBIC)
    h, w = image.shape[:2]
    image_area = float(h * w)

    detector = cv2.QRCodeDetector()

    # Try multi-QR detection first
    retval, decoded_info, points, _ = detector.detectAndDecodeMulti(image)
    found = []  # list of (text, bbox_area)
    if retval and decoded_info:
        for text, pts in zip(decoded_info, points):
            if text and pts is not None:
                xs, ys = pts[:, 0], pts[:, 1]
                area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
                found.append((text, area))

    if len(found) >= 2:
        # Sort by bounding-box size: largest = outer, smallest = inner
        found.sort(key=lambda x: x[1], reverse=True)
        return found[0][0], found[1][0]

    outer_text: Optional[str] = None
    inner_text: Optional[str] = None

    if len(found) == 1:
        text, bbox_area = found[0]
        ratio = bbox_area / image_area
        # Inner QR is ~22% of the outer's side → ~5% of image area.
        # Outer QR typically fills >15% of the image.
        if ratio > 0.10:
            outer_text = text   # large bounding box → this is the outer QR
        else:
            inner_text = text   # small bounding box → this is the inner QR

    # If outer not yet found, try full-frame single decode
    if outer_text is None:
        ft, ft_pts, _ = detector.detectAndDecode(image)
        if ft and ft != inner_text:
            if ft_pts is not None:
                xs, ys = ft_pts[0][:, 0], ft_pts[0][:, 1]
                area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
                if area / image_area > 0.10:
                    outer_text = ft
                elif inner_text is None:
                    inner_text = ft   # still small → treat as inner
            else:
                outer_text = ft

    # Inner QR: centre crop at multiple sizes with aggressive preprocessing
    if inner_text is None:
        cx, cy = w // 2, h // 2
        for frac in (0.28, 0.35, 0.22):
            ch = int(min(w, h) * frac)
            x1 = max(0, cx - ch); y1 = max(0, cy - ch)
            x2 = min(w, cx + ch); y2 = min(h, cy + ch)
            crop = image[y1:y2, x1:x2]
            result = _decode_small_qr(crop)
            if result and result != outer_text:
                inner_text = result
                break

    return outer_text, inner_text


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _create_request_store() -> Any:
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    mongo_db = os.getenv("MONGODB_DB", "quick_response_code")
    try:
        return MongoRequestStore(uri=mongo_uri, db_name=mongo_db)
    except PyMongoError:
        return InMemoryRequestStore()


def _save_locally(image_bytes: bytes, filename: str, storage_dir: Path) -> str:
    storage_dir.mkdir(parents=True, exist_ok=True)
    output_path = storage_dir / filename
    output_path.write_bytes(image_bytes)
    return str(output_path)


def _maybe_upload_to_s3(image_bytes: bytes, filename: str) -> Optional[str]:
    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        return None

    import boto3

    key_prefix = os.getenv("AWS_S3_PREFIX", "qrcodes")
    key = f"{key_prefix.rstrip('/')}/{filename}"
    s3_client = boto3.client("s3")
    s3_client.put_object(Bucket=bucket, Key=key, Body=image_bytes, ContentType="image/png")
    return f"s3://{bucket}/{key}"


def create_app(request_store: Optional[Any] = None, storage_dir: Optional[Path] = None) -> FastAPI:
    app = FastAPI(title="Quick Response Code API")

    resolved_store = request_store or _create_request_store()
    resolved_storage_dir = storage_dir or Path(
        os.getenv("QR_STORAGE_DIR", str(Path(__file__).resolve().parent / "storage" / "qrcodes"))
    )

    app.state.request_store = resolved_store
    app.state.storage_dir = resolved_storage_dir

    _trace_store: Dict[str, Any] = {}

    _GENERATOR_JS = """
    // ── QR Scanner ──────────────────────────────────────────────
    let _scanStream = null;

    function openScanner() {
      document.getElementById('scanner-modal').style.display = 'flex';
      document.getElementById('scan-result-box').style.display = 'none';
      document.getElementById('scan-error').style.display = 'none';
      document.getElementById('scan-capture-btn').disabled = false;
      startCamera();
    }

    function closeScanner() {
      stopCamera();
      document.getElementById('scanner-modal').style.display = 'none';
    }

    async function startCamera() {
      try {
        _scanStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
        });
        const video = document.getElementById('scanner-video');
        video.srcObject = _scanStream;
        await video.play();
      } catch (err) {
        showScanError('Camera access denied or unavailable: ' + err.message);
      }
    }

    function stopCamera() {
      if (_scanStream) {
        _scanStream.getTracks().forEach(t => t.stop());
        _scanStream = null;
      }
      const video = document.getElementById('scanner-video');
      video.srcObject = null;
    }

    async function captureAndScan() {
      const video = document.getElementById('scanner-video');
      const btn = document.getElementById('scan-capture-btn');
      document.getElementById('scan-error').style.display = 'none';
      document.getElementById('scan-result-box').style.display = 'none';

      if (!video.videoWidth) {
        showScanError('Camera not ready yet. Please wait a moment.');
        return;
      }

      btn.disabled = true;
      btn.textContent = 'Scanning\u2026';

      try {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);

        const blob = await new Promise((res, rej) =>
          canvas.toBlob(b => b ? res(b) : rej(new Error('Failed to capture frame')), 'image/png')
        );

        const formData = new FormData();
        formData.append('file', blob, 'capture.png');

        const response = await fetch('/scan_qr_code', { method: 'POST', body: formData });
        const body = await response.json();

        if (!response.ok) throw new Error(body.detail || 'Scan failed.');

        stopCamera();
        const resultBox = document.getElementById('scan-result-box');
        const resultText = document.getElementById('scan-result-text');
        resultText.textContent = body.result;
        resultBox.style.display = 'block';
        document.getElementById('scan-capture-btn').style.display = 'none';
      } catch (err) {
        showScanError(err.message);
        btn.disabled = false;
        btn.textContent = 'Capture & Scan';
      }
    }

    function showScanError(msg) {
      const el = document.getElementById('scan-error');
      el.textContent = msg;
      el.style.display = 'block';
    }

    // ── QR Generator ─────────────────────────────────────────────
    async function generateQR(inputId, btnId, errorId, resultId, imgId, dlId) {
      const input = document.getElementById(inputId);
      const btn = document.getElementById(btnId);
      const errorEl = document.getElementById(errorId);
      const resultEl = document.getElementById(resultId);
      const img = document.getElementById(imgId);
      const dl = document.getElementById(dlId);

      const data = input.value.trim();
      if (!data) { showError(errorEl, 'Please enter some text or a URL.'); return; }

      errorEl.style.display = 'none';
      resultEl.style.display = 'none';
      btn.disabled = true;
      const original = btn.textContent;
      btn.textContent = 'Generating\u2026';

      try {
        const response = await fetch('/generate_qr_code', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ data }),
        });
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          throw new Error(body.detail || 'Failed to generate QR code.');
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        img.src = url;
        dl.href = url;
        resultEl.style.display = 'flex';
      } catch (err) {
        showError(errorEl, err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = original;
      }
    }

    function showError(el, msg) {
      el.textContent = msg;
      el.style.display = 'block';
    }
    """

    @app.get("/qr_generator_page", response_class=HTMLResponse)
    def qr_generator_page() -> HTMLResponse:
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>QR Code Generator</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: system-ui, sans-serif;
      background: #f3f4f6;
    }}
    .card {{
      background: #fff;
      border-radius: 12px;
      box-shadow: 0 4px 24px rgba(0,0,0,.08);
      padding: 2.5rem;
      width: 100%;
      max-width: 480px;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }}
    h1 {{ font-size: 1.5rem; color: #111; }}
    input[type="text"] {{
      width: 100%;
      padding: .75rem 1rem;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      font-size: 1rem;
      outline: none;
      transition: border-color .2s;
    }}
    input[type="text"]:focus {{ border-color: #1a3c6e; }}
    button {{
      padding: .75rem 1rem;
      background: #1a3c6e;
      color: #fff;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      cursor: pointer;
      transition: background .2s;
    }}
    button:hover {{ background: #122d54; }}
    button:disabled {{ background: #7a9cc4; cursor: not-allowed; }}
    #sp-error {{ color: #dc2626; font-size: .9rem; display: none; }}
    #sp-result {{ display: none; flex-direction: column; align-items: center; gap: 1rem; }}
    #sp-result img {{ width: 220px; height: 220px; border: 1px solid #e5e7eb; border-radius: 8px; }}
    #sp-download {{
      background: #10b981;
      text-decoration: none;
      color: #fff;
      padding: .6rem 1.25rem;
      border-radius: 8px;
      font-size: .95rem;
    }}
    #sp-download:hover {{ background: #059669; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>QR Code Generator</h1>
    <input type="text" id="sp-input" placeholder="Enter text or URL\u2026"
           onkeydown="if(event.key==='Enter') generateQR('sp-input','sp-btn','sp-error','sp-result','sp-img','sp-download')" />
    <button id="sp-btn" onclick="generateQR('sp-input','sp-btn','sp-error','sp-result','sp-img','sp-download')">
      Generate QR Code
    </button>
    <p id="sp-error"></p>
    <div id="sp-result">
      <img id="sp-img" src="" alt="QR Code" />
      <a id="sp-download" download="qrcode.png">Download</a>
    </div>
  </div>
  <script>{_GENERATOR_JS}</script>
</body>
</html>"""
        return HTMLResponse(content=html)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RuDron &ndash; QR Code Generator</title>
  <link rel="icon" href="https://rudron.com/favicon.ico" />
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --brand: #1a3c6e;
      --brand-dark: #122d54;
      --accent: #e8a020;
      --text: #1e293b;
      --muted: #64748b;
      --bg: #f8fafc;
      --white: #ffffff;
    }}
    body {{ font-family: system-ui, -apple-system, sans-serif; color: var(--text); background: var(--bg); }}

    /* ── NAV ── */
    nav {{
      position: sticky; top: 0; z-index: 100;
      display: flex; align-items: center; justify-content: space-between;
      padding: .9rem 2rem;
      background: var(--white);
      box-shadow: 0 1px 8px rgba(0,0,0,.07);
    }}
    .nav-logo {{
      display: flex; align-items: center; gap: .55rem;
      text-decoration: none;
    }}
    .nav-logo img {{ width: 28px; height: 28px; object-fit: contain; }}
    .nav-logo span {{
      font-size: 1.25rem; font-weight: 700; letter-spacing: -.5px;
      color: var(--brand);
    }}
    .nav-logo span em {{ color: var(--accent); font-style: normal; }}
    .nav-links {{ display: flex; align-items: center; gap: 1.75rem; list-style: none; }}
    .nav-links a {{ text-decoration: none; color: var(--muted); font-size: .95rem; transition: color .2s; }}
    .nav-links a:hover {{ color: var(--brand); }}
    .nav-cta {{
      padding: .5rem 1.25rem;
      background: var(--brand); color: var(--white);
      border-radius: 6px; text-decoration: none; font-size: .9rem; font-weight: 600;
      transition: background .2s;
    }}
    .nav-cta:hover {{ background: var(--brand-dark); }}

    /* ── HERO ── */
    .hero {{
      background: linear-gradient(135deg, var(--brand) 0%, #2a5298 60%, #1e3a5f 100%);
      color: var(--white);
      padding: 5rem 2rem 4rem;
    }}
    .hero-inner {{
      max-width: 1100px; margin: 0 auto;
      display: grid; grid-template-columns: 1fr 420px; gap: 4rem; align-items: center;
    }}
    .hero-copy h1 {{
      font-size: clamp(2rem, 3.5vw, 2.9rem);
      font-weight: 800; line-height: 1.2; margin-bottom: 1.1rem;
    }}
    .hero-copy h1 span {{ color: var(--accent); }}
    .hero-copy p {{ font-size: 1.1rem; opacity: .88; line-height: 1.6; margin-bottom: 1.75rem; }}
    .hero-bullets {{ list-style: none; display: flex; flex-direction: column; gap: .6rem; margin-bottom: 2rem; }}
    .hero-bullets li {{ display: flex; align-items: center; gap: .6rem; font-size: .97rem; opacity: .9; }}
    .hero-bullets li::before {{ content: "\\2713"; font-weight: 700; color: var(--accent); font-size: 1rem; }}
    .hero-btn {{
      display: inline-block;
      padding: .85rem 2.2rem;
      background: var(--accent); color: #111;
      border-radius: 8px; text-decoration: none; font-weight: 700; font-size: 1rem;
      transition: filter .2s;
    }}
    .hero-btn:hover {{ filter: brightness(1.1); }}

    /* ── GENERATOR CARD ── */
    .gen-card {{
      background: var(--white);
      border-radius: 16px;
      padding: 2rem;
      box-shadow: 0 8px 40px rgba(0,0,0,.18);
      display: flex; flex-direction: column; gap: 1rem;
    }}
    .gen-card h2 {{ font-size: 1.15rem; color: var(--brand); font-weight: 700; }}
    .gen-card input[type="text"] {{
      width: 100%; padding: .75rem 1rem;
      border: 1.5px solid #d1d5db; border-radius: 8px;
      font-size: 1rem; outline: none; transition: border-color .2s;
      color: var(--text);
    }}
    .gen-card input[type="text"]:focus {{ border-color: var(--brand); }}
    .gen-card button {{
      padding: .8rem 1rem;
      background: var(--brand); color: var(--white);
      border: none; border-radius: 8px; font-size: 1rem; font-weight: 600;
      cursor: pointer; transition: background .2s;
    }}
    .gen-card button:hover {{ background: var(--brand-dark); }}
    .gen-card button:disabled {{ background: #7a9cc4; cursor: not-allowed; }}
    #hero-error {{ color: #dc2626; font-size: .88rem; display: none; }}
    #hero-result {{
      display: none; flex-direction: column; align-items: center; gap: .85rem;
      padding-top: .5rem;
    }}
    #hero-img {{
      width: 190px; height: 190px;
      border: 1px solid #e5e7eb; border-radius: 10px;
    }}
    #hero-download {{
      background: #10b981; color: var(--white);
      text-decoration: none; padding: .55rem 1.4rem;
      border-radius: 8px; font-size: .93rem; font-weight: 600;
    }}
    #hero-download:hover {{ background: #059669; }}

    /* ── STEPS ── */
    .steps {{
      max-width: 1100px; margin: 5rem auto; padding: 0 2rem;
      text-align: center;
    }}
    .steps h2 {{
      font-size: 1.85rem; font-weight: 800; color: var(--brand); margin-bottom: .5rem;
    }}
    .steps .subtitle {{ color: var(--muted); margin-bottom: 3rem; font-size: 1.05rem; }}
    .steps-grid {{
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem;
    }}
    .step {{
      background: var(--white); border-radius: 14px;
      padding: 2rem 1.5rem;
      box-shadow: 0 2px 16px rgba(0,0,0,.06);
      display: flex; flex-direction: column; align-items: center; gap: .85rem;
    }}
    .step-num {{
      width: 52px; height: 52px; border-radius: 50%;
      background: linear-gradient(135deg, var(--brand), #2a5298);
      color: var(--white); font-size: 1.35rem; font-weight: 800;
      display: flex; align-items: center; justify-content: center;
    }}
    .step h3 {{ font-size: 1.05rem; font-weight: 700; color: var(--text); }}
    .step p {{ font-size: .9rem; color: var(--muted); line-height: 1.55; }}

    /* ── FEATURES ── */
    .features {{
      background: linear-gradient(180deg, #f0f4ff 0%, var(--bg) 100%);
      padding: 5rem 2rem;
    }}
    .features-inner {{ max-width: 1100px; margin: 0 auto; }}
    .features h2 {{
      font-size: 1.85rem; font-weight: 800; color: var(--brand);
      text-align: center; margin-bottom: .5rem;
    }}
    .features .subtitle {{
      text-align: center; color: var(--muted); margin-bottom: 3rem; font-size: 1.05rem;
    }}
    .features-grid {{
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem;
    }}
    .feature {{
      background: var(--white); border-radius: 14px;
      padding: 1.75rem 1.5rem;
      box-shadow: 0 2px 12px rgba(0,0,0,.05);
    }}
    .feature-icon {{
      font-size: 1.8rem; margin-bottom: .75rem;
    }}
    .feature h3 {{ font-size: 1rem; font-weight: 700; margin-bottom: .4rem; color: var(--brand); }}
    .feature p {{ font-size: .88rem; color: var(--muted); line-height: 1.55; }}

    /* ── FOOTER ── */
    footer {{
      background: var(--brand-dark); color: rgba(255,255,255,.75);
      text-align: center; padding: 2rem 1rem;
      font-size: .88rem; line-height: 1.7;
    }}
    footer a {{ color: var(--accent); text-decoration: none; }}
    footer a:hover {{ text-decoration: underline; }}

    /* ── RESPONSIVE ── */
    @media (max-width: 800px) {{
      .hero-inner {{ grid-template-columns: 1fr; gap: 2.5rem; }}
      .gen-card {{ width: 100%; }}
      .steps-grid, .features-grid {{ grid-template-columns: 1fr; }}
      .nav-links {{ display: none; }}
    }}
  </style>
</head>
<body>

  <!-- NAV -->
  <nav>
    <a class="nav-logo" href="https://rudron.com" target="_blank" rel="noopener">
      <img src="https://rudron.com/favicon.ico" alt="RuDron logo"
           onerror="this.style.display='none'" />
      <span>Ru<em>Dron</em></span>
    </a>
    <ul class="nav-links">
      <li><a href="https://rudron.com/about/" target="_blank" rel="noopener">About</a></li>
      <li><a href="https://rudron.com/services/" target="_blank" rel="noopener">Services</a></li>
      <li><a href="https://rudron.com/contact/" target="_blank" rel="noopener">Contact</a></li>
    </ul>
    <a class="nav-cta" href="#generator">Create QR Code</a>
  </nav>

  <!-- HERO -->
  <section class="hero">
    <div class="hero-inner">
      <div class="hero-copy">
        <h1>We make <span>QR codes</span><br>easy &amp; secure</h1>
        <p>Powered by RuDron Technolabs &mdash; trusted anti-counterfeiting experts.<br>
           Generate high-quality QR codes instantly, right in your browser.</p>
        <ul class="hero-bullets">
          <li>Works with any text, URL, or data</li>
          <li>Instant PNG download &mdash; no sign-up required</li>
          <li>Private &amp; secure &mdash; nothing stored in the browser</li>
        </ul>
        <div style="display:flex;gap:1rem;flex-wrap:wrap;">
          <a class="hero-btn" href="#generator">Generate for free</a>
          <button class="hero-btn" style="background:var(--accent);border:none;cursor:pointer;"
                  onclick="openScanner()">&#128247;&nbsp;Scan QR Code</button>
          <a class="hero-btn" href="/specialized_codes"
             style="background:#10b981;color:#fff;">&#128203;&nbsp;Specialized QR Codes</a>
        </div>
      </div>

      <!-- INLINE GENERATOR -->
      <div class="gen-card" id="generator">
        <h2>Generate your QR code</h2>
        <input type="text" id="hero-input" placeholder="Enter text or URL\u2026"
               onkeydown="if(event.key==='Enter') generateQR('hero-input','hero-btn','hero-error','hero-result','hero-img','hero-download')" />
        <button id="hero-btn"
                onclick="generateQR('hero-input','hero-btn','hero-error','hero-result','hero-img','hero-download')">
          Generate QR Code
        </button>
        <p id="hero-error"></p>
        <div id="hero-result">
          <img id="hero-img" src="" alt="Generated QR Code" />
          <a id="hero-download" download="qrcode.png">&#8681;&nbsp;Download PNG</a>
        </div>
      </div>
    </div>
  </section>

  <!-- HOW IT WORKS -->
  <section class="steps">
    <h2>Create your QR code in three simple steps</h2>
    <p class="subtitle">Fast, free, and no account needed.</p>
    <div class="steps-grid">
      <div class="step">
        <div class="step-num">1</div>
        <h3>Enter your content</h3>
        <p>Type or paste any text, website URL, phone number, or any data you want to encode.</p>
      </div>
      <div class="step">
        <div class="step-num">2</div>
        <h3>Generate instantly</h3>
        <p>Click &ldquo;Generate QR Code&rdquo; and your QR code is created in milliseconds.</p>
      </div>
      <div class="step">
        <div class="step-num">3</div>
        <h3>Download &amp; share</h3>
        <p>Save the PNG and use it in print, on screens, or share it digitally anywhere.</p>
      </div>
    </div>
  </section>

  <!-- FEATURES -->
  <section class="features">
    <div class="features-inner">
      <h2>Why choose RuDron QR Generator?</h2>
      <p class="subtitle">Built on the same trust that protects governments and global brands.</p>
      <div class="features-grid">
        <div class="feature">
          <div class="feature-icon">&#9889;</div>
          <h3>Instant generation</h3>
          <p>QR codes are generated server-side in real time &mdash; no waiting, no queues.</p>
        </div>
        <div class="feature">
          <div class="feature-icon">&#128274;</div>
          <h3>Secure &amp; private</h3>
          <p>RuDron is a leader in anti-counterfeiting technology. Your data is handled with the highest standards.</p>
        </div>
        <div class="feature">
          <div class="feature-icon">&#128247;</div>
          <h3>High-resolution output</h3>
          <p>Download crisp, scan-ready PNG images suitable for both digital use and high-quality print.</p>
        </div>
      </div>
    </div>
  </section>

  <!-- FOOTER -->
  <footer>
    <p>&copy; 2026 <a href="https://rudron.com" target="_blank" rel="noopener">RuDron Technolabs Pvt. Ltd.</a>
       &mdash; All rights reserved.</p>
    <p style="margin-top:.35rem; font-size:.8rem; opacity:.65;">
      &lsquo;QR Code&rsquo; is a trademark of DENSO WAVE INCORPORATED
    </p>
  </footer>

  <!-- SCANNER MODAL -->
  <div id="scanner-modal" style="display:none;position:fixed;inset:0;z-index:1000;
       background:rgba(0,0,0,.75);align-items:center;justify-content:center;">
    <div style="background:#fff;border-radius:16px;padding:2rem;width:min(520px,95vw);
                display:flex;flex-direction:column;gap:1rem;position:relative;">
      <button onclick="closeScanner()" style="position:absolute;top:1rem;right:1rem;
              background:none;border:none;font-size:1.4rem;cursor:pointer;color:#64748b;"
              title="Close">&times;</button>
      <h2 style="color:#1a3c6e;font-size:1.2rem;font-weight:700;">Scan a QR Code</h2>
      <p style="font-size:.88rem;color:#64748b;">Point your camera at a QR code then click <strong>Capture &amp; Scan</strong>.</p>
      <video id="scanner-video" playsinline muted
             style="width:100%;border-radius:10px;background:#000;max-height:320px;object-fit:cover;"></video>
      <button id="scan-capture-btn" onclick="captureAndScan()"
              style="padding:.8rem 1rem;background:#1a3c6e;color:#fff;border:none;
                     border-radius:8px;font-size:1rem;font-weight:600;
                     cursor:pointer;transition:background .2s;">
        Capture &amp; Scan
      </button>
      <p id="scan-error" style="color:#dc2626;font-size:.88rem;display:none;"></p>
      <div id="scan-result-box" style="display:none;background:#f0fdf4;border:1px solid #86efac;
           border-radius:10px;padding:1.1rem;">
        <p style="font-size:.82rem;font-weight:600;color:#166534;margin-bottom:.4rem;">&#10003;&nbsp;QR Code detected</p>
        <p id="scan-result-text" style="font-size:.97rem;color:#1e293b;word-break:break-all;"></p>
        <div style="margin-top:.85rem;display:flex;gap:.75rem;flex-wrap:wrap;">
          <button onclick="navigator.clipboard.writeText(document.getElementById('scan-result-text').textContent)"
                  style="padding:.45rem 1rem;background:#1a3c6e;color:#fff;border:none;
                         border-radius:6px;font-size:.88rem;cursor:pointer;">Copy</button>
          <button onclick="closeScanner();document.getElementById('scan-capture-btn').style.display='block';"
                  style="padding:.45rem 1rem;background:#e5e7eb;color:#1e293b;border:none;
                         border-radius:6px;font-size:.88rem;cursor:pointer;">Close</button>
        </div>
      </div>
    </div>
  </div>

  <script>{_GENERATOR_JS}</script>
</body>
</html>"""
        return HTMLResponse(content=html)

    @app.post("/generate_qr_code", response_class=Response)
    def generate_qr_code(payload: GenerateQRCodeRequest) -> Response:
        event_id = str(uuid.uuid4())
        created_at = _utc_now()
        filename = f"{event_id}.png"

        try:
            qr_image = qrcode.make(payload.data)
            image_buffer = io.BytesIO()
            qr_image.save(image_buffer, format="PNG")
            image_bytes = image_buffer.getvalue()

            local_path = _save_locally(image_bytes=image_bytes, filename=filename, storage_dir=app.state.storage_dir)
            stored_path = _maybe_upload_to_s3(image_bytes=image_bytes, filename=filename) or local_path

            app.state.request_store.insert_generate_request(
                {
                    "event_id": event_id,
                    "request_data": payload.data,
                    "status": "completed",
                    "result_path": stored_path,
                    "created_at": created_at,
                    "completed_at": _utc_now(),
                }
            )

            return Response(content=image_bytes, media_type="image/png", headers={"X-Event-Id": event_id})
        except Exception as exc:  # noqa: BLE001
            try:
                app.state.request_store.insert_generate_request(
                    {
                        "event_id": event_id,
                        "request_data": payload.data,
                        "status": "failed",
                        "error": str(exc),
                        "created_at": created_at,
                        "completed_at": _utc_now(),
                    }
                )
            except Exception:  # noqa: BLE001
                pass
            raise HTTPException(status_code=500, detail="Failed to generate QR code") from exc

    @app.post("/scan_qr_code")
    async def scan_qr_code(file: UploadFile = File(...)) -> JSONResponse:
        event_id = str(uuid.uuid4())
        created_at = _utc_now()

        try:
            file_bytes = await file.read()
            image_array = np.frombuffer(file_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Invalid image file")

            detector = cv2.QRCodeDetector()
            decoded_text, _, _ = detector.detectAndDecode(image)
            if not decoded_text:
                raise ValueError("No QR code content detected")

            filename = f"{event_id}.png"
            stored_path = _save_locally(
                image_bytes=file_bytes, filename=filename, storage_dir=app.state.storage_dir
            )

            app.state.request_store.insert_scan_request(
                {
                    "event_id": event_id,
                    "file_name": file.filename,
                    "status": "completed",
                    "result": decoded_text,
                    "stored_path": stored_path,
                    "created_at": created_at,
                    "completed_at": _utc_now(),
                }
            )

            return JSONResponse(content={"event_id": event_id, "result": decoded_text})
        except ValueError as exc:
            try:
                app.state.request_store.insert_scan_request(
                    {
                        "event_id": event_id,
                        "file_name": file.filename,
                        "status": "failed",
                        "error": str(exc),
                        "created_at": created_at,
                        "completed_at": _utc_now(),
                    }
                )
            except Exception:  # noqa: BLE001
                pass
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            try:
                app.state.request_store.insert_scan_request(
                    {
                        "event_id": event_id,
                        "file_name": file.filename,
                        "status": "failed",
                        "error": str(exc),
                        "created_at": created_at,
                        "completed_at": _utc_now(),
                    }
                )
            except Exception:  # noqa: BLE001
                pass
            raise HTTPException(status_code=500, detail="Failed to scan QR code") from exc

    # ── AssureTag endpoint ────────────────────────────────────────────────
    @app.post("/assure_tag/generate", response_class=Response)
    def assure_tag_generate(payload: AssureTagRequest) -> Response:
        event_id = str(uuid.uuid4())
        created_at = _utc_now()
        serial = str(random.randint(10**9, 10**10 - 1))
        image_bytes = _generate_nested_qr(outer_text=payload.data, inner_text=serial)
        filename = f"assure_tag_{event_id}.png"
        stored_path = _save_locally(image_bytes=image_bytes, filename=filename, storage_dir=app.state.storage_dir)
        app.state.request_store.insert_generate_request(
            {
                "event_id": event_id,
                "type": "assure_tag",
                "request_data": payload.data,
                "inner_serial": serial,
                "status": "completed",
                "result_path": stored_path,
                "created_at": created_at,
                "completed_at": _utc_now(),
            }
        )
        return Response(
            content=image_bytes,
            media_type="image/png",
            headers={"X-Event-Id": event_id, "X-Serial": serial},
        )

    # ── AssureTrace endpoints ─────────────────────────────────────────────
    @app.post("/assure_trace/generate", response_class=Response)
    def assure_trace_generate(payload: AssureTraceRequest) -> Response:
        event_id = str(uuid.uuid4())
        created_at = _utc_now()
        serial = f"AT-{uuid.uuid4().hex[:8].upper()}"
        combined = f"{payload.data}|SERIAL:{serial}"
        image_bytes = _generate_nested_qr(outer_text=combined, inner_text=serial)
        filename = f"assure_trace_{event_id}.png"
        stored_path = _save_locally(image_bytes=image_bytes, filename=filename, storage_dir=app.state.storage_dir)
        _trace_store[event_id] = {"data": payload.data, "serial": serial}
        app.state.request_store.insert_generate_request(
            {
                "event_id": event_id,
                "type": "assure_trace",
                "request_data": payload.data,
                "inner_serial": serial,
                "status": "completed",
                "result_path": stored_path,
                "created_at": created_at,
                "completed_at": _utc_now(),
            }
        )
        return Response(
            content=image_bytes,
            media_type="image/png",
            headers={"X-Event-Id": event_id, "X-Serial": serial},
        )

    @app.get("/assure_trace/simulate/{event_id}")
    def assure_trace_simulate(event_id: str) -> JSONResponse:
        info = _trace_store.get(event_id)
        if not info:
            raise HTTPException(status_code=404, detail="Trace record not found")
        cities = random.sample(_INDIA_CITIES, 3)
        base_time = datetime.now(timezone.utc) - timedelta(hours=48)
        statuses = ["Manufactured", "In Transit", "Delivered"]
        events = [
            {
                "city": city["city"],
                "state": city["state"],
                "lat": city["lat"],
                "lng": city["lng"],
                "timestamp": (base_time + timedelta(hours=i * 16 + random.randint(0, 3))).isoformat(),
                "serial": info["serial"],
                "status": statuses[i],
            }
            for i, city in enumerate(cities)
        ]
        return JSONResponse(content={"serial": info["serial"], "data": info["data"], "events": events})

    # ── AssureTag / AssureTrace scan endpoints ────────────────────────────
    @app.post("/assure_tag/scan")
    async def assure_tag_scan(file: UploadFile = File(...)) -> JSONResponse:
        event_id = str(uuid.uuid4())
        created_at = _utc_now()
        file_bytes = await file.read()
        try:
            outer_text, inner_text = _scan_dual_qr(file_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        matched = app.state.request_store.find_by_inner_serial(inner_text or "", "assure_tag") if inner_text else None
        authentic = matched is not None

        app.state.request_store.insert_scan_request(
            {
                "event_id": event_id,
                "type": "assure_tag_scan",
                "file_name": file.filename,
                "outer_code": outer_text,
                "inner_code": inner_text,
                "authentic": authentic,
                "created_at": created_at,
                "completed_at": _utc_now(),
            }
        )
        return JSONResponse(content={
            "event_id": event_id,
            "outer_code": outer_text,
            "inner_code": inner_text,
            "authentic": authentic,
        })

    @app.post("/assure_trace/scan")
    async def assure_trace_scan(file: UploadFile = File(...)) -> JSONResponse:
        event_id = str(uuid.uuid4())
        created_at = _utc_now()
        file_bytes = await file.read()
        try:
            outer_text, inner_text = _scan_dual_qr(file_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        matched = app.state.request_store.find_by_inner_serial(inner_text or "", "assure_trace") if inner_text else None
        authentic = matched is not None

        app.state.request_store.insert_scan_request(
            {
                "event_id": event_id,
                "type": "assure_trace_scan",
                "file_name": file.filename,
                "outer_code": outer_text,
                "inner_code": inner_text,
                "authentic": authentic,
                "created_at": created_at,
                "completed_at": _utc_now(),
            }
        )
        return JSONResponse(content={
            "event_id": event_id,
            "outer_code": outer_text,
            "inner_code": inner_text,
            "authentic": authentic,
        })

    # ── Specialized codes landing page ────────────────────────────────────
    @app.get("/specialized_codes", response_class=HTMLResponse)
    def specialized_codes() -> HTMLResponse:  # noqa: PLR0915
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RuDron AssureSeries\u2122 \u2013 Specialized QR Solutions</title>
  <link rel="icon" href="https://rudron.com/favicon.ico" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/jsqr@1.4.0/dist/jsQR.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --brand: #1a3c6e; --brand-dark: #122d54; --accent: #e8a020;
      --green: #10b981; --text: #1e293b; --muted: #64748b;
      --bg: #f8fafc; --white: #ffffff; --border: #e5e7eb;
    }}
    body {{ font-family: system-ui, -apple-system, sans-serif; color: var(--text); background: var(--bg); }}
    nav {{
      position: sticky; top: 0; z-index: 100;
      display: flex; align-items: center; justify-content: space-between;
      padding: .9rem 2rem; background: var(--white);
      box-shadow: 0 1px 8px rgba(0,0,0,.07);
    }}
    .nav-logo {{ display: flex; align-items: center; gap: .55rem; text-decoration: none; }}
    .nav-logo img {{ width: 28px; height: 28px; object-fit: contain; }}
    .nav-logo span {{ font-size: 1.25rem; font-weight: 700; color: var(--brand); }}
    .nav-logo span em {{ color: var(--accent); font-style: normal; }}
    .nav-back {{ text-decoration: none; color: var(--muted); font-size: .9rem; transition: color .2s; }}
    .nav-back:hover {{ color: var(--brand); }}
    .page-hero {{
      background: linear-gradient(135deg, var(--brand) 0%, #2a5298 100%);
      color: #fff; text-align: center; padding: 3.5rem 2rem 3rem;
    }}
    .page-hero h1 {{ font-size: clamp(1.7rem, 3vw, 2.5rem); font-weight: 800; margin-bottom: .65rem; }}
    .page-hero h1 span {{ color: var(--accent); }}
    .page-hero p {{ font-size: 1.05rem; opacity: .88; max-width: 640px; margin: 0 auto; }}
    .products-section {{ max-width: 1280px; margin: 3rem auto 2rem; padding: 0 1.5rem; }}
    .products-grid {{
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.75rem; align-items: start;
    }}
    .product-card {{
      background: var(--white); border-radius: 16px;
      box-shadow: 0 2px 20px rgba(0,0,0,.08);
      display: flex; flex-direction: column;
    }}
    .card-header {{ padding: 1.5rem 1.5rem 0; }}
    .badge {{
      display: inline-flex; align-items: center; gap: .4rem;
      padding: .3rem .85rem; border-radius: 20px; font-size: .75rem;
      font-weight: 700; letter-spacing: .5px; text-transform: uppercase; margin-bottom: .85rem;
    }}
    .badge-tag {{ background: #dbeafe; color: #1d4ed8; }}
    .badge-trace {{ background: #fef9c3; color: #a16207; }}
    .badge-stock {{ background: #d1fae5; color: #065f46; }}
    .product-card h2 {{ font-size: 1.3rem; font-weight: 800; color: var(--brand); margin-bottom: .35rem; }}
    .product-card .tagline {{ font-size: .88rem; color: var(--muted); line-height: 1.55; margin-bottom: 1.1rem; }}
    .divider {{ height: 1px; background: var(--border); margin: 0 1.5rem; }}
    .card-demo {{ padding: 1.25rem 1.5rem 1.5rem; display: flex; flex-direction: column; gap: .85rem; flex: 1; }}
    .demo-label {{ font-size: .75rem; font-weight: 700; color: var(--brand); text-transform: uppercase; letter-spacing: .6px; }}
    .demo-input {{
      width: 100%; padding: .65rem .9rem; border: 1.5px solid var(--border);
      border-radius: 8px; font-size: .95rem; outline: none; transition: border-color .2s;
    }}
    .demo-input:focus {{ border-color: var(--brand); }}
    .btn {{
      display: inline-flex; align-items: center; justify-content: center; gap: .4rem;
      padding: .65rem 1.1rem; border: none; border-radius: 8px;
      font-size: .9rem; font-weight: 600; cursor: pointer; transition: filter .2s;
      text-decoration: none; white-space: nowrap;
    }}
    .btn:disabled {{ opacity: .55; cursor: not-allowed; }}
    .btn-primary {{ background: var(--brand); color: #fff; }}
    .btn-primary:hover:not(:disabled) {{ filter: brightness(1.12); }}
    .btn-accent {{ background: var(--accent); color: #111; }}
    .btn-accent:hover:not(:disabled) {{ filter: brightness(1.1); }}
    .btn-green {{ background: var(--green); color: #fff; }}
    .btn-green:hover:not(:disabled) {{ filter: brightness(1.1); }}
    .btn-ghost {{ background: #f1f5f9; color: var(--text); }}
    .btn-ghost:hover:not(:disabled) {{ background: #e2e8f0; }}
    .btn-row {{ display: flex; gap: .6rem; flex-wrap: wrap; }}
    .qr-result-box {{ display: none; flex-direction: column; align-items: center; gap: .75rem; padding-top: .25rem; }}
    .qr-result-box img {{ width: 180px; height: 180px; border: 1px solid var(--border); border-radius: 10px; }}
    .serial-badge {{
      background: #f0f4ff; border: 1px solid #c7d2fe; border-radius: 8px;
      padding: .45rem .9rem; font-size: .82rem; color: var(--brand);
      font-family: monospace; word-break: break-all; text-align: center; width: 100%;
    }}
    .inline-error {{ color: #dc2626; font-size: .85rem; display: none; }}
    #trace-map-section {{ max-width: 1280px; margin: 0 auto 3rem; padding: 0 1.5rem; display: none; }}
    .trace-map-card {{ background: var(--white); border-radius: 16px; box-shadow: 0 2px 20px rgba(0,0,0,.08); padding: 1.5rem; }}
    .trace-map-card h3 {{ font-size: 1.1rem; font-weight: 700; color: var(--brand); margin-bottom: 1rem; }}
    #trace-map {{ height: 360px; border-radius: 12px; z-index: 1; }}
    .trace-grid {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 1.5rem; align-items: start; }}
    .timeline-label {{ font-size: .75rem; font-weight: 700; color: var(--brand); text-transform: uppercase; letter-spacing: .5px; margin-bottom: .75rem; }}
    #stock-scanner-area {{ display: none; flex-direction: column; gap: .75rem; }}
    #stock-video {{ width: 100%; border-radius: 8px; background: #000; max-height: 220px; object-fit: cover; }}
    #stock-last-scan-box {{
      display: none; background: #f0fdf4; border: 1px solid #86efac;
      border-radius: 8px; padding: .6rem .9rem; font-size: .85rem; color: #166534;
    }}
    .stock-table {{ width: 100%; border-collapse: collapse; font-size: .87rem; }}
    .stock-table th {{
      text-align: left; padding: .45rem .9rem; background: #f8fafc;
      font-weight: 700; color: var(--muted); font-size: .75rem; text-transform: uppercase;
      border-bottom: 2px solid var(--border);
    }}
    .stock-table td {{ padding: .5rem .9rem; border-bottom: 1px solid var(--border); }}
    #stock-results {{ display: none; }}
    footer {{
      background: var(--brand-dark); color: rgba(255,255,255,.75);
      text-align: center; padding: 2rem 1rem; font-size: .88rem; line-height: 1.7;
    }}
    footer a {{ color: var(--accent); text-decoration: none; }}
    @media (max-width: 960px) {{
      .products-grid {{ grid-template-columns: 1fr; max-width: 520px; margin: 0 auto; }}
      .trace-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>

<nav>
  <a class="nav-logo" href="/">
    <img src="https://rudron.com/favicon.ico" alt="RuDron" onerror="this.style.display='none'" />
    <span>Ru<em>Dron</em></span>
  </a>
  <a class="nav-back" href="/">&#8592; Back to Home</a>
</nav>

<section class="page-hero">
  <h1>RuDron <span>AssureSeries\u2122</span></h1>
  <p>Specialized QR solutions for product authentication, supply-chain traceability,
     and real-time inventory management \u2014 trusted by governments and global brands.</p>
</section>

<section class="products-section">
  <div class="products-grid">

    <!-- ── ASSURETAG ── -->
    <div class="product-card">
      <div class="card-header">
        <span class="badge badge-tag">&#128215; AssureTag\u2122</span>
        <h2>AssureTag\u2122</h2>
        <p class="tagline">Product serialization QR code technology that guarantees your products cannot be
        counterfeited. Each unit carries a unique nested dual-layer QR \u2014 an outer QR encodes your data
        while an inner QR holds a tamper-proof random serial.</p>
      </div>
      <div class="divider"></div>
      <div class="card-demo">
        <p class="demo-label">Create Demo Code</p>
        <input class="demo-input" id="at-input" placeholder="Enter product name or URL\u2026"
               onkeydown="if(event.key==='Enter') atGenerate()" />
        <button class="btn btn-primary" id="at-btn" onclick="atGenerate()" style="width:100%;">
          Create QR Code
        </button>
        <p class="inline-error" id="at-error"></p>
        <div class="qr-result-box" id="at-result">
          <img id="at-img" src="" alt="AssureTag nested QR" />
          <div class="serial-badge">
            <strong>AssureTag Inner Serial:</strong><br/><span id="at-serial"></span>
          </div>
          <p style="font-size:.75rem;color:var(--muted);text-align:center;">
            Outer QR \u2192 your text &nbsp;&bull;&nbsp; Inner QR \u2192 tamper-proof serial
          </p>
          <a id="at-dl" download="assuretag.png" class="btn btn-green" style="width:100%;">
            &#8681; Download PNG
          </a>
        </div>
        <div class="divider" style="margin:0;"></div>
        <div class="card-demo">
          <p class="demo-label">&#128269; Verify / Authenticate</p>
          <div class="btn-row">
            <button class="btn btn-primary" onclick="openAssureScanner('at')" style="flex:1;">
              &#128247; Scan to Verify
            </button>
          </div>
          <div id="at-scan-camera" style="display:none;flex-direction:column;gap:.75rem;">
            <div style="position:relative;">
              <video id="at-scan-video" playsinline muted
                     style="width:100%;border-radius:8px;background:#000;max-height:220px;object-fit:contain;display:block;"></video>
              <canvas id="at-scan-overlay" style="position:absolute;top:0;left:0;width:100%;height:100%;border-radius:8px;pointer-events:none;"></canvas>
            </div>
            <div style="display:flex;gap:.5rem;flex-wrap:wrap;">
              <span id="at-outer-badge" style="display:inline-flex;align-items:center;gap:.35rem;padding:.3rem .75rem;
                    border-radius:20px;font-size:.78rem;font-weight:700;background:#f1f5f9;color:#94a3b8;">
                &#9711; Outer QR
              </span>
              <span id="at-inner-badge" style="display:inline-flex;align-items:center;gap:.35rem;padding:.3rem .75rem;
                    border-radius:20px;font-size:.78rem;font-weight:700;background:#f1f5f9;color:#94a3b8;">
                &#9711; Inner QR
              </span>
            </div>
            <div id="at-scan-guide" style="font-size:.82rem;color:#64748b;min-height:1.2rem;"></div>
            <div id="at-auto-bar" style="display:none;">
              <div style="font-size:.75rem;color:#64748b;margin-bottom:.3rem;" id="at-auto-label">Hold steady…</div>
              <div style="height:6px;background:#e5e7eb;border-radius:4px;overflow:hidden;">
                <div id="at-auto-progress" style="height:100%;width:0%;background:#22c55e;border-radius:4px;transition:width .05s linear;"></div>
              </div>
            </div>
            <div class="btn-row">
              <button class="btn btn-ghost" onclick="closeAssureScanner('at')" style="font-size:.85rem;">&#9632; Stop Camera</button>
              <button class="btn btn-accent" id="at-capture-btn"
                      onclick="triggerAssureCapture('at','assure_tag')"
                      style="font-size:.9rem;display:none;">&#128247; Capture Now</button>
            </div>
          </div>
          <div id="at-scan-result" style="display:none;"></div>
          <div style="display:flex;align-items:center;gap:.6rem;margin-top:.25rem;">
            <div style="flex:1;height:1px;background:#e5e7eb;"></div>
            <span style="font-size:.75rem;color:#94a3b8;">or</span>
            <div style="flex:1;height:1px;background:#e5e7eb;"></div>
          </div>
          <div>
            <input type="file" id="at-upload-input" accept="image/*" style="display:none;"
                   onchange="handleAssureUpload(event,'at','assure_tag')" />
            <button class="btn btn-ghost" style="width:100%;font-size:.88rem;"
                    onclick="document.getElementById('at-upload-input').click()">
              &#128228; Upload QR Code Image
            </button>
          </div>
        </div>
      </div>
    </div>

    <div class="product-card">
      <div class="card-header">
        <span class="badge badge-trace">&#128202; AssureTrace\u2122</span>
        <h2>AssureTrace\u2122</h2>
        <p class="tagline">Tracing technology that tracks your products through their full logistics lifecycle
        \u2014 from manufacture to final delivery \u2014 with tamper-evident serialized nested QR codes and
        real-time geographic event tracking.</p>
      </div>
      <div class="divider"></div>
      <div class="card-demo">
        <p class="demo-label">Create &amp; Track Demo</p>
        <input class="demo-input" id="atrac-input" placeholder="Enter product or batch ID\u2026"
               onkeydown="if(event.key==='Enter') atracGenerate()" />
        <div class="btn-row">
          <button class="btn btn-primary" id="atrac-btn" onclick="atracGenerate()" style="flex:1;">
            Create Serialized QR
          </button>
          <button class="btn btn-accent" id="atrac-sim-btn" onclick="atracSimulate()" style="display:none;">
            &#127758; Simulate Track
          </button>
        </div>
        <p class="inline-error" id="atrac-error"></p>
        <div class="qr-result-box" id="atrac-result">
          <img id="atrac-img" src="" alt="AssureTrace QR" />
          <div class="serial-badge">
            <strong>Trace Serial:</strong><br/><span id="atrac-serial"></span>
          </div>
          <p style="font-size:.75rem;color:var(--muted);text-align:center;">
            Click <strong>Simulate Track</strong> to animate the product journey across India
          </p>
          <a id="atrac-dl" download="assuretrace.png" class="btn btn-green" style="width:100%;">
            &#8681; Download PNG
          </a>
        </div>
        <div class="divider" style="margin:0;"></div>
        <div class="card-demo">
          <p class="demo-label">&#128269; Verify / Authenticate</p>
          <div class="btn-row">
            <button class="btn btn-primary" onclick="openAssureScanner('atrac')" style="flex:1;">
              &#128247; Scan to Verify
            </button>
          </div>
          <div id="atrac-scan-camera" style="display:none;flex-direction:column;gap:.75rem;">
            <div style="position:relative;">
              <video id="atrac-scan-video" playsinline muted
                     style="width:100%;border-radius:8px;background:#000;max-height:220px;object-fit:contain;display:block;"></video>
              <canvas id="atrac-scan-overlay" style="position:absolute;top:0;left:0;width:100%;height:100%;border-radius:8px;pointer-events:none;"></canvas>
            </div>
            <div style="display:flex;gap:.5rem;flex-wrap:wrap;">
              <span id="atrac-outer-badge" style="display:inline-flex;align-items:center;gap:.35rem;padding:.3rem .75rem;
                    border-radius:20px;font-size:.78rem;font-weight:700;background:#f1f5f9;color:#94a3b8;">
                &#9711; Outer QR
              </span>
              <span id="atrac-inner-badge" style="display:inline-flex;align-items:center;gap:.35rem;padding:.3rem .75rem;
                    border-radius:20px;font-size:.78rem;font-weight:700;background:#f1f5f9;color:#94a3b8;">
                &#9711; Inner QR
              </span>
            </div>
            <div id="atrac-scan-guide" style="font-size:.82rem;color:#64748b;min-height:1.2rem;"></div>
            <div id="atrac-auto-bar" style="display:none;">
              <div style="font-size:.75rem;color:#64748b;margin-bottom:.3rem;" id="atrac-auto-label">Hold steady…</div>
              <div style="height:6px;background:#e5e7eb;border-radius:4px;overflow:hidden;">
                <div id="atrac-auto-progress" style="height:100%;width:0%;background:#22c55e;border-radius:4px;transition:width .05s linear;"></div>
              </div>
            </div>
            <div class="btn-row">
              <button class="btn btn-ghost" onclick="closeAssureScanner('atrac')" style="font-size:.85rem;">&#9632; Stop Camera</button>
              <button class="btn btn-accent" id="atrac-capture-btn"
                      onclick="triggerAssureCapture('atrac','assure_trace')"
                      style="font-size:.9rem;display:none;">&#128247; Capture Now</button>
            </div>
          </div>
          <div id="atrac-scan-result" style="display:none;"></div>
          <div style="display:flex;align-items:center;gap:.6rem;margin-top:.25rem;">
            <div style="flex:1;height:1px;background:#e5e7eb;"></div>
            <span style="font-size:.75rem;color:#94a3b8;">or</span>
            <div style="flex:1;height:1px;background:#e5e7eb;"></div>
          </div>
          <div>
            <input type="file" id="atrac-upload-input" accept="image/*" style="display:none;"
                   onchange="handleAssureUpload(event,'atrac','assure_trace')" />
            <button class="btn btn-ghost" style="width:100%;font-size:.88rem;"
                    onclick="document.getElementById('atrac-upload-input').click()">
              &#128228; Upload QR Code Image
            </button>
          </div>
        </div>
      </div>
    </div>

    <div class="product-card">
      <div class="card-header">
        <span class="badge badge-stock">&#128179; AssureStock\u2122</span>
        <h2>AssureStock\u2122</h2>
        <p class="tagline">Complete real-time visibility into your warehouse inventory. Scan multiple QR
        codes with your camera to instantly build a live stock-count table with unique item identification
        and running totals.</p>
      </div>
      <div class="divider"></div>
      <div class="card-demo">
        <p class="demo-label">Inventory Scanner Demo</p>
        <div class="btn-row">
          <button class="btn btn-primary" id="stock-open-btn" onclick="openStockScanner()">
            &#128247; Start Scanning
          </button>
          <button class="btn btn-ghost" id="stock-clear-btn" onclick="clearStock()" style="display:none;">
            &#128465; Clear
          </button>
        </div>
        <div id="stock-scanner-area">
          <video id="stock-video" playsinline muted></video>
          <div class="btn-row">
            <button class="btn btn-accent" id="stock-capture-btn" onclick="stockCapture()" style="flex:1;">
              &#128247; Capture QR Code
            </button>
            <button class="btn btn-ghost" onclick="closeStockScanner()">&#9632; Stop</button>
          </div>
          <div id="stock-last-scan-box">
            &#10003; Scanned: <strong id="stock-last-scan"></strong>
          </div>
        </div>
        <div id="stock-results">
          <p id="stock-total" style="font-size:.82rem;color:var(--muted);margin-bottom:.5rem;"></p>
          <table class="stock-table">
            <thead>
              <tr>
                <th>#</th><th>QR Content</th>
                <th style="text-align:center;">Count</th>
              </tr>
            </thead>
            <tbody id="stock-table-body"></tbody>
          </table>
        </div>
      </div>
    </div>

  </div>
</section>

<!-- TRACE MAP (revealed after simulate) -->
<div id="trace-map-section">
  <div class="trace-map-card">
    <h3>&#127758; Product Journey \u2014 Track &amp; Trace Simulation</h3>
    <div class="trace-grid">
      <div id="trace-map"></div>
      <div>
        <p class="timeline-label">Logistics Timeline</p>
        <div id="trace-timeline"></div>
      </div>
    </div>
  </div>
</div>

<footer>
  <p>&copy; 2026 <a href="https://rudron.com" target="_blank" rel="noopener">RuDron Technolabs Pvt. Ltd.</a>
     &mdash; All rights reserved.</p>
  <p style="margin-top:.35rem;font-size:.8rem;opacity:.65;">
    &lsquo;QR Code&rsquo; is a trademark of DENSO WAVE INCORPORATED
  </p>
</footer>

<script>
// ── ASSURE SCANNER (shared for AssureTag & AssureTrace) ──────────────────────
const _assureStreams  = {{}};
const _assureLoops   = {{}};
const _offCanvas     = {{}};
const _stableCounts  = {{}};
const STABLE_NEEDED  = 20;   // frames outer QR must be held steady before auto-capture
const _capturing     = {{}};  // prevent double-trigger
const _fallbackTimers = {{}};

async function openAssureScanner(prefix) {{
  const cam = document.getElementById(prefix + '-scan-camera');
  cam.style.display = 'flex';
  document.getElementById(prefix + '-scan-result').style.display = 'none';
  _setBadge(prefix, 'outer', 'idle');
  _setBadge(prefix, 'inner', 'idle');
  _setGuide(prefix, '&#128247; Point the camera at the QR code\u2026');
  _setProgress(prefix, 0, false);
  _stableCounts[prefix] = 0;
  _capturing[prefix]    = false;
  const manBtn = document.getElementById(prefix + '-capture-btn');
  if (manBtn) manBtn.style.display = 'none';
  // 5-second fallback — show Capture Now button if no QR detected yet
  clearTimeout(_fallbackTimers[prefix]);
  _fallbackTimers[prefix] = setTimeout(() => {{
    if (_capturing[prefix]) return;
    const btn = document.getElementById(prefix + '-capture-btn');
    if (btn) {{
      btn.style.display = 'inline-flex';
      _setGuide(prefix, '&#9888;&#65039; Struggling to detect? Click <strong>Capture Now</strong> to process the image as-is.');
    }}
  }}, 5000);
  try {{
    const stream = await navigator.mediaDevices.getUserMedia({{
      video: {{ facingMode: 'environment', width: {{ ideal: 1920 }}, height: {{ ideal: 1080 }} }}
    }});
    _assureStreams[prefix] = stream;
    const video = document.getElementById(prefix + '-scan-video');
    video.srcObject = stream;
    await video.play();
    _offCanvas[prefix] = document.createElement('canvas');
    _assureLoops[prefix] = true;
    requestAnimationFrame(() => _scanLoop(prefix));
  }} catch (e) {{ alert('Camera error: ' + e.message); }}
}}

function closeAssureScanner(prefix) {{
  clearTimeout(_fallbackTimers[prefix]);
  _assureLoops[prefix] = false;
  _capturing[prefix]   = false;
  if (_assureStreams[prefix]) {{
    _assureStreams[prefix].getTracks().forEach(t => t.stop());
    delete _assureStreams[prefix];
  }}
  const v = document.getElementById(prefix + '-scan-video');
  if (v) v.srcObject = null;
  const overlay = document.getElementById(prefix + '-scan-overlay');
  if (overlay) overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);
  document.getElementById(prefix + '-scan-camera').style.display = 'none';
  _setProgress(prefix, 0, false);
}}

function _scanLoop(prefix) {{
  if (!_assureLoops[prefix] || _capturing[prefix]) return;
  const video   = document.getElementById(prefix + '-scan-video');
  const overlay = document.getElementById(prefix + '-scan-overlay');
  if (!video.videoWidth) {{ requestAnimationFrame(() => _scanLoop(prefix)); return; }}

  const oc = _offCanvas[prefix];
  const w = video.videoWidth, h = video.videoHeight;
  oc.width = w; oc.height = h;
  const ctx = oc.getContext('2d', {{ willReadFrequently: true }});
  ctx.drawImage(video, 0, 0);

  // ─ outer scan (full frame)
  const fullData = ctx.getImageData(0, 0, w, h);
  const outerQR  = typeof jsQR !== 'undefined' ? jsQR(fullData.data, w, h) : null;

  // ─ inner scan preview (centre crop, upscaled)
  const cropFrac = 0.32;
  const cx = Math.round(w / 2), cy = Math.round(h / 2);
  const ch = Math.round(Math.min(w, h) * cropFrac);
  const x1 = Math.max(0, cx - ch), y1 = Math.max(0, cy - ch);
  const cw2 = Math.min(w - x1, ch * 2), ch2 = Math.min(h - y1, ch * 2);
  const cropData = ctx.getImageData(x1, y1, cw2, ch2);
  const tmpC = document.createElement('canvas');
  const scale = Math.max(2, Math.round(400 / Math.max(cw2, ch2)));
  tmpC.width = cw2 * scale; tmpC.height = ch2 * scale;
  const tctx = tmpC.getContext('2d');
  const miniC = document.createElement('canvas');
  miniC.width = cw2; miniC.height = ch2;
  miniC.getContext('2d').putImageData(cropData, 0, 0);
  tctx.imageSmoothingEnabled = false;
  tctx.drawImage(miniC, 0, 0, tmpC.width, tmpC.height);
  const upData  = tctx.getImageData(0, 0, tmpC.width, tmpC.height);
  const innerQR = typeof jsQR !== 'undefined' ? jsQR(upData.data, tmpC.width, tmpC.height) : null;
  const innerValid = innerQR && (!outerQR || innerQR.data !== outerQR.data);

  // ─ draw overlay
  overlay.width = w; overlay.height = h;
  const octx = overlay.getContext('2d');
  octx.clearRect(0, 0, w, h);
  if (outerQR) _drawQRBox(octx, outerQR.location, '#22c55e', 3);
  const innerColor = innerValid ? '#22c55e' : (outerQR ? '#f59e0b' : '#94a3b8');
  octx.strokeStyle = innerColor; octx.lineWidth = 2;
  octx.setLineDash([6, 4]);
  octx.strokeRect(x1, y1, cw2, ch2);
  octx.setLineDash([]);
  octx.fillStyle = innerColor; octx.font = 'bold 11px system-ui';
  octx.fillText('Inner QR zone', x1 + 4, y1 - 4);

  // ─ stability & auto-capture logic
  if (outerQR) {{
    _stableCounts[prefix] = (_stableCounts[prefix] || 0) + 1;
    const pct = Math.min(100, Math.round(_stableCounts[prefix] / STABLE_NEEDED * 100));
    _setProgress(prefix, pct, true);

    if (innerValid) {{
      _setBadge(prefix, 'inner', 'ok');
      _setGuide(prefix, '&#10003; Both codes visible \u2014 auto-capturing\u2026');
    }} else {{
      _setBadge(prefix, 'inner', 'warn');
      _setGuide(prefix, '&#128269; Outer detected. Move <strong>closer to centre</strong> for inner code. Auto-capturing when stable\u2026');
    }}
    _setBadge(prefix, 'outer', 'ok');

    const manBtn = document.getElementById(prefix + '-capture-btn');
    if (manBtn && _stableCounts[prefix] >= 5) manBtn.style.display = 'inline-flex';

    if (_stableCounts[prefix] >= STABLE_NEEDED) {{
      // ── AUTO-CAPTURE ──
      _capturing[prefix] = true;
      _assureLoops[prefix] = false;
      _setGuide(prefix, '&#9889; QR code locked \u2014 processing\u2026');
      // freeze current frame into a blob and send
      oc.toBlob(blob => {{
        const endpointMap = {{ 'at': 'assure_tag', 'atrac': 'assure_trace' }};
        const ep = endpointMap[prefix] || prefix;
        _submitScan(prefix, ep, blob);
      }}, 'image/png');
      return; // stop loop
    }}
  }} else {{
    // lost outer QR — reset stability
    _stableCounts[prefix] = 0;
    _setBadge(prefix, 'outer', 'idle');
    _setBadge(prefix, 'inner', 'idle');
    _setGuide(prefix, '&#128247; Point camera at the QR code \u2014 keep it centred and steady.');
    _setProgress(prefix, 0, true);
    const manBtn = document.getElementById(prefix + '-capture-btn');
    if (manBtn) manBtn.style.display = 'none';
  }}

  requestAnimationFrame(() => _scanLoop(prefix));
}}

function _setProgress(prefix, pct, show) {{
  const bar  = document.getElementById(prefix + '-auto-bar');
  const fill = document.getElementById(prefix + '-auto-progress');
  const lbl  = document.getElementById(prefix + '-auto-label');
  if (!bar || !fill) return;
  bar.style.display  = show ? 'block' : 'none';
  fill.style.width   = pct + '%';
  fill.style.background = pct >= 100 ? '#22c55e' : (pct > 50 ? '#f59e0b' : '#1a3c6e');
  if (lbl) lbl.textContent = pct >= 100 ? 'Capturing\u2026' : (pct > 0 ? 'Hold steady\u2026 ' + pct + '%' : 'Hold steady\u2026');
}}

async function triggerAssureCapture(prefix, endpoint) {{
  // manual fallback
  if (_capturing[prefix]) return;
  _capturing[prefix] = true;
  _assureLoops[prefix] = false;
  const oc = _offCanvas[prefix];
  if (!oc || !oc.width) {{ alert('No frame captured yet.'); _capturing[prefix] = false; return; }}
  oc.toBlob(blob => _submitScan(prefix, endpoint, blob), 'image/png');
}}

async function _submitScan(prefix, endpoint, blob) {{
  closeAssureScanner(prefix);
  const fd = new FormData();
  fd.append('file', blob, 'scan.png');
  const resultBox = document.getElementById(prefix + '-scan-result');
  resultBox.innerHTML = '<div style="display:flex;align-items:center;gap:.6rem;color:#64748b;font-size:.88rem;">' +
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
    '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>Analysing image\u2026</div>';
  resultBox.style.display = 'block';
  try {{
    const res  = await fetch('/' + endpoint + '/scan', {{ method: 'POST', body: fd }});
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Scan failed');
    renderAssureScanResult(resultBox, data);
  }} catch (e) {{
    resultBox.innerHTML = `<div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;padding:.85rem;">
      <p style="color:#dc2626;font-weight:700;">&#9888;&#65039; Error</p>
      <p style="font-size:.88rem;color:#991b1b;">${{escHtml(e.message)}}</p></div>`;
  }}
}}

// manual capture (old name kept for any inline refs)
async function captureAssureScan(prefix, endpoint) {{ triggerAssureCapture(prefix, endpoint); }}

// ── UPLOAD HANDLER ───────────────────────────────────────────
async function handleAssureUpload(event, prefix, endpoint) {{
  const file = event.target.files[0];
  // reset input so same file can be re-selected
  event.target.value = '';
  if (!file) return;
  const resultBox = document.getElementById(prefix + '-scan-result');
  resultBox.innerHTML = '<div style="display:flex;align-items:center;gap:.6rem;color:#64748b;font-size:.88rem;">' +
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
    '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>Analysing image…</div>';
  resultBox.style.display = 'block';
  const fd = new FormData();
  fd.append('file', file, file.name);
  try {{
    const res  = await fetch('/' + endpoint + '/scan', {{ method: 'POST', body: fd }});
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Scan failed');
    renderAssureScanResult(resultBox, data);
  }} catch (e) {{
    resultBox.innerHTML = `<div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;padding:.85rem;">
      <p style="color:#dc2626;font-weight:700;">&#9888;&#65039; Error</p>
      <p style="font-size:.88rem;color:#991b1b;">${{escHtml(e.message)}}</p></div>`;
  }}
}}

function _drawQRBox(ctx, loc, color, lw) {{
  if (!loc) return;
  ctx.strokeStyle = color; ctx.lineWidth = lw;
  ctx.beginPath();
  ctx.moveTo(loc.topLeftCorner.x, loc.topLeftCorner.y);
  ctx.lineTo(loc.topRightCorner.x, loc.topRightCorner.y);
  ctx.lineTo(loc.bottomRightCorner.x, loc.bottomRightCorner.y);
  ctx.lineTo(loc.bottomLeftCorner.x, loc.bottomLeftCorner.y);
  ctx.closePath(); ctx.stroke();
}}

function _setBadge(prefix, which, state) {{
  const el = document.getElementById(prefix + '-' + which + '-badge');
  if (!el) return;
  const label = which === 'outer' ? 'Outer QR' : 'Inner QR';
  if (state === 'ok') {{
    el.style.background = '#dcfce7'; el.style.color = '#166534';
    el.innerHTML = '&#10003; ' + label;
  }} else if (state === 'warn') {{
    el.style.background = '#fef9c3'; el.style.color = '#a16207';
    el.innerHTML = '&#128269; ' + label + ' \u2014 move closer';
  }} else {{
    el.style.background = '#f1f5f9'; el.style.color = '#94a3b8';
    el.innerHTML = '&#9711; ' + label;
  }}
}}

function _setGuide(prefix, html) {{
  const el = document.getElementById(prefix + '-scan-guide');
  if (el) el.innerHTML = html;
}}

function renderAssureScanResult(box, data) {{
  const outer = data.outer_code;
  const inner = data.inner_code;
  const authentic = data.authentic;

  // ── row helpers ───────────────────────────────────────────────
  function scanRow(label, value, detected) {{
    const icon  = detected ? '&#10003;' : '&#9711;';
    const color = detected ? '#166534'  : '#94a3b8';
    const bg    = detected ? '#f0fdf4'  : '#f8fafc';
    const border= detected ? '#86efac'  : '#e2e8f0';
    const val   = detected ? `<code style="word-break:break-all;font-size:.8rem;">${{escHtml(value)}}</code>`
                           : `<span style="color:#94a3b8;font-size:.83rem;">(not detected)</span>`;
    return `<div style="background:${{bg}};border:1.5px solid ${{border}};border-radius:8px;padding:.7rem .9rem;display:flex;gap:.75rem;align-items:flex-start;">
      <span style="font-size:1rem;color:${{color}};margin-top:.05rem;">${{icon}}</span>
      <div style="flex:1;min-width:0;">
        <p style="font-weight:700;font-size:.82rem;color:${{color}};text-transform:uppercase;letter-spacing:.04em;margin-bottom:.25rem;">${{label}}</p>
        ${{val}}
      </div>
    </div>`;
  }}

  // ── DB verification row ───────────────────────────────────────
  let dbRow = '';
  if (inner) {{
    if (authentic) {{
      dbRow = `<div style="background:#f0fdf4;border:1.5px solid #86efac;border-radius:8px;padding:.7rem .9rem;display:flex;gap:.75rem;align-items:center;">
        <span style="font-size:1.1rem;">&#9989;</span>
        <p style="font-weight:700;color:#166534;font-size:.88rem;">Inner serial verified in RuDron database — <em>Authentic</em></p>
      </div>`;
    }} else {{
      dbRow = `<div style="background:#fef2f2;border:1.5px solid #fca5a5;border-radius:8px;padding:.7rem .9rem;display:flex;gap:.75rem;align-items:center;">
        <span style="font-size:1.1rem;">&#10060;</span>
        <p style="font-weight:700;color:#dc2626;font-size:.88rem;">Inner serial NOT found in database — <em>Possibly tampered</em></p>
      </div>`;
    }}
  }} else {{
    dbRow = `<div style="background:#fffbeb;border:1.5px solid #fcd34d;border-radius:8px;padding:.7rem .9rem;display:flex;gap:.75rem;align-items:center;">
      <span style="font-size:1.1rem;">&#9888;&#65039;</span>
      <p style="font-weight:600;color:#92400e;font-size:.88rem;">Inner QR not detected — could not verify against database</p>
    </div>`;
  }}

  box.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:.6rem;">
      ${{scanRow('Outer QR Scan', outer, !!outer)}}
      ${{scanRow('Inner QR Scan', inner, !!inner)}}
      ${{dbRow}}
    </div>`;
}}

// ── ASSURETAG ──────────────────────────────────────────────────────────────
async function atGenerate() {{
  const data = document.getElementById('at-input').value.trim();
  const errEl = document.getElementById('at-error');
  errEl.style.display = 'none';
  if (!data) {{ errEl.textContent = 'Please enter some text.'; errEl.style.display = 'block'; return; }}
  const btn = document.getElementById('at-btn');
  btn.disabled = true; btn.textContent = 'Generating\u2026';
  document.getElementById('at-result').style.display = 'none';
  try {{
    const res = await fetch('/assure_tag/generate', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{data: data}})
    }});
    if (!res.ok) {{ const b = await res.json().catch(() => ({{}})); throw new Error(b.detail || 'Failed'); }}
    const serial = res.headers.get('X-Serial');
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    document.getElementById('at-img').src = url;
    document.getElementById('at-dl').href = url;
    document.getElementById('at-serial').textContent = serial;
    document.getElementById('at-result').style.display = 'flex';
  }} catch (e) {{ errEl.textContent = e.message; errEl.style.display = 'block'; }}
  finally {{ btn.disabled = false; btn.textContent = 'Create QR Code'; }}
}}

// ── ASSURETRACE ────────────────────────────────────────────────────────────
let _traceId = null;
let _traceMap = null;

async function atracGenerate() {{
  const data = document.getElementById('atrac-input').value.trim();
  const errEl = document.getElementById('atrac-error');
  errEl.style.display = 'none';
  if (!data) {{ errEl.textContent = 'Please enter a product or batch ID.'; errEl.style.display = 'block'; return; }}
  const btn = document.getElementById('atrac-btn');
  btn.disabled = true; btn.textContent = 'Generating\u2026';
  document.getElementById('atrac-result').style.display = 'none';
  document.getElementById('atrac-sim-btn').style.display = 'none';
  try {{
    const res = await fetch('/assure_trace/generate', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{data: data}})
    }});
    if (!res.ok) {{ const b = await res.json().catch(() => ({{}})); throw new Error(b.detail || 'Failed'); }}
    _traceId = res.headers.get('X-Event-Id');
    const serial = res.headers.get('X-Serial');
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    document.getElementById('atrac-img').src = url;
    document.getElementById('atrac-dl').href = url;
    document.getElementById('atrac-serial').textContent = serial;
    document.getElementById('atrac-result').style.display = 'flex';
    document.getElementById('atrac-sim-btn').style.display = 'inline-flex';
  }} catch (e) {{ errEl.textContent = e.message; errEl.style.display = 'block'; }}
  finally {{ btn.disabled = false; btn.textContent = 'Create Serialized QR'; }}
}}

async function atracSimulate() {{
  if (!_traceId) return;
  const btn = document.getElementById('atrac-sim-btn');
  btn.disabled = true; btn.textContent = 'Simulating\u2026';
  try {{
    const res = await fetch('/assure_trace/simulate/' + _traceId);
    if (!res.ok) throw new Error('Simulation failed');
    const data = await res.json();
    renderTraceMap(data);
  }} catch (e) {{ alert('Simulation error: ' + e.message); }}
  finally {{ btn.disabled = false; btn.textContent = '🌎 Simulate Track'; }}
}}

function renderTraceMap(data) {{
  const section = document.getElementById('trace-map-section');
  section.style.display = 'block';
  if (_traceMap) {{ _traceMap.remove(); _traceMap = null; }}
  _traceMap = L.map('trace-map').setView([22.5, 80.0], 4);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }}).addTo(_traceMap);
  const colors = ['#1a3c6e', '#e8a020', '#10b981'];
  const points = [];
  data.events.forEach((ev, i) => {{
    const color = colors[i % colors.length];
    L.circleMarker([ev.lat, ev.lng], {{
      radius: 14, fillColor: color, color: '#fff', weight: 2.5, fillOpacity: 1
    }}).addTo(_traceMap)
      .bindPopup(`<div style="min-width:160px"><strong style="color:${{color}}">${{ev.status}}</strong><br/>${{ev.city}}, ${{ev.state}}<br/><small>${{new Date(ev.timestamp).toLocaleString('en-IN', {{timeZone:'Asia/Kolkata'}})}}</small></div>`);
    points.push([ev.lat, ev.lng]);
  }});
  L.polyline(points, {{color: '#1a3c6e', weight: 2, dashArray: '8,6', opacity: .7}}).addTo(_traceMap);
  document.getElementById('trace-timeline').innerHTML = data.events.map((ev, i) => `
    <div style="display:flex;gap:.75rem;align-items:flex-start;padding:.75rem 0;${{i < data.events.length - 1 ? 'border-bottom:1px solid #e5e7eb' : ''}}">
      <div style="width:34px;height:34px;border-radius:50%;background:${{colors[i % colors.length]}};color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.9rem;flex-shrink:0;">${{i + 1}}</div>
      <div>
        <div style="font-weight:700;color:#1a3c6e;">${{ev.status}}</div>
        <div style="font-size:.88rem;">${{ev.city}}, ${{ev.state}}</div>
        <div style="font-size:.8rem;color:#64748b;">${{new Date(ev.timestamp).toLocaleString('en-IN', {{timeZone:'Asia/Kolkata'}})}}</div>
        <div style="font-size:.76rem;color:#94a3b8;font-family:monospace;">Serial: ${{ev.serial}}</div>
      </div>
    </div>`).join('');
  section.scrollIntoView({{behavior: 'smooth'}});
}}

// ── ASSURESTOCK ────────────────────────────────────────────────────────────
let _stockStream = null;
let _stockInventory = {{}};

function openStockScanner() {{
  document.getElementById('stock-scanner-area').style.display = 'flex';
  document.getElementById('stock-open-btn').style.display = 'none';
  document.getElementById('stock-clear-btn').style.display = 'inline-flex';
  startStockCamera();
}}

async function startStockCamera() {{
  try {{
    _stockStream = await navigator.mediaDevices.getUserMedia({{
      video: {{facingMode: 'environment', width: {{ideal: 1280}}, height: {{ideal: 720}}}}
    }});
    const video = document.getElementById('stock-video');
    video.srcObject = _stockStream;
    await video.play();
  }} catch (e) {{ alert('Camera access denied: ' + e.message); }}
}}

function stopStockCamera() {{
  if (_stockStream) {{ _stockStream.getTracks().forEach(t => t.stop()); _stockStream = null; }}
  document.getElementById('stock-video').srcObject = null;
}}

async function stockCapture() {{
  const video = document.getElementById('stock-video');
  if (!video.videoWidth) {{ alert('Camera not ready yet.'); return; }}
  const btn = document.getElementById('stock-capture-btn');
  btn.disabled = true; btn.textContent = 'Scanning\u2026';
  try {{
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const blob = await new Promise((res, rej) =>
      canvas.toBlob(b => b ? res(b) : rej(new Error('Capture failed')), 'image/png')
    );
    const fd = new FormData();
    fd.append('file', blob, 'stock_scan.png');
    const res = await fetch('/scan_qr_code', {{method: 'POST', body: fd}});
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || 'No QR code detected');
    const content = body.result;
    _stockInventory[content] = (_stockInventory[content] || 0) + 1;
    updateStockTable();
    document.getElementById('stock-last-scan').textContent = content;
    document.getElementById('stock-last-scan-box').style.display = 'block';
  }} catch (e) {{ alert(e.message); }}
  finally {{ btn.disabled = false; btn.textContent = '&#128247; Capture QR Code'; }}
}}

function updateStockTable() {{
  const entries = Object.entries(_stockInventory).sort((a, b) => b[1] - a[1]);
  document.getElementById('stock-table-body').innerHTML = entries.map(([content, count], i) => `
    <tr>
      <td>${{i + 1}}</td>
      <td style="font-family:monospace;word-break:break-all;">${{escHtml(content)}}</td>
      <td style="text-align:center;">
        <span style="background:#1a3c6e;color:#fff;border-radius:12px;padding:.2rem .75rem;font-size:.85rem;font-weight:700;">${{count}}</span>
      </td>
    </tr>`).join('');
  const total = Object.values(_stockInventory).reduce((a, b) => a + b, 0);
  document.getElementById('stock-total').textContent =
    total + ' scan' + (total === 1 ? '' : 's') + ' \u2014 ' +
    entries.length + ' unique item' + (entries.length === 1 ? '' : 's');
  document.getElementById('stock-results').style.display = 'block';
}}

function escHtml(str) {{
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}}

function clearStock() {{
  _stockInventory = {{}};
  document.getElementById('stock-table-body').innerHTML = '';
  document.getElementById('stock-results').style.display = 'none';
  document.getElementById('stock-last-scan-box').style.display = 'none';
  document.getElementById('stock-total').textContent = '';
}}

function closeStockScanner() {{
  stopStockCamera();
  document.getElementById('stock-scanner-area').style.display = 'none';
  document.getElementById('stock-open-btn').style.display = 'inline-flex';
}}
</script>
</body>
</html>"""
        return HTMLResponse(content=html)

    return app


app = create_app()
