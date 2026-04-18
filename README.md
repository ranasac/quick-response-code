# quick-response-code

FastAPI service that:

- Generates QR codes from text/URL input (`/generate_qr_code`)
- Scans uploaded QR images (`/scan_qr_code`)
- Persists request/status/result data in MongoDB collections:
  - `generate_qr_code_requests`
  - `scan_qr_code_requests`

## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## API

### `POST /generate_qr_code`
Request body:

```json
{
  "data": "https://example.com"
}
```

Response:

- Returns QR PNG image directly
- Includes `X-Event-Id` response header
- Saves image under `storage/qrcodes/<event_id>.png` by default
- If `AWS_S3_BUCKET` is configured, uploads QR image to S3 and stores S3 path

### `POST /scan_qr_code`
Form-data:

- `file`: PNG/JPG image containing a QR code

Response:

```json
{
  "event_id": "<uuid>",
  "result": "decoded text or url"
}
```

## Environment variables

- `MONGODB_URI` (default: `mongodb://localhost:27017`)
- `MONGODB_DB` (default: `quick_response_code`)
- `QR_STORAGE_DIR` (default: `storage/qrcodes`)
- `AWS_S3_BUCKET` (optional)
- `AWS_S3_PREFIX` (optional, default: `qrcodes`)
