from __future__ import annotations

import io
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import qrcode
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import PyMongoError


class GenerateQRCodeRequest(BaseModel):
    data: str = Field(min_length=1, description="Text or URL to encode")


class InMemoryRequestStore:
    def __init__(self) -> None:
        self.generate_requests: List[Dict[str, Any]] = []
        self.scan_requests: List[Dict[str, Any]] = []

    def insert_generate_request(self, document: Dict[str, Any]) -> None:
        self.generate_requests.append(document)

    def insert_scan_request(self, document: Dict[str, Any]) -> None:
        self.scan_requests.append(document)


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

            app.state.request_store.insert_scan_request(
                {
                    "event_id": event_id,
                    "file_name": file.filename,
                    "status": "completed",
                    "result": decoded_text,
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

    return app


app = create_app()
