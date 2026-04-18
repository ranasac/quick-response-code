import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from main import InMemoryRequestStore, create_app


class QRCodeAPITestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = InMemoryRequestStore()
        self.app = create_app(request_store=self.store, storage_dir=Path(self.temp_dir.name))
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_generate_qr_code_returns_png_and_persists_record(self) -> None:
        response = self.client.post("/generate_qr_code", json={"data": "https://example.com"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertIn("x-event-id", response.headers)
        self.assertGreater(len(response.content), 0)

        self.assertEqual(len(self.store.generate_requests), 1)
        saved_record = self.store.generate_requests[0]
        self.assertEqual(saved_record["status"], "completed")
        self.assertTrue(saved_record["result_path"].endswith(".png"))
        self.assertTrue(Path(saved_record["result_path"]).exists())

    def test_scan_qr_code_returns_decoded_text_and_persists_record(self) -> None:
        generate_response = self.client.post("/generate_qr_code", json={"data": "hello qr"})

        scan_response = self.client.post(
            "/scan_qr_code",
            files={"file": ("generated.png", generate_response.content, "image/png")},
        )

        self.assertEqual(scan_response.status_code, 200)
        payload = scan_response.json()
        self.assertEqual(payload["result"], "hello qr")
        self.assertIn("event_id", payload)

        self.assertEqual(len(self.store.scan_requests), 1)
        scan_record = self.store.scan_requests[0]
        self.assertEqual(scan_record["status"], "completed")
        self.assertEqual(scan_record["result"], "hello qr")


if __name__ == "__main__":
    unittest.main()
