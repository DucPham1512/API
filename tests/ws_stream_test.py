"""
Webcam → VSR WebSocket stream client.

Main thread  : camera capture + PyQt6 preview window with MediaPipe overlay
Worker thread: WebSocket connect / send / recv  (no Qt / cv2 display touching)

Usage:
    python tests/ws_stream_test.py
    python tests/ws_stream_test.py --camera 1
    python tests/ws_stream_test.py --url ws://192.168.1.10:8000/vsr/stream
"""

import argparse
import json
import os
import queue
import sys
import threading
import time

# Make `app` importable when run as: python tests/ws_stream_test.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import websockets.sync.client as wss
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

from app.preprocessor.face_localizer import FaceLocalizer, LocalizerResult, download_model

WS_URL = "ws://127.0.0.1:8000/vsr/stream"
TARGET_FPS = 25
JPEG_QUALITY = 80

_GREEN  = (0, 255, 0)    # lip bounding box
_CYAN   = (255, 255, 0)  # lip landmark points
_YELLOW = (0, 255, 255)  # alignment anchor points
_RED    = (0, 0, 255)    # "no face" text


def draw_localizer_result(frame, result: "LocalizerResult | None"):
    """Return a copy of frame with lip bbox, landmarks, and anchors drawn."""
    vis = frame.copy()
    if result is None:
        cv2.putText(vis, "No face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, _RED, 2, cv2.LINE_AA)
        return vis

    bbox = result.bbox
    x1, y1 = int(bbox.x), int(bbox.y)
    x2, y2 = int(bbox.x + bbox.w), int(bbox.y + bbox.h)
    cv2.rectangle(vis, (x1, y1), (x2, y2), _GREEN, 2)

    for pt in result.landmarks:
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, _CYAN, -1)

    for pt in result.anchor_landmarks:
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, _YELLOW, 2)

    return vis


class PreviewWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VSR stream — press Q to stop")
        self.label = QLabel()
        self.setCentralWidget(self.label)

    def update_frame(self, frame):
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        self.label.setPixmap(QPixmap.fromImage(img))
        self.resize(w, h)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Q:
            self.close()


def ws_worker(frame_queue: queue.Queue, url: str, stop_event: threading.Event):
    """Owns the WebSocket entirely — never touches Qt."""
    time.sleep(0.5)  # give main thread a moment to start camera capture
    try:
        with wss.connect(url, open_timeout=5.0) as ws:
            ws.send(json.dumps({"type": "config", "fps": TARGET_FPS}))

            while not stop_event.is_set():
                # send next frame if available
                try:
                    data = frame_queue.get(timeout=0.1)
                    ws.send(data)
                except queue.Empty:
                    pass

                # poll for transcript segments
                try:
                    msg = ws.recv(timeout=0.01)
                    seg = json.loads(msg)
                    if seg.get("is_final"):
                        break
                    print(f"  [{seg['start_ms']:.0f}–{seg['end_ms']:.0f} ms]  {seg['text']}")
                except TimeoutError:
                    pass

            # drain remaining segments after stop
            ws.send(json.dumps({"type": "end"}))
            print("\n[end sent — draining …]")
            try:
                while True:
                    msg = ws.recv(timeout=5.0)
                    seg = json.loads(msg)
                    if seg.get("is_final"):
                        print("[stream complete]")
                        break
                    print(f"  [{seg['start_ms']:.0f}–{seg['end_ms']:.0f} ms]  {seg['text']}")
            except TimeoutError:
                pass

    except Exception as e:
        print(f"[ws error] {e}", file=sys.stderr)
    finally:
        stop_event.set()


def main():
    parser = argparse.ArgumentParser(description="Webcam → VSR WebSocket stream")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--url", default=WS_URL)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}", file=sys.stderr)
        sys.exit(1)

    for _ in range(15):          # let the sensor warm up
        cap.read()

    print(f"Camera {args.camera} ready. Press Q to stop.\n")

    download_model()

    frame_queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()

    worker = threading.Thread(target=ws_worker,
                              args=(frame_queue, args.url, stop_event),
                              daemon=True)
    worker.start()

    app = QApplication(sys.argv)
    window = PreviewWindow()
    window.show()

    last_send = [time.perf_counter()]  # list so tick() can mutate it
    frame_interval = 1.0 / TARGET_FPS
    start_time = time.perf_counter()

    with FaceLocalizer() as localizer:
        def tick():
            if stop_event.is_set():
                app.quit()
                return
            ret, frame = cap.read()
            if not ret:
                return

            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            result = localizer.locate(frame, timestamp_ms)
            window.update_frame(draw_localizer_result(frame, result))

            now = time.perf_counter()
            if now - last_send[0] >= frame_interval:
                last_send[0] = now
                ok, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    try:
                        frame_queue.put_nowait(buf.tobytes())
                    except queue.Full:
                        pass            # drop frame — worker busy, keep preview smooth

        timer = QTimer()
        timer.timeout.connect(tick)
        timer.start(int(1000 / TARGET_FPS))

        app.exec()
        # FaceLocalizer.__exit__ closes the landmarker cleanly here

    stop_event.set()
    cap.release()
    worker.join(timeout=15.0)


if __name__ == "__main__":
    main()
