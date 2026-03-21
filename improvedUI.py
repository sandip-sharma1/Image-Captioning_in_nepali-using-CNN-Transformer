"""
sabai same nai ho..just motion detection add gareko ho..camera bata live feed ma motion detect garera caption 
generate garne..motion detect bhayepachi 8 second samma arko caption generate hudaina..motion detect bhayepachi tyo 
frame save garera background thread ma caption generate garne..caption generate bhayepachi tyo caption speak garne..
camera close garda sabai resource release garne..
"""





import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QPushButton, QLabel, QFileDialog, QHBoxLayout, QFrame,
                            QDialog, QMessageBox, QLineEdit)
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt, QTimer
from PIL import Image
import pyttsx3
from gtts import gTTS
import tempfile
import threading
import time
from preprocessing import *
from Transformer import *
from loadingwight import model as caption_model

# ── Global variables 
vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1
valid_images = list(valid_data.keys())
speak = 'कृपया फोटो हाल्नु होस् '

# settings for motion detection 
MOTION_THRESHOLD  = 25      # pixel diff to count as changed
MOTION_MIN_AREA   = 2000    # min contour area — filters noise/shadows
COOLDOWN_SECONDS  = 8       # seconds between captions
BLUR_KERNEL       = (21,21) # gaussian blur before diff


class IPInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter IP Address")
        self.setFixedSize(300, 100)

        layout = QVBoxLayout()
        self.label = QLabel("Enter camera URL (e.g., 192.168.1.75:8080):")
        layout.addWidget(self.label)

        self.ip_input = QLineEdit(self)
        self.ip_input.setPlaceholderText("192.168.x.x:8080")
        layout.addWidget(self.ip_input)

        self.ok_button = QPushButton("Connect", self)
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)


class CameraChoiceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Camera")
        self.setFixedSize(300, 150)

        layout = QVBoxLayout()

        self.laptop_button = QPushButton("Laptop Camera")
        self.laptop_button.clicked.connect(lambda: self.accept_choice("laptop"))
        layout.addWidget(self.laptop_button)

        self.ip_button = QPushButton("IP Camera")
        self.ip_button.clicked.connect(lambda: self.accept_choice("ip"))
        layout.addWidget(self.ip_button)

        self.setLayout(layout)
        self.choice = None

    def accept_choice(self, choice):
        self.choice = choice
        self.accept()


class CaptureDialog(QDialog):
    """
    Live camera preview with motion detection on every frame.
    Identical logic for both laptop and IP camera — only the
    source (0 vs URL) changes.

    Motion detection:
      1. Build a running background average of the still scene
      2. Each frame: blur → diff vs background → threshold → find contours
      3. Any contour bigger than MOTION_MIN_AREA = something moved
      4. Save frame → generate Nepali caption in background thread → speak
      5. COOLDOWN_SECONDS prevents re-triggering immediately after
    """
    def __init__(self, parent=None, is_ip_camera=False, ip_url=None):
        super().__init__(parent)
        self.setWindowTitle("Camera — Motion Detection Active")
        self.setGeometry(200, 200, 700, 560)

        layout = QVBoxLayout()

        # Live feed display
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.preview_label)

        # Motion status
        self.status_label = QLabel("Monitoring... waiting for motion")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "color: green; font-size: 12px; padding: 4px;")
        layout.addWidget(self.status_label)

        # Caption display
        self.caption_label = QLabel("")
        self.caption_label.setAlignment(Qt.AlignCenter)
        self.caption_label.setWordWrap(True)
        self.caption_label.setStyleSheet(
            "color: white; background-color: #1a1a2e; font-size: 13px; "
            "padding: 8px; border-radius: 4px;")
        layout.addWidget(self.caption_label)

        # Buttons
        btn_layout = QHBoxLayout()

        self.capture_button = QPushButton("Capture Now")
        self.capture_button.setStyleSheet(
            "background-color: #2196F3; color: white; "
            "padding: 8px; border-radius: 4px;")
        self.capture_button.clicked.connect(self.manual_capture)
        btn_layout.addWidget(self.capture_button)

        self.close_button = QPushButton("Close")
        self.close_button.setStyleSheet(
            "background-color: #e74c3c; color: white; "
            "padding: 8px; border-radius: 4px;")
        self.close_button.clicked.connect(self.close)
        btn_layout.addWidget(self.close_button)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # Open camera — laptop=0, IP=url, same code either way
        source = ip_url if (is_ip_camera and ip_url) else 0
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            QMessageBox.critical(
                self, "Error",
                "Could not open camera.\n"
                "For IP camera check the URL and that the app is running.")
            self.close()
            return

        # Motion detection state
        self.bg_avg       = None   # running background average (float)
        self.last_caption = 0.0   # timestamp of last caption event
        self.captioning   = False  # lock — prevents double trigger
        self.captured_image = None

        # Single timer drives both display and motion detection
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(33)  # ~30 fps

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # ── Step 1: prepare grayscale blurred frame ---------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)

        # First frame ever — set as background, nothing to compare yet
        if self.bg_avg is None:
            self.bg_avg = gray.astype(float)
            self._show(frame)
            return

        # ── Step 2: diff against background average ---------------
        diff = cv2.absdiff(gray, cv2.convertScaleAbs(self.bg_avg))
        _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Dilate fills small holes so one person isn't detected as 10 fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # ── Step 3: find contours = moved regions ---------------
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        annotated = frame.copy()

        for cnt in contours:
            if cv2.contourArea(cnt) < MOTION_MIN_AREA:
                continue  # too small — dust, shadow, compression noise
            motion_detected = True
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(annotated, "MOTION", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ── Step 4: update background ──────────────────────────
        # Faster update when motion — background adapts after someone stops
        alpha = 0.05 if motion_detected else 0.02
        cv2.accumulateWeighted(gray.astype(float), self.bg_avg, alpha)

        # ── Step 5: trigger caption if motion + cooldown ok ────
        now = time.time()
        cooldown_ok = (now - self.last_caption) >= COOLDOWN_SECONDS

        if motion_detected and cooldown_ok and not self.captioning:
            self.last_caption = now
            snap = "motion_snapshot.jpg"
            cv2.imwrite(snap, frame)
            self._trigger_caption(snap)
            self.status_label.setText("⚡ Motion! Generating caption...")
            self.status_label.setStyleSheet(
                "color: orange; font-size: 12px; padding: 4px;")
        elif not motion_detected:
            self.status_label.setText("Monitoring... waiting for motion")
            self.status_label.setStyleSheet(
                "color: green; font-size: 12px; padding: 4px;")

        self._show(annotated)

    def _show(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        self.preview_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(640, 420, Qt.KeepAspectRatio))

    def _trigger_caption(self, image_path):
        """Runs caption + TTS in background so feed stays live."""
        self.captioning = True

        def _worker():
            caption = self._generate_caption(image_path)
            self.caption_label.setText(f"Caption: {caption}")
            self.status_label.setText("Monitoring... waiting for motion")
            self.status_label.setStyleSheet(
                "color: green; font-size: 12px; padding: 4px;")
            self._speak(caption)
            self.captioning = False

        threading.Thread(target=_worker, daemon=True).start()

    def _generate_caption(self, image_path):
        try:
            sample_img = decode_and_resize(image_path)
            img = tf.expand_dims(sample_img, 0)
            img = caption_model.cnn_model(img)
            encoded_img = caption_model.encoder(img, training=False)

            decoded_caption = "<start> "
            for i in range(max_decoded_sentence_length):
                tokenized_caption = vectorization([decoded_caption])[:, :-1]
                mask = tf.math.not_equal(tokenized_caption, 0)
                predictions = caption_model.decoder(
                    tokenized_caption, encoded_img, training=False, mask=mask)
                sampled_token_index = np.argmax(predictions[0, i, :])
                sampled_token = index_lookup[sampled_token_index]
                if sampled_token == "<end>":
                    break
                decoded_caption += " " + sampled_token

            decoded_caption = decoded_caption.replace("<start> ", "")
            decoded_caption = decoded_caption.replace(" <end>", "").strip()
            return decoded_caption
        except Exception as e:
            return f"Error: {str(e)}"

    def _speak(self, text):
        try:
            tts = gTTS(text=text, lang="ne")
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.close()
            tts.save(tmp.name)
            if sys.platform == "darwin":
                os.system(f'afplay "{tmp.name}"')
            elif sys.platform == "win32":
                os.system(f'start /wait wmplayer "{tmp.name}"')
            else:
                if os.system(f'mpg123 -q "{tmp.name}"') != 0:
                    os.system(f'mpg321 -q "{tmp.name}"')
            os.unlink(tmp.name)
        except Exception as e:
            print(f"[TTS error] {e}")

    def manual_capture(self):
        ret, frame = self.cap.read()
        if ret:
            self.captured_image = "captured_image.jpg"
            cv2.imwrite(self.captured_image, frame)
            self.accept()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()


class ImageCaptionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Caption Generator")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("background-color: black;")

        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        self.title_label = QLabel("Image Caption Generator")
        self.title_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.title_label)

        self.image_frame = QFrame()
        self.image_frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.image_frame.setLineWidth(2)
        self.image_layout = QVBoxLayout(self.image_frame)
        self.image_label = QLabel(
            "No image loaded\n\nClick 'Choose Photo' or 'Capture Photo' to begin")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "background-color: white; padding: 10px; color: black;")
        self.image_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.image_frame, stretch=3)

        self.caption_frame = QFrame()
        self.caption_frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.caption_layout = QVBoxLayout(self.caption_frame)
        self.caption_label = QLabel("Caption will appear here")
        self.caption_label.setFont(QFont("Arial", 12))
        self.caption_label.setAlignment(Qt.AlignCenter)
        self.caption_label.setStyleSheet("padding: 10px;")
        self.caption_layout.addWidget(self.caption_label)
        self.main_layout.addWidget(self.caption_frame, stretch=1)

        self.button_frame = QFrame()
        self.button_layout = QHBoxLayout(self.button_frame)
        self.button_layout.setSpacing(15)

        self.capture_button = QPushButton("Capture Photo")
        self.capture_button.setFont(QFont("Arial", 10))
        self.capture_button.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 8px; border-radius: 4px;")
        self.capture_button.clicked.connect(self.capture_photo)
        self.button_layout.addWidget(self.capture_button)

        self.choose_button = QPushButton("Choose Photo")
        self.choose_button.setFont(QFont("Arial", 10))
        self.choose_button.setStyleSheet(
            "background-color: #2196F3; color: white; padding: 8px; border-radius: 4px;")
        self.choose_button.clicked.connect(self.choose_photo)
        self.button_layout.addWidget(self.choose_button)

        self.speak_button = QPushButton("Speak Caption")
        self.speak_button.setFont(QFont("Arial", 10))
        self.speak_button.setStyleSheet(
            "background-color: #FF9800; color: white; padding: 8px; border-radius: 4px;")
        self.speak_button.clicked.connect(self.speak_caption)
        self.button_layout.addWidget(self.speak_button)

        self.main_layout.addWidget(self.button_frame)

        self.current_image_path = None
        self.current_caption = ""

    def process_image(self, image_path):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(600, 400, Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
            caption = self.generate_caption(image_path)
            global speak
            speak = caption
            self.caption_label.setStyleSheet("color: white;")
            self.caption_label.setText(f"Predicted Caption: {caption}")
        else:
            self.image_label.setText("Failed to load image")
            self.caption_label.setText("Error: Could not load image")

    def choose_photo(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.current_image_path = file_name
            self.process_image(file_name)

    def capture_photo(self):
        choice_dialog = CameraChoiceDialog(self)
        if choice_dialog.exec_() == QDialog.Accepted:
            if choice_dialog.choice == "laptop":
                # source=0, same CaptureDialog as IP
                self.capture_dialog = CaptureDialog(self, is_ip_camera=False)
                if self.capture_dialog.exec_() == QDialog.Accepted:
                    if self.capture_dialog.captured_image:
                        self.current_image_path = self.capture_dialog.captured_image
                        self.process_image(self.current_image_path)

            elif choice_dialog.choice == "ip":
                ip_dialog = IPInputDialog(self)
                if ip_dialog.exec_() == QDialog.Accepted:
                    ip_address = ip_dialog.ip_input.text().strip()
                    # Accept full URL or just IP:port
                    url = ip_address if ip_address.startswith("http") \
                          else f"http://{ip_address}/video"
                    # Exact same CaptureDialog — same motion detection
                    self.capture_dialog = CaptureDialog(
                        self, is_ip_camera=True, ip_url=url)
                    if self.capture_dialog.exec_() == QDialog.Accepted:
                        if self.capture_dialog.captured_image:
                            self.current_image_path = self.capture_dialog.captured_image
                            self.process_image(self.current_image_path)

    def generate_caption(self, image_path):
        try:
            sample_img = decode_and_resize(image_path)
            img = tf.expand_dims(sample_img, 0)
            img = caption_model.cnn_model(img)
            encoded_img = caption_model.encoder(img, training=False)

            decoded_caption = "<start> "
            for i in range(max_decoded_sentence_length):
                tokenized_caption = vectorization([decoded_caption])[:, :-1]
                mask = tf.math.not_equal(tokenized_caption, 0)
                predictions = caption_model.decoder(
                    tokenized_caption, encoded_img, training=False, mask=mask)
                sampled_token_index = np.argmax(predictions[0, i, :])
                sampled_token = index_lookup[sampled_token_index]
                if sampled_token == "<end>":
                    break
                decoded_caption += " " + sampled_token

            decoded_caption = decoded_caption.replace("<start> ", "")
            decoded_caption = decoded_caption.replace(" <end>", "").strip()
            return decoded_caption
        except Exception as e:
            return f"Error generating caption: {str(e)}"

    def speak_caption(self):
        text = speak
        try:
            tts = gTTS(text=text, lang="ne")
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.close()
            tts.save(tmp.name)
            if sys.platform == "darwin":
                os.system(f'afplay "{tmp.name}"')
            elif sys.platform == "win32":
                os.system(f'start /wait wmplayer "{tmp.name}"')
            else:
                if os.system(f'mpg123 -q "{tmp.name}"') != 0:
                    os.system(f'mpg321 -q "{tmp.name}"')
            os.unlink(tmp.name)
        except Exception as e:
            print(f"[TTS error] {e}")

    def closeEvent(self, event):
        self.tts_engine.stop()
        for f in ["captured_image.jpg", "motion_snapshot.jpg"]:
            if os.path.exists(f):
                os.remove(f)
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = ImageCaptionWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
