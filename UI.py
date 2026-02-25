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
from preprocessing import *
from Transformer import *
from loadingweight import model as caption_model

# Global variables
vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1
valid_images = list(valid_data.keys())
speak = 'कृपया फोटो हाल्नु होस् '

class IPInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter IP Address")
        self.setFixedSize(300, 100)
        
        layout = QVBoxLayout()
        
        self.label = QLabel("Enter phone's IP address (e.g., 192.168.1.75:8080):")
        layout.addWidget(self.label)
        
        self.ip_input = QLineEdit(self)
        self.ip_input.setText("192.168.1.75:8080")
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
    def __init__(self, parent=None, is_ip_camera=False, ip_url=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Preview")
        self.setGeometry(200, 200, 640, 480)
        self.is_ip_camera = is_ip_camera
        
        self.layout = QVBoxLayout()
        self.preview_label = QLabel()
        self.layout.addWidget(self.preview_label)
        
        self.capture_button = QPushButton("Capture")
        self.capture_button.setStyleSheet("background-color: black; color: white;")
        self.capture_button.clicked.connect(self.capture)
        self.layout.addWidget(self.capture_button)
        
        self.setLayout(self.layout)
        
        if is_ip_camera and ip_url:
            self.cap = cv2.VideoCapture(ip_url)
        else:
            self.cap = cv2.VideoCapture(0)
            
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open camera")
            self.close()
            return
            
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        
        self.captured_image = None
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.preview_label.setPixmap(QPixmap.fromImage(image).scaled(
                600, 400, Qt.KeepAspectRatio))
            
    def capture(self):
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

        # Main layout setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        # Title
        self.title_label = QLabel("Image Caption Generator")
        self.title_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.title_label)

        # Image frame
        self.image_frame = QFrame()
        self.image_frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.image_frame.setLineWidth(2)
        self.image_layout = QVBoxLayout(self.image_frame)
        self.image_label = QLabel("No image loaded\n\nClick 'Choose Photo' or 'Capture Photo' to begin")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: white; padding: 10px; color: black;")
        self.image_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.image_frame, stretch=3)

        # Caption frame
        self.caption_frame = QFrame()
        self.caption_frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.caption_layout = QVBoxLayout(self.caption_frame)
        self.caption_label = QLabel("Caption will appear here")
        self.caption_label.setFont(QFont("Arial", 12))
        self.caption_label.setAlignment(Qt.AlignCenter)
        self.caption_label.setStyleSheet("padding: 10px;")
        self.caption_layout.addWidget(self.caption_label)
        self.main_layout.addWidget(self.caption_frame, stretch=1)

        # Button frame
        self.button_frame = QFrame()
        self.button_layout = QHBoxLayout(self.button_frame)
        self.button_layout.setSpacing(15)

        self.capture_button = QPushButton("Capture Photo")
        self.capture_button.setFont(QFont("Arial", 10))
        self.capture_button.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 8px; border-radius: 4px;"
        )
        self.capture_button.clicked.connect(self.capture_photo)
        self.button_layout.addWidget(self.capture_button)

        self.choose_button = QPushButton("Choose Photo")
        self.choose_button.setFont(QFont("Arial", 10))
        self.choose_button.setStyleSheet(
            "background-color: #2196F3; color: white; padding: 8px; border-radius: 4px;"
        )
        self.choose_button.clicked.connect(self.choose_photo)
        self.button_layout.addWidget(self.choose_button)

        self.speak_button = QPushButton("Speak Caption")
        self.speak_button.setFont(QFont("Arial", 10))
        self.speak_button.setStyleSheet(
            "background-color: #FF9800; color: white; padding: 8px; border-radius: 4px;"
        )
        self.speak_button.clicked.connect(self.speak_caption)
        self.speak_button.setEnabled(True)
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
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_name:
            self.current_image_path = file_name
            self.process_image(file_name)

    def capture_photo(self):
        choice_dialog = CameraChoiceDialog(self)
        if choice_dialog.exec_() == QDialog.Accepted:
            if choice_dialog.choice == "laptop":
                self.capture_dialog = CaptureDialog(self, is_ip_camera=False)
                if self.capture_dialog.exec_() == QDialog.Accepted:
                    if self.capture_dialog.captured_image:
                        self.current_image_path = self.capture_dialog.captured_image
                        self.process_image(self.current_image_path)
            elif choice_dialog.choice == "ip":
                ip_dialog = IPInputDialog(self)
                if ip_dialog.exec_() == QDialog.Accepted:
                    ip_address = ip_dialog.ip_input.text()
                    url = f"http://{ip_address}/video"
                    self.capture_dialog = CaptureDialog(self, is_ip_camera=True, ip_url=url)
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
                    tokenized_caption, encoded_img, training=False, mask=mask
                )
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
        tts = gTTS(text=text, lang="ne")
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            tts.save(temp_file.name)
            os.system(f"mpg321 {temp_file.name}")

    def closeEvent(self, event):
        self.tts_engine.stop()
        if os.path.exists("captured_image.jpg"):
            os.remove("captured_image.jpg")
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = ImageCaptionWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()