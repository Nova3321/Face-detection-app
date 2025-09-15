import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import datetime
import os

# Charger Haar cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

save_dir = "saved_faces"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

st.title("Face Detection with Viola-Jones in Streamlit")

# Paramètres ajustables
scale_factor = st.slider("Choisissez le paramètre scaleFactor", 1.01, 2.0, 1.3, 0.01)
min_neighbors = st.slider("Choisissez le paramètre minNeighbors", 1, 10, 5)
rect_color_hex = st.color_picker("Choisissez la couleur des rectangles", "#00FF00")
save_images = st.checkbox("Sauvegarder les visages détectés")

# Convertir hex → BGR
hex_color = rect_color_hex.lstrip('#')
rect_color_bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

# Transformer pour appliquer la détection en temps réel
class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
        )

        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(img, (x, y), (x+w, y+h), rect_color_bgr, 2)

            if save_images:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_dir, f"face_{timestamp}_{i}.jpg")
                cv2.imwrite(filename, img[y:y+h, x:x+w])

        return img

# Lancer le flux webcam dans le navigateur
webrtc_streamer(key="face-detection", video_transformer_factory=FaceDetectionTransformer)
