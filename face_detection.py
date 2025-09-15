import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import datetime
import os

# Charger le classificateur Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if face_cascade.empty():
    st.error("Erreur : le fichier Haar cascade n'a pas été chargé ! Vérifie le chemin.")

# Créer un dossier pour sauvegarder les images si nécessaire
save_dir = "saved_faces"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

st.title("👀 Face Detection with Viola-Jones (Streamlit + WebRTC)")
st.markdown("""
**Instructions :**  
1. Autorisez l'accès à votre webcam.  
2. Ajustez les paramètres dans la barre latérale.  
3. Les visages seront détectés en temps réel.  
4. Activez l'option pour sauvegarder les visages détectés.  
""")

# Paramètres ajustables
st.sidebar.header("⚙️ Paramètres de détection")
scale_factor = st.sidebar.slider("Paramètre scaleFactor", 1.01, 2.0, 1.3, 0.01)
min_neighbors = st.sidebar.slider("Paramètre minNeighbors", 1, 10, 5)
rect_color_hex = st.sidebar.color_picker("Couleur des rectangles", "#00FF00")
save_images = st.sidebar.checkbox("💾 Sauvegarder les visages détectés")

# Conversion couleur HEX → BGR
hex_color = rect_color_hex.lstrip('#')
rect_color_bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

# Compteur global (stocké dans la session Streamlit)
if "total_faces_detected" not in st.session_state:
    st.session_state.total_faces_detected = 0

# Définir le transformateur vidéo
class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_counter = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Améliorer le contraste
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Détection des visages
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(50, 50)
        )

        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(img, (x, y), (x + w, y + h), rect_color_bgr, 2)

            if save_images:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_dir, f"face_{timestamp}_{i}.jpg")
                cv2.imwrite(filename, img[y:y+h, x:x+w])

        # ➕ Mise à jour du compteur global
        st.session_state.total_faces_detected += len(faces)

        return img

# Lancer le flux WebRTC
webrtc_streamer(
    key="face-detection",
    video_transformer_factory=FaceDetectionTransformer
)

# Afficher le compteur total
st.markdown(f"### 👤 Nombre total de visages détectés pendant la session : **{st.session_state.total_faces_detected}**")
