import cv2
import streamlit as st
import os
import datetime

# Charger le classificateur Haar depuis le package OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    st.error("Erreur : le fichier Haar cascade n'a pas été chargé ! Vérifie le chemin.")

# Créer un dossier pour sauvegarder les images si nécessaire
save_dir = "saved_faces"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

st.title("Face Detection with Viola-Jones in Streamlit")
st.markdown("""
**Instructions :**  
1. Cliquez sur **Start Webcam** pour lancer la détection.  
2. Ajustez les paramètres `scaleFactor` et `minNeighbors` pour améliorer la détection.  
3. Choisissez la couleur des rectangles autour des visages.  
4. Cochez la case pour sauvegarder les visages détectés.  
5. Cliquez sur **Stop Webcam** pour arrêter la détection.
""")

# Paramètres ajustables par l'utilisateur
scale_factor = st.slider("Choisissez le paramètre scaleFactor", 1.01, 2.0, 1.3, 0.01)
min_neighbors = st.slider("Choisissez le paramètre minNeighbors", 1, 10, 5)
rect_color_hex = st.color_picker("Choisissez la couleur des rectangles", "#00FF00")
save_images = st.checkbox("Sauvegarder les visages détectés")

# Convertir la couleur hex en BGR pour OpenCV
hex_color = rect_color_hex.lstrip('#')
rect_color_bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

# Zone pour afficher le flux vidéo
frame_placeholder = st.empty()
# Zone pour afficher le nombre de visages
face_count_placeholder = st.empty()

# Boutons de contrôle
start = st.button("Start Webcam")
stop = st.button("Stop Webcam")

# Si l'utilisateur clique sur Start
if start:
    cap = cv2.VideoCapture(0)
    st.write("Webcam démarrée.")

    # Utiliser une boucle tant que Stop n'est pas cliqué
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Impossible d'accéder à la webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        # Compter les visages détectés
        face_count = len(faces)
        face_count_placeholder.markdown(f"**Nombre de visages détectés : {face_count}**")

        for i, (x, y, w, h) in enumerate(faces):
            # Dessiner le rectangle autour du visage
            cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color_bgr, 2)

            # Sauvegarder le visage si demandé
            if save_images:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_dir, f"face_{timestamp}_{i}.jpg")
                cv2.imwrite(filename, frame[y:y+h, x:x+w])

        # Convertir BGR en RGB pour Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Vérifier si Stop a été cliqué
        if stop:
            st.write("Webcam arrêtée.")
            break

    cap.release()
