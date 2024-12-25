import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Dictionnaire de correspondance entre noms affichés, fichiers, et tailles d'entrée pour les modèles Keras
model_mapping = {
    "Modèle personnel": {"file": "dog_breed_cnn_3.h5", "input_size": (180, 180)},
    "MobileNet": {"file": "dog_breed_mobilenet_3.keras", "input_size": (160, 160)},
    "VGG16_P6": {"file": "dog_breed_vgg16_p6.h5", "input_size": (224, 224)},
    "VGG16_P7": {"file": "dog_breed_vgg16_p7.h5", "input_size": (224, 224)},
    "ResNet": {"file": "dog_breed_resnet_50.h5", "input_size": (160, 160)},
}

# Charger le modèle YOLO
model_yolo = YOLO("/Users/randaalsabbagh/Desktop/MACHINE_LEARNING/P6/modelyolo.pt")

# Initialiser st.session_state si nécessaire
if "model" not in st.session_state:
    st.session_state.model = None

if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = None

# Ajouter un sélecteur d'onglets
tab1, tab2 = st.tabs(["Classification", "Dashboard"])

with tab1:
    # Sélection du modèle
    selected_model_name = st.sidebar.selectbox(
        "Choisissez un modèle pour la prédiction :",
        list(model_mapping.keys()) + ["YOLO11"]
    )

    # Bouton pour charger le modèle
    if st.sidebar.button("Charger le modèle"):
        try:
            if selected_model_name == "YOLO11":
                st.session_state.model = model_yolo
                st.session_state.selected_model_name = selected_model_name
                st.sidebar.success(f"Modèle '{selected_model_name}' chargé avec succès.")
            else:
                # Charger le modèle Keras sélectionné
                selected_file = model_mapping[selected_model_name]["file"]
                st.session_state.model = tf.keras.models.load_model(selected_file)
                st.session_state.selected_model_name = selected_model_name
                st.sidebar.success(f"Modèle '{selected_model_name}' chargé avec succès.")
        except Exception as e:
            st.sidebar.error(f"Erreur lors du chargement du modèle : {e}")

    # Classes de prédiction pour les modèles Keras
    CLASS_NAMES = ['n02085936-Maltese_dog', 'n02088094-Afghan_hound', 'n02092002-Scottish_deerhound']

    def predict_race(image, model, input_size):
        image = image.resize(input_size)
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        predictions = tf.nn.softmax(predictions, axis=-1)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions)
        return predicted_class, confidence

    def predict_yolo(image, confidence_threshold=0.3):
        results = model_yolo(image)
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
        else:
            st.error("Aucun résultat valide retourné par le modèle.")
            return "Aucun résultat valide", 0.0

        if hasattr(result, "probs") and result.probs is not None:
            probs = result.probs
            class_names = result.names
            predicted_class_index = probs.top1
            predicted_class_confidence = probs.top1conf
            predicted_class = class_names[predicted_class_index]

            if predicted_class_confidence < confidence_threshold:
                st.error("Aucune classe détectée avec une confiance suffisante.")
                return "Aucune classe détectée", float(predicted_class_confidence)

            return predicted_class, float(predicted_class_confidence)
        else:
            st.error("Les probabilités de classe ne sont pas disponibles.")
            return "Aucune classe détectée", 0.0

    # Titre de l'application
    st.title("Classification de races de chiens")

    # Uploader une image
    uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image téléchargée", use_container_width=True)

        if st.session_state.selected_model_name:
            if st.session_state.selected_model_name == "YOLO11":
                predicted_class, confidence = predict_yolo(image)
                st.subheader("Résultat de la prédiction YOLO")
                st.write(f"Objet détecté : **{predicted_class}**")
                st.write(f"Confiance : **{confidence * 100:.2f}%**")
            else:
                input_size = model_mapping[st.session_state.selected_model_name]["input_size"]
                predicted_class, confidence = predict_race(image, st.session_state.model, input_size)
                st.subheader("Résultat de la prédiction")
                st.write(f"Race prédite : **{predicted_class}**")
                st.write(f"Confiance : **{confidence * 100:.2f}%**")
        else:
            st.warning("Veuillez d'abord charger un modèle.")

    st.text("Développé avec ❤️ par Randa AL SABBAGH")

with tab2:
    # Onglet pour le dashboard
    st.subheader("Dashboard")
    st.write("Voici les graphiques et images.")
    
    # Afficher vos images prédéfinies
    image_paths = ["/Users/randaalsabbagh/runs/classify/train109/confusion_matrix_normalized.png", 
                   "/Users/randaalsabbagh/runs/classify/train109/results.png"]  # Chemins des images
    for img_path in image_paths:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            st.image(img, caption=f"Image: {img_path}", use_container_width=True)
            #graph à ajouter avec Dash 

