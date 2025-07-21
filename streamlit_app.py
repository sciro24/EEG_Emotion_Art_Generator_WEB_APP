import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import io
import joblib
import matplotlib.pyplot as plt
import os

from support_function import *

# Configurazione pagina
st.set_page_config(
    page_title="Analisi EEG e Predizione Emozioni",
    page_icon="🧠",
    layout="wide"
)

# Titolo principale
st.title("🧠 Analisi EEG e Predizione Emozioni")
st.write("---")

# Sezione informativa
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("ℹ️ Informazioni sull'applicazione")
    st.markdown(
        """
        Questa applicazione analizza i dati EEG e predice lo stato emozionale utilizzando:
        - **Classificazione** delle emozioni tramite MLP  
        - **Logica Fuzzy** per affinare le predizioni  
        - **Feature Importance** per interpretabilità  
        - **Arte Astratta** generata sulla base delle emozioni rilevate  
        """
    )

with col2:
    st.subheader("📊 Come funziona")
    st.markdown(
        """
        1. Prepara un file `.csv` con i dati EEG  
        2. Caricalo nell'applicazione  
        3. Visualizza le emozioni predette e le analisi  
        """
    )

st.write("---")

# Sezione caricamento file
st.subheader("📁 Caricamento Dati EEG")
st.markdown("Trascina qui il file o clicca per selezionarlo dal tuo dispositivo.")

uploaded_file = st.file_uploader(
    label="**Upload file EEG (.csv)**",
    type=['csv'],
    label_visibility="collapsed",
    help="Carica un file CSV contenente un campione EEG"
)



# Session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

@st.cache_resource
def load_model_and_scaler():
    try:
        model_path = "emotion_classifier_model.keras"
        scaler_path = "scaler.pkl"
        if not os.path.exists(model_path):
            model_path = os.path.join("INTERFACCIA WEB", "emotion_classifier_model.keras")
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join("INTERFACCIA WEB", "scaler.pkl")

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler, "Modello e scaler caricati con successo!"
        else:
            class DummyModel(tf.keras.Model):
                def __init__(self):
                    super(DummyModel, self).__init__()
                    self.dense1 = tf.keras.layers.Dense(10, activation='relu')
                    self.dense2 = tf.keras.layers.Dense(3, activation='softmax')

                def call(self, inputs):
                    x = self.dense1(inputs)
                    return self.dense2(x)

            model = DummyModel()
            model.build(input_shape=(None, 100))
            scaler = MinMaxScaler()
            scaler.fit(np.random.rand(100, 100))
            return model, scaler, "⚠️ Utilizzo di modelli placeholder (file originali non trovati)"
    except Exception as e:
        st.error(f"Errore nel caricamento: {e}")
        return None, None, f"Errore: {e}"

def create_feature_importance_plot(feature_importance, feature_names, top_n=20):
    top_indices = np.argsort(feature_importance)[-top_n:][::-1]
    top_importance = feature_importance[top_indices]
    feature_labels = [feature_names[idx] if idx < len(feature_names) else f'Feature {idx}' for idx in top_indices]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_importance)), top_importance, color='steelblue')
    ax.set_xlabel('Importanza Feature (Gradiente Assoluto)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Top {top_n} Feature più Importanti per la Predizione', fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(top_importance)))
    ax.set_yticklabels(feature_labels)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig

def create_emotion_probability_plot(probabilities, classes):
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, probabilities, color='lightblue')
    bars[np.argmax(probabilities)].set_color('salmon')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom')
    ax.set_title('Probabilità Emozioni Predette dal Modello', pad=20)
    ax.set_xlabel('Emozioni', labelpad=10)
    ax.set_ylabel('Probabilità', labelpad=10)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def create_fuzzy_membership_plot_streamlit(fuzzy_value):
    import skfuzzy as fuzz
    x = np.arange(0, 1.01, 0.01)
    basso = fuzz.trimf(x, [0, 0, 0.5])
    medio = fuzz.trimf(x, [0.2, 0.5, 0.8])
    alto = fuzz.trimf(x, [0.5, 1, 1])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, basso, label='Basso', linewidth=2)
    ax.plot(x, medio, label='Medio', linewidth=2)
    ax.plot(x, alto, label='Alto', linewidth=2)
    ax.axvline(fuzzy_value, color='red', linestyle='--', linewidth=2, label=f'Valore: {fuzzy_value:.2f}')
    ax.set_title('Appartenenza Fuzzy dello Stato Emotivo', pad=15)
    ax.set_xlabel('Output Normalizzato', labelpad=10)
    ax.set_ylabel('Grado di Appartenenza', labelpad=10)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

def resize_generated_art(image, target_width=800, target_height=480):
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

# Caricamento modello e scaler
model, scaler, load_message = load_model_and_scaler()
if "⚠️" in load_message:
    st.warning(load_message)
elif model is not None and scaler is not None:
    st.success(load_message)
else:
    st.error(load_message)

if uploaded_file is not None and model is not None and scaler is not None:
    try:
        eeg_data = pd.read_csv(uploaded_file)
        st.success(f"File caricato con successo!")
        feature_names = list(eeg_data.columns)
        processed_eeg_data = preprocess_eeg_data(eeg_data, scaler)

        with st.spinner("🔄 Elaborazione in corso..."):
            predictions = model.predict(processed_eeg_data)
            model_classes = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
            probabilities = predictions[0]
            dominant_emotion, dominant_intensity, fuzzy_intensities = get_fuzzy_emotion_intensity(probabilities, model_classes)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("### 🎯 Emozioni Predette")
            st.metric("Emozione Dominante", dominant_emotion)
            prob_df = pd.DataFrame({
                'Emozione': model_classes,
                'Probabilità': probabilities
            }).sort_values('Probabilità', ascending=False)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            st.pyplot(create_emotion_probability_plot(probabilities, model_classes))

        with col2:
            st.markdown("### 🌊 Analisi Fuzzy")
            if dominant_intensity is not None:
                st.metric("Intensità Emozione", f"{dominant_intensity:.2f}")
                fuzzy_df = pd.DataFrame({
                    'Emozione': list(fuzzy_intensities.keys()),
                    'Intensità': list(fuzzy_intensities.values())
                }).sort_values('Intensità', ascending=False)
                st.dataframe(fuzzy_df, use_container_width=True, hide_index=True)
                st.pyplot(create_fuzzy_membership_plot_streamlit(dominant_intensity))
            else:
                st.error("Errore nel calcolo dell'intensità fuzzy")

        col3, col4 = st.columns(2, gap="large")

        with col3:
            st.markdown("### 📊 Feature Importance")
            try:
                predicted_class_index = model_classes.index(dominant_emotion)
                feature_importance = compute_feature_importance_ffnn(model, processed_eeg_data[0], predicted_class_index)
                most_important_idx = np.argmax(feature_importance)
                most_important_feature = feature_names[most_important_idx] if len(feature_names) > most_important_idx else f'Feature {most_important_idx}'
                st.metric("Feature più Rilevante", most_important_feature)
                fig = create_feature_importance_plot(feature_importance, feature_names)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Errore nel calcolo della feature importance: {e}")

        with col4:
            st.markdown("### 🎨 Arte Generativa")
            try:
                generated_image = generate_abstract_art(dominant_emotion, fuzzy_intensities)
                resized_image = resize_generated_art(generated_image, target_width=800, target_height=480)
                st.markdown(f"**Arte generata per:** {dominant_emotion}")

                # Forza altezza minima al container immagine con CSS inline (480px)
                st.markdown(
                    """
                    <style>
                    div[data-testid="stImage"] > img {
                        height: 480px !important;
                        object-fit: contain;
                        width: 100% !important;
                    }
                    </style>
                    """, unsafe_allow_html=True
                )
                
                st.image(resized_image, use_container_width=True)
                img_buffer = io.BytesIO()
                generated_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
            except Exception as e:
                st.error(f"Errore nella generazione dell'arte: {e}")



    except Exception as e:
        st.error(f"Errore durante l'elaborazione del file: {e}")

st.markdown("---")
st.markdown("Scirocco Diego")
