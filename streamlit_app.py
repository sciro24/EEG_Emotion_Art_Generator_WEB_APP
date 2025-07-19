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
import tempfile
import shutil

# Importa le funzioni di supporto
from support_function import *

# Configurazione della pagina
st.set_page_config(
    page_title="Analisi EEG e Predizione Emozioni",
    page_icon="🧠",
    layout="wide"
)

# Titolo principale
st.title("🧠 Analisi EEG e Predizione Emozioni")
st.markdown("Carica un file CSV con dati EEG per ottenere predizioni emozionali e visualizzazioni")

# Sidebar per informazioni
st.sidebar.header("ℹ️ Informazioni")
st.sidebar.markdown("""
Questa applicazione analizza i dati EEG e predice le emozioni utilizzando:
- **Modello di Machine Learning** per la classificazione
- **Logica Fuzzy** per l'intensità emotiva
- **Visualizzazioni** dei risultati
- **Arte Generativa** basata sulle emozioni
""")

# Inizializzazione del session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

@st.cache_resource
def load_model_and_scaler():
    """Carica il modello e lo scaler con caching"""
    try:
        # Cerca i file nella directory corrente
        model_path = "emotion_classifier_model.keras"
        scaler_path = "scaler.pkl"
        
        # Se non trovati, cerca nella directory INTERFACCIA WEB
        if not os.path.exists(model_path):
            model_path = os.path.join("INTERFACCIA WEB", "emotion_classifier_model.keras")
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join("INTERFACCIA WEB", "scaler.pkl")
            
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler, "Modello e scaler caricati con successo!"
        else:
            # Crea modelli dummy se i file non esistono
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
    """Crea un grafico delle feature più importanti"""
    top_indices = np.argsort(feature_importance)[-top_n:][::-1]
    top_importance = feature_importance[top_indices]
    
    if feature_names is not None and len(feature_names) > max(top_indices):
        feature_labels = [feature_names[idx] for idx in top_indices]
    else:
        feature_labels = [f'Feature {idx}' for idx in top_indices]
    
    fig, ax = plt.subplots(figsize=(12, 8))
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
    """Crea un grafico a barre delle probabilità delle emozioni"""
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(classes, probabilities, color='lightblue')

    # Evidenzia la barra con probabilità più alta
    bars[np.argmax(probabilities)].set_color('salmon')

    # Aggiungi i valori sulle barre
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')

    ax.set_title('Probabilità Emozioni Predette dal Modello', pad=20)
    ax.set_xlabel('Emozioni', labelpad=10)
    ax.set_ylabel('Probabilità', labelpad=10)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig

def create_fuzzy_membership_plot_streamlit(fuzzy_value):
    """Crea un grafico che mostra le curve di appartenenza fuzzy"""
    import skfuzzy as fuzz
    
    # Universo dell'output fuzzy
    x = np.arange(0, 1.01, 0.01)

    # Funzioni di appartenenza
    basso = fuzz.trimf(x, [0, 0, 0.5])
    medio = fuzz.trimf(x, [0.2, 0.5, 0.8])
    alto = fuzz.trimf(x, [0.5, 1, 1])

    # Creazione grafico
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Disegna le curve di appartenenza
    ax.plot(x, basso, label='Basso', linewidth=2)
    ax.plot(x, medio, label='Medio', linewidth=2)
    ax.plot(x, alto, label='Alto', linewidth=2)

    # Linea verticale rossa tratteggiata che mostra il valore fuzzy calcolato
    ax.axvline(fuzzy_value, color='red', linestyle='--', linewidth=2, label=f'Valore: {fuzzy_value:.2f}')

    # Impostazioni grafiche
    ax.set_title('Appartenenza Fuzzy dello Stato Emotivo', pad=15)
    ax.set_xlabel('Output Normalizzato', labelpad=10)
    ax.set_ylabel('Grado di Appartenenza', labelpad=10)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    return fig

# Carica modello e scaler
model, scaler, load_message = load_model_and_scaler()

# Mostra il messaggio di caricamento
if "⚠️" in load_message:
    st.warning(load_message)
else:
    st.success(load_message)

# Upload del file
st.header("📁 Carica File EEG")
uploaded_file = st.file_uploader(
    "Seleziona un file CSV con dati EEG",
    type=['csv'],
    help="Il file deve contenere dati EEG in formato CSV"
)

if uploaded_file is not None and model is not None and scaler is not None:
    try:
        # Leggi il file CSV
        eeg_data = pd.read_csv(uploaded_file)
        
        st.success(f"File caricato con successo! Dimensioni: {eeg_data.shape}")
        
        # Mostra anteprima dei dati
        with st.expander("👀 Anteprima Dati EEG"):
            st.dataframe(eeg_data.head())
            st.write(f"**Colonne:** {list(eeg_data.columns)}")
        
        # Processa i dati
        feature_names = list(eeg_data.columns)
        processed_eeg_data = preprocess_eeg_data(eeg_data, scaler)
        
        # Predizione del modello
        with st.spinner("🔄 Elaborazione in corso..."):
            predictions = model.predict(processed_eeg_data)
            
            # Classi del modello
            model_classes = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
            
            # Ottieni le probabilità per ogni classe
            probabilities = predictions[0]
            
            # Calcola le intensità fuzzy
            dominant_emotion, dominant_intensity, fuzzy_intensities = get_fuzzy_emotion_intensity(
                probabilities, model_classes
            )
        
        # Layout a colonne per i risultati
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🎯 Risultati Predizione")
            
            # Risultati del modello
            st.subheader("Modello di Machine Learning")
            st.metric("Emozione Dominante", dominant_emotion)
            
            # Mostra le probabilità
            prob_df = pd.DataFrame({
                'Emozione': model_classes,
                'Probabilità': probabilities
            })
            st.dataframe(prob_df, use_container_width=True)
            
            # Grafico delle probabilità
            prob_fig = create_emotion_probability_plot(probabilities, model_classes)
            st.pyplot(prob_fig)
            
        with col2:
            st.header("🌊 Analisi Fuzzy")
            
            if dominant_intensity is not None:
                st.subheader("Intensità Fuzzy")
                st.metric("Intensità Dominante", f"{dominant_intensity:.3f}")
                
                # Mostra le intensità fuzzy
                fuzzy_df = pd.DataFrame({
                    'Emozione': list(fuzzy_intensities.keys()),
                    'Intensità': list(fuzzy_intensities.values())
                })
                st.dataframe(fuzzy_df, use_container_width=True)
                
                # Grafico fuzzy
                fuzzy_fig = create_fuzzy_membership_plot_streamlit(dominant_intensity)
                st.pyplot(fuzzy_fig)
            else:
                st.error("Errore nel calcolo dell'intensità fuzzy")
        
        # Feature Importance
        st.header("📊 Importanza delle Feature")
        try:
            predicted_class_index = model_classes.index(dominant_emotion)
            feature_importance = compute_feature_importance_ffnn(
                model, processed_eeg_data[0], predicted_class_index
            )
            
            most_important_idx = np.argmax(feature_importance)
            most_important_feature = feature_names[most_important_idx] if len(feature_names) > most_important_idx else f'Feature {most_important_idx}'
            
            st.metric("Feature più Importante", most_important_feature)
            
            # Grafico feature importance
            importance_fig = create_feature_importance_plot(feature_importance, feature_names)
            st.pyplot(importance_fig)
            
        except Exception as e:
            st.error(f"Errore nel calcolo della feature importance: {e}")
        
        # Arte Generativa
        st.header("🎨 Arte Generativa")
        try:
            generated_image = generate_abstract_art(dominant_emotion, fuzzy_intensities)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(generated_image, caption=f"Arte basata su: {dominant_emotion}", use_container_width=True)
                
                # Pulsante per scaricare l'immagine
                img_buffer = io.BytesIO()
                generated_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label="💾 Scarica Arte Generata",
                    data=img_buffer.getvalue(),
                    file_name=f"arte_emotiva_{dominant_emotion.lower()}.png",
                    mime="image/png"
                )
                
        except Exception as e:
            st.error(f"Errore nella generazione dell'arte: {e}")
            
    except Exception as e:
        st.error(f"Errore durante l'elaborazione del file: {e}")

# Footer
st.markdown("---")
st.markdown("🧠 **Analisi EEG e Predizione Emozioni** - Powered by Streamlit")

