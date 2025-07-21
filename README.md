# 🧠 EEG Emotion Art Generator 🎨

[🔗 Vai all'applicazione](https://eeg-emotion-art-generator.streamlit.app/)

## Descrizione

**EEG Emotion Art Generator** è un'app interattiva sviluppata con Streamlit che classifica i dati EEG in **stati emotivi** e li rappresenta sotto forma di **arte astratta generata dinamicamente**.  

---

## 🚀 Funzionalità principali

- 🎯 **Predizione delle emozioni** da segnali EEG tramite un modello MLP
- 📊 **Analisi fuzzy** per stimare l’intensità dell’emozione
- 🔍 **Feature importance** per interpretare le decisioni del modello
- 🎨 **Generazione artistica** basata sull’emozione dominante

---

## 🧪 Come funziona

1. Prepara un file `.csv` contenente un campione di dati EEG (una riga = un campione)
2. Caricalo nell'applicazione tramite drag-and-drop
3. Visualizza:
   - L'emozione dominante (NEGATIVE, NEUTRAL, POSITIVE)
   - L’intensità dell’emozione (basso, medio, alto)
   - Le feature EEG più rilevanti
   - Un'immagine astratta generata in base all'emozione

---

## 📎 Provalo online

👉 **[Accedi all'applicazione](https://eeg-emotion-art-generator.streamlit.app/)**

---

## 🛠️ Tecnologie utilizzate

- Python + Streamlit
- TensorFlow (rete neurale MLP)
- Scikit-learn
- Logica fuzzy (`scikit-fuzzy`)
- PIL + Matplotlib per la visualizzazione
- Joblib per la serializzazione del modello/scaler

---

## 👨‍💻 Autore

Realizzato da **Scirocco Diego**  

---
