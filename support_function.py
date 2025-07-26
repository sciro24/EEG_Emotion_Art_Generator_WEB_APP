import numpy as np
import random
from PIL import Image, ImageDraw
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import tensorflow as tf

# --- Definizione variabili linguistiche e universi ---

# Input: probabilità predette dal modello (range 0-1)
prob_negative = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'prob_negative')
prob_neutral  = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'prob_neutral')
prob_positive = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'prob_positive')

# Output: intensità fuzzy per ogni emozione (range 0-1)
intensita_negativa = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'intensita_negativa')
intensita_neutra   = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'intensita_neutra')
intensita_positiva = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'intensita_positiva')

# --- Funzioni di appartenenza per input ---
for var in [prob_negative, prob_neutral, prob_positive]:
    var['bassa'] = fuzz.trimf(var.universe, [0, 0, 0.5])
    var['media'] = fuzz.trimf(var.universe, [0.3, 0.5, 0.7])
    var['alta']  = fuzz.trimf(var.universe, [0.5, 1, 1])

# --- Funzioni di appartenenza per output (intensità) ---
for out_var in [intensita_negativa, intensita_neutra, intensita_positiva]:
    out_var['bassa'] = fuzz.trimf(out_var.universe, [0, 0, 0.5])
    out_var['media'] = fuzz.trimf(out_var.universe, [0.3, 0.5, 0.7])
    out_var['alta']  = fuzz.trimf(out_var.universe, [0.5, 1, 1])

# --- Regole fuzzy per intensità negativa ---
rule_neg_1 = ctrl.Rule(prob_negative['alta'], intensita_negativa['alta'])
rule_neg_2 = ctrl.Rule(prob_negative['media'], intensita_negativa['media'])
rule_neg_3 = ctrl.Rule(prob_negative['bassa'], intensita_negativa['bassa'])

# --- Regole fuzzy per intensità neutra ---
rule_neu_1 = ctrl.Rule(prob_neutral['alta'], intensita_neutra['alta'])
rule_neu_2 = ctrl.Rule(prob_neutral['media'], intensita_neutra['media'])
rule_neu_3 = ctrl.Rule(prob_neutral['bassa'], intensita_neutra['bassa'])

# --- Regole fuzzy per intensità positiva ---
rule_pos_1 = ctrl.Rule(prob_positive['alta'], intensita_positiva['alta'])
rule_pos_2 = ctrl.Rule(prob_positive['media'], intensita_positiva['media'])
rule_pos_3 = ctrl.Rule(prob_positive['bassa'], intensita_positiva['bassa'])

# --- Sistema di controllo e simulazione ---
emotion_ctrl_system = ctrl.ControlSystem([
    rule_neg_1, rule_neg_2, rule_neg_3,
    rule_neu_1, rule_neu_2, rule_neu_3,
    rule_pos_1, rule_pos_2, rule_pos_3,
])

emotion_simulation = ctrl.ControlSystemSimulation(emotion_ctrl_system)

def get_fuzzy_emotion_intensity(probabilities, emotion_classes):
    emotion_simulation = ctrl.ControlSystemSimulation(emotion_ctrl_system)  # ← Spostato qui

    # Mappa le probabilità ai rispettivi input fuzzy
    mapping = {'NEGATIVE': 'prob_negative', 'NEUTRAL': 'prob_neutral', 'POSITIVE': 'prob_positive'}
    for emo in emotion_classes:
        if emo in mapping:
            emotion_simulation.input[mapping[emo]] = probabilities[emotion_classes.index(emo)]

    try:
        emotion_simulation.compute()
        # Estrazione output fuzzy
        intensities = {
            'NEGATIVE': emotion_simulation.output['intensita_negativa'],
            'NEUTRAL': emotion_simulation.output['intensita_neutra'],
            'POSITIVE': emotion_simulation.output['intensita_positiva'],
        }
        # Scelta emozione dominante sulla base dell'intensità fuzzy
        dominant_emotion = max(intensities, key=intensities.get)
        dominant_intensity = intensities[dominant_emotion]
        return dominant_emotion, dominant_intensity, intensities
    except Exception as e:
        print(f"Errore calcolo fuzzy: {e}\nInput forniti: {emotion_simulation.input}")
        return None, None, None


def generate_abstract_art(emotion_label, fuzzy_intensities, width=512, height=512):
    # Prendo il valore fuzzy per l'emozione indicata, default 0.5 se non trovato
    if isinstance(fuzzy_intensities, dict):
        fuzzy_value = fuzzy_intensities.get(emotion_label, 0.5)
    else:
        fuzzy_value = fuzzy_intensities  # per retrocompatibilità
    
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    
    params = {
        'POSITIVE': {
            'color': (int(255*fuzzy_value), int(200+50*fuzzy_value), int(50+100*fuzzy_value)),
            'num_shapes': int(70+50*fuzzy_value),
            'size_range': (20,120),
            'line_width': int(2+2*fuzzy_value),
            'shape': 'polygon'
        },
        'NEUTRAL': {
            'color': (int(150+100*fuzzy_value), int(200+50*fuzzy_value), int(255*fuzzy_value)),
            'num_shapes': int(20+30*fuzzy_value),
            'size_range': (30,150),
            'line_width': int(1+3*fuzzy_value),
            'shape': 'circle'
        },
        'NEGATIVE': {
            'color': (int(150+100*(1-fuzzy_value)), int(50+100*(1-fuzzy_value)), int(50+100*(1-fuzzy_value))),
            'num_shapes': int(80+70*(1-fuzzy_value)),
            'size_range': (5,80),
            'line_width': int(3+5*(1-fuzzy_value)),
            'shape': 'line'
        }
    }
    
    p = params.get(emotion_label, {'color': (150,150,150), 'num_shapes':50, 'size_range':(10,100), 'line_width':1, 'shape':'random'})
    
    for _ in range(p['num_shapes']):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        size = random.randint(*p['size_range'])
        x2, y2 = x1 + size, y1 + size
        base_color = p['color']
        color = tuple(max(0, min(255, base_color[i] + random.randint(-50,50))) for i in range(3))
        
        shape_type = p['shape'] if p['shape'] != 'random' else random.choice(['circle', 'rectangle', 'line', 'polygon'])
        
        if shape_type == 'circle':
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=color, width=p['line_width'])
        elif shape_type == 'rectangle':
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=color, width=p['line_width'])
        elif shape_type == 'line':
            x3, y3 = random.randint(0, width), random.randint(0, height)
            draw.line([x1, y1, x3, y3], fill=color, width=p['line_width'])
        elif shape_type == 'polygon':
            points = [(random.randint(0, width), random.randint(0, height)) for _ in range(random.randint(3,6))]
            draw.polygon(points, fill=color, outline=color)
    
    return image

def compute_feature_importance_ffnn(model, input_data_np, class_idx):
    input_tensor = tf.convert_to_tensor(input_data_np, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        preds = model(input_tensor)
        class_output = preds[0, class_idx]
    gradients = tape.gradient(class_output, input_tensor)
    return np.abs(gradients.numpy())[0]

def preprocess_eeg_data(eeg_df, scaler):
    eeg_np = eeg_df.values
    if eeg_np.ndim == 1:
        eeg_np = eeg_np.reshape(1, -1)
    scaled_eeg = scaler.transform(eeg_np)
    return scaled_eeg


def generate_art_from_emotion(predicted_emotion_label):
    # Questa funzione deve mappare l'etichetta dell'emozione predetta
    # alle categorie usate da generate_abstract_art (POSITIVE, NEUTRAL, NEGATIVE)
    # e fornire un valore fuzzy (placeholder per ora).

    # Mappatura delle emozioni predette alle categorie di arte
    # Adatta questa logica in base alle tue etichette di emozione e come vuoi che l'arte sia generata.
    if predicted_emotion_label == 'felicità':
        emotion_category = 'POSITIVE'
        fuzzy_val = 0.8 # Valore fuzzy alto per felicità
    elif predicted_emotion_label == 'tristezza':
        emotion_category = 'NEGATIVE'
        fuzzy_val = 0.2 # Valore fuzzy basso per tristezza
    elif predicted_emotion_label == 'rabbia':
        emotion_category = 'NEGATIVE'
        fuzzy_val = 0.6 # Valore fuzzy medio-alto per rabbia (può essere più intenso)
    elif predicted_emotion_label == 'neutro':
        emotion_category = 'NEUTRAL'
        fuzzy_val = 0.5 # Valore fuzzy medio per neutro
    else:
        emotion_category = 'NEUTRAL' # Default
        fuzzy_val = 0.5

    # Chiama la funzione di generazione arte con la categoria e il valore fuzzy
    return generate_abstract_art(emotion_category, fuzzy_val)

def create_fuzzy_membership_plot(fuzzy_value):
    """Crea un grafico che mostra le curve di appartenenza fuzzy 
    e una linea verticale tratteggiata rossa per il valore fuzzy calcolato."""

    import numpy as np
    import skfuzzy as fuzz
    import matplotlib.pyplot as plt

    # Universo dell'output fuzzy (es. stato emotivo normalizzato)
    x = np.arange(0, 1.01, 0.01)

    # Funzioni di appartenenza
    basso = fuzz.trimf(x, [0, 0, 0.5])
    medio = fuzz.trimf(x, [0.2, 0.5, 0.8])
    alto = fuzz.trimf(x, [0.5, 1, 1])

    # Creazione grafico
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Disegna le curve di appartenenza
    ax.plot(x, basso, label='Basso', linewidth=2)
    ax.plot(x, medio, label='Medio', linewidth=2)
    ax.plot(x, alto, label='Alto', linewidth=2)

    # Linea verticale rossa tratteggiata che mostra il valore fuzzy calcolato
    line = ax.axvline(fuzzy_value, color='red', linestyle='--', linewidth=1.5)

    # Aggiungi riquadro giallo con il valore
    bbox_props = dict(boxstyle="round,pad=0.3", fc="yellow", ec="red", lw=1, alpha=0.8)
    ax.text(fuzzy_value + 0.02, 0.95, f'{fuzzy_value:.2f}', 
            ha="left", va="top", bbox=bbox_props, transform=ax.get_xaxis_transform())

    # Impostazioni grafiche
    ax.set_title('Appartenenza fuzzy dello stato emotivo', pad=15)
    ax.set_xlabel('Output normalizzato', labelpad=10)
    ax.set_ylabel('Grado di appartenenza', labelpad=10)
    
    # Legenda in alto a sinistra
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
    
    # Griglia
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Migliora il layout e i margini
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Salva il grafico
    plot_filename = f"fuzzy_output_{np.random.randint(1000, 9999)}.png"
    plot_path = f"uploads/{plot_filename}"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return plot_filename