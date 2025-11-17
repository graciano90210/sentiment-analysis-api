import json
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI
from pydantic import BaseModel
import nltk
from nltk.corpus import stopwords

# --- 1. Inicializar la Aplicación FastAPI ---
app = FastAPI(
    title="API de Análisis de Sentimiento",
    description="Un proyecto MLOps con Deep Learning y GitHub Student Pack.",
    version="1.0"
)

# --- 2. Cargar los Modelos al Iniciar ---
# (Esto solo se ejecuta una vez, cuando la API arranca)

print("Cargando stopwords de NLTK...")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

print("Cargando el modelo de Keras...")
model_path = "src/model/sentiment_model.keras"  # Ruta DESDE LA RAÍZ
model = load_model(model_path)

print("Cargando el tokenizer...")
tokenizer_path = "src/model/tokenizer.json"  # Ruta DESDE LA RAÍZ
with open(tokenizer_path, 'r', encoding='utf-8') as f:
    tokenizer_config = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_config)

print("¡Carga inicial completa! API lista.")

# Definimos los parámetros que usamos en el entrenamiento
MAX_SEQ_LENGTH = 50
LABELS = ['negative', 'neutral', 'positive'] # Basado en el LabelEncoder (0, 1, 2)


# --- 3. Función de Limpieza de Texto ---
# (La misma que usamos en el notebook)
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'@\w+', '', texto)
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    texto = ' '.join(palabra for palabra in texto.split() if palabra not in stop_words)
    return texto


# --- 4. Definir el Modelo de Entrada (Pydantic) ---
# Esto le dice a FastAPI cómo debe ser el JSON que recibimos
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    text: str


# --- 5. Definir el Endpoint de Predicción ---
@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    
    # 1. Limpiar el texto de entrada
    texto_limpio = limpiar_texto(request.text)
    
    # 2. Convertir a secuencia de números
    secuencia = tokenizer.texts_to_sequences([texto_limpio])
    
    # 3. Rellenar la secuencia (Padding)
    secuencia_pad = pad_sequences(secuencia, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
    
    # 4. Hacer la predicción
    prediccion_probs = model.predict(secuencia_pad)[0]
    
    # 5. Obtener la clase con mayor probabilidad
    clase_predicha = np.argmax(prediccion_probs)
    
    # 6. Obtener la etiqueta (nombre) y la confianza
    sentimiento = LABELS[clase_predicha]
    confianza = float(np.max(prediccion_probs))
    
    # 7. Devolver la respuesta
    return SentimentResponse(
        sentiment=sentimiento,
        confidence=confianza,
        text=request.text
    )

# --- 6. Endpoint Raíz (para saber que la API vive) ---
@app.get("/")
def read_root():
    return {"message": "¡Bienvenido a la API de Análisis de Sentimiento!"}