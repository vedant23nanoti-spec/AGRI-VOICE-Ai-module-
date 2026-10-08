import os
import requests
import ollama
import streamlit as st

from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader
import chromadb


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Agri Voice",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_NAME = "Agri Voice"

LLM_MODEL = "qwen2.5:1.5b"
WHISPER_MODEL = "base"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

PLANT_DISEASE_MODEL = (
    "A2H0H0R1/mobilenet_v2_1.0_224-plant-disease"
)

LATITUDE = 20.90
LONGITUDE = 79.00


# ============================================================
# LANGUAGE CONFIGURATION
# ============================================================

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada"
}

TTS_LANGUAGES = {
    "en": "en",
    "hi": "hi",
    "mr": "mr",
    "te": "te",
    "ta": "ta",
    "kn": "kn"
}


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    whisper_model = WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
        cpu_threads=2,
        num_workers=1
    )

    disease_model = pipeline(
        "image-classification",
        model=PLANT_DISEASE_MODEL
    )

    return (
        embedding_model,
        whisper_model,
        disease_model
    )


# ============================================================
# LOAD CHROMADB
# ============================================================

@st.cache_resource
def load_database():

    chroma_client = chromadb.PersistentClient(
        path="db"
    )

    collection = chroma_client.get_or_create_collection(
        name="agriculture_knowledge"
    )

    return collection


# ============================================================
# INITIALIZE
# ============================================================

with st.spinner("Loading Agri Voice AI models..."):

    (
        embedding_model,
        whisper_model,
        disease_model
    ) = load_models()

    collection = load_database()


# ============================================================
# AGRICULTURAL SEARCH
# ============================================================

def search_agriculture(question, top_k=4):

    query_embedding = embedding_model.encode(
        ["query: " + question],
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    return documents


# ============================================================
# WEATHER
# ============================================================

def get_weather():

    try:

        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "precipitation"
                ),
                "timezone": "auto"
            },
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        current = data["current"]

        return {
            "temperature": current["temperature_2m"],
            "humidity": current["relative_humidity_2m"],
            "precipitation": current["precipitation"]
        }

    except Exception as error:

        return {
            "error": str(error)
        }


# ============================================================
# AGRICULTURE QUESTION → QWEN
# ============================================================

def generate_agriculture_answer(
    question,
    language="en"
):

    documents = search_agriculture(
        question,
        top_k=4
    )

    context = "\n\n".join(documents)

    language_name = LANGUAGE_NAMES.get(
        language,
        "English"
    )

    prompt = f"""
You are Agri Voice, an agriculture assistant
for Indian farmers.

Answer the farmer's question using only the
agricultural context provided below.

Detected language:
{language_name}

IMPORTANT:
- Answer in {language_name}.
- Use simple farmer-friendly language.
- Do not invent agricultural facts.
- Do not invent pesticide names.
- Do not provide pesticide dosages.
- Give safe general agricultural advice.
- If information is insufficient, say so.
- Recommend consulting a local KVK or
  agriculture expert when required.

AGRICULTURAL CONTEXT:

{context}

FARMER QUESTION:

{question}

ANSWER:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ============================================================
# LEAF DISEASE DETECTION
# ============================================================

def detect_leaf_disease(
    image,
    top_k=3
):

    predictions = disease_model(
        image,
        top_k=top_k
    )

    best_prediction = predictions[0]

    disease = best_prediction["label"]

    confidence = best_prediction["score"]

    return (
        disease,
        confidence,
        predictions
    )


# ============================================================
# DISEASE → AGRICULTURAL ADVICE
# ============================================================

def generate_disease_advice(
    disease,
    confidence
):

    documents = search_agriculture(
        disease,
        top_k=4
    )

    context = "\n\n".join(documents)

    prompt = f"""
You are Agri Voice, an agriculture assistant
for Indian farmers.

A plant disease image classifier detected:

Disease:
{disease}

Confidence:
{confidence:.2%}

Use the agricultural knowledge below.

AGRICULTURAL KNOWLEDGE:

{context}

Instructions:

- Explain the detected condition.
- Mention symptoms only if supported.
- Give safe general prevention and management.
- Do not invent pesticide names.
- Do not provide pesticide dosage.
- Do not claim the image diagnosis is 100% certain.
- Recommend KVK/agriculture expert confirmation
  when appropriate.
- Keep the answer concise and practical.

FARMER ADVICE:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ============================================================
# ROUTER
# ============================================================

WEATHER_KEYWORDS = [
    "weather",
    "rain",
    "rainfall",
    "temperature",
    "humidity",
    "forecast",
    "climate",
    "बारिश",
    "वर्षा",
    "मौसम",
    "तापमान",
    "पाऊस",
    "हवामान"
]


def route_query(question):

    question_lower = question.lower()

    for keyword in WEATHER_KEYWORDS:

        if keyword in question_lower:
            return "weather"

    return "agriculture"


# ============================================================
# STREAMLIT HEADER
# ============================================================

st.title("🌾 Agri Voice")

st.subheader(
    "AI-Powered Voice Assistant for Farmers"
)

st.write(
    "Ask agricultural questions, check weather, "
    "or analyze a crop leaf."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌾 Agri Voice")

st.sidebar.write(
    "AI agriculture assistant"
)

option = st.sidebar.radio(
    "Select Service",
    [
        "🎤 Ask Agriculture Question",
        "🌦️ Weather",
        "🌿 Leaf Disease Detection"
    ]
)


# ============================================================
# AGRICULTURE QUESTION
# ============================================================

if option == "🎤 Ask Agriculture Question":

    st.header("🎤 Ask an Agriculture Question")

    question = st.text_area(
        "Enter your question",
        placeholder=(
            "Example: How can I control "
            "pink bollworm in cotton?"
        )
    )

    language = st.selectbox(
        "Answer Language",
        list(LANGUAGE_NAMES.keys()),
        format_func=lambda x: LANGUAGE_NAMES[x]
    )

    if st.button(
        "🤖 Ask Agri Voice"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            intent = route_query(
                question
            )

            if intent == "weather":

                weather = get_weather()

                if "error" in weather:

                    st.error(
                        weather["error"]
                    )

                else:

                    st.success(
                        "🌦️ Weather information"
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Temperature",
                        f"{weather['temperature']} °C"
                    )

                    col2.metric(
                        "Humidity",
                        f"{weather['humidity']} %"
                    )

                    col3.metric(
                        "Rain",
                        f"{weather['precipitation']} mm"
                    )

            else:

                with st.spinner(
                    "Searching agricultural knowledge..."
                ):

                    answer = generate_agriculture_answer(
                        question,
                        language
                    )

                st.success(
                    "🌾 Agri Voice Response"
                )

                st.write(answer)


# ============================================================
# WEATHER
# ============================================================

elif option == "🌦️ Weather":

    st.header("🌦️ Current Weather")

    if st.button(
        "🔄 Get Weather"
    ):

        weather = get_weather()

        if "error" in weather:

            st.error(
                weather["error"]
            )

        else:

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "🌡️ Temperature",
                f"{weather['temperature']} °C"
            )

            col2.metric(
                "💧 Humidity",
                f"{weather['humidity']} %"
            )

            col3.metric(
                "🌧️ Precipitation",
                f"{weather['precipitation']} mm"
            )


# ============================================================
# LEAF DISEASE
# ============================================================

elif option == "🌿 Leaf Disease Detection":

    st.header("🌿 Crop Leaf Disease Detection")

    uploaded_file = st.file_uploader(
        "Upload a crop leaf image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded Leaf",
            use_container_width=True
        )

        if st.button(
            "🔍 Analyze Leaf"
        ):

            with st.spinner(
                "Analyzing leaf image..."
            ):

                disease, confidence, predictions = (
                    detect_leaf_disease(
                        uploaded_file
                    )
                )

            st.subheader(
                "🌿 Detection Result"
            )

            st.write(
                "**Detected condition:**",
                disease
            )

            st.write(
                "**Confidence:**",
                f"{confidence:.2%}"
            )

            st.subheader(
                "🔎 Top Predictions"
            )

            for i, prediction in enumerate(
                predictions,
                1
            ):

                st.write(
                    f"{i}. "
                    f"{prediction['label']} "
                    f"→ "
                    f"{prediction['score']:.2%}"
                )

            if confidence < 0.60:

                st.warning(
                    "The model confidence is low. "
                    "Please consult a local KVK or "
                    "agriculture expert for confirmation."
                )

            else:

                with st.spinner(
                    "Generating agricultural advice..."
                ):

                    advice = generate_disease_advice(
                        disease,
                        confidence
                    )

                st.subheader(
                    "🌾 Agricultural Advice"
                )

                st.write(advice)


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Agri Voice | AI + RAG + Computer Vision"
)