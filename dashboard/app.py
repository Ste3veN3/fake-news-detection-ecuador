"""
Dashboard: Sistema de Deteccion de Noticias Falsas
Proyecto de Titulacion, Maestria en Ciencia de Datos, USFQ
Autor: Steeven Quezada | Tutor: Alejandro Proano

Como correrlo (desde la raiz del repo, con el venv activado):
    python -m streamlit run dashboard/app.py
"""

import glob
from pathlib import Path

import numpy as np
import streamlit as st

# ============================================================
# CONFIGURACION DE PAGINA
# ============================================================
st.set_page_config(
    page_title="Detector de Noticias Falsas | Ecuador",
    page_icon="📰",
    layout="wide",
)

# Este archivo vive en dashboard/app.py, la raiz del repo es un nivel arriba.
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
SRC_DIR = BASE_DIR / "src"

# Orden real de los topicos del LDA (notebook 09, indice 0 = Topico 1).
NOMBRES_TOPICOS = {
    0: "Política Internacional (Noboa, Trump, México)",
    1: "Economía y Finanzas",
    2: "Elecciones 2023 (Noboa vs González)",
    3: "Metodología (TikTok, fuentes, verificación)",
    4: "Consulta Popular",
    5: "Caso Correa-Glas (Fiscalía, Justicia)",
    6: "Salud y Ciencia",
    7: "Proceso Electoral (Lasso, Asamblea)",
}

# Stopwords de dominio usadas al entrenar el LDA (notebook 09).
STOP_CUSTOM_LDA = {
    "ecuador", "chequea", "verificación", "publicación",
    "información", "según", "afirma", "dice", "dijo",
    "señala", "indica", "asegura", "contenido", "dato",
    "imagen", "video", "post", "facebook", "twitter",
    "redes", "sociales", "viral", "circula", "usuarios",
    "también", "además", "sino", "aunque", "embargo",
}


# ============================================================
# CARGA DE MODELOS (cada uno se carga UNA sola vez por sesion)
# ============================================================

@st.cache_resource(show_spinner="Cargando modelo SVM baseline...")
def load_svm_model():
    """LinearSVC entrenado en el notebook 03."""
    import pickle
    path = MODELS_DIR / "baseline_svm_best.pkl"
    if not path.exists():
        st.error(f"No se encontró {path}.")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Cargando vectorizador TF-IDF...")
def load_tfidf_vectorizer():
    """
    Vectorizador TF-IDF del SVM. Debe ser el MISMO con el que se entreno el
    SVM (enfoque Titulo+Texto, notebook 03), no cualquier tfidf*.pkl.
    """
    import pickle
    path = MODELS_DIR / "tfidf_title_text_clean.pkl"
    if not path.exists():
        st.error(f"No se encontró {path}.")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Cargando BERT (inglés)...")
def load_bert_en():
    """BERT fine-tuned en ingles (notebook 04)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = MODELS_DIR / "bert_finetuned"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Cargando BERT multilingüe (Ecuador)...")
def load_bert_es():
    """BERT multilingue fine-tuned sobre el corpus ecuatoriano (notebook 08)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = MODELS_DIR / "bert_ecuador"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Cargando modelo LDA de tópicos...")
def load_lda_model():
    """LDA + CountVectorizer guardados desde el notebook 09 (None si no existen)."""
    import joblib
    lda_path = MODELS_DIR / "lda_model.pkl"
    vec_path = MODELS_DIR / "lda_vectorizer.pkl"
    if not lda_path.exists() or not vec_path.exists():
        return None, None
    return joblib.load(lda_path), joblib.load(vec_path)


@st.cache_resource(show_spinner=False)
def load_stopwords_es():
    """Stopwords en espanol (NLTK) + stopwords de dominio del notebook 09."""
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("spanish")).union(STOP_CUSTOM_LDA)


# ============================================================
# UTILIDADES
# ============================================================

def detectar_idioma(texto: str) -> str:
    """Devuelve 'es' si el texto esta en espanol, 'en' en cualquier otro caso."""
    from langdetect import detect
    try:
        return "es" if detect(texto) == "es" else "en"
    except Exception:
        return "en"


def predecir_bert(texto: str, idioma: str):
    """Clasifica un texto con el BERT correspondiente al idioma detectado."""
    import torch
    tokenizer, model = load_bert_es() if idioma == "es" else load_bert_en()
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
    return int(np.argmax(probs)), probs


def preprocesar_para_svm(texto: str) -> str:
    """Mismo pipeline de limpieza del notebook 02 (src/preprocessing.py)."""
    import sys
    sys.path.append(str(SRC_DIR))
    from preprocessing import preprocess
    return preprocess(texto, lemmatize=True)


def limpiar_texto_es(texto: str) -> str:
    """Misma limpieza del notebook 09, para que el LDA vea el texto como en el entrenamiento."""
    import re
    stop_words = load_stopwords_es()
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return " ".join(w for w in texto.split() if w not in stop_words and len(w) > 3)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("📰 Detector de Noticias Falsas")
st.sidebar.markdown("**Proyecto de Titulación | USFQ**")
st.sidebar.markdown("Maestría en Ciencia de Datos")
st.sidebar.markdown("---")
st.sidebar.markdown("**Autor:** Steeven Quezada")
st.sidebar.markdown("**Tutor:** Alejandro Proaño")

pagina = st.sidebar.radio(
    "Navegación",
    ["🔍 Clasificador", "🧠 Explicabilidad (XAI)", "📊 Tópicos (LDA)", "📈 Estadísticas del Corpus"],
)

# ============================================================
# PAGINA 1: CLASIFICADOR
# ============================================================
if pagina == "🔍 Clasificador":
    st.title("Clasificador de Noticias")
    st.markdown(
        "Pega el titular y/o cuerpo de una noticia. El sistema detecta el "
        "idioma automáticamente y usa el modelo BERT correspondiente."
    )

    texto = st.text_area("Texto de la noticia", height=200, placeholder="Pega aquí el titular + contenido...")

    if st.button("Clasificar", type="primary"):
        if not texto.strip():
            st.warning("Ingresa un texto primero.")
        else:
            idioma = detectar_idioma(texto)
            with st.spinner(f"Clasificando (idioma detectado: {'español' if idioma == 'es' else 'inglés'})..."):
                pred, probs = predecir_bert(texto, idioma)

            etiqueta = "🟢 REAL" if pred == 1 else "🔴 FALSA"
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicción", etiqueta)
            col2.metric("Confianza", f"{probs[pred] * 100:.1f}%")
            col3.metric("Modelo usado", "BERT-ES (Ecuador)" if idioma == "es" else "BERT-EN")

# ============================================================
# PAGINA 2: EXPLICABILIDAD (XAI)
# ============================================================
elif pagina == "🧠 Explicabilidad (XAI)":
    st.title("Explicabilidad del Modelo (SHAP / LIME)")
    st.markdown(
        "Estas explicaciones usan el modelo **SVM + TF-IDF** (notebook 06), "
        "no BERT. Es sobre el que se corrió el análisis XAI original. "
        "Funciona con texto en **inglés**, igual que el dataset de entrenamiento."
    )

    tab_global, tab_local = st.tabs(["Vista Global (SHAP)", "Explicación Individual (LIME)"])

    with tab_global:
        st.markdown("**Importancia global de features**, calculada sobre 200 muestras del test set.")
        shap_img = REPORTS_DIR / "11_shap_global.png"
        if shap_img.exists():
            st.image(str(shap_img), caption="Top 20 features más importantes (SHAP)")
        else:
            st.info(f"No se encontró {shap_img}.")
        st.warning(
            "⚠️ Hallazgo clave: la palabra 'reuters' tuvo ~3x más peso que cualquier otra "
            "feature. El modelo aprendió la fuente, no el contenido."
        )

    with tab_local:
        st.markdown("Escribe una noticia en inglés y LIME muestra qué palabras influyeron en la predicción del SVM.")
        texto_lime = st.text_area("Texto para explicar", height=150, key="lime_input")

        if st.button("Explicar con LIME"):
            if not texto_lime.strip():
                st.warning("Ingresa un texto primero.")
            else:
                from lime.lime_text import LimeTextExplainer
                from scipy.special import expit

                svm = load_svm_model()
                vectorizer = load_tfidf_vectorizer()

                def predict_proba_svm(textos):
                    limpios = [preprocesar_para_svm(t) for t in textos]
                    X = vectorizer.transform(limpios)
                    scores = svm.decision_function(X)
                    p1 = expit(scores)  # pseudo-probabilidad de clase "Real"
                    return np.column_stack([1 - p1, p1])

                explainer = LimeTextExplainer(class_names=["Falsa", "Real"])
                with st.spinner("Calculando explicación LIME..."):
                    exp = explainer.explain_instance(texto_lime, predict_proba_svm, num_features=10)
                st.components.v1.html(exp.as_html(), height=400, scrolling=True)

# ============================================================
# PAGINA 3: TOPICOS (LDA)
# ============================================================
elif pagina == "📊 Tópicos (LDA)":
    st.title("Análisis de Tópicos: Corpus Ecuatoriano (LDA)")

    st.markdown("**8 tópicos identificados en 1,543 verificaciones de Ecuador Chequea:**")
    for i, nombre in NOMBRES_TOPICOS.items():
        st.markdown(f"- **Tópico {i + 1}:** {nombre}")

    lda_img = REPORTS_DIR / "16_lda_topicos_ecuador.png"
    if lda_img.exists():
        st.image(str(lda_img))

    st.markdown("---")
    lda, vec = load_lda_model()
    if lda is None:
        st.info(
            "⚠️ No se encontraron lda_model.pkl y lda_vectorizer.pkl en models/. "
            "Corre la celda de joblib.dump al final del notebook 09."
        )
    else:
        st.markdown("Clasifica un texto nuevo en uno de los 8 tópicos:")
        texto_lda = st.text_area("Texto (español)", height=150, key="lda_input")
        if st.button("Detectar tópico"):
            if not texto_lda.strip():
                st.warning("Ingresa un texto primero.")
            else:
                texto_limpio = limpiar_texto_es(texto_lda)
                if not texto_limpio:
                    st.warning("El texto quedó vacío tras la limpieza. Prueba con un texto más largo.")
                else:
                    dist = lda.transform(vec.transform([texto_limpio]))[0]
                    topico = int(np.argmax(dist))
                    st.success(
                        f"Tópico dominante: **{NOMBRES_TOPICOS.get(topico, topico)}** "
                        f"({dist[topico] * 100:.1f}% de probabilidad)"
                    )

# ============================================================
# PAGINA 4: ESTADISTICAS DEL CORPUS
# ============================================================
else:
    st.title("Estadísticas del Corpus")

    tab1, tab2 = st.tabs(["Fake & Real News (Kaggle)", "Ecuador Chequea"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de registros", "38,473")
        col2.metric("Reducción tras preprocesamiento", "43.4%")
        col3.metric("Mejor modelo baseline", "SVM (99.45%)")

        for img_name in ["01_distribucion_basica.png", "02_wordcloud.png", "04_eda_completo.png"]:
            img_path = REPORTS_DIR / img_name
            if img_path.exists():
                st.image(str(img_path))

    with tab2:
        col1, col2, col3 = st.columns(3)
        col1.metric("Verificaciones", "1,543")
        col2.metric("Período cubierto", "2016-2026")
        col3.metric("Mejora con fine-tuning", "+18.29 pts")

        for img_name in ["13_corpus_ecuatoriano.png", "17_lda_distribucion_topicos.png"]:
            img_path = REPORTS_DIR / img_name
            if img_path.exists():
                st.image(str(img_path))