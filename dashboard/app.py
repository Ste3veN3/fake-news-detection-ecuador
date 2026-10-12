"""
Verificador de noticias: dashboard del proyecto de titulacion.
Maestria en Ciencia de Datos, USFQ. Autor: Steeven Quezada. Tutor: Alejandro Proano.

Como correrlo (desde la raiz del repo, con el venv activado):
    python -m streamlit run dashboard/app.py
"""

from pathlib import Path

import numpy as np
import streamlit as st

import ui

# ============================================================
# CONFIGURACION
# ============================================================
st.set_page_config(
    page_title="Verificador de noticias",
    page_icon="📰",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
SRC_DIR = BASE_DIR / "src"
ASSETS = Path(__file__).resolve().parent / "assets"
# Logo de la universidad: se prefiere el SVG (no se pixela) y, si no hay, el PNG.
LOGO = next((p for p in (ASSETS / "usfq_logo.svg", ASSETS / "usfq_logo.png") if p.exists()), None)

# Marca de respaldo para la parte superior de la barra lateral (si no hay logo en assets/).
MARCA_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 230 48" width="230" height="48">'
    '<path d="M24 4 8 10v12c0 10 7 18 16 22 9-4 16-12 16-22V10L24 4z" fill="#1F4FBF"/>'
    '<path d="M16 24l6 6 11-12" fill="none" stroke="#fff" stroke-width="4" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
    '<text x="54" y="22" font-family="Arial, sans-serif" font-size="17" font-weight="700" '
    'fill="#14213D">Verificador</text>'
    '<text x="54" y="40" font-family="Arial, sans-serif" font-size="13" '
    'fill="#5B6578">de noticias</text></svg>'
)

# Etiquetas de cada modelo: (clase 0, clase 1).
# El corpus de Ecuador Chequea se etiqueto como Desinformacion (0) / Verificado (1);
# Fake & Real News como Falsa (0) / Real (1).
ETIQUETAS = {
    "es": ("Desinformación", "Verificado"),
    "en": ("Falsa", "Real"),
}

# Orden real de los topicos del LDA (notebook 09, indice 0 = Topico 1).
NOMBRES_TOPICOS = {
    0: "Política internacional (Noboa, Trump, México)",
    1: "Economía y finanzas",
    2: "Elecciones 2023 (Noboa vs González)",
    3: "Metodología (TikTok, fuentes, verificación)",
    4: "Consulta popular",
    5: "Caso Correa-Glas (Fiscalía, Justicia)",
    6: "Salud y ciencia",
    7: "Proceso electoral (Lasso, Asamblea)",
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

# Precision de LIME: numero de variantes del texto que se evaluan.
PRECISIONES = {
    "Rápida (1,500 variantes)": 1500,
    "Completa (5,000 variantes)": 5000,
}

# Textos de ejemplo para demostraciones en vivo.
EJEMPLO_ES = (
    "Circula en TikTok que el Consejo Nacional Electoral anuló las elecciones "
    "por un supuesto fraude masivo en todo el país."
)
EJEMPLO_EN = (
    "You won't believe what this politician did behind closed doors! Sources say "
    "the entire government is covering up the truth about this shocking scandal!"
)
REUTERS_CON = (
    "WASHINGTON (Reuters) - The U.S. Federal Reserve held interest rates steady on "
    "Wednesday, saying that inflation remains above its long-term target even as the "
    "labor market shows signs of cooling. Fed Chair said in a statement that "
    "policymakers would continue to monitor economic data closely before deciding on "
    "any future changes to borrowing costs. The decision was widely expected by "
    "analysts, who had predicted that the central bank would wait for further "
    "evidence that price pressures are easing. Stocks on Wall Street were little "
    "changed after the announcement, while Treasury yields edged lower. Economists "
    "polled by Reuters said they expect the central bank to begin cutting rates later "
    "this year if inflation continues to decline. The Fed has raised rates sharply "
    "over the past two years to bring down prices, a campaign that has increased the "
    "cost of mortgages and business loans. Officials said they remain committed to "
    "returning inflation to two percent, and that the path forward will depend on "
    "incoming data, the outlook for growth and the balance of risks facing the economy."
)
REUTERS_SIN = REUTERS_CON.replace("WASHINGTON (Reuters) - ", "").replace(
    "polled by Reuters", "polled by analysts"
)


# ============================================================
# CARGA DE MODELOS (cada uno se carga una sola vez por sesion)
# ============================================================

@st.cache_resource(show_spinner="Cargando el modelo SVM...")
def load_svm_model():
    import pickle
    path = MODELS_DIR / "baseline_svm_best.pkl"
    if not path.exists():
        st.error(f"No se encontró {path}. Copia el modelo a la carpeta models/.")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Cargando el vectorizador TF-IDF...")
def load_tfidf_vectorizer():
    """Debe ser el mismo con el que se entrenó el SVM (título + texto, notebook 03)."""
    import pickle
    path = MODELS_DIR / "tfidf_title_text_clean.pkl"
    if not path.exists():
        st.error(f"No se encontró {path}. Copia el vectorizador a la carpeta models/.")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Cargando BERT en inglés...")
def load_bert_en():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = MODELS_DIR / "bert_finetuned"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Cargando BERT multilingüe (Ecuador)...")
def load_bert_es():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = MODELS_DIR / "bert_ecuador"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Cargando el modelo de temas...")
def load_lda_model():
    """LDA y CountVectorizer del notebook 09 (None si no están en models/)."""
    import joblib
    lda_path = MODELS_DIR / "lda_model.pkl"
    vec_path = MODELS_DIR / "lda_vectorizer.pkl"
    if not lda_path.exists() or not vec_path.exists():
        return None, None
    return joblib.load(lda_path), joblib.load(vec_path)


@st.cache_resource(show_spinner=False)
def load_stopwords_es():
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("spanish")).union(STOP_CUSTOM_LDA)


# ============================================================
# LOGICA
# ============================================================

def detectar_idioma(texto: str) -> str:
    from langdetect import detect
    try:
        return "es" if detect(texto) == "es" else "en"
    except Exception:
        return "en"


def probs_bert(textos, idioma: str, lote: int = 16):
    """Probabilidades [clase 0, clase 1] del BERT del idioma indicado, para una lista de textos."""
    import torch
    tokenizer, model = load_bert_es() if idioma == "es" else load_bert_en()
    salida = []
    for i in range(0, len(textos), lote):
        enc = tokenizer(
            list(textos[i:i + lote]), return_tensors="pt",
            truncation=True, max_length=256, padding=True,
        )
        with torch.no_grad():
            salida.append(torch.softmax(model(**enc).logits, dim=1).numpy())
    return np.vstack(salida)


@st.cache_data(show_spinner=False)
def explicar_svm(texto: str, num_samples: int = 5000):
    """
    LIME sobre el SVM. El resultado queda guardado por texto: repetir el mismo texto es
    instantaneo. random_state fijo para que el ranking sea reproducible.
    Devuelve (probabilidad de Real, {palabra: peso}); peso positivo = empuja hacia Real.
    """
    from lime.lime_text import LimeTextExplainer
    p_real = float(prob_real_svm([texto])[0][1])
    explainer = LimeTextExplainer(class_names=list(ETIQUETAS["en"]), random_state=42)
    exp = explainer.explain_instance(
        texto, prob_real_svm, num_features=15, num_samples=num_samples
    )
    return p_real, dict(exp.as_list())


def preprocesar_para_svm(texto: str) -> str:
    import sys
    sys.path.append(str(SRC_DIR))
    from preprocessing import preprocess
    return preprocess(texto, lemmatize=True)


def prob_real_svm(textos):
    """Puntaje del SVM convertido a probabilidad aproximada con una sigmoide."""
    from scipy.special import expit
    svm = load_svm_model()
    vectorizer = load_tfidf_vectorizer()
    limpios = [preprocesar_para_svm(t) for t in textos]
    scores = svm.decision_function(vectorizer.transform(limpios))
    p1 = expit(scores)
    return np.column_stack([1 - p1, p1])


def limpiar_texto_es(texto: str) -> str:
    """Misma limpieza del notebook 09, para que el LDA vea el texto como en el entrenamiento."""
    import re
    stop_words = load_stopwords_es()
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return " ".join(w for w in texto.split() if w not in stop_words and len(w) > 3)


def panel_temas(texto: str) -> None:
    """Para textos en español: los tres temas del corpus ecuatoriano más cercanos al texto."""
    lda, vec = load_lda_model()
    md(ui.card_h3("De qué trata"))
    if lda is None:
        md(ui.card_note("Falta el modelo de temas (lda_model.pkl) en la carpeta models/."))
        return
    limpio = limpiar_texto_es(texto)
    if not limpio:
        md(ui.card_note("El texto es muy corto para identificar un tema."))
        return
    dist = lda.transform(vec.transform([limpio]))[0]
    orden = np.argsort(dist)[::-1][:3]
    md(ui.card_note("Los tres temas del corpus ecuatoriano más cercanos a este texto."))
    md(ui.topic_bars(
        [(NOMBRES_TOPICOS.get(int(i), f"Tema {int(i) + 1}"), float(dist[i])) for i in orden]
    ))


def _cargar(clave: str, valor: str) -> None:
    st.session_state[clave] = valor


def md(html_str: str) -> None:
    st.markdown(html_str, unsafe_allow_html=True)


# ============================================================
# PAGINAS
# ============================================================

def page_verificar():
    with st.container(key="card_entrada"):
        md(ui.card_title(
            "Verifica una noticia",
            "Pega el titular y el cuerpo. El sistema detecta si está en español o en inglés "
            "y aplica el modelo correspondiente.",
        ))
        st.text_area(
            "Texto de la noticia", key="texto_verificar", height=150,
            placeholder="Pega aquí el titular y el contenido de la noticia que deseas analizar...",
        )
        with st.container(horizontal=True):
            verificar = st.button("Verificar", type="primary", icon=":material/search:", key="btn_verificar")
            st.button("🇪🇸 Ejemplo en español", key="ej_es", on_click=_cargar, args=("texto_verificar", EJEMPLO_ES))
            st.button("🇬🇧 Ejemplo en inglés", key="ej_en", on_click=_cargar, args=("texto_verificar", EJEMPLO_EN))

    if verificar:
        texto = st.session_state.get("texto_verificar", "")
        if not texto.strip():
            st.session_state.pop("res_ver", None)
            st.warning("Escribe o pega un texto antes de verificar.")
        else:
            idioma = detectar_idioma(texto)
            with st.spinner("Analizando el texto..."):
                p_pos = float(probs_bert([texto], idioma)[0][1])
            st.session_state["res_ver"] = {
                "texto": texto, "idioma": idioma, "p_pos": p_pos,
            }

    res = st.session_state.get("res_ver")
    if not res:
        with st.container(key="card_resultado"):
            md(ui.card_title(
                "Resultado del análisis",
                "Pega un texto y pulsa Verificar: el resultado y las palabras que más pesaron "
                "aparecerán aquí.",
            ))
        return

    neg, pos = ETIQUETAS[res["idioma"]]
    nombre_idioma = "español" if res["idioma"] == "es" else "inglés"
    modelo = (
        "BERT multilingüe con fine-tuning y pesos por clase sobre el corpus de Ecuador Chequea"
        if res["idioma"] == "es" else "BERT base con fine-tuning sobre Fake & Real News"
    )
    with st.container(key="card_resultado"):
        md(ui.card_title("Resultado del análisis"))
        izq, der = st.columns([1, 1.2], gap="large")
        with izq:
            md(ui.result_panel(
                neg, pos, res["p_pos"],
                f"Idioma detectado: {nombre_idioma}. Modelo: {modelo}. "
                "Es una estimación estadística, no una verificación de hechos.",
            ))
        with der:
            if res["idioma"] == "es":
                panel_temas(res["texto"])
            else:
                md(ui.card_h3("Qué palabras pesaron"))
                md(ui.card_note(
                    "El análisis palabra por palabra se hace con el SVM, el modelo del análisis "
                    "XAI, y puede diferir del veredicto de BERT."
                ))
                if st.button("Explicar este texto con el SVM", key="btn_ir_explicar", icon=":material/lightbulb:"):
                    st.session_state["texto_lime"] = res["texto"]
                    st.switch_page(PAG_EXPLICAR)


def page_explicar():
    with st.container(key="card_entrada_exp"):
        md(ui.card_title(
            "Mira en qué se fijó el SVM",
            "Resalta las palabras que empujaron la decisión del SVM con TF-IDF, el modelo del "
            "análisis XAI original. Funciona con texto en inglés, igual que el dataset de entrenamiento.",
        ))
        st.text_area(
            "Texto en inglés", key="texto_lime", height=150,
            placeholder="Pega aquí un texto en inglés...",
        )
        with st.container(horizontal=True):
            explicar = st.button("Explicar", type="primary", icon=":material/lightbulb:", key="btn_explicar")
            st.button("Ejemplo con la fuente", key="ej_con", on_click=_cargar, args=("texto_lime", REUTERS_CON))
            st.button("Mismo texto sin la fuente", key="ej_sin", on_click=_cargar, args=("texto_lime", REUTERS_SIN))
        precision = st.radio(
            "Precisión de la explicación",
            list(PRECISIONES),
            index=1,
            horizontal=True,
            key="lime_precision",
            help="Más variantes dan un ranking de palabras más estable, pero tardan más. "
                 "Cada texto se calcula una sola vez por precisión; repetirlo es instantáneo.",
        )

    if explicar:
        texto = st.session_state.get("texto_lime", "")
        if not texto.strip():
            st.session_state.pop("res_exp", None)
            st.warning("Escribe o pega un texto en inglés antes de explicar.")
        else:
            n_variantes = PRECISIONES[precision]
            with st.spinner("Calculando la explicación. La primera vez tarda; después queda guardada."):
                p_real, pesos = explicar_svm(texto, n_variantes)
            st.session_state["res_exp"] = {
                "texto": texto, "p_pos": p_real, "pesos": pesos, "n": n_variantes,
            }

    res = st.session_state.get("res_exp")
    if not res:
        with st.container(key="card_resultado_exp"):
            md(ui.card_title(
                "Resultado del análisis",
                "Pega un texto en inglés y pulsa Explicar, o prueba los dos ejemplos para ver "
                "cómo cambia la decisión al quitar la fuente.",
            ))
    else:
        neg, pos = ETIQUETAS["en"]
        with st.container(key="card_resultado_exp"):
            md(ui.card_title("Resultado del análisis"))
            izq, der = st.columns([1, 1.2], gap="large")
            with izq:
                md(ui.result_panel(
                    neg, pos, res["p_pos"],
                    "Modelo: SVM con TF-IDF. El SVM entrega un puntaje que se convierte en "
                    "probabilidad aproximada con una sigmoide: sirve para comparar textos, "
                    f"no como probabilidad calibrada. Explicación con LIME, {res.get('n', 5000):,} variantes.",
                    alerta=False,
                ))
            with der:
                md(ui.card_h3("Palabras que pesaron en la decisión"))
                md(ui.annotate(res["texto"], res["pesos"], neg, pos))
                md(ui.card_h3("Las más influyentes", espacio=True))
                md(ui.top_words(res["pesos"], n=8))

    with st.container(key="card_general"):
        md(ui.card_title(
            "Qué aprendió el modelo en general",
            "En el análisis global con SHAP (200 muestras del conjunto de prueba), la palabra "
            "«reuters» pesó unas tres veces más que cualquier otra. Prueba los dos botones de "
            "ejemplo: con esos textos, quitar la fuente baja la probabilidad de Real de 91% a "
            "74%, pero el texto sigue clasificándose como Real. El modelo aprendió la fuente y "
            "el estilo de redacción de agencia, no si la información es cierta.",
        ))
        shap_img = REPORTS_DIR / "11_shap_global.png"
        if shap_img.exists():
            st.image(str(shap_img), caption="Las 20 variables más importantes según SHAP")
        else:
            st.info(f"No se encontró {shap_img.name} en la carpeta reports/.")


def page_temas():
    lda, vec = load_lda_model()
    with st.container(key="card_temas"):
        md(ui.card_title(
            "Temas de las verificaciones en Ecuador",
            "El modelo LDA encontró ocho temas en 1,543 verificaciones de Ecuador Chequea.",
        ))
        if lda is None:
            st.info(
                "Faltan lda_model.pkl y lda_vectorizer.pkl en la carpeta models/. "
                "Genéralos corriendo la última celda del notebook 09."
            )
        else:
            palabras = vec.get_feature_names_out()
            temas = []
            for i, comp in enumerate(lda.components_):
                top = [palabras[j] for j in comp.argsort()[:-7:-1]]
                temas.append((f"{i + 1}. {NOMBRES_TOPICOS.get(i, f'Tema {i + 1}')}", ", ".join(top)))
            md(ui.topic_list(temas))

    if lda is not None:
        with st.container(key="card_temas_prueba"):
            md(ui.card_title(
                "Prueba con un texto",
                "Pega un texto en español y mira cómo se reparte entre los ocho temas.",
            ))
            st.text_area(
                "Texto en español", key="texto_lda", height=120,
                placeholder="Pega aquí un texto en español...",
            )
            if st.button("Ver temas", type="primary", icon=":material/search:", key="btn_temas"):
                texto = st.session_state.get("texto_lda", "")
                limpio = limpiar_texto_es(texto) if texto.strip() else ""
                if not limpio:
                    st.warning("Escribe un texto en español con algunas palabras más para poder analizarlo.")
                else:
                    dist = lda.transform(vec.transform([limpio]))[0]
                    filas = [(NOMBRES_TOPICOS.get(i, f"Tema {i + 1}"), float(p)) for i, p in enumerate(dist)]
                    md(ui.topic_bars(sorted(filas, key=lambda f: f[1], reverse=True)))

    with st.expander("Ver los gráficos del análisis"):
        for nombre in ["16_lda_topicos_ecuador.png", "17_lda_distribucion_topicos.png"]:
            ruta = REPORTS_DIR / nombre
            if ruta.exists():
                st.image(str(ruta))


def page_datos():
    with st.container(key="card_corpus"):
        md(ui.card_title("Los corpus"))
        md(ui.stats([
            ("38,473", "noticias en Fake & Real News (Kaggle), tras la limpieza"),
            ("43.4%", "menos texto después del preprocesamiento"),
            ("1,543", "verificaciones de Ecuador Chequea, 2016-2026"),
            ("0.735", "F1 macro del BERT en español con pesos por clase (5 particiones)"),
        ]))

    with st.container(key="card_modelos"):
        md(ui.card_title(
            "Modelos en inglés",
            "Entrenados y evaluados con Fake & Real News. Precision, recall y F1 ponderados por clase. "
            "Resaltado: el mejor baseline.",
        ))
        md(ui.table(
            ["Modelo", "Accuracy", "Precision", "Recall", "F1"],
            [
                ["Naive Bayes", "94.71%", "0.95", "0.95", "0.9471"],
                ["Logistic Regression", "98.69%", "0.99", "0.99", "0.9869"],
                ["Random Forest", "99.21%", "0.99", "0.99", "0.9921"],
                ["SVM (mejor baseline)", "99.45%", "0.99", "0.99", "0.9945"],
                ["BERT base, inglés", "99.91%", "n/d", "n/d", "0.9991"],
                ["SVM, prueba externa (AG News)", "49.00%", "0.49", "0.49", "0.46"],
            ],
            key_rows=(3,),
            nota="n/d: no se calculó en el notebook original. La prueba externa del SVM usó 100 noticias "
                 "y muestra que el modelo no generaliza fuera de su dominio.",
        ))

    with st.container(key="card_modelos_es"):
        md(ui.card_title(
            "Modelos en español",
            "Corpus de Ecuador Chequea: 1,543 artículos, de los cuales solo el 14% son Verificado. "
            "Validación cruzada de 5 particiones; se muestra la media y, cuando se calculó, la "
            "desviación entre particiones.",
        ))
        md(ui.table(
            ["Modelo", "Accuracy", "F1 macro", "Recall de Verificado"],
            [
                ["Siempre la clase mayoritaria", "0.855 ± 0.001", "0.461 ± 0.000", "0.000 ± 0.000"],
                ["TF-IDF + Regresión logística", "0.872 ± 0.010", "0.720 ± 0.043", "0.484 ± 0.131"],
                ["TF-IDF + SVM", "0.881 ± 0.010", "0.687 ± 0.045", "0.332 ± 0.093"],
                ["BERT multilingüe con pesos por clase", "0.843 ± 0.040", "0.735 ± 0.043", "0.695 ± 0.082"],
            ],
            key_rows=(3,),
            nota="El accuracy no es la métrica principal: responder siempre Desinformación ya acierta el "
                 "85.5%. Por eso se usan F1 macro y el recall de la clase minoritaria. Con pesos por clase "
                 "el BERT detecta más contenido verificado, a costa de más falsas alarmas. La primera "
                 "versión del BERT, sin pesos y evaluada en una sola partición de 232 artículos, tuvo "
                 "F1 macro 0.63 y recall de Verificado 0.26.",
        ))

    with st.expander("Ver los gráficos del análisis exploratorio"):
        for nombre in ["01_distribucion_basica.png", "02_wordcloud.png", "04_eda_completo.png",
                       "13_corpus_ecuatoriano.png", "15_bert_ecuador_resultados.png"]:
            ruta = REPORTS_DIR / nombre
            if ruta.exists():
                st.image(str(ruta))


# ============================================================
# ARRANQUE: tema, navegacion lateral y cabecera comun
# ============================================================
ui.inject_css()
st.logo(str(LOGO) if LOGO else MARCA_SVG, size="large")

PAG_EXPLICAR = st.Page(page_explicar, title="Explicar", icon=":material/lightbulb:", url_path="explicar")
paginas = [
    st.Page(page_verificar, title="Verificar", icon=":material/search:", url_path="verificar", default=True),
    PAG_EXPLICAR,
    st.Page(page_temas, title="Temas", icon=":material/push_pin:", url_path="temas"),
    st.Page(page_datos, title="Datos", icon=":material/bar_chart:", url_path="datos"),
]
pagina = st.navigation(paginas)

with st.sidebar:
    md(ui.side_footer())

md(ui.masthead())
pagina.run()