# Sistema de Detección y Análisis de Noticias Falsas

**Proyecto de Titulación - Maestría en Ciencia de Datos**  
Universidad San Francisco de Quito (USFQ)  
Quito, Ecuador · 2026

---

## Información del proyecto

| | |
|---|---|
| **Autor** | Steeven Quezada |
| **Tutor** | Alejandro Proaño |
| **Programa** | Maestría en Ciencia de Datos |
| **Estado** | En desarrollo |
| **Período** | 2026 |

---

## Descripción

Este proyecto desarrolla un sistema integral para la detección automática de noticias falsas aplicando técnicas de Procesamiento de Lenguaje Natural (NLP), aprendizaje automático y Explainable AI (XAI). El sistema clasifica contenido como verdadero o falso, analiza los patrones lingüísticos que caracterizan la desinformación, y explica las decisiones de clasificación mediante métodos interpretables (SHAP, LIME).

El sistema es evaluado sobre el dataset Fake & Real News (Kaggle) y validado mediante una prueba de generalización con noticias de AG News y noticias generadas con un modelo de lenguaje. Como contribución original, se construye un corpus ecuatoriano de 1,543 verificaciones etiquetadas de Ecuador Chequea, organización signataria verificada de la International Fact-Checking Network (IFCN), que cubre de 2016 a 2026.

---

## Objetivos

**General:** Desarrollar un sistema integral de detección automática de noticias falsas que integre modelos Transformer, análisis de patrones lingüísticos y técnicas de explicabilidad en IA, evaluado sobre múltiples datasets públicos de referencia y validado sobre contenido ecuatoriano.

**Específicos:**
1. Diseñar y ejecutar un pipeline completo de preprocesamiento y análisis exploratorio de noticias etiquetadas.
2. Implementar y comparar modelos de clasificación desde enfoques clásicos (TF-IDF + ML) hasta arquitecturas avanzadas (BERT).
3. Aplicar técnicas de Explainable AI (SHAP, LIME) para identificar patrones lingüísticos asociados a la desinformación.
4. Construir un corpus original de verificaciones ecuatorianas mediante web scraping de fuentes certificadas.
5. Desarrollar un dashboard interactivo que permita clasificar noticias en tiempo real con explicaciones visuales.

---

## Estructura del repositorio

```
fake-news-detection-ecuador/
│
├── data/
│   ├── raw/              # Datasets originales sin modificar
│   ├── processed/        # Datos limpios para modelado
│   └── external/         # Corpus ecuatoriano construido
│
├── notebooks/
│   ├── 01_exploracion_inicial.ipynb        # EDA completo
│   ├── 02_preprocesamiento.ipynb           # Pipeline NLP - 3 enfoques
│   ├── 03_modelos_baseline.ipynb           # SVM | LR | NB | RF
│   ├── 04_bert_finetuning.ipynb            # BERT base GPU T4 - inglés
│   ├── 05_evaluacion_generalizacion.ipynb  # Generalización - fuera del dominio
│   ├── 06_xai_shap_lime.ipynb              # SHAP y LIME - explicabilidad
│   ├── 07_corpus_ecuatoriano.ipynb         # Corpus Ecuador Chequea 1,500+ verificaciones
│   ├── 08_bert_finetuning_ecuador.ipynb    # BERT multilingüe corpus Ecuador
│   ├── 09_analisis_linguistico_ecuador.ipynb  # LDA - 8 tópicos corpus Ecuador
│   ├── 10_bert_pesos_por_clase_cv.ipynb    # BERT con pesos por clase, validación cruzada y líneas base
│   ├── 11_figuras_articulo.ipynb           # Figuras del artículo en PDF vectorial y conteo de plataformas
│   └── 12_bert_sin_veredicto_cv.ipynb      # BERT sin palabras de veredicto, mismas particiones del cuaderno 10
│
├── references/
│   └── references.bib                      # Referencias bibliográficas (BibTeX)
│
├── reports/                                # Visualizaciones del EDA y modelos (PNG)
│   ├── pdf/                                # Figuras del artículo en PDF vectorial (cuaderno 11)
│   ├── 01_distribucion_basica.png
│   ├── 02_wordcloud.png
│   ├── 03_sentimiento.png
│   ├── 04_eda_completo.png
│   ├── 05_features_linguisticas.png
│   ├── 06_baseline_comparacion.png
│   ├── 07_confusion_matrix_baseline.png
│   ├── 08_bert_historial.png
│   ├── 09_prueba_externa.png
│   ├── 10_lime_explicaciones.png
│   ├── 11_shap_global.png
│   ├── 12_shap_direccion.png
│   ├── 13_corpus_ecuatoriano.png
│   ├── 14_bert_corpus_ecuatoriano.png
│   ├── 15_bert_ecuador_resultados.png
│   ├── 16_lda_topicos_ecuador.png
│   └── 17_lda_distribucion_topicos.png
│
├── src/
│   ├── preprocessing.py              # Pipeline de preprocesamiento NLP
│   ├── feature_extraction.py         # Vectorización TF-IDF
│   └── scraper_ecuador_chequea.py    # Web scraping Ecuador Chequea
│
├── models/                           # Modelos entrenados (local, no en GitHub)
│   ├── baseline_svm_best.pkl         # SVM - mejor baseline
│   ├── tfidf_*.pkl                   # Vectorizadores TF-IDF
│   ├── lda_model.pkl                 # LDA - 8 tópicos
│   ├── lda_vectorizer.pkl            # CountVectorizer del LDA
│   ├── bert_finetuned/               # BERT fine-tuned inglés
│   └── bert_ecuador/                 # BERT multilingüe con pesos por clase (español Ecuador)
│
├── thesis/                           # Documento de tesis (local, no en GitHub)
├── dashboard/                        # Aplicación Streamlit (v1 funcional en local)
│   ├── app.py                        # Páginas, carga de modelos y lógica
│   ├── ui.py                         # Estilos y componentes visuales
│   └── assets/                       # Logo de la universidad (usfq_logo.png o .svg)
├── .streamlit/
│   └── config.toml                   # Tema del dashboard
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Resultados obtenidos

**Inglés (Fake & Real News y prueba externa)**

| Modelo | Enfoque | Accuracy | F1-score |
|---|---|---|---|
| Naive Bayes | Título+Texto | 94.71% | 0.9471 |
| Logistic Regression | Título+Texto | 98.69% | 0.9869 |
| Random Forest | Título+Texto | 99.21% | 0.9921 |
| SVM | Título+Texto | 99.45% | 0.9945 |
| **BERT base (inglés)** | **Título+Texto** | **99.91%** | **0.9991** |
| SVM (generalización externa) | AG News + LLM | 49.00% | - |

**Español (corpus Ecuador Chequea, validación cruzada de 5 particiones)**

Solo el 14.5% de los 1,543 artículos es Verificado, así que responder siempre Desinformación ya da 85.5% de accuracy. Por eso la métrica principal es F1 macro y el recall de la clase minoritaria.

| Modelo | Accuracy | F1 macro | Recall de Verificado |
|---|---|---|---|
| Siempre la clase mayoritaria | 0.855 ± 0.001 | 0.461 ± 0.000 | 0.000 ± 0.000 |
| TF-IDF + Regresión logística | 0.872 ± 0.010 | 0.720 ± 0.043 | 0.484 ± 0.131 |
| TF-IDF + SVM | 0.881 ± 0.010 | 0.687 ± 0.045 | 0.332 ± 0.093 |
| **BERT multilingüe con pesos por clase** | 0.843 ± 0.040 | **0.735 ± 0.043** | **0.695 ± 0.082** |

Con pesos por clase, el BERT empata con la regresión logística en F1 macro y detecta más contenido verificado, a costa de más falsas alarmas. La primera versión del BERT, sin pesos y evaluada en una sola partición de 232 artículos, tuvo F1 macro 0.63 y recall de Verificado 0.26.

**Dependencia del vocabulario de veredicto (corpus Ecuador Chequea)**

Las etiquetas se asignaron con reglas de palabras clave aplicadas al propio texto (cuaderno 07). Se repitió la validación cruzada quitando esas palabras (cuadernos 11 y 12):

| Modelo | Entrada | F1 macro | Recall de Verificado |
|---|---|---|---|
| Regresión logística | Texto completo | 0.718 | 0.480 |
| Regresión logística | Sin veredicto | 0.651 | 0.390 |
| BERT multilingüe | Texto completo | 0.735 | 0.695 |
| BERT multilingüe | Sin veredicto | 0.715 | 0.659 |

En el BERT la caída de F1 macro (0.020) no se distingue del ruido entre particiones (prueba t pareada, p = 0.52); la regresión logística baja más. Es una sola corrida, y quitar esas palabras no elimina otras pistas del estilo de las verificaciones. La regresión logística aparece con 0.718 en esta ejecución local y con 0.720 en el cuaderno 10 (Colab).

---

## Limitaciones conocidas

- **Dominio:** Fake & Real News es política de EE. UU. de 2016 y 2017. Fuera de ese dominio, el SVM baja de 99.45% a 49.00% de accuracy.
- **Fuente y estilo:** en el análisis con SHAP, la palabra "reuters" pesa unas tres veces más que cualquier otra (0.2282 contra 0.0779). El modelo aprendió la fuente y el estilo de redacción, no la veracidad.
- **Español:** el corpus es pequeño y desbalanceado (14.5% Verificado), y los modelos se entrenaron con el texto de las verificaciones, no con afirmaciones sueltas.
- **Etiquetas:** las del corpus ecuatoriano se asignaron con reglas de palabras clave y no se midió cuántas son incorrectas. Además, se descartó el 59% de los artículos descargados (2,256 de 3,799) por no tener una etiqueta clara.
- **Plataformas:** las verificaciones mencionan Facebook (58.2%), Twitter/X (39.3%) y TikTok (26.2%), pero eso mide menciones en el texto, no el canal por el que circuló el contenido.
- **Correcciones:** los cuadernos 06, 07, 08 y 09 tienen notas de corrección donde un resultado o una frase inicial quedó superado. Las figuras vigentes del artículo están en el cuaderno 11.

---

## Datasets utilizados

| Capa | Dataset | Tamaño | Fuente | Estado |
|---|---|---|---|---|
| Entrenamiento | Fake & Real News | 44,898 artículos (38,473 tras la limpieza) | Kaggle | ✅ Implementado |
| Generalización | AG News + noticias generadas con un LLM | 50 reales (de 7,600 de la partición de prueba) y 50 generadas | HuggingFace (noticias de múltiples fuentes) | ✅ Implementado |
| Ecuador 🇪🇨 | Corpus Ecuador Chequea | 1,543 verificaciones | Web scraping (API REST) | ✅ Implementado |

---

## Stack tecnológico

- **Lenguaje:** Python 3.14
- **NLP y ML:** scikit-learn, NLTK, spaCy, HuggingFace Transformers
- **Deep Learning:** PyTorch, BERT base uncased, BERT multilingüe
- **Explicabilidad:** SHAP, LIME
- **Web Scraping:** requests, BeautifulSoup
- **Visualización:** matplotlib, seaborn, Streamlit
- **Cómputo GPU:** Google Colab T4

---

## Cómo reproducir el entorno

```bash
git clone https://github.com/Ste3veN3/fake-news-detection-ecuador
cd fake-news-detection-ecuador
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Cómo correr el dashboard

Requiere los modelos entrenados en `models/` (no incluidos en GitHub por su tamaño):
`baseline_svm_best.pkl`, `tfidf_title_text_clean.pkl`, `lda_model.pkl`, `lda_vectorizer.pkl`,
`bert_finetuned/` y `bert_ecuador/`.

Desde la raíz del repositorio, con el entorno virtual activado:

```bash
python -m streamlit run dashboard/app.py
```

El dashboard tiene cuatro secciones, en la barra lateral:

- **Verificar:** detecta el idioma del texto y lo clasifica con el BERT correspondiente (BERT base en inglés: Falsa o Real; BERT multilingüe Ecuador en español: Desinformación o Verificado). Muestra las dos probabilidades y marca como poco concluyentes los resultados por debajo de 65%. En español muestra además los tres temas LDA más cercanos al texto; en inglés ofrece llevar el texto a la sección Explicar.
- **Explicar:** resalta las palabras que empujaron la decisión del SVM con TF-IDF mediante LIME (precisión Rápida de 1,500 variantes o Completa de 5,000, con semilla fija 42) y muestra la importancia global de SHAP. Incluye un par de ejemplos, con y sin la fuente, para mostrar el sesgo hacia "Reuters". Funciona con texto en inglés.
- **Temas:** los ocho temas del modelo LDA sobre el corpus ecuatoriano, con sus palabras clave, y una prueba para ver cómo se reparte un texto nuevo entre ellos.
- **Datos:** cifras de los corpus, tablas de modelos en inglés y en español (con validación cruzada) y los gráficos del análisis.

Para mostrar el logo de la universidad, guarda la imagen como `dashboard/assets/usfq_logo.png` (o `.svg`). Si no existe, el dashboard usa una marca de respaldo. El tema visual se define en `.streamlit/config.toml`.

---

## Estado del proyecto

- [x] Propuesta aprobada
- [x] Repositorio inicializado
- [x] Entorno configurado
- [x] EDA completado
- [x] Preprocesamiento NLP completado (3 enfoques)
- [x] Marco Teórico redactado (7 trabajos principales)
- [x] Modelos baseline completados (mejor: SVM)
- [x] BERT fine-tuning completado en inglés
- [x] Prueba de generalización externa (sesgo de dominio documentado)
- [x] XAI implementado (SHAP + LIME)
- [x] Corpus ecuatoriano construido (1,500+ verificaciones Ecuador Chequea)
- [x] Evaluación del BERT en inglés sobre el corpus ecuatoriano (transferencia sin adaptación)
- [x] Fine-tuning BERT multilingüe con pesos por clase en el corpus ecuatoriano (F1 macro 0.735, validación cruzada de 5 particiones)
- [x] Análisis lingüístico LDA completado (8 tópicos corpus ecuatoriano)
- [x] Dashboard Streamlit v1 (verificar, explicar, temas LDA, datos)
- [x] Figuras del artículo en PDF vectorial (notebook 11) y notas de corrección en los cuadernos 06 a 09
- [x] Chequeo de fuga de etiqueta con regresión logística y BERT (notebooks 11 y 12)
- [ ] Artículo en Overleaf (formato USFQ, en redacción)
- [ ] Dashboard deployed
- [ ] Tesis entregada

---

## Contacto

**Steeven Quezada**  
bquezadah@estud.usfq.edu.ec  
Maestría en Ciencia de Datos · USFQ · Quito, Ecuador