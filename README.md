# Sistema de Detección y Análisis de Noticias Falsas mediante NLP y Deep Learning

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

El sistema es evaluado sobre el dataset Fake & Real News (Kaggle) y validado mediante una prueba de generalización con AG News (Reuters/AP). Como contribución original, se construye el primer corpus ecuatoriano de desinformación etiquetada con más de 1,500 verificaciones de Ecuador Chequea, organización certificada por la International Fact-Checking Network (IFCN), cubriendo el período 2016-2026.

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
│   └── 09_analisis_linguistico_ecuador.ipynb  # LDA - 8 tópicos corpus Ecuador
│
├── references/
│   └── references.bib                      # Referencias bibliográficas APA
│
├── reports/                                # Visualizaciones del EDA y modelos
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
│   └── bert_ecuador/                 # BERT fine-tuned español Ecuador
│
├── thesis/                           # Documento de tesis (local, no en GitHub)
├── dashboard/                        # Aplicación Streamlit (v1 funcional en local)
│   └── app.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Resultados obtenidos

| Modelo | Enfoque | Accuracy | F1-score |
|---|---|---|---|
| Naive Bayes | Título+Texto | 94.71% | 0.9471 |
| Logistic Regression | Título+Texto | 98.69% | 0.9869 |
| Random Forest | Título+Texto | 99.21% | 0.9921 |
| SVM | Título+Texto | 99.45% | 0.9945 |
| **BERT base (inglés)** | **Título+Texto** | **99.91%** | **0.9991** |
| SVM (generalización externa) | AG News + LLM | 49.00% | - |
| BERT sin fine-tuning (Ecuador) | Contenido español | 66.62% | 0.7041 |
| **BERT multilingüe (Ecuador)** | **Contenido español** | **84.91%** | **0.8305** |

---

## Datasets utilizados

| Capa | Dataset | Tamaño | Fuente | Estado |
|---|---|---|---|---|
| Entrenamiento | Fake & Real News | ~45,000 artículos | Kaggle | ✅ Implementado |
| Generalización | AG News | 7,600 artículos | HuggingFace (Reuters/AP) | ✅ Implementado |
| Ecuador 🇪🇨 | Corpus Ecuador Chequea | 1,500+ verificaciones | Web scraping | ✅ Implementado |

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

El dashboard incluye cuatro secciones: clasificador de noticias con detección automática de idioma (BERT inglés o BERT multilingüe Ecuador), explicabilidad con SHAP y LIME, clasificación de tópicos LDA y estadísticas de los corpus.

---

## Estado del proyecto

- [x] Propuesta aprobada
- [x] Repositorio inicializado
- [x] Entorno configurado
- [x] EDA completado
- [x] Preprocesamiento NLP completado (3 enfoques)
- [x] Marco Teórico redactado (7 papers, 14 referencias)
- [x] Modelos baseline completados (mejor: SVM)
- [x] BERT fine-tuning completado en inglés
- [x] Prueba de generalización externa (sesgo de dominio documentado)
- [x] XAI implementado (SHAP + LIME)
- [x] Corpus ecuatoriano construido (1,500+ verificaciones Ecuador Chequea)
- [x] Evaluación BERT sin fine-tuning en español
- [x] Fine-tuning BERT multilingüe corpus ecuatoriano (+18 puntos de mejora)
- [x] Análisis lingüístico LDA completado (8 tópicos corpus ecuatoriano)
- [x] Dashboard Streamlit v1 (clasificador, XAI, tópicos LDA, estadísticas)
- [ ] Dashboard deployed
- [ ] Tesis entregada

---

## Contacto

**Steeven Quezada**  
bquezadah@estud.usfq.edu.ec  
Maestría en Ciencia de Datos · USFQ · Quito, Ecuador