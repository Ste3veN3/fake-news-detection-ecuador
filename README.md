# Sistema de Detección y Análisis de Noticias Falsas mediante NLP y Deep Learning

**Proyecto de Titulación — Maestría en Ciencia de Datos**  
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

El sistema es evaluado sobre datasets internacionales de referencia (Fake & Real News, LIAR, FakeNewsNet) y validado sobre contenido en español. Como contribución original, se construye el primer corpus ecuatoriano de desinformación etiquetada, recopilado de las plataformas de fact-checking Ecuador Chequea y Lupa Media, ambas certificadas por la International Fact-Checking Network (IFCN).

---

## Objetivos

**General:** Desarrollar un sistema integral de detección automática de noticias falsas que integre modelos Transformer, análisis de patrones lingüísticos y técnicas de explicabilidad en IA, evaluado sobre múltiples datasets públicos de referencia y validado sobre contenido ecuatoriano.

**Específicos:**
1. Diseñar y ejecutar un pipeline completo de preprocesamiento y análisis exploratorio de cuatro datasets de noticias etiquetadas.
2. Implementar y comparar modelos de clasificación desde enfoques clásicos (TF-IDF + ML) hasta arquitecturas avanzadas (LSTM, BERT/RoBERTa).
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
│   ├── 01_exploracion_inicial.ipynb   # EDA completo
│   ├── 02_preprocesamiento.ipynb      # Pipeline NLP - 3 enfoques
│   ├── 03_modelos_baseline.ipynb      # SVM 99.45% | LR | NB | RF
│   └── 04_bert_finetuning.ipynb       # BERT base GPU T4 - 99.91%
│
├── references/
│   └── references.bib                 # Referencias bibliográficas APA
│
├── reports/                           # Visualizaciones del EDA y modelos
│   ├── 01_distribucion_basica.png
│   ├── 02_wordcloud.png
│   ├── 03_sentimiento.png
│   ├── 04_eda_completo.png
│   ├── 05_features_linguisticas.png
│   ├── 06_baseline_comparacion.png
│   └── 07_confusion_matrix_baseline.png
│
├── src/
│   ├── preprocessing.py              # Pipeline de preprocesamiento NLP
│   └── feature_extraction.py         # Vectorización TF-IDF
│
├── models/                           # Modelos entrenados (local, no en GitHub)
│   ├── baseline_svm_best.pkl         # SVM - mejor baseline
│   ├── tfidf_*.pkl                   # Vectorizadores TF-IDF
│   └── bert_finetuned/               # BERT fine-tuned (427MB)
│
├── thesis/                           # Documento de tesis (local, no en GitHub)
├── dashboard/                        # Aplicación Streamlit (pendiente)
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
| **BERT base** | **Título+Texto** | **99.91%** | **0.9991** |

---

## Datasets utilizados

| Capa | Dataset | Tamaño | Fuente | Estado |
|---|---|---|---|---|
| Base (inglés) | Fake & Real News | ~45,000 artículos | Kaggle | ✅ Implementado |
| Base (inglés) | LIAR / LIARx | 12,836 declaraciones | Zenodo | ⏳ Pendiente |
| Base (inglés) | FakeNewsNet | Variable | GitHub | ⏳ Pendiente |
| Base (inglés) | COVID-19 Fake News | ~10,700 noticias | Kaggle | ⏳ Pendiente |
| Español | Spanish Fake News Dataset | — | Univ. Complutense de Madrid (2025) | ⏳ Pendiente |
| Ecuador 🇪🇨 | Corpus Ecuador Chequea | 1,500+ verificaciones | Web scraping | ⏳ Pendiente |
| Ecuador 🇪🇨 | Corpus Lupa Media | 247+ casos | Web scraping | ⏳ Pendiente |

---

## Stack tecnológico

- **Lenguaje:** Python 3.14
- **NLP y ML:** scikit-learn, NLTK, spaCy, HuggingFace Transformers
- **Deep Learning:** PyTorch, BERT base uncased
- **Explicabilidad:** SHAP, LIME
- **Análisis de redes:** NetworkX
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

## Estado del proyecto

- [x] Propuesta aprobada
- [x] Repositorio inicializado
- [x] Entorno configurado
- [x] EDA completado
- [x] Preprocesamiento NLP completado (3 enfoques)
- [x] Marco Teórico redactado (7 papers, 14 referencias)
- [x] Modelos baseline completados (mejor: SVM 99.45%)
- [x] BERT fine-tuning completado (99.91% accuracy, F1 0.9991)
- [ ] XAI implementado (SHAP + LIME)
- [ ] Corpus ecuatoriano construido
- [ ] Dashboard deployed
- [ ] Tesis entregada

---

## Contacto

**Steeven Quezada**  
bquezadah@estud.usfq.edu.ec  
Maestría en Ciencia de Datos · USFQ · Quito, Ecuador