# Sistema de Detección y Análisis de Noticias Falsas mediante NLP y Deep Learning

**Proyecto de Titulación — Maestría en Ciencia de Datos**  
Universidad San Francisco de Quito (USFQ)  
Quito, Ecuador · 2026

---

## Información del proyecto

| | |
|---|---|
| **Autor** | [Steeven Quezada] |
| **Tutor** | [Alejandro Proaño] |
| **Programa** | Maestría en Ciencia de Datos |
| **Estado** | En desarrollo |
| **Período** | 2026 |

---

## Descripción

Este proyecto desarrolla un sistema integral para la detección automática de noticias falsas aplicando técnicas de Procesamiento de Lenguaje Natural (NLP), aprendizaje automático y Explainable AI (XAI). El sistema clasifica contenido como verdadero o falso, analiza los patrones lingüísticos que caracterizan la desinformación, y explica las decisiones de clasificación mediante métodos interpretables (SHAP, LIME).

Como contribución original, se construye el primer corpus ecuatoriano de desinformación etiquetada, recopilado de las plataformas de fact-checking Ecuador Chequea y Lupa Media, ambas certificadas por la International Fact-Checking Network (IFCN).

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
├── notebooks/            # Jupyter Notebooks por fase
├── src/                  # Módulos Python reutilizables
├── models/               # Modelos entrenados serializados
├── reports/              # Figuras y visualizaciones para la tesis
├── dashboard/            # Aplicación Streamlit
└── requirements.txt
```

---

## Datasets utilizados

| Capa | Dataset | Tamaño | Fuente |
|---|---|---|---|
| Base (inglés) | Fake & Real News | ~45,000 artículos | Kaggle |
| Base (inglés) | LIAR / LIARx | 12,836 declaraciones | Zenodo |
| Base (inglés) | FakeNewsNet | Variable | GitHub |
| Base (inglés) | COVID-19 Fake News | ~10,700 noticias | Kaggle |
| Español | Spanish Fake News Dataset | — | Univ. Complutense de Madrid (2025) |
| Ecuador 🇪🇨 | Corpus Ecuador Chequea | 1,500+ verificaciones | Web scraping |
| Ecuador 🇪🇨 | Corpus Lupa Media | 247+ casos | Web scraping |

---

## Stack tecnológico

- **Lenguaje:** Python 3.10+
- **NLP y ML:** scikit-learn, NLTK, spaCy, HuggingFace Transformers
- **Deep Learning:** TensorFlow/Keras, PyTorch
- **Explicabilidad:** SHAP, LIME
- **Análisis de redes:** NetworkX
- **Visualización:** matplotlib, seaborn, Streamlit
- **Cómputo GPU:** Google Colab / Kaggle Notebooks

---

## Cómo reproducir el entorno

```bash
git clone https://github.com/TU_USUARIO/fake-news-detection-ecuador.git
cd fake-news-detection-ecuador
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Estado del proyecto

- [x] Propuesta aprobada
- [x] Repositorio inicializado
- [x] Entorno configurado
- [ ] EDA completado
- [ ] Modelos baseline
- [ ] BERT fine-tuning
- [ ] XAI implementado
- [ ] Dashboard deployed
- [ ] Tesis entregada

---

## Contacto

**[Steeven Quezada]**  
[bquezadah@estud.usfq.edu.ec]  
Maestría en Ciencia de Datos · USFQ · Quito, Ecuador
