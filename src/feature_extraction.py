# ================================================
# src/feature_extraction.py
# Vectorización de texto con TF-IDF
# Proyecto: Detección de Noticias Falsas - USFQ
# Autor: Steeven Quezada
# ================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os

def get_tfidf_features(
    df,
    text_column,
    label_column='label',
    max_features=10000,
    test_size=0.2,
    random_state=42
):
    """
    Convierte texto limpio en vectores TF-IDF y divide
    en conjuntos de entrenamiento y prueba.

    Args:
        df:            DataFrame con texto preprocesado
        text_column:   columna a vectorizar 
                       ('text_clean', 'title_clean', 'title_text_clean')
        label_column:  columna objetivo (default: 'label')
        max_features:  máximo de features TF-IDF (default: 10,000)
        test_size:     proporción del conjunto de prueba (default: 0.2)
        random_state:  semilla para reproducibilidad (default: 42)

    Returns:
        X_train, X_test, y_train, y_test, vectorizer
    """
    # Separar texto y etiquetas
    X = df[text_column].fillna('')
    y = df[label_column]

    # Dividir en train y test
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # mantiene proporciones de clases
    )

    # Vectorizar con TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # unigramas y bigramas
        min_df=2,            # ignorar términos < 2 documentos
        max_df=0.95,         # ignorar términos en > 95% documentos
        sublinear_tf=True    # aplica log en frecuencia de términos
    )

    X_train = vectorizer.fit_transform(X_train_raw)
    X_test  = vectorizer.transform(X_test_raw)

    print(f"Enfoque: '{text_column}'")
    print(f"  Train: {X_train.shape[0]:,} muestras | {X_train.shape[1]:,} features")
    print(f"  Test:  {X_test.shape[0]:,} muestras  | {X_test.shape[1]:,} features")
    print(f"  Balance train — Falsas: {(y_train==0).sum():,} | Reales: {(y_train==1).sum():,}")

    return X_train, X_test, y_train, y_test, vectorizer


def save_vectorizer(vectorizer, name, path='models/'):
    """
    Guarda el vectorizador entrenado para reutilizarlo
    en el dashboard y en los modelos de BERT.
    """
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f'tfidf_{name}.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"  ✓ Vectorizador guardado en {filepath}")


def load_vectorizer(name, path='models/'):
    """
    Carga un vectorizador previamente guardado.
    """
    filepath = os.path.join(path, f'tfidf_{name}.pkl')
    with open(filepath, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer


# ================================================
# TEST RÁPIDO — ejecutar desde la raíz del proyecto
# python src/feature_extraction.py
# ================================================
if __name__ == "__main__":
    import sys
    sys.path.append('src')

    # Cargar dataset preprocesado
    df = pd.read_csv('data/processed/dataset_preprocessed.csv', low_memory=False)
    print(f"Dataset cargado: {len(df):,} registros\n")

    # Probar los tres enfoques
    enfoques = ['text_clean', 'title_clean', 'title_text_clean']

    for enfoque in enfoques:
        print(f"{'='*50}")
        X_train, X_test, y_train, y_test, vec = get_tfidf_features(df, enfoque)
        save_vectorizer(vec, enfoque)
        print()

    print("✓ feature_extraction.py funcionando correctamente")