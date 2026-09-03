# ================================================
# src/scraper_ecuador_chequea.py
# Web scraping de Ecuador Chequea via WordPress API
# Proyecto: Detección de Noticias Falsas - USFQ
# Autor: Steeven Quezada
# ================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

BASE_API = "https://ecuadorchequea.com/wp-json/wp/v2/posts"
CAT_ID   = 4289  # Verificaciones
HEADERS  = {"User-Agent": "Mozilla/5.0"}


def extraer_etiqueta_completa(titulo, contenido):
    """
    Extrae la etiqueta buscando en título y contenido.
    Orden de prioridad: título → contenido → OTROS
    """
    t     = titulo.upper()
    c     = contenido.upper()[:1000]
    texto = t + ' ' + c

    # Buscar en título primero
    if 'FALSO' in t:                                        return 'FALSO'
    elif 'ENGAÑOSO' in t:                                   return 'ENGAÑOSO'
    elif 'CIERTO' in t:                                     return 'CIERTO'
    elif 'VERDADERO' in t:                                  return 'VERDADERO'
    elif 'IMPRECISO' in t:                                  return 'IMPRECISO'
    elif 'FALTA CONTEXTO' in t:                             return 'FALTA CONTEXTO'

    # Buscar patrón "Ecuador Chequea califica" en contenido
    elif 'CALIFICA EL CONTENIDO COMO FALSO' in texto or \
         'CALIFICA COMO FALSO' in texto or \
         'ES FALSO' in texto:                               return 'FALSO'
    elif 'CALIFICA EL CONTENIDO COMO ENGAÑOSO' in texto or \
         'CALIFICA COMO ENGAÑOSO' in texto or \
         'ES ENGAÑOSO' in texto:                            return 'ENGAÑOSO'
    elif 'CALIFICA EL CONTENIDO COMO CIERTO' in texto or \
         'CALIFICA COMO CIERTO' in texto or \
         'ES CIERTO' in texto:                              return 'CIERTO'
    elif 'CALIFICA EL CONTENIDO COMO IMPRECISO' in texto or \
         'CALIFICA COMO IMPRECISO' in texto or \
         'ES IMPRECISO' in texto:                           return 'IMPRECISO'
    elif 'CALIFICA EL CONTENIDO COMO VERDADERO' in texto or \
         'CALIFICA COMO VERDADERO' in texto:                return 'VERDADERO'
    elif 'FALTA CONTEXTO' in texto:                         return 'FALTA CONTEXTO'

    # Inferir del título como último recurso
    elif ' NO ' in t or 'NO ES' in t:                      return 'FALSO'
    elif ' SÍ ' in t or ' SI ' in t:                       return 'CIERTO'
    elif 'MEZCLA' in t or 'COMBINA' in t:                  return 'ENGAÑOSO'
    else:                                                    return 'OTROS'


def limpiar_html(html):
    """Elimina tags HTML y retorna texto limpio."""
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator=' ', strip=True)


def scrape(max_pages=190, per_page=20, delay=1.0):
    """
    Scraping completo via WordPress REST API.

    Args:
        max_pages: páginas de la API a recorrer (190 = corpus completo)
        per_page:  artículos por página (max 100)
        delay:     segundos entre requests
    """
    all_articles = []

    total_esperado = max_pages * per_page
    print(f"Iniciando scraping de Ecuador Chequea...")
    print(f"Páginas API: {max_pages} | Artículos esperados: ~{total_esperado}\n")

    for page in range(1, max_pages + 1):
        url = (f"{BASE_API}?categories={CAT_ID}"
               f"&per_page={per_page}&page={page}")

        try:
            response = requests.get(url, headers=HEADERS, timeout=15)

            if response.status_code != 200:
                print(f"Página {page}: Error {response.status_code}")
                break

            data = response.json()

            if not data:
                print(f"Página {page}: Sin más artículos")
                break

            print(f"Página {page}/{max_pages}: {len(data)} artículos")

            for art in data:
                titulo    = limpiar_html(art['title']['rendered'])
                contenido = limpiar_html(art['content']['rendered'])
                etiqueta  = extraer_etiqueta_completa(titulo, contenido)
                fecha     = art['date'][:10]
                url_art   = art['link']

                all_articles.append({
                    'titulo':    titulo,
                    'contenido': contenido,
                    'etiqueta':  etiqueta,
                    'fecha':     fecha,
                    'url':       url_art,
                })

                print(f"  {etiqueta:<15} → {titulo[:60]}")

            time.sleep(delay)

        except Exception as e:
            print(f"Error en página {page}: {e}")
            break

    # Guardar resultado
    df = pd.DataFrame(all_articles)

    if len(df) > 0:
        os.makedirs('data/external', exist_ok=True)
        df.to_csv('data/external/corpus_ecuador_chequea.csv',
                  index=False, encoding='utf-8-sig')

        print(f"\n{'='*55}")
        print(f"✓ Scraping completado")
        print(f"  Total artículos:  {len(df):,}")
        print(f"  Distribución de etiquetas:")
        print(df['etiqueta'].value_counts().to_string())
        print(f"\n  Guardado en: data/external/corpus_ecuador_chequea.csv")
    else:
        print("No se obtuvieron artículos.")

    return df


if __name__ == "__main__":
    df = scrape(max_pages=190, per_page=20, delay=1.0)
    if len(df) > 0:
        print(f"\nPrimeros 5 artículos:")
        print(df[['etiqueta', 'titulo']].head())