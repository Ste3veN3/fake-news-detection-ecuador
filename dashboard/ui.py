"""
Capa de presentacion del dashboard (estilos y componentes HTML).
La logica de modelos vive en app.py; aqui solo se construye lo que se ve.
"""

import html
import re

import streamlit as st

# ============================================================
# TOKENS DE DISENO
# ============================================================
POS = "#1B8A5A"      # clase positiva (Real / Verificado)
NEG = "#C8372D"      # clase negativa (Falsa / Desinformacion)
POS_RGB = "27, 138, 90"
NEG_RGB = "200, 55, 45"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');

:root {
  --paper: #F4F6FA;
  --side: #ECEFF5;
  --card: #FFFFFF;
  --ink: #14213D;
  --muted: #5B6578;
  --rule: #DDE2EA;
  --primary: #1F4FBF;
  --pos: #1B8A5A;
  --neg: #C8372D;
  --sans: 'Public Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}

/* Base */
html { font-size: 17px !important; }
.stApp, .stApp p, .stApp label, .stApp li, .stApp textarea,
.stApp button, .stApp a, .stApp [role="tab"] { font-family: var(--sans); }
.stApp { background: var(--paper); color: var(--ink); }
.block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem; }
[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

/* Barra lateral */
[data-testid="stSidebar"] { background: var(--side); border-right: 1px solid var(--rule); }
[data-testid="stSidebarNavLink"] {
  border-radius: 8px; padding: .55rem .8rem; margin-bottom: .25rem; color: var(--ink);
}
[data-testid="stSidebarNavLink"][aria-current="page"] { background: var(--ink) !important; }
[data-testid="stSidebarNavLink"][aria-current="page"],
[data-testid="stSidebarNavLink"][aria-current="page"] * { color: #FFFFFF !important; }
.side-foot { color: var(--muted); font-size: .82rem; line-height: 1.55; margin-top: 2.2rem; }
.side-foot b { color: var(--ink); font-weight: 600; }

/* Logo en la barra lateral y barra superior */
[data-testid="stSidebarHeader"] { height: auto !important; padding-top: 1.2rem; padding-bottom: .2rem; }
/* Sube el menú: más negativo = más arriba (prueba entre -1rem y -4rem) */
[data-testid="stSidebarNav"] { margin-top: -2.5rem; }
/* Tamaño del logo: cambia el ancho (width) para hacerlo más grande o más chico */
[data-testid="stSidebarHeader"] img, [data-testid="stLogo"], [data-testid="stSidebarLogo"] {
  width: 12.5rem !important; max-width: 100% !important;
  height: auto !important; max-height: none !important; object-fit: contain;
}
[data-testid="stAppDeployButton"] { display: none !important; }

/* Cabecera */
.mh { display: flex; align-items: flex-start; justify-content: space-between; gap: 1.2rem; margin-bottom: 1.5rem; }
.mh-left { display: flex; gap: .95rem; align-items: center; min-width: 0; flex: 1; }
.mh-icon svg { width: 3.1rem; height: 3.1rem; display: block; }
.mh-title { font-size: 2.3rem; font-weight: 700; letter-spacing: -0.015em; line-height: 1.1; }
.mh-sub { color: var(--muted); margin: .4rem 0 0 0; max-width: 50ch; line-height: 1.5; }
.chip {
  border: 1px solid var(--rule); background: var(--card); border-radius: 999px;
  padding: .3rem .9rem; font-size: .82rem; color: var(--muted); white-space: nowrap; flex-shrink: 0;
}

/* Tarjetas */
[class*="st-key-card_"] {
  background: var(--card); border: 1px solid var(--rule); border-radius: 14px;
  box-shadow: 0 1px 2px rgba(20, 33, 61, .05), 0 8px 22px rgba(20, 33, 61, .06);
  padding: 1.5rem 1.7rem !important; margin-bottom: 1.2rem;
}
.card-title { font-size: 1.4rem; font-weight: 700; line-height: 1.25; margin: 0 0 .3rem 0; }
.card-sub { color: var(--muted); line-height: 1.55; margin: 0 0 1rem 0; max-width: 66ch; }
.card-h3 { font-size: 1.05rem; font-weight: 600; margin: 0 0 .5rem 0; }
.card-h3.sp { margin-top: 1.6rem; }
.card-note { color: var(--muted); font-size: .9rem; line-height: 1.55; margin: 0 0 .8rem 0; }

/* Campos y botones */
.stApp textarea {
  font-size: 1rem; line-height: 1.6; border-radius: 10px;
  background: #F8FAFD; border: 1px solid var(--rule);
}
.stApp [data-testid="stBaseButton-primary"], .stApp button[kind="primary"] {
  background-color: var(--primary) !important; border-color: var(--primary) !important;
  border-radius: 8px; font-weight: 600;
}
.stApp [data-testid="stBaseButton-primary"] p, .stApp button[kind="primary"] p,
.stApp [data-testid="stBaseButton-primary"] span { color: #FFFFFF !important; }
.stApp [data-testid="stBaseButton-primary"]:hover { background-color: #163E96 !important; border-color: #163E96 !important; }
.stApp [data-testid="stBaseButton-secondary"], .stApp button[kind="secondary"] {
  border-radius: 8px; border-color: var(--rule); background: #F8FAFD;
}

/* Resultado */
.badge {
  display: inline-block; font-size: 1.5rem; font-weight: 700; line-height: 1;
  padding: .5rem .9rem; border-radius: 8px; margin-bottom: .4rem;
}
.badge.pos { color: var(--pos); background: rgba(27, 138, 90, .12); }
.badge.neg { color: var(--neg); background: rgba(200, 55, 45, .11); }
.badge.mid { color: var(--ink); background: #E6EAF1; }
.meter { margin: .85rem 0; }
.meter-label { display: flex; justify-content: space-between; font-size: .95rem; margin-bottom: .3rem; }
.meter-label b { font-weight: 600; }
.meter-track { height: 10px; border-radius: 5px; background: #E6EAF1; overflow: hidden; }
.meter-fill { height: 100%; border-radius: 5px; }
.meter-fill.pos { background: var(--pos); }
.meter-fill.neg { background: var(--neg); }
.alert { font-weight: 600; margin: 1rem 0 .2rem 0; }
.alert.neg { color: var(--neg); }
.alert.pos { color: var(--pos); }
.alert.mid { color: var(--muted); }
.verdict-note { color: var(--muted); font-size: .88rem; line-height: 1.55; margin: .8rem 0 0 0; }

/* Texto anotado */
.annot {
  font-size: 1rem; line-height: 1.95; background: #F8FAFD; border: 1px solid var(--rule);
  border-radius: 10px; padding: 1rem 1.2rem; margin: 0;
}
.annot .w { padding: .05em .18em; border-radius: 4px; border-bottom: 2px solid transparent; }
.annot .w.n { border-bottom-color: var(--neg); }
.annot .w.p { border-bottom-color: var(--pos); }
.legend { display: flex; gap: 1.4rem; flex-wrap: wrap; font-size: .85rem; color: var(--muted); margin-top: .7rem; }
.legend i { display: inline-block; width: 1.3rem; height: .65rem; border-radius: 3px; margin-right: .4rem; vertical-align: -1px; }
.legend i.n { background: rgba(200, 55, 45, .35); border-bottom: 2px solid var(--neg); }
.legend i.p { background: rgba(27, 138, 90, .35); border-bottom: 2px solid var(--pos); }

/* Filas con barra */
.rows { margin-top: .4rem; }
.row { display: grid; grid-template-columns: minmax(6rem, 13rem) 1fr 3.2rem; gap: .8rem; align-items: center; padding: .3rem 0; font-size: .92rem; }
.row .bar { height: 8px; border-radius: 4px; }
.row .num { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
.row.top .name { font-weight: 600; }

/* Lista de temas */
.topics { margin: .2rem 0 0 0; }
.topic { padding: .65rem 0; border-top: 1px solid var(--rule); }
.topic:last-child { border-bottom: 1px solid var(--rule); }
.topic b { font-weight: 600; }
.topic span { display: block; color: var(--muted); font-size: .9rem; margin-top: .15rem; }

/* Cifras */
.stats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.6rem 2.4rem; margin: .6rem 0 .2rem 0; }
.stat b { font-size: 2rem; font-weight: 700; display: block; line-height: 1.1; color: var(--ink); }
.stat span { color: var(--muted); font-size: .88rem; }

/* Tabla */
.tbl-wrap { overflow-x: auto; }
.tbl { width: 100%; border-collapse: collapse; font-size: .92rem; }
.tbl th { text-align: left; font-weight: 600; color: var(--muted); padding: .5rem .6rem; border-bottom: 1.5px solid var(--ink); }
.tbl th.n { text-align: right; }
.tbl td { padding: .55rem .6rem; border-bottom: 1px solid var(--rule); }
.tbl .n { text-align: right; font-variant-numeric: tabular-nums; }
.tbl tr.key td { font-weight: 700; background: #F3F6FC; }
.tbl-note { color: var(--muted); font-size: .85rem; margin-top: .6rem; }

/* Pie */
.foot { color: var(--muted); font-size: .85rem; margin-top: 2rem; text-align: right; }

@media (max-width: 900px) {
  .mh { flex-wrap: wrap; }
}
@media (max-width: 700px) {
  .mh-title { font-size: 1.8rem; }
  .row { grid-template-columns: 6rem 1fr 3rem; }
}
"""


# ============================================================
# UTILIDADES
# ============================================================
def _flat(s: str) -> str:
    """Una sola linea sin sangrias, para que Streamlit no lo trate como bloque de codigo."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


def _esc(texto: str) -> str:
    """Escapa HTML y el signo $ (Streamlit lo interpretaria como formula)."""
    return html.escape(str(texto)).replace("$", "&#36;")


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ============================================================
# COMPONENTES
# ============================================================
_ICONO = (
    '<svg viewBox="0 0 48 48" aria-hidden="true">'
    '<path d="M24 4 8 10v12c0 10 7 18 16 22 9-4 16-12 16-22V10L24 4z" fill="#1F4FBF"/>'
    '<path d="M16 24l6 6 11-12" fill="none" stroke="#fff" stroke-width="4" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


def masthead() -> str:
    return _flat(f"""
        <div class="mh">
          <div class="mh-left">
            <div class="mh-icon">{_ICONO}</div>
            <div>
              <div class="mh-title">Verificador de noticias</div>
              <p class="mh-sub">Detecta desinformación en español e inglés y muestra
              en qué se fijó el modelo para decidir.</p>
            </div>
          </div>
          <div class="chip">Maestría en Ciencia de Datos | USFQ</div>
        </div>
    """)


def side_footer() -> str:
    return _flat("""
        <div class="side-foot">
          <b>Autor:</b> Steeven Quezada<br>
          <b>Tutor:</b> Alejandro Proaño<br>
          Proyecto de titulación, USFQ
        </div>
    """)


def footer() -> str:
    return '<div class="foot">Steeven Quezada. Tutor: Alejandro Proaño. Proyecto de titulación, Maestría en Ciencia de Datos, USFQ.</div>'


def card_title(titulo: str, texto: str = "") -> str:
    out = f'<div class="card-title" role="heading" aria-level="2">{_esc(titulo)}</div>'
    if texto:
        out += f'<p class="card-sub">{_esc(texto)}</p>'
    return out


def card_h3(titulo: str, espacio: bool = False) -> str:
    cls = "card-h3 sp" if espacio else "card-h3"
    return f'<div class="{cls}" role="heading" aria-level="3">{_esc(titulo)}</div>'


def card_note(texto: str) -> str:
    return f'<p class="card-note">{_esc(texto)}</p>'


UMBRAL_CONCLUYENTE = 0.65  # por debajo de esta probabilidad el resultado se marca como poco concluyente


def result_panel(neg_label: str, pos_label: str, p_pos: float, detalle: str, alerta: bool = True) -> str:
    """Etiqueta ganadora, dos medidores complementarios y una nota."""
    p_pos = max(0.0, min(1.0, p_pos))
    es_pos = p_pos >= 0.5
    ganadora = pos_label if es_pos else neg_label
    p_ganadora = p_pos if es_pos else 1 - p_pos
    concluyente = p_ganadora >= UMBRAL_CONCLUYENTE
    cls = ("pos" if es_pos else "neg") if concluyente else "mid"
    if not alerta:
        bloque_alerta = ""
    elif not concluyente:
        bloque_alerta = '<p class="alert mid">Resultado poco concluyente: el modelo duda entre las dos clases.</p>'
    elif es_pos:
        bloque_alerta = '<p class="alert pos">El modelo no encontró señales de desinformación.</p>'
    else:
        bloque_alerta = '<p class="alert neg">Alerta: posible desinformación.</p>'
    return _flat(f"""
        <div class="badge {cls}">{_esc(ganadora)}</div>
        <div class="meter">
          <div class="meter-label"><span>Probabilidad de {_esc(pos_label.lower())}</span><b>{p_pos * 100:.0f}%</b></div>
          <div class="meter-track"><div class="meter-fill pos" style="width:{p_pos * 100:.0f}%"></div></div>
        </div>
        <div class="meter">
          <div class="meter-label"><span>Probabilidad de {_esc(neg_label.lower())}</span><b>{(1 - p_pos) * 100:.0f}%</b></div>
          <div class="meter-track"><div class="meter-fill neg" style="width:{(1 - p_pos) * 100:.0f}%"></div></div>
        </div>
        {bloque_alerta}
        <p class="verdict-note">{_esc(detalle)}</p>
    """)


def annotate(texto: str, pesos: dict, neg_label: str, pos_label: str) -> str:
    """Texto original con cada palabra influyente marcada segun su peso (positivo = clase positiva)."""
    if not pesos:
        return f'<div class="annot">{_esc(texto)}</div>'
    max_w = max(abs(w) for w in pesos.values()) or 1.0
    partes = []
    for token in re.split(r"(\W+)", texto):
        if not token:
            continue
        if token in pesos:
            w = pesos[token]
            alpha = 0.14 + 0.5 * abs(w) / max_w
            rgb, cls = (POS_RGB, "p") if w > 0 else (NEG_RGB, "n")
            partes.append(
                f'<span class="w {cls}" style="background:rgba({rgb},{alpha:.2f})" '
                f'title="{w:+.3f}">{_esc(token)}</span>'
            )
        else:
            partes.append(_esc(token).replace("\r", "").replace("\n", "<br>"))
    leyenda = _flat(f"""
        <div class="legend">
          <span><i class="n"></i>Empuja hacia {_esc(neg_label.lower())}</span>
          <span><i class="p"></i>Empuja hacia {_esc(pos_label.lower())}</span>
        </div>
    """)
    return f'<div class="annot">{"".join(partes)}</div>{leyenda}'


def top_words(pesos: dict, n: int = 8) -> str:
    items = sorted(pesos.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]
    if not items:
        return ""
    max_w = max(abs(w) for _, w in items) or 1.0
    filas = []
    for palabra, w in items:
        color = POS if w > 0 else NEG
        filas.append(
            f'<div class="row"><span class="name">{_esc(palabra)}</span>'
            f'<span class="bar" style="width:{abs(w) / max_w * 100:.0f}%;background:{color}"></span>'
            f'<span class="num">{w:+.2f}</span></div>'
        )
    return f'<div class="rows">{"".join(filas)}</div>'


def topic_list(temas: list) -> str:
    filas = "".join(
        f'<div class="topic"><b>{_esc(n)}</b><span>{_esc(p)}</span></div>' for n, p in temas
    )
    return f'<div class="topics">{filas}</div>'


def topic_bars(distribucion: list) -> str:
    if not distribucion:
        return ""
    top = max(range(len(distribucion)), key=lambda i: distribucion[i][1])
    filas = []
    for i, (nombre, p) in enumerate(distribucion):
        color = "var(--primary)" if i == top else "var(--rule)"
        cls = "row top" if i == top else "row"
        filas.append(
            f'<div class="{cls}"><span class="name">{_esc(nombre)}</span>'
            f'<span class="bar" style="width:{p * 100:.0f}%;background:{color}"></span>'
            f'<span class="num">{p * 100:.0f}%</span></div>'
        )
    return f'<div class="rows">{"".join(filas)}</div>'


def stats(items: list) -> str:
    bloques = "".join(
        f'<div class="stat"><b>{_esc(v)}</b><span>{_esc(d)}</span></div>' for v, d in items
    )
    return f'<div class="stats">{bloques}</div>'


def table(headers: list, rows: list, key_rows: tuple = (), nota: str = "") -> str:
    th = "".join(f'<th class="{"n" if i else ""}">{_esc(h)}</th>' for i, h in enumerate(headers))
    trs = []
    for idx, fila in enumerate(rows):
        celdas = "".join(f'<td class="{"n" if i else ""}">{_esc(c)}</td>' for i, c in enumerate(fila))
        cls = ' class="key"' if idx in key_rows else ""
        trs.append(f"<tr{cls}>{celdas}</tr>")
    out = f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>'
    if nota:
        out += f'<p class="tbl-note">{_esc(nota)}</p>'
    return out