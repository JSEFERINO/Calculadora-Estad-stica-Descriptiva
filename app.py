import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import re

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="📊 Calculadora Estadística Descriptiva",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def detectar_outliers(datos):
    """Detecta outliers usando el método IQR"""
    if len(datos) < 2:
        return {'outliers': [], 'limite_inferior': None, 'limite_superior': None,
                'q1': None, 'q3': None, 'iqr': None}

    q1 = np.percentile(datos, 25)
    q3 = np.percentile(datos, 75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    outliers = datos[(datos < limite_inferior) | (datos > limite_superior)]

    return {
        'outliers': outliers,
        'limite_inferior': limite_inferior,
        'limite_superior': limite_superior,
        'q1': q1,
        'q3': q3,
        'iqr': iqr
    }

def procesar_datos_lista(texto):
    """Procesa datos en formato lista"""
    texto = re.sub(r'[^0-9.,\-]', ' ', texto)
    texto = texto.replace(',', '.')
    numeros = [float(x) for x in texto.split() if x and x != '.']
    if len(numeros) < 2:
        return None
    return np.array(numeros)

def procesar_datos_agrupados(texto):
    """Procesa datos agrupados"""
    pares = [p.strip() for p in texto.split(';') if p.strip()]
    clases = []
    frecuencias = []

    for par in pares:
        elementos = [e.strip() for e in par.split(',') if e.strip()]
        if len(elementos) == 2:
            clase = elementos[0]
            try:
                freq = float(elementos[1])
                if freq > 0:
                    clases.append(clase)
                    frecuencias.append(freq)
            except:
                continue

    if len(clases) < 2:
        return None

    # Extraer límites y marcas de clase
    limites_inferiores = []
    limites_superiores = []
    marcas_clase = []

    for clase in clases:
        if '-' in clase:
            limites = clase.split('-')
            if len(limites) == 2:
                try:
                    lim_inf = float(limites[0].strip())
                    lim_sup = float(limites[1].strip())
                    limites_inferiores.append(lim_inf)
                    limites_superiores.append(lim_sup)
                    marcas_clase.append((lim_inf + lim_sup) / 2)
                except:
                    marcas_clase.append(np.nan)
                    limites_inferiores.append(np.nan)
                    limites_superiores.append(np.nan)

    return {
        'clases': clases,
        'frecuencias': frecuencias,
        'marcas_clase': marcas_clase,
        'limites_inferiores': limites_inferiores,
        'limites_superiores': limites_superiores
    }

def procesar_datos_cualitativos(texto):
    """Procesa datos cualitativos"""
    texto = re.sub(r'[^a-zA-ZáéíóúñÁÉÍÓÚÑ\s,]', '', texto)
    categorias = [c.strip() for c in texto.split(',') if c.strip()]
    if len(categorias) < 2:
        return None
    return categorias

def procesar_cuali_frec(texto):
    """Procesa datos cualitativos con frecuencias"""
    pares = [p.strip() for p in texto.split(';') if p.strip()]
    categorias = []
    frecuencias = []

    for par in pares:
        elementos = [e.strip() for e in par.split(',') if e.strip()]
        if len(elementos) == 2:
            cat = elementos[0]
            try:
                freq = float(elementos[1])
                if freq > 0:
                    categorias.append(cat)
                    frecuencias.append(freq)
            except:
                continue

    if len(categorias) < 2:
        return None
    return {'categorias': categorias, 'frecuencias': frecuencias}

def procesar_contingencia(texto, filas, columnas):
    """Procesa tabla de contingencia"""
    if not texto:
        return None

    filas_vec = [f.strip() for f in filas.split(',') if f.strip()]
    columnas_vec = [c.strip() for c in columnas.split(',') if c.strip()]

    if len(filas_vec) < 2 or len(columnas_vec) < 2:
        return None

    pares = [p.strip() for p in texto.split(';') if p.strip()]
    matriz = np.zeros((len(filas_vec), len(columnas_vec)))

    for par in pares:
        elementos = [e.strip() for e in par.split(',') if e.strip()]
        if len(elementos) == 2:
            clave = elementos[0]
            try:
                valor = float(elementos[1])
                if valor >= 0:
                    partes = clave.split('_')
                    if len(partes) == 2:
                        fila = partes[0].strip()
                        columna = partes[1].strip()
                        if fila in filas_vec and columna in columnas_vec:
                            i = filas_vec.index(fila)
                            j = columnas_vec.index(columna)
                            matriz[i, j] = valor
            except:
                continue

    if np.sum(matriz) == 0:
        return None

    return {
        'matriz': matriz,
        'filas': filas_vec,
        'columnas': columnas_vec
    }

def procesar_datos_biv(texto_x, texto_y):
    """Procesa datos bivariados"""
    x = procesar_datos_lista(texto_x)
    y = procesar_datos_lista(texto_y)
    if x is None or y is None or len(x) != len(y) or len(x) < 3:
        return None
    return {'x': x, 'y': y}

# ============================================================
# INTERFAZ DE USUARIO
# ============================================================

st.title("📊 Calculadora Estadística Descriptiva")
st.markdown("---")

# Crear todas las pestañas
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Datos en Lista",
    "📊 Datos Agrupados",
    "🎯 Cualitativas (Datos)",
    "📊 Cualitativas (Frecuencias)",
    "📊 Tabla de Contingencia",
    "🔗 Bivariado (Cuantitativo)",
    "📖 Guía de Interpretación"
])

# ============================================================
# PESTAÑA 1: DATOS EN LISTA
# ============================================================

with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Ingreso de Datos")
        datos_texto = st.text_area(
            "Ingresa los datos separados por comas o espacios",
            placeholder="Ejemplo: 12, 15, 18, 20, 22",
            height=150
        )

        if st.button("🔍 Calcular", key="calcular_lista"):
            datos = procesar_datos_lista(datos_texto)

            if datos is None:
                st.error("❌ Error: Ingresa al menos 2 números válidos.")
            else:
                n = len(datos)
                media = np.mean(datos)
                mediana = np.median(datos)

                # Moda
                valores, conteos = np.unique(datos, return_counts=True)
                moda = valores[conteos == conteos.max()]

                varianza = np.var(datos, ddof=1)
                desviacion = np.std(datos, ddof=1)
                rango = np.max(datos) - np.min(datos)

                q1 = np.percentile(datos, 25)
                q3 = np.percentile(datos, 75)
                iqr = q3 - q1

                outliers_info = detectar_outliers(datos)

                # Crear tabla de frecuencias
                n_clases = int(min(20, max(5, np.ceil(1 + 3.322 * np.log10(n)))))
                hist_data = np.histogram(datos, bins=n_clases)

                clases = []
                for i in range(len(hist_data[0])):
                    if i == 0:
                        clase = f"[{hist_data[1][i]:.2f}, {hist_data[1][i+1]:.2f}]"
                    else:
                        clase = f"({hist_data[1][i]:.2f}, {hist_data[1][i+1]:.2f}]"
                    clases.append(clase)

                marcas_clase = (hist_data[1][:-1] + hist_data[1][1:]) / 2
                freq_acum = np.cumsum(hist_data[0])

                st.session_state['resultados_lista'] = {
                    'datos': datos,
                    'n': n,
                    'media': media,
                    'mediana': mediana,
                    'moda': moda,
                    'varianza': varianza,
                    'desviacion': desviacion,
                    'rango': rango,
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr,
                    'outliers_info': outliers_info,
                    'clases': clases,
                    'frecuencias': hist_data[0].tolist(),
                    'marcas_clase': marcas_clase,
                    'freq_acum': freq_acum.tolist()
                }

                st.success("✅ Cálculo completado exitosamente.")

    with col2:
        if 'resultados_lista' in st.session_state:
            res = st.session_state['resultados_lista']

            st.subheader("📌 Resumen de Resultados")
            col_res1, col_res2 = st.columns(2)

            with col_res1:
                st.metric("Total de datos", res['n'])
                st.metric("Media", f"{res['media']:.4f}")
                st.metric("Mediana", f"{res['mediana']:.4f}")
                st.metric("Moda", f"{', '.join([str(round(m, 2)) for m in res['moda']])}")

            with col_res2:
                st.metric("Varianza", f"{res['varianza']:.4f}")
                st.metric("Desviación Estándar", f"{res['desviacion']:.4f}")
                st.metric("Rango", f"{res['rango']:.4f}")
                st.metric("IQR", f"{res['iqr']:.4f}")

            # Outliers
            if len(res['outliers_info']['outliers']) > 0:
                st.warning(f"⚠️ Outliers detectados: {', '.join([str(round(o, 2)) for o in res['outliers_info']['outliers']])}")
                st.info(f"📊 Límite inferior: {res['outliers_info']['limite_inferior']:.4f} | Límite superior: {res['outliers_info']['limite_superior']:.4f}")

            # Tabla de frecuencias
            st.subheader("📊 Tabla de Frecuencias")
            df_frec = pd.DataFrame({
                'Clase': res['clases'],
                'Frecuencia': res['frecuencias'],
                'Marca': [round(m, 2) for m in res['marcas_clase']],
                'F. Acumulada': res['freq_acum']
            })
            st.dataframe(df_frec, use_container_width=True)

            # Gráficos
            st.subheader("📈 Visualizaciones")

            # Boxplot
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=res['datos'],
                name="Datos",
                boxmean='sd',
                marker_color='#3498db',
                boxpoints='outliers'
            ))
            fig_box.update_layout(title="Diagrama de Caja (Boxplot)", height=350)
            st.plotly_chart(fig_box, use_container_width=True)

            # Histograma con polígono
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=res['datos'],
                nbinsx=len(res['clases']),
                marker_color='#3498db',
                opacity=0.7,
                name="Histograma"
            ))
            # Polígono de frecuencias
            fig_hist.add_trace(go.Scatter(
                x=res['marcas_clase'],
                y=res['frecuencias'],
                mode='lines+markers',
                name='Polígono',
                line=dict(color='#e74c3c', width=2),
                marker=dict(size=6)
            ))
            fig_hist.update_layout(title="Histograma y Polígono de Frecuencias", height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

            # Ojiva
            fig_ojiva = go.Figure()
            fig_ojiva.add_trace(go.Scatter(
                x=res['marcas_clase'],
                y=res['freq_acum'],
                mode='lines+markers',
                name='Ojiva',
                line=dict(color='#2ecc71', width=2),
                marker=dict(size=6)
            ))
            fig_ojiva.update_layout(title="Ojiva (Frecuencias Acumuladas)", height=350)
            st.plotly_chart(fig_ojiva, use_container_width=True)

# ============================================================
# PESTAÑA 2: DATOS AGRUPADOS
# ============================================================

with tab2:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Ingreso de Datos Agrupados")
        st.caption("Formato: clase1, frecuencia1; clase2, frecuencia2; ...")
        datos_agrupados_texto = st.text_area(
            "Ingresa los datos agrupados",
            placeholder="Ejemplo: 10-20, 5; 20-30, 8; 30-40, 12",
            height=150
        )

        if st.button("🔍 Calcular", key="calcular_agrupados"):
            resultado = procesar_datos_agrupados(datos_agrupados_texto)

            if resultado is None:
                st.error("❌ Error: Ingresa al menos 2 clases con frecuencias válidas.")
            else:
                clases = resultado['clases']
                frecuencias = resultado['frecuencias']
                marcas_clase = resultado['marcas_clase']
                limites_inferiores = resultado['limites_inferiores']
                limites_superiores = resultado['limites_superiores']

                n = sum(frecuencias)

                # Media
                media = np.average(marcas_clase, weights=frecuencias)

                # Mediana
                n_mitad = n / 2
                acumulado = 0
                clase_mediana = 0
                for i, f in enumerate(frecuencias):
                    acumulado += f
                    if acumulado >= n_mitad:
                        clase_mediana = i
                        break

                if clase_mediana > 0:
                    f_anterior = sum(frecuencias[:clase_mediana])
                else:
                    f_anterior = 0

                f_clase = frecuencias[clase_mediana]
                L_mediana = limites_inferiores[clase_mediana]
                A_mediana = limites_superiores[clase_mediana] - limites_inferiores[clase_mediana]
                mediana = L_mediana + ((n_mitad - f_anterior) / f_clase) * A_mediana

                # Moda
                clase_modal = np.argmax(frecuencias)
                f_modal = frecuencias[clase_modal]
                f_pre = frecuencias[clase_modal - 1] if clase_modal > 0 else 0
                f_post = frecuencias[clase_modal + 1] if clase_modal < len(frecuencias) - 1 else 0
                L_modal = limites_inferiores[clase_modal]
                A_modal = limites_superiores[clase_modal] - limites_inferiores[clase_modal]

                d1 = f_modal - f_pre
                d2 = f_modal - f_post
                if d1 + d2 > 0:
                    moda = L_modal + (d1 / (d1 + d2)) * A_modal
                else:
                    moda = marcas_clase[clase_modal]

                # Varianza y desviación
                varianza = np.average((np.array(marcas_clase) - media)**2, weights=frecuencias) * n / (n - 1)
                desviacion = np.sqrt(varianza)

                # Cuantiles
                def calcular_cuantil(p):
                    pos = p * n
                    acumulado = 0
                    clase_cuantil = 0
                    for i, f in enumerate(frecuencias):
                        acumulado += f
                        if acumulado >= pos:
                            clase_cuantil = i
                            break

                    if clase_cuantil > 0:
                        f_ant = sum(frecuencias[:clase_cuantil])
                    else:
                        f_ant = 0

                    f_clase = frecuencias[clase_cuantil]
                    L_cuantil = limites_inferiores[clase_cuantil]
                    A_cuantil = limites_superiores[clase_cuantil] - limites_inferiores[clase_cuantil]
                    return L_cuantil + ((pos - f_ant) / f_clase) * A_cuantil

                q1 = calcular_cuantil(0.25)
                q3 = calcular_cuantil(0.75)
                iqr = q3 - q1

                limite_inf = q1 - 1.5 * iqr
                limite_sup = q3 + 1.5 * iqr

                # Frecuencias relativas y acumuladas
                freq_rel = [f/n*100 for f in frecuencias]
                freq_acum = np.cumsum(frecuencias)
                freq_rel_acum = np.cumsum(freq_rel)

                st.session_state['resultados_agrupados'] = {
                    'clases': clases,
                    'frecuencias': frecuencias,
                    'marcas_clase': marcas_clase,
                    'limites_inferiores': limites_inferiores,
                    'limites_superiores': limites_superiores,
                    'n': n,
                    'media': media,
                    'mediana': mediana,
                    'moda': moda,
                    'varianza': varianza,
                    'desviacion': desviacion,
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr,
                    'limite_inf': limite_inf,
                    'limite_sup': limite_sup,
                    'clase_modal': clase_modal,
                    'freq_rel': freq_rel,
                    'freq_acum': freq_acum.tolist(),
                    'freq_rel_acum': freq_rel_acum.tolist()
                }

                st.success(f"✅ Cálculo completado. Total de datos: {n}")

    with col2:
        if 'resultados_agrupados' in st.session_state:
            res = st.session_state['resultados_agrupados']

            st.subheader("📌 Resumen de Resultados")
            col_res1, col_res2 = st.columns(2)

            with col_res1:
                st.metric("Total de datos", res['n'])
                st.metric("Media", f"{res['media']:.4f}")
                st.metric("Mediana", f"{res['mediana']:.4f}")
                st.metric("Moda", f"{res['moda']:.4f}")

            with col_res2:
                st.metric("Varianza", f"{res['varianza']:.4f}")
                st.metric("Desviación Estándar", f"{res['desviacion']:.4f}")
                st.metric("Q1", f"{res['q1']:.4f}")
                st.metric("Q3", f"{res['q3']:.4f}")

            # Tabla de frecuencias
            st.subheader("📊 Tabla de Frecuencias")
            df_frec = pd.DataFrame({
                'Clase': res['clases'],
                'Marca': [round(m, 2) for m in res['marcas_clase']],
                'f': res['frecuencias'],
                'fr(%)': [round(f, 2) for f in res['freq_rel']],
                'F': res['freq_acum'],
                'F(%)': [round(f, 2) for f in res['freq_rel_acum']]
            })
            st.dataframe(df_frec, use_container_width=True)

            # Gráficos
            st.subheader("📈 Visualizaciones")

            # Histograma
            fig_hist = px.bar(
                df_frec,
                x='Clase',
                y='f',
                title='Histograma de Frecuencias',
                color_discrete_sequence=['#3498db']
            )
            fig_hist.update_layout(height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

            # Polígono y Ojiva
            fig_pol_ojiva = make_subplots(rows=1, cols=2, subplot_titles=('Polígono', 'Ojiva'))

            fig_pol_ojiva.add_trace(
                go.Scatter(
                    x=res['marcas_clase'],
                    y=res['frecuencias'],
                    mode='lines+markers',
                    name='Polígono',
                    line=dict(color='#e74c3c', width=2)
                ),
                row=1, col=1
            )

            fig_pol_ojiva.add_trace(
                go.Scatter(
                    x=res['marcas_clase'],
                    y=res['freq_acum'],
                    mode='lines+markers',
                    name='Ojiva',
                    line=dict(color='#2ecc71', width=2)
                ),
                row=1, col=2
            )

            fig_pol_ojiva.update_layout(height=400)
            st.plotly_chart(fig_pol_ojiva, use_container_width=True)

# ============================================================
# PESTAÑA 3: CUALITATIVAS (DATOS)
# ============================================================

with tab3:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🎯 Ingreso de Datos Cualitativos")
        st.caption("Ingresa las categorías separadas por comas")
        datos_cuali_texto = st.text_area(
            "Datos cualitativos",
            placeholder="Ejemplo: Rojo, Azul, Rojo, Verde, Azul, Azul",
            height=150
        )

        if st.button("🔍 Calcular", key="calcular_cuali"):
            categorias = procesar_datos_cualitativos(datos_cuali_texto)

            if categorias is None:
                st.error("❌ Error: Ingresa al menos 2 categorías válidas.")
            else:
                valores, conteos = np.unique(categorias, return_counts=True)
                n = len(categorias)
                porcentajes = conteos / n * 100

                # Moda
                max_freq = max(conteos)
                modas = valores[conteos == max_freq]

                st.session_state['resultados_cuali'] = {
                    'nombres': valores.tolist(),
                    'frecuencias': conteos.tolist(),
                    'porcentajes': porcentajes.tolist(),
                    'n': n,
                    'modas': modas.tolist(),
                    'max_freq': max_freq
                }

                st.success(f"✅ Cálculo completado. Total de datos: {n}")

    with col2:
        if 'resultados_cuali' in st.session_state:
            res = st.session_state['resultados_cuali']

            st.subheader("📌 Resumen de Resultados")
            col_res1, col_res2 = st.columns(2)

            with col_res1:
                st.metric("Total de datos", res['n'])
                st.metric("Número de categorías", len(res['nombres']))

            with col_res2:
                modas_str = ", ".join(res['modas'])
                st.metric("Moda", f"{modas_str} (f: {res['max_freq']})")

            # Tabla de frecuencias
            st.subheader("📊 Tabla de Frecuencias")
            df_cuali = pd.DataFrame({
                'Categoría': res['nombres'],
                'Frecuencia': res['frecuencias'],
                'Porcentaje': [round(p, 2) for p in res['porcentajes']]
            })
            st.dataframe(df_cuali, use_container_width=True)

            # Gráficos
            st.subheader("📈 Visualizaciones")

            col_graf1, col_graf2 = st.columns(2)

            with col_graf1:
                # Gráfico de barras
                fig_bar = px.bar(
                    df_cuali,
                    x='Categoría',
                    y='Frecuencia',
                    color='Categoría',
                    text=[f"{f} ({round(p, 1)}%)" for f, p in zip(res['frecuencias'], res['porcentajes'])],
                    title='Gráfico de Barras'
                )
                fig_bar.update_traces(textposition='outside')
                fig_bar.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_graf2:
                # Gráfico circular
                fig_pie = px.pie(
                    df_cuali,
                    values='Frecuencia',
                    names='Categoría',
                    title='Gráfico Circular',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textinfo='label+percent')
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)

# ============================================================
# PESTAÑA 4: CUALITATIVAS (FRECUENCIAS)
# ============================================================

with tab4:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Ingreso de Categorías con Frecuencias")
        st.caption("Formato: categoría1, frecuencia1; categoría2, frecuencia2; ...")
        datos_cuali_frec = st.text_area(
            "Datos con frecuencias",
            placeholder="Ejemplo: Rojo, 10; Verde, 35; Azul, 55",
            height=150
        )

        if st.button("🔍 Calcular", key="calcular_cuali_frec"):
            resultado = procesar_cuali_frec(datos_cuali_frec)

            if resultado is None:
                st.error("❌ Error: Ingresa al menos 2 categorías con frecuencias válidas.")
            else:
                n = sum(resultado['frecuencias'])
                porcentajes = [f/n*100 for f in resultado['frecuencias']]

                # Moda
                max_freq = max(resultado['frecuencias'])
                modas = [resultado['categorias'][i] for i, f in enumerate(resultado['frecuencias']) if f == max_freq]

                st.session_state['resultados_cuali_frec'] = {
                    'categorias': resultado['categorias'],
                    'frecuencias': resultado['frecuencias'],
                    'porcentajes': porcentajes,
                    'n': n,
                    'modas': modas,
                    'max_freq': max_freq
                }

                st.success(f"✅ Cálculo completado. Total de datos: {n}")

    with col2:
        if 'resultados_cuali_frec' in st.session_state:
            res = st.session_state['resultados_cuali_frec']

            st.subheader("📌 Resumen de Resultados")
            col_res1, col_res2 = st.columns(2)

            with col_res1:
                st.metric("Total de datos", res['n'])
                st.metric("Número de categorías", len(res['categorias']))

            with col_res2:
                modas_str = ", ".join(res['modas'])
                st.metric("Moda", f"{modas_str} (f: {res['max_freq']})")

            # Tabla de frecuencias
            st.subheader("📊 Tabla de Frecuencias")
            df_cuali_frec = pd.DataFrame({
                'Categoría': res['categorias'],
                'Frecuencia': res['frecuencias'],
                'Porcentaje': [round(p, 2) for p in res['porcentajes']]
            })
            st.dataframe(df_cuali_frec, use_container_width=True)

            # Gráficos
            st.subheader("📈 Visualizaciones")

            col_graf1, col_graf2 = st.columns(2)

            with col_graf1:
                fig_bar = px.bar(
                    df_cuali_frec,
                    x='Categoría',
                    y='Frecuencia',
                    color='Categoría',
                    text=[f"{f} ({round(p, 1)}%)" for f, p in zip(res['frecuencias'], res['porcentajes'])],
                    title='Gráfico de Barras'
                )
                fig_bar.update_traces(textposition='outside')
                fig_bar.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_graf2:
                fig_pie = px.pie(
                    df_cuali_frec,
                    values='Frecuencia',
                    names='Categoría',
                    title='Gráfico Circular',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textinfo='label+percent')
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)

# ============================================================
# PESTAÑA 5: TABLA DE CONTINGENCIA
# ============================================================

with tab5:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Ingreso de Tabla de Contingencia")
        st.caption("Formato: fila1_col1, valor; fila1_col2, valor; ...")

        datos_contingencia = st.text_area(
            "Datos de la tabla",
            placeholder='Ejemplo: Hombre_Matemáticas, 15; Hombre_Ingeniería, 30; Mujer_Matemáticas, 15; Mujer_Ingeniería, 20',
            height=150
        )

        nombres_filas = st.text_input("Nombres de filas (separados por comas):", value="Hombre, Mujer")
        nombres_columnas = st.text_input("Nombres de columnas (separados por comas):", value="Matemáticas, Ingeniería")

        if st.button("🔍 Calcular", key="calcular_contingencia"):
            resultado = procesar_contingencia(datos_contingencia, nombres_filas, nombres_columnas)

            if resultado is None:
                st.error("❌ Error: Verifica el formato de la tabla de contingencia.")
            else:
                st.session_state['resultados_contingencia'] = resultado
                st.success("✅ Tabla de contingencia calculada exitosamente.")

    with col2:
        if 'resultados_contingencia' in st.session_state:
            res = st.session_state['resultados_contingencia']

            st.subheader("📌 Resumen")
            col_res1, col_res2, col_res3 = st.columns(3)

            with col_res1:
                st.metric("Filas", len(res['filas']))
            with col_res2:
                st.metric("Columnas", len(res['columnas']))
            with col_res3:
                st.metric("Total", int(np.sum(res['matriz'])))

            # Tabla con totales
            df_cont = pd.DataFrame(res['matriz'], index=res['filas'], columns=res['columnas'])
            df_cont['Total Fila'] = df_cont.sum(axis=1)
            df_cont.loc['Total Columna'] = df_cont.sum()

            st.dataframe(df_cont, use_container_width=True)

            # Gráficos
            st.subheader("📈 Visualizaciones")

            # Datos para gráficos
            df_long = pd.DataFrame(res['matriz'], index=res['filas'], columns=res['columnas'])
            df_long = df_long.reset_index().melt(id_vars='index', var_name='Columna', value_name='Frecuencia')
            df_long.columns = ['Fila', 'Columna', 'Frecuencia']

            # Gráficos en pestañas
            graf_tab1, graf_tab2, graf_tab3, graf_tab4 = st.tabs([
                "Barras Agrupadas", "Barras Apiladas", "Porcentual", "Mapa de Calor"
            ])

            with graf_tab1:
                fig_grouped = px.bar(
                    df_long,
                    x='Fila',
                    y='Frecuencia',
                    color='Columna',
                    barmode='group',
                    text='Frecuencia',
                    title='Barras Agrupadas',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_grouped.update_traces(textposition='outside')
                fig_grouped.update_layout(height=400)
                st.plotly_chart(fig_grouped, use_container_width=True)

            with graf_tab2:
                fig_stacked = px.bar(
                    df_long,
                    x='Fila',
                    y='Frecuencia',
                    color='Columna',
                    barmode='stack',
                    text='Frecuencia',
                    title='Barras Apiladas',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_stacked.update_traces(textposition='inside')
                fig_stacked.update_layout(height=400)
                st.plotly_chart(fig_stacked, use_container_width=True)

            with graf_tab3:
                # Calcular porcentajes por fila
                df_pct = df_long.copy()
                for fila in res['filas']:
                    mask = df_pct['Fila'] == fila
                    total = df_pct[mask]['Frecuencia'].sum()
                    if total > 0:
                        df_pct.loc[mask, 'Frecuencia'] = df_pct.loc[mask, 'Frecuencia'] / total * 100

                fig_pct = px.bar(
                    df_pct,
                    x='Fila',
                    y='Frecuencia',
                    color='Columna',
                    barmode='stack',
                    text=[f"{round(v, 1)}%" for v in df_pct['Frecuencia']],
                    title='Barras Apiladas (Porcentajes)',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_pct.update_traces(textposition='inside')
                fig_pct.update_layout(height=400, yaxis_title="Porcentaje (%)")
                st.plotly_chart(fig_pct, use_container_width=True)

            with graf_tab4:
                fig_heatmap = px.imshow(
                    res['matriz'],
                    x=res['columnas'],
                    y=res['filas'],
                    text_auto=True,
                    aspect="auto",
                    title="Mapa de Calor",
                    color_continuous_scale='Blues'
                )
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)

# ============================================================
# PESTAÑA 6: BIVARIADO (CUANTITATIVO)
# ============================================================

with tab6:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Ingreso de Datos Bivariados")

        datos_x = st.text_area(
            "Variable 1 (X) - Cuantitativa:",
            placeholder="Ejemplo: 12, 15, 18, 20, 22",
            height=80
        )

        datos_y = st.text_area(
            "Variable 2 (Y) - Cuantitativa:",
            placeholder="Ejemplo: 25, 30, 35, 40, 45",
            height=80
        )

        if st.button("🔍 Calcular", key="calcular_biv"):
            resultado = procesar_datos_biv(datos_x, datos_y)

            if resultado is None:
                st.error("❌ Error: Ambas variables deben tener al menos 3 valores y la misma longitud.")
            else:
                x = resultado['x']
                y = resultado['y']
                n = len(x)

                # Estadísticas
                media_x = np.mean(x)
                media_y = np.mean(y)
                var_x = np.var(x, ddof=1)
                var_y = np.var(y, ddof=1)
                sd_x = np.std(x, ddof=1)
                sd_y = np.std(y, ddof=1)

                # Correlación
                cor_pearson, p_value = stats.pearsonr(x, y)

                # Regresión
                slope, intercept, r_value, p_value_reg, std_err = stats.linregress(x, y)
                r2 = r_value**2
                r2_adj = 1 - (1 - r2) * (n - 1) / (n - 2)

                # Predicciones
                x_pred = np.linspace(min(x) - 0.1*(max(x)-min(x)), max(x) + 0.1*(max(x)-min(x)), 100)
                y_pred = intercept + slope * x_pred

                # Residuos
                y_pred_train = intercept + slope * x
                residuos = y - y_pred_train

                st.session_state['resultados_biv'] = {
                    'x': x,
                    'y': y,
                    'n': n,
                    'media_x': media_x,
                    'media_y': media_y,
                    'var_x': var_x,
                    'var_y': var_y,
                    'sd_x': sd_x,
                    'sd_y': sd_y,
                    'cor_pearson': cor_pearson,
                    'p_value': p_value,
                    'intercepto': intercept,
                    'pendiente': slope,
                    'r2': r2,
                    'r2_adj': r2_adj,
                    'x_pred': x_pred,
                    'y_pred': y_pred,
                    'residuos': residuos
                }

                st.success("✅ Cálculo bivariado completado exitosamente.")

    with col2:
        if 'resultados_biv' in st.session_state:
            res = st.session_state['resultados_biv']

            st.subheader("📌 Resumen de Resultados")

            col_res1, col_res2 = st.columns(2)

            with col_res1:
                st.metric("Tamaño de muestra", res['n'])
                st.metric("Media X", f"{res['media_x']:.4f}")
                st.metric("Media Y", f"{res['media_y']:.4f}")
                st.metric("Correlación (r)", f"{res['cor_pearson']:.4f}")

            with col_res2:
                st.metric("Var X", f"{res['var_x']:.4f}")
                st.metric("Var Y", f"{res['var_y']:.4f}")
                st.metric("R²", f"{res['r2']:.4f}")
                st.metric("R² ajustado", f"{res['r2_adj']:.4f}")

            # Interpretación de la correlación
            st.subheader("📊 Interpretación")

            r_abs = abs(res['cor_pearson'])
            if r_abs >= 0.9:
                fuerza = "Muy fuerte"
            elif r_abs >= 0.7:
                fuerza = "Fuerte"
            elif r_abs >= 0.5:
                fuerza = "Moderada"
            elif r_abs >= 0.3:
                fuerza = "Débil"
            else:
                fuerza = "Muy débil"

            direccion = "positiva" if res['cor_pearson'] > 0 else "negativa" if res['cor_pearson'] < 0 else "nula"

            col_int1, col_int2 = st.columns(2)
            with col_int1:
                st.info(f"**Correlación {fuerza} ({direccion})**")
            with col_int2:
                if res['p_value'] < 0.05:
                    st.success("✅ Estadísticamente significativa (p < 0.05)")
                else:
                    st.warning("❌ No significativa (p ≥ 0.05)")

            # Datos
            st.subheader("📊 Datos")
            df_biv = pd.DataFrame({'X': res['x'], 'Y': res['y']})
            st.dataframe(df_biv, use_container_width=True)

            # Gráficos
            st.subheader("📈 Visualizaciones")

            graf_biv1, graf_biv2, graf_biv3 = st.tabs([
                "Dispersión + Regresión", "Correlación", "Residuos"
            ])

            with graf_biv1:
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=res['x'],
                    y=res['y'],
                    mode='markers',
                    name='Datos',
                    marker=dict(color='#3498db', size=10)
                ))
                fig_scatter.add_trace(go.Scatter(
                    x=res['x_pred'],
                    y=res['y_pred'],
                    mode='lines',
                    name='Regresión',
                    line=dict(color='#e74c3c', width=2)
                ))
                fig_scatter.update_layout(
                    title=f"Y = {res['intercepto']:.4f} + {res['pendiente']:.4f} * X",
                    xaxis_title="Variable X",
                    yaxis_title="Variable Y",
                    height=450
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            with graf_biv2:
                # Matriz de correlación
                matriz_cor = np.array([[1, res['cor_pearson']], [res['cor_pearson'], 1]])
                fig_cor = px.imshow(
                    matriz_cor,
                    x=['X', 'Y'],
                    y=['X', 'Y'],
                    text_auto=True,
                    aspect="auto",
                    title="Matriz de Correlación",
                    color_continuous_scale='RdBu',
                    zmin=-1,
                    zmax=1
                )
                fig_cor.update_layout(height=450)
                st.plotly_chart(fig_cor, use_container_width=True)

            with graf_biv3:
                fig_residuos = go.Figure()
                fig_residuos.add_trace(go.Scatter(
                    x=res['x'],
                    y=res['residuos'],
                    mode='markers',
                    name='Residuos',
                    marker=dict(color='#e74c3c', size=10)
                ))
                fig_residuos.add_hline(y=0, line_dash="dash", line_color="black")
                fig_residuos.update_layout(
                    title="Gráfico de Residuos",
                    xaxis_title="Variable X",
                    yaxis_title="Residuos",
                    height=450
                )
                st.plotly_chart(fig_residuos, use_container_width=True)

# ============================================================
# PESTAÑA 7: GUÍA DE INTERPRETACIÓN
# ============================================================

with tab7:
    st.markdown("""
    # 📖 Guía de Interpretación de Medidas Estadísticas

    ## 📈 Medidas de Tendencia Central

    ### Media (Promedio)
    - Es el promedio aritmético de todos los datos
    - Representa el centro de masa de la distribución
    - **Sensible a outliers**: Valores extremos pueden distorsionarla

    ### Mediana
    - Valor que divide los datos en dos partes iguales (50% cada una)
    - **Robusta a outliers**: No se ve afectada por valores extremos
    - Ideal para distribuciones asimétricas

    ### Moda
    - Valor que aparece con mayor frecuencia
    - Puede haber más de una moda (bimodal, multimodal)
    - Útil para datos categóricos

    ## 📊 Medidas de Dispersión

    ### Varianza
    - Mide la dispersión promedio al cuadrado
    - **Sensible a outliers**

    ### Desviación Estándar
    - Raíz cuadrada de la varianza
    - Misma unidad que los datos originales
    - Más interpretable que la varianza

    ### Rango
    - Diferencia entre el máximo y el mínimo
    - **Muy sensible a outliers**

    ### Rango Intercuartil (IQR)
    - Q3 - Q1 (diferencia entre el percentil 75 y 25)
    - **Robusto a outliers**
    - Contiene el 50% central de los datos

    ## 📈 Diagrama de Caja (Boxplot)

    - **Caja**: Contiene el 50% central de los datos (Q1 a Q3)
    - **Línea en la caja**: Mediana (Q2)
    - **Bigotes**: Se extienden hasta 1.5 × IQR
    - **Puntos fuera**: Outliers (valores atípicos)

    ## 🔍 Interpretación de Outliers

    - Un outlier es un valor que se aleja significativamente del resto
    - Se detecta cuando está fuera de: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]

    ## 📊 Tabla de Contingencia

    - **Barras Agrupadas**: Compara categorías lado a lado
    - **Barras Apiladas**: Muestra composición dentro de cada categoría
    - **Barras Porcentuales**: Muestra distribución porcentual
    - **Mapa de Calor**: Visualización de intensidad de frecuencias

    ## 🔗 Análisis Bivariado (Cuantitativo)

    ### Coeficiente de Correlación de Pearson (r)

    | Valor de r | Interpretación |
    |------------|----------------|
    | 0.90 - 1.00 | Correlación muy fuerte |
    | 0.70 - 0.89 | Correlación fuerte |
    | 0.50 - 0.69 | Correlación moderada |
    | 0.30 - 0.49 | Correlación débil |
    | 0.00 - 0.29 | Correlación muy débil |

    ### Regresión Lineal

    - **Ecuación**: Y = β₀ + β₁X
    - **β₀ (Intercepto)**: Valor de Y cuando X = 0
    - **β₁ (Pendiente)**: Cambio en Y por cada unidad de X
    - **R²**: Proporción de varianza explicada por el modelo

    ### Significancia Estadística

    - **p-value < 0.05**: Resultado estadísticamente significativo
    - **p-value ≥ 0.05**: No hay evidencia suficiente para rechazar H₀
    """)
