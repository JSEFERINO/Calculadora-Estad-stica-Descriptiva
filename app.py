import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import re
import requests
import io
import csv

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
# INICIALIZAR SESSION STATE
# ============================================================
if 'df_cargado' not in st.session_state:
    st.session_state.df_cargado = None
if 'columna_seleccionada' not in st.session_state:
    st.session_state.columna_seleccionada = None
if 'columna_cat_seleccionada' not in st.session_state:
    st.session_state.columna_cat_seleccionada = None

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def calcular_moda(datos):
    """Calcula la moda de manera robusta"""
    try:
        if isinstance(datos, np.ndarray):
            valores, conteos = np.unique(datos, return_counts=True)
            if len(valores) == 0:
                return np.nan
            max_count = np.max(conteos)
            if max_count == 1:
                return np.nan
            modas = valores[conteos == max_count]
            if len(modas) == 1:
                return modas[0]
            else:
                return modas
        else:
            from collections import Counter
            counter = Counter(datos)
            if not counter:
                return np.nan
            max_count = max(counter.values())
            if max_count == 1:
                return np.nan
            modas = [k for k, v in counter.items() if v == max_count]
            if len(modas) == 1:
                return modas[0]
            return modas
    except Exception as e:
        return np.nan

def detectar_separador(texto):
    """Detecta automáticamente el separador en un texto CSV"""
    lineas = [line.strip() for line in texto.split('\n') if line.strip()]
    if not lineas:
        return ','

    separadores = [',', ';', '\t', '|', ' ']
    conteos = {}

    for sep in separadores:
        conteos[sep] = sum(linea.count(sep) for linea in lineas)

    mejor_sep = max(conteos, key=conteos.get)
    return mejor_sep if conteos[mejor_sep] > 0 else ','

def cargar_datos_desde_csv(archivo, separador=None):
    """Carga datos desde un archivo CSV subido con manejo de errores"""
    try:
        contenido = archivo.getvalue().decode('utf-8', errors='ignore')

        if separador is None:
            separador = detectar_separador(contenido)

        try:
            df = pd.read_csv(io.StringIO(contenido), sep=separador)
        except:
            for sep in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(io.StringIO(contenido), sep=sep)
                    if len(df.columns) > 1:
                        separador = sep
                        break
                except:
                    continue
            else:
                df = pd.read_csv(io.StringIO(contenido), sep=separador, engine='python')

        df.columns = [col.strip().replace(' ', '_').replace(';', '').replace(',', '') for col in df.columns]

        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        return df
    except Exception as e:
        st.error(f"❌ Error al cargar CSV: {str(e)}")
        return None

def cargar_datos_desde_url(url, separador=None):
    """Carga datos desde una URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text

        if separador is None:
            separador = detectar_separador(content)

        try:
            df = pd.read_csv(io.StringIO(content), sep=separador)
        except:
            for sep in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=sep)
                    if len(df.columns) > 1:
                        break
                except:
                    continue
            else:
                df = pd.read_csv(io.StringIO(content), sep=separador, engine='python')

        df.columns = [col.strip().replace(' ', '_').replace(';', '').replace(',', '') for col in df.columns]

        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        return df
    except Exception as e:
        st.error(f"❌ Error al cargar desde URL: {str(e)}")
        return None

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

def identificar_distribucion(datos):
    """Identifica la distribución de los datos usando pruebas estadísticas"""
    # Inicializar con valores por defecto
    resultado_base = {
        "distribucion": "Datos insuficientes",
        "distribuciones_posibles": [],
        "pruebas": {},
        "skewness": 0,
        "kurtosis": 0
    }
    
    if len(datos) < 8:
        return resultado_base

    resultados = {}

    try:
        shapiro_stat, shapiro_p = stats.shapiro(datos)
        resultados['Shapiro-Wilk'] = {
            'estadistico': shapiro_stat,
            'p_valor': shapiro_p,
            'es_normal': shapiro_p > 0.05
        }
    except:
        resultados['Shapiro-Wilk'] = {'estadistico': None, 'p_valor': None, 'es_normal': False}

    try:
        ks_stat, ks_p = stats.kstest(datos, 'norm', args=(np.mean(datos), np.std(datos)))
        resultados['Kolmogorov-Smirnov'] = {
            'estadistico': ks_stat,
            'p_valor': ks_p,
            'es_normal': ks_p > 0.05
        }
    except:
        resultados['Kolmogorov-Smirnov'] = {'estadistico': None, 'p_valor': None, 'es_normal': False}

    try:
        anderson_result = stats.anderson(datos, dist='norm')
        resultados['Anderson-Darling'] = {
            'estadistico': anderson_result.statistic,
            'es_normal': anderson_result.statistic < anderson_result.critical_values[2]
        }
    except:
        resultados['Anderson-Darling'] = {'estadistico': None, 'es_normal': False}

    distribuciones_posibles = []

    es_normal = resultados['Shapiro-Wilk']['es_normal'] if resultados['Shapiro-Wilk']['p_valor'] is not None else False
    if es_normal:
        distribuciones_posibles.append("Normal")

    try:
        uni_stat, uni_p = stats.kstest(datos, 'uniform', args=(min(datos), max(datos)-min(datos)))
        if uni_p > 0.05:
            distribuciones_posibles.append("Uniforme")
    except:
        pass

    try:
        exp_stat, exp_p = stats.kstest(datos, 'expon', args=(0, np.mean(datos)))
        if exp_p > 0.05:
            distribuciones_posibles.append("Exponencial")
    except:
        pass

    # Calcular skewness y kurtosis siempre
    try:
        skewness = stats.skew(datos) if len(datos) > 2 else 0
    except:
        skewness = 0
    
    try:
        kurtosis = stats.kurtosis(datos) if len(datos) > 3 else 0
    except:
        kurtosis = 0

    if not distribuciones_posibles:
        if abs(skewness) > 1:
            distribuciones_posibles.append("Asimétrica (sesgo significativo)")
        elif abs(kurtosis) > 3:
            distribuciones_posibles.append("Con colas pesadas (leptocúrtica)")
        else:
            distribuciones_posibles.append("Sin distribución clara")

    return {
        "distribucion": distribuciones_posibles[0] if distribuciones_posibles else "No determinada",
        "distribuciones_posibles": distribuciones_posibles,
        "pruebas": resultados,
        "skewness": skewness,
        "kurtosis": kurtosis
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

def procesar_datos_biv(texto_x, texto_y):
    """Procesa datos bivariados"""
    x = procesar_datos_lista(texto_x)
    y = procesar_datos_lista(texto_y)
    if x is None or y is None or len(x) != len(y) or len(x) < 3:
        return None
    return {'x': x, 'y': y}

# ============================================================
# INTERFAZ DE USUARIO - BARRA LATERAL
# ============================================================

st.sidebar.header("📥 Carga de Datos")

tipo_carga = st.sidebar.radio(
    "Selecciona el método de carga:",
    ["✏️ Ingresar manualmente", "📁 Subir archivo CSV", "🔗 Desde URL"]
)

df_cargado = None
nombre_archivo = ""

if tipo_carga == "📁 Subir archivo CSV":
    archivo = st.sidebar.file_uploader("Selecciona un archivo CSV", type=['csv', 'txt'])
    if archivo is not None:
        with st.sidebar.expander("⚙️ Opciones avanzadas"):
            separador_manual = st.selectbox(
                "Separador:",
                ['Auto-detectar', ',', ';', '\t', '|', ' ']
            )

        if st.sidebar.button("📊 Cargar archivo"):
            with st.spinner("Cargando archivo..."):
                sep = None if separador_manual == 'Auto-detectar' else separador_manual
                df_cargado = cargar_datos_desde_csv(archivo, sep)
                if df_cargado is not None:
                    st.session_state.df_cargado = df_cargado
                    nombre_archivo = archivo.name
                    st.sidebar.success(f"✅ Archivo cargado: {nombre_archivo}")
                    st.sidebar.info(f"📊 {len(df_cargado)} filas, {len(df_cargado.columns)} columnas")

                    with st.sidebar.expander("📋 Vista previa"):
                        st.dataframe(df_cargado.head())

elif tipo_carga == "🔗 Desde URL":
    url = st.sidebar.text_input("URL del archivo CSV:",
                                placeholder="https://ejemplo.com/datos.csv")
    separador_url = st.sidebar.selectbox("Separador:", ['Auto-detectar', ',', ';', '\t', '|'], index=0)
    if url and st.sidebar.button("📊 Cargar desde URL"):
        with st.spinner("Cargando datos desde URL..."):
            sep = None if separador_url == 'Auto-detectar' else separador_url
            df_cargado = cargar_datos_desde_url(url, sep)
            if df_cargado is not None:
                st.session_state.df_cargado = df_cargado
                nombre_archivo = url.split('/')[-1]
                st.sidebar.success("✅ Datos cargados desde URL")
                st.sidebar.info(f"📊 {len(df_cargado)} filas, {len(df_cargado.columns)} columnas")

                with st.sidebar.expander("📋 Vista previa"):
                    st.dataframe(df_cargado.head())

# Usar el DataFrame de session_state si existe
if st.session_state.df_cargado is not None:
    df_cargado = st.session_state.df_cargado

# ============================================================
# FUNCIONES PARA ANÁLISIS DESCRIPTIVO DE CSV
# ============================================================

def mostrar_analisis_descriptivo(df):
    """Muestra análisis descriptivo completo de un DataFrame"""
    st.subheader("📊 Análisis Descriptivo del DataFrame")

    # Información general
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Filas", len(df))
    with col_info2:
        st.metric("Columnas", len(df.columns))
    with col_info3:
        st.metric("Memoria", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    # Tipos de datos
    st.subheader("📋 Tipos de Datos")
    tipos_df = pd.DataFrame({
        'Columna': df.columns,
        'Tipo': df.dtypes.astype(str),
        'Valores Nulos': df.isnull().sum().values,
        'Porcentaje Nulos': (df.isnull().sum() / len(df) * 100).round(2).values
    })
    st.dataframe(tipos_df, use_container_width=True)

    # Selección de columnas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if columnas_numericas:
        st.subheader("📊 Análisis de Variables Numéricas")

        if st.session_state.columna_seleccionada is None or st.session_state.columna_seleccionada not in columnas_numericas:
            st.session_state.columna_seleccionada = columnas_numericas[0]

        columna_seleccionada = st.selectbox(
            "Selecciona una variable numérica para analizar:",
            columnas_numericas,
            index=columnas_numericas.index(st.session_state.columna_seleccionada) if st.session_state.columna_seleccionada in columnas_numericas else 0,
            key="selector_numerico"
        )
        st.session_state.columna_seleccionada = columna_seleccionada

        if columna_seleccionada:
            datos = df[columna_seleccionada].dropna()

            if len(datos) > 0:
                moda_valor = calcular_moda(datos)
                if isinstance(moda_valor, (np.ndarray, list)):
                    moda_str = ', '.join([str(round(float(m), 4)) if isinstance(m, (int, float)) else str(m) for m in moda_valor])
                else:
                    moda_str = f"{moda_valor:.4f}" if isinstance(moda_valor, (int, float)) else str(moda_valor)

                stats_df = pd.DataFrame({
                    'Estadística': ['N', 'Media', 'Mediana', 'Moda', 'Varianza', 'Desviación',
                                  'Mínimo', 'Máximo', 'Rango', 'Q1', 'Q3', 'IQR'],
                    'Valor': [
                        len(datos),
                        f"{np.mean(datos):.4f}",
                        f"{np.median(datos):.4f}",
                        moda_str,
                        f"{np.var(datos, ddof=1):.4f}",
                        f"{np.std(datos, ddof=1):.4f}",
                        f"{np.min(datos):.4f}",
                        f"{np.max(datos):.4f}",
                        f"{np.max(datos) - np.min(datos):.4f}",
                        f"{np.percentile(datos, 25):.4f}",
                        f"{np.percentile(datos, 75):.4f}",
                        f"{np.percentile(datos, 75) - np.percentile(datos, 25):.4f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True)

                if len(datos) > 2:
                    skewness = stats.skew(datos)
                    kurtosis = stats.kurtosis(datos)

                    col_skew1, col_skew2 = st.columns(2)
                    with col_skew1:
                        st.metric("Asimetría", f"{skewness:.4f}")
                        if skewness > 0:
                            st.caption("📈 Sesgo positivo (cola a la derecha)")
                        elif skewness < 0:
                            st.caption("📉 Sesgo negativo (cola a la izquierda)")
                        else:
                            st.caption("📊 Simétrica")
                    with col_skew2:
                        st.metric("Curtosis", f"{kurtosis:.4f}")
                        if kurtosis > 3:
                            st.caption("📊 Leptocúrtica (colas pesadas)")
                        elif kurtosis < 3:
                            st.caption("📊 Platicúrtica (colas ligeras)")
                        else:
                            st.caption("📊 Mesocúrtica (normal)")

                st.subheader("📈 Identificación de Distribución")

                with st.spinner("Analizando distribución..."):
                    distribucion = identificar_distribucion(datos)

                col_dist1, col_dist2, col_dist3 = st.columns(3)
                with col_dist1:
                    st.metric("Distribución más probable", distribucion.get('distribucion', 'No determinada'))
                with col_dist2:
                    st.metric("Asimetría", f"{distribucion.get('skewness', 0):.3f}")
                with col_dist3:
                    st.metric("Curtosis", f"{distribucion.get('kurtosis', 0):.3f}")

                st.write("**Pruebas de normalidad:**")
                for prueba, resultados in distribucion.get('pruebas', {}).items():
                    if 'p_valor' in resultados and resultados['p_valor'] is not None:
                        if resultados['es_normal']:
                            st.success(f"✅ {prueba}: p = {resultados['p_valor']:.4f} → Distribución normal")
                        else:
                            st.warning(f"⚠️ {prueba}: p = {resultados['p_valor']:.4f} → No normal")

                st.subheader("📊 Visualizaciones")

                col_graf1, col_graf2 = st.columns(2)

                with col_graf1:
                    fig_hist = px.histogram(datos, nbins=30, title="Histograma",
                                           marginal='box', color_discrete_sequence=['#3498db'])
                    fig_hist.update_layout(height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)

                with col_graf2:
                    if len(datos) > 1:
                        fig_qq = go.Figure()
                        n = len(datos)
                        quantiles = np.linspace(0.5/n, 1-0.5/n, n)
                        q_theoretical = stats.norm.ppf(quantiles, np.mean(datos), np.std(datos))
                        q_empirical = np.sort(datos)

                        fig_qq.add_trace(go.Scatter(
                            x=q_theoretical,
                            y=q_empirical,
                            mode='markers',
                            name='Datos',
                            marker=dict(color='#3498db')
                        ))
                        min_val = min(np.min(q_theoretical), np.min(q_empirical))
                        max_val = max(np.max(q_theoretical), np.max(q_empirical))
                        fig_qq.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Referencia',
                            line=dict(color='red', dash='dash')
                        ))
                        fig_qq.update_layout(
                            title="Gráfico Q-Q (Normal)",
                            xaxis_title="Cuantiles Teóricos",
                            yaxis_title="Cuantiles Empíricos",
                            height=400
                        )
                        st.plotly_chart(fig_qq, use_container_width=True)

                col_graf3, col_graf4 = st.columns(2)

                with col_graf3:
                    fig_box = px.box(datos, title="Boxplot", color_discrete_sequence=['#e74c3c'])
                    fig_box.update_layout(height=400)
                    st.plotly_chart(fig_box, use_container_width=True)

                with col_graf4:
                    if len(datos) > 1:
                        fig_violin = px.violin(datos, box=True, title="Violin Plot",
                                              color_discrete_sequence=['#2ecc71'])
                        fig_violin.update_layout(height=400)
                        st.plotly_chart(fig_violin, use_container_width=True)

    if columnas_categoricas:
        st.subheader("📊 Análisis de Variables Categóricas")

        if st.session_state.columna_cat_seleccionada is None or st.session_state.columna_cat_seleccionada not in columnas_categoricas:
            st.session_state.columna_cat_seleccionada = columnas_categoricas[0]

        columna_cat = st.selectbox(
            "Selecciona una variable categórica para analizar:",
            columnas_categoricas,
            index=columnas_categoricas.index(st.session_state.columna_cat_seleccionada) if st.session_state.columna_cat_seleccionada in columnas_categoricas else 0,
            key="selector_categorico"
        )
        st.session_state.columna_cat_seleccionada = columna_cat

        if columna_cat:
            datos_cat = df[columna_cat].dropna()

            if len(datos_cat) > 0:
                freq_table = datos_cat.value_counts().reset_index()
                freq_table.columns = ['Categoría', 'Frecuencia']
                freq_table['Porcentaje'] = (freq_table['Frecuencia'] / len(datos_cat) * 100).round(2)
                freq_table['Porcentaje Acumulado'] = freq_table['Porcentaje'].cumsum().round(2)

                st.subheader("📊 Tabla de Frecuencias")
                st.dataframe(freq_table, use_container_width=True)

                col_cat1, col_cat2, col_cat3 = st.columns(3)
                with col_cat1:
                    st.metric("Total", len(datos_cat))
                with col_cat2:
                    st.metric("Categorías Únicas", len(datos_cat.unique()))
                with col_cat3:
                    moda = datos_cat.mode()[0] if len(datos_cat.mode()) > 0 else "N/A"
                    st.metric("Moda", moda)

                st.subheader("📊 Visualizaciones")

                col_cat_graf1, col_cat_graf2 = st.columns(2)

                with col_cat_graf1:
                    fig_bar = px.bar(
                        freq_table,
                        x='Categoría',
                        y='Frecuencia',
                        text='Porcentaje',
                        title="Gráfico de Barras",
                        color='Categoría'
                    )
                    fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig_bar.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)

                with col_cat_graf2:
                    fig_pie = px.pie(
                        freq_table,
                        values='Frecuencia',
                        names='Categoría',
                        title="Gráfico Circular",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_traces(textinfo='label+percent')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)

# ============================================================
# CONTENIDO PRINCIPAL
# ============================================================

# Si hay datos cargados, mostrar análisis
if df_cargado is not None:
    mostrar_analisis_descriptivo(df_cargado)
    st.markdown("---")

# ============================================================
# PESTAÑAS COMPLETAS
# ============================================================

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

# PESTAÑA 1: DATOS EN LISTA
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Ingreso de Datos")
        st.caption("Ingresa los datos separados por comas o espacios")
        datos_texto = st.text_area(
            "Datos:",
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

                valores, conteos = np.unique(datos, return_counts=True)
                moda = valores[conteos == conteos.max()] if len(valores) > 0 else np.array([])

                varianza = np.var(datos, ddof=1)
                desviacion = np.std(datos, ddof=1)
                rango = np.max(datos) - np.min(datos)

                q1 = np.percentile(datos, 25)
                q3 = np.percentile(datos, 75)
                iqr = q3 - q1

                outliers_info = detectar_outliers(datos)
                distribucion = identificar_distribucion(datos)

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

                moda_str = ', '.join([str(round(m, 2)) for m in moda]) if len(moda) > 0 else "Sin moda"

                st.session_state['resultados_lista'] = {
                    'datos': datos,
                    'n': n,
                    'media': media,
                    'mediana': mediana,
                    'moda': moda,
                    'moda_str': moda_str,
                    'varianza': varianza,
                    'desviacion': desviacion,
                    'rango': rango,
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr,
                    'outliers_info': outliers_info,
                    'distribucion': distribucion,
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
                st.metric("Moda", res['moda_str'])

            with col_res2:
                st.metric("Varianza", f"{res['varianza']:.4f}")
                st.metric("Desviación Estándar", f"{res['desviacion']:.4f}")
                st.metric("Rango", f"{res['rango']:.4f}")
                st.metric("IQR", f"{res['iqr']:.4f}")

            st.subheader("📈 Identificación de Distribución")
            col_dist1, col_dist2 = st.columns(2)
            with col_dist1:
                st.metric("Distribución más probable", res['distribucion'].get('distribucion', 'No determinada'))
            with col_dist2:
                st.metric("Asimetría", f"{res['distribucion'].get('skewness', 0):.3f}")

            if len(res['outliers_info']['outliers']) > 0:
                st.warning(f"⚠️ Outliers detectados: {', '.join([str(round(o, 2)) for o in res['outliers_info']['outliers']])}")
                st.info(f"📊 Límite inferior: {res['outliers_info']['limite_inferior']:.4f} | Límite superior: {res['outliers_info']['limite_superior']:.4f}")

            st.subheader("📊 Tabla de Frecuencias")
            df_frec = pd.DataFrame({
                'Clase': res['clases'],
                'Frecuencia': res['frecuencias'],
                'Marca': [round(m, 2) for m in res['marcas_clase']],
                'F. Acumulada': res['freq_acum']
            })
            st.dataframe(df_frec, use_container_width=True)

            st.subheader("📈 Visualizaciones")

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

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=res['datos'],
                nbinsx=len(res['clases']),
                marker_color='#3498db',
                opacity=0.7,
                name="Histograma"
            ))
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

# PESTAÑA 2: DATOS AGRUPADOS
with tab2:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Ingreso de Datos Agrupados")
        st.caption("Formato: clase1, frecuencia1; clase2, frecuencia2; ...")
        datos_agrupados_texto = st.text_area(
            "Datos agrupados:",
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
                media = np.average(marcas_clase, weights=frecuencias)

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

                varianza = np.average((np.array(marcas_clase) - media)**2, weights=frecuencias) * n / (n - 1)
                desviacion = np.sqrt(varianza)

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

                freq_rel = [f/n*100 for f in frecuencias]
                freq_acum = np.cumsum(frecuencias)
                freq_rel_acum = np.cumsum(freq_rel)

                st.session_state['resultados_agrupados'] = {
                    'clases': clases,
                    'frecuencias': frecuencias,
                    'marcas_clase': marcas_clase,
                    'n': n,
                    'media': media,
                    'mediana': mediana,
                    'moda': moda,
                    'varianza': varianza,
                    'desviacion': desviacion,
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr,
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

            st.subheader("📈 Visualizaciones")

            fig_hist = px.bar(
                df_frec,
                x='Clase',
                y='f',
                title='Histograma de Frecuencias',
                color_discrete_sequence=['#3498db']
            )
            fig_hist.update_layout(height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

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

# PESTAÑA 3: CUALITATIVAS (DATOS)
with tab3:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🎯 Ingreso de Datos Cualitativos")
        st.caption("Ingresa las categorías separadas por comas")
        datos_cuali_texto = st.text_area(
            "Datos cualitativos:",
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

            st.subheader("📊 Tabla de Frecuencias")
            df_cuali = pd.DataFrame({
                'Categoría': res['nombres'],
                'Frecuencia': res['frecuencias'],
                'Porcentaje': [round(p, 2) for p in res['porcentajes']]
            })
            st.dataframe(df_cuali, use_container_width=True)

            st.subheader("📈 Visualizaciones")

            col_graf1, col_graf2 = st.columns(2)

            with col_graf1:
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

# PESTAÑA 4: CUALITATIVAS (FRECUENCIAS)
with tab4:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Ingreso de Categorías con Frecuencias")
        st.caption("Formato: categoría1, frecuencia1; categoría2, frecuencia2; ...")
        datos_cuali_frec = st.text_area(
            "Datos con frecuencias:",
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

            st.subheader("📊 Tabla de Frecuencias")
            df_cuali_frec = pd.DataFrame({
                'Categoría': res['categorias'],
                'Frecuencia': res['frecuencias'],
                'Porcentaje': [round(p, 2) for p in res['porcentajes']]
            })
            st.dataframe(df_cuali_frec, use_container_width=True)

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

# PESTAÑA 5: TABLA DE CONTINGENCIA
with tab5:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Ingreso de Tabla de Contingencia")
        st.caption("Formato: fila1_col1, valor; fila1_col2, valor; ...")

        datos_contingencia = st.text_area(
            "Datos de la tabla:",
            placeholder='Ejemplo: Hombre_Matemáticas, 15; Hombre_Ingeniería, 30; Mujer_Matemáticas, 15; Mujer_Ingeniería, 20',
            height=150
        )

        nombres_filas = st.text_input("Nombres de filas (separados por comas):", value="Hombre, Mujer")
        nombres_columnas = st.text_input("Nombres de columnas (separados por comas):", value="Matemáticas, Ingeniería")

        if st.button("🔍 Calcular", key="calcular_contingencia"):
            # Procesar tabla de contingencia
            filas_vec = [f.strip() for f in nombres_filas.split(',') if f.strip()]
            columnas_vec = [c.strip() for c in nombres_columnas.split(',') if c.strip()]

            if len(filas_vec) < 2 or len(columnas_vec) < 2:
                st.error("❌ Error: Se necesitan al menos 2 filas y 2 columnas.")
            else:
                pares = [p.strip() for p in datos_contingencia.split(';') if p.strip()]
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
                    st.error("❌ Error: No se encontraron datos válidos.")
                else:
                    st.session_state['resultados_contingencia'] = {
                        'matriz': matriz,
                        'filas': filas_vec,
                        'columnas': columnas_vec
                    }
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

            df_cont = pd.DataFrame(res['matriz'], index=res['filas'], columns=res['columnas'])
            df_cont['Total Fila'] = df_cont.sum(axis=1)
            df_cont.loc['Total Columna'] = df_cont.sum()

            st.dataframe(df_cont, use_container_width=True)

            st.subheader("📈 Visualizaciones")

            df_long = pd.DataFrame(res['matriz'], index=res['filas'], columns=res['columnas'])
            df_long = df_long.reset_index().melt(id_vars='index', var_name='Columna', value_name='Frecuencia')
            df_long.columns = ['Fila', 'Columna', 'Frecuencia']

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

# PESTAÑA 6: BIVARIADO
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

                media_x = np.mean(x)
                media_y = np.mean(y)
                var_x = np.var(x, ddof=1)
                var_y = np.var(y, ddof=1)
                sd_x = np.std(x, ddof=1)
                sd_y = np.std(y, ddof=1)

                cor_pearson, p_value = stats.pearsonr(x, y)

                slope, intercept, r_value, p_value_reg, std_err = stats.linregress(x, y)
                r2 = r_value**2
                r2_adj = 1 - (1 - r2) * (n - 1) / (n - 2)

                x_pred = np.linspace(min(x) - 0.1*(max(x)-min(x)), max(x) + 0.1*(max(x)-min(x)), 100)
                y_pred = intercept + slope * x_pred
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

            st.subheader("📊 Datos")
            df_biv = pd.DataFrame({'X': res['x'], 'Y': res['y']})
            st.dataframe(df_biv, use_container_width=True)

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

# PESTAÑA 7: GUÍA DE INTERPRETACIÓN
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

    ## 🔗 Análisis Bivariado (Cuantitativo)

    ### Coeficiente de Correlación de Pearson (r)

    | Valor de r | Interpretación |
    |------------|----------------|
    | 0.90 - 1.00 | Correlación muy fuerte |
    | 0.70 - 0.89 | Correlación fuerte |
    | 0.50 - 0.69 | Correlación moderada |
    | 0.30 - 0.49 | Correlación débil |
    | 0.00 - 0.29 | Correlación muy débil |

    ### Significancia Estadística

    - **p-value < 0.05**: Resultado estadísticamente significativo
    - **p-value ≥ 0.05**: No hay evidencia suficiente para rechazar H₀
    """)
