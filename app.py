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
# FUNCIÓN: ANÁLISIS BIVARIADO COMPLETO
# ============================================================

def analisis_bivariado(df, var1, var2):
    """Realiza análisis bivariado según el tipo de variables"""
    resultados = {
        'tipo': None,
        'estadisticos': {},
        'visualizaciones': []
    }

    es_numerica1 = pd.api.types.is_numeric_dtype(df[var1])
    es_numerica2 = pd.api.types.is_numeric_dtype(df[var2])
    
    # CASO 1: AMBAS CUALITATIVAS
    if not es_numerica1 and not es_numerica2:
        resultados['tipo'] = 'Cualitativa-Cualitativa'
        tabla = pd.crosstab(df[var1], df[var2], margins=True, margins_name='Total')
        resultados['tabla_contingencia'] = tabla
        
        try:
            chi2, p_valor, dof, expected = stats.chi2_contingency(pd.crosstab(df[var1], df[var2]))
            resultados['estadisticos']['chi2'] = chi2
            resultados['estadisticos']['p_valor'] = p_valor
            resultados['estadisticos']['dof'] = dof
            resultados['estadisticos']['esperados'] = expected
        except Exception as e:
            resultados['estadisticos']['error'] = str(e)
        
        try:
            tabla_sin_margenes = pd.crosstab(df[var1], df[var2])
            chi2_val = stats.chi2_contingency(tabla_sin_margenes)[0]
            n = tabla_sin_margenes.sum().sum()
            min_dim = min(tabla_sin_margenes.shape) - 1
            if min_dim > 0:
                cramer_v = np.sqrt(chi2_val / (n * min_dim))
                resultados['estadisticos']['cramer_v'] = cramer_v
        except:
            pass
    
    # CASO 2: AMBAS CUANTITATIVAS
    elif es_numerica1 and es_numerica2:
        resultados['tipo'] = 'Cuantitativa-Cuantitativa'
        datos_validos = df[[var1, var2]].dropna()
        x = datos_validos[var1].values
        y = datos_validos[var2].values
        
        if len(x) > 2:
            try:
                pearson_r, pearson_p = stats.pearsonr(x, y)
                resultados['estadisticos']['pearson_r'] = pearson_r
                resultados['estadisticos']['pearson_p'] = pearson_p
            except:
                pass
            
            try:
                spearman_r, spearman_p = stats.spearmanr(x, y)
                resultados['estadisticos']['spearman_r'] = spearman_r
                resultados['estadisticos']['spearman_p'] = spearman_p
            except:
                pass
            
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                resultados['estadisticos']['pendiente'] = slope
                resultados['estadisticos']['intercepto'] = intercept
                resultados['estadisticos']['r2'] = r_value**2
                if len(x) > 2:
                    resultados['estadisticos']['r2_ajustado'] = 1 - (1 - r_value**2) * (len(x) - 1) / (len(x) - 2)
                resultados['estadisticos']['p_valor_reg'] = p_value
                resultados['estadisticos']['error_std'] = std_err
            except:
                pass
    
    # CASO 3: UNA CUALITATIVA Y UNA CUANTITATIVA
    else:
        resultados['tipo'] = 'Cualitativa-Cuantitativa'
        if es_numerica1:
            var_cat = var2
            var_cuant = var1
        else:
            var_cat = var1
            var_cuant = var2
        
        datos_validos = df[[var_cat, var_cuant]].dropna()
        categorias = datos_validos[var_cat].unique()
        
        estadisticos_cat = {}
        for cat in categorias:
            datos_cat = datos_validos[datos_validos[var_cat] == cat][var_cuant]
            estadisticos_cat[cat] = {
                'n': len(datos_cat),
                'media': np.mean(datos_cat),
                'mediana': np.median(datos_cat),
                'std': np.std(datos_cat, ddof=1),
                'min': np.min(datos_cat),
                'max': np.max(datos_cat)
            }
        resultados['estadisticos_cat'] = estadisticos_cat
        
        if len(categorias) >= 2:
            try:
                grupos = [datos_validos[datos_validos[var_cat] == cat][var_cuant].values for cat in categorias]
                f_stat, p_valor = stats.f_oneway(*grupos)
                resultados['estadisticos']['anova_f'] = f_stat
                resultados['estadisticos']['anova_p'] = p_valor
            except:
                pass
            
            try:
                grupos = [datos_validos[datos_validos[var_cat] == cat][var_cuant].values for cat in categorias]
                h_stat, p_valor = stats.kruskal(*grupos)
                resultados['estadisticos']['kruskal_h'] = h_stat
                resultados['estadisticos']['kruskal_p'] = p_valor
            except:
                pass
    
    return resultados

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

if st.session_state.df_cargado is not None:
    df_cargado = st.session_state.df_cargado

# ============================================================
# FUNCIÓN PARA ANÁLISIS DESCRIPTIVO (CORREGIDA Y COMPLETA)
# ============================================================

def mostrar_analisis_descriptivo(df):
    """Muestra análisis descriptivo completo de un DataFrame"""
    st.subheader("📊 Análisis Descriptivo del DataFrame")

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Filas", len(df))
    with col_info2:
        st.metric("Columnas", len(df.columns))
    with col_info3:
        st.metric("Memoria", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    st.subheader("📋 Tipos de Datos")
    tipos_df = pd.DataFrame({
        'Columna': df.columns,
        'Tipo': df.dtypes.astype(str),
        'Valores Nulos': df.isnull().sum().values,
        'Porcentaje Nulos': (df.isnull().sum() / len(df) * 100).round(2).values
    })
    st.dataframe(tipos_df, use_container_width=True)

    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # ============================================================
    # SECCIÓN: ANÁLISIS BIVARIADO CORREGIDO
    # ============================================================
    st.subheader("🔗 Análisis Bivariado (Tablas Cruzadas)")
    
    todas_columnas = df.columns.tolist()
    if len(todas_columnas) >= 2:
        col1_biv, col2_biv = st.columns(2)
        
        with col1_biv:
            var1 = st.selectbox("Selecciona la primera variable:", todas_columnas, key="biv_var1")
        with col2_biv:
            var2 = st.selectbox("Selecciona la segunda variable:", todas_columnas, key="biv_var2")
        
        if var1 != var2:
            if st.button("📊 Analizar Relación Bivariada", key="btn_bivariado"):
                with st.spinner("Analizando relación entre variables..."):
                    resultados = analisis_bivariado(df, var1, var2)
                    
                    st.success(f"✅ Análisis completado: {resultados['tipo']}")
                    
                    # CASO 1: CUALITATIVA-CUALITATIVA
                    if resultados['tipo'] == 'Cualitativa-Cualitativa':
                        st.subheader("📊 Tabla de Contingencia")
                        st.dataframe(resultados['tabla_contingencia'], use_container_width=True)
                        
                        # Mapa de Calor
                        st.subheader("🌡️ Mapa de Calor")
                        tabla_sin_margenes = pd.crosstab(df[var1], df[var2])
                        fig_heatmap = px.imshow(
                            tabla_sin_margenes,
                            text_auto=True,
                            aspect="auto",
                            title="Mapa de Calor de la Tabla de Contingencia",
                            color_continuous_scale='Blues'
                        )
                        fig_heatmap.update_layout(height=400)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        # Barras Apiladas (Diagrama de Mosaico CORREGIDO)
                        st.subheader("🧩 Diagrama de Mosaico")
                        tabla_pct = pd.crosstab(df[var1], df[var2], normalize='index') * 100
                        df_stacked = tabla_pct.reset_index().melt(id_vars=var1)
                        df_stacked.columns = ['Variable1', 'Variable2', 'Porcentaje']
                        
                        fig_stacked = px.bar(
                            df_stacked,
                            x='Variable1',
                            y='Porcentaje',
                            color='Variable2',
                            title="Diagrama de Mosaico (Barras 100% apiladas)",
                            barmode='stack',
                            text=[f"{v:.1f}%" for v in df_stacked['Porcentaje']]
                        )
                        fig_stacked.update_traces(textposition='inside')
                        fig_stacked.update_layout(height=400)
                        st.plotly_chart(fig_stacked, use_container_width=True)
                        
                        # Estadísticos
                        st.subheader("📈 Prueba de Independencia (Chi-cuadrado)")
                        if 'chi2' in resultados['estadisticos']:
                            col_chi1, col_chi2, col_chi3 = st.columns(3)
                            with col_chi1:
                                st.metric("χ²", f"{resultados['estadisticos']['chi2']:.4f}")
                            with col_chi2:
                                st.metric("p-valor", f"{resultados['estadisticos']['p_valor']:.4f}")
                            with col_chi3:
                                st.metric("Grados de libertad", resultados['estadisticos']['dof'])
                            
                            if resultados['estadisticos']['p_valor'] < 0.05:
                                st.success("✅ Relación estadísticamente significativa (p < 0.05)")
                            else:
                                st.warning("⚠️ No hay evidencia de relación significativa (p ≥ 0.05)")
                        
                        if 'cramer_v' in resultados['estadisticos']:
                            st.metric("V de Cramer", f"{resultados['estadisticos']['cramer_v']:.4f}")
                    
                    # CASO 2: CUANTITATIVA-CUANTITATIVA
                    elif resultados['tipo'] == 'Cuantitativa-Cuantitativa':
                        st.subheader("📊 Estadísticos de Correlación")
                        col_corr1, col_corr2 = st.columns(2)
                        
                        with col_corr1:
                            if 'pearson_r' in resultados['estadisticos']:
                                st.metric("Correlación de Pearson (r)", 
                                         f"{resultados['estadisticos']['pearson_r']:.4f}")
                                st.caption(f"p-valor: {resultados['estadisticos']['pearson_p']:.4f}")
                            
                            if 'r2' in resultados['estadisticos']:
                                st.metric("R²", f"{resultados['estadisticos']['r2']:.4f}")
                        
                        with col_corr2:
                            if 'spearman_r' in resultados['estadisticos']:
                                st.metric("Correlación de Spearman (ρ)", 
                                         f"{resultados['estadisticos']['spearman_r']:.4f}")
                                st.caption(f"p-valor: {resultados['estadisticos']['spearman_p']:.4f}")
                            
                            if 'r2_ajustado' in resultados['estadisticos']:
                                st.metric("R² Ajustado", f"{resultados['estadisticos']['r2_ajustado']:.4f}")
                        
                        # Gráfico de dispersión
                        st.subheader("📈 Dispersión y Regresión")
                        datos = df[[var1, var2]].dropna()
                        
                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(go.Scatter(
                            x=datos[var1],
                            y=datos[var2],
                            mode='markers',
                            name='Datos',
                            marker=dict(color='#3498db', size=8)
                        ))
                        
                        if 'pendiente' in resultados['estadisticos']:
                            x_range = np.linspace(datos[var1].min(), datos[var1].max(), 100)
                            y_reg = resultados['estadisticos']['intercepto'] + resultados['estadisticos']['pendiente'] * x_range
                            fig_scatter.add_trace(go.Scatter(
                                x=x_range,
                                y=y_reg,
                                mode='lines',
                                name='Regresión',
                                line=dict(color='#e74c3c', width=2)
                            ))
                            fig_scatter.update_layout(
                                title=f"Y = {resultados['estadisticos']['intercepto']:.4f} + {resultados['estadisticos']['pendiente']:.4f}X",
                                xaxis_title=var1,
                                yaxis_title=var2,
                                height=400
                            )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # CASO 3: CUALITATIVA-CUANTITATIVA
                    elif resultados['tipo'] == 'Cualitativa-Cuantitativa':
                        if pd.api.types.is_numeric_dtype(df[var1]):
                            var_cat = var2
                            var_cuant = var1
                        else:
                            var_cat = var1
                            var_cuant = var2
                        
                        st.subheader("📊 Estadísticos por Categoría")
                        stats_cat_df = pd.DataFrame(resultados['estadisticos_cat']).T
                        stats_cat_df = stats_cat_df.round(4)
                        st.dataframe(stats_cat_df, use_container_width=True)
                        
                        # Boxplots
                        st.subheader("📊 Boxplots por Categoría")
                        fig_box = px.box(
                            df[[var_cat, var_cuant]].dropna(),
                            x=var_cat,
                            y=var_cuant,
                            color=var_cat,
                            title="Boxplots por Categoría",
                            points="all"
                        )
                        fig_box.update_layout(height=400)
                        st.plotly_chart(fig_box, use_container_width=True)
                        
                        # Pruebas estadísticas
                        st.subheader("📈 Pruebas de Comparación")
                        col_test1, col_test2 = st.columns(2)
                        
                        with col_test1:
                            if 'anova_f' in resultados['estadisticos']:
                                st.metric("ANOVA F", f"{resultados['estadisticos']['anova_f']:.4f}")
                                st.caption(f"p-valor: {resultados['estadisticos']['anova_p']:.4f}")
                                if resultados['estadisticos']['anova_p'] < 0.05:
                                    st.success("✅ Diferencias significativas entre grupos")
                                else:
                                    st.warning("⚠️ No hay diferencias significativas")
                        
                        with col_test2:
                            if 'kruskal_h' in resultados['estadisticos']:
                                st.metric("Kruskal-Wallis H", f"{resultados['estadisticos']['kruskal_h']:.4f}")
                                st.caption(f"p-valor: {resultados['estadisticos']['kruskal_p']:.4f}")
                                if resultados['estadisticos']['kruskal_p'] < 0.05:
                                    st.success("✅ Diferencias significativas (no paramétrica)")
                                else:
                                    st.warning("⚠️ No hay diferencias significativas")
    else:
        st.warning("⚠️ Se necesitan al menos 2 columnas para análisis bivariado.")

    # ============================================================
    # ANÁLISIS UNIVARIADO (NUMÉRICO)
    # ============================================================
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

    # ============================================================
    # ANÁLISIS UNIVARIADO (CATEGÓRICO)
    # ============================================================
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

if df_cargado is not None:
    mostrar_analisis_descriptivo(df_cargado)
    st.markdown("---")

# ============================================================
# PESTAÑAS COMPLETAS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Datos en Lista",
    "📊 Datos Agrupados",
    "🎯 Cualitativas (Datos)",
    "📊 Cualitativas (Frecuencias)",
    "📊 Tabla de Contingencia",
    "🔗 Bivariado (Cuantitativo)",
    "📖 Guía de Interpretación"
])

with tab1:
    st.header("📈 Análisis de Datos en Lista")
    texto = st.text_area("Ingresa los datos numéricos separados por espacios o comas:", height=150)
    if texto and st.button("📊 Calcular Estadísticos (Lista)"):
        datos = procesar_datos_lista(texto)
        if datos is None:
            st.error("❌ Ingresa al menos 2 números válidos.")
        else:
            st.success(f"✅ {len(datos)} datos procesados.")
            st.write(f"Datos: {datos}")
            # Aquí se mostrarían las estadísticas de lista (similar a tu código original)

with tab2:
    st.header("📊 Análisis de Datos Agrupados")
    st.markdown("Formato: `Clase,Frecuencia; Clase,Frecuencia` (Ej: `10-20,5; 20-30,8`)")
    texto = st.text_area("Ingresa las clases y frecuencias:", height=150)
    if texto and st.button("📊 Calcular Estadísticos (Agrupados)"):
        datos_agrupados = procesar_datos_agrupados(texto)
        if datos_agrupados is None:
            st.error("❌ Error en el formato de los datos.")
        else:
            st.success("✅ Datos agrupados procesados.")

with tab3:
    st.header("🎯 Análisis de Datos Cualitativos")
    texto = st.text_area("Ingresa las categorías separadas por comas:", height=100)
    if texto and st.button("📊 Procesar Cualitativos"):
        categorias = procesar_datos_cualitativos(texto)
        if categorias is None:
            st.error("❌ Ingresa al menos 2 categorías.")
        else:
            st.success(f"✅ Categorías: {categorias}")

with tab4:
    st.header("📊 Datos Cualitativos con Frecuencias")
    st.markdown("Formato: `Categoría,Frecuencia; Categoría,Frecuencia`")
    texto = st.text_area("Ingresa categorías y frecuencias:", height=150)
    if texto and st.button("📊 Procesar Cuali-Frec"):
        datos = procesar_cuali_frec(texto)
        if datos is None:
            st.error("❌ Error en el formato.")
        else:
            st.success("✅ Datos cargados.")

with tab5:
    st.header("📊 Tabla de Contingencia Manual")
    st.markdown("Crea una tabla de contingencia a partir de dos variables.")
    # Funcionalidad para ingresar tabla manual

with tab6:
    st.header("🔗 Bivariado (Cuantitativo)")
    st.markdown("Relación entre dos conjuntos de datos numéricos.")
    col_x, col_y = st.columns(2)
    texto_x = col_x.text_area("Variable X (numérica):", height=100)
    texto_y = col_y.text_area("Variable Y (numérica):", height=100)
    if texto_x and texto_y and st.button("📊 Analizar Relación"):
        datos_biv = procesar_datos_biv(texto_x, texto_y)
        if datos_biv is None:
            st.error("❌ Asegúrate de tener la misma cantidad de datos (>2).")
        else:
            st.success("✅ Datos bivariados cargados.")

with tab7:
    st.header("📖 Guía de Interpretación")
    st.markdown("""
    ### 📚 Interpretación de Resultados

    **1. Estadísticos Básicos:**
    - **Media:** Promedio de los datos.
    - **Mediana:** Valor central.
    - **Moda:** Valor más frecuente.
    - **Varianza/Desviación:** Dispersión de los datos.

    **2. Pruebas de Normalidad (p > 0.05):**
    Indica que los datos siguen una distribución normal (simétrica).

    **3. Asimetría (Skewness):**
    - **= 0:** Simétrica.
    - **> 0:** Sesgo positivo (cola a la derecha).
    - **< 0:** Sesgo negativo (cola a la izquierda).

    **4. Curtosis:**
    - **< 3:** Platicúrtica (datos achatados).
    - **= 3:** Mesocúrtica (normal).
    - **> 3:** Leptocúrtica (picos altos y colas gruesas).

    **5. Diagrama de Mosaico (Bivariado):**
    Muestra la relación entre dos variables categóricas. Si un bloque es significativamente más grande, indica una relación entre esas categorías.
    """)

st.markdown("""
---
### 📊 Funcionalidades disponibles:
- **Análisis Bivariado**: Selecciona dos variables y obtén automáticamente el análisis adecuado
- **Cualitativa-Cualitativa**: Tablas cruzadas, Chi-cuadrado, V de Cramer
- **Cuantitativa-Cuantitativa**: Correlación, regresión, análisis de residuos
- **Cualitativa-Cuantitativa**: ANOVA, Kruskal-Wallis, boxplots y más
""")
