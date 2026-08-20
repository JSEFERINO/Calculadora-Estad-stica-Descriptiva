import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import gmean, hmean, trim_mean
import re
import requests
import io
import csv

# ============================================================
# CONFIGURACION DE LA PAGINA
# ============================================================
st.set_page_config(
    page_title="Calculadora Estadistica Descriptiva",
    page_icon=":chart_with_upwards_trend:",
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
if 'resultados_lista' not in st.session_state:
    st.session_state.resultados_lista = None
if 'resultados_cuant_frec' not in st.session_state:
    st.session_state.resultados_cuant_frec = None

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def calcular_moda(datos):
    """Calcula la moda de manera robusta"""
    try:
        if isinstance(datos, np.ndarray):
            valores, conteos = np.unique(datos, return_counts=True)
            if len(valores) == 0: return np.nan
            max_count = np.max(conteos)
            if max_count == 1: return np.nan
            modas = valores[conteos == max_count]
            return modas[0] if len(modas) == 1 else modas
        else:
            from collections import Counter
            counter = Counter(datos)
            if not counter: return np.nan
            max_count = max(counter.values())
            if max_count == 1: return np.nan
            modas = [k for k, v in counter.items() if v == max_count]
            return modas[0] if len(modas) == 1 else modas
    except:
        return np.nan

def calcular_estadisticas_avanzadas(datos):
    """Calcula estadisticas avanzadas para una variable numerica"""
    n = len(datos)
    datos_ordenados = np.sort(datos)

    media = np.mean(datos)
    mediana = np.median(datos)
    moda = calcular_moda(datos)
    varianza = np.var(datos, ddof=1)
    desviacion = np.std(datos, ddof=1)
    rango = np.max(datos) - np.min(datos)
    q1 = np.percentile(datos, 25)
    q3 = np.percentile(datos, 75)
    iqr = q3 - q1

    try:
        media_geometrica = gmean(datos) if np.all(datos > 0) else np.nan
    except:
        media_geometrica = np.nan

    try:
        media_armonica = hmean(datos) if np.all(datos > 0) else np.nan
    except:
        media_armonica = np.nan

    try:
        media_truncada_5 = trim_mean(datos, 0.05)
    except:
        media_truncada_5 = np.nan

    try:
        from scipy.stats.mstats import winsorize
        datos_winsor = winsorize(datos, limits=[0.05, 0.05])
        media_winsorizada_5 = np.mean(datos_winsor)
    except:
        media_winsorizada_5 = np.nan

    cv = (desviacion / media * 100) if media != 0 else np.nan

    try:
        datos_gini = np.sort(datos)
        n_gini = len(datos_gini)
        sumatoria = np.sum((2 * np.arange(1, n_gini + 1) - n_gini - 1) * datos_gini)
        gini = sumatoria / (n_gini * np.sum(datos_gini)) if np.sum(datos_gini) != 0 else np.nan
    except:
        gini = np.nan

    error_estandar = desviacion / np.sqrt(n) if n > 1 else np.nan

    try:
        log_datos = np.log(datos[datos > 0])
        gsd = np.exp(np.std(log_datos, ddof=1)) if len(log_datos) > 1 else np.nan
    except:
        gsd = np.nan

    try:
        sigma_winsor = np.std(datos_winsor, ddof=1) if 'datos_winsor' in locals() else np.nan
    except:
        sigma_winsor = np.nan

    mad_media = np.mean(np.abs(datos - media))
    mad = np.median(np.abs(datos - mediana))

    try:
        sbi = np.mean(datos) / np.std(datos, ddof=1) if np.std(datos, ddof=1) != 0 else np.nan
    except:
        sbi = np.nan

    sextil_1_6 = np.percentile(datos, 100/6)
    sextil_5_6 = np.percentile(datos, 500/6)
    isr = sextil_5_6 - sextil_1_6

    try:
        skewness = stats.skew(datos)
    except:
        skewness = np.nan

    try:
        skewness_std = skewness / np.sqrt(6/n) if n > 1 else np.nan
    except:
        skewness_std = np.nan

    try:
        kurtosis = stats.kurtosis(datos)
    except:
        kurtosis = np.nan

    try:
        kurtosis_std = kurtosis / np.sqrt(24/n) if n > 1 else np.nan
    except:
        kurtosis_std = np.nan

    suma = np.sum(datos)
    suma_cuadrados = np.sum(datos**2)

    return {
        'n': n,
        'media': media,
        'mediana': mediana,
        'moda': moda,
        'media_geometrica': media_geometrica,
        'media_armonica': media_armonica,
        'media_truncada_5': media_truncada_5,
        'media_winsorizada_5': media_winsorizada_5,
        'varianza': varianza,
        'desviacion': desviacion,
        'cv': cv,
        'gini': gini,
        'error_estandar': error_estandar,
        'gsd': gsd,
        'sigma_winsor': sigma_winsor,
        'mad_media': mad_media,
        'mad': mad,
        'sbi': sbi,
        'minimo': np.min(datos),
        'maximo': np.max(datos),
        'rango': rango,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'sextil_1_6': sextil_1_6,
        'sextil_5_6': sextil_5_6,
        'isr': isr,
        'skewness': skewness,
        'skewness_std': skewness_std,
        'kurtosis': kurtosis,
        'kurtosis_std': kurtosis_std,
        'suma': suma,
        'suma_cuadrados': suma_cuadrados
    }

def calcular_medidas_agrupadas(res):
    """
    Calcula y muestra las medidas descriptivas para datos agrupados
    incluyendo Media, Mediana, Moda, Q1, Q3, Percentiles
    """
    st.subheader("📊 Medidas Descriptivas para Datos Agrupados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📌 Total de datos (n)", res['n'])
        st.metric("📊 Media", f"{res['media']:.4f}")
        st.metric("📊 Mediana", f"{res['mediana']:.4f}")
        st.metric("📊 Moda (aproximada)", f"{res['moda']:.4f}")
    
    with col2:
        st.metric("📊 Varianza", f"{res['varianza']:.4f}")
        st.metric("📊 Desviación Estándar", f"{res['desviacion']:.4f}")
        st.metric("📊 Q1 (Cuartil 1)", f"{res['q1']:.4f}")
        st.metric("📊 Q3 (Cuartil 3)", f"{res['q3']:.4f}")
    
    with col3:
        st.metric("📊 Rango", f"{res['rango']:.4f}")
        st.metric("📊 IQR", f"{res['iqr']:.4f}")
        st.metric("📊 Percentil 10", f"{calcular_cuantil_agrupado(res, 0.10):.4f}" if 'calcular_cuantil_agrupado' in globals() else "N/A")
        st.metric("📊 Percentil 90", f"{calcular_cuantil_agrupado(res, 0.90):.4f}" if 'calcular_cuantil_agrupado' in globals() else "N/A")
    
    # Percentiles adicionales
    st.subheader("📊 Percentiles Personalizados para Datos Agrupados")
    
    col_perc1, col_perc2 = st.columns(2)
    with col_perc1:
        percentiles_seleccionados = st.multiselect(
            "Selecciona percentiles:",
            [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95],
            default=[10, 25, 50, 75, 90],
            key="percentiles_agrupados"
        )
    with col_perc2:
        if st.button("Calcular Percentiles Agrupados", key="calc_percentiles_agrupados"):
            if percentiles_seleccionados:
                st.write("**Resultados:**")
                cols = st.columns(min(4, len(percentiles_seleccionados)))
                for i, p in enumerate(sorted(percentiles_seleccionados)):
                    with cols[i % len(cols)]:
                        valor = calcular_cuantil_agrupado(res, p/100)
                        st.metric(f"Percentil {p}%", f"{valor:.4f}")

def calcular_cuantil_agrupado(res, p):
    """
    Calcula un cuantil específico para datos agrupados
    p: proporción (0-1)
    """
    if 'limites_inferiores' not in res or 'limites_superiores' not in res:
        return np.nan
    
    n = res['n']
    pos = p * n
    frecuencias = res['frecuencias']
    limites_inf = res['limites_inferiores']
    limites_sup = res['limites_superiores']
    
    acumulado = 0
    clase_cuantil = 0
    for i, f in enumerate(frecuencias):
        acumulado += f
        if acumulado >= pos:
            clase_cuantil = i
            break
    
    f_ant = sum(frecuencias[:clase_cuantil]) if clase_cuantil > 0 else 0
    f_clase = frecuencias[clase_cuantil]
    L_cuantil = limites_inf[clase_cuantil]
    A_cuantil = limites_sup[clase_cuantil] - limites_inf[clase_cuantil]
    
    if f_clase == 0:
        return np.nan
    
    return L_cuantil + ((pos - f_ant) / f_clase) * A_cuantil

# ============================================================
# FUNCION PARA HISTOGRAMA INTERACTIVO
# ============================================================

def histograma_interactivo(datos, nombre_variable="Variable"):
    """Crea un histograma interactivo con controles para el usuario"""

    if len(datos) < 2:
        st.warning("No hay suficientes datos para generar el histograma.")
        return

    st.subheader(f"Histograma Interactivo - {nombre_variable}")

    # Controles en una fila
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col1:
        # Número de clases
        n_clases_default = calcular_n_clases_sturges(len(datos))
        n_clases = st.number_input(
            "Number of Classes:",
            min_value=2,
            max_value=50,
            value=n_clases_default,
            step=1,
            key=f"n_clases_{nombre_variable}"
        )

    with col2:
        # Límite inferior
        min_val = float(np.min(datos))
        max_val = float(np.max(datos))
        lower_limit = st.number_input(
            "Lower Limit:",
            value=min_val,
            step=0.5,
            key=f"lower_limit_{nombre_variable}"
        )

    with col3:
        # Límite superior
        upper_limit = st.number_input(
            "Upper Limit:",
            value=max_val,
            step=0.5,
            key=f"upper_limit_{nombre_variable}"
        )

    with col4:
        # Tipo de frecuencia
        freq_type = st.selectbox(
            "Counts",
            ["Absolute", "Relative", "Cumulative"],
            key=f"freq_type_{nombre_variable}"
        )

    with col5:
        # Tipo de gráfico
        plot_type = st.selectbox(
            "Plot Type",
            ["Histogram", "Polygon", "Both"],
            key=f"plot_type_{nombre_variable}"
        )

    # Botón para actualizar
    if st.button("Update Histogram", key=f"update_hist_{nombre_variable}"):
        # Filtrar datos según límites
        datos_filtrados = datos[(datos >= lower_limit) & (datos <= upper_limit)]

        if len(datos_filtrados) < 2:
            st.warning("No hay datos en el rango seleccionado. Ajusta los límites.")
            return

        # Calcular histograma
        hist, bin_edges = np.histogram(datos_filtrados, bins=n_clases, range=(lower_limit, upper_limit))

        # Calcular marcas de clase
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Crear etiquetas de clase [a,b)
        class_labels = []
        for i in range(len(hist)):
            if i == len(hist) - 1:
                class_labels.append(f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]")
            else:
                class_labels.append(f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})")

        # Calcular frecuencias según el tipo seleccionado
        n = len(datos_filtrados)
        if freq_type == "Absolute":
            freq_values = hist
            y_label = "Frequency"
            title_suffix = "Frecuencias Absolutas"
        elif freq_type == "Relative":
            freq_values = hist / n * 100
            y_label = "Relative Frequency (%)"
            title_suffix = "Frecuencias Relativas (%)"
        else:  # Cumulative
            freq_values = np.cumsum(hist)
            y_label = "Cumulative Frequency"
            title_suffix = "Frecuencias Acumuladas"

        # Crear figura
        fig = go.Figure()

        # Agregar histograma
        if plot_type in ["Histogram", "Both"]:
            fig.add_trace(go.Bar(
                x=class_labels,
                y=freq_values,
                name='Histogram',
                marker_color='#3498db',
                opacity=0.8,
                text=[f"{v:.2f}" for v in freq_values] if freq_type != "Absolute" else [str(int(v)) for v in freq_values],
                textposition='outside'
            ))

        # Agregar polígono
        if plot_type in ["Polygon", "Both"]:
            fig.add_trace(go.Scatter(
                x=bin_centers,
                y=freq_values,
                mode='lines+markers',
                name='Polygon',
                line=dict(color='#e74c3c', width=2),
                marker=dict(size=8, color='#e74c3c')
            ))

        # Configurar layout
        fig.update_layout(
            title=f"Histograma de {title_suffix} - {nombre_variable}",
            xaxis_title="Clases",
            yaxis_title=y_label,
            height=450,
            showlegend=True,
            bargap=0 if plot_type in ["Histogram", "Both"] else None,
            bargroupgap=0 if plot_type in ["Histogram", "Both"] else None,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Mostrar tabla de frecuencias
        st.subheader("Tabla de Frecuencias")
        df_freq = pd.DataFrame({
            'Clase': class_labels,
            'Marca de Clase': [f"{c:.2f}" for c in bin_centers],
            'Frecuencia': hist,
            'Frecuencia Relativa (%)': [f"{h/n*100:.2f}" for h in hist],
            'Frecuencia Acumulada': np.cumsum(hist)
        })
        st.dataframe(df_freq, use_container_width=True)

        # Mostrar estadísticas del rango seleccionado
        st.subheader("Estadísticas del Rango Seleccionado")
        col_est1, col_est2, col_est3 = st.columns(3)
        with col_est1:
            st.metric("Datos en rango", len(datos_filtrados))
            st.metric("Media", f"{np.mean(datos_filtrados):.4f}")
            st.metric("Mediana", f"{np.median(datos_filtrados):.4f}")
        with col_est2:
            st.metric("Desviación Estándar", f"{np.std(datos_filtrados, ddof=1):.4f}")
            st.metric("Mínimo", f"{np.min(datos_filtrados):.4f}")
            st.metric("Máximo", f"{np.max(datos_filtrados):.4f}")
        with col_est3:
            st.metric("Rango", f"{np.max(datos_filtrados) - np.min(datos_filtrados):.4f}")
            st.metric("Q1", f"{np.percentile(datos_filtrados, 25):.4f}")
            st.metric("Q3", f"{np.percentile(datos_filtrados, 75):.4f}")

def detectar_separador(texto):
    """Detecta automaticamente el separador en un texto CSV"""
    lineas = [line.strip() for line in texto.split('\n') if line.strip()]
    if not lineas: return ','
    separadores = [',', ';', '\t', '|', ' ']
    conteos = {sep: sum(linea.count(sep) for linea in lineas) for sep in separadores}
    mejor_sep = max(conteos, key=conteos.get)
    return mejor_sep if conteos[mejor_sep] > 0 else ','

def cargar_datos_desde_csv(archivo, separador=None):
    try:
        contenido = archivo.getvalue().decode('utf-8', errors='ignore')
        if separador is None: separador = detectar_separador(contenido)
        try:
            df = pd.read_csv(io.StringIO(contenido), sep=separador)
        except:
            for sep in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(io.StringIO(contenido), sep=sep)
                    if len(df.columns) > 1: break
                except: continue
            else:
                df = pd.read_csv(io.StringIO(contenido), sep=separador, engine='python')
        df.columns = [col.strip().replace(' ', '_').replace(';', '').replace(',', '') for col in df.columns]
        for col in df.columns:
            try: df[col] = pd.to_numeric(df[col])
            except: pass
        return df
    except Exception as e:
        st.error(f"Error al cargar CSV: {str(e)}")
        return None

def cargar_datos_desde_url(url, separador=None):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text
        if separador is None: separador = detectar_separador(content)
        try:
            df = pd.read_csv(io.StringIO(content), sep=separador)
        except:
            for sep in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=sep)
                    if len(df.columns) > 1: break
                except: continue
            else:
                df = pd.read_csv(io.StringIO(content), sep=separador, engine='python')
        df.columns = [col.strip().replace(' ', '_').replace(';', '').replace(',', '') for col in df.columns]
        for col in df.columns:
            try: df[col] = pd.to_numeric(df[col])
            except: pass
        return df
    except Exception as e:
        st.error(f"Error al cargar desde URL: {str(e)}")
        return None

def identificar_distribucion(datos):
    """Identifica la distribucion de los datos usando pruebas estadisticas"""
    resultado_base = {"distribucion": "Datos insuficientes", "distribuciones_posibles": [], "pruebas": {}, "skewness": 0, "kurtosis": 0}
    if len(datos) < 8: return resultado_base
    resultados = {}
    try:
        shapiro_stat, shapiro_p = stats.shapiro(datos)
        resultados['Shapiro-Wilk'] = {'estadistico': shapiro_stat, 'p_valor': shapiro_p, 'es_normal': shapiro_p > 0.05}
    except: resultados['Shapiro-Wilk'] = {'estadistico': None, 'p_valor': None, 'es_normal': False}
    try:
        ks_stat, ks_p = stats.kstest(datos, 'norm', args=(np.mean(datos), np.std(datos)))
        resultados['Kolmogorov-Smirnov'] = {'estadistico': ks_stat, 'p_valor': ks_p, 'es_normal': ks_p > 0.05}
    except: resultados['Kolmogorov-Smirnov'] = {'estadistico': None, 'p_valor': None, 'es_normal': False}
    try:
        anderson_result = stats.anderson(datos, dist='norm')
        resultados['Anderson-Darling'] = {'estadistico': anderson_result.statistic, 'es_normal': anderson_result.statistic < anderson_result.critical_values[2]}
    except: resultados['Anderson-Darling'] = {'estadistico': None, 'es_normal': False}
    distribuciones_posibles = []
    es_normal = resultados['Shapiro-Wilk']['es_normal'] if resultados['Shapiro-Wilk']['p_valor'] is not None else False
    if es_normal: distribuciones_posibles.append("Normal")
    try:
        uni_stat, uni_p = stats.kstest(datos, 'uniform', args=(min(datos), max(datos)-min(datos)))
        if uni_p > 0.05: distribuciones_posibles.append("Uniforme")
    except: pass
    try:
        exp_stat, exp_p = stats.kstest(datos, 'expon', args=(0, np.mean(datos)))
        if exp_p > 0.05: distribuciones_posibles.append("Exponencial")
    except: pass
    try: skewness = stats.skew(datos) if len(datos) > 2 else 0
    except: skewness = 0
    try: kurtosis = stats.kurtosis(datos) if len(datos) > 3 else 0
    except: kurtosis = 0
    if not distribuciones_posibles:
        if abs(skewness) > 1: distribuciones_posibles.append("Asimetrica (sesgo significativo)")
        elif abs(kurtosis) > 3: distribuciones_posibles.append("Con colas pesadas (leptocurtica)")
        else: distribuciones_posibles.append("Sin distribucion clara")
    return {"distribucion": distribuciones_posibles[0] if distribuciones_posibles else "No determinada", "distribuciones_posibles": distribuciones_posibles, "pruebas": resultados, "skewness": skewness, "kurtosis": kurtosis}

def analisis_bivariado(df, var1, var2):
    """Realiza analisis bivariado segun el tipo de variables"""
    resultados = {'tipo': None, 'estadisticos': {}, 'visualizaciones': []}
    es_numerica1 = pd.api.types.is_numeric_dtype(df[var1])
    es_numerica2 = pd.api.types.is_numeric_dtype(df[var2])
    if not es_numerica1 and not es_numerica2:
        resultados['tipo'] = 'Cualitativa-Cualitativa'
        tabla = pd.crosstab(df[var1], df[var2], margins=True, margins_name='Total')
        resultados['tabla_contingencia'] = tabla
        try:
            chi2, p_valor, dof, expected = stats.chi2_contingency(pd.crosstab(df[var1], df[var2]))
            resultados['estadisticos']['chi2'] = chi2
            resultados['estadisticos']['p_valor'] = p_valor
            resultados['estadisticos']['dof'] = dof
        except: pass
        try:
            tabla_sin_margenes = pd.crosstab(df[var1], df[var2])
            chi2_val = stats.chi2_contingency(tabla_sin_margenes)[0]
            n = tabla_sin_margenes.sum().sum()
            min_dim = min(tabla_sin_margenes.shape) - 1
            if min_dim > 0:
                cramer_v = np.sqrt(chi2_val / (n * min_dim))
                resultados['estadisticos']['cramer_v'] = cramer_v
        except: pass
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
            except: pass
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                resultados['estadisticos']['pendiente'] = slope
                resultados['estadisticos']['intercepto'] = intercept
                resultados['estadisticos']['r2'] = r_value**2
            except: pass
    else:
        resultados['tipo'] = 'Cualitativa-Cuantitativa'
        var_cat = var2 if es_numerica1 else var1
        var_cuant = var1 if es_numerica1 else var2
        datos_validos = df[[var_cat, var_cuant]].dropna()
        categorias = datos_validos[var_cat].unique()
        estadisticos_cat = {}
        for cat in categorias:
            datos_cat = datos_validos[datos_validos[var_cat] == cat][var_cuant]
            estadisticos_cat[cat] = {'n': len(datos_cat), 'media': np.mean(datos_cat), 'mediana': np.median(datos_cat), 'std': np.std(datos_cat, ddof=1)}
        resultados['estadisticos_cat'] = estadisticos_cat
        if len(categorias) >= 2:
            try:
                grupos = [datos_validos[datos_validos[var_cat] == cat][var_cuant].values for cat in categorias]
                f_stat, p_valor = stats.f_oneway(*grupos)
                resultados['estadisticos']['anova_f'] = f_stat
                resultados['estadisticos']['anova_p'] = p_valor
            except: pass
    return resultados

def detectar_outliers(datos):
    """Detecta outliers usando el metodo IQR"""
    if len(datos) < 2:
        return {'outliers': [], 'limite_inferior': None, 'limite_superior': None,
                'q1': None, 'q3': None, 'iqr': None}
    q1 = np.percentile(datos, 25)
    q3 = np.percentile(datos, 75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    outliers = datos[(datos < limite_inferior) | (datos > limite_superior)]
    return {'outliers': outliers, 'limite_inferior': limite_inferior, 'limite_superior': limite_superior, 'q1': q1, 'q3': q3, 'iqr': iqr}

def calcular_percentiles_personalizados(datos, percentiles):
    """Calcula percentiles personalizados"""
    resultados = {}
    for p in percentiles:
        if 0 <= p <= 100:
            valor = np.percentile(datos, p)
            resultados[p] = valor
    return resultados

# ============================================================
# CORRECCIÓN: PROCESAR DATOS EN LISTA
# ============================================================

def procesar_datos_lista(texto):
    """Procesa datos en formato lista (acepta enteros, decimales y negativos)"""
    texto = texto.replace(',', '.')
    numeros = re.findall(r'-?\d+\.?\d*', texto)
    resultado = []
    for num in numeros:
        try:
            resultado.append(float(num))
        except ValueError:
            continue
    if len(resultado) < 2:
        return None
    return np.array(resultado)

def procesar_datos_agrupados(texto):
    """Procesa datos agrupados con intervalos [a,b)"""
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
        elif ',' in clase:
            limites = clase.split(',')
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

def procesar_datos_cuantitativos_frec(texto):
    """Procesa datos cuantitativos con frecuencias (valor, frecuencia)"""
    pares = [p.strip() for p in texto.split(';') if p.strip()]
    valores = []
    frecuencias = []
    for par in pares:
        elementos = [e.strip() for e in par.split(',') if e.strip()]
        if len(elementos) == 2:
            try:
                valor = float(elementos[0].strip())
                freq = float(elementos[1].strip())
                if freq > 0 and not np.isnan(valor):
                    valores.append(valor)
                    frecuencias.append(freq)
            except:
                continue
    if len(valores) < 2:
        return None
    return {'valores': np.array(valores), 'frecuencias': np.array(frecuencias)}

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
    return {'matriz': matriz, 'filas': filas_vec, 'columnas': columnas_vec}

def procesar_datos_biv(texto_x, texto_y):
    """Procesa datos bivariados"""
    x = procesar_datos_lista(texto_x)
    y = procesar_datos_lista(texto_y)
    if x is None or y is None or len(x) != len(y) or len(x) < 3:
        return None
    return {'x': x, 'y': y}

# ============================================================
# REGLA DE STURGES PARA CALCULAR NUMERO DE CLASES
# ============================================================
def calcular_n_clases_sturges(n):
    """Calcula el numero de clases usando la regla de Sturges: k = 1 + 3.322 * log10(n)"""
    if n < 2:
        return 1
    k = int(np.ceil(1 + 3.322 * np.log10(n)))
    return max(2, min(50, k))

# ============================================================
# TALLO Y HOJAS + GRAFICO DE PUNTOS
# ============================================================

def grafico_tallo_hojas(datos):
    """Genera un diagrama de Tallo y Hojas"""
    if len(datos) < 2:
        return None

    datos_ordenados = np.sort(datos)
    max_val = int(np.max(datos_ordenados))
    min_val = int(np.min(datos_ordenados))

    if max_val < 100:
        factor = 10
        tallos = range(min_val // 10 * 10, (max_val // 10 + 2) * 10, 10)
        tallo_str = lambda t: str(t // 10)
        hoja_str = lambda v, t: str(int(v % 10))
    elif max_val < 1000:
        factor = 100
        tallos = range(min_val // 100 * 100, (max_val // 100 + 2) * 100, 100)
        tallo_str = lambda t: str(t // 100)
        hoja_str = lambda v, t: str(int((v % 100) / 10)) + str(int(v % 10))
    else:
        factor = 1000
        tallos = range(min_val // 1000 * 1000, (max_val // 1000 + 2) * 1000, 1000)
        tallo_str = lambda t: str(t // 1000)
        hoja_str = lambda v, t: str(int((v % 1000) / 100)) + str(int((v % 100) / 10)) + str(int(v % 10))

    resultado = []
    for t in tallos:
        hojas = [hoja_str(v, t) for v in datos_ordenados if t <= v < t + factor]
        if hojas:
            resultado.append(f"{tallo_str(t)} | " + " ".join(hojas))
    return "\n".join(resultado)

def grafico_puntos(datos, titulo="Grafico de Puntos"):
    """Genera un grafico de puntos (dot plot)"""
    if len(datos) < 2:
        return None

    df_puntos = pd.DataFrame({'Valor': datos})
    freq = df_puntos['Valor'].value_counts().sort_index().reset_index()
    freq.columns = ['Valor', 'Frecuencia']

    fig = go.Figure()
    for i, row in freq.iterrows():
        for j in range(int(row['Frecuencia'])):
            fig.add_trace(go.Scatter(
                x=[row['Valor']],
                y=[j + 0.5],
                mode='markers',
                marker=dict(size=12, color='#3498db', symbol='circle'),
                showlegend=False,
                hoverinfo='text',
                text=f"Valor: {row['Valor']}<br>Frecuencia: {row['Frecuencia']}"
            ))

    fig.update_layout(
        title=titulo,
        xaxis_title="Valor",
        yaxis_title="Frecuencia",
        height=400,
        plot_bgcolor='white',
        showlegend=False
    )
    return fig

# ============================================================
# FUNCION PARA GRAFICOS DE DATOS CUANTITATIVOS CON FRECUENCIAS
# ============================================================
def graficar_cuantitativos_frec(valores, frecuencias, titulo="Datos Cuantitativos con Frecuencias"):
    """Genera todos los graficos para datos cuantitativos con frecuencias"""
    datos_expandidos = np.repeat(valores, frecuencias.astype(int))
    n = int(np.sum(frecuencias))

    media = np.mean(datos_expandidos)
    mediana = np.median(datos_expandidos)
    moda = calcular_moda(datos_expandidos)
    if isinstance(moda, np.ndarray):
        moda_str = ', '.join([str(round(m, 4)) for m in moda])
    else:
        moda_str = f"{moda:.4f}" if isinstance(moda, (int, float)) else str(moda)

    varianza = np.var(datos_expandidos, ddof=1)
    desviacion = np.std(datos_expandidos, ddof=1)
    q1 = np.percentile(datos_expandidos, 25)
    q3 = np.percentile(datos_expandidos, 75)
    iqr = q3 - q1
    rango = np.max(datos_expandidos) - np.min(datos_expandidos)

    freq_rel = frecuencias / n * 100
    freq_acum = np.cumsum(frecuencias)
    freq_rel_acum = np.cumsum(freq_rel)

    df_frec = pd.DataFrame({
        'Valor (xi)': valores,
        'fi': frecuencias,
        'fir': np.round(frecuencias / n, 4),
        'fir (%)': np.round(freq_rel, 2),
        'Fi': freq_acum,
        'Fir': np.round(freq_acum / n, 4),
        'Fir (%)': np.round(freq_rel_acum, 2)
    })

    col1, col2 = st.columns(2)

    with col1:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=datos_expandidos,
            name="Datos",
            boxmean='sd',
            marker_color='#3498db',
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8
        ))
        fig_box.update_layout(
            title=f"Diagrama de Caja - {titulo}",
            height=400,
            yaxis_title="Valores",
            showlegend=False
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Resumen de Estadisticas")
        col_est1, col_est2 = st.columns(2)
        with col_est1:
            st.metric("Total de datos (n)", n)
            st.metric("Media", f"{media:.4f}")
            st.metric("Mediana", f"{mediana:.4f}")
            st.metric("Moda", moda_str)
        with col_est2:
            st.metric("Varianza", f"{varianza:.4f}")
            st.metric("Desviacion Estandar", f"{desviacion:.4f}")
            st.metric("Rango", f"{rango:.4f}")
            st.metric("IQR", f"{iqr:.4f}")

    with col2:
        st.subheader("Tabla de Frecuencias")
        st.dataframe(df_frec, use_container_width=True)

    st.subheader("Diagrama de Tallo y Hojas")
    tallo_hojas = grafico_tallo_hojas(datos_expandidos)
    if tallo_hojas:
        st.code(tallo_hojas, language="text")
    else:
        st.info("No hay suficientes datos para generar el diagrama de tallo y hojas.")

    st.subheader("Grafico de Puntos")
    fig_puntos = grafico_puntos(datos_expandidos, titulo)
    if fig_puntos:
        st.plotly_chart(fig_puntos, use_container_width=True)

    st.subheader("Histogramas de Frecuencias")

    tabs_hist = st.tabs([
        "Frecuencias Absolutas",
        "Frecuencias Relativas",
        "Frecuencias Acumuladas",
        "Frecuencias Relativas Acumuladas"
    ])

    etiquetas = [f"{v:.1f}" for v in valores]

    with tabs_hist[0]:
        fig_abs = go.Figure()
        fig_abs.add_trace(go.Bar(
            x=etiquetas,
            y=frecuencias,
            marker_color='#3498db',
            opacity=0.9,
            name='Frecuencia Absoluta',
            text=frecuencias.astype(int),
            textposition='outside'
        ))
        fig_abs.update_layout(
            title="Histograma de Frecuencias Absolutas",
            xaxis_title="Valor (xi)",
            yaxis_title="Frecuencia (fi)",
            height=400,
            showlegend=False,
            bargap=0,
            bargroupgap=0
        )
        st.plotly_chart(fig_abs, use_container_width=True)
        fig_pol_abs = go.Figure()
        fig_pol_abs.add_trace(go.Scatter(
            x=valores,
            y=frecuencias,
            mode='lines+markers',
            name='Poligono',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=10, color='#e74c3c')
        ))
        fig_pol_abs.update_layout(
            title="Poligono de Frecuencias Absolutas",
            xaxis_title="Valor (xi)",
            yaxis_title="Frecuencia (fi)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_pol_abs, use_container_width=True)

    with tabs_hist[1]:
        fig_rel = go.Figure()
        fig_rel.add_trace(go.Bar(
            x=etiquetas,
            y=freq_rel,
            marker_color='#2ecc71',
            opacity=0.9,
            name='Frecuencia Relativa (%)',
            text=[f"{f:.2f}%" for f in freq_rel],
            textposition='outside'
        ))
        fig_rel.update_layout(
            title="Histograma de Frecuencias Relativas",
            xaxis_title="Valor (xi)",
            yaxis_title="Frecuencia Relativa (%)",
            height=400,
            showlegend=False,
            bargap=0,
            bargroupgap=0
        )
        st.plotly_chart(fig_rel, use_container_width=True)
        fig_pol_rel = go.Figure()
        fig_pol_rel.add_trace(go.Scatter(
            x=valores,
            y=freq_rel,
            mode='lines+markers',
            name='Poligono',
            line=dict(color='#e67e22', width=2),
            marker=dict(size=10, color='#e67e22')
        ))
        fig_pol_rel.update_layout(
            title="Poligono de Frecuencias Relativas",
            xaxis_title="Valor (xi)",
            yaxis_title="Frecuencia Relativa (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_pol_rel, use_container_width=True)

    with tabs_hist[2]:
        fig_acum = go.Figure()
        fig_acum.add_trace(go.Bar(
            x=etiquetas,
            y=freq_acum,
            marker_color='#e74c3c',
            opacity=0.9,
            name='Frecuencia Acumulada',
            text=freq_acum.astype(int),
            textposition='outside'
        ))
        fig_acum.update_layout(
            title="Histograma de Frecuencias Acumuladas",
            xaxis_title="Valor (xi)",
            yaxis_title="Frecuencia Acumulada (Fi)",
            height=400,
            showlegend=False,
            bargap=0,
            bargroupgap=0
        )
        st.plotly_chart(fig_acum, use_container_width=True)
        fig_ojiva = go.Figure()
        fig_ojiva.add_trace(go.Scatter(
            x=valores,
            y=freq_acum,
            mode='lines+markers',
            name='Ojiva',
            line=dict(color='#8e44ad', width=3),
            marker=dict(size=10, color='#8e44ad')
        ))
        fig_ojiva.update_layout(
            title="Ojiva (Frecuencias Acumuladas)",
            xaxis_title="Valor (xi)",
            yaxis_title="Frecuencia Acumulada (Fi)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_ojiva, use_container_width=True)

    with tabs_hist[3]:
        fig_rel_acum = go.Figure()
        fig_rel_acum.add_trace(go.Bar(
            x=etiquetas,
            y=freq_rel_acum,
            marker_color='#f39c12',
            opacity=0.9,
            name='Frecuencia Relativa Acumulada (%)',
            text=[f"{f:.2f}%" for f in freq_rel_acum],
            textposition='outside'
        ))
        fig_rel_acum.update_layout(
            title="Histograma de Frecuencias Relativas Acumuladas",
            xaxis_title="Valor (xi)",
            yaxis_title="Frecuencia Relativa Acumulada (%)",
            height=400,
            showlegend=False,
            bargap=0,
            bargroupgap=0
        )
        st.plotly_chart(fig_rel_acum, use_container_width=True)
        fig_ojiva_pct = go.Figure()
        fig_ojiva_pct.add_trace(go.Scatter(
            x=valores,
            y=freq_rel_acum,
            mode='lines+markers',
            name='Ojiva (%)',
            line=dict(color='#1abc9c', width=3),
            marker=dict(size=10, color='#1abc9c')
        ))
        fig_ojiva_pct.update_layout(
            title="Ojiva Porcentual (Frecuencias Relativas Acumuladas)",
            xaxis_title="Valor (xi)",
            yaxis_title="Frecuencia Relativa Acumulada (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_ojiva_pct, use_container_width=True)

    return {
        'datos_expandidos': datos_expandidos,
        'df_frecuencias': df_frec,
        'media': media,
        'mediana': mediana,
        'moda': moda,
        'varianza': varianza,
        'desviacion': desviacion,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'rango': rango,
        'n': n
    }

# ============================================================
# FUNCION: ANALISIS COMPLETO PARA VARIABLES NUMERICAS
# ============================================================

def analisis_completo_variable_numerica(datos, nombre_variable="Variable"):
    """Realiza analisis completo de una variable numerica con todos los graficos y estadisticas avanzadas"""

    if len(datos) < 2:
        st.warning("No hay suficientes datos para el analisis.")
        return

    # Calcular todas las estadisticas avanzadas
    stats_avanzadas = calcular_estadisticas_avanzadas(datos)
    outliers_info = detectar_outliers(datos)

    # Mostrar estadisticas
    st.subheader(f"Estadisticas de {nombre_variable}")

    # Primera fila de estadisticas (basicas)
    col_est1, col_est2, col_est3, col_est4 = st.columns(4)
    with col_est1:
        st.metric("Count (n)", stats_avanzadas['n'])
        st.metric("Average", f"{stats_avanzadas['media']:.4f}")
        st.metric("Median", f"{stats_avanzadas['mediana']:.4f}")
        moda_val = stats_avanzadas['moda']
        if isinstance(moda_val, np.ndarray):
            moda_str = ', '.join([str(round(m, 4)) for m in moda_val])
        else:
            moda_str = f"{moda_val:.4f}" if isinstance(moda_val, (int, float)) else str(moda_val)
        st.metric("Mode", moda_str)

    with col_est2:
        st.metric("Geometric mean", f"{stats_avanzadas['media_geometrica']:.4f}" if not np.isnan(stats_avanzadas['media_geometrica']) else "N/A")
        st.metric("Harmonic mean", f"{stats_avanzadas['media_armonica']:.4f}" if not np.isnan(stats_avanzadas['media_armonica']) else "N/A")
        st.metric("5% Trimmed mean", f"{stats_avanzadas['media_truncada_5']:.4f}" if not np.isnan(stats_avanzadas['media_truncada_5']) else "N/A")
        st.metric("5% Winsorized mean", f"{stats_avanzadas['media_winsorizada_5']:.4f}" if not np.isnan(stats_avanzadas['media_winsorizada_5']) else "N/A")

    with col_est3:
        st.metric("Variance", f"{stats_avanzadas['varianza']:.4f}")
        st.metric("Standard deviation", f"{stats_avanzadas['desviacion']:.4f}")
        st.metric("Coeff. of variation", f"{stats_avanzadas['cv']:.4f}%" if not np.isnan(stats_avanzadas['cv']) else "N/A")
        st.metric("Gini coefficient", f"{stats_avanzadas['gini']:.8f}" if not np.isnan(stats_avanzadas['gini']) else "N/A")

    with col_est4:
        st.metric("Standard error", f"{stats_avanzadas['error_estandar']:.6f}" if not np.isnan(stats_avanzadas['error_estandar']) else "N/A")
        st.metric("Geometric std dev", f"{stats_avanzadas['gsd']:.5f}" if not np.isnan(stats_avanzadas['gsd']) else "N/A")
        st.metric("5% Winsorized sigma", f"{stats_avanzadas['sigma_winsor']:.5f}" if not np.isnan(stats_avanzadas['sigma_winsor']) else "N/A")
        st.metric("Mean absolute deviation", f"{stats_avanzadas['mad_media']:.8f}")

    # Segunda fila de estadisticas
    st.subheader("Medidas de Posición y Dispersión")
    col_pos1, col_pos2, col_pos3, col_pos4 = st.columns(4)

    with col_pos1:
        st.metric("MAD", f"{stats_avanzadas['mad']:.4f}")
        st.metric("Sbi", f"{stats_avanzadas['sbi']:.4f}" if not np.isnan(stats_avanzadas['sbi']) else "N/A")
        st.metric("Minimum", f"{stats_avanzadas['minimo']:.4f}")
        st.metric("Maximum", f"{stats_avanzadas['maximo']:.4f}")

    with col_pos2:
        st.metric("Range", f"{stats_avanzadas['rango']:.4f}")
        st.metric("Lower quartile (Q1)", f"{stats_avanzadas['q1']:.4f}")
        st.metric("Upper quartile (Q3)", f"{stats_avanzadas['q3']:.4f}")
        st.metric("Interquartile range (IQR)", f"{stats_avanzadas['iqr']:.4f}")

    with col_pos3:
        st.metric("1/6 sextile", f"{stats_avanzadas['sextil_1_6']:.4f}")
        st.metric("5/6 sextile", f"{stats_avanzadas['sextil_5_6']:.4f}")
        st.metric("Intersextile range (ISR)", f"{stats_avanzadas['isr']:.4f}")
        st.metric("Skewness", f"{stats_avanzadas['skewness']:.5f}" if not np.isnan(stats_avanzadas['skewness']) else "N/A")

    with col_pos4:
        st.metric("Stnd. skewness", f"{stats_avanzadas['skewness_std']:.5f}" if not np.isnan(stats_avanzadas['skewness_std']) else "N/A")
        st.metric("Kurtosis", f"{stats_avanzadas['kurtosis']:.6f}" if not np.isnan(stats_avanzadas['kurtosis']) else "N/A")
        st.metric("Stnd. kurtosis", f"{stats_avanzadas['kurtosis_std']:.5f}" if not np.isnan(stats_avanzadas['kurtosis_std']) else "N/A")
        st.metric("Sum", f"{stats_avanzadas['suma']:.4f}")

    st.metric("Sum of squares", f"{stats_avanzadas['suma_cuadrados']:.4f}")

    # ============================================================
    # HISTOGRAMA INTERACTIVO
    # ============================================================
    histograma_interactivo(datos, nombre_variable)

    # PERCENTILES PERSONALIZADOS
    st.subheader("Percentiles Personalizados")

    col_perc1, col_perc2, col_perc3 = st.columns([2, 2, 1])
    with col_perc1:
        perc_rapidos = st.multiselect(
            "Percentiles rápidos:",
            [5, 10, 20, 30, 40, 60, 70, 80, 90, 95],
            default=[10, 90],
            key=f"perc_rapidos_{nombre_variable}"
        )
    with col_perc2:
        perc_manual = st.text_input(
            "O ingresa percentiles separados por comas:",
            placeholder="Ejemplo: 15, 25, 75, 85",
            key=f"perc_manual_{nombre_variable}"
        )
    with col_perc3:
        if st.button("Calcular Percentiles", key=f"btn_percentiles_{nombre_variable}"):
            percentiles_a_calcular = set(perc_rapidos)

            if perc_manual.strip():
                try:
                    manuales = [float(p.strip()) for p in perc_manual.split(',') if p.strip()]
                    for p in manuales:
                        if 0 <= p <= 100:
                            percentiles_a_calcular.add(p)
                except:
                    st.error("Error en el formato de percentiles manuales")

            if percentiles_a_calcular:
                resultados_percentiles = calcular_percentiles_personalizados(datos, sorted(percentiles_a_calcular))

                st.subheader("Resultados de Percentiles")
                cols_perc = st.columns(min(4, len(resultados_percentiles)))
                for i, (p, valor) in enumerate(sorted(resultados_percentiles.items())):
                    with cols_perc[i % len(cols_perc)]:
                        st.metric(f"Percentil {p}%", f"{valor:.4f}")

                df_percentiles = pd.DataFrame({
                    'Percentil (%)': list(resultados_percentiles.keys()),
                    'Valor': [f"{v:.4f}" for v in resultados_percentiles.values()]
                })
                st.dataframe(df_percentiles, use_container_width=True)

                fig_perc = go.Figure()
                fig_perc.add_trace(go.Scatter(
                    x=list(resultados_percentiles.keys()),
                    y=list(resultados_percentiles.values()),
                    mode='lines+markers',
                    name='Percentiles',
                    line=dict(color='#9b59b6', width=2),
                    marker=dict(size=12, color='#9b59b6')
                ))
                fig_perc.update_layout(
                    title="Curva de Percentiles",
                    xaxis_title="Percentil (%)",
                    yaxis_title="Valor",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig_perc, use_container_width=True)

    if len(outliers_info['outliers']) > 0:
        st.warning(f"Outliers detectados: {', '.join([str(round(o, 2)) for o in outliers_info['outliers']])}")
        st.info(f"Limite inferior: {outliers_info['limite_inferior']:.4f} | Limite superior: {outliers_info['limite_superior']:.4f}")

    # Tallo y Hojas
    st.subheader("Diagrama de Tallo y Hojas")
    tallo_hojas = grafico_tallo_hojas(datos)
    if tallo_hojas:
        st.code(tallo_hojas, language="text")
    else:
        st.info("No hay suficientes datos para generar el diagrama de tallo y hojas.")

    # Grafico de Puntos
    st.subheader("Grafico de Puntos")
    fig_puntos = grafico_puntos(datos, f"Grafico de Puntos - {nombre_variable}")
    if fig_puntos:
        st.plotly_chart(fig_puntos, use_container_width=True)

    # Diagrama de Caja
    st.subheader("Diagrama de Caja (Boxplot)")
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=datos,
        name=nombre_variable,
        boxmean='sd',
        marker_color='#3498db',
        boxpoints='outliers',
        jitter=0.3,
        pointpos=-1.8
    ))
    fig_box.update_layout(
        title=f"Diagrama de Caja - {nombre_variable}",
        height=400,
        yaxis_title="Valores",
        showlegend=False
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Tabla de Frecuencias (usando regla de Sturges)
    st.subheader("Tabla de Frecuencias (Sturges)")

    n_clases = calcular_n_clases_sturges(stats_avanzadas['n'])
    hist_data = np.histogram(datos, bins=n_clases)

    clases = []
    limites_inf = []
    limites_sup = []
    for i in range(len(hist_data[0])):
        lim_inf = hist_data[1][i]
        lim_sup = hist_data[1][i+1]
        limites_inf.append(lim_inf)
        limites_sup.append(lim_sup)
        if i == len(hist_data[0]) - 1:
            clase = f"[{lim_inf:.2f}, {lim_sup:.2f}]"
        else:
            clase = f"[{lim_inf:.2f}, {lim_sup:.2f})"
        clases.append(clase)

    marcas_clase = (np.array(limites_inf) + np.array(limites_sup)) / 2
    frecuencias = hist_data[0].tolist()
    freq_acum = np.cumsum(frecuencias).tolist()
    freq_rel = [f / stats_avanzadas['n'] * 100 for f in frecuencias]
    freq_rel_acum = np.cumsum(freq_rel).tolist()

    df_frec = pd.DataFrame({
        'Clase': clases,
        'f': frecuencias,
        'fr(%)': [round(f, 2) for f in freq_rel],
        'F': freq_acum,
        'F(%)': [round(f, 2) for f in freq_rel_acum],
        'Marca': [round(m, 2) for m in marcas_clase]
    })
    st.dataframe(df_frec, use_container_width=True)

# ============================================================
# INTERFAZ DE USUARIO - BARRA LATERAL
# ============================================================

st.sidebar.header("Carga de Datos")

tipo_carga = st.sidebar.radio(
    "Selecciona el metodo de carga:",
    ["Ingresar manualmente", "Subir archivo CSV", "Desde URL"]
)

df_cargado = None

if tipo_carga == "Subir archivo CSV":
    archivo = st.sidebar.file_uploader("Selecciona un archivo CSV", type=['csv', 'txt'])
    if archivo is not None:
        with st.sidebar.expander("Opciones avanzadas"):
            separador_manual = st.selectbox(
                "Separador:",
                ['Auto-detectar', ',', ';', '\t', '|', ' ']
            )
        if st.sidebar.button("Cargar archivo"):
            with st.spinner("Cargando archivo..."):
                sep = None if separador_manual == 'Auto-detectar' else separador_manual
                df_cargado = cargar_datos_desde_csv(archivo, sep)
                if df_cargado is not None:
                    st.session_state.df_cargado = df_cargado
                    st.sidebar.success(f"Archivo cargado: {archivo.name}")
                    st.sidebar.info(f"{len(df_cargado)} filas, {len(df_cargado.columns)} columnas")

elif tipo_carga == "Desde URL":
    url = st.sidebar.text_input("URL del archivo CSV:",
                                placeholder="https://ejemplo.com/datos.csv")
    if url and st.sidebar.button("Cargar desde URL"):
        with st.spinner("Cargando datos desde URL..."):
            df_cargado = cargar_datos_desde_url(url)
            if df_cargado is not None:
                st.session_state.df_cargado = df_cargado
                st.sidebar.success("Datos cargados desde URL")
                st.sidebar.info(f"{len(df_cargado)} filas, {len(df_cargado.columns)} columnas")

# ============================================================
# FUNCION PARA ANALISIS DESCRIPTIVO DE DATAFRAMES CARGADOS
# ============================================================

def mostrar_analisis_descriptivo(df):
    """Muestra analisis descriptivo completo de un DataFrame"""
    st.subheader("Analisis Descriptivo del DataFrame")

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1: st.metric("Filas", len(df))
    with col_info2: st.metric("Columnas", len(df.columns))
    with col_info3: st.metric("Memoria", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    st.subheader("Tipos de Datos")
    tipos_df = pd.DataFrame({
        'Columna': df.columns, 'Tipo': df.dtypes.astype(str),
        'Valores Nulos': df.isnull().sum().values,
        'Porcentaje Nulos': (df.isnull().sum() / len(df) * 100).round(2).values
    })
    st.dataframe(tipos_df, use_container_width=True)

    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # SECCION: ANALISIS BIVARIADO
    st.subheader("Analisis Bivariado (Tablas Cruzadas)")
    todas_columnas = df.columns.tolist()
    if len(todas_columnas) >= 2:
        col1_biv, col2_biv = st.columns(2)
        with col1_biv: var1 = st.selectbox("Selecciona la primera variable:", todas_columnas, key="biv_var1")
        with col2_biv: var2 = st.selectbox("Selecciona la segunda variable:", todas_columnas, key="biv_var2")
        if var1 != var2 and st.button("Analizar Relacion Bivariada", key="btn_bivariado"):
            resultados = analisis_bivariado(df, var1, var2)
            st.success(f"Analisis completado: {resultados['tipo']}")
            if resultados['tipo'] == 'Cualitativa-Cualitativa':
                st.subheader("Tabla de Contingencia")
                st.dataframe(resultados['tabla_contingencia'], use_container_width=True)
                st.subheader("Mapa de Calor")
                tabla_sin_margenes = pd.crosstab(df[var1], df[var2])
                fig_heatmap = px.imshow(tabla_sin_margenes, text_auto=True, aspect="auto", color_continuous_scale='Blues')
                st.plotly_chart(fig_heatmap, use_container_width=True)
                st.subheader("Diagrama de Mosaico")
                tabla_pct = pd.crosstab(df[var1], df[var2], normalize='index') * 100
                df_stacked = tabla_pct.reset_index().melt(id_vars=var1)
                df_stacked.columns = ['Variable1', 'Variable2', 'Porcentaje']
                fig_stacked = px.bar(df_stacked, x='Variable1', y='Porcentaje', color='Variable2', title="Diagrama de Mosaico (Barras 100% apiladas)", barmode='stack', text=[f"{v:.1f}%" for v in df_stacked['Porcentaje']])
                fig_stacked.update_traces(textposition='inside'); fig_stacked.update_layout(height=400)
                st.plotly_chart(fig_stacked, use_container_width=True)
                if 'chi2' in resultados['estadisticos']:
                    st.metric("Chi2", f"{resultados['estadisticos']['chi2']:.4f}")
                    st.metric("p-valor", f"{resultados['estadisticos']['p_valor']:.4f}")
            elif resultados['tipo'] == 'Cuantitativa-Cuantitativa':
                st.subheader("Estadisticos de Correlacion")
                col_corr1, col_corr2 = st.columns(2)
                with col_corr1:
                    if 'pearson_r' in resultados['estadisticos']:
                        st.metric("Correlacion de Pearson (r)", f"{resultados['estadisticos']['pearson_r']:.4f}")
                        st.caption(f"p-valor: {resultados['estadisticos']['pearson_p']:.4f}")
                    if 'r2' in resultados['estadisticos']:
                        st.metric("R2", f"{resultados['estadisticos']['r2']:.4f}")
                with col_corr2:
                    if 'spearman_r' in resultados['estadisticos']:
                        st.metric("Correlacion de Spearman (rho)", f"{resultados['estadisticos']['spearman_r']:.4f}")
                        st.caption(f"p-valor: {resultados['estadisticos']['spearman_p']:.4f}")
                st.subheader("Dispersion y Regresion")
                datos = df[[var1, var2]].dropna()
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(x=datos[var1], y=datos[var2], mode='markers', name='Datos', marker=dict(color='#3498db', size=8)))
                if 'pendiente' in resultados['estadisticos']:
                    x_range = np.linspace(datos[var1].min(), datos[var1].max(), 100)
                    y_reg = resultados['estadisticos']['intercepto'] + resultados['estadisticos']['pendiente'] * x_range
                    fig_scatter.add_trace(go.Scatter(x=x_range, y=y_reg, mode='lines', name='Regresion', line=dict(color='#e74c3c', width=2)))
                    fig_scatter.update_layout(title=f"Y = {resultados['estadisticos']['intercepto']:.4f} + {resultados['estadisticos']['pendiente']:.4f}X", xaxis_title=var1, yaxis_title=var2, height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
            elif resultados['tipo'] == 'Cualitativa-Cuantitativa':
                if pd.api.types.is_numeric_dtype(df[var1]): var_cat = var2; var_cuant = var1
                else: var_cat = var1; var_cuant = var2
                st.subheader("Estadisticos por Categoria")
                stats_cat_df = pd.DataFrame(resultados['estadisticos_cat']).T.round(4)
                st.dataframe(stats_cat_df, use_container_width=True)
                st.subheader("Boxplots por Categoria")
                fig_box = px.box(df[[var_cat, var_cuant]].dropna(), x=var_cat, y=var_cuant, color=var_cat, title="Boxplots por Categoria", points="all")
                fig_box.update_layout(height=400); st.plotly_chart(fig_box, use_container_width=True)
                if 'anova_f' in resultados['estadisticos']:
                    st.metric("ANOVA F", f"{resultados['estadisticos']['anova_f']:.4f}")

    # ============================================================
    # ANALISIS UNIVARIADO (NUMERICO) - CON REGLA DE STURGES
    # ============================================================
    if columnas_numericas:
        st.subheader("Analisis de Variables Numericas")
        columna_seleccionada = st.selectbox("Selecciona una variable numerica:", columnas_numericas)

        if columna_seleccionada:
            datos = df[columna_seleccionada].dropna()
            if len(datos) > 0:
                analisis_completo_variable_numerica(datos, columna_seleccionada)
            else:
                st.warning("La variable seleccionada no tiene datos validos.")

    # ANALISIS UNIVARIADO (CATEGORICO)
    if columnas_categoricas:
        st.subheader("Analisis de Variables Categoricas")
        if st.session_state.columna_cat_seleccionada is None or st.session_state.columna_cat_seleccionada not in columnas_categoricas:
            st.session_state.columna_cat_seleccionada = columnas_categoricas[0]

        columna_cat = st.selectbox(
            "Selecciona una variable categorica:",
            columnas_categoricas,
            index=columnas_categoricas.index(st.session_state.columna_cat_seleccionada) if st.session_state.columna_cat_seleccionada in columnas_categoricas else 0
        )
        st.session_state.columna_cat_seleccionada = columna_cat

        datos_cat = df[columna_cat].dropna()
        if len(datos_cat) > 0:
            freq_table = datos_cat.value_counts().reset_index()
            freq_table.columns = ['Categoria', 'Frecuencia']
            freq_table['Porcentaje'] = (freq_table['Frecuencia'] / len(datos_cat) * 100).round(2)

            st.subheader("Tabla de Frecuencias")
            st.dataframe(freq_table, use_container_width=True)

            st.subheader("Visualizaciones")
            col_cat_graf1, col_cat_graf2 = st.columns(2)

            with col_cat_graf1:
                fig_bar = px.bar(
                    freq_table,
                    x='Categoria',
                    y='Frecuencia',
                    color='Categoria',
                    text='Porcentaje',
                    title='Grafico de Barras'
                )
                fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_bar.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_cat_graf2:
                fig_pie = px.pie(
                    freq_table,
                    values='Frecuencia',
                    names='Categoria',
                    title='Grafico Circular',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textinfo='label+percent')
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)

# ============================================================
# FLUJO PRINCIPAL DE LA APP
# ============================================================

if st.session_state.df_cargado is not None:
    df_cargado = st.session_state.df_cargado
    mostrar_analisis_descriptivo(df_cargado)
    st.markdown("---")
else:
    st.info("Carga un archivo CSV o URL para comenzar el analisis.")

# ============================================================
# PESTANAS COMPLETAS (ANALISIS MANUAL FUNCIONAL)
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Datos en Lista",
    "Datos Agrupados",
    "Datos Cuantitativos con Frecuencias",
    "Cualitativas (Datos)",
    "Cualitativas (Frecuencias)",
    "Tabla de Contingencia",
    "Guia de Interpretacion"
])

# ============================================================
# PESTANA 1: DATOS EN LISTA - CON ANALISIS COMPLETO
# ============================================================

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Ingreso de Datos")
        datos_texto = st.text_area("Ingresa los datos separados por comas o espacios", placeholder="Ejemplo: 12, 15, 18, 20, 22", height=150)
        if st.button("Calcular", key="calcular_lista"):
            datos = procesar_datos_lista(datos_texto)
            if datos is None:
                st.error("Error: Ingresa al menos 2 numeros validos.")
            else:
                st.session_state.resultados_lista = datos
                st.success("Calculo completado exitosamente.")
    
    with col2:
        if st.session_state.resultados_lista is not None:
            datos = st.session_state.resultados_lista
            analisis_completo_variable_numerica(datos, "Datos en Lista")
            
            if st.button("🗑️ Limpiar resultados", key="limpiar_lista"):
                st.session_state.resultados_lista = None
                st.rerun()

# ============================================================
# PESTANA 2: DATOS AGRUPADOS - CON MEDIDAS DESCRIPTIVAS
# ============================================================

with tab2:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Ingreso de Datos Agrupados")
        st.caption("Formato: clase1, frecuencia1; clase2, frecuencia2; ...")
        datos_agrupados_texto = st.text_area("Ingresa los datos agrupados", placeholder="Ejemplo: [10,20), 5; [20,30), 8; [30,40), 12", height=150)
        if st.button("Calcular", key="calcular_agrupados"):
            resultado = procesar_datos_agrupados(datos_agrupados_texto)
            if resultado is None:
                st.error("Error: Ingresa al menos 2 clases con frecuencias validas.")
            else:
                clases = resultado['clases']; frecuencias = resultado['frecuencias']; marcas_clase = resultado['marcas_clase']
                limites_inferiores = resultado['limites_inferiores']; limites_superiores = resultado['limites_superiores']
                n = sum(frecuencias); media = np.average(marcas_clase, weights=frecuencias)
                
                # Calcular Mediana
                n_mitad = n / 2; acumulado = 0; clase_mediana = 0
                for i, f in enumerate(frecuencias):
                    acumulado += f
                    if acumulado >= n_mitad: 
                        clase_mediana = i
                        break
                f_anterior = sum(frecuencias[:clase_mediana]) if clase_mediana > 0 else 0
                f_clase = frecuencias[clase_mediana]
                L_mediana = limites_inferiores[clase_mediana]
                A_mediana = limites_superiores[clase_mediana] - limites_inferiores[clase_mediana]
                mediana = L_mediana + ((n_mitad - f_anterior) / f_clase) * A_mediana
                
                # Calcular Moda
                clase_modal = np.argmax(frecuencias)
                f_modal = frecuencias[clase_modal]
                f_pre = frecuencias[clase_modal - 1] if clase_modal > 0 else 0
                f_post = frecuencias[clase_modal + 1] if clase_modal < len(frecuencias) - 1 else 0
                L_modal = limites_inferiores[clase_modal]
                A_modal = limites_superiores[clase_modal] - limites_inferiores[clase_modal]
                d1 = f_modal - f_pre
                d2 = f_modal - f_post
                moda = L_modal + (d1 / (d1 + d2)) * A_modal if (d1 + d2) > 0 else marcas_clase[clase_modal]
                
                # Calcular Varianza y Desviación
                varianza = np.average((np.array(marcas_clase) - media)**2, weights=frecuencias) * n / (n - 1)
                desviacion = np.sqrt(varianza)
                rango = limites_superiores[-1] - limites_inferiores[0]
                
                # Calcular Cuantiles
                def calcular_cuantil(p):
                    pos = p * n
                    acumulado = 0
                    clase_cuantil = 0
                    for i, f in enumerate(frecuencias):
                        acumulado += f
                        if acumulado >= pos:
                            clase_cuantil = i
                            break
                    f_ant = sum(frecuencias[:clase_cuantil]) if clase_cuantil > 0 else 0
                    f_clase = frecuencias[clase_cuantil]
                    L_cuantil = limites_inferiores[clase_cuantil]
                    A_cuantil = limites_superiores[clase_cuantil] - limites_inferiores[clase_cuantil]
                    return L_cuantil + ((pos - f_ant) / f_clase) * A_cuantil
                
                q1 = calcular_cuantil(0.25)
                q3 = calcular_cuantil(0.75)
                iqr = q3 - q1
                
                # Frecuencias relativas y acumuladas
                freq_rel = [f/n*100 for f in frecuencias]
                freq_acum = np.cumsum(frecuencias)
                freq_rel_acum = np.cumsum(freq_rel)
                
                # Formatear clases
                clases_formateadas = []
                for i in range(len(clases)):
                    if i == len(clases) - 1:
                        clases_formateadas.append(f"[{limites_inferiores[i]:.2f}, {limites_superiores[i]:.2f}]")
                    else:
                        clases_formateadas.append(f"[{limites_inferiores[i]:.2f}, {limites_superiores[i]:.2f})")
                
                # Guardar en session state incluyendo límites para percentiles
                st.session_state['resultados_agrupados'] = {
                    'clases': clases_formateadas,
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
                    'rango': rango,
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr,
                    'freq_rel': freq_rel,
                    'freq_acum': freq_acum.tolist(),
                    'freq_rel_acum': freq_rel_acum.tolist()
                }
                st.success(f"Calculo completado. Total de datos: {n}")
    
    with col2:
        if 'resultados_agrupados' in st.session_state:
            res = st.session_state['resultados_agrupados']
            
            # ============================================================
            # MEDIDAS DESCRIPTIVAS PARA DATOS AGRUPADOS
            # ============================================================
            st.subheader("📊 Medidas Descriptivas para Datos Agrupados")
            
            col1_m, col2_m, col3_m = st.columns(3)
            
            with col1_m:
                st.metric("📌 Total de datos (n)", res['n'])
                st.metric("📊 Media", f"{res['media']:.4f}")
                st.metric("📊 Mediana", f"{res['mediana']:.4f}")
                st.metric("📊 Moda (aproximada)", f"{res['moda']:.4f}")
            
            with col2_m:
                st.metric("📊 Varianza", f"{res['varianza']:.4f}")
                st.metric("📊 Desviación Estándar", f"{res['desviacion']:.4f}")
                st.metric("📊 Q1 (Cuartil 1)", f"{res['q1']:.4f}")
                st.metric("📊 Q3 (Cuartil 3)", f"{res['q3']:.4f}")
            
            with col3_m:
                st.metric("📊 Rango", f"{res['rango']:.4f}")
                st.metric("📊 IQR", f"{res['iqr']:.4f}")
                # Calcular percentiles adicionales usando la función de cuantil
                def calc_cuantil(p):
                    pos = p * res['n']
                    acumulado = 0
                    clase_cuantil = 0
                    for i, f in enumerate(res['frecuencias']):
                        acumulado += f
                        if acumulado >= pos:
                            clase_cuantil = i
                            break
                    f_ant = sum(res['frecuencias'][:clase_cuantil]) if clase_cuantil > 0 else 0
                    f_clase = res['frecuencias'][clase_cuantil]
                    L_cuantil = res['limites_inferiores'][clase_cuantil]
                    A_cuantil = res['limites_superiores'][clase_cuantil] - res['limites_inferiores'][clase_cuantil]
                    return L_cuantil + ((pos - f_ant) / f_clase) * A_cuantil
                
                st.metric("📊 Percentil 10", f"{calc_cuantil(0.10):.4f}")
                st.metric("📊 Percentil 90", f"{calc_cuantil(0.90):.4f}")
            
            # ============================================================
            # PERCENTILES PERSONALIZADOS PARA DATOS AGRUPADOS
            # ============================================================
            st.subheader("📊 Percentiles Personalizados para Datos Agrupados")
            
            col_perc1, col_perc2 = st.columns(2)
            with col_perc1:
                percentiles_seleccionados = st.multiselect(
                    "Selecciona percentiles:",
                    [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95],
                    default=[10, 25, 50, 75, 90],
                    key="percentiles_agrupados"
                )
            with col_perc2:
                if st.button("Calcular Percentiles Agrupados", key="calc_percentiles_agrupados"):
                    if percentiles_seleccionados:
                        st.write("**Resultados:**")
                        cols = st.columns(min(4, len(percentiles_seleccionados)))
                        for i, p in enumerate(sorted(percentiles_seleccionados)):
                            with cols[i % len(cols)]:
                                valor = calc_cuantil(p/100)
                                st.metric(f"Percentil {p}%", f"{valor:.4f}")
            
            # ============================================================
            # TABLA DE FRECUENCIAS
            # ============================================================
            st.subheader("Tabla de Frecuencias")
            df_frec = pd.DataFrame({
                'Clase': res['clases'],
                'Marca': [round(m, 2) for m in res['marcas_clase']],
                'f': res['frecuencias'],
                'fr(%)': [round(f, 2) for f in res['freq_rel']],
                'F': res['freq_acum'],
                'F(%)': [round(f, 2) for f in res['freq_rel_acum']]
            })
            st.dataframe(df_frec, use_container_width=True)
            
            # ============================================================
            # VISUALIZACIONES
            # ============================================================
            st.subheader("Visualizaciones")
            
            # Histograma con barras unidas
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(
                x=res['clases'],
                y=res['frecuencias'],
                marker_color='#3498db',
                opacity=0.9,
                name='Histograma'
            ))
            fig_hist.update_layout(
                title='Histograma de Frecuencias',
                height=350,
                xaxis_title="Clases",
                yaxis_title="Frecuencia",
                bargap=0,
                bargroupgap=0
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Poligono y Ojiva
            fig_pol_ojiva = make_subplots(rows=1, cols=2, subplot_titles=('Poligono', 'Ojiva'))
            fig_pol_ojiva.add_trace(go.Scatter(
                x=res['marcas_clase'], 
                y=res['frecuencias'], 
                mode='lines+markers', 
                name='Poligono', 
                line=dict(color='#e74c3c', width=2)
            ), row=1, col=1)
            fig_pol_ojiva.add_trace(go.Scatter(
                x=res['marcas_clase'], 
                y=res['freq_acum'], 
                mode='lines+markers', 
                name='Ojiva', 
                line=dict(color='#2ecc71', width=2)
            ), row=1, col=2)
            fig_pol_ojiva.update_layout(height=400)
            st.plotly_chart(fig_pol_ojiva, use_container_width=True)
            
            # Botón para limpiar
            if st.button("🗑️ Limpiar resultados", key="limpiar_agrupados"):
                del st.session_state['resultados_agrupados']
                st.rerun()

# ============================================================
# PESTANA 3: DATOS CUANTITATIVOS CON FRECUENCIAS - CON ANALISIS COMPLETO
# ============================================================

with tab3:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Ingreso de Datos Cuantitativos con Frecuencias")
        st.caption("Formato: valor1, frecuencia1; valor2, frecuencia2; ...")
        st.caption("Ejemplo: 12, 5; 15, 8; 18, 12; 20, 6; 22, 4")
        datos_cuant_frec_texto = st.text_area(
            "Ingresa los valores y sus frecuencias:",
            placeholder="12, 5; 15, 8; 18, 12; 20, 6; 22, 4",
            height=150
        )

        if st.button("Calcular", key="calcular_cuant_frec"):
            resultado = procesar_datos_cuantitativos_frec(datos_cuant_frec_texto)
            if resultado is None:
                st.error("Error: Ingresa al menos 2 pares (valor, frecuencia) validos.")
            else:
                # Expandir datos para análisis completo
                datos_expandidos = np.repeat(resultado['valores'], resultado['frecuencias'].astype(int))
                st.session_state.resultados_cuant_frec = datos_expandidos
                st.success(f"Datos cargados. Total de datos: {int(np.sum(resultado['frecuencias']))}")
    
    with col2:
        if st.session_state.resultados_cuant_frec is not None:
            datos = st.session_state.resultados_cuant_frec
            analisis_completo_variable_numerica(datos, "Datos Cuantitativos con Frecuencias")
            
            if st.button("🗑️ Limpiar resultados", key="limpiar_cuant_frec"):
                st.session_state.resultados_cuant_frec = None
                st.rerun()

# ============================================================
# PESTANA 4: CUALITATIVAS (DATOS)
# ============================================================

with tab4:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Ingreso de Datos Cualitativos")
        st.caption("Ingresa las categorias separadas por comas")
        datos_cuali_texto = st.text_area("Datos cualitativos", placeholder="Ejemplo: Rojo, Azul, Rojo, Verde, Azul, Azul", height=150)
        if st.button("Calcular", key="calcular_cuali"):
            categorias = procesar_datos_cualitativos(datos_cuali_texto)
            if categorias is None:
                st.error("Error: Ingresa al menos 2 categorias validas.")
            else:
                valores, conteos = np.unique(categorias, return_counts=True); n = len(categorias)
                porcentajes = conteos / n * 100
                max_freq = max(conteos); modas = valores[conteos == max_freq]
                st.session_state['resultados_cuali'] = {
                    'nombres': valores.tolist(), 'frecuencias': conteos.tolist(),
                    'porcentajes': porcentajes.tolist(), 'n': n, 'modas': modas.tolist(), 'max_freq': max_freq
                }
                st.success(f"Calculo completado. Total de datos: {n}")
    with col2:
        if 'resultados_cuali' in st.session_state:
            res = st.session_state['resultados_cuali']
            st.subheader("Resumen de Resultados")
            col_res1, col_res2 = st.columns(2)
            with col_res1: st.metric("Total de datos", res['n']); st.metric("Numero de categorias", len(res['nombres']))
            with col_res2: modas_str = ", ".join(res['modas']); st.metric("Moda", f"{modas_str} (f: {res['max_freq']})")
            st.subheader("Tabla de Frecuencias")
            df_cuali = pd.DataFrame({'Categoria': res['nombres'], 'Frecuencia': res['frecuencias'], 'Porcentaje': [round(p, 2) for p in res['porcentajes']]})
            st.dataframe(df_cuali, use_container_width=True)
            st.subheader("Visualizaciones")
            col_graf1, col_graf2 = st.columns(2)
            with col_graf1:
                fig_bar = px.bar(df_cuali, x='Categoria', y='Frecuencia', color='Categoria', text=[f"{f} ({round(p, 1)}%)" for f, p in zip(res['frecuencias'], res['porcentajes'])], title='Grafico de Barras')
                fig_bar.update_traces(textposition='outside'); fig_bar.update_layout(showlegend=False, height=400); st.plotly_chart(fig_bar, use_container_width=True)
            with col_graf2:
                fig_pie = px.pie(df_cuali, values='Frecuencia', names='Categoria', title='Grafico Circular', color_discrete_sequence=px.colors.qualitative.Set3)
                fig_pie.update_traces(textinfo='label+percent'); fig_pie.update_layout(height=400); st.plotly_chart(fig_pie, use_container_width=True)
            
            if st.button("🗑️ Limpiar resultados", key="limpiar_cuali"):
                del st.session_state['resultados_cuali']
                st.rerun()

# ============================================================
# PESTANA 5: CUALITATIVAS (FRECUENCIAS)
# ============================================================

with tab5:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Ingreso de Categorias con Frecuencias")
        st.caption("Formato: categoria1, frecuencia1; categoria2, frecuencia2; ...")
        datos_cuali_frec = st.text_area("Datos con frecuencias", placeholder="Ejemplo: Rojo, 10; Verde, 35; Azul, 55", height=150)
        if st.button("Calcular", key="calcular_cuali_frec"):
            resultado = procesar_cuali_frec(datos_cuali_frec)
            if resultado is None:
                st.error("Error: Ingresa al menos 2 categorias con frecuencias validas.")
            else:
                n = sum(resultado['frecuencias']); porcentajes = [f/n*100 for f in resultado['frecuencias']]
                max_freq = max(resultado['frecuencias']); modas = [resultado['categorias'][i] for i, f in enumerate(resultado['frecuencias']) if f == max_freq]
                st.session_state['resultados_cuali_frec'] = {
                    'categorias': resultado['categorias'], 'frecuencias': resultado['frecuencias'],
                    'porcentajes': porcentajes, 'n': n, 'modas': modas, 'max_freq': max_freq
                }
                st.success(f"Calculo completado. Total de datos: {n}")
    with col2:
        if 'resultados_cuali_frec' in st.session_state:
            res = st.session_state['resultados_cuali_frec']
            st.subheader("Resumen de Resultados")
            col_res1, col_res2 = st.columns(2)
            with col_res1: st.metric("Total de datos", res['n']); st.metric("Numero de categorias", len(res['categorias']))
            with col_res2: modas_str = ", ".join(res['modas']); st.metric("Moda", f"{modas_str} (f: {res['max_freq']})")
            st.subheader("Tabla de Frecuencias")
            df_cuali_frec = pd.DataFrame({'Categoria': res['categorias'], 'Frecuencia': res['frecuencias'], 'Porcentaje': [round(p, 2) for p in res['porcentajes']]})
            st.dataframe(df_cuali_frec, use_container_width=True)
            st.subheader("Visualizaciones")
            col_graf1, col_graf2 = st.columns(2)
            with col_graf1:
                fig_bar = px.bar(df_cuali_frec, x='Categoria', y='Frecuencia', color='Categoria', text=[f"{f} ({round(p, 1)}%)" for f, p in zip(res['frecuencias'], res['porcentajes'])], title='Grafico de Barras')
                fig_bar.update_traces(textposition='outside'); fig_bar.update_layout(showlegend=False, height=400); st.plotly_chart(fig_bar, use_container_width=True)
            with col_graf2:
                fig_pie = px.pie(df_cuali_frec, values='Frecuencia', names='Categoria', title='Grafico Circular', color_discrete_sequence=px.colors.qualitative.Set3)
                fig_pie.update_traces(textinfo='label+percent'); fig_pie.update_layout(height=400); st.plotly_chart(fig_pie, use_container_width=True)
            
            if st.button("🗑️ Limpiar resultados", key="limpiar_cuali_frec"):
                del st.session_state['resultados_cuali_frec']
                st.rerun()

# ============================================================
# PESTANA 6: TABLA DE CONTINGENCIA
# ============================================================

with tab6:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Ingreso de Tabla de Contingencia")
        st.caption("Formato: fila1_col1, valor; fila1_col2, valor; ...")
        datos_contingencia = st.text_area("Datos de la tabla", placeholder='Ejemplo: Hombre_Matematicas, 15; Hombre_Ingenieria, 30; Mujer_Matematicas, 15; Mujer_Ingenieria, 20', height=150)
        nombres_filas = st.text_input("Nombres de filas (separados por comas):", value="Hombre, Mujer")
        nombres_columnas = st.text_input("Nombres de columnas (separados por comas):", value="Matematicas, Ingenieria")
        if st.button("Calcular", key="calcular_contingencia"):
            resultado = procesar_contingencia(datos_contingencia, nombres_filas, nombres_columnas)
            if resultado is None:
                st.error("Error: Verifica el formato de la tabla de contingencia.")
            else:
                st.session_state['resultados_contingencia'] = resultado
                st.success("Tabla de contingencia calculada exitosamente.")
    with col2:
        if 'resultados_contingencia' in st.session_state:
            res = st.session_state['resultados_contingencia']
            st.subheader("Resumen")
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1: st.metric("Filas", len(res['filas']))
            with col_res2: st.metric("Columnas", len(res['columnas']))
            with col_res3: st.metric("Total", int(np.sum(res['matriz'])))
            df_cont = pd.DataFrame(res['matriz'], index=res['filas'], columns=res['columnas'])
            df_cont['Total Fila'] = df_cont.sum(axis=1); df_cont.loc['Total Columna'] = df_cont.sum()
            st.dataframe(df_cont, use_container_width=True)
            st.subheader("Visualizaciones")
            df_long = pd.DataFrame(res['matriz'], index=res['filas'], columns=res['columnas']).reset_index().melt(id_vars='index', var_name='Columna', value_name='Frecuencia')
            df_long.columns = ['Fila', 'Columna', 'Frecuencia']
            graf_tab1, graf_tab2, graf_tab3, graf_tab4 = st.tabs(["Barras Agrupadas", "Barras Apiladas", "Porcentual", "Mapa de Calor"])
            with graf_tab1:
                fig_grouped = px.bar(df_long, x='Fila', y='Frecuencia', color='Columna', barmode='group', text='Frecuencia', title='Barras Agrupadas', color_discrete_sequence=px.colors.qualitative.Set2)
                fig_grouped.update_traces(textposition='outside'); fig_grouped.update_layout(height=400); st.plotly_chart(fig_grouped, use_container_width=True)
            with graf_tab2:
                fig_stacked = px.bar(df_long, x='Fila', y='Frecuencia', color='Columna', barmode='stack', text='Frecuencia', title='Barras Apiladas', color_discrete_sequence=px.colors.qualitative.Set2)
                fig_stacked.update_traces(textposition='inside'); fig_stacked.update_layout(height=400); st.plotly_chart(fig_stacked, use_container_width=True)
            with graf_tab3:
                df_pct = df_long.copy()
                for fila in res['filas']:
                    mask = df_pct['Fila'] == fila
                    total = df_pct[mask]['Frecuencia'].sum()
                    if total > 0: df_pct.loc[mask, 'Frecuencia'] = df_pct.loc[mask, 'Frecuencia'] / total * 100
                fig_pct = px.bar(df_pct, x='Fila', y='Frecuencia', color='Columna', barmode='stack', text=[f"{round(v, 1)}%" for v in df_pct['Frecuencia']], title='Barras Apiladas (Porcentajes)', color_discrete_sequence=px.colors.qualitative.Set2)
                fig_pct.update_traces(textposition='inside'); fig_pct.update_layout(height=400, yaxis_title="Porcentaje (%)"); st.plotly_chart(fig_pct, use_container_width=True)
            with graf_tab4:
                fig_heatmap = px.imshow(res['matriz'], x=res['columnas'], y=res['filas'], text_auto=True, aspect="auto", title="Mapa de Calor", color_continuous_scale='Blues')
                fig_heatmap.update_layout(height=400); st.plotly_chart(fig_heatmap, use_container_width=True)
            
            if st.button("🗑️ Limpiar resultados", key="limpiar_contingencia"):
                del st.session_state['resultados_contingencia']
                st.rerun()

# ============================================================
# PESTANA 7: GUIA DE INTERPRETACION
# ============================================================

with tab7:
    st.markdown("""
    # Guia de Interpretacion de Medidas Estadisticas

    ## Medidas de Tendencia Central
    ### Media (Promedio)
    - Es el promedio aritmetico de todos los datos
    - Representa el centro de masa de la distribucion
    - **Sensible a outliers**: Valores extremos pueden distorsionarla
    ### Mediana
    - Valor que divide los datos en dos partes iguales (50% cada una)
    - **Robusta a outliers**: No se ve afectada por valores extremos
    - Ideal para distribuciones asimetricas
    ### Moda
    - Valor que aparece con mayor frecuencia
    - Puede haber mas de una moda (bimodal, multimodal)
    - Util para datos categoricos

    ## Medidas de Dispersion
    ### Varianza
    - Mide la dispersion promedio al cuadrado
    - **Sensible a outliers**
    ### Desviacion Estandar
    - Raiz cuadrada de la varianza
    - Misma unidad que los datos originales
    - Mas interpretable que la varianza
    ### Rango
    - Diferencia entre el maximo y el minimo
    - **Muy sensible a outliers**
    ### Rango Intercuartil (IQR)
    - Q3 - Q1 (diferencia entre el percentil 75 y 25)
    - **Robusto a outliers**
    - Contiene el 50% central de los datos

    ## Medidas Avanzadas
    ### Media Geometrica
    - Util para datos de crecimiento o tasas
    - Siempre menor o igual que la media aritmetica
    ### Media Armonica
    - Util para promedios de tasas y velocidades
    - Siempre menor o igual que la media geometrica
    ### Media Truncada (5%)
    - Elimina el 5% inferior y superior de los datos
    - **Robusta a outliers**
    ### Media Winsorizada (5%)
    - Reemplaza el 5% inferior y superior por los valores mas cercanos
    - **Robusta a outliers**

    ### Coeficiente de Variacion (CV)
    - (Desviacion Estandar / Media) * 100
    - Mide la variabilidad relativa
    - CV < 15%: Baja variabilidad
    - CV 15-30%: Variabilidad moderada
    - CV > 30%: Alta variabilidad

    ### Coeficiente de Gini
    - Mide la desigualdad en la distribucion
    - 0 = Igualdad perfecta
    - 1 = Desigualdad maxima

    ### MAD (Median Absolute Deviation)
    - Mediana de las desviaciones absolutas respecto a la mediana
    - **Robusta a outliers**
    - Alternativa robusta a la desviacion estandar

    ### Sesgo (Skewness)
    - Mide la asimetria de la distribucion
    - Skewness = 0: Simetrica
    - Skewness > 0: Sesgo positivo (cola a la derecha)
    - Skewness < 0: Sesgo negativo (cola a la izquierda)

    ### Curtosis (Kurtosis)
    - Mide la concentracion en los extremos
    - Kurtosis = 0: Mesocurtica (normal)
    - Kurtosis > 0: Leptocurtica (colas pesadas)
    - Kurtosis < 0: Platicurtica (colas ligeras)

    ## Histograma Interactivo
    - **Number of Classes**: Controla el numero de intervalos
    - **Lower/Upper Limit**: Filtra los datos en un rango especifico
    - **Counts**: Muestra frecuencias absolutas
    - **Relative**: Muestra frecuencias relativas (%)
    - **Cumulative**: Muestra frecuencias acumuladas
    - **Histogram**: Muestra barras
    - **Polygon**: Muestra el poligono de frecuencias
    - **Both**: Muestra barras y poligono

    ## Diagrama de Caja (Boxplot)
    - **Caja**: Contiene el 50% central de los datos (Q1 a Q3)
    - **Linea en la caja**: Mediana (Q2)
    - **Bigotes**: Se extienden hasta 1.5 x IQR
    - **Puntos fuera**: Outliers (valores atipicos)

    ## Interpretacion de Outliers
    - Un outlier es un valor que se aleja significativamente del resto
    - Se detecta cuando esta fuera de: [Q1 - 1.5xIQR, Q3 + 1.5xIQR]

    ## Percentiles
    - El percentil p es el valor por debajo del cual se encuentra el p% de los datos
    - Percentil 50 = Mediana
    - Percentil 25 = Q1, Percentil 75 = Q3
    - Util para describir la posicion relativa de un valor

    ## Sextiles
    - Dividen los datos en 6 partes iguales
    - 1/6 sextil = Percentil 16.67
    - 5/6 sextil = Percentil 83.33
    - Intersextile range (ISR) = 5/6 sextil - 1/6 sextil

    ## Medidas Descriptivas para Datos Agrupados
    ### Media para datos agrupados
    - Se calcula usando las marcas de clase y las frecuencias
    - Fórmula: Media = Σ(marca_clase × frecuencia) / n
    ### Mediana para datos agrupados
    - Se utiliza interpolación lineal dentro de la clase mediana
    - Fórmula: Mediana = L + ((n/2 - F_anterior) / f) × A
    ### Moda para datos agrupados
    - Se estima usando la clase modal
    - Fórmula: Moda = L + (d1 / (d1 + d2)) × A
    ### Percentiles para datos agrupados
    - Similar a la mediana, usando interpolación
    - Fórmula: P = L + ((p×n/100 - F_anterior) / f) × A
    """)
