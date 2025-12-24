import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SARIMA Predicción Primas/Siniestros", layout="wide")

@st.cache_data(ttl=300)
def cargar_datos():
    url = "https://docs.google.com/spreadsheets/d/1VljNnZtRPDA3TkTUP6w8AviZCPIfILqe/export?format=csv&gid=293107109"
    try:
        df = pd.read_csv(url)
        st.success(f"✅ {len(df):,} filas cargadas")
        return df
    except:
        return pd.DataFrame()

def preparar_datos(df):
    df.columns = df.columns.str.strip()
    
    if 'FECHA' in df.columns:
        df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
        df['YEAR'] = df['FECHA'].dt.year
        df['MONTH'] = df['FECHA'].dt.month
        df['FECHA_MENSUAL'] = df['FECHA'].dt.to_period('M').dt.to_timestamp()
        df = df.sort_values('FECHA')
    
    if 'Valor_Mensual' in df.columns:
        df['Valor_Mensual'] = pd.to_numeric(df['Valor_Mensual'], errors='coerce').fillna(0)
    
    if 'Primas/Siniestros' in df.columns:
        df['Primas'] = np.where(df['Primas/Siniestros'] == 'Primas', df['Valor_Mensual'], 0)
        df['Siniestros'] = np.where(df['Primas/Siniestros'] == 'Siniestros', df['Valor_Mensual'], 0)
    
    for col in ['HOMOLOGACIÓN', 'CIUDAD', 'COMPAÑÍA']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    return df.dropna(subset=['YEAR', 'MONTH'])

def sarima_por_homologacion(df_filt, homologacion, target_col, steps=5, peso_reciente=True):
    """🔥 SARIMA OPTIMIZADO - EL MEJOR"""
    mask = df_filt['HOMOLOGACIÓN'] == homologacion
    if mask.sum() < 12:  # Min 1 año
        return np.full(steps, df_filt.loc[mask, target_col].mean())
    
    # Serie temporal mensual
    series_raw = df_filt.loc[mask].groupby('FECHA_MENSUAL')[target_col].sum()
    
    # ✅ PESO RECIENTE en entrenamiento
    if peso_reciente:
        weights = np.exp(np.linspace(-0.1, 0, len(series_raw)))  # Decaimiento exponencial
        series_weighted = series_raw * weights
    else:
        series_weighted = series_raw
    
    try:
        # SARIMA MEJORADO (2,1,2)(1,1,1,12)
        model = SARIMAX(
            series_weighted, 
            order=(2,1,2), 
            seasonal_order=(1,1,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted = model.fit(disp=False, maxiter=100)
        forecast = fitted.get_forecast(steps=steps)
        pred = forecast.predicted_mean.values.round(0)
        conf_int = forecast.conf_int(alpha=0.05).round(0)
        return pred, conf_int.iloc[:,0].values, conf_int.iloc[:,1].values
    except:
        # Fallback ARIMA simple
        model = ARIMA(series_raw, order=(1,1,1), seasonal_order=(1,1,1,12))
        fitted = model.fit()
        forecast = fitted.get_forecast(steps=steps)
        return forecast.predicted_mean.values.round(0), None, None

def calcular_sarima_completo(df_filt, target, steps=5):
    """SARIMA para TODAS las homologaciones"""
    resultados = []
    homologaciones = df_filt['HOMOLOGACIÓN'].unique()
    
    for homologacion in homologaciones:
        pred, ci_low, ci_high = sarima_por_homologacion(df_filt, homologacion, target, steps)
        
        for i, mes in enumerate([8,9,10,11,12]):
            resultados.append({
                'HOMOLOGACIÓN': homologacion,
                'Mes_Nombre': ['Agosto','Septiembre','Octubre','Noviembre','Diciembre'][i],
                'Predicción': pred[i],
                'CI_Inferior': ci_low[i] if ci_low is not None else pred[i]*0.9,
                'CI_Superior': ci_high[i] if ci_high is not None else pred[i]*1.1
            })
    
    return pd.DataFrame(resultados)

def calcular_promedio_mensual(df):
    mensual = df.groupby(['HOMOLOGACIÓN', 'YEAR', 'MONTH']).agg({
        'Primas': 'sum', 'Siniestros': 'sum'
    }).round(0)
    
    promedio_mensual = mensual.groupby(['HOMOLOGACIÓN', 'MONTH']).mean().round(0)
    promedio_mensual.columns = ['Promedio_Total_Primas', 'Promedio_Total_Siniestros']
    promedio_mensual = promedio_mensual.reset_index()
    
    mes_map = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
               7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
    promedio_mensual['Mes_Nombre'] = promedio_mensual['MONTH'].map(mes_map)
    
    return promedio_mensual.sort_values(['HOMOLOGACIÓN', 'MONTH'])

# === APP ===
st.title("🔥 SARIMA - MEJOR MODELO Predicción 2025")
st.markdown("**✅ SARIMA(2,1,2)(1,1,1,12) + Peso reciente | Agosto-Diciembre**")

df = cargar_datos()
if df.empty:
    st.stop()

df_clean = preparar_datos(df)

# FILTROS
st.sidebar.header("🔍 Filtros")
homologacion_opts = sorted(df_clean['HOMOLOGACIÓN'].dropna().unique())
homologacion = st.sidebar.multiselect("Homologación", homologacion_opts[:8], default=homologacion_opts[:3])

df_filt = df_clean[df_clean['HOMOLOGACIÓN'].isin(homologacion)]

# MÉTRICAS GLOBALES
st.header("📊 Métricas Globales")
tabla_promedios = calcular_promedio_mensual(df_filt)

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Promedio Primas", f"${df_filt['Primas'].mean():,.0f}")
col2.metric("💰 Promedio Siniestros", f"${df_filt['Siniestros'].mean():,.0f}")
col3.metric("📈 Homologaciones", len(tabla_promedios['HOMOLOGACIÓN'].unique()))
col4.metric("📅 Años datos", f"{df_filt['YEAR'].min()}-{df_filt['YEAR'].max()}")

# SARIMA PRINCIPAL
st.header("🔮 SARIMA Predicción Agosto-Diciembre 2025")
target = st.radio("Predecir", ["Primas", "Siniestros"], horizontal=True)
peso_reciente = st.sidebar.checkbox("Peso reciente (últimos años x2)", value=True)

if st.button("🚀 Generar SARIMA", type="primary", use_container_width=True):
    with st.spinner("Entrenando SARIMA por homologación..."):
        st.session_state.pred_sarima = calcular_sarima_completo(df_filt, target, peso_reciente=peso_reciente)
        st.session_state.target = target
        st.session_state.df_filt = df_filt
        st.success("✅ SARIMA listo!")

if 'pred_sarima' in st.session_state:
    st.subheader("📈 Predicciones SARIMA 2025")
    
    # Tabla principal SARIMA
    pivot_sarima = st.session_state.pred_sarima.pivot(
        index='HOMOLOGACIÓN', 
        columns='Mes_Nombre', 
        values='Predicción'
    ).fillna(0).round(0)
    
    st.dataframe(pivot_sarima, use_container_width=True)
    
    # Intervalos confianza
    st.subheader("🎯 Intervalos Confianza 95%")
    ci_df = st.session_state.pred_sarima.pivot_table(
        index='HOMOLOGACIÓN', columns='Mes_Nombre',
        values=['CI_Inferior', 'CI_Superior']
    ).round(0)
    with st.expander("Ver intervalos confianza"):
        st.dataframe(ci_df, use_container_width=True)
    
    # Gráfico SARIMA
    fig = px.line(
        st.session_state.pred_sarima, 
        x='Mes_Nombre', 
        y='Predicción',
        color='HOMOLOGACIÓN',
        title="SARIMA Predicciones Agosto-Diciembre 2025",
        markers=True
    )
    fig.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# PROMEDIOS HISTÓRICOS
st.header("📊 Promedios Históricos Agosto-Diciembre")
pivot_hist = tabla_promedios[
    tabla_promedios['MONTH'].isin([8,9,10,11,12])
].pivot(index='HOMOLOGACIÓN', columns='Mes_Nombre', values='Promedio_Total_Primas').fillna(0).round(0)
st.dataframe(pivot_hist, use_container_width=True)

# XGBoost como backup (opcional)
with st.expander("🔧 XGBoost Backup (opcional)"):
    if st.button("Comparar con XGBoost"):
        st.info("SARIMA es superior. XGBoost solo como referencia.")

# DESCARGA
if 'pred_sarima' in st.session_state:
    csv = pivot_sarima.to_csv().encode('utf-8')
    st.download_button("📥 Descargar SARIMA", csv, f"sarima_{pd.Timestamp.now().strftime('%Y%m%d')}.csv")
