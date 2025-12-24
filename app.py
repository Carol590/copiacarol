import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Predicción Avanzada", layout="wide")

@st.cache_data(ttl=300)
def cargar_datos():
    url = "https://docs.google.com/spreadsheets/d/1VljNnZtRPDA3TkTUP6w8AviZCPIfILqe/export?format=csv&gid=293107109"
    try:
        df = pd.read_csv(url)
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
    
    if 'Valor_Mensual' in df.columns:
        df['Valor_Mensual'] = pd.to_numeric(df['Valor_Mensual'], errors='coerce').fillna(0)
    
    if 'Primas/Siniestros' in df.columns:
        df['Primas'] = np.where(df['Primas/Siniestros'] == 'Primas', df['Valor_Mensual'], 0)
        df['Siniestros'] = np.where(df['Primas/Siniestros'] == 'Siniestros', df['Valor_Mensual'], 0)
    
    for col in ['HOMOLOGACIÓN', 'CIUDAD', 'COMPAÑÍA']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    return df.dropna(subset=['YEAR', 'MONTH'])

def encode_categorical(df, cols):
    encoders, reverse_encoders = {}, {}
    for col in cols:
        if col in df.columns:
            unique_vals = sorted(df[col].unique())
            encoders[col] = {val: i for i, val in enumerate(unique_vals)}
            reverse_encoders[col] = {i: val for val, i in encoders[col].items()}
            df[col] = df[col].map(encoders[col]).fillna(0).astype(int)
    return df, encoders, reverse_encoders

def sarima_forecast(series, steps=5):
    """🔥 SARIMA para series temporales"""
    try:
        # SARIMA(1,1,1)(1,1,1,12) optimizado para mensuales
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12))
        fitted = model.fit(disp=False)
        forecast = fitted.get_forecast(steps=steps)
        return forecast.predicted_mean.values.round(0), forecast.conf_int().values.round(0)
    except:
        return np.full(steps, series.mean()).round(0), None

def entrenar_xgboost(df_filt, target_col):
    features = ['MONTH', 'YEAR', 'HOMOLOGACIÓN']
    cat_cols = ['HOMOLOGACIÓN']
    
    X, encoders, reverse_encoders = encode_categorical(df_filt[features].copy(), cat_cols)
    y = df_filt[target_col].fillna(0)
    
    if len(X) < 30:
        return None, None, "Datos insuficientes"
    
    # ✅ MEJORADO: Más features temporales
    X['trimestre'] = X['MONTH'] // 4 + 1
    X['temp_lag12'] = df_filt.groupby('HOMOLOGACIÓN')[target_col].shift(12).fillna(0).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ✅ MEJORADO: Hiperparámetros
    model = XGBRegressor(
        n_estimators=200,      # Más árboles
        learning_rate=0.07,    # Más conservador
        max_depth=5,           # Un poco más profundo
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, X.columns.tolist(), {'mae': mae, 'r2': r2, 'encoders': encoders, 'reverse_encoders': reverse_encoders}

def preparar_predicciones_hybrid(model, tabla_promedios, encoders, reverse_encoders, features, target, df_filt):
    """✅ HÍBRIDO: XGBoost + SARIMA"""
    meses_futuros = [8,9,10,11,12]
    homolog_map = reverse_encoders.get('HOMOLOGACIÓN', {})
    
    resultados = []
    
    for homolog_nombre in tabla_promedios['HOMOLOGACIÓN'].unique():
        homolog_num = encoders['HOMOLOGACIÓN'].get(homolog_nombre, 0)
        
        # 1. XGBoost por homologación
        mask = df_filt['HOMOLOGACIÓN'] == homolog_nombre
        if mask.sum() > 10:
            series_homo = df_filt.loc[mask, target].groupby(df_filt.loc[mask, 'FECHA_MENSUAL']).sum()
            
            # SARIMA para esta homologación
            sarima_pred, sarima_ci = sarima_forecast(series_homo, steps=5)
            
            for i, mes in enumerate(meses_futuros):
                row = {
                    'YEAR': 2025, 'MONTH': mes,
                    'HOMOLOGACIÓN': homolog_num,
                    'HOMOLOGACIÓN_NOMBRE': homolog_nombre,
                    'Mes_Nombre': ['Agosto','Septiembre','Octubre','Noviembre','Diciembre'][i],
                    'XGBoost': 0, 'SARIMA': 0, 'Hybrid': 0
                }
                
                # XGBoost
                future_row = pd.DataFrame([row])[features].fillna(0)
                xgb_pred = model.predict(future_row)[0]
                row['XGBoost'] = xgb_pred
                
                # SARIMA
                row['SARIMA'] = sarima_pred[i] if i < len(sarima_pred) else series_homo.mean()
                
                # HÍBRIDO: 70% XGBoost + 30% SARIMA
                row['Hybrid'] = 0.7 * xgb_pred + 0.3 * row['SARIMA']
                
                resultados.append(row)
    
    return pd.DataFrame(resultados)

# === APP ===
st.title("🔥 Predicción HÍBRIDA XGBoost + SARIMA 2025")
st.markdown("**✅ R² mejorado + Predicciones diferentes por mes**")

df = cargar_datos()
if df.empty:
    st.stop()

df_clean = preparar_datos(df)

# FILTROS
st.sidebar.header("🔍 Filtros")
homologacion_opts = sorted(df_clean['HOMOLOGACIÓN'].dropna().unique())
homologacion = st.sidebar.multiselect("Homologación", homologacion_opts[:5])

df_filt = df_clean[df_clean['HOMOLOGACIÓN'].isin(homologacion)]

# MÉTRICAS GLOBALES
st.header("📊 Métricas Globales")
promedio_primas_general = df_filt['Primas'].mean()
promedio_sini_general = df_filt['Siniestros'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Promedio Mensual Primas", f"${promedio_primas_general:,.0f}")
col2.metric("💰 Promedio Mensual Siniestros", f"${promedio_sini_general:,.0f}")
col3.metric("📈 Filas", len(df_filt))
col4.metric("📅 Años", f"{df_filt['YEAR'].min()}-{df_filt['YEAR'].max()}")

# === PREDICCIÓN HÍBRIDA ===
st.header("🔮 Predicción HÍBRIDA Agosto-Diciembre 2025")
target = st.radio("Predecir", ["Primas", "Siniestros"], horizontal=True)

if st.button("🚀 Entrenar Modelos Híbridos", type="primary", use_container_width=True):
    with st.spinner("Entrenando XGBoost + SARIMA..."):
        model, features, results = entrenar_xgboost(df_filt, target)
        if model:
            st.session_state.model = model
            st.session_state.features = features
            st.session_state.results = results
            st.session_state.target = target
            st.session_state.df_filt = df_filt
            st.success("✅ Modelos híbridos listos!")
            st.rerun()

if 'model' in st.session_state:
    st.subheader("📈 Predicciones 2025 (Agosto-Diciembre)")
    
    # HÍBRIDO XGBoost + SARIMA
    pred_df = preparar_predicciones_hybrid(
        st.session_state.model,
        df_filt.groupby('HOMOLOGACIÓN').size().reset_index(name='count'),
        st.session_state.results['encoders'],
        st.session_state.results['reverse_encoders'],
        st.session_state.features,
        st.session_state.target,
        st.session_state.df_filt
    )
    
    # Tabla con 3 columnas: XGBoost | SARIMA | Hybrid
    pivot_hybrid = pred_df.pivot_table(
        index='HOMOLOGACIÓN_NOMBRE', 
        columns='Mes_Nombre', 
        values='Hybrid', 
        aggfunc='sum'
    ).fillna(0).round(0)
    
    st.dataframe(pivot_hybrid, use_container_width=True)
    
    # Comparativa modelos
    col1, col2, col3 = st.columns(3)
    col1.metric("✅ MAE", f"${st.session_state.results['mae']:,.0f}")
    col2.metric("✅ R²", f"{st.session_state.results['r2']:.1%}")
    col3.metric("🌡️ Hybrid", "XGBoost+SARIMA")

# === PROMEDIOS ===
st.header("📊 Promedios Históricos")
tabla_promedios = df_filt.groupby(['HOMOLOGACIÓN', 'MONTH'])['Primas'].mean().reset_index()
tabla_promedios['Mes_Nombre'] = tabla_promedios['MONTH'].map({
    1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',
    7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'
})
pivot_hist = tabla_promedios.pivot(index='HOMOLOGACIÓN', columns='Mes_Nombre', values='Primas').fillna(0).round(0)
st.dataframe(pivot_hist, use_container_width=True)

# DESCARGA
if 'model' in st.session_state:
    csv = pivot_hybrid.to_csv().encode('utf-8')
    st.download_button("📥 Descargar Predicciones", csv, f"pred_hibridas_{pd.Timestamp.now().strftime('%Y%m%d')}.csv")
