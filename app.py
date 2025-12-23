import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuración
st.set_page_config(
    page_title="🔮 Predicción Primas/Siniestros", 
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Predicción XGBoost Primas y Siniestros")
st.markdown("**Datos 2020-2025 → Predicción 2026** | Seguros Colombia")

@st.cache_data(show_spinner=False)
def load_google_sheet():
    """Carga y limpia Google Sheet"""
    sheet_id = "1VljNnZtRPDA3TkTUP6w8AviZCPIfILqe"
    gid = "293107109"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    
    try:
        df = pd.read_csv(url)
        st.success(f"✅ Datos cargados: {len(df):,} filas")
    except:
        st.error("❌ Error cargando Google Sheet")
        st.stop()
    
    # Limpieza robusta
    df.columns = df.columns.str.strip()
    
    # FECHA
    df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
    df['YEAR'] = df['FECHA'].dt.year
    df['MONTH'] = df['FECHA'].dt.month
    
    # Valor_Mensual numérico
    df['Valor_Mensual'] = pd.to_numeric(df['Valor_Mensual'], errors='coerce')
    
    # Targets separados
    df['Primas'] = np.where(df['Primas/Siniestros'] == 'Primas', df['Valor_Mensual'], 0)
    df['Siniestros'] = np.where(df['Primas/Siniestros'] == 'Siniestros', df['Valor_Mensual'], 0)
    
    # Encoding categóricas
    cat_cols = ['HOMOLOGACIÓN', 'COMPAÑÍA', 'CIUDAD', 'RAMOS', 'DEPARTAMENTO']
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col].astype(str))
    
    return df.dropna(subset=['YEAR', 'MONTH'])

# Cargar datos
df = load_google_sheet()
st.sidebar.markdown("---")
st.sidebar.metric("Filas", len(df))
st.sidebar.metric("Años", f"{df['YEAR'].min()}-{df['YEAR'].max()}")
st.sidebar.markdown("---")

# Sidebar filtros
st.sidebar.header("⚙️ Configuración")
target = st.sidebar.selectbox("🎯 Predecir", ["Primas", "Siniestros"])
max_year = st.sidebar.slider("📅 Año máx. entrenamiento", 2020, 2025, 2025)

df_filtered = df[df['YEAR'] <= max_year].copy()

# Preview datos
col1, col2 = st.columns([2,1])
with col1:
    st.markdown("### 📋 Datos procesados")
    st.dataframe(
        df_filtered[['YEAR', 'MONTH', 'COMPAÑÍA', 'CIUDAD', 'RAMOS', target]].head(10),
        use_container_width=True
    )
with col2:
    st.markdown("### 📈 Distribución Target")
    fig_dist = px.histogram(df_filtered, x=target, nbins=50, title=f"Distribución {target}")
    st.plotly_chart(fig_dist, use_container_width=True)

def prepare_features(df, target_col):
    """Features para XGBoost"""
    feature_cols = ['YEAR', 'MONTH', 'HOMOLOGACIÓN', 'COMPAÑÍA', 'CIUDAD', 'RAMOS', 'DEPARTAMENTO']
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features].fillna(0)
    y = df[target_col].fillna(0)
    
    return X, y, available_features

def train_model(X, y):
    """Entrena XGBoost optimizado"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.07,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mae, r2, X_test, y_test, y_pred

# Entrenar
if st.button("🚀 Entrenar XGBoost", type="primary"):
    with st.spinner("Entrenando modelo..."):
        X, y, features = prepare_features(df_filtered, target)
        model, mae, r2, X_test, y_test, y_pred = train_model(X, y)
        
        st.session_state.model = model
        st.session_state.mae = mae
        st.session_state.r2 = r2
        st.session_state.features = features
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.target = target
        st.success("✅ Modelo entrenado!")

# Resultados
if 'model' in st.session_state:
    st.markdown("---")
    st.header("📊 Resultados del Modelo")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAE", f"${st.session_state.mae:,.0f}")
    with col2:
        st.metric("R²", f"{st.session_state.r2:.2%}")
    with col3:
        st.metric("Features", len(st.session_state.features))
    
    # Scatter plot
    fig_scatter = px.scatter(
        x=st.session_state.y_test, 
        y=st.session_state.y_pred,
        labels={'x': 'Real', 'y': 'Predicho'},
        title=f"🔍 Predicción vs Real ({st.session_state.target})"
    )
    fig_scatter.add_shape(
        type="line", x0=0, y0=0, 
        x1=max(st.session_state.y_test.max(), st.session_state.y_pred.max()),
        y1=max(st.session_state.y_test.max(), st.session_state.y_pred.max()),
        line=dict(color="red", dash="dash")
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': st.session_state.features,
        'Importancia': st.session_state.model.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    fig_importance = px.bar(importance.head(10), x='Importancia', y='Feature',
                           orientation='h', title="📊 Importancia Features")
    st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown("---")
    st.header("🔮 Predicciones 2026")
    
    # Datos futuros 2026
    future_data = pd.DataFrame({
        'YEAR': [2026] * 12,
        'MONTH': list(range(1, 13))
    })
    
    # Features más frecuentes
    top_values = df_filtered.mode().iloc[0]
    for feature in st.session_state.features:
        if feature not in ['YEAR', 'MONTH']:
            future_data[feature] = top_values.get(feature, 0)
    
    future_data = future_data[st.session_state.features]
    predictions_2026 = st.session_state.model.predict(future_data)
    
    future_df = pd.DataFrame({
        'Mes': [f"2026-{m:02d}" for m in range(1, 13)],
        f'Predicción {st.session_state.target}': predictions_2026.round(0)
    })
    
    st.dataframe(future_df, use_container_width=True)
    
    # Gráfico 2026
    fig_2026 = px.bar(future_df, x='Mes', y=f'Predicción {st.session_state.target}',
                     title="📈 Predicción Mensual 2026", color=f'Predicción {st.session_state.target}')
    st.plotly_chart(fig_2026, use_container_width=True)
    
    # Total 2026
    total_2026 = predictions_2026.sum()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total 2026", f"${total_2026:,.0f}")
    with col2:
        st.metric("Promedio mensual", f"${total_2026/12:,.0f}")
    
    # Descarga Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        future_df.to_excel(writer, sheet_name='Predicciones_2026', index=False)
        importance.to_excel(writer, sheet_name='Importancia_Features', index=False)
        pd.DataFrame({
            'MAE': [st.session_state.mae],
            'R2': [st.session_state.r2],
            'Total_2026': [total_2026]
        }).to_excel(writer, sheet_name='Métricas', index=False)
    
    st.download_button(
        label="📥 Descargar Excel Completo",
        data=output.getvalue(),
        file_name=f"predicciones_{st.session_state.target.lower()}_2026.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.markdown("---")
st.markdown("*Desarrollado para seguros Colombia | XGBoost optimizado para series temporales*")
