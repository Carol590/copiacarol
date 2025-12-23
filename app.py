import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import xgboost as xgb
import os
import pickle
from sklearn.preprocessing import LabelEncoder
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests
from io import StringIO

# Configuración de la página
st.set_page_config(
    page_title="Predicción Primas y Siniestros",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Crear directorios necesarios
os.makedirs('models', exist_ok=True)

# ==================== CONFIGURACIÓN DE GOOGLE SHEETS ====================

# Opción 1: Conexión directa con URL público (más simple)
def load_data_from_gsheets_public():
    """Carga datos desde Google Sheets usando el enlace de exportación CSV"""
    try:
        # ID de tu Google Sheets (extraído de la URL)
        SHEET_ID = "1VljNnZtRPDA3TkTUP6w8AviZCPIfIlqe"
        
        # URL de exportación CSV
        CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
        
        # Cargar datos
        df = pd.read_csv(CSV_URL)
        
        if df.empty:
            st.warning("⚠️ El Google Sheets está vacío o no es accesible")
            return pd.DataFrame()
            
        return df
        
    except Exception as e:
        st.error(f"❌ Error al cargar datos desde Google Sheets: {str(e)}")
        st.info("💡 Verifica que el documento sea público o que el ID sea correcto")
        return pd.DataFrame()

# Opción 2: Conexión con Service Account (más segura, recomendada)
def load_data_from_gsheets_service_account():
    """Carga datos usando credenciales de Service Account"""
    try:
        # Configurar credenciales
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        # Verificar si existe el archivo de credenciales
        creds_file = "credentials.json"
        if not os.path.exists(creds_file):
            st.error(f"❌ No se encontró '{creds_file}'")
            st.info("📄 Para usar esta opción, necesitas:")
            st.markdown("""
            1. Crear un proyecto en [Google Cloud Console](https://console.cloud.google.com/)
            2. Habilitar Google Sheets API y Google Drive API
            3. Crear una Service Account y descargar el JSON como `credentials.json`
            4. Compartir tu Google Sheets con el email de la Service Account
            """)
            return pd.DataFrame()
        
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)
        client = gspread.authorize(creds)
        
        # Abrir el spreadsheet
        spreadsheet = client.open_by_key("1VljNnZtRPDA3TkTUP6w8AviZCPIfIlqe")
        worksheet = spreadsheet.get_worksheet(0)  # Primera hoja
        
        # Convertir a DataFrame
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error con Service Account: {str(e)}")
        return pd.DataFrame()

# Función principal de carga de datos
@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_data():
    """Carga datos desde Google Sheets (Intenta ambos métodos)"""
    
    # Intentar método público primero
    df = load_data_from_gsheets_public()
    
    if df.empty:
        st.warning("⚠️ Intentando con Service Account...")
        df = load_data_from_gsheets_service_account()
    
    return df

# ==================== PREPROCESAMIENTO ====================

@st.cache_data
def preprocess_data(df):
    if df.empty:
        return df
    
    # Limpiar nombres de columnas (eliminar espacios extras)
    df.columns = df.columns.str.strip()
    
    # Convertir FECHA a datetime
    try:
        # Intentar múltiples formatos
        df['FECHA'] = pd.to_datetime(df['FECHA'], format='%d/%m/%Y %I:%M:%S %p', errors='coerce')
        
        # Si falla, intentar formato simple
        if df['FECHA'].isna().all():
            df['FECHA'] = pd.to_datetime(df['FECHA'], format='%d/%m/%Y', errors='coerce')
    except:
        st.error("❌ Error al convertir la columna FECHA. Verifica el formato.")
    
    # Extraer mes y año
    df['Mes'] = df['FECHA'].dt.month
    df['Año'] = df['FECHA'].dt.year
    
    # Limpiar Valor_Mensual
    df['Valor_Mensual'] = pd.to_numeric(df['Valor_Mensual'], errors='coerce').fillna(0)
    
    # Crear columna de fecha para Prophet (ds)
    df['ds'] = df['FECHA'].dt.to_period('M').dt.to_timestamp()
    
    # Crear columna de valor para Prophet (y)
    df['y'] = df['Valor_Mensual']
    
    return df

# ==================== FUNCIONES DE MODELOS ====================
def train_prophet_model(df_subset, model_name):
    model_path = f"models/{model_name}_prophet.pkl"
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    prophet_df = df_subset[['ds', 'y']].rename(columns={'Valor_Mensual': 'y'})
    
    model = Prophet(
        yearly_seasonality=True,
        monthly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    model.fit(prophet_df)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model

def predict_prophet(model, periods=12):
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

def train_xgboost_model(df_subset):
    df_train = df_subset.copy()
    
    le_compania = LabelEncoder()
    le_ciudad = LabelEncoder()
    le_ramo = LabelEncoder()
    le_tipo = LabelEncoder()
    
    df_train['compania_enc'] = le_compania.fit_transform(df_train['COMPAÑÍA'])
    df_train['ciudad_enc'] = le_ciudad.fit_transform(df_train['CIUDAD'])
    df_train['ramo_enc'] = le_ramo.fit_transform(df_train['RAMOS'])
    df_train['tipo_enc'] = le_tipo.fit_transform(df_train['Primas/Siniestros'])
    
    features = ['Año', 'Mes', 'compania_enc', 'ciudad_enc', 'ramo_enc', 'tipo_enc']
    X = df_train[features]
    y = df_train['Valor_Mensual']
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X, y)
    
    return model, le_compania, le_ciudad, le_ramo, le_tipo

# ==================== APLICACIÓN PRINCIPAL ====================

# Título
st.title("📊 Predicción de Primas y Siniestros")
st.subheader("Mercado Asegurador Colombiano")

# Cargar datos
df_raw = load_data()
df = preprocess_data(df_raw)

# Sidebar - Filtros
st.sidebar.header("🔧 Filtros")

if not df.empty:
    # Mostrar última actualización
    ultima_act = df['FECHA'].max().strftime('%Y-%m-%d')
    st.sidebar.success(f"✅ Datos actualizados hasta: {ultima_act}")
    
    companias = sorted(df['COMPAÑÍA'].unique().tolist())
    selected_companias = st.sidebar.multiselect(
        "Compañía(s)",
        options=companias,
        default=companias[:3] if len(companias) >= 3 else companias
    )
    
    ciudades = sorted(df['CIUDAD'].unique().tolist())
    selected_ciudades = st.sidebar.multiselect(
        "Ciudad(es)",
        options=ciudades,
        default=ciudades[:3] if len(ciudades) >= 3 else ciudades
    )
    
    ramos = sorted(df['RAMOS'].unique().tolist())
    selected_ramos = st.sidebar.multiselect(
        "Ramo(s)",
        options=ramos,
        default=ramos[:2] if len(ramos) >= 2 else ramos
    )
    
    df_filtered = df[
        (df['COMPAÑÍA'].isin(selected_companias)) &
        (df['CIUDAD'].isin(selected_ciudades)) &
        (df['RAMOS'].isin(selected_ramos))
    ]
else:
    df_filtered = pd.DataFrame()
    st.warning("⚠️ No hay datos para mostrar. Verifica la conexión a Google Sheets.")

# Menú de navegación
page = st.selectbox(
    "Selecciona la página:",
    ["🏠 Inicio", "📋 Predicciones por Homologación", "🏙️ Análisis de Ciudades", "🏢 Vista de Competidores"]
)

# Página de Inicio
if page == "🏠 Inicio":
    st.header("Bienvenido al Sistema de Predicción")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Compañías", len(df['COMPAÑÍA'].unique()) if not df.empty else 0)
    
    with col2:
        st.metric("Total Ciudades", len(df['CIUDAD'].unique()) if not df.empty else 0)
    
    with col3:
        st.metric("Total Ramos", len(df['RAMOS'].unique()) if not df.empty else 0)
    
    with col4:
        total_registros = len(df) if not df.empty else 0
        st.metric("Total Registros", f"{total_registros:,}")
    
    if not df_filtered.empty:
        st.subheader("📈 Vista Previa de Datos Filtrados")
        
        # Mostrar sample con formato
        st.dataframe(
            df_filtered[['HOMOLOGACIÓN', 'Año', 'COMPAÑÍA', 'CIUDAD', 
                        'RAMOS', 'Primas/Siniestros', 'FECHA', 'Valor_Mensual']].head(10),
            use_container_width=True
        )
        
        # Gráfico de evolución temporal
        st.subheader("Evolución Temporal")
        df_temporal = df_filtered.groupby(['FECHA', 'Primas/Siniestros'])['Valor_Mensual'].sum().reset_index()
        
        fig = px.line(
            df_temporal,
            x='FECHA',
            y='Valor_Mensual',
            color='Primas/Siniestros',
            title="Evolución de Primas vs Siniestros"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # KPIs
        st.subheader("📊 KPIs Generales")
        col1, col2, col3 = st.columns(3)
        
        if 'Primas' in df_filtered['Primas/Siniestros'].values:
            total_primas = df_filtered[df_filtered['Primas/Siniestros'] == 'Primas']['Valor_Mensual'].sum()
            col1.metric("Total Primas", f"${total_primas:,.0f}")
        
        if 'Siniestros' in df_filtered['Primas/Siniestros'].values:
            total_siniestros = df_filtered[df_filtered['Primas/Siniestros'] == 'Siniestros']['Valor_Mensual'].sum()
            col2.metric("Total Siniestros", f"${total_siniestros:,.0f}")
            
            if 'total_primas' in locals() and total_primas > 0:
                loss_ratio = (total_siniestros / total_primas) * 100
                col3.metric("Loss Ratio", f"{loss_ratio:.1f}%")

# Página 1: Predicciones por Homologación
elif page == "📋 Predicciones por Homologación":
    st.header("Predicciones por Homologación")
    
    if df_filtered.empty:
        st.warning("No hay datos para analizar con los filtros seleccionados")
    else:
        df_homo = df_filtered.groupby(['HOMOLOGACIÓN', 'FECHA', 'Primas/Siniestros'])['Valor_Mensual'].sum().reset_index()
        
        tab1, tab2 = st.tabs(["💰 Primas", "🚨 Siniestros"])
        
        with tab1:
            st.subheader("Predicciones para Primas")
            
            df_primas = df_homo[df_homo['Primas/Siniestros'] == 'Primas']
            
            if not df_primas.empty:
                homologaciones = df_primas['HOMOLOGACIÓN'].unique()
                resultados = []
                
                for homo in homologaciones:
                    df_h = df_primas[df_primas['HOMOLOGACIÓN'] == homo]
                    
                    if len(df_h) > 10:
                        model = train_prophet_model(df_h, f"homo_{homo}_primas")
                        pred = predict_prophet(model, 12)
                        
                        ultimo_valor = df_h['Valor_Mensual'].iloc[-1]
                        prediccion_6m = pred['yhat'].iloc[5]
                        crecimiento = ((prediccion_6m - ultimo_valor) / ultimo_valor * 100) if ultimo_valor > 0 else 0
                        
                        resultados.append({
                            'HOMOLOGACIÓN': homo,
                            'Último Valor': f"${ultimo_valor:,.0f}",
                            'Pred 6M': f"${prediccion_6m:,.0f}",
                            'Crecimiento %': f"{crecimiento:.1f}%",
                            'Tendencia': '📈' if crecimiento > 0 else '📉',
                            'Confianza': 'Alta' if len(df_h) > 30 else 'Media'
                        })
                
                if resultados:
                    df_result = pd.DataFrame(resultados)
                    st.dataframe(df_result, use_container_width=True)
                    
                    # Gráfico de barras
                    fig = px.bar(
                        df_result,
                        x='HOMOLOGACIÓN',
                        y=[float(x.replace('%', '')) for x in df_result['Crecimiento %']],
                        title="Crecimiento Estimado por Homologación (6 meses)"
                    )
                    st.plotly_chart(fig)
            else:
                st.info("No hay datos de primas para los filtros seleccionados")
        
        with tab2:
            st.subheader("Predicciones para Siniestros")
            
            df_siniestros = df_homo[df_homo['Primas/Siniestros'] == 'Siniestros']
            
            if not df_siniestros.empty:
                homologaciones = df_siniestros['HOMOLOGACIÓN'].unique()
                resultados = []
                
                for homo in homologaciones:
                    df_h = df_siniestros[df_siniestros['HOMOLOGACIÓN'] == homo]
                    
                    if len(df_h) > 10:
                        model = train_prophet_model(df_h, f"homo_{homo}_siniestros")
                        pred = predict_prophet(model, 12)
                        
                        ultimo_valor = df_h['Valor_Mensual'].iloc[-1]
                        prediccion_6m = pred['yhat'].iloc[5]
                        crecimiento = ((prediccion_6m - ultimo_valor) / ultimo_valor * 100) if ultimo_valor > 0 else 0
                        
                        resultados.append({
                            'HOMOLOGACIÓN': homo,
                            'Último Valor': f"${ultimo_valor:,.0f}",
                            'Pred 6M': f"${prediccion_6m:,.0f}",
                            'Crecimiento %': f"{crecimiento:.1f}%",
                            'Tendencia': '📈' if crecimiento > 0 else '📉',
                            'Confianza': 'Alta' if len(df_h) > 30 else 'Media'
                        })
                
                if resultados:
                    df_result = pd.DataFrame(resultados)
                    st.dataframe(df_result, use_container_width=True)
                    
                    # Gráfico
                    fig = px.bar(
                        df_result,
                        x='HOMOLOGACIÓN',
                        y=[float(x.replace('%', '')) for x in df_result['Crecimiento %']],
                        title="Crecimiento Estimado de Siniestros por Homologación"
                    )
                    st.plotly_chart(fig)
            else:
                st.info("No hay datos de siniestros para los filtros seleccionados")

# Página 2: Análisis de Ciudades
elif page == "🏙️ Análisis de Ciudades":
    st.header("Análisis de Ciudades Principales")
    
    ciudades_objetivo = ['BOGOTA', 'MEDELLIN', 'CALI', 'BUCARAMANGA', 
                        'BARRANQUILLA', 'CARTAGENA', 'TUNJA']
    
    ciudades_disponibles = [c for c in ciudades_objetivo if c in df_filtered['CIUDAD'].str.upper().unique()]
    
    if not ciudades_disponibles:
        st.warning("No hay datos para las ciudades principales con los filtros seleccionados")
    else:
        df_ciudades = df_filtered[df_filtered['CIUDAD'].str.upper().isin(ciudades_disponibles)]
        df_ciudad_resumen = df_ciudades.groupby(['CIUDAD', 'Primas/Siniestros', 'FECHA'])['Valor_Mensual'].sum().reset_index()
        
        tab1, tab2 = st.tabs(["📊 Comparativa", "🔮 Predicciones"])
        
        with tab1:
            # Gráfico comparativo
            fig = px.line(
                df_ciudad_resumen,
                x='FECHA',
                y='Valor_Mensual',
                color='CIUDAD',
                line_dash='Primas/Siniestros',
                title="Evolución por Ciudad"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap
            st.subheader("🗺️ Heatmap de Actividad")
            ciudad_periodo = df_ciudad_resumen.groupby(['CIUDAD', 'FECHA'])['Valor_Mensual'].sum().reset_index()
            pivot_data = ciudad_periodo.pivot(index='CIUDAD', columns='FECHA', values='Valor_Mensual')
            
            fig_heatmap = px.imshow(
                pivot_data.values,
                x=[col.strftime('%Y-%m') for col in pivot_data.columns],
                y=pivot_data.index,
                color_continuous_scale='Viridis',
                title="Intensidad de Primas/Siniestros por Ciudad"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Tabla resumen
            st.subheader("Resumen Últimos 12 Meses")
            df_ultimos = df_ciudad_resumen[df_ciudad_resumen['FECHA'] >= (datetime.now() - timedelta(days=365))]
            resumen = df_ultimos.groupby(['CIUDAD', 'Primas/Siniestros'])['Valor_Mensual'].sum().unstack(fill_value=0)
            
            if 'Primas' in resumen.columns and 'Siniestros' in resumen.columns:
                resumen['Loss Ratio'] = (resumen['Siniestros'] / resumen['Primas'] * 100).round(1)
            
            st.dataframe(resumen, use_container_width=True)
        
        with tab2:
            st.subheader("Predicciones por Ciudad")
            
            ciudad_seleccionada = st.selectbox("Selecciona ciudad para predicción detallada:", ciudades_disponibles)
            
            if ciudad_seleccionada:
                df_ciudad = df_ciudad_resumen[df_ciudad_resumen['CIUDAD'].str.upper() == ciudad_seleccionada.upper()]
                
                if not df_ciudad.empty:
                    model = train_prophet_model(df_ciudad, f"ciudad_{ciudad_seleccionada}")
                    pred = predict_prophet(model, 12)
                    
                    # Gráfico
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_ciudad['FECHA'],
                        y=df_ciudad['Valor_Mensual'],
                        mode='lines+markers',
                        name='Histórico',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=pred['ds'],
                        y=pred['yhat'],
                        mode='lines+markers',
                        name='Predicción',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Intervalo de confianza
                    fig.add_trace(go.Scatter(
                        x=pred['ds'],
                        y=pred['yhat_upper'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(255,0,0,0)',
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=pred['ds'],
                        y=pred['yhat_lower'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(255,0,0,0)',
                        name='Intervalo Confianza',
                        fillcolor='rgba(255,0,0,0.2)'
                    ))
                    
                    fig.update_layout(
                        title=f"Predicción para {ciudad_seleccionada}",
                        xaxis_title="Fecha",
                        yaxis_title="Valor Mensual"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabla de predicciones
                    pred_tabla = pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                    pred_tabla['Mes'] = pred_tabla['ds'].dt.strftime('%Y-%m')
                    pred_tabla = pred_tabla[['Mes', 'yhat', 'yhat_lower', 'yhat_upper']]
                    pred_tabla.columns = ['Mes', 'Predicción', 'Límite Inferior', 'Límite Superior']
                    pred_tabla = pred_tabla.round(0)
                    
                    st.subheader("📅 Tabla de Predicciones")
                    st.dataframe(pred_tabla, use_container_width=True)
                    
                    # KPIs de predicción
                    col1, col2, col3 = st.columns(3)
                    pred_total = pred_tabla['Predicción'].sum()
                    crecimiento_pred = ((pred_tabla['Predicción'].iloc[-1] - pred_tabla['Predicción'].iloc[0]) / pred_tabla['Predicción'].iloc[0] * 100) if pred_tabla['Predicción'].iloc[0] > 0 else 0
                    
                    col1.metric("Predicción Total 12M", f"${pred_total:,.0f}")
                    col2.metric("Crecimiento Estimado", f"{crecimiento_pred:.1f}%")
                    col3.metric("Volatilidad", f"{pred_tabla['Predicción'].std():,.0f}")

# Página 3: Vista de Competidores
elif page == "🏢 Vista de Competidores":
    st.header("Vista de Competidores Principales")
    
    competidores_objetivo = ['ESTADO', 'MAPFRE GENERALES', 'LIBERTY', 'AXA GENERALES', 'MUNDIAL', 'PREVISORA']
    
    df_filtered['COMP_NORMALIZADO'] = df_filtered['COMPAÑÍA'].str.upper().str.strip()
    competidores_disponibles = [c for c in competidores_objetivo if c in df_filtered['COMP_NORMALIZADO'].unique()]
    
    if not competidores_disponibles:
        st.warning("No hay datos para los competidores principales con los filtros seleccionados")
        st.info("Compañías disponibles: " + ", ".join(sorted(df_filtered['COMP_NORMALIZADO'].unique()[:15])))
    else:
        df_competidores = df_filtered[df_filtered['COMP_NORMALIZADO'].isin(competidores_disponibles)]
        df_comp_resumen = df_competidores.groupby(['COMPAÑÍA', 'Primas/Siniestros', 'FECHA'])['Valor_Mensual'].sum().reset_index()
        
        tab1, tab2, tab3 = st.tabs(["📈 Comparativa", "📊 Market Share", "🎯 Predicciones"])
        
        with tab1:
            # Gráfico de líneas
            fig = px.line(
                df_comp_resumen,
                x='FECHA',
                y='Valor_Mensual',
                color='COMPAÑÍA',
                line_dash='Primas/Siniestros',
                title="Evolución de Competidores",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla de métricas
            st.subheader("Métricas Últimos 12 Meses")
            df_ultimos = df_comp_resumen[df_comp_resumen['FECHA'] >= (datetime.now() - timedelta(days=365))]
            metricas = df_ultimos.groupby(['COMPAÑÍA', 'Primas/Siniestros'])['Valor_Mensual'].agg(['sum', 'mean', 'std']).round(0)
            st.dataframe(metricas, use_container_width=True)
        
        with tab2:
            st.subheader("Market Share - Últimos 12 Meses")
            
            # Calcular market share
            df_totales = df_ultimos.groupby(['COMPAÑÍA'])['Valor_Mensual'].sum().reset_index()
            df_totales = df_totales.sort_values('Valor_Mensual', ascending=False)
            df_totales['Market_Share_%'] = (df_totales['Valor_Mensual'] / df_totales['Valor_Mensual'].sum() * 100).round(1)
            
            # Gráfico de torta
            col1, col2 = st.columns([1, 2])
            
            with col1:
                fig = px.pie(
                    df_totales,
                    values='Market_Share_%',
                    names='COMPAÑÍA',
                    title="Market Share Total",
                    hole=0.4,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Gráfico de barras
                fig_bar = px.bar(
                    df_totales,
                    x='COMPAÑÍA',
                    y='Market_Share_%',
                    title="Market Share por Compañía",
                    color='Market_Share_%',
                    color_continuous_scale='Viridis',
                    height=400
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Tabla detallada
            st.subheader("📋 Detalle de Market Share")
            df_totales['Valor_Mensual'] = df_totales['Valor_Mensual'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(df_totales, use_container_width=True)
        
        with tab3:
            st.subheader("Predicciones por Competidor")
            
            competidor_seleccionado = st.selectbox("Selecciona competidor para predicción:", competidores_disponibles)
            
            if competidor_seleccionado:
                # Filtrar datos del competidor
                df_comp = df_comp_resumen[
                    df_comp_resumen['COMPAÑÍA'].str.upper().str.strip() == competidor_seleccionado.upper()
                ]
                
                if not df_comp.empty:
                    # Crear pestañas para Primas/Siniestros
                    tab_primas, tab_siniestros = st.tabs(["💰 Predicción Primas", "🚨 Predicción Siniestros"])
                    
                    for tab, tipo in [(tab_primas, 'Primas'), (tab_siniestros, 'Siniestros')]:
                        with tab:
                            df_tipo = df_comp[df_comp['Primas/Siniestros'] == tipo]
                            
                            if not df_tipo.empty:
                                model = train_prophet_model(df_tipo, f"comp_{competidor_seleccionado}_{tipo}")
                                pred = predict_prophet(model, 12)
                                
                                # Gráfico
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=df_tipo['FECHA'],
                                    y=df_tipo['Valor_Mensual'],
                                    mode='lines+markers',
                                    name='Histórico',
                                    line=dict(color='blue')
                                ))
                                fig.add_trace(go.Scatter(
                                    x=pred['ds'],
                                    y=pred['yhat'],
                                    mode='lines+markers',
                                    name='Predicción',
                                    line=dict(color='green', dash='dash')
                                ))
                                
                                fig.update_layout(
                                    title=f"Predicción para {competidor_seleccionado} - {tipo}",
                                    xaxis_title="Fecha",
                                    yaxis_title="Valor Mensual"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Tabla
                                pred_tabla = pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                                pred_tabla['ds'] = pred_tabla['ds'].dt.strftime('%Y-%m')
                                pred_tabla.columns = ['Mes', 'Predicción', 'Límite Inferior', 'Límite Superior']
                                st.dataframe(pred_tabla.round(0), use_container_width=True)
                                
                                # KPIs
                                pred_total = pred_tabla['Predicción'].sum()
                                crecimiento = ((pred_tabla['Predicción'].iloc[-1] - pred_tabla['Predicción'].iloc[0]) / pred_tabla['Predicción'].iloc[0] * 100) if pred_tabla['Predicción'].iloc[0] > 0 else 0
                                
                                col1, col2 = st.columns(2)
                                col1.metric(f"Total 12M {tipo}", f"${pred_total:,.0f}")
                                col2.metric("Crecimiento Estimado", f"{crecimiento:.1f}%")
                            else:
                                st.info(f"No hay datos de {tipo.lower()} para este competidor")
                
                else:
                    st.info("No hay suficientes datos para generar predicciones")

# ==================== FOOTER ====================

st.sidebar.markdown("---")
st.sidebar.info("""
**📊 Sistema de Predicción**
- **Fuente**: Google Sheets
- **Modelos**: Prophet & XGBoost
- **Actualización**: Automática
""")

# Botón de recarga
if st.sidebar.button("🔄 Recargar Datos"):
    st.cache_data.clear()
    st.experimental_rerun()

# Mostrar estado de datos
if st.sidebar.checkbox("Mostrar info de datos"):
    if not df.empty:
        st.sidebar.write(f"📈 Registros: {len(df):,}")
        st.sidebar.write(f"📅 Período: {df['FECHA'].min().strftime('%Y-%m')} a {df['FECHA'].max().strftime('%Y-%m')}")
        st.sidebar.write(f"🏢 Compañías: {len(df['COMPAÑÍA'].unique())}")
        st.sidebar.write(f"🌆 Ciudades: {len(df['CIUDAD'].unique())}")
        st.sidebar.write(f"📋 Ramos: {len(df['RAMOS'].unique())}")
    else:
        st.sidebar.error("No hay datos cargados")

# Créditos
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; font-size: 12px;">
Desarrollado con ❤️ usando Streamlit<br>
Prophet | XGBoost | Plotly
</div>
""", unsafe_allow_html=True)
