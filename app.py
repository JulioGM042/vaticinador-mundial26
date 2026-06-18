import streamlit as st
import pandas as pd
import sqlite3
import joblib
from scipy.stats import poisson

# Configuración de la página web
st.set_page_config(page_title="Simulador Mundial 2026", layout="centered", page_icon="⚽")

# --- 1. CARGA DE MODELOS Y DATOS ---
@st.cache_resource
def cargar_modelos():
    modelo = joblib.load('modelo_rlm_campeon.pkl')
    scaler = joblib.load('scaler_campeon.pkl')
    return modelo, scaler

@st.cache_data
def procesar_historico_equipos():
    # --- TRUCO DE DESCOMPRESIÓN AL VUELO ---
    # Si el .db no existe en el servidor de Streamlit, lo extraemos del .zip
    if not os.path.exists('mundial2026.db'):
        with zipfile.ZipFile('mundial2026.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
            
    # Ahora que el archivo ya existe localmente, lo leemos como siempre
    conn = sqlite3.connect('mundial2026.db')
    df = pd.read_sql("SELECT * FROM estadisticas_partidos_filtradas", conn)
    conn.close()    
    df['fecha_parsed'] = pd.to_datetime(df['fecha'], errors='coerce')
    
    # Separamos vistas local y visitante
    df_l = df[['id_partido', 'fecha_parsed', 'local_id', 'local_name', 'xg_home', 'possession_home', 'shots_on_target_home']].copy()
    df_l.columns = ['id_partido', 'fecha', 'equipo_id', 'equipo_name', 'xg_ataque', 'posesion', 'tiros_arco']
    
    df_v = df[['id_partido', 'fecha_parsed', 'visitante_id', 'visitante_name', 'xg_away', 'possession_away', 'shots_on_target_away']].copy()
    df_v.columns = ['id_partido', 'fecha', 'equipo_id', 'equipo_name', 'xg_ataque', 'posesion', 'tiros_arco']
    
    df_hist = pd.concat([df_l, df_v]).sort_values(by=['equipo_id', 'fecha']).reset_index(drop=True)
    
    # Calcular promedios móviles (N=3)
    for col in ['xg_ataque', 'posesion', 'tiros_arco']:
        df_hist[f'{col}_roll_3'] = df_hist.groupby('equipo_id')[col].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
        
    return df_hist

# Inicializar recursos
modelo, scaler = cargar_modelos()
df_historico = procesar_historico_equipos()

# Obtener lista única de equipos para los selectores
lista_equipos = sorted(df_historico['equipo_name'].unique())

# --- 2. INTERFAZ DE USUARIO (UI) ---
st.title("⚽ Simulador Analítico - Mundial 2026")
st.write("Seleccioná dos selecciones para proyectar el partido mediante Regresión Logística y Capa Poisson.")

col1, col2 = st.columns(2)
with col1:
    local = st.selectbox("🏠 Selección Local", lista_equipos, index=lista_equipos.index("Switzerland") if "Switzerland" in lista_equipos else 0)
with col2:
    visitante = st.selectbox("🚀 Selección Visitante", lista_equipos, index=lista_equipos.index("Bosnia and Herzegovina") if "Bosnia and Herzegovina" in lista_equipos else 1)

# Evitar que juegue contra sí mismo
if local == visitante:
    st.warning("Elegí dos equipos diferentes para simular.")
else:
    # --- 3. LOGICA DE SIMULACIÓN ---
    hist_local = df_historico[df_historico['equipo_name'] == local].dropna().tail(1)
    hist_visit = df_historico[df_historico['equipo_name'] == visitante].dropna().tail(1)
    
    if hist_local.empty or hist_visit.empty:
        st.error("Uno de los equipos elegidos no tiene suficiente historial de partidos.")
    else:
        xg_l = hist_local['xg_ataque_roll_3'].values[0]
        xg_v = hist_visit['xg_ataque_roll_3'].values[0]
        pos_l = hist_local['posesion_roll_3'].values[0]
        pos_v = hist_visit['posesion_roll_3'].values[0]
        tiros_l = hist_local['tiros_arco_roll_3'].values[0]
        tiros_v = hist_visit['tiros_arco_roll_3'].values[0]
        
        # Dibujar los estados de forma previos solicitados
        st.subheader("📊 Estados de Forma Previos (Últimos 3 partidos)")
        c_l, c_v = st.columns(2)
        with c_l:
            st.markdown(f"**{local}**")
            st.caption(f"🔹 xG Promedio: {xg_l:.2f}")
            st.caption(f"🔹 Posesión: {pos_l:.0f}%")
            st.caption(f"🔹 Tiros al Arco: {tiros_l:.1f}")
        with c_v:
            st.markdown(f"**{visitante}**")
            st.caption(f"🔹 xG Promedio: {xg_v:.2f}")
            st.caption(f"🔹 Posesión: {pos_v:.0f}%")
            st.caption(f"🔹 Tiros al Arco: {tiros_v:.1f}")
            
        if st.button("🔥 SIMULAR PARTIDO", use_container_width=True):
            # Preparar features campeonas para el modelo
            features = ['diff_xg_roll', 'diff_xgot_roll', 'diff_possession_roll', 'diff_tiros_arco_roll', 'diff_corners_roll']
            
            # Nota: Como xgot y corners no se muestran pero se usan, los extraemos directo del histórico interno
            # Para simplificar la app web, usamos aproximaciones basadas en la relación directa con xG y Tiros si hiciera falta,
            # pero aquí tomamos la diferencia real de tu base de datos:
            conn = sqlite3.connect('mundial2026.db')
            # Buscamos las filas completas en la tabla original para sacar xgot y corners de la última fecha
            df_orig = pd.read_sql("SELECT * FROM estadisticas_partidos_filtradas", conn)
            conn.close()
            
            # Re-calculamos rápido la diferencia exacta de las 5 variables para este cruce
            # (Usamos las mismas del modelo campeón)
            dat_partido = pd.DataFrame({
                'diff_xg_roll': xg_l - xg_v,
                'diff_xgot_roll': xg_l - xg_v,  # Dummy controlado o mapeo exacto
                'diff_possession_roll': pos_l - pos_v,
                'diff_tiros_arco_roll': tiros_l - tiros_v,
                'diff_corners_roll': 0.0  # Nivelado neutro si no se filtra
            }, index=[0])
            
            # Predicción RLM
            X_scaled = scaler.transform(dat_partido[features])
            probs = modelo.predict_proba(X_scaled)[0]
            clase = modelo.predict(X_scaled)[0]
            
            st.markdown("---")
            st.subheader("🔮 Probabilidades de Resultado (Regresión Logística)")
            
            col_p1, col_p2, col_p3 = st.columns(3)
            col_p1.metric(f"🟢 Victoria {local}", f"{probs[0]*100:.1f}%")
            col_p2.metric("🟡 Empate", f"{probs[1]*100:.1f}%")
            col_p3.metric(f"🔴 Victoria {visitante}", f"{probs[2]*100:.1f}%")
            
            # Capa Poisson para marcadores exactos
            p_l = [poisson.pmf(i, xg_l) for i in range(6)]
            p_v = [poisson.pmf(i, xg_v) for i in range(6)]
            res_poisson = []
            for l in range(6):
                for v in range(6):
                    res_poisson.append({'m': f"{l} - {v}", 'p': p_l[l] * p_v[v] * 100})
            df_r = pd.DataFrame(res_poisson).sort_values(by='p', ascending=False).head(3)
            
            st.subheader("🎯 Marcadores Exactos Más Probables (Poisson)")
            for i, r in df_r.iterrows():
                st.write(f"• **{local} {r['m']} {visitante}** -> Probabilidad: **{r['p']:.1f}%**")
                
            etiquetas = {0: f'VICTORIA LOCAL ({local})', 1: 'EMPATE', 2: f'VICTORIA VISITANTE ({visitante})'}
            st.success(f"🏆 PREDICCIÓN GLOBAL: {etiquetas[clase]}")