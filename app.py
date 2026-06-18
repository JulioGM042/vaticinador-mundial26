import streamlit as st
import pandas as pd
import sqlite3
import joblib
import zipfile
import os
from scipy.stats import poisson

st.set_page_config(page_title="Simulador Mundial 2026 v2", layout="centered", page_icon="⚽")

# ==============================================================================
# 1. DICCIONARIO DE LOS 48 CLASIFICADOS (Mapeo e Imputación)
# ==============================================================================
# Nota: Aquí mapeas el nombre en Español con su ID real de FotMob y su nombre en la DB.
# Podés completar los 48 equipos siguiendo este mismo formato exacto.
DICT_EQUIPOS_2026 = {
    "Argentina": {"id": 8474, "db_name": "Argentina"},
    "Alemania": {"id": 8113, "db_name": "Germany"},
    "Bosnia y Herzegovina": {"id": 10106, "db_name": "Bosnia and Herzegovina"},
    "Brasil": {"id": 8346, "db_name": "Brazil"},
    "Canadá": {"id": 9152, "db_name": "Canada"},
    "Estados Unidos": {"id": 8072, "db_name": "USA"},
    "Francia": {"id": 8480, "db_name": "France"},
    "México": {"id": 8419, "db_name": "Mexico"},
    "Portugal": {"id": 8449, "db_name": "Portugal"},
    "Suiza": {"id": 6717, "db_name": "Switzerland"},
    # ... Agrega aquí el resto de las selecciones clasificadas según tu DB
}

# Perfil Base de Emergencia (Si un equipo no tiene ningún dato histórico en la DB)
PERFIL_BASE_MUNDIAL = {
    'xg_ataque_roll_3': 1.25,
    'xgot_ataque_roll_3': 1.20,
    'posesion_roll_3': 50.0,
    'tiros_arco_roll_3': 4.5,
    'corners_roll_3': 4.0
}

# ==============================================================================
# 2. PROCESAMIENTO DE DATOS CON MITIGACIÓN DE ERRORES (FALLBACK)
# ==============================================================================
@st.cache_resource
def cargar_modelos():
    modelo = joblib.load('modelo_rlm_campeon.pkl')
    scaler = joblib.load('scaler_campeon.pkl')
    return modelo, scaler

@st.cache_data
def procesar_historico_completo():
    if not os.path.exists('mundial2026.db'):
        with zipfile.ZipFile('mundial2026.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
            
    conn = sqlite3.connect('mundial2026.db')
    df = pd.read_sql("SELECT * FROM estadisticas_partidos_filtradas", conn)
    conn.close()
    
    df['fecha_parsed'] = pd.to_datetime(df['fecha'], errors='coerce')
    
    df_l = df[['id_partido', 'fecha_parsed', 'local_id', 'local_name', 'xg_home', 'xgot_home', 'possession_home', 'shots_on_target_home', 'corners_home']].copy()
    df_l.columns = ['id_partido', 'fecha', 'equipo_id', 'equipo_name', 'xg_ataque', 'xgot_ataque', 'posesion', 'tiros_arco', 'corners']
    
    df_v = df[['id_partido', 'fecha_parsed', 'visitante_id', 'visitante_name', 'xg_away', 'xgot_away', 'possession_away', 'shots_on_target_away', 'corners_away']].copy()
    df_v.columns = ['id_partido', 'fecha', 'equipo_id', 'equipo_name', 'xg_ataque', 'xgot_ataque', 'posesion', 'tiros_arco', 'corners']
    
    df_hist = pd.concat([df_l, df_v]).sort_values(by=['equipo_id', 'fecha']).reset_index(drop=True)
    
    # Calculamos las métricas móviles habituales
    columnas = ['xg_ataque', 'xgot_ataque', 'posesion', 'tiros_arco', 'corners']
    for col in columnas:
        df_hist[f'{col}_roll_3'] = df_hist.groupby('equipo_id')[col].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
        
    return df_hist

modelo, scaler = cargar_modelos()
df_historico = procesar_historico_completo()

# ==============================================================================
# 3. INTERFAZ DE USUARIO (UI) CON FILTRO EN ESPAÑOL
# ==============================================================================
st.title("⚽ Simulador Analítico - Mundial 2026 (V2)")
st.write("Proyecciones estadísticas basadas en Big Data y Machine Learning.")

# El selector ahora usa las llaves en español del diccionario configurado
lista_espanol = sorted(list(DICT_EQUIPOS_2026.keys()))

col1, col2 = st.columns(2)
with col1:
    local_es = st.selectbox("🏠 Selección Local", lista_espanol, index=lista_espanol.index("Suiza") if "Suiza" in lista_espanol else 0)
with col2:
    visitante_es = st.selectbox("🚀 Selección Visitante", lista_espanol, index=lista_espanol.index("Bosnia y Herzegovina") if "Bosnia y Herzegovina" in lista_espanol else 1)

if local_es == visitante_es:
    st.warning("Elegí dos selecciones diferentes para la simulación.")
else:
    # Obtener metadatos internos desde el diccionario de mapeo
    meta_local = DICT_EQUIPOS_2026[local_es]
    meta_visit = DICT_EQUIPOS_2026[visitante_es]
    
    # LÓGICA DE EXTRACCIÓN CON FALLBACK (Subsumir falta de historial)
    def extraer_metricas_equipo(meta_equipo):
        # Intentar buscar por ID o por nombre en inglés en el histórico elaborado
        df_team = df_historico[df_historico['equipo_id'] == meta_equipo['id']].copy()
        if df_team.empty:
            df_team = df_historico[df_historico['equipo_name'] == meta_equipo['db_name']].copy()
            
        # Filtro de última fila sin limpiar por dropna general
        ultima_fila = df_team.tail(1)
        
        # Validar si la fila contiene datos utilizables en el rolling, si no, ir al plan B
        if not ultima_fila.empty and not pd.isna(ultima_fila['xg_ataque_roll_3'].values[0]):
            return {
                'xg': ultima_fila['xg_ataque_roll_3'].values[0],
                'xgot': ultima_fila['xgot_ataque_roll_3'].values[0],
                'pos': ultima_fila['posesion_roll_3'].values[0],
                'tiros': ultima_fila['tiros_arco_roll_3'].values[0],
                'corners': ultima_fila['corners_roll_3'].values[0],
                'origen': "Métricas Móviles (OK)"
            }
        elif not df_team.empty:
            # Plan B: Si no tiene rolling por falta de partidos previos, usamos su promedio histórico general bruto
            return {
                'xg': df_team['xg_ataque'].mean(),
                'xgot': df_team['xgot_ataque'].mean(),
                'pos': df_team['posesion'].mean(),
                'tiros': df_team['tiros_arco'].mean(),
                'corners': df_team['corners'].mean(),
                'origen': "Promedio Histórico Total (Datos insuficientes para Rolling)"
            }
        else:
            # Plan C: Si el equipo no figura en la base de datos, se le asigna el Perfil Base Neutral
            res = PERFIL_BASE_MUNDIAL.copy()
            return {
                'xg': res['xg_ataque_roll_3'], 'xgot': res['xgot_ataque_roll_3'],
                'pos': res['posesion_roll_3'], 'tiros': res['tiros_arco_roll_3'],
                'corners': res['corners_roll_3'], 'origen': "Perfil Base Asignado (Sin registros en DB)"
            }

    medidas_l = extraer_metricas_equipo(meta_local)
    medidas_v = extraer_metricas_equipo(meta_visit)

    # UI de Estados de Forma
    st.subheader("📊 Estados de Forma Previos")
    c_l, c_v = st.columns(2)
    with c_l:
        st.markdown(f"**{local_es}**")
        st.caption(f"🔹 xG Promedio: {medidas_l['xg']:.2f}")
        st.caption(f"🔹 Posesión: {medidas_l['pos']:.0f}%")
        st.caption(f"🔹 Tiros al Arco: {medidas_l['tiros']:.1f}")
        st.text(f"ℹ️ Fuente: {medidas_l['origen']}")
    with c_v:
        st.markdown(f"**{visitante_es}**")
        st.caption(f"🔹 xG Promedio: {medidas_v['xg']:.2f}")
        st.caption(f"🔹 Posesión: {medidas_v['pos']:.0f}%")
        st.caption(f"🔹 Tiros al Arco: {medidas_v['tiros']:.1f}")
        st.text(f"ℹ️ Fuente: {medidas_v['origen']}")
        
    if st.button("🔥 SIMULAR PARTIDO", use_container_width=True):
        features = ['diff_xg_roll', 'diff_xgot_roll', 'diff_possession_roll', 'diff_tiros_arco_roll', 'diff_corners_roll']
        
        dat_partido = pd.DataFrame({
            'diff_xg_roll': medidas_l['xg'] - medidas_v['xg'],
            'diff_xgot_roll': medidas_l['xgot'] - medidas_v['xgot'],
            'diff_possession_roll': medidas_l['pos'] - medidas_v['pos'],
            'diff_tiros_arco_roll': medidas_l['tiros'] - medidas_v['tiros'],
            'diff_corners_roll': medidas_l['corners'] - medidas_v['corners']
        }, index=[0])
        
        X_scaled = scaler.transform(dat_partido[features])
        probs = modelo.predict_proba(X_scaled)[0]
        clase = modelo.predict(X_scaled)[0]
        
        st.markdown("---")
        st.subheader("🔮 Probabilidades de Resultado (Regresión Logística)")
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric(f"🟢 Victoria {local_es}", f"{probs[0]*100:.1f}%")
        col_p2.metric("🟡 Empate", f"{probs[1]*100:.1f}%")
        col_p3.metric(f"🔴 Victoria {visitante_es}", f"{probs[2]*100:.1f}%")
        
        # Poisson
        p_l = [poisson.pmf(i, medidas_l['xg']) for i in range(6)]
        p_v = [poisson.pmf(i, medidas_v['xg']) for i in range(6)]
        res_poisson = []
        for l in range(6):
            for v in range(6):
                res_poisson.append({'m': f"{l} - {v}", 'p': p_l[l] * p_v[v] * 100})
        df_r = pd.DataFrame(res_poisson).sort_values(by='p', ascending=False).head(3)
        
        st.subheader("🎯 Marcadores Exactos Más Probables (Poisson)")
        for i, r in df_r.iterrows():
            st.write(f"• **{local_es} {r['m']} {visitante_es}** -> Probabilidad: **{r['p']:.1f}%**")
            
        etiquetas = {0: f'VICTORIA {local_es.upper()}', 1: 'EMPATE', 2: f'VICTORIA {visitante_es.upper()}'}
        st.success(f"🏆 PREDICCIÓN GLOBAL: {etiquetas[clase]}")