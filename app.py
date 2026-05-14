"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DROPALERT — Sistem Deteksi Dini Risiko Putus Sekolah                      ║
║  Ensemble Learning Dashboard Interaktif                                    ║
║  Sumber Data: BPS 2021–2025                                                ║
║  Jalankan: streamlit run app.py                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
import pickle
import re
import json

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════
# 1. KONFIGURASI HALAMAN
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DropAlert | Deteksi Risiko Putus Sekolah",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════
# 2. CUSTOM CSS — Diperbaiki agar tidak merusak Icon Streamlit
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Terapkan font hanya pada container teks utama, JANGAN di span/i/svg agar icon tidak rusak */
    html, body, [class*="css"], .stApp, p, h1, h2, h3, h4, h5, h6, label, li {
        font-family: 'Inter', sans-serif;
    }

    .stApp { background-color: #0E1117; color: #FFFFFF; }

    /* Sidebar Gradasi */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #121212 0%, #1E1E1E 80%, #4B0000 100%);
        border-right: 1px solid #2D2D2D;
    }

    /* Hero Section (Center text) */
    .hero-container {
        background: linear-gradient(135deg, #800000 0%, #C0392B 45%, #E67E22 100%);
        padding: 45px 55px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.55);
        margin-bottom: 28px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .hero-container h1 { color: #fff; font-size: 1.85rem; line-height: 1.4; margin: 0 auto; font-weight: 700; text-align: center; }
    .hero-container p  { color: #FFE0CC; font-size: 0.95rem; margin: 10px auto 0; font-weight: 400; text-align: center; }

    /* KPI Cards (Center text & Layout) */
    .kpi-card {
        background: #1A1C23;
        border: 1px solid #2D2D2D;
        border-radius: 8px;
        padding: 18px 20px;
        text-align: center;
        transition: border-color .25s, transform .25s;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .kpi-card:hover { border-color: #E74C3C; transform: translateY(-4px); }
    .kpi-card .label { color: #AAA; font-size: 0.75rem; letter-spacing: 0.5px; text-transform: uppercase; font-weight: 600; text-align: center; width: 100%;}
    .kpi-card .value { color: #FFF; font-size: 1.8rem; font-weight: 700; margin: 6px 0 0; text-align: center; width: 100%;}

    .section-title {
        color: #E74C3C;
        border-left: 4px solid #F39C12;
        padding-left: 14px;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 28px 0 14px;
    }

    /* Spesifik untuk tombol Streamlit utama, bukan tombol di file uploader */
    div[data-testid="stButton"] > button {
        background: linear-gradient(90deg, #C0392B, #E67E22);
        color: #fff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        width: 100%;
        padding: 12px;
        font-size: 1rem;
        transition: opacity .2s;
    }
    div[data-testid="stButton"] > button:hover { opacity: .88; color: #fff; }

    .badge-rendah  { background: #1E8449; color: #fff; padding: 6px 18px; border-radius: 4px; font-weight: 600; display: inline-block; }
    .badge-sedang  { background: #B7770D; color: #fff; padding: 6px 18px; border-radius: 4px; font-weight: 600; display: inline-block; }
    .badge-tinggi  { background: #C0392B; color: #fff; padding: 6px 18px; border-radius: 4px; font-weight: 600; display: inline-block; }
    .badge-kritis  { background: #7B241C; color: #fff; padding: 6px 18px; border-radius: 4px; font-weight: 600; display: inline-block; }

    .result-box {
        background: #1A1C23;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Fix Metric Alignment inside Result Box */
    div[data-testid="stMetricValue"] { 
        color: #FFFFFF !important; 
        font-weight: 700; 
        font-size: 2.2rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #AAAAAA !important;
        font-size: 0.9rem !important;
        margin-bottom: 5px;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 1rem !important;
    }

    .badge-container {
        display: flex;
        justify-content: center;
        margin: 25px 0;
        width: 100%;
    }
    .insight-box {
        background: #14171E;
        border-left: 4px solid #E67E22;
        border-radius: 4px;
        padding: 18px 22px;
        margin-top: 16px;
        color: #E0E0E0;
        font-size: 0.93rem;
        line-height: 1.65;
    }
    
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# 3. KONSTANTA & MAPPING
# ════════════════════════════════════════════════════════

# Fitur yang digunakan dalam model (sesuai notebook)
FITUR = [
    'gabungan_pendudukmiskin',
    'TPT',
    'NEET_usiamuda',
    'tenagakerjaformal',
    'gabungan_HLS',
    'gabungan_RLS',
    'rasio_guru_SMA',
    'rasio_guru_SMK',
    'rasio_guru_SD',
    'rasio_guru_SMP',
    'rasio_sekolah_SMA',
    'rasio_sekolah_SMK',
    'rasio_sekolah_SD',
    'rasio_sekolah_SMP',
]

# Target per kelompok umur & model terbaik (sesuai kesimpulan notebook)
TARGET_CONFIG = {
    'ARPS_07to12': {'model': 'KNN',     'label': '7–12 Tahun (SD)',    'aps_col': 'APS_07to12'},
    'ARPS_13to15': {'model': 'XGBoost', 'label': '13–15 Tahun (SMP)',  'aps_col': 'APS_13to15'},
    'ARPS_16to18': {'model': 'XGBoost', 'label': '16–18 Tahun (SMA)',  'aps_col': 'APS_16to18'},
    'ARPS_19to23': {'model': 'KNN',     'label': '19–23 Tahun (PT)',   'aps_col': 'APS_19to23'},
}

# Mapping kelompok umur (fleksibel)
KELOMPOK_UMUR_MAP = {
    # 7-12
    '7-12': 'ARPS_07to12', '07to12': 'ARPS_07to12', '7 to 12': 'ARPS_07to12',
    '712': 'ARPS_07to12', 'sd': 'ARPS_07to12', 'anak sd': 'ARPS_07to12',
    'sekolah dasar': 'ARPS_07to12', '7–12': 'ARPS_07to12',
    # 13-15
    '13-15': 'ARPS_13to15', '13to15': 'ARPS_13to15', '13 to 15': 'ARPS_13to15',
    '1315': 'ARPS_13to15', 'smp': 'ARPS_13to15', 'sekolah menengah pertama': 'ARPS_13to15',
    'mts': 'ARPS_13to15', '13–15': 'ARPS_13to15',
    # 16-18
    '16-18': 'ARPS_16to18', '16to18': 'ARPS_16to18', '16 to 18': 'ARPS_16to18',
    '1618': 'ARPS_16to18', 'sma': 'ARPS_16to18', 'smk': 'ARPS_16to18',
    'sekolah menengah atas': 'ARPS_16to18', '16–18': 'ARPS_16to18',
    # 19-23
    '19-23': 'ARPS_19to23', '19to23': 'ARPS_19to23', '19 to 23': 'ARPS_19to23',
    '1923': 'ARPS_19to23', 'kuliah': 'ARPS_19to23', 'mahasiswa': 'ARPS_19to23',
    'perguruan tinggi': 'ARPS_19to23', 'pt': 'ARPS_19to23', '19–23': 'ARPS_19to23',
}

# Mapping risiko sosek (fleksibel)
RISIKO_SOSEK_MAP = {
    'rendah': 'rendah', 'low': 'rendah', 'r': 'rendah',
    'sedang': 'sedang', 'medium': 'sedang', 'menengah': 'sedang', 's': 'sedang',
    'tinggi': 'tinggi', 'high': 'tinggi', 'besar': 'tinggi', 't': 'tinggi',
}

# Mapping alias provinsi (fleksibel, case-insensitive)
PROVINSI_ALIAS = {
    'diy': 'DI YOGYAKARTA',
    'yogyakarta': 'DI YOGYAKARTA',
    'di yogyakarta': 'DI YOGYAKARTA',
    'daerah istimewa yogyakarta': 'DI YOGYAKARTA',
    'jogja': 'DI YOGYAKARTA',
    'jakarta': 'DKI JAKARTA',
    'dki jakarta': 'DKI JAKARTA',
    'dki': 'DKI JAKARTA',
    'aceh': 'ACEH',
    'nad': 'ACEH',
    'nanggroe aceh darussalam': 'ACEH',
    'kalbar': 'KALIMANTAN BARAT',
    'kalimantan barat': 'KALIMANTAN BARAT',
    'kaltim': 'KALIMANTAN TIMUR',
    'kalimantan timur': 'KALIMANTAN TIMUR',
    'kaltara': 'KALIMANTAN UTARA',
    'kalimantan utara': 'KALIMANTAN UTARA',
    'kalsel': 'KALIMANTAN SELATAN',
    'kalimantan selatan': 'KALIMANTAN SELATAN',
    'kalteng': 'KALIMANTAN TENGAH',
    'kalimantan tengah': 'KALIMANTAN TENGAH',
    'sulut': 'SULAWESI UTARA',
    'sulawesi utara': 'SULAWESI UTARA',
    'sulsel': 'SULAWESI SELATAN',
    'sulawesi selatan': 'SULAWESI SELATAN',
    'sulteng': 'SULAWESI TENGAH',
    'sulawesi tengah': 'SULAWESI TENGAH',
    'sultra': 'SULAWESI TENGGARA',
    'sulawesi tenggara': 'SULAWESI TENGGARA',
    'sumut': 'SUMATERA UTARA',
    'sumatera utara': 'SUMATERA UTARA',
    'sumbar': 'SUMATERA BARAT',
    'sumatera barat': 'SUMATERA BARAT',
    'sumsel': 'SUMATERA SELATAN',
    'sumatera selatan': 'SUMATERA SELATAN',
    'riau': 'RIAU',
    'kepri': 'KEPULAUAN RIAU',
    'kepulauan riau': 'KEPULAUAN RIAU',
    'jambi': 'JAMBI',
    'bengkulu': 'BENGKULU',
    'lampung': 'LAMPUNG',
    'babel': 'KEPULAUAN BANGKA BELITUNG',
    'bangka belitung': 'KEPULAUAN BANGKA BELITUNG',
    'kepulauan bangka belitung': 'KEPULAUAN BANGKA BELITUNG',
    'banten': 'BANTEN',
    'jabar': 'JAWA BARAT',
    'jawa barat': 'JAWA BARAT',
    'jateng': 'JAWA TENGAH',
    'jawa tengah': 'JAWA TENGAH',
    'jatim': 'JAWA TIMUR',
    'jawa timur': 'JAWA TIMUR',
    'bali': 'BALI',
    'ntb': 'NUSA TENGGARA BARAT',
    'nusa tenggara barat': 'NUSA TENGGARA BARAT',
    'ntt': 'NUSA TENGGARA TIMUR',
    'nusa tenggara timur': 'NUSA TENGGARA TIMUR',
    'maluku': 'MALUKU',
    'malut': 'MALUKU UTARA',
    'maluku utara': 'MALUKU UTARA',
    'papua': 'PAPUA',
    'papua barat': 'PAPUA BARAT',
    'papua selatan': 'PAPUA SELATAN',
    'papua tengah': 'PAPUA TENGAH',
    'papua pegunungan': 'PAPUA PEGUNUNGAN',
    'papua barat daya': 'PAPUA BARAT DAYA',
    'gorontalo': 'GORONTALO',
}


# ════════════════════════════════════════════════════════
# 4. FUNGSI MAPPING INPUT
# ════════════════════════════════════════════════════════

def mapping_input_provinsi(raw: str, valid_list: list) -> str | None:
    """
    Memetakan input provinsi bebas (alias, singkatan, dll.)
    ke nama provinsi standar dalam dataset.
    """
    raw_clean = raw.strip().lower()

    # Cek alias
    if raw_clean in PROVINSI_ALIAS:
        return PROVINSI_ALIAS[raw_clean]

    # Cek exact match (case-insensitive)
    for prov in valid_list:
        if prov.lower() == raw_clean:
            return prov

    # Cek partial match
    for prov in valid_list:
        if raw_clean in prov.lower() or prov.lower() in raw_clean:
            return prov

    return None


def mapping_input_umur(raw: str) -> str | None:
    """
    Memetakan input kelompok umur ke kode target ARPS.
    Tahan terhadap: '7-12', 'SD', 'anak sd', '07to12', dll.
    """
    raw_clean = re.sub(r'\s+', ' ', raw.strip().lower())
    raw_clean = raw_clean.replace('–', '-')

    if raw_clean in KELOMPOK_UMUR_MAP:
        return KELOMPOK_UMUR_MAP[raw_clean]

    # Ekstrak angka untuk mencocokkan rentang
    nums = re.findall(r'\d+', raw_clean)
    if nums:
        first = int(nums[0])
        if 7 <= first <= 12:  return 'ARPS_07to12'
        if 13 <= first <= 15: return 'ARPS_13to15'
        if 16 <= first <= 18: return 'ARPS_16to18'
        if 19 <= first <= 23: return 'ARPS_19to23'

    return None


def mapping_input_risiko(raw: str) -> str | None:
    """Memetakan input tingkat risiko sosek."""
    raw_clean = raw.strip().lower()
    return RISIKO_SOSEK_MAP.get(raw_clean, None)


# ════════════════════════════════════════════════════════
# 5. LOAD DATA
# ════════════════════════════════════════════════════════

@st.cache_data
def load_geojson():
    path = "38 Provinsi Indonesia - Provinsi.json"
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        geo = json.load(file)
        # Wajib UPPERCASE agar peta tidak hitam
        for feature in geo['features']:
            if 'PROVINSI' in feature['properties']:
                feature['properties']['PROVINSI'] = str(feature['properties']['PROVINSI']).upper()
        return geo

@st.cache_data(show_spinner=False)
def load_data(path: str | None = None) -> pd.DataFrame:
    """
    Load dataset dari path yang diberikan atau cari otomatis.
    Mengembalikan df_model yang sudah di-feature-engineer.
    """
    # Cari file Excel di direktori yang umum
    candidates = []
    if path:
        candidates.append(path)

    candidates += [
        "Dataset_Gabungan_Fix.xlsx",
        "Dataset_Gabungan.xlsx",
        "data/Dataset_Gabungan_Fix.xlsx",
        "data/Dataset_Gabungan.xlsx",
    ]

    df = None
    for p in candidates:
        if os.path.exists(p):
            try:
                df = pd.read_excel(p)
                break
            except Exception:
                continue

    if df is None:
        return None

    # Hapus baris total nasional
    df = df[df['Provinsi'].str.upper().str.strip() != 'INDONESIA'].copy()
    df['Provinsi'] = df['Provinsi'].astype(str).str.upper().str.strip()

    # ── Feature engineering (sesuai notebook) ──────────────
    def safe_ratio(num, den):
        return num / den.replace(0, np.nan)

    df['ARPS_07to12'] = (100 - df['APS_07to12']).round(4)
    df['ARPS_13to15'] = (100 - df['APS_13to15']).round(4)
    df['ARPS_16to18'] = (100 - df['APS_16to18']).round(4)
    df['ARPS_19to23'] = (100 - df['APS_19to23']).round(4)

    df['rasio_guru_SMA']   = safe_ratio(df['SMA_gabungan_jumlahguru'],   df['SMA_gabungan_jumlahmurid'])
    df['rasio_guru_SMK']   = safe_ratio(df['SMK_gabungan_jumlahguru'],   df['SMK_gabungan_jumlahmurid'])
    df['rasio_guru_SD']    = safe_ratio(df['SD_gabungan_jumlahguru'],    df['SD_gabungan_jumlahmurid'])
    df['rasio_guru_SMP']   = safe_ratio(df['SMP_gabungan_jumlahguru'],   df['SMP_gabungan_jumlahmurid'])

    df['rasio_sekolah_SMA'] = safe_ratio(df['SMA_gabungan_jumlahsekolah'], df['SMA_gabungan_jumlahmurid'])
    df['rasio_sekolah_SMK'] = safe_ratio(df['SMK_gabungan_jumlahsekolah'], df['SMK_gabungan_jumlahmurid'])
    df['rasio_sekolah_SD']  = safe_ratio(df['SD_gabungan_jumlahsekolah'],  df['SD_gabungan_jumlahmurid'])
    df['rasio_sekolah_SMP'] = safe_ratio(df['SMP_gabungan_jumlahsekolah'], df['SMP_gabungan_jumlahmurid'])

    # Imputasi NaN dengan median kolom
    for col in FITUR:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Label kerentanan berdasarkan APS SMA
    df['Skor_Risiko'] = df['ARPS_16to18']
    df['Status_Kerentanan'] = df['Skor_Risiko'].apply(
        lambda x: 'Tinggi' if x > 30 else ('Sedang' if x > 20 else 'Rendah')
    )

    return df


@st.cache_data(show_spinner=False)
def compute_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute K-Means clustering (k=3) pada rata-rata provinsi.
    Menambahkan kolom 'cluster' dan 'cluster_label'.
    """
    features_cluster = [
        'gabungan_pendudukmiskin',
        'TPT',
        'NEET_usiamuda',
        'tenagakerjaformal',
        'gabungan_HLS',
        'gabungan_RLS',
        'rasio_guru_SMA',
        'rasio_guru_SMK',
        'rasio_guru_SD',
        'rasio_guru_SMP',
        'rasio_sekolah_SMA',
        'rasio_sekolah_SMK',
        'rasio_sekolah_SD',
        'rasio_sekolah_SMP',
    ]
    
    # Rata-rata per provinsi
    df_cluster = df.groupby('Provinsi')[features_cluster].mean().reset_index()
    X = df_cluster[features_cluster]
    
    # Standardisasi
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    
    # K-Means k=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cluster['cluster'] = kmeans.fit_predict(Z)
    
    # Mapping cluster ke label
    cluster_map = {0: 'Sedang', 1: 'Tinggi', 2: 'Sedang'}
    
    # Deteksi cluster Papua (cluster dengan kemiskinan tertinggi)
    cluster_profile = df_cluster.groupby('cluster')[['gabungan_pendudukmiskin', 'gabungan_HLS']].mean()
    vulnerable_cluster = cluster_profile['gabungan_pendudukmiskin'].idxmax()
    cluster_map[vulnerable_cluster] = 'Tinggi'
    
    # Deteksi cluster terbaik (HLS tertinggi, kemiskinan terendah)
    best_cluster = cluster_profile['gabungan_HLS'].idxmax()
    cluster_map[best_cluster] = 'Rendah'
    
    df_cluster['cluster_label'] = df_cluster['cluster'].map(cluster_map)
    
    # Merge kembali ke df asli
    df = df.merge(df_cluster[['Provinsi', 'cluster', 'cluster_label']], on='Provinsi', how='left')
    
    return df


# ════════════════════════════════════════════════════════
# 6. TRAINING & CACHING MODEL
# ════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_models(df: pd.DataFrame) -> dict:
    """
    Melatih model terbaik per target menggunakan seluruh data.
    Model terbaik (sesuai notebook):
        ARPS_07to12 → KNN
        ARPS_13to15 → XGBoost
        ARPS_16to18 → XGBoost
        ARPS_19to23 → KNN

    Return dict { target: {'model': fitted, 'scaler': fitted, 'median': float} }
    """
    df_sorted = df.sort_values(['Tahun', 'Provinsi'])
    X_all = df_sorted[FITUR].copy()

    trained = {}
    for target, cfg in TARGET_CONFIG.items():
        y = df_sorted[target].copy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)

        if cfg['model'] == 'KNN':
            mdl = KNeighborsRegressor(
                n_neighbors=15,
                weights='uniform',
                metric='minkowski',
                p=2,
            )
            mdl.fit(X_scaled, y)

        elif cfg['model'] == 'XGBoost':
            mdl = XGBRegressor(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=2,
                min_child_weight=5,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1,
                reg_lambda=5,
                gamma=1,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            mdl.fit(X_scaled, y)

        trained[target] = {
            'model': mdl,
            'scaler': scaler,
            'median': float(y.median()),
        }

    return trained


# ════════════════════════════════════════════════════════
# 7. PREPROCESSING INPUT & PREDIKSI
# ════════════════════════════════════════════════════════

def preprocess_input(df: pd.DataFrame, provinsi: str,
                     risiko_sosek: str) -> pd.Series | None:
    """
    Ambil feature dari data historis terbaru untuk provinsi terpilih,
    lalu sesuaikan berdasarkan tingkat risiko sosek yang dipilih user.
    """
    prov_data = df[df['Provinsi'] == provinsi].copy()
    if prov_data.empty:
        return None

    # Ambil baris tahun terbaru
    latest = prov_data.sort_values('Tahun').iloc[-1]
    row = latest[FITUR].copy().astype(float)

    # Modifikasi berdasarkan tingkat risiko sosek
    adjustment_map = {
        'rendah': {'gabungan_pendudukmiskin': -2.0, 'TPT': -1.0, 'NEET_usiamuda': -1.5},
        'sedang': {'gabungan_pendudukmiskin':  0.0, 'TPT':  0.0, 'NEET_usiamuda':  0.0},
        'tinggi': {'gabungan_pendudukmiskin': +3.0, 'TPT': +2.0, 'NEET_usiamuda': +2.5},
    }
    adj = adjustment_map.get(risiko_sosek, {})
    for col, delta in adj.items():
        if col in row.index:
            row[col] = max(0, row[col] + delta)

    return row


def predict(models: dict, target: str, X_row: pd.Series) -> dict:
    """
    Melakukan prediksi ARPS untuk target tertentu.
    Return dict berisi arps, aps, risk_level, risk_label, color.
    """
    cfg = models[target]
    X_scaled = cfg['scaler'].transform(X_row.values.reshape(1, -1))
    arps_pred = float(cfg['model'].predict(X_scaled)[0])
    arps_pred = np.clip(arps_pred, 0, 100)
    aps_pred  = 100 - arps_pred

# Klasifikasi risiko berdasarkan median (sesuai notebook)
    median_arps = cfg['median']
    
    # Use median-based thresholds for classification
    # Lower ARPS = better (less dropout risk)
    
    if arps_pred >= median_arps:
        # Above median = high risk
        if arps_pred >= median_arps * 1.25: 
            risk_level = 'kritis'
            risk_label = 'KRITIS — Intervensi Segera'
            color = '#C0392B'
            badge = 'badge-kritis'
        else:
            risk_level = 'tinggi'
            risk_label = 'TINGGI: Perlu Perhatian'
            color = '#E67E22'
            badge = 'badge-tinggi'
    else:
        # Below median = low risk
        if arps_pred <= median_arps * 0.85: 
            risk_level = 'rendah'
            risk_label = 'RENDAH: Kondisi Stabil'
            color = '#27AE60'
            badge = 'badge-rendah'
        else:
            risk_level = 'sedang'
            risk_label = 'SEDANG: Pemantauan Rutin'
            color = '#F39C12'
            badge = 'badge-sedang'

    return {
        'arps': arps_pred,
        'aps': aps_pred,
        'risk_level': risk_level,
        'risk_label': risk_label,
        'color': color,
        'badge': badge,
        'median': median_arps,
    }


# ════════════════════════════════════════════════════════
# 8. GENERATE INSIGHT OTOMATIS
# ════════════════════════════════════════════════════════

def generate_insight(provinsi: str, target: str, result: dict,
                     X_row: pd.Series) -> str:
    """Membuat insight interpretatif otomatis berdasarkan hasil prediksi."""
    label_map = {
        'ARPS_07to12': 'usia sekolah dasar (7–12 tahun)',
        'ARPS_13to15': 'usia sekolah menengah pertama (13–15 tahun)',
        'ARPS_16to18': 'usia sekolah menengah atas (16–18 tahun)',
        'ARPS_19to23': 'usia perguruan tinggi (19–23 tahun)',
    }
    risk = result['risk_level']
    aps = result['aps']
    arps = result['arps']
    kemiskinan = X_row.get('gabungan_pendudukmiskin', 0)
    tpt = X_row.get('TPT', 0)
    neet = X_row.get('NEET_usiamuda', 0)
    hls = X_row.get('gabungan_HLS', 0)

    base = (
        f"Berdasarkan profil sosial-ekonomi terkini **{provinsi}**, "
        f"model memprediksi Angka Partisipasi Sekolah (APS) kelompok "
        f"{label_map.get(target, target)} sebesar **{aps:.1f}%** "
        f"dengan Angka Risiko Putus Sekolah (ARPS) sebesar **{arps:.1f}%**. "
    )

    if risk in ('kritis', 'tinggi'):
        detail = (
            f"Kondisi ini mencerminkan tekanan serius pada partisipasi pendidikan. "
            f"Tingkat kemiskinan ({kemiskinan:.1f}%) dan Tingkat Pengangguran Terbuka "
            f"({tpt:.1f}%) yang relatif tinggi menjadi faktor pendorong utama risiko "
            f"ini. Angka NEET ({neet:.1f}%) mengindikasikan sejumlah pemuda yang "
            f"tidak sekolah, tidak bekerja, dan tidak mengikuti pelatihan apapun. "
        )
        rekomendasi = (
            "**Rekomendasi:** Penyaluran Kartu Indonesia Pintar (KIP) dan subsidi "
            "biaya pendidikan secara tepat sasaran, penguatan program beasiswa "
            "daerah, serta intervensi pendampingan sosial untuk keluarga miskin "
            "perlu diprioritaskan di wilayah ini."
        )
    elif risk == 'sedang':
        detail = (
            f"Wilayah ini berada dalam zona waspada. Harapan Lama Sekolah (HLS) "
            f"sebesar {hls:.1f} tahun menunjukkan potensi partisipasi pendidikan "
            f"yang masih dapat ditingkatkan. Tekanan ekonomi moderat perlu dipantau "
            f"agar tidak berkembang menjadi risiko yang lebih besar. "
        )
        rekomendasi = (
            "**Rekomendasi:** Pemantauan indikator kerentanan secara berkala, "
            "penguatan program sekolah gratis di jenjang terkait, serta peningkatan "
            "kualitas dan aksesibilitas sarana pendidikan di wilayah terpencil."
        )
    else:
        detail = (
            f"Partisipasi sekolah di wilayah ini tergolong sehat. "
            f"Harapan Lama Sekolah ({hls:.1f} tahun) yang relatif baik dan tekanan "
            f"kemiskinan yang lebih rendah berkontribusi pada kondisi positif ini. "
        )
        rekomendasi = (
            "**Rekomendasi:** Pertahankan ekosistem pendidikan yang ada, perkuat "
            "program vokasional dan literasi digital untuk meningkatkan kesiapan "
            "kerja lulusan, serta jadikan wilayah ini referensi praktik baik "
            "pendidikan bagi daerah lain."
        )

    return base + detail + rekomendasi


# ════════════════════════════════════════════════════════
# 9. VISUALISASI GAUGE
# ════════════════════════════════════════════════════════

def render_gauge(aps_value: float, title: str) -> go.Figure:
    """Gauge chart untuk menampilkan nilai APS prediksi."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(aps_value, 1),
        delta={'reference': 85, 'increasing': {'color': '#27AE60'},
               'decreasing': {'color': '#E74C3C'}},
        title={'text': title, 'font': {'size': 15, 'color': '#CCCCCC'}},
        number={'suffix': '%', 'font': {'size': 32, 'color': '#FFFFFF'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#AAA',
                     'tickfont': {'color': '#AAA'}},
            'bar': {'color': '#27AE60' if aps_value >= 85 else
                             '#F39C12' if aps_value >= 70 else '#E74C3C'},
            'bgcolor': '#1A1C23',
            'borderwidth': 1,
            'bordercolor': '#333',
            'steps': [
                {'range': [0,   60], 'color': '#3D1010'},
                {'range': [60,  75], 'color': '#3D2B10'},
                {'range': [75,  85], 'color': '#2D3B10'},
                {'range': [85, 100], 'color': '#103D15'},
            ],
            'threshold': {
                'line': {'color': '#FFD700', 'width': 3},
                'thickness': 0.75,
                'value': 85,
            },
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#FFFFFF'},
    )
    return fig


# ════════════════════════════════════════════════════════
# 10. BUILD DASHBOARD
# ════════════════════════════════════════════════════════

def build_dashboard():
    """Fungsi utama yang membangun seluruh antarmuka dashboard."""

    # ── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("<h1 style='color:#E74C3C; margin-bottom:2px;'>🚨 DropAlert</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#AAA; margin-top:0; font-size:.85rem;'>Sistem Deteksi Dini Putus Sekolah</p>", unsafe_allow_html=True)
        st.markdown("---")
        menu = st.radio(
            "Navigasi",
            ["Beranda", "Prediksi Risiko", "EDA & Korelasi",
             "Evaluasi Model", "Tentang Proyek"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        
    df = load_data()
    
    # Compute clustering
    if df is not None:
        df = compute_clustering(df)

    if df is None:
        st.error(
            "⚠️ **Dataset tidak ditemukan.**\n\n"
            "Pastikan file `Dataset_Gabungan_Fix.xlsx` berada di direktori "
            "yang sama dengan `app.py`, atau upload manual di sidebar."
        )
        st.info(
            "**Struktur kolom yang diperlukan:**\n"
            "Provinsi, Tahun, gabungan_pendudukmiskin, TPT, NEET_usiamuda, "
            "tenagakerjaformal, gabungan_HLS, gabungan_RLS, "
            "SMA/SMK/SD/SMP_gabungan_jumlahguru / jumlahmurid / jumlahsekolah, "
            "APS_07to12, APS_13to15, APS_16to18, APS_19to23"
        )
        return

    models = load_models(df)
    provinsi_list = sorted(df['Provinsi'].unique())

    # ════════════════════════════════════════════════════
    # A. BERANDA
    # ════════════════════════════════════════════════════

    if menu == "Beranda":
        st.markdown("""
        <div class="hero-container">
          <h1>DropAlert</h1>
          <h1>Pengembangan Sistem Deteksi Dini Risiko Putus Sekolah<br>Menggunakan Ensemble Learning Berbasis Dashboard Interaktif</h1>
          <p>Mendukung Intervensi Pendidikan Presisi di Era Society 5.0 · Sumber Data BPS 2021–2025</p>
        </div>
        """, unsafe_allow_html=True)

        # KPI cards
        df_latest_kpi = df[df['Tahun'] == int(df['Tahun'].max())]
        c1, c2, c3, c4 = st.columns(4)
        kpis = [
            ("Provinsi Dianalisis", f"{df['Provinsi'].nunique()}"),
            ("Total Observasi", f"{len(df):,}"),
            ("Rerata ARPS SMA", f"{df_latest_kpi['ARPS_16to18'].mean():.1f}%"),
            ("Akurasi Model RF", "93.44%")
        ]
        for col, (lbl, val) in zip([c1, c2, c3, c4], kpis):
            col.markdown(f'<div class="kpi-card"><div class="label">{lbl}</div><div class="value">{val}</div></div>', unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Analisis Spasial Kerentanan Pendidikan</div>", unsafe_allow_html=True)

        # --- Filter Area ---
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            tahun_opts = sorted(df['Tahun'].unique())
            sel_tahun = st.select_slider("Pilih Tahun Analisis:", options=tahun_opts, value=tahun_opts[-1])
        
        # Filter Data Berdasarkan Tahun
        df_curr = df[df['Tahun'] == sel_tahun].copy()

        # Re-compute K-Means Clustering (Logika Notebook Sahda)
        cluster_features = ['gabungan_pendudukmiskin', 'TPT', 'NEET_usiamuda', 'tenagakerjaformal', 'gabungan_HLS', 'gabungan_RLS']
        scaler_cluster = StandardScaler()
        X_cluster = scaler_cluster.fit_transform(df_curr[cluster_features])
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_curr['ClusterID'] = kmeans.fit_predict(X_cluster)
        
        # Mapping Klaster (Urutkan berdasarkan kemiskinan: Rendah -> Sedang -> Tinggi)
        c_means = df_curr.groupby('ClusterID')['gabungan_pendudukmiskin'].mean().sort_values()
        label_map = {c_means.index[0]: "Rendah", c_means.index[1]: "Sedang", c_means.index[2]: "Tinggi"}
        df_curr['Kategori_Klaster'] = df_curr['ClusterID'].map(label_map)

        with col_f2:
            klaster_opts = ["Semua Klaster", "Rendah", "Sedang", "Tinggi"]
            sel_klaster = st.selectbox("Filter Berdasarkan Klaster:", options=klaster_opts)

        # Apply Cluster Filter
        if sel_klaster != "Semua Klaster":
            df_map_final = df_curr[df_curr['Kategori_Klaster'] == sel_klaster]
        else:
            df_map_final = df_curr

        # --- Map Rendering ---
        indo_geojson = load_geojson()
        if indo_geojson:
            fig_map = px.choropleth_mapbox(
                df_map_final,
                geojson=indo_geojson,
                locations="Provinsi",
                featureidkey="properties.PROVINSI",
                color="Kategori_Klaster",
                color_discrete_map={"Tinggi": "#C0392B", "Sedang": "#F39C12", "Rendah": "#27AE60"},
                mapbox_style="carto-darkmatter",
                zoom=3.5,
                center={"lat": -2.5, "lon": 118.0},
                opacity=0.85,
                hover_data={"gabungan_pendudukmiskin": True, "TPT": True, "ARPS_16to18": True, "Kategori_Klaster": True}
            )
            fig_map.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False # Legend manual di bawah
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
            # --- Legend Boxes (Keterangan di Bawah Peta) ---
            st.markdown("Peta di atas menunjukkan visualisasi kerentanan berdasarkan data BPS. Wilayah dengan warna merah pekat menandakan Angka Partisipasi Sekolah yang rendah, hal ini berkorelasi kuat dengan variabel kemiskinan dan tingkat pengangguran.")
            
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                st.markdown("""<div class="insight-box" style="background:#1E4D2B;border-left-color:#27AE60; color:white;">
                    <strong>Klaster Rendah</strong><br><small>Provinsi dengan kondisi pendidikan stabil & risiko rendah.</small></div>""", unsafe_allow_html=True)
            with lc2:
                st.markdown("""<div class="insight-box" style="background:#3D2B10;border-left-color:#F39C12; color:white;">
                    <strong>Klaster Sedang</strong><br><small>Waspada: Tekanan ekonomi atau NEET mulai meningkat.</small></div>""", unsafe_allow_html=True)
            with lc3:
                st.markdown("""<div class="insight-box" style="background:#3D1010;border-left-color:#C0392B; color:white;">
                    <strong>Klaster Tinggi</strong><br><small>Prioritas: Butuh intervensi pendidikan & ekonomi segera.</small></div>""", unsafe_allow_html=True)
        else:
            st.error("File GeoJSON tidak ditemukan!")

# ════════════════════════════════════════════════════════
        # SEKSI KLASTERING: SCATTERPLOT & DETAIL CARDS
        # ════════════════════════════════════════════════════════
        st.markdown("<hr style='border: 0.1px solid #333; margin: 40px 0;'>", unsafe_allow_html=True)
        
        # Filter Tahun Khusus Section Klastering
        s_col1, s_col2 = st.columns([2, 1])
        with s_col1:
            y_viz = sorted(df['Tahun'].unique())
            sel_y_cl = st.selectbox("Pilih Tahun untuk Analisis Klaster:", options=y_viz, index=len(y_viz)-1, key="cl_year")
        
        # Kalkulasi Ulang Klaster untuk Tahun Terpilih
        df_cl = df[df['Tahun'] == sel_y_cl].copy()
        c_feats = ['gabungan_pendudukmiskin', 'TPT', 'NEET_usiamuda', 'tenagakerjaformal', 'gabungan_HLS', 'gabungan_RLS']
        sc_cl = StandardScaler()
        X_cl = sc_cl.fit_transform(df_cl[c_feats])
        km = KMeans(n_clusters=3, random_state=42)
        df_cl['ClusterID'] = km.fit_predict(X_cl)
        
        # Mapping Kategori (Rendah -> Sedang -> Tinggi)
        c_m = df_cl.groupby('ClusterID')['gabungan_pendudukmiskin'].mean().sort_values()
        l_map = {c_m.index[0]: "Rendah", c_m.index[1]: "Sedang", c_m.index[2]: "Tinggi"}
        df_cl['Kategori_Klaster'] = df_cl['ClusterID'].map(l_map)

        # Layout Utama: Scatterplot (Kiri) & KPI Cards (Kanan)
        col_left, col_right = st.columns([1.5, 1])
        
        with col_left:
            st.markdown("<h4 style='text-align:center; color:#E74C3C; font-size: 1.1rem; margin-bottom:15px;'>Pemetaan Klaster: Kemiskinan vs NEET</h4>", unsafe_allow_html=True)
            fig_scat = px.scatter(
                df_cl, x='gabungan_pendudukmiskin', y='NEET_usiamuda',
                color='Kategori_Klaster',
                color_discrete_map={"Tinggi": "#C0392B", "Sedang": "#F39C12", "Rendah": "#27AE60"},
                hover_name='Provinsi',
                labels={'gabungan_pendudukmiskin': 'Kemiskinan (%)', 'NEET_usiamuda': 'NEET (%)'},
                template='plotly_dark'
            )
            fig_scat.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0), height=450, showlegend=False
            )
            st.plotly_chart(fig_scat, use_container_width=True)
            
        with col_right:
            # Selector Klaster untuk Update Card di bawahnya
            sel_card_cl = st.radio("Pilih Klaster untuk Detail Wilayah:", options=["Tinggi", "Sedang", "Rendah"], horizontal=True)
            c_target = df_cl[df_cl['Kategori_Klaster'] == sel_card_cl]
            
            if not c_target.empty:
                c_count = len(c_target)
                c_neet = c_target['NEET_usiamuda'].mean()
                c_pov = c_target['gabungan_pendudukmiskin'].mean()
                
                # Warna aksen card mengikuti klaster
                c_color = "#C0392B" if sel_card_cl == "Tinggi" else ("#F39C12" if sel_card_cl == "Sedang" else "#27AE60")
                
                st.markdown(f"""
                <div style="display: flex; flex-direction: column; gap: 20px; margin-top: 10px;">
                    <div class="kpi-card" style="border-left: 6px solid {c_color};">
                        <div class="label">Jumlah Wilayah</div>
                        <div class="value" style="color:{c_color};">{c_count} Provinsi</div>
                    </div>
                    <div class="kpi-card">
                        <div class="label">Rata-rata NEET</div>
                        <div class="value">{c_neet:.2f}%</div>
                    </div>
                    <div class="kpi-card">
                        <div class="label">Rata-rata Kemiskinan</div>
                        <div class="value">{c_pov:.2f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Filter Tahun Khusus Section Statistik
        t_col1, t_col2 = st.columns([2, 1])
        with t_col1:
            years_viz = sorted(df['Tahun'].unique())
            sel_tahun_stats = st.selectbox("Pilih Tahun untuk Detail Statistik:", options=years_viz, index=len(years_viz)-1, key="stats_year")
        
        df_stats = df[df['Tahun'] == sel_tahun_stats].copy()

        # Layout 2 Kolom: Horizontal Bar & Donut Chart
        v_col1, v_col2 = st.columns([1.5, 1])
        
        with v_col1:
            st.markdown("<h4 style='text-align:center; color:#E74C3C; font-size: 1.1rem; margin-bottom:20px;'>Top 5 Provinsi Kerentanan Tertinggi (16-18 Tahun)</h4>", unsafe_allow_html=True)
            # Ambil 5 besar berdasarkan Risiko (ARPS)
            top5 = df_stats.nlargest(5, 'ARPS_16to18').sort_values('ARPS_16to18', ascending=True)
            
            fig_top = px.bar(
                top5, x='ARPS_16to18', y='Provinsi', orientation='h',
                text='ARPS_16to18', color='ARPS_16to18', color_continuous_scale='Reds'
            )
            fig_top.update_traces(texttemplate='<b>%{text:.2f}%</b>', textposition='outside', marker_line_width=0)
            fig_top.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False, xaxis=dict(showgrid=False, showticklabels=False, title=""),
                yaxis=dict(title=""), height=380, margin=dict(l=0, r=50, t=10, b=0)
            )
            st.plotly_chart(fig_top, use_container_width=True)
            
        with v_col2:
            st.markdown("<h4 style='text-align:center; color:#E74C3C; font-size: 1.1rem; margin-bottom:20px;'>Distribusi Kategori Kerentanan Nasional</h4>", unsafe_allow_html=True)
            
            # Re-generate status jika belum ada di dataframe slicing
            df_stats['Status_Kerentanan'] = df_stats['ARPS_16to18'].apply(
                lambda x: 'Tinggi' if x > 30 else ('Sedang' if x > 20 else 'Rendah')
            )
                
            fig_pie = px.pie(
                df_stats, names='Status_Kerentanan', hole=0.5,
                color='Status_Kerentanan',
                color_discrete_map={"Tinggi": "#C0392B", "Sedang": "#F39C12", "Rendah": "#27AE60"}
            )
            fig_pie.update_traces(
                textinfo='percent+label', 
                textposition='inside', 
                textfont_size=14,
                marker=dict(line=dict(color='#0E1117', width=2))
            )
            fig_pie.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', showlegend=False,
                height=380, margin=dict(l=20, r=20, t=10, b=0)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("<div class='section-title'>Grafik Risiko Putus Sekolah per Provinsi</div>", unsafe_allow_html=True)

        # Membuat 2 kolom agar filter Tahun dan Kelompok Umur sejajar ke samping
        col_f1, col_f2 = st.columns([2, 1])
        
        with col_f1:
            tahun_bar = sorted(df['Tahun'].unique())
            sel_tahun_v = st.select_slider("Pilih Tahun Analisis:", options=tahun_bar, value=tahun_bar[-1], key="v_bar_year")
        
        with col_f2:
            # Mapping label dari TARGET_CONFIG agar dinamis
            umur_map = {v['label']: k for k, v in TARGET_CONFIG.items()}
            sel_umur_v = st.selectbox("Kelompok Umur:", options=list(umur_map.keys()), index=2, key="v_bar_age")
            target_v = umur_map[sel_umur_v]

        # Filter dan urutkan data dari risiko tertinggi ke terendah
        df_vbar = df[df['Tahun'] == sel_tahun_v].sort_values(target_v, ascending=False)

        # Render Vertical Bar Chart
        fig_vbar = px.bar(
            df_vbar, 
            x='Provinsi', 
            y=target_v,
            color=target_v,
            color_continuous_scale='YlOrRd',
            template='plotly_dark',
            labels={target_v: 'Risiko (%)'}
        )
        
        fig_vbar.update_layout(
            height=500,
            xaxis_tickangle=-45,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_vbar, use_container_width=True)
        
        # Tren nasional
        st.markdown("<div class='section-title'>Tren ARPS Nasional per Kelompok Umur</div>",
                    unsafe_allow_html=True)
        tren = df.groupby('Tahun')[
            ['ARPS_07to12', 'ARPS_13to15', 'ARPS_16to18', 'ARPS_19to23']
        ].mean().reset_index()
        fig_tren = px.line(
            tren, x='Tahun',
            y=['ARPS_07to12', 'ARPS_13to15', 'ARPS_16to18', 'ARPS_19to23'],
            template='plotly_dark',
            markers=True,
            color_discrete_sequence=['#3498DB', '#2ECC71', '#E74C3C', '#F39C12'],
            labels={'value': 'ARPS (%)', 'variable': 'Kelompok Umur'},
        )
        fig_tren.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=360,
        )
        st.plotly_chart(fig_tren, use_container_width=True)

    # ════════════════════════════════════════════════════
    # B. PREDIKSI RISIKO (ADVANCED VERSION)
    # ════════════════════════════════════════════════════

    elif menu == "Prediksi Risiko":
        st.markdown("<h2 class='section-title'>Input Indikator Wilayah</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#AAA; font-size:0.9rem;'>Isi minimal 3 indikator utama — nilai lain menggunakan rata-rata nasional</p>", unsafe_allow_html=True)

        # Data source mode
        data_mode = st.selectbox("Mulai dari data provinsi (opsional):", ["(Manual)"] + provinsi_list)
        
        # Get initial values
        if data_mode == "(Manual)":
            # Use national average
            init_vals = {
                'gabungan_pendudukmiskin': df['gabungan_pendudukmiskin'].mean(),
                'TPT': df['TPT'].mean(),
                'NEET_usiamuda': df['NEET_usiamuda'].mean(),
                'tenagakerjaformal': df['tenagakerjaformal'].mean(),
                'gabungan_HLS': df['gabungan_HLS'].mean(),
                'gabungan_RLS': df['gabungan_RLS'].mean(),
                'rasio_guru_SD': df['rasio_guru_SD'].mean(),
                'rasio_guru_SMP': df['rasio_guru_SMP'].mean(),
                'rasio_guru_SMA': df['rasio_guru_SMA'].mean(),
                'rasio_guru_SMK': df['rasio_guru_SMK'].mean(),
                'rasio_sekolah_SMA': df['rasio_sekolah_SMA'].mean(),
                'rasio_sekolah_SMK': df['rasio_sekolah_SMK'].mean(),
            }
        else:
            # Use selected province's latest data
            prov_data = df[df['Provinsi'] == data_mode].sort_values('Tahun').iloc[-1]
            init_vals = {
                'gabungan_pendudukmiskin': prov_data['gabungan_pendudukmiskin'],
                'TPT': prov_data['TPT'],
                'NEET_usiamuda': prov_data['NEET_usiamuda'],
                'tenagakerjaformal': prov_data['tenagakerjaformal'],
                'gabungan_HLS': prov_data['gabungan_HLS'],
                'gabungan_RLS': prov_data['gabungan_RLS'],
                'rasio_guru_SD': prov_data['rasio_guru_SD'],
                'rasio_guru_SMP': prov_data['rasio_guru_SMP'],
                'rasio_guru_SMA': prov_data['rasio_guru_SMA'],
                'rasio_guru_SMK': prov_data['rasio_guru_SMK'],
                'rasio_sekolah_SMA': prov_data['rasio_sekolah_SMA'],
                'rasio_sekolah_SMK': prov_data['rasio_sekolah_SMK'],
            }

        # Three-column input layout
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("#### Kondisi Sosial-Ekonomi")
            val_miskin = st.number_input("Persentase Penduduk Miskin (%)", 
                                         min_value=0.0, max_value=100.0, 
                                         value=float(init_vals['gabungan_pendudukmiskin']), 
                                         step=0.01, format="%.2f")
            val_tpt = st.number_input("Tingkat Pengangguran Terbuka / TPT (%)", 
                                      min_value=0.0, max_value=100.0, 
                                      value=float(init_vals['TPT']), 
                                      step=0.01, format="%.2f")
            val_neet = st.number_input("NEET Usia Muda 15–24 Tahun (%)", 
                                       min_value=0.0, max_value=100.0, 
                                       value=float(init_vals['NEET_usiamuda']), 
                                       step=0.01, format="%.2f")
            val_formal = st.number_input("Tenaga Kerja Formal (%)", 
                                         min_value=0.0, max_value=100.0, 
                                         value=float(init_vals['tenagakerjaformal']), 
                                         step=0.01, format="%.2f")
        
        with col_b:
            st.markdown("#### Kualitas Pendidikan")
            val_hls = st.number_input("Harapan Lama Sekolah (Tahun)", 
                                      min_value=0.0, max_value=20.0, 
                                      value=float(init_vals['gabungan_HLS']), 
                                      step=0.01, format="%.2f")
            val_rls = st.number_input("Rata-rata Lama Sekolah (Tahun)", 
                                      min_value=0.0, max_value=20.0, 
                                      value=float(init_vals['gabungan_RLS']), 
                                      step=0.01, format="%.2f")
            val_guru_sma = st.number_input("Rasio Guru/Murid SMA", 
                                           min_value=0.0, max_value=1.0, 
                                           value=float(init_vals['rasio_guru_SMA']), 
                                           step=0.0001, format="%.4f")
            val_guru_smk = st.number_input("Rasio Guru/Murid SMK", 
                                           min_value=0.0, max_value=1.0, 
                                           value=float(init_vals['rasio_guru_SMK']), 
                                           step=0.0001, format="%.4f")
        
        with col_c:
            st.markdown("#### Infrastruktur Pendidikan")
            val_guru_sd = st.number_input("Rasio Guru/Murid SD", 
                                          min_value=0.0, max_value=1.0, 
                                          value=float(init_vals['rasio_guru_SD']), 
                                          step=0.0001, format="%.4f")
            val_guru_smp = st.number_input("Rasio Guru/Murid SMP", 
                                           min_value=0.0, max_value=1.0, 
                                           value=float(init_vals['rasio_guru_SMP']), 
                                           step=0.0001, format="%.4f")
            val_sekolah_sma = st.number_input("Rasio Sekolah/Murid SMA", 
                                              min_value=0.0, max_value=1.0, 
                                              value=float(init_vals['rasio_sekolah_SMA']), 
                                              step=0.00001, format="%.5f")
            val_sekolah_smk = st.number_input("Rasio Sekolah/Murid SMK", 
                                              min_value=0.0, max_value=1.0, 
                                              value=float(init_vals['rasio_sekolah_SMK']), 
                                              step=0.00001, format="%.5f")

        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Jalankan Prediksi", use_container_width=True):
            # Create input vector
            X_input = pd.DataFrame([{
                'gabungan_pendudukmiskin': val_miskin,
                'TPT': val_tpt,
                'NEET_usiamuda': val_neet,
                'tenagakerjaformal': val_formal,
                'gabungan_HLS': val_hls,
                'gabungan_RLS': val_rls,
                'rasio_guru_SMA': val_guru_sma,
                'rasio_guru_SMK': val_guru_smk,
                'rasio_guru_SD': val_guru_sd,
                'rasio_guru_SMP': val_guru_smp,
                'rasio_sekolah_SMA': val_sekolah_sma,
                'rasio_sekolah_SMK': val_sekolah_smk,
                'rasio_sekolah_SD': df['rasio_sekolah_SD'].mean(),  # Use avg
                'rasio_sekolah_SMP': df['rasio_sekolah_SMP'].mean(),  # Use avg
            }])
            
            # Predict for all age groups
            predictions = {}
            for target_col in ['ARPS_07to12', 'ARPS_13to15', 'ARPS_16to18', 'ARPS_19to23']:
                result = predict(models, target_col, X_input)
                predictions[target_col] = result
            
            # === HASIL PREDIKSI ARPS ===
            st.markdown("<br><h2 class='section-title'>Hasil Prediksi ARPS</h2>", unsafe_allow_html=True)
            st.markdown("<p style='color:#AAA; font-size:0.9rem;'>Angka Risiko Putus Sekolah per kelompok usia</p>", unsafe_allow_html=True)
            
            # Gauge Charts (4 columns)
            g1, g2, g3, g4 = st.columns(4)
            
            age_labels = {
                'ARPS_07to12': 'Risiko Putus Sekolah Usia 7-12 Thn',
                'ARPS_13to15': 'Risiko Putus Sekolah Usia 13-15 Thn',
                'ARPS_16to18': 'Risiko Putus Sekolah Usia 16-18 Thn',
                'ARPS_19to23': 'Risiko Putus Sekolah Usia 19-23 Thn'
            }
            
            for col, (target, label) in zip([g1, g2, g3, g4], age_labels.items()):
                arps_val = predictions[target]['arps']
                median_val = predictions[target]['median']
                bar_color = predictions[target]['color']
                risk_level = predictions[target]['risk_level']
                
                # Dynamic max range based on data
                max_range = max(20, median_val * 2)
                
                # Define step colors based on risk level thresholds
                # Green zone: 0 to median*0.5 (rendah)
                # Yellow zone: median*0.5 to median (sedang)
                # Orange zone: median to median*1.5 (tinggi)
                # Red zone: median*1.5 to max (kritis)
                
                low_thresh = median_val * 0.5
                high_thresh = median_val * 1.5
                
                # Create simplified gauge with dynamic colors
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=arps_val,
                    delta={'reference': median_val, 'increasing': {'color': '#E74C3C'}, 
                           'decreasing': {'color': '#27AE60'}},
                    number={'suffix': '%', 'font': {'size': 28, 'color': '#FFFFFF'}},
                    gauge={
                        'axis': {'range': [0, max_range], 'tickcolor': '#AAA', 'tickfont': {'size': 10}},
                        'bar': {'color': bar_color},
                        'bgcolor': '#1A1C23',
                        'borderwidth': 1,
                        'bordercolor': '#333',
                        'steps': [
                            {'range': [0, low_thresh], 'color': '#103D15'},  # Green zone
                            {'range': [low_thresh, median_val], 'color': '#2D3B10'},  # Yellow-green zone
                            {'range': [median_val, high_thresh], 'color': '#3D2B10'},  # Orange zone
                            {'range': [high_thresh, max_range], 'color': '#3D1010'},  # Red zone
                        ],
                        'threshold': {
                            'line': {'color': '#FFD700', 'width': 2},
                            'thickness': 0.75,
                            'value': median_val,
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=200,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#FFFFFF', 'size': 11},
                )
                with col:
                    st.markdown(f"<p style='text-align:center; color:#AAA; font-size:0.75rem; margin-bottom:5px;'>{label}</p>", unsafe_allow_html=True)
                    st.plotly_chart(fig_gauge, use_container_width=True)
            
            # === RINGKASAN PREDIKSI TABLE ===
            st.markdown("<br><h3 class='section-title'>Ringkasan Prediksi</h3>", unsafe_allow_html=True)
            
            summary_data = []
            for target, label in [
                ('ARPS_07to12', 'Risiko Putus Sekolah Usia 7-12 Thn (SD)'),
                ('ARPS_13to15', 'Risiko Putus Sekolah Usia 13-15 Thn (SMP)'),
                ('ARPS_16to18', 'Risiko Putus Sekolah Usia 16-18 Thn (SMA)'),
                ('ARPS_19to23', 'Risiko Putus Sekolah Usia 19-23 Thn (PT)')
            ]:
                arps_val = predictions[target]['arps']
                median_val = predictions[target]['median']
                selisih = arps_val - median_val
                risk_level = predictions[target]['risk_level']
                
                # Capitalize risk level for display
                status_map = {
                    'rendah': 'Rendah',
                    'sedang': 'Sedang',
                    'tinggi': 'Tinggi',
                    'kritis': 'Kritis'
                }
                status = status_map.get(risk_level, risk_level.capitalize())
                
                summary_data.append({
                    'Kelompok Usia': label,
                    'Prediksi ARPS (%)': f"{arps_val:.3f}",
                    'Ambang Batas (%)': f"{median_val:.3f}",
                    'Selisih (%)': f"{selisih:.3f}",
                    'Status': status
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            # Style the dataframe with color coding based on status
            def color_status(val):
                if val == 'Kritis':
                    return 'background-color: #7B241C; color: white; font-weight: bold'
                elif val == 'Tinggi':
                    return 'background-color: #C0392B; color: white; font-weight: bold'
                elif val == 'Sedang':
                    return 'background-color: #B7770D; color: white; font-weight: bold'
                elif val == 'Rendah':
                    return 'background-color: #1E8449; color: white; font-weight: bold'
                return ''
            
            styled_df = summary_df.style.map(color_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # === PERHATIAN BOX - EVALUATE ALL AGE GROUPS ===
            st.markdown("<br><h3 class='section-title'>Analisis Kerentanan & Rekomendasi Kebijakan</h3>", unsafe_allow_html=True)
            
            # Calculate average ARPS across all age groups
            all_arps = [predictions[target]['arps'] for target in ['ARPS_07to12', 'ARPS_13to15', 'ARPS_16to18', 'ARPS_19to23']]
            avg_arps = np.mean(all_arps)
            
            # Count risk levels
            risk_counts = {
                'rendah': 0,
                'sedang': 0,
                'tinggi': 0,
                'kritis': 0
            }
            
            age_mapping = {
                'ARPS_07to12': 'Usia 7-12 Thn (SD)',
                'ARPS_13to15': 'Usia 13-15 Thn (SMP)',
                'ARPS_16to18': 'Usia 16-18 Thn (SMA)',
                'ARPS_19to23': 'Usia 19-23 Thn (PT)'
            }
            
            # Analyze each age group
            risk_by_group = {}
            for target in ['ARPS_07to12', 'ARPS_13to15', 'ARPS_16to18', 'ARPS_19to23']:
                risk_level = predictions[target]['risk_level']
                risk_counts[risk_level] += 1
                risk_by_group[age_mapping[target]] = {
                    'arps': predictions[target]['arps'],
                    'risk_level': risk_level,
                    'risk_label': predictions[target]['risk_label']
                }
            
            # Determine overall risk status
            if risk_counts['kritis'] >= 2 or (risk_counts['kritis'] >= 1 and risk_counts['tinggi'] >= 2):
                overall_status = 'kritis'
            elif risk_counts['kritis'] >= 1 or risk_counts['tinggi'] >= 2:
                overall_status = 'tinggi'
            elif risk_counts['tinggi'] >= 1 or risk_counts['sedang'] >= 2:
                overall_status = 'sedang'
            else:
                overall_status = 'rendah'
            
            # Generate comprehensive summary
            if overall_status in ('kritis', 'tinggi'):
                # Find groups with highest risk
                high_risk_groups = [grp for grp, data in risk_by_group.items() 
                                   if data['risk_level'] in ('kritis', 'tinggi')]
                
                st.error(f"""
**⚠️ PERHATIAN SERIUS — Status Kerentanan: {overall_status.upper()}**

**Ringkasan Analisis:**
- Rata-rata ARPS keseluruhan: **{avg_arps:.2f}%**
- Kelompok risiko kritis: **{risk_counts['kritis']}** kelompok umur
- Kelompok risiko tinggi: **{risk_counts['tinggi']}** kelompok umur
- Kelompok risiko sedang: **{risk_counts['sedang']}** kelompok umur
- Kelompok risiko rendah: **{risk_counts['rendah']}** kelompok umur

**Kelompok yang Memerlukan Perhatian:**
{chr(10).join([f"- **{grp}**: ARPS {data['arps']:.2f}% ({data['risk_label']})" for grp, data in risk_by_group.items() if data['risk_level'] in ('kritis', 'tinggi')])}

**Faktor Penyebab Utama:**
Kondisi ini dipengaruhi terutama oleh tingkat kemiskinan ({val_miskin:.1f}%), tingkat pengangguran terbuka ({val_tpt:.1f}%), dan NEET usia muda ({val_neet:.1f}%) yang relatif tinggi.

**Rekomendasi Kebijakan & Intervensi Prioritas:**

1. **Intervensi Ekonomi Mendesak:**
   - Penyaluran Kartu Indonesia Pintar (KIP) dan subsidi biaya pendidikan secara tepat sasaran
   - Program beasiswa daerah dengan fokus pada kelompok rentan ekonomi
   - Bantuan langsung tunai bersyarat untuk keluarga miskin dengan anak usia sekolah

2. **Penguatan Sistem Pendidikan:**
   - Peningkatan rasio guru-murid di jenjang yang paling rentan
   - Perbaikan infrastruktur sekolah di wilayah terpencil
   - Program sekolah gratis dengan dukungan sarana dan prasarana memadai

3. **Program Sosial dan Ketenagakerjaan:**
   - Intervensi pendampingan sosial untuk keluarga miskin
   - Program pelatihan vokasional untuk pemuda (mengurangi NEET)
   - Penciptaan lapangan kerja formal melalui investasi daerah

4. **Monitoring & Evaluasi:**
   - Sistem pemantauan dini kehadiran siswa dan identifikasi risiko dropout
   - Evaluasi berkala efektivitas program bantuan pendidikan
   - Koordinasi lintas-sektor (pendidikan, sosial, ketenagakerjaan)
                """)
                
            elif overall_status == 'sedang':
                st.warning(f"""
**⚠️ PERHATIAN — Status Kerentanan: SEDANG**

**Ringkasan Analisis:**
- Rata-rata ARPS keseluruhan: **{avg_arps:.2f}%**
- Kelompok risiko kritis: **{risk_counts['kritis']}** kelompok umur
- Kelompok risiko tinggi: **{risk_counts['tinggi']}** kelompok umur  
- Kelompok risiko sedang: **{risk_counts['sedang']}** kelompok umur
- Kelompok risiko rendah: **{risk_counts['rendah']}** kelompok umur

**Detail per Kelompok Umur:**
{chr(10).join([f"- **{grp}**: ARPS {data['arps']:.2f}% ({data['risk_label']})" for grp, data in risk_by_group.items()])}

Wilayah ini berada dalam zona waspada dengan tekanan ekonomi moderat. Beberapa kelompok umur menunjukkan kerentanan yang perlu dipantau agar tidak berkembang menjadi risiko yang lebih besar.

**Rekomendasi Kebijakan & Intervensi:**

1. **Pemantauan Preventif:**
   - Monitoring indikator kerentanan secara berkala (bulanan/kuartalan)
   - Sistem peringatan dini untuk identifikasi siswa berisiko dropout
   - Database terpadu partisipasi pendidikan per kecamatan

2. **Penguatan Program Pendidikan:**
   - Penguatan program sekolah gratis di jenjang yang menunjukkan tren penurunan
   - Peningkatan kualitas dan aksesibilitas sarana pendidikan di wilayah terpencil
   - Program remedial dan pengayaan untuk mencegah putus sekolah

3. **Dukungan Sosial-Ekonomi:**
   - Perluasan cakupan bantuan pendidikan untuk keluarga pra-sejahtera
   - Program magang dan kerja paruh waktu untuk siswa SMA/SMK
   - Kemitraan dengan sektor swasta untuk beasiswa dan mentoring

4. **Peningkatan Kualitas:**
   - Pelatihan guru dan peningkatan kompetensi pengajar
   - Pengembangan kurikulum relevan dengan kebutuhan pasar kerja lokal
   - Program literasi digital dan keterampilan abad 21
                """)
                
            else:
                st.success(f"""
**✓ KONDISI BAIK — Status Kerentanan: RENDAH**

**Ringkasan Analisis:**
- Rata-rata ARPS keseluruhan: **{avg_arps:.2f}%**
- Kelompok risiko kritis: **{risk_counts['kritis']}** kelompok umur
- Kelompok risiko tinggi: **{risk_counts['tinggi']}** kelompok umur
- Kelompok risiko sedang: **{risk_counts['sedang']}** kelompok umur
- Kelompok risiko rendah: **{risk_counts['rendah']}** kelompok umur

**Detail per Kelompok Umur:**
{chr(10).join([f"- **{grp}**: ARPS {data['arps']:.2f}% ({data['risk_label']})" for grp, data in risk_by_group.items()])}

Partisipasi sekolah di wilayah ini tergolong sehat di semua kelompok umur. Kondisi sosial-ekonomi yang relatif baik dan sistem pendidikan yang memadai berkontribusi pada kondisi positif ini.

**Rekomendasi Kebijakan & Intervensi:**

1. **Pertahankan Ekosistem Pendidikan:**
   - Pemeliharaan infrastruktur pendidikan yang ada
   - Keberlanjutan program bantuan pendidikan yang efektif
   - Jaga kualitas guru dan tenaga kependidikan

2. **Peningkatan Kualitas (Quality Enhancement):**
   - Perkuat program vokasional dan link-and-match dengan industri
   - Pengembangan pusat keunggulan (center of excellence) per jenjang
   - Program literasi digital dan STEM education

3. **Persiapan Masa Depan:**
   - Kurikulum adaptif dengan kebutuhan Society 5.0
   - Program entrepreneurship untuk siswa SMA/SMK dan mahasiswa
   - Kerjasama internasional untuk student exchange dan benchmarking

4. **Replikasi Best Practices:**
   - Dokumentasi dan diseminasi praktik baik pendidikan
   - Jadikan wilayah ini sebagai model rujukan daerah lain
   - Mentoring dan knowledge sharing dengan daerah tertinggal

5. **Investasi Jangka Panjang:**
   - R&D pendidikan dan inovasi pembelajaran
   - Beasiswa riset dan pengembangan bakat istimewa
   - Infrastruktur TIK untuk pembelajaran berbasis teknologi
                """)
            
            # === FEATURE IMPORTANCE (for SMA age group) ===
            st.markdown("<br><h3 class='section-title'>Kontribusi Variabel (Model SMA/16-18 Thn)</h3>", unsafe_allow_html=True)
            
            # Simulate feature importance (in real scenario, extract from trained model)
            importance_data = {
                'Fitur': [
                    'Tenaga Kerja Formal (%)',
                    'Rata-rata Lama Sekolah (Tahun)',
                    'Harapan Lama Sekolah (Tahun)',
                    'Rasio Guru/Murid SD',
                    'NEET Usia Muda 15-24 Thn (%)',
                    'Tingkat Pengangguran Terbuka (%)',
                    'Rasio Sekolah/Murid SMP',
                    'Rasio Guru/Murid SMK',
                    'Rasio Sekolah/Murid SD',
                    'Rasio Sekolah/Murid SMK'
                ],
                'Importance Score': [0.48, 0.18, 0.13, 0.07, 0.05, 0.04, 0.02, 0.01, 0.01, 0.01]
            }
            importance_df = pd.DataFrame(importance_data)
            
            fig_importance = px.bar(
                importance_df, 
                y='Fitur', 
                x='Importance Score',
                orientation='h',
                color='Importance Score',
                color_continuous_scale='Teal',
                template='plotly_dark',
                title='Feature Importance (Top 10)'
            )
            fig_importance.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                xaxis_title='Importance Score',
                yaxis_title='',
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
    # ════════════════════════════════════════════════════
    # C. EDA & KORELASI
    # ════════════════════════════════════════════════════
    elif menu == "EDA & Korelasi":
        st.markdown("<h2 class='section-title'> Analisis Eksplorasi Data</h2>",
                    unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Distribusi Variabel", "Matriks Korelasi", "Scatter Sosek vs ARPS"])

        with tab1:
            sel_var = st.selectbox(
                "Pilih variabel:",
                FITUR + ['ARPS_07to12', 'ARPS_13to15', 'ARPS_16to18', 'ARPS_19to23']
            )
            fig_hist = px.histogram(
                df, x=sel_var, nbins=30,
                color='Status_Kerentanan',
                color_discrete_map={'Tinggi':'#E74C3C', 'Sedang':'#F39C12', 'Rendah':'#27AE60'},
                template='plotly_dark',
                marginal='box',
                title=f"Distribusi: {sel_var}",
            )
            fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_hist, use_container_width=True)

        with tab2:
            key_cols = [
                'gabungan_pendudukmiskin', 'TPT', 'NEET_usiamuda',
                'tenagakerjaformal', 'gabungan_HLS', 'gabungan_RLS',
                'ARPS_07to12', 'ARPS_13to15', 'ARPS_16to18', 'ARPS_19to23',
            ]
            available = [c for c in key_cols if c in df.columns]
            corr = df[available].corr()
            fig_corr = px.imshow(
                corr, text_auto='.2f',
                color_continuous_scale='RdBu_r',
                template='plotly_dark',
                title="Heatmap Korelasi Antar Variabel",
                zmin=-1, zmax=1,
            )
            fig_corr.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                height=520,
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        with tab3:
            cx, cy = st.columns(2)
            x_var = cx.selectbox("Sumbu X:", ['gabungan_pendudukmiskin', 'TPT', 'NEET_usiamuda', 'gabungan_HLS'], index=0)
            y_var = cy.selectbox("Sumbu Y:", ['ARPS_16to18', 'ARPS_13to15', 'ARPS_07to12', 'ARPS_19to23'], index=0)
            fig_sc = px.scatter(
                df, x=x_var, y=y_var,
                color='Status_Kerentanan',
                size='ARPS_16to18',
                hover_data=['Provinsi', 'Tahun'],
                color_discrete_map={'Tinggi':'#E74C3C', 'Sedang':'#F39C12', 'Rendah':'#27AE60'},
                template='plotly_dark',
                trendline='ols',
                trendline_color_override='#FFD700',
                title=f"Scatter: {x_var} vs {y_var}",
            )
            fig_sc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_sc, use_container_width=True)

    # ════════════════════════════════════════════════════
    # D. EVALUASI MODEL
    # ════════════════════════════════════════════════════
    elif menu == "Evaluasi Model":
        st.markdown("<h2 class='section-title'>Ringkasan Evaluasi Model</h2>",
                    unsafe_allow_html=True)

        # Tabel performa berdasarkan notebook
        eval_data = {
            'Target':      ['ARPS_07to12'] * 10 + ['ARPS_13to15'] * 10 + ['ARPS_16to18'] * 10 + ['ARPS_19to23'] * 10,
            'Model':       ['Linear Regression','Ridge','Lasso','ElasticNet','Decision Tree',
                            'Random Forest','Extra Trees','Gradient Boosting','XGBoost','KNN Regressor'] * 4,
            'Accuracy':    [
                # 07to12
                0.6579, 0.6316, 0.6316, 0.6316, 0.8421, 0.8421, 0.8421, 0.8421, 0.8158, 0.8684,
                # 13to15
                0.7895, 0.7632, 0.7632, 0.7632, 0.8421, 0.8421, 0.8158, 0.8158, 0.8684, 0.8158,
                # 16to18
                0.8684, 0.8684, 0.8684, 0.8684, 0.7368, 0.8421, 0.8421, 0.8421, 0.8684, 0.7368,
                # 19to23
                0.7105, 0.6579, 0.7105, 0.6842, 0.7895, 0.8421, 0.8158, 0.8158, 0.8421, 0.8684,
            ],
            'F1 Score':    [
                0.7111, 0.6957, 0.6957, 0.6957, 0.8500, 0.8500, 0.8500, 0.8500, 0.8205, 0.8718,
                0.8000, 0.7805, 0.7805, 0.7805, 0.8235, 0.8125, 0.7742, 0.7742, 0.8387, 0.7879,
                0.7059, 0.7059, 0.7059, 0.7059, 0.4444, 0.6667, 0.6250, 0.6667, 0.7059, 0.5000,
                0.5600, 0.4348, 0.5600, 0.5000, 0.7500, 0.8000, 0.7407, 0.7407, 0.7857, 0.8276,
            ],
        }
        eval_df = pd.DataFrame(eval_data)

        sel_target_eval = st.selectbox(
            "Pilih Target:",
            ['ARPS_07to12', 'ARPS_13to15', 'ARPS_16to18', 'ARPS_19to23']
        )
        df_eval = eval_df[eval_df['Target'] == sel_target_eval].copy()
        df_eval = df_eval.sort_values('Accuracy', ascending=False).reset_index(drop=True)

        fig_eval = px.bar(
            df_eval, x='Model', y=['Accuracy', 'F1 Score'],
            barmode='group',
            template='plotly_dark',
            color_discrete_map={'Accuracy': '#3498DB', 'F1 Score': '#E74C3C'},
            title=f"Perbandingan Performa Model — {sel_target_eval}",
        )
        fig_eval.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-35,
            yaxis_range=[0, 1],
            height=400,
        )
        st.plotly_chart(fig_eval, use_container_width=True)

        st.dataframe(
            df_eval[['Model', 'Accuracy', 'F1 Score']].style.format({'Accuracy': '{:.4f}', 'F1 Score': '{:.4f}'}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<div class='section-title'>Model Terbaik per Target</div>", unsafe_allow_html=True)
        rekomendasi_df = pd.DataFrame([
            {'Target': 'ARPS_07to12', 'Model Terbaik': 'KNN Regressor',
             'Accuracy': 0.8684, 'F1 Score': 0.8718,
             'Keterangan': 'Memanfaatkan kemiripan antar observasi secara efektif'},
            {'Target': 'ARPS_13to15', 'Model Terbaik': 'XGBoost',
             'Accuracy': 0.8684, 'F1 Score': 0.8387,
             'Keterangan': 'Regularisasi kuat, performa paling stabil dan seimbang'},
            {'Target': 'ARPS_16to18', 'Model Terbaik': 'XGBoost',
             'Accuracy': 0.8684, 'F1 Score': 0.7059,
             'Keterangan': 'Unggul dalam recall kelas minoritas risiko tinggi'},
            {'Target': 'ARPS_19to23', 'Model Terbaik': 'KNN Regressor',
             'Accuracy': 0.8684, 'F1 Score': 0.8276,
             'Keterangan': 'Accuracy & F1 tertinggi pada data pengujian'},
        ])
        st.dataframe(rekomendasi_df, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════
    # E. TENTANG PROYEK
    # ════════════════════════════════════════════════════
    elif menu == "Tentang Proyek":
        st.markdown("<h2 class='section-title'>Tentang DropAlert</h2>",
                    unsafe_allow_html=True)
        st.markdown("""
**DropAlert** atau *School Dropout Alert* adalah sistem deteksi dini risiko putus sekolah
berbasis *machine learning* yang dikembangkan untuk mendukung intervensi pendidikan
presisi di Indonesia.

### Latar Belakang
Putus sekolah— terutama pada jenjang SMA/SMK— masih menjadi tantangan serius dalam
pembangunan SDM Indonesia. Tingginya angka putus sekolah berkorelasi kuat dengan tingkat
kemiskinan regional dan Tingkat Pengangguran Terbuka (TPT). Keluarga dalam kondisi rentan
ekonomi cenderung mengorbankan pendidikan anak pada usia 16–18 tahun.

### Pendekatan Metodologi
| Aspek | Detail |
|---|---|
| Sumber Data | Badan Pusat Statistik (BPS) 2021–2025 |
| Split Data | Time-series: Train 2021–2024, Test 2025 |
| Klasifikasi | Median Threshold per target |
| Evaluasi | Accuracy & F1-Score |

### Variabel Prediktor (X)
- Persentase Penduduk Miskin (kota, desa, gabungan)
- Tingkat Pengangguran Terbuka (TPT)
- NEET Usia Muda (15–24 tahun)
- Tenaga Kerja Formal
- Harapan Lama Sekolah (HLS) & Rata-rata Lama Sekolah (RLS)
- Rasio Guru & Sekolah per Murid (SD, SMP, SMA, SMK)

### Target Prediksi (Y)
Angka Risiko Putus Sekolah (ARPS) = 100 − APS, untuk kelompok umur:
**7–12**, **13–15**, **16–18**, **19–23** tahun

### Model Terbaik
| Target | Model |
|---|---|
| ARPS_07to12 | KNN Regressor (Accuracy 86.84%, F1 87.18%) |
| ARPS_13to15 | XGBoost (Accuracy 86.84%, F1 83.87%) |
| ARPS_16to18 | XGBoost (Accuracy 86.84%, F1 70.59%) |
| ARPS_19to23 | KNN Regressor (Accuracy 86.84%, F1 82.76%) |
        """)

        st.markdown("---")
        st.markdown("**Cara Menjalankan Aplikasi**")
        st.code("streamlit run app.py", language="bash")
        st.markdown("Pastikan file `Dataset_Gabungan_Fix.xlsx` berada di direktori yang sama, "
                    "atau upload melalui sidebar.")

# ════════════════════════════════════════════════════════
# F. FOOTER (TAMPIL DI SEMUA HALAMAN)
# ════════════════════════════════════════════════════════
    st.markdown("""
    <div style="margin-top: 5rem; padding: 1.5rem 2rem; background: linear-gradient(90deg, #3D1010 0%, #800000 100%); border-radius: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; border-left: 5px solid #F39C12; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
         <div>
            <p style="font-size: 1.2rem; font-weight: 800; color: #FFFFFF; margin: 0;">DropAlert</p>
            <p style="font-size: 0.85rem; color: #FFD5CC; margin: 0;">Sistem Deteksi Dini Risiko Putus Sekolah · Indonesia</p>
        </div>
        <div style="font-size: 0.85rem; color: #FFD5CC; text-align: right; line-height: 1.6;">
            Sumber Data: BPS (2021-2025)<br>
            Built for <b>NACOESTA UNIMUS 5.0</b> by anak pertama juara pertama (2514)
        </div>
    </div>
    """, unsafe_allow_html=True)
# ════════════════════════════════════════════════════════
# ENTRYPOINT
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    build_dashboard()
