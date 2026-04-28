"""
Kuveyt Türk AI Lab - Model Envanteri Benzerlik Analiz Sistemi
Streamlit Arayüz Modülü
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Proje kök dizinini Python yoluna ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    APP_ICON,
    APP_TITLE,
    COMPARE_METHOD,
    EMBEDDING_MODEL,
    INVENTORY_PATH,
    MAX_RESULTS,
    THRESHOLD,
)
from app.similarity import analyze_similarity

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sayfa Konfigürasyonu
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS — Kuveyt Türk Teması
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

KT_CSS = """
<style>
/* ── Font Ayarları (İkonları bozmayan güvenli seçiciler) ── */
p, h1, h2, h3, h4, h5, h6, label, input, textarea, button, li {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background-color: #f7f9fa;
}
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* ── Kuveyt Türk Başlık Alanı ── */
.header-card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
    border-top: 5px solid #16a086;
    border-bottom: 1px solid #eaebec;
    border-left: 1px solid #eaebec;
    border-right: 1px solid #eaebec;
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
    overflow: hidden;
}
.header-card::after {
    content: '🏦';
    position: absolute;
    right: 2rem;
    bottom: -1rem;
    font-size: 8rem;
    opacity: 0.03;
    pointer-events: none;
}
.header-title {
    color: #1a1a1a !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    margin: 0 0 0.5rem 0 !important;
    letter-spacing: -0.5px !important;
}
.header-title-accent {
    color: #16a086;
}
.header-subtitle {
    color: #555555 !important;
    font-size: 1.05rem !important;
    font-weight: 400 !important;
    margin: 0 !important;
    max-width: 800px;
    line-height: 1.5;
}

.method-badge {
    align-self: flex-start;
    display: inline-block;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-top: 1.2rem;
    background: #e8f5e9;
    color: #2e7d32;
    border: 1px solid #c8e6c9;
}

/* ── İstatistik Kartları ── */
.stat-card {
    background: #ffffff;
    border: 1px solid #eaebec;
    border-radius: 10px;
    padding: 1.5rem 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    border-bottom: 3px solid #16a086;
    transition: transform 0.2s ease;
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(22,160,134,0.1);
}
.stat-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #16a086;
    margin: 0 0 0.2rem 0;
}
.stat-label {
    font-size: 0.8rem;
    color: #777777;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}

/* ── Sonuç Kartları ── */
.result-card {
    background: #ffffff;
    border: 1px solid #eaebec;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    border-left: 4px solid #eaebec;
    transition: all 0.2s;
}
.result-card:hover {
    border-left-color: #16a086;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.result-model-name {
    color: #1a1a1a;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-model-id {
    display: inline-block;
    background: #f0f2f5;
    color: #555555;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}
.result-purpose {
    color: #444444;
    font-size: 0.9rem;
    line-height: 1.6;
    margin-bottom: 0;
}

/* ── Skor Badge ── */
.score-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
}
.score-high { background: #fff1f0; color: #d32f2f; border: 1px solid #ffcdd2; }
.score-medium { background: #fff8e1; color: #e65100; border: 1px solid #ffe0b2; }
.score-low { background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }

/* ── Bildirim Kutuları ── */
.warning-box {
    background: #fff8f6; border: 1px solid #ffcdd2; border-left: 4px solid #d32f2f;
    border-radius: 8px; padding: 1.2rem 1.5rem; margin: 1.5rem 0;
}
.warning-box h4 { color: #c62828; margin: 0 0 0.3rem 0; font-size: 1.05rem; font-weight: 700; }
.warning-box p { color: #555555; margin: 0; font-size: 0.9rem; line-height: 1.5; }

.success-box {
    background: #f1f9f5; border: 1px solid #c8e6c9; border-left: 4px solid #16a086;
    border-radius: 8px; padding: 1.2rem 1.5rem; margin: 1.5rem 0;
}
.success-box h4 { color: #12826c; margin: 0 0 0.3rem 0; font-size: 1.05rem; font-weight: 700; }
.success-box p { color: #555555; margin: 0; font-size: 0.9rem; line-height: 1.5; }

.llm-rationale {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;
    padding: 0.8rem 1rem; margin-top: 1rem; font-size: 0.85rem;
    color: #475569; font-style: italic; line-height: 1.5;
}

/* ── Sidebar İyileştirmeleri ── */
[data-testid="stSidebar"] {
    background-color: #14917a;
    background-image: linear-gradient(180deg, #14917a 0%, #16a086 100%);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] label {
    color: #ffffff !important;
}
[data-testid="stSidebar"] small {
    color: rgba(255, 255, 255, 0.8) !important;
}

/* Sidebar kapatma butonunu sürekli görünür yap */
[data-testid="stSidebarCollapseButton"], 
[data-testid="baseButton-header"], 
button[kind="header"] {
    opacity: 1 !important;
    visibility: visible !important;
    color: #ffffff !important;
}
[data-testid="stSidebarCollapseButton"]:hover, 
[data-testid="baseButton-header"]:hover, 
button[kind="header"]:hover {
    background: rgba(255, 255, 255, 0.1) !important;
}
[data-testid="stSidebarCollapseButton"] * {
    color: #ffffff !important;
}
/* Slider değerleri görünürlüğü */
[data-testid="stSidebar"] [data-testid="stTickBar"] div {
    color: rgba(255, 255, 255, 0.8) !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] {
    margin-top: 5px;
}
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] > div * {
    color: #ffffff !important;
}

/* ── Buton ve Girdiler ── */
.stButton > button[kind="primary"] {
    background-color: #ff9500 !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 6px rgba(255, 149, 0, 0.2) !important;
    font-size: 0.95rem !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #e68600 !important;
    box-shadow: 0 6px 12px rgba(255, 149, 0, 0.3) !important;
    transform: translateY(-1px) !important;
}

.stTextArea textarea {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    color: #1a1a1a !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: #16a086 !important;
    box-shadow: 0 0 0 2px rgba(22, 160, 134, 0.2) !important;
}

.custom-divider {
    border: 0;
    height: 1px;
    background: #e5e7eb;
    margin: 2rem 0;
}
.sidebar-divider {
    border: 0;
    height: 1px;
    background: rgba(255, 255, 255, 0.2);
    margin: 1.5rem 0;
}
</style>
"""

st.markdown(KT_CSS, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Yardımcı Fonksiyonlar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data
def load_inventory(path: str) -> pd.DataFrame:
    inventory_path = PROJECT_ROOT / path
    return pd.read_csv(inventory_path)


def get_score_class(score: float) -> str:
    if score >= 80:
        return "score-high"
    elif score >= 60:
        return "score-medium"
    else:
        return "score-low"


def get_method_display(method: str) -> tuple[str, str, str]:
    mapping = {
        "text": ("📝 Text (Fuzzy Matching)", "method-text", "Kelime bazlı benzerlik — thefuzz kütüphanesi ile token set ratio"),
        "embedding": ("🧠 Embedding (Semantic)", "method-embedding", f"Semantik benzerlik — {EMBEDDING_MODEL} modeli kosinüs benzerliği"),
        "llm": ("🤖 LLM (Mantıksal Analiz)", "method-llm", "Büyük dil modeli tabanlı fonksiyonel analiz (mock simülasyon)"),
    }
    return mapping.get(method, ("❓ Bilinmeyen", "method-text", ""))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("### ⚙️ Analiz Ayarları")
    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    method_options = {
        "📝 Text (Fuzzy Matching)": "text",
        "🧠 Embedding (Semantic)": "embedding",
        "🤖 LLM (Mantıksal Analiz)": "llm",
    }
    default_display_keys = list(method_options.keys())
    default_values = list(method_options.values())
    default_idx = default_values.index(COMPARE_METHOD) if COMPARE_METHOD in default_values else 0

    selected_method_label = st.selectbox(
        "Karşılaştırma Metodu",
        options=default_display_keys,
        index=default_idx,
        help="Analiz için kullanılacak yöntemi seçin.",
    )
    selected_method = method_options[selected_method_label]

    st.markdown("")

    threshold = st.slider(
        "Benzerlik Eşiği (%)",
        min_value=30, max_value=100, value=THRESHOLD, step=5,
        help="Bu eşiğin üzerindeki modeller mükerrer risk taşıyan olarak işaretlenir.",
    )

    st.markdown("")

    max_results = st.slider(
        "Maksimum Sonuç Sayısı",
        min_value=3, max_value=35, value=MAX_RESULTS, step=1,
        help="Gösterilecek en fazla benzer model sayısı.",
    )

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    method_display, method_css, method_desc = get_method_display(selected_method)
    st.markdown("**Seçili Metot Detayı:**")
    st.markdown(f"<small>{method_desc}</small>", unsafe_allow_html=True)

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<small>© 2026 Kuveyt Türk AI Lab<br>Model Envanter Yönetim Sistemi v1.0</small>",
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ana Başlık
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
method_display, method_css, _ = get_method_display(selected_method)

st.markdown(
    f"""
    <div class="header-card">
        <h1 class="header-title">Model Envanteri <span class="header-title-accent">Benzerlik Analizi</span></h1>
        <p class="header-subtitle">
            Banka içindeki model geliştirme taleplerinin mevcut envanterle mükerrerlik riskini analiz eden akıllı platform.
        </p>
        <div class="method-badge">{method_display}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Veri Yükleme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
try:
    inventory_df = load_inventory(INVENTORY_PATH)
except FileNotFoundError:
    st.error(f"❌ Envanter dosyası bulunamadı: `{INVENTORY_PATH}`")
    st.stop()
except Exception as e:
    st.error(f"❌ Veri yükleme hatası: {e}")
    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# İstatistik Kartları
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f'<div class="stat-card"><p class="stat-value">{len(inventory_df)}</p><p class="stat-label">Kayıtlı Model</p></div>',
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f'<div class="stat-card"><p class="stat-value">%{threshold}</p><p class="stat-label">Risk Eşiği</p></div>',
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f'<div class="stat-card"><p class="stat-value">{selected_method.upper()}</p><p class="stat-label">Aktif Algoritma</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Kullanıcı Giriş Alanı
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("### 📋 Yeni Model Talebi")

query = st.text_area(
    "Model talep açıklamanızı girin:",
    placeholder="Örnek: Operasyonel süreçlerdeki müşteri şikayetlerini doğal dil işleme ile kategorize eden yeni bir AI modeli...",
    height=140,
    label_visibility="collapsed"
)

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    analyze_btn = st.button("Analiz Et", type="primary", use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analiz ve Sonuçlar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if analyze_btn:
    if not query or not query.strip():
        st.warning("⚠️ Lütfen analiz için bir model talep açıklaması girin.")
    else:
        if selected_method == "llm":
            from app.config import LLM_API_KEY
            if not LLM_API_KEY or LLM_API_KEY == "":
                st.warning("⚠️ **LLM API Anahtarı Eksik!**\nYapay zeka (LLM) analizini bilgisayarınızda kullanabilmek için lütfen `.streamlit/secrets.toml` dosyası oluşturup içine `LLM_API_KEY` değerini giriniz. Detaylar için README.md dosyasına bakabilirsiniz.", icon="🔑")
                st.stop()
                
        with st.spinner(f"Arkaplanda {method_display} yöntemi ile envanter taranıyor, metinler karşılaştırılıyor..."):
            try:
                results_df = analyze_similarity(query, inventory_df, method=selected_method)
            except Exception as e:
                st.error(f"❌ Analiz sırasında hata oluştu: {e}")
                st.stop()

        risk_df = results_df[results_df["Benzerlik_Skoru"] >= threshold].head(max_results)

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        if len(risk_df) > 0:
            st.markdown(
                f"""
                <div class="warning-box">
                    <h4>⚠️ Mükerrer Risk Tespit Edildi</h4>
                    <p>
                        Girilen talep ile envanterdeki <strong>{len(risk_df)}</strong> model arasında
                        %{threshold} ve üzeri benzerlik skoru tespit edildi.
                        Lütfen ilgili ekiplerle iletişime geçerek mükerrer efor riskini değerlendirin.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### 🚨 Kritik Risk Taşıyan Modeller")

            for idx, row in risk_df.iterrows():
                score = row["Benzerlik_Skoru"]
                score_class = get_score_class(score)

                card_html = f"""
<div class="result-card">
    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div style="flex:1; padding-right: 1rem;">
            <p class="result-model-name">{row['Model_Adı']}</p>
            <span class="result-model-id">{row['Model_ID']}</span>
            <p class="result-purpose">{row['Model_Amacı']}</p>
        </div>
        <div style="text-align: right;">
            <span class="score-badge {score_class}">%{score}</span>
        </div>
    </div>
"""

                if selected_method == "llm" and "LLM_Gerekçe" in row:
                    card_html += f"""
<div class="llm-rationale">
    <strong>AI Analiz Yorumu:</strong> {row['LLM_Gerekçe']}
</div>
"""

                card_html += "</div>"
                st.markdown(card_html, unsafe_allow_html=True)

        else:
            st.markdown(
                f"""
                <div class="success-box">
                    <h4>✅ Mükerrer Risk Bulunamadı</h4>
                    <p>
                        Girilen talep ile mevcut envanterdeki hiçbir model arasında
                        %{threshold} üzerinde bir benzerlik tespit edilmedi.
                        Yeni model talebini onay sürecine iletebilirsiniz.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        with st.expander("📊 Tüm Analiz Detaylarını Göster (Veri Tablosu)", expanded=False):
            display_df = results_df.head(max_results).copy()

            display_columns = {
                "Model_ID": "ID",
                "Model_Adı": "Model Adı",
                "Model_Amacı": "Kapsam ve Amaç",
                "Benzerlik_Skoru": "Benzerlik (%)",
            }
            display_df = display_df.rename(columns=display_columns)

            if "LLM_Gerekçe" in display_df.columns:
                display_df = display_df.rename(columns={"LLM_Gerekçe": "LLM Gerekçesi"})

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Benzerlik (%)": st.column_config.ProgressColumn(
                        "Skor",
                        help="0-100 arası benzerlik skoru",
                        format="%d%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Envanter Görüntüleme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("<br>", unsafe_allow_html=True)

with st.expander("📁 Mevcut Model Envanterini Görüntüle", expanded=False):
    st.dataframe(
        inventory_df,
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Sistemde toplam {len(inventory_df)} aktif model kaydı bulunmaktadır.")
