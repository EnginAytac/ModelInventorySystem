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

st.markdown("""<style>
html, body, p, h1, h2, h3, h4, h5, h6, span, div, label, input, textarea, button {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp { background-color: #ffffff; }
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
h1, h2, h3, h4, h5, h6 { color: #1a1a1a !important; }
.main p, .main span, .main label { color: #333333; }

.header-container {
    background: linear-gradient(135deg, #009e7e 0%, #00a884 50%, #00b890 100%);
    padding: 2rem 2.5rem; border-radius: 14px; margin-bottom: 1.8rem;
    box-shadow: 0 4px 20px rgba(0,168,132,0.3); position: relative; overflow: hidden;
}
.header-container::before {
    content:''; position:absolute; top:-60%; right:-15%;
    width:350px; height:350px;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    border-radius:50%;
}
.header-title {
    color:#ffffff; font-size:1.85rem; font-weight:800; margin:0;
    position:relative; z-index:1;
}
.header-subtitle {
    color:rgba(255,255,255,0.9); font-size:0.95rem; margin-top:0.5rem;
    font-weight:400; position:relative; z-index:1;
}
.method-badge {
    display:inline-block; padding:5px 16px; border-radius:20px;
    font-size:0.75rem; font-weight:600; letter-spacing:0.5px;
    text-transform:uppercase; margin-top:0.7rem; position:relative; z-index:1;
    background:rgba(255,255,255,0.2); color:#ffffff;
    border:1px solid rgba(255,255,255,0.4);
}

.result-card {
    background:#ffffff; border:1px solid #e0e0e0; border-radius:10px;
    padding:1.3rem 1.5rem; margin-bottom:0.9rem;
    transition:all 0.25s ease; box-shadow:0 1px 4px rgba(0,0,0,0.06);
}
.result-card:hover {
    border-color:#00a884; box-shadow:0 4px 16px rgba(0,168,132,0.15);
    transform:translateY(-1px);
}
.result-model-name { color:#1a1a1a; font-size:1.05rem; font-weight:700; margin-bottom:0.25rem; }
.result-model-id { color:#00a884; font-size:0.78rem; font-weight:600; margin-bottom:0.5rem; }
.result-purpose { color:#555555; font-size:0.88rem; line-height:1.55; margin-bottom:0.7rem; }

.score-badge { display:inline-block; padding:5px 16px; border-radius:20px; font-size:0.85rem; font-weight:700; }
.score-high { background:#fff1f0; color:#d32f2f; border:1px solid #ffcdd2; }
.score-medium { background:#fff8e1; color:#e65100; border:1px solid #ffe0b2; }
.score-low { background:#e8f5e9; color:#2e7d32; border:1px solid #c8e6c9; }

.stat-card {
    background:#00a884; border:none; border-radius:10px;
    padding:1.2rem 1.3rem; text-align:center;
    box-shadow:0 2px 10px rgba(0,168,132,0.2);
}
.stat-value { font-size:1.9rem; font-weight:800; color:#ffffff; margin:0; }
.stat-label { font-size:0.75rem; color:rgba(255,255,255,0.85); text-transform:uppercase; letter-spacing:0.8px; margin-top:0.3rem; font-weight:500; }

.warning-box {
    background:#fff8f6; border:1px solid #ffcdd2; border-left:4px solid #d32f2f;
    border-radius:8px; padding:1.1rem 1.4rem; margin:1.5rem 0;
}
.warning-box h4 { color:#c62828; margin:0 0 0.3rem 0; font-size:1rem; font-weight:700; }
.warning-box p { color:#555555; margin:0; font-size:0.88rem; line-height:1.5; }

.success-box {
    background:#f1f9f5; border:1px solid #c8e6c9; border-left:4px solid #00a884;
    border-radius:8px; padding:1.1rem 1.4rem; margin:1.5rem 0;
}
.success-box h4 { color:#00896b; margin:0 0 0.3rem 0; font-size:1rem; font-weight:700; }
.success-box p { color:#555555; margin:0; font-size:0.88rem; line-height:1.5; }

.llm-rationale {
    background:#f3f0ff; border:1px solid #d9d0f5; border-radius:8px;
    padding:0.7rem 1rem; margin-top:0.7rem; font-size:0.84rem;
    color:#5b21b6; font-style:italic; line-height:1.5;
}

[data-testid="stSidebar"] { background:#00a884 !important; }
[data-testid="stSidebar"] * { color:#ffffff !important; }
[data-testid="stSidebar"] small { color:rgba(255,255,255,0.75) !important; }

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background:#f7941e !important; border:none !important; color:#ffffff !important;
    font-weight:700 !important; border-radius:25px !important;
    padding:0.55rem 2rem !important; transition:all 0.25s ease !important;
    box-shadow:0 2px 8px rgba(247,148,30,0.3) !important; font-size:0.9rem !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background:#e8850e !important;
    box-shadow:0 4px 14px rgba(247,148,30,0.4) !important;
    transform:translateY(-1px) !important;
}

.custom-divider { border:0; height:1px; background:linear-gradient(90deg, transparent, #e0e0e0, transparent); margin:1.5rem 0; }

.stTextArea textarea {
    background:#ffffff !important; border:1.5px solid #d0d0d0 !important;
    border-radius:8px !important; color:#1a1a1a !important; font-size:0.9rem !important;
}
.stTextArea textarea:focus { border-color:#00a884 !important; box-shadow:0 0 0 2px rgba(0,168,132,0.15) !important; }
.stTextArea label p { color:#333333 !important; }

div[data-testid="stExpander"] { background:#ffffff; border:1px solid #e0e0e0; border-radius:10px; }
div[data-testid="stExpander"] summary span p { color:#333333 !important; }
</style>""", unsafe_allow_html=True)


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
        "embedding": ("🧠 Embedding (Semantic)", "method-embedding", f"Semantik benzerlik — {EMBEDDING_MODEL} modeli ile kosinüs benzerliği"),
        "llm": ("🤖 LLM (Mantıksal Analiz)", "method-llm", "Büyük dil modeli tabanlı fonksiyonel analiz (mock simülasyon)"),
    }
    return mapping.get(method, ("❓ Bilinmeyen", "method-text", ""))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("### ⚙️ Analiz Ayarları")
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

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
        help="Benzerlik analizi için kullanılacak yöntemi seçin.",
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

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    method_display, method_css, method_desc = get_method_display(selected_method)
    st.markdown("**Seçili Metot Detayı:**")
    st.markdown(f"<small>{method_desc}</small>", unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<small style='color: rgba(255,255,255,0.7);'>© 2026 Kuveyt Türk AI Lab<br>Model Envanter Yönetim Sistemi v1.0</small>",
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ana Başlık
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
method_display, method_css, _ = get_method_display(selected_method)
st.markdown(
    f"""
    <div class="header-container">
        <h1 class="header-title">{APP_ICON} {APP_TITLE}</h1>
        <p class="header-subtitle">
            Banka içi mükerrer model taleplerini tespit eden akıllı benzerlik analiz platformu
        </p>
        <span class="method-badge">{method_display}</span>
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
        f'<div class="stat-card"><p class="stat-value">{len(inventory_df)}</p><p class="stat-label">Envanterdeki Model</p></div>',
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f'<div class="stat-card"><p class="stat-value">%{threshold}</p><p class="stat-label">Benzerlik Eşiği</p></div>',
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f'<div class="stat-card"><p class="stat-value">{selected_method.upper()}</p><p class="stat-label">Aktif Metot</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Kullanıcı Giriş Alanı
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("### 📋 Yeni Model Talebi")

query = st.text_area(
    "Model talep açıklamanızı girin:",
    placeholder="Örnek: Müşteri kredi başvurularında temerrüt riskini tahmin eden bir makine öğrenmesi modeli geliştirmek istiyoruz.",
    height=120,
    help="Talebin ne kadar detaylı girilirse, benzerlik analizi o kadar doğru sonuç üretir.",
)

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    analyze_btn = st.button("🔍 Analiz Et", type="primary", use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analiz ve Sonuçlar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if analyze_btn:
    if not query or not query.strip():
        st.warning("⚠️ Lütfen bir model talep açıklaması girin.")
    else:
        with st.spinner(f"🔄 {method_display} yöntemi ile envanter taranıyor..."):
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
                        Lütfen bu modelleri inceleyerek mükerrer geliştirme riskini değerlendirin.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### 🚨 Mükerrer Risk Taşıyan Modeller")

            for idx, row in risk_df.iterrows():
                score = row["Benzerlik_Skoru"]
                score_class = get_score_class(score)

                card_html = f"""
                <div class="result-card">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div style="flex:1;">
                            <p class="result-model-name">{row['Model_Adı']}</p>
                            <p class="result-model-id">{row['Model_ID']}</p>
                            <p class="result-purpose">{row['Model_Amacı']}</p>
                        </div>
                        <div>
                            <span class="score-badge {score_class}">%{score}</span>
                        </div>
                    </div>
                """

                if selected_method == "llm" and "LLM_Gerekçe" in row:
                    card_html += f"""
                    <div class="llm-rationale">
                        💡 <strong>LLM Gerekçesi:</strong> {row['LLM_Gerekçe']}
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
                        Girilen talep ile envanterdeki hiçbir model arasında
                        %{threshold} üzeri benzerlik tespit edilmedi.
                        Yeni model talebi güvenle oluşturulabilir.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        with st.expander("📊 Tüm Analiz Sonuçlarını Görüntüle", expanded=False):
            display_df = results_df.head(max_results).copy()

            display_columns = {
                "Model_ID": "Model ID",
                "Model_Adı": "Model Adı",
                "Model_Amacı": "Model Amacı",
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
                        "Benzerlik (%)",
                        help="Benzerlik skoru",
                        format="%d%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Envanter Görüntüleme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

with st.expander("📁 Mevcut Model Envanterini Görüntüle", expanded=False):
    st.dataframe(
        inventory_df,
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Toplam {len(inventory_df)} model kayıtlı.")
