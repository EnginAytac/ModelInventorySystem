"""
Kuveyt Türk AI Lab - Model Envanteri Benzerlik Analiz Sistemi
Konfigürasyon Dosyası

Bu dosya, sistem genelinde kullanılan temel ayarları ve metot seçimlerini içerir.
"""

# ──────────────────────────────────────────────────────────────
# Benzerlik Karşılaştırma Metodu
# Kullanılabilir değerler: "text", "embedding", "llm"
#   - "text"      : Fuzzy string matching (thefuzz kütüphanesi)
#   - "embedding" : Semantik benzerlik (sentence-transformers)
#   - "llm"       : LLM tabanlı mantıksal analiz (mock/simülasyon)
# ──────────────────────────────────────────────────────────────
COMPARE_METHOD: str = "text"

# ──────────────────────────────────────────────────────────────
# Benzerlik Eşik Değeri (Threshold)
# 0-100 arasında bir değer. Bu eşiğin üzerindeki modeller
# "mükerrer risk" taşıyan olarak işaretlenir.
# ──────────────────────────────────────────────────────────────
THRESHOLD: int = 70

# ──────────────────────────────────────────────────────────────
# Envanter Veri Dosyası Yolu
# ──────────────────────────────────────────────────────────────
INVENTORY_PATH: str = "data/inventory.csv"

# ──────────────────────────────────────────────────────────────
# Embedding Model Ayarları
# ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"

# ──────────────────────────────────────────────────────────────
# LLM API Ayarları (Groq Llama 3)
# ──────────────────────────────────────────────────────────────
import streamlit as st
import os

try:
    # Bulut ortamında (Streamlit Cloud) şifreli kasadan okur
    LLM_API_KEY: str = st.secrets["LLM_API_KEY"]
except Exception:
    # Lokal ortamda veya kasa bulunamazsa çevre değişkenlerinden okur
    LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")

LLM_MODEL: str = "llama-3.1-8b-instant"

# ──────────────────────────────────────────────────────────────
# Uygulama Genel Ayarları
# ──────────────────────────────────────────────────────────────
APP_TITLE: str = "Kuveyt Türk AI Lab - Model Envanteri Benzerlik Analizi"
APP_ICON: str = "🏦"
MAX_RESULTS: int = 10  # Gösterilecek maksimum benzer model sayısı
