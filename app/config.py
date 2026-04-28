"""
Kuveyt Türk AI Lab - Model Envanteri Benzerlik Analiz Sistemi
Konfigürasyon Modülü

Bu modül, sistem genelinde kullanılan tüm sabit değerleri, API anahtarı
çözümleme mantığını ve uygulama ayarlarını merkezi olarak tanımlar.
Ortam değişkeni önceliği: Streamlit Secrets → OS Environment Variable.
"""

import os

import streamlit as st


def _resolve_llm_api_key() -> str:
    """Groq LLM API anahtarını ortama göre çözümler.

    Öncelik sırası:
        1. Streamlit Cloud secrets kasası (``st.secrets["LLM_API_KEY"]``)
        2. İşletim sistemi ortam değişkeni (``LLM_API_KEY``)
        3. Boş string (anahtar bulunamazsa)

    Returns:
        API anahtarı string'i; bulunamazsa boş string.
    """
    try:
        return st.secrets["LLM_API_KEY"]
    except Exception:
        return os.environ.get("LLM_API_KEY", "")


# ──────────────────────────────────────────────────────────────
# Benzerlik Karşılaştırma Metodu
# Kullanılabilir değerler: "text", "embedding", "llm"
#   - "text"      : Fuzzy string matching (thefuzz kütüphanesi)
#   - "embedding" : Semantik benzerlik (sentence-transformers)
#   - "llm"       : LLM tabanlı mantıksal analiz (Groq Llama 3)
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
LLM_API_KEY: str = _resolve_llm_api_key()
LLM_MODEL: str = "llama-3.3-70b-versatile"

# ──────────────────────────────────────────────────────────────
# Uygulama Genel Ayarları
# ──────────────────────────────────────────────────────────────
APP_TITLE: str = "Kuveyt Türk AI Lab - Model Envanteri Benzerlik Analizi"
APP_ICON: str = "🏦"
MAX_RESULTS: int = 10  # Gösterilecek maksimum benzer model sayısı
