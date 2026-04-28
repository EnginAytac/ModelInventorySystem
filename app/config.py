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
# LLM API Ayarları (Gemini 2.5 Flash)
# ──────────────────────────────────────────────────────────────
LLM_API_KEY: str = "AIzaSyDupleiJrfsYhaAC6oLG_yve1jjVRpsiIw"
LLM_MODEL: str = "gemini-2.5-flash"

# ──────────────────────────────────────────────────────────────
# Uygulama Genel Ayarları
# ──────────────────────────────────────────────────────────────
APP_TITLE: str = "Kuveyt Türk AI Lab - Model Envanteri Benzerlik Analizi"
APP_ICON: str = "🏦"
MAX_RESULTS: int = 10  # Gösterilecek maksimum benzer model sayısı
