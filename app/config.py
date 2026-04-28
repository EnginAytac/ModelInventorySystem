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
COMPARE_METHOD: str = "embedding"

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
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ──────────────────────────────────────────────────────────────
# LLM API Ayarları (Şimdilik mock olarak çalışır)
# ──────────────────────────────────────────────────────────────
LLM_API_KEY: str = ""
LLM_MODEL: str = "gpt-4"

# ──────────────────────────────────────────────────────────────
# Uygulama Genel Ayarları
# ──────────────────────────────────────────────────────────────
APP_TITLE: str = "Kuveyt Türk AI Lab - Model Envanteri Benzerlik Analizi"
APP_ICON: str = "🏦"
MAX_RESULTS: int = 10  # Gösterilecek maksimum benzer model sayısı
