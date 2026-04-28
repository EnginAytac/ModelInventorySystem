"""
Kuveyt Türk AI Lab - Model Envanteri Benzerlik Analiz Sistemi
Benzerlik Algoritmaları Modülü

Bu modül, 3 farklı benzerlik analiz yöntemini içerir:
  1. TEXT   — Fuzzy string matching (thefuzz)
  2. EMBEDDING — Semantik benzerlik (sentence-transformers + cosine similarity)
  3. LLM   — Büyük dil modeli tabanlı mantıksal analiz (mock/simülasyon)
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Union

import numpy as np
import pandas as pd
from thefuzz import fuzz

from app.config import EMBEDDING_MODEL


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1) TEXT — Fuzzy String Matching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_text_similarity(query: str, inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    thefuzz kütüphanesi ile fuzzy string matching uygular.

    Hem model adı hem de model amacı üzerinden ayrı ayrı benzerlik
    skoru hesaplar ve ağırlıklı ortalama ile tek bir skor üretir.

    Ağırlıklar:
        - Model Adı  : %40
        - Model Amacı : %60

    Args:
        query: Kullanıcının girdiği yeni model talep açıklaması.
        inventory_df: Envanter DataFrame'i (Model_ID, Model_Adı, Model_Amacı).

    Returns:
        Benzerlik skoru (Benzerlik_Skoru sütunu) eklenmiş DataFrame.
    """
    results = []

    for _, row in inventory_df.iterrows():
        # Token set ratio — kelime sırası bağımsız eşleştirme
        name_score = fuzz.token_set_ratio(query.lower(), str(row["Model_Adı"]).lower())
        purpose_score = fuzz.token_set_ratio(query.lower(), str(row["Model_Amacı"]).lower())

        # Ağırlıklı ortalama (Model adı ağırlığı düşürüldü, amaca odaklanıldı)
        combined_score = round(name_score * 0.15 + purpose_score * 0.85, 1)

        results.append(
            {
                "Model_ID": row["Model_ID"],
                "Model_Adı": row["Model_Adı"],
                "Model_Amacı": row["Model_Amacı"],
                "Benzerlik_Skoru": combined_score,
            }
        )

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("Benzerlik_Skoru", ascending=False).reset_index(drop=True)
    return result_df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2) EMBEDDING — Semantik Benzerlik
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Lazy-loaded model cache
_embedding_model = None


def _get_embedding_model():
    """Sentence-transformer modelini tembel yükleme ile döndürür."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """İki vektör arasındaki kosinüs benzerliğini hesaplar."""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compute_embedding_similarity(query: str, inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sentence-transformers (all-MiniLM-L6-v2) ile semantik benzerlik hesaplar.

    Her envanter kaydı için (Model_Adı + Model_Amacı) birleşik metnin
    embedding vektörünü çıkarır ve sorgu ile kosinüs benzerliğini hesaplar.

    Args:
        query: Kullanıcının girdiği yeni model talep açıklaması.
        inventory_df: Envanter DataFrame'i.

    Returns:
        Benzerlik skoru (0-100) eklenmiş, azalan sırada sıralanmış DataFrame.
    """
    model = _get_embedding_model()

    # Sorgu embeddingi
    query_embedding = model.encode(query, convert_to_numpy=True)

    # Embedding performansını artırmak için SADECE Model Amacını vektörize ediyoruz.
    # Kullanıcılar isim değil işlem ("kredi riski hesaplama" vb.) yazdığı için
    # model adı gürültü (noise) yaratıyor.
    inventory_texts = inventory_df["Model_Amacı"].astype(str).tolist()

    # Toplu embedding
    inventory_embeddings = model.encode(inventory_texts, convert_to_numpy=True)

    results = []
    for idx, row in inventory_df.iterrows():
        sim = _cosine_similarity(query_embedding, inventory_embeddings[idx])
        score = round(sim * 100, 1)  # Yüzdelik dönüşüm

        results.append(
            {
                "Model_ID": row["Model_ID"],
                "Model_Adı": row["Model_Adı"],
                "Model_Amacı": row["Model_Amacı"],
                "Benzerlik_Skoru": score,
            }
        )

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("Benzerlik_Skoru", ascending=False).reset_index(drop=True)
    return result_df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3) LLM — Büyük Dil Modeli Tabanlı Analiz (Mock / Simülasyon)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_llm_prompt(query: str, model_name: str, model_purpose: str) -> str:
    """LLM'e gönderilecek karşılaştırma promptunu oluşturur."""
    return f"""Sen bir bankacılık model yönetimi uzmanısın.

Aşağıda yeni bir model talebi ve mevcut envanterdeki bir model bilgisi verilmiştir.
Bu iki modelin birbirinin mükerrer veya çok benzeri olup olmadığını analiz et.

### Yeni Model Talebi:
{query}

### Mevcut Envanter Modeli:
- **Model Adı:** {model_name}
- **Model Amacı:** {model_purpose}

### Görev:
1. İki modelin fonksiyonel örtüşme oranını 0-100 arasında bir skor ile değerlendir.
2. Değerlendirme yaparken "Model Amacı"na %90, "Model Adı"na %10 ağırlık ver.
3. Kısa bir gerekçe yaz.

Yanıtını şu JSON formatında ver:
{{"skor": <0-100>, "gerekce": "<kısa açıklama>"}}
"""


def _mock_llm_response(query: str, model_name: str, model_purpose: str) -> dict:
    """
    LLM API çağrısını simüle eder.

    Gerçek bir API entegrasyonu olmadığında, metin benzerliğine dayalı
    deterministik bir mock skor üretir. Hash tabanlı deterministik
    rastgelelik kullanarak tutarlı sonuçlar sağlar.
    """
    # Deterministik skor üretimi (aynı girdi → aynı skor)
    combined = f"{query}|{model_name}|{model_purpose}"
    hash_val = int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)

    # Fuzzy matching bazlı amaca yönelik temel skor (Model adının ağırlığı çok düşük)
    name_score = fuzz.token_set_ratio(query.lower(), model_name.lower())
    purpose_score = fuzz.token_set_ratio(query.lower(), model_purpose.lower())
    base_score = (name_score * 0.10) + (purpose_score * 0.90)

    # Hash'e dayalı küçük varyasyon (-5, +5 arası)
    variation = (hash_val % 11) - 5
    score = max(0, min(100, base_score + variation))

    # Gerekçe şablonları
    if score >= 80:
        gerekce = (
            f"Yeni talep ile '{model_name}' arasında yüksek fonksiyonel örtüşme tespit edildi. "
            "Her iki model de benzer veri kaynaklarını kullanarak aynı iş problemini çözmeye yönelik."
        )
    elif score >= 60:
        gerekce = (
            f"Yeni talep ile '{model_name}' arasında kısmi benzerlik mevcut. "
            "Modellerin kapsamları örtüşse de farklı metodolojiler veya hedef kitleler söz konusu olabilir."
        )
    elif score >= 40:
        gerekce = (
            f"Yeni talep ile '{model_name}' arasında sınırlı benzerlik bulunmakta. "
            "Aynı alan içinde farklı problemlere odaklanıyorlar."
        )
    else:
        gerekce = (
            f"Yeni talep ile '{model_name}' arasında anlamlı bir benzerlik tespit edilmedi. "
            "Modeller farklı iş alanlarına ve farklı problemlere yönelik."
        )

    return {"skor": score, "gerekce": gerekce}


def compute_llm_similarity(query: str, inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    LLM tabanlı mantıksal benzerlik analizi (Mock / Simülasyon).

    Gerçek bir LLM API entegrasyonu için prompt yapısını gösterir.
    Şu an mock yanıtlar döndürür; gerçek API anahtarı eklendiğinde
    _mock_llm_response yerine gerçek API çağrısı yapılabilir.

    Args:
        query: Kullanıcının girdiği yeni model talep açıklaması.
        inventory_df: Envanter DataFrame'i.

    Returns:
        Benzerlik skoru ve LLM gerekçesi eklenmiş DataFrame.
    """
    results = []

    for _, row in inventory_df.iterrows():
        model_name = str(row["Model_Adı"])
        model_purpose = str(row["Model_Amacı"])

        # Prompt oluştur (loglama / debug için saklanabilir)
        _prompt = _build_llm_prompt(query, model_name, model_purpose)

        # Mock LLM yanıtı (gerçek entegrasyonda API çağrısı yapılır)
        response = _mock_llm_response(query, model_name, model_purpose)

        results.append(
            {
                "Model_ID": row["Model_ID"],
                "Model_Adı": row["Model_Adı"],
                "Model_Amacı": row["Model_Amacı"],
                "Benzerlik_Skoru": response["skor"],
                "LLM_Gerekçe": response["gerekce"],
            }
        )

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("Benzerlik_Skoru", ascending=False).reset_index(drop=True)
    return result_df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ana Dispatcher Fonksiyonu
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def analyze_similarity(
    query: str,
    inventory_df: pd.DataFrame,
    method: str = "text",
) -> pd.DataFrame:
    """
    Seçilen metoda göre uygun benzerlik analiz fonksiyonunu çağırır.

    Args:
        query: Yeni model talep açıklaması.
        inventory_df: Model envanteri DataFrame'i.
        method: Karşılaştırma metodu ("text", "embedding", "llm").

    Returns:
        Skorlanmış ve sıralanmış DataFrame.

    Raises:
        ValueError: Geçersiz metot seçimi durumunda.
    """
    method = method.lower().strip()

    dispatch = {
        "text": compute_text_similarity,
        "embedding": compute_embedding_similarity,
        "llm": compute_llm_similarity,
    }

    if method not in dispatch:
        raise ValueError(
            f"Geçersiz karşılaştırma metodu: '{method}'. "
            f"Desteklenen metodlar: {list(dispatch.keys())}"
        )

    return dispatch[method](query, inventory_df)
