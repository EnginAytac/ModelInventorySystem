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
        - Model Adı  : %15
        - Model Amacı : %85

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
        import streamlit as st
        
        # Kullanıcıya indirme/yükleme işlemi için bilgi veriyoruz (Bittiğinde kaybolacak)
        placeholder = st.empty()
        placeholder.info("📦 **Yapay Zeka Modeli Hazırlanıyor...** İlk kullanım olduğu için çok dilli semantik model (yaklaşık 470 MB) indiriliyor veya belleğe yükleniyor. Bu işlem 1-2 dakika sürebilir, lütfen bekleyiniz.")
        
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Yükleme bittiğinde bilgi mesajını temizle
        placeholder.empty()
        
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

    # Ad ve Amaç metinlerini listelere ayır
    name_texts = inventory_df["Model_Adı"].astype(str).tolist()
    purpose_texts = inventory_df["Model_Amacı"].astype(str).tolist()

    # İki alanı ayrı ayrı vektörize et (Bu sayede ağırlıkları tam kontrol edebiliriz)
    name_embeddings = model.encode(name_texts, convert_to_numpy=True)
    purpose_embeddings = model.encode(purpose_texts, convert_to_numpy=True)

    results = []
    for idx, row in inventory_df.iterrows():
        # Ayrı ayrı kosinüs benzerliklerini hesapla
        name_sim = _cosine_similarity(query_embedding, name_embeddings[idx])
        purpose_sim = _cosine_similarity(query_embedding, purpose_embeddings[idx])
        
        # Ağırlıklı birleştirme: İsim %15, Amaç %85 etki etsin
        combined_sim = (name_sim * 0.15) + (purpose_sim * 0.85)
        
        score = round(combined_sim * 100, 1)  # Yüzdelik dönüşüm

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


def compute_llm_similarity(query: str, inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Google Gemini (LLM) tabanlı mantıksal benzerlik analizi (Toplu İşlem).
    """
    import json
    import google.generativeai as genai
    from app.config import LLM_API_KEY, LLM_MODEL

    if not LLM_API_KEY:
        raise ValueError("LLM_API_KEY bulunamadı. Lütfen config.py dosyasını kontrol ediniz.")

    genai.configure(api_key=LLM_API_KEY)
    
    # Gemini 1.5 Flash - JSON modu aktif
    model = genai.GenerativeModel(
        LLM_MODEL,
        generation_config={"response_mime_type": "application/json"}
    )

    # Modelleri toplu (batch) prompt için listele
    inventory_data = []
    for _, row in inventory_df.iterrows():
        inventory_data.append(
            f'- {{"Model_ID": "{row["Model_ID"]}", "Model_Adı": "{row["Model_Adı"]}", "Model_Amacı": "{row["Model_Amacı"]}"}}'
        )
    
    inventory_str = "\n".join(inventory_data)

    prompt = f"""Sen Kuveyt Türk Bankası için çalışan bir Yapay Zeka Model Yönetim Uzmanısın.
Görev: Kullanıcının talep ettiği YENİ MODEL ile envanterdeki MEVCUT MODELLERİ karşılaştırmak.

[YENİ MODEL TALEBİ]:
{query}

[MEVCUT MODELLER ENVANTERİ (Toplam {len(inventory_df)} Model)]:
{inventory_str}

Lütfen yukarıdaki TÜM mevcut modelleri dikkatlice incele.
Her bir mevcut model için, yeni taleple olan benzerlik skorunu (0-100 arası) hesapla.
Kural: Model amacına %90, ismine %10 ağırlık ver.

PUANLAMA KRİTERİ VE GEREKÇE UZUNLUĞU (HIZ OPTİMİZASYONU): 
1. Birebir aynı iş problemine hizmet eden modellere %85-100 arası puan ver.
2. Bankacılıkta tematik olarak birbirine benzeyen ancak UYGULAMA ALANLARI farklı olan (Örn: İkisi de güvenlik modelidir ama biri Kredi Kartı Sahteciliği, diğeri Kara Para Aklamadır) "kısmen benzer" modellere %40-65 arası daha düşük ve dengeli puan ver. İkisi de aynı ana konuya ait diye yüksek puan verme!
3. Amaçları FARKLI olan modellere yüksek puan vermekten KESİNLİKLE kaçın. İlgisiz veya alakasız modellere katı davranarak %0-35 arası puan ver.

HIZ VE TOKEN OPTİMİZASYONU (ÇOK ÖNEMLİ):
- Skoru %40 ve üzerinde olan modeller için 1-2 cümlelik normal açıklayıcı bir gerekçe yaz.
- Skoru %40'ın altında olan modeller için uzun cümleler kurma! Vakitten tasarruf etmek için MAKSİMUM 3-5 KELİMELİK çok kısa ve öz gerekçeler yaz (Örn: "Odak alanları tamamen farklı", "Farklı departmanlara hizmet ediyor", "Alakasız bir veri modeli").

Yanıtın KESİNLİKLE aşağıdaki JSON formatında geçerli bir LİSTE olmalıdır. JSON harici hiçbir metin yazma:
[
  {{"Model_ID": "MOD-1", "skor": 45, "gerekce": "Kısa ve net Türkçe açıklama..."}},
  ... (TÜM modeller için eksiksiz liste)
]
"""

    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        
        # Markdown bloklarını temizle (bazen model yine de koyabiliyor)
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.replace("```", "").strip()
            
        json_results = json.loads(raw_text)
        llm_df = pd.DataFrame(json_results)
        
        # Orijinal DataFrame ile birleştir (Model_ID bazlı)
        merged_df = pd.merge(inventory_df, llm_df, on="Model_ID", how="left")
        
        # Boş kalanlar olursa 0 ata
        merged_df["skor"] = merged_df["skor"].fillna(0).astype(int)
        merged_df["gerekce"] = merged_df["gerekce"].fillna("LLM bu modeli değerlendiremedi.")
        
        # UI ile uyumlu hale getir
        merged_df = merged_df.rename(columns={
            "skor": "Benzerlik_Skoru", 
            "gerekce": "LLM_Gerekçe"
        })
        
        merged_df = merged_df.sort_values("Benzerlik_Skoru", ascending=False).reset_index(drop=True)
        return merged_df

    except Exception as e:
        import streamlit as st
        st.error(f"LLM API Hatası: {str(e)}")
        # Uygulama çökmesin diye boş DataFrame dön
        empty_df = inventory_df.copy()
        empty_df["Benzerlik_Skoru"] = 0
        empty_df["LLM_Gerekçe"] = "API Hatası."
        return empty_df


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
