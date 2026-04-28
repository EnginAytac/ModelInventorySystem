"""
Kuveyt Türk AI Lab - Model Envanteri Benzerlik Analiz Sistemi
Benzerlik Algoritmaları Modülü

Bu modül, 3 farklı benzerlik analiz yöntemini içerir:
  1. TEXT      — Fuzzy string matching (thefuzz)
  2. EMBEDDING — Semantik benzerlik (sentence-transformers + cosine similarity)
  3. LLM       — Büyük dil modeli tabanlı mantıksal analiz (Groq Llama 3)
"""

import json

import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
from thefuzz import fuzz

from app.config import EMBEDDING_MODEL, LLM_API_KEY, LLM_MODEL


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1) TEXT — Fuzzy String Matching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_text_similarity(query: str, inventory_df: pd.DataFrame) -> pd.DataFrame:
    """thefuzz kütüphanesi ile fuzzy string matching uygular.

    Hem model adı hem de model amacı üzerinden ayrı ayrı benzerlik
    skoru hesaplar ve ağırlıklı ortalama ile tek bir skor üretir.

    Ağırlıklar:
        - Model Adı   : %15
        - Model Amacı : %85

    Args:
        query: Kullanıcının girdiği yeni model talep açıklaması.
        inventory_df: Envanter DataFrame'i (Model_ID, Model_Adı, Model_Amacı).

    Returns:
        Benzerlik skoru (Benzerlik_Skoru sütunu) eklenmiş, azalan sırada
        sıralanmış DataFrame.
    """
    query_lower = query.lower()
    results: list[dict] = []

    for _, row in inventory_df.iterrows():
        # Token set ratio — kelime sırası bağımsız eşleştirme
        name_score = fuzz.token_set_ratio(query_lower, str(row["Model_Adı"]).lower())
        purpose_score = fuzz.token_set_ratio(query_lower, str(row["Model_Amacı"]).lower())

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

# Lazy-loaded model cache (modül seviyesinde tek örnek)
_embedding_model = None


def _get_embedding_model():
    """Sentence-transformer modelini tembel yükleme (lazy loading) ile döndürür.

    İlk çağrıda modeli indirip belleğe yükler ve bir sonraki çağrılarda
    önbellekteki örneği yeniden kullanır. Yükleme süresince kullanıcıya
    Streamlit bilgi mesajı gösterilir.

    Returns:
        Yüklenmiş ``SentenceTransformer`` model örneği.
    """
    global _embedding_model

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        placeholder = st.empty()
        placeholder.info(
            "📦 **Yapay Zeka Modeli Hazırlanıyor...** İlk kullanım olduğu için "
            "çok dilli semantik model (yaklaşık 470 MB) indiriliyor veya belleğe "
            "yükleniyor. Bu işlem 1-2 dakika sürebilir, lütfen bekleyiniz."
        )
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        placeholder.empty()

    return _embedding_model


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """İki vektör arasındaki kosinüs benzerliğini hesaplar.

    Args:
        vec_a: Birinci vektör.
        vec_b: İkinci vektör.

    Returns:
        0.0 ile 1.0 arasında kosinüs benzerlik skoru; sıfır normlu
        vektörler için 0.0 döner.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def compute_embedding_similarity(query: str, inventory_df: pd.DataFrame) -> pd.DataFrame:
    """Sentence-transformers ile semantik benzerlik hesaplar.

    Her envanter kaydı için (Model_Adı + Model_Amacı) alanlarının embedding
    vektörlerini ayrı ayrı çıkarır ve sorgu ile ağırlıklı kosinüs benzerliğini
    hesaplar.

    Ağırlıklar:
        - Model Adı   : %15
        - Model Amacı : %85

    Args:
        query: Kullanıcının girdiği yeni model talep açıklaması.
        inventory_df: Envanter DataFrame'i.

    Returns:
        Benzerlik skoru (0–100) eklenmiş, azalan sırada sıralanmış DataFrame.
    """
    model = _get_embedding_model()

    query_embedding: np.ndarray = model.encode(query, convert_to_numpy=True)

    name_texts = inventory_df["Model_Adı"].astype(str).tolist()
    purpose_texts = inventory_df["Model_Amacı"].astype(str).tolist()

    # İki alanı ayrı ayrı vektörize et (ağırlık kontrolü için)
    name_embeddings: np.ndarray = model.encode(name_texts, convert_to_numpy=True)
    purpose_embeddings: np.ndarray = model.encode(purpose_texts, convert_to_numpy=True)

    results: list[dict] = []
    # enumerate kullanarak index'i güvenli şekilde sıfırdan alıyoruz
    for i, (_, row) in enumerate(inventory_df.iterrows()):
        name_sim = _cosine_similarity(query_embedding, name_embeddings[i])
        purpose_sim = _cosine_similarity(query_embedding, purpose_embeddings[i])

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
# 3) LLM — Büyük Dil Modeli Tabanlı Analiz (Groq Llama 3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_inventory_prompt_lines(inventory_df: pd.DataFrame) -> str:
    """Envanter DataFrame'inden LLM prompt'u için liste formatı oluşturur.

    Args:
        inventory_df: Envanter DataFrame'i.

    Returns:
        Her satırı JSON benzeri bir string olan, satır sonları ile
        ayrılmış envanter metin bloğu.
    """
    lines = [
        f'- {{"Model_ID": "{row["Model_ID"]}", "Model_Adı": "{row["Model_Adı"]}", '
        f'"Model_Amacı": "{row["Model_Amacı"]}"}}'
        for _, row in inventory_df.iterrows()
    ]
    return "\n".join(lines)


def _build_llm_prompt(query: str, inventory_df: pd.DataFrame) -> str:
    """Groq API'ye gönderilecek analiz prompt metnini oluşturur.

    Args:
        query: Kullanıcının girdiği yeni model talep açıklaması.
        inventory_df: Envanter DataFrame'i.

    Returns:
        Tam prompt metni.
    """
    inventory_str = _build_inventory_prompt_lines(inventory_df)
    return f"""Sen Kuveyt Türk Bankası için çalışan bir Yapay Zeka Model Yönetim Uzmanısın.
Görev: Kullanıcının talep ettiği YENİ MODEL ile envanterdeki MEVCUT MODELLERİ karşılaştırmak.

[YENİ MODEL TALEBİ]:
{query}

[MEVCUT MODELLER ENVANTERİ (Toplam {len(inventory_df)} Model)]:
{inventory_str}

Lütfen yukarıdaki TÜM mevcut modelleri dikkatlice incele.
Her bir mevcut model için, yeni taleple olan benzerlik skorunu (0-100 arası) hesapla.
Kural: Model amacına %90, ismine %10 ağırlık ver.

PUANLAMA KRİTERİ VE GEREKÇE UZUNLUĞU:
1. Birebir aynı veya çok büyük ölçüde örtüşen iş problemine hizmet eden modellere %80-100 arası yüksek puan ver.
2. Belirli fonksiyonları veya kullanım alanları (domain) kesişen, tematik olarak benzer modellere %40-75 arası orta düzey puanlar ver. (Örn: Biri kredi kartı sahteciliği, diğeri POS sahteciliği ise %60-70 civarı).
3. Çok ufak tefek mantıksal veya yöntemsel benzerlikleri olan modellere %15-35 arası düşük puanlar ver.
4. Hiçbir şekilde ilgisi olmayan modellere %0-10 arası puan ver.
ÖNEMLİ KURAL: Lütfen sadece 0 ve 95 gibi uç değerler kullanmaktan kaçın. Benzerlik seviyesine göre 10, 25, 45, 60, 80 gibi ara değerleri ve geniş bir puan yelpazesini kullan.

HIZ VE TOKEN OPTİMİZASYONU:
- Tüm modeller için çok kısa, 1-2 cümlelik (en fazla 10-15 kelime) gerekçeler yaz. "Kullanım alanları farklı", "Kısmi örtüşme var", "Aynı amaca hizmet ediyor" gibi net ifadeler kullan.

Yanıtın KESİNLİKLE aşağıdaki JSON formatında geçerli bir JSON objesi olmalıdır. JSON harici hiçbir metin yazma:
{{
  "sonuclar": [
    {{"Model_ID": "MOD-1", "skor": 45, "gerekce": "Kısa ve net Türkçe açıklama..."}},
    ... (TÜM modeller için eksiksiz liste)
  ]
}}
"""


def _parse_llm_response(raw_text: str, inventory_df: pd.DataFrame) -> pd.DataFrame:
    """LLM'den dönen ham JSON metnini parse edip envanter ile birleştirir.

    Args:
        raw_text: Groq API'den alınan ham JSON string.
        inventory_df: Orijinal envanter DataFrame'i.

    Returns:
        Benzerlik skoru ve LLM gerekçesi eklenmiş, azalan sırada
        sıralanmış DataFrame.
    """
    json_data = json.loads(raw_text)
    json_results = json_data.get("sonuclar", [])
    llm_df = pd.DataFrame(json_results)

    # Orijinal DataFrame ile Model_ID bazlı birleştirme
    merged_df = pd.merge(inventory_df, llm_df, on="Model_ID", how="left")

    # Eksik değerleri varsayılanlarla doldur
    merged_df["skor"] = merged_df["skor"].fillna(0).astype(int)
    merged_df["gerekce"] = merged_df["gerekce"].fillna("LLM bu modeli değerlendiremedi.")

    # UI ile uyumlu sütun adlarına çevir
    merged_df = merged_df.rename(
        columns={"skor": "Benzerlik_Skoru", "gerekce": "LLM_Gerekçe"}
    )
    merged_df = merged_df.sort_values("Benzerlik_Skoru", ascending=False).reset_index(drop=True)
    return merged_df


def compute_llm_similarity(query: str, inventory_df: pd.DataFrame) -> pd.DataFrame:
    """Groq (Llama 3) tabanlı mantıksal benzerlik analizi yapar (Toplu İşlem).

    Tüm envanter modelleri tek bir prompt içinde LLM'e gönderilir;
    dönen JSON yanıtı parse edilerek benzerlik skoru ve gerekçe
    sütunları DataFrame'e eklenir.

    Args:
        query: Kullanıcının girdiği yeni model talep açıklaması.
        inventory_df: Envanter DataFrame'i.

    Returns:
        Benzerlik skoru ve LLM gerekçesi eklenmiş, azalan sırada
        sıralanmış DataFrame. API hatası durumunda tüm skorlar 0
        olarak döner.

    Raises:
        ValueError: LLM_API_KEY tanımlı değilse.
    """
    if not LLM_API_KEY:
        raise ValueError("LLM_API_KEY bulunamadı. Lütfen config.py dosyasını kontrol ediniz.")

    client = Groq(api_key=LLM_API_KEY)
    prompt = _build_llm_prompt(query, inventory_df)

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that strictly outputs JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw_text = response.choices[0].message.content.strip()
        return _parse_llm_response(raw_text, inventory_df)

    except Exception as e:
        st.error(f"LLM API Hatası: {str(e)}")
        # Uygulama çökmesin diye hata durumunda boş skorlu DataFrame dön
        fallback_df = inventory_df.copy()
        fallback_df["Benzerlik_Skoru"] = 0
        fallback_df["LLM_Gerekçe"] = "API Hatası."
        return fallback_df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ana Dispatcher Fonksiyonu
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_DISPATCH: dict[str, callable] = {
    "text": compute_text_similarity,
    "embedding": compute_embedding_similarity,
    "llm": compute_llm_similarity,
}


def analyze_similarity(
    query: str,
    inventory_df: pd.DataFrame,
    method: str = "text",
) -> pd.DataFrame:
    """Seçilen metoda göre uygun benzerlik analiz fonksiyonunu çağırır.

    Args:
        query: Yeni model talep açıklaması.
        inventory_df: Model envanteri DataFrame'i.
        method: Karşılaştırma metodu (``"text"``, ``"embedding"``, ``"llm"``).

    Returns:
        Skorlanmış ve azalan sırada sıralanmış DataFrame.

    Raises:
        ValueError: Geçersiz metot seçimi durumunda.
    """
    method = method.lower().strip()

    if method not in _DISPATCH:
        raise ValueError(
            f"Geçersiz karşılaştırma metodu: '{method}'. "
            f"Desteklenen metodlar: {list(_DISPATCH.keys())}"
        )

    return _DISPATCH[method](query, inventory_df)
