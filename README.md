# 🏦 Kuveyt Türk AI Lab — Model Envanteri Benzerlik Analiz Sistemi

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.56.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Groq-Llama%203.3%2070B-00AA55?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Kuveyt%20T%C3%BCrk-AI%20Lab-16a086?style=for-the-badge"/>
</p>

---

## Yönetici Özeti (Executive Summary)

Büyük ölçekli finansal kuruluşlarda, birden fazla iş birimi tarafından bağımsız olarak başlatılan model geliştirme projeleri, farkında olmadan birbirinin işlevsel kopyasını üretebilmektedir. Bu **mükerrer efor** sorunu; bütçe ve zaman kaybının ötesinde, model yaşam döngüsü yönetimini karmaşıklaştırmakta, regülasyon uyumunu (BDDK, SPK) zorlaştırmakta ve kurumun yapay zeka portföyünü anlamsız biçimde şişirmektedir.

**Model Envanteri Benzerlik Analiz Sistemi**, Kuveyt Türk Yapay Zeka Laboratuvarı tarafından bu problemi çözmek amacıyla geliştirilmiş bir karar destek platformudur. Sistem; yeni bir model geliştirme talebi geldiğinde, bu talebi mevcut **35 kayıtlı modeli** kapsayan canlı envantere karşı üç farklı yapay zeka tekniğiyle otomatik olarak analiz eder ve olası mükerrerlik riskini %0–100 arasında bir skor ile raporlar.

**Temel iş değerleri:**

- ⏱️ Potansiyel mükerrer projeleri ön değerlendirme aşamasında tespit ederek **haftalarca sürebilecek kayıpları önler**
- 💰 Gereksiz model geliştirme bütçelerini elimine ederek **kaynak verimliliğini artırır**
- 📋 Model portföyünü denetlenebilir ve sürdürülebilir kılar, **regülasyon uyumunu kolaylaştırır**
- 🤖 Manuel inceleme yükünü azaltarak AI Lab ekibinin **stratejik işlere odaklanmasını** sağlar

---

## Çözüm Mimarisi — 3 Analiz Case'i

Sistem, farklı senaryolara ve kaynak kısıtlarına göre üç bağımsız benzerlik motoru sunar. Her motor, `analyze_similarity()` dispatcher fonksiyonu üzerinden tek bir arayüzden çağrılır.

### Case 1 — 📝 Text: Fuzzy String Matching

| Özellik | Detay |
|---|---|
| **Kütüphane** | `thefuzz` |
| **Algoritma** | Token Set Ratio (Levenshtein tabanlı) |
| **Ağırlıklar** | Model Adı %15 · Model Amacı %85 |
| **Çıktı** | 0–100 arası normalize benzerlik skoru |

**Çalışma Mantığı:** Her envanter kaydının *Model Adı* ve *Model Amacı* alanları, kullanıcı sorgusuna karşı kelime sırası bağımsız (token set) Levenshtein mesafesi hesabıyla karşılaştırılır. İki ayrı skor ağırlıklı ortalama ile birleştirilir.

**Avantajları:** Sıfır infrastructure maliyeti, milisaniye düzeyinde yanıt süresi, internet bağlantısı gerektirmez.

**Kısıtları:** Yalnızca yüzeysel metin benzerliğini ölçer. "Temerrüt tahmini" ile "geri ödeme riski" gibi farklı kelimelerle ifade edilen eşdeğer kavramları yakalayamaz.

**Önerilen Senaryo:** Hızlı ilk tarama, düşük kaynak ortamları, yüksek hacimli toplu analiz.

---

### Case 2 — 🧠 Embedding: Semantik Benzerlik

| Özellik | Detay |
|---|---|
| **Kütüphane** | `sentence-transformers` |
| **Model** | `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB) |
| **Algoritma** | Kosinüs benzerliği (Cosine Similarity) |
| **Ağırlıklar** | Model Adı %15 · Model Amacı %85 |
| **Çıktı** | 0–100 arası normalize benzerlik skoru |

**Çalışma Mantığı:** Sorgu ve her envanter kaydının Model Adı ile Model Amacı alanları ayrı ayrı yüksek boyutlu vektörlere (embedding) dönüştürülür. Vektörler arasındaki açı (kosinüs benzerliği) hesaplanarak anlamsal yakınlık ölçülür. Çok dilli model olduğu için Türkçe metinleri yerel bağlamda doğru temsil eder.

**Avantajları:** Farklı sözcüklerle ifade edilen semantik eşdeğerlikleri yakalar. Kelime bazlı eşleşmelerin gözden kaçırdığı kavramsal benzerlikleri tespit eder. Model ilk yüklemeden sonra tamamen lokal çalışır.

**Kısıtları:** İlk yüklemede ~470 MB model indirmesi ve 1–2 dakika bekleme süresi gerektirir. Saf metin karşılaştırmasına göre daha yüksek işlemci/bellek tüketimi.

**Önerilen Senaryo:** Derin anlamsal analiz, farklı departmanlardan gelen terminolojik çeşitlilik içeren talepler.

---

### Case 3 — 🤖 LLM: Büyük Dil Modeli Tabanlı Analiz

| Özellik | Detay |
|---|---|
| **Sağlayıcı** | Groq Cloud API |
| **Model** | `llama-3.3-70b-versatile` |
| **Strateji** | Toplu (batch) prompt — tek API çağrısı, tüm envanter |
| **Çıktı Formatı** | JSON (`sonuclar: [{Model_ID, skor, gerekce}]`) |
| **Ağırlıklar** | Model Amacı %90 · Model Adı %10 (prompt kuralı) |

**Çalışma Mantığı:** Tüm 35 envanter kaydı tek bir yapılandırılmış prompt içinde Llama 3.3 70B modeline iletilir. Model; iş fonksiyonları, hedef kitleler ve metodolojik yaklaşımları karşılaştırarak her kayıt için 0–100 arası bir skor ve kısa Türkçe gerekçe üretir. Yanıt `json_object` modunda alınarak parse edilir.

**Avantajları:** İki modelin yalnızca kelimelerini değil, gerçek iş fonksiyonlarını ve bağlamını karşılaştırır. Puanlama gerekçeleri sayesinde kararlar insan tarafından denetlenebilir (explainability). Üç yöntem arasında en yüksek doğruluk potansiyeline sahiptir.

**Kısıtları:** Groq API anahtarı ve internet bağlantısı gerektirir. Diğer yöntemlere kıyasla en yüksek yanıt süresi (~3–8 saniye).

**Önerilen Senaryo:** Yönetim onayına sunulacak kritik kararlar, stratejik model portföyü değerlendirmeleri, audit süreçleri.

---

### Yöntem Karşılaştırma Tablosu

| Kriter | Text | Embedding | LLM |
|---|:---:|:---:|:---:|
| Hız | ⚡ Çok Hızlı | 🔄 Orta | 🐢 Yavaş |
| Doğruluk | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Infrastructure Maliyeti | 🟢 Sıfır | 🟡 Orta | 🔴 API Maliyeti |
| Açıklanabilirlik | 🟡 Düşük | 🟡 Düşük | 🟢 Yüksek |
| Offline Çalışma | ✅ | ✅ | ❌ |
| Çok Dilli Destek | 🟡 Kısmi | ✅ | ✅ |

---

## Teknoloji Yığını

| Katman | Teknoloji | Versiyon | Rol |
|---|---|---|---|
| **Dil** | Python | 3.12+ | Uygulama dili |
| **UI Framework** | Streamlit | 1.56.0 | Web arayüzü |
| **Fuzzy Matching** | thefuzz | 0.22.1 | Case 1 motoru |
| **Semantik Model** | sentence-transformers | 3.4.1 | Case 2 motoru |
| **Derin Öğrenme** | PyTorch | 2.11.0 | Embedding altyapısı |
| **LLM API** | Groq (Llama 3.3 70B) | — | Case 3 motoru |
| **Veri İşleme** | pandas / numpy | 2.2.3 / — | DataFrame operasyonları |
| **Konfigürasyon** | Streamlit Secrets | — | Güvenli API yönetimi |

---

## Proje Yapısı

```
ModelInventorySystem/
│
├── app/                          # Ana uygulama paketi
│   ├── __init__.py               # Paket tanım dosyası
│   ├── config.py                 # Merkezi konfigürasyon ve sabitler
│   ├── similarity.py             # 3 benzerlik motoru + dispatcher
│   └── main.py                   # Streamlit arayüzü (UI katmanı)
│
├── data/
│   └── inventory.csv             # 35 kayıtlı model envanteri
│
├── .streamlit/
│   ├── config.toml               # Streamlit tema ve sunucu ayarları
│   ├── secrets.toml              # API anahtarları — Git'e eklenmez
│   └── secrets.toml.example      # API anahtar şablonu
│
├── requirements.txt              # Python bağımlılıkları
└── README.md                     # Bu dokümantasyon
```

**Modül sorumlulukları:**

- **`config.py`** — `COMPARE_METHOD`, `THRESHOLD`, `EMBEDDING_MODEL`, `LLM_MODEL` gibi tüm sistem parametrelerini ve `_resolve_llm_api_key()` ile güvenli API anahtar çözümlemesini barındırır.
- **`similarity.py`** — `compute_text_similarity()`, `compute_embedding_similarity()`, `compute_llm_similarity()` motorlarını ve `analyze_similarity()` dispatcher'ını içerir. UI katmanından tamamen bağımsızdır.
- **`main.py`** — Streamlit sayfa yapılandırması, CSS tema, sidebar kontrolleri, girdi formu ve sonuç render mantığını yönetir. Hiçbir iş mantığı içermez.

---

## Kurulum ve Çalıştırma

### Gereksinimler

- Python **3.12** veya üzeri
- `pip` paket yöneticisi

### Adım 1 — Depoyu Klonlayın

```bash
git clone https://github.com/EnginAytac/ModelInventorySystem.git
cd ModelInventorySystem
```

### Adım 2 — Sanal Ortam Oluşturun

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Adım 3 — Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

> ⚠️ `sentence-transformers` ilk kurulumda PyTorch ile birlikte yaklaşık **1–2 GB** indirir. İnternet bağlantınızın stabil olduğundan emin olun.

### Adım 4 — LLM API Anahtarı (Opsiyonel)

**Text** ve **Embedding** modları API anahtarı gerektirmez. **LLM** modunu kullanmak için:

```bash
# .streamlit/secrets.toml.example dosyasını kopyalayın
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

`.streamlit/secrets.toml` dosyasını açıp Groq API anahtarınızı girin:

```toml
LLM_API_KEY = "gsk_BURAYA_KENDI_ANAHTARINIZI_YAZIN"
```

> Groq API anahtarını [console.groq.com](https://console.groq.com) adresinden ücretsiz alabilirsiniz.

### Adım 5 — Uygulamayı Başlatın

```bash
streamlit run app/main.py
```

Uygulama tarayıcınızda otomatik açılacaktır: **`http://localhost:8501`**

---

## Kullanım Kılavuzu

1. Sol panelden **Karşılaştırma Metodunu** seçin (Text / Embedding / LLM)
2. **Benzerlik Eşiğini** belirleyin (varsayılan: %70 — bu eşiğin üzerindeki modeller mükerrer risk taşıyan olarak işaretlenir)
3. Ana ekrandaki metin alanına yeni model talebinizin açıklamasını girin
4. **"Analiz Et"** butonuna tıklayın
5. Sistem; eşiği aşan modelleri **⚠️ Mükerrer Risk Tespit Edildi** bölümünde kart formatında listeler
6. Tüm sonuçları **📊 Tüm Analiz Detaylarını Göster** bölümündeki interaktif tablodan inceleyebilirsiniz

---

## Konfigürasyon Referansı

`app/config.py` dosyasından merkezi parametreler yönetilir:

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `COMPARE_METHOD` | `"text"` | Varsayılan karşılaştırma motoru |
| `THRESHOLD` | `70` | Mükerrer risk eşiği (%) |
| `EMBEDDING_MODEL` | `"paraphrase-multilingual-MiniLM-L12-v2"` | Çok dilli semantik model |
| `LLM_MODEL` | `"llama-3.3-70b-versatile"` | Groq üzerindeki LLM modeli |
| `MAX_RESULTS` | `10` | Arayüzde gösterilecek maksimum sonuç sayısı |
| `INVENTORY_PATH` | `"data/inventory.csv"` | Envanter veri dosyası yolu |

---

## ☁️ Streamlit Cloud Canlı Dağıtımı

Uygulama **Streamlit Cloud** üzerinde çalışacak şekilde yapılandırılmıştır. Cloud ortamında API anahtarı Streamlit Secrets kasasından güvenli biçimde okunur; kullanıcının herhangi bir ek ayar yapmasına gerek kalmaz.

Canlı sürümde **LLM Analiz** özelliği doğrudan kullanılabilir durumdadır.

---

## Lisans

Bu proje, **Kuveyt Türk Yapay Zeka Laboratuvarı** bünyesinde kurumsal iç kullanım ve değerlendirme amacıyla geliştirilmiştir. İzinsiz dağıtım ve ticari kullanım yasaktır.

---

<p align="center">
  <sub>© 2026 Kuveyt Türk AI Lab · Model Envanter Yönetim Sistemi v1.0</sub>
</p>
