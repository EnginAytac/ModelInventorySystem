# 🏦 Kuveyt Türk AI Lab — Model Envanteri Benzerlik Analiz Sistemi

## Proje Amacı

Bu sistem, **Kuveyt Türk Yapay Zeka Laboratuvarı** bünyesinde geliştirilen ve planlanan makine öğrenmesi / yapay zeka modellerinin envanterini yöneterek, yeni model taleplerinin mevcut envanterdeki modellerle **mükerrer olup olmadığını** otomatik olarak tespit etmeyi amaçlamaktadır.

Banka içindeki farklı birimlerden gelen model geliştirme talepleri zaman zaman birbirinin tekrarı olabilmektedir. Bu durum:

- **Kaynak israfına** (zaman, bütçe, insan kaynağı)
- **Model yönetim karmaşıklığına**
- **Regülasyon uyum risklerine**

yol açmaktadır. Bu sistem, "Vibe Coding" prensiplerine uygun, modüler ve modern bir mimariyle bu problemi çözer.

---

## Mimari ve Kullanılan Yöntemler

Sistem, 3 farklı benzerlik analiz yöntemi sunar. Her yöntem farklı bir ihtiyacı karşılamak üzere tasarlanmıştır:

### 1. 📝 Text — Fuzzy String Matching

- **Kütüphane:** `thefuzz` (python-Levenshtein destekli)
- **Yöntem:** Token Set Ratio
- **Neden:** Hızlı ve basit karşılaştırmalar için idealdir. Kelime sırası bağımsız çalışır, küçük yazım farklarını tolere eder. Deployment maliyeti sıfırdır.
- **Kullanım Alanı:** İlk tarama, hızlı prototipleme, düşük kaynak ortamları.

### 2. 🧠 Embedding — Semantik Benzerlik

- **Kütüphane:** `sentence-transformers` (all-MiniLM-L6-v2)
- **Yöntem:** Kosinüs benzerliği (cosine similarity)
- **Neden:** Kelimelerin anlamsal yakınlığını yakalayarak, farklı kelimelerle ifade edilmiş benzer kavramları tespit eder. "Kredi riski" ile "temerrüt olasılığı" gibi semantik eşleşmeleri bulabilir.
- **Kullanım Alanı:** Derin analiz, anlamsal örtüşme tespiti.

### 3. 🤖 LLM — Büyük Dil Modeli Tabanlı Analiz

- **Yöntem:** Yapılandırılmış prompt ile fonksiyonel karşılaştırma
- **Neden:** İki modelin iş fonksiyonlarını, hedef kitlelerini ve metodolojilerini mantıksal olarak karşılaştırabilir. En yüksek doğruluk potansiyeline sahiptir ancak API maliyeti ve latency'si yüksektir.
- **Mevcut Durum:** Mock/simülasyon olarak çalışır. OpenAI veya Anthropic API anahtarı eklenerek gerçek entegrasyona geçilebilir.
- **Kullanım Alanı:** Kritik kararlar, yönetim raporları, final onay süreci.

---

## Dizin Yapısı

```
```text
model_inventory_system/
├── app/
│   ├── __init__.py        # Paket tanımı
│   ├── main.py            # Streamlit arayüzü
│   ├── similarity.py      # Benzerlik algoritmaları
│   └── config.py          # Konfigürasyon ve ayarlar
├── data/
│   └── inventory.csv      # Sentetik model envanteri (35 kayıt)
├── requirements.txt       # Python bağımlılıkları
└── README.md              # Bu dosya
```

---

## Kurulum ve Çalıştırma

### Gereksinimler

- Python 3.10+
- pip

### Adımlar

```bash
# 1. Proje dizinine gidin
cd ModelInventorySystem

# 2. Sanal ortam oluşturun (önerilir)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. (Opsiyonel) LLM API Kurulumu
# LLM tabanlı analizi bilgisayarınızda test etmek isterseniz:
# `.streamlit/secrets.toml.example` dosyasının adını `.streamlit/secrets.toml` olarak değiştirin
# ve içerisine Groq API anahtarınızı yazın.

# 5. Uygulamayı başlatın
streamlit run app/main.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde açılacaktır.

---

## ☁️ Streamlit Cloud Canlı Sürümü
Uygulamayı indirmeden test etmek isterseniz, canlı Streamlit Cloud versiyonunda API anahtarı sisteme gömülü olduğu için ekstra bir ayar yapmadan **LLM Analiz** özelliğini kullanabilirsiniz.

---

## Kullanım

1. Sol paneldeki **Analiz Ayarları**'ndan karşılaştırma metodunu ve benzerlik eşiğini seçin.
2. Ana ekrandaki metin alanına yeni model talebinizin açıklamasını girin.
3. **"🔍 Analiz Et"** butonuna tıklayın.
4. Sonuçlar, eşik değerinin üzerindeki modelleri **"Mükerrer Risk Taşıyan Modeller"** bölümünde kart formatında gösterir.
5. Tüm sonuçları "📊 Tüm Analiz Sonuçlarını Görüntüle" bölümünden inceleyebilirsiniz.

---

## Konfigürasyon

`app/config.py` dosyasından aşağıdaki parametreler değiştirilebilir:

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `COMPARE_METHOD` | `"text"` | Varsayılan karşılaştırma yöntemi |
| `THRESHOLD` | `70` | Mükerrer risk eşiği (%) |
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | Semantik analiz modeli |
| `MAX_RESULTS` | `10` | Maksimum gösterilen sonuç sayısı |

---

## Teknoloji Yığını

| Katman | Teknoloji |
|--------|-----------|
| UI Framework | Streamlit |
| Fuzzy Matching | thefuzz + python-Levenshtein |
| Semantik Analiz | sentence-transformers |
| Veri İşleme | pandas, numpy |
| Dil | Python 3.10+ |

---

## Lisans

Bu proje Kuveyt Türk Yapay Zeka Laboratuvarı bünyesinde iç kullanım amacıyla geliştirilmiştir.

---

*Kuveyt Türk AI Lab — 2026*
