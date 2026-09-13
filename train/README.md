# Train üretimi: Gemini 3.8 Flash + iki bağımsız judge

Testten bağımsız Python/config/veri hattı. `test` modülleri import edilmez ve test
dosyalarına yazılmaz. Test JSONL'leri yalnız **yerel dışlama indeksi** için okunur;
test query/adayları hiçbir generator veya judge promptuna gönderilmez.

## Sırasıyla mimari

1. **Prepare:** 600 korunan family'den metin/kök/şablon/ek-zinciri dışlama snapshot'ı.
2. **Plan:** tekrar üretimde değişmeyen slotlar; aynı kota özellikleri korunur.
3. **Approve:** insan kontrolünün bittiği test sürümü checksum + isim ile onaylanır.
4. **Gemini:** bir query, positive, iki morfolojik hard ve bir semantik hard üretir.
5. **Yerel guard:** schema, metin/kök sızıntısı, uzunluk, soru/bildirim, minimal-pair kontrolü.
6. **Paralel iki judge:** DeepSeek semantik, GLM morfoloji. Gold/slot etiketleri gizlenir.
7. **Karar:** kabul / aday düzeltme / aynı kotada yeni family / teknik erteleme.
8. **SQLite → JSONL:** güvenli devam, provenance, maliyet ve eğitim görünümü.

Kodun çevrimdışı testleri vardır; gerçek model kalitesi/maliyet/süre ancak ücretli pilotta
ölçülür. Fine-tuning/encoder eğitimi bu üretim hattının parçası değildir.

## Varsayılan 1.000 train planı

| Boyut | Dağılım |
|---|---|
| Adaylar | 1 positive + 2 morph hard + 1 semantic hard |
| Yapı | 400 strict minimal + 300 controlled diverse + 300 natural retrieval |
| Query | 750 tek cümle + 250 iki cümle |
| Aday metni | 300/300/300/100 family için 1/2/3/4 cümle |
| Fenomen | Dışlama sonrası uygun hedefler arasında yaklaşık eşit |

`catalog.json` 76 fenomenin bağımsız, sürümlü train kopyasıdır; test koduyla otomatik
senkronizasyon/import yoktur. Mevcut test snapshot'ı **18 composition_holdout** hedefini
dışlar; train'de 58 hedef kalır. Zincirin bileşenleri uygun tekil hedeflerde öğrenilir,
saklanan tam zincirler train'e alınmaz. Yeni zincir eklemek ayrı bir plan kararıdır.
Kota train'e aittir; testin aday sayısı veya split'i bu kod tarafından değiştirilmez.
Şu an bu hat ayrı validation üretmez. Yalnız açıkça development olarak ayrılmış veriyle
ayar yapılmalıdır; bütün 600 sealed ise eğitim deneyinden önce ayrıca bağımsız validation
planlanmalıdır. Sealed örneklerle hiperparametre seçilmez.

## Komutlar (repo kökünden)

```bash
# API yok: mevcut 600'den taslak plan çıkarır, hiçbir insan onayı uydurmaz.
python3 train/workflow.py prepare --run-id pilot1000 --source test/data/final_shards --size 1000
python3 train/workflow.py status --run-id pilot1000

# YALNIZ insan kontrolü gerçekten bitince. Checksum prepare çıktısındadır.
python3 train/workflow.py approve --run-id pilot1000 --source-sha CHECKSUM --reviewer ISIM --confirm-human-review-complete

# ÜCRETLİ. Anahtar OPENROUTER_API_KEY ortam değişkeninde veya repo .env dosyasında.
# En fazla 10 YENİ kabul; en fazla 30 cache-dışı mantıksal LLM çağrısı.
python3 train/workflow.py run --run-id pilot1000 --limit 10 --max-calls 30

# Aynı komutla devam: kabul edilen slotlar atlanır.
python3 train/workflow.py export --run-id pilot1000

# Ücretsiz kontroller
python3 train/production.py --check-config
python3 -m unittest discover -s train -p 'test_*.py' -v
```

Test metninde bozuk karakter varsa approve engellenir. İnsanlar test verisini düzenlerse
eski snapshot geçersizleşir; **yeni run-id ile prepare** gerekir. Sırf devam etmek için
checksum/manifest elle değiştirilmez. Aynı şekilde kaynak kod/config/plan değişirse eski
üretim sözleşmesiyle sessiz devam edilmez. İnsan kontrolü kod tarafından kanıtlanmaz;
approve kaydı sorumlu kişinin açık beyanıdır.

## Judge politikası

- İkisi de **pass, confidence ≥80** → kabul.
- Herhangi biri somut adaya bağlı **fail, confidence ≥80** → yalnız ilgili adayı düzelt.
- Düşük güven, abstain, eksik/çelişkili rapor → sınırlı tekrar, otomatik kabul yok.
- **En fazla 3 judge turu / 2 düzeltme**, ardından ret.
- Reddedilen family yerine **aynı slotta en fazla 3 generation denemesi**. Sonra `exhausted`;
  raporda görünür, kota doldurulmuş sayılmaz. Sonsuz üretim/harcama döngüsü yoktur.
- Query/hedef/diğer adaylar patch ile değişmez. Kritik kelime/lemma/cümle metadata'sı
  düzeltilen metinle birlikte güncellenir; guard ve iki judge bütün family'ye tekrar bakar.
- API kesintisi, boş/kesilmiş cevap veri hatası değildir; `pending` kalır ve devam edilebilir.

Confidence ölçülmüş hata oranı değildir. Human-review bekleme kuyruğu yoktur;
otomatik etiketler kusursuz kabul edilmez, pilot örneklemesi önerilir.

## Sızıntı kontrollerinin kapsamı ve sınırı

Korunan 600'ün query ve bütün adayları ile daha önce kabul edilmiş yerel train verileri
kontrol edilir: normalize exact match, token bigram Jaccard ≥0.65, karakter trigram
Jaccard ≥0.85 veya token sıra benzerliği ≥0.85. Tam pasaj ve tekil cümleler indekslenir.
Lemma holdout hem kritik lemma metadata'sında hem konservatif yüzey taramasında aranır
(4+ karakter kökler için prefix; kısa köklerde exact). Bu tarama gerçek morfolojik
çözümleyici değildir; yanlış ret veya kaçırma olabilir. Lemma/şablon/zincirin gerçek
metindeki kullanımı ayrıca GLM'e denetletilir. IDs'yi farklı yazmak tek başına disjointlik
kanıtı değildir; tamamen farklı kelimeli anlamsal kopyaları bu filtreler garantiyle yakalamaz.
**"Sıfır leakage garantisi" iddiası yoktur.**

Kota kontrollü schema/uzunluk/metin ve doğru cümle tipi zorunludur. Strict minimal
positive–morph_1 aynı lemma, tam bir token değişimi ister. Generator lemma açıklamaları
judge için doğrulanacak iddiadır, ground truth kabul edilmez.

## Dosyalar / kayıt

- `production.py`: API, generator/judge promptları, kabul ve aday düzeltmesi.
- `workflow.py`: plan, guard, SQLite, cache, CLI ve export.
- `production_config.json`, `catalog.json`: bağımsız ayarlar ve fenomen kataloğu.
- `test_*.py`: sentetik fixture'larla offline testler; fixture'lar eğitime yazılmaz.
- `docs/`: eski literatür arşivi, aktif üretim talimatı değil.

Çalışma verileri `train/runs/RUN_ID/` içinde: `manifest.json`, `protected.json`, `plan.json`,
`state.sqlite3`; dışa aktarımda `accepted.jsonl` ve `report.json`. Bu klasör Git ignore'dadır;
otomatik push edilmez. Veriyi paylaşırken bilinçli bir sürümleme/export kararı verilir.
`accepted.jsonl` tek dosyada query/positive/negatives eğitim görünümünü ve ayrıntılı provenance'ı taşır.
SQLite'ta slot durumları, denemeler, ret nedenleri, cache ve bütün çağrı olayları tutulur.

Yerel run'lar arası kabul edilmiş train metinleri de duplicate kontrolüne katılır.
Global yerel kilit iki üreticinin aynı anda kabul yazarak kopya kaçırmasını önler;
family'ler sıralı, **iki judge paralel** çalışır. Bu merkezi çok-makineli DB değildir:
başka bilgisayara aktarırken durdurulmuş run'ın tamamını taşıyın; eşzamanlı ayrı kopyalarda üretmeyin.

Cache aynı tamamlanmış isteği yeniden harcatmaz. API cevabı ile cache commit'i arasındaki
ani kapanmada son istek belirsiz kalabilir; mutlak exactly-once API garantisi yoktur.
`request_started` olayları bu boşluğu raporda gösterir. Bilinmeyen maliyet sıfır sayılmaz.
Rapor yalnız telemetride dönen maliyeti toplar; eksik maliyet sayısını ayrıca verir.

## Provider / süre / maliyet

Gemini 3.8 Flash low; DeepSeek V4 Flash 0731 reasoning kapalı isteği; GLM 5.3 Flash low.
`exclude:true` reasoning'i kapatmaz, görünür yanıttan çıkarır. `sort:price` en ucuz uygun
sağlayıcıyı önceliklendirir; fallback açık. Generator için milyon token başına $0.75 giriş /
$3.75 çıkış tavanı vardır. Flex ile async Batch farklıdır; bu hat normal API kullanır.
Desteklenmeyen parametre sessizce atılmasın diye `require_parameters:true` kullanılır.

Mantıksal çağrı başına en fazla 3 teknik deneme vardır; `length` token bütçesini en fazla
4 katına çıkarabilir. `--max-calls 30` dolayısıyla en fazla 90 sağlayıcı denemesidir,
30 dolar veya 30 token değildir. Kaynak kod/API canlı pilotta test edilmeden süre veya
kabul oranı taahhüt edilmez. Anahtar hiçbir rapora yazılmaz.

Referans: https://openrouter.ai/docs/guides/routing/provider-selection
