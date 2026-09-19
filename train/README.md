# Train üretimi: Gemini 3.8 Flash + iki bağımsız judge

Generator, Luna ve GLM düşük reasoning ile çalışır. Semantik judge GPT-5.6 Luna
Flex'tir; normal Luna ile aynı modeldir, daha düşük öncelikli/değişken gecikmeli
servis karşılığında daha düşük fiyat hedeflenir.
Gemini için ucuz/Flex fiyat tavanı $0.375/M girdi ve $1.875/M çıktıdır;
uygun endpoint yoksa pahalı standart sağlayıcıya sessiz geçilmez.
GLM endpoint'i reasoning kapatmayı desteklemez (API 400 döndürür).
GLM sağlayıcı seçiminde Wafer dışlanır; hız önceliklidir ve fiyat üst sınırı
$0.15/M girdi, $0.50/M çıktıdır. Negatiflerin ilgisizliği judge hatası sayılmaz;
bağımsız relevant_ids amaçlanan positive ile ayrıca karşılaştırılır.

Testten bağımsız Python/config/veri hattı. `test` modülleri import edilmez ve test
dosyalarına yazılmaz. Test JSONL'leri yalnız **yerel dışlama indeksi** için okunur;
test query/adayları hiçbir generator veya judge promptuna gönderilmez.

## Sırasıyla mimari

1. **Prepare:** 600 korunan family'den metin/kök/şablon/ek-zinciri dışlama snapshot'ı.
2. **Plan:** tekrar üretimde değişmeyen slotlar; aynı kota özellikleri korunur.
3. **Approve:** insan kontrolünün bittiği test sürümü checksum + isim ile onaylanır.
4. **Gemini:** ortak olay bilgisi, query, tek ortak bağlam ve dört kritik aday cümlesi üretir; pasajları Python birleştirir.
5. **Yerel guard:** schema, metin/kök sızıntısı, soru/bildirim ve strict minimal-pair kontrolü.
6. **Paralel iki judge:** Luna semantik turda etiketler gizli; GLM morfoloji turunda slot amaçlarını denetler.
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
senkronizasyon/import yoktur. **Train-v4 kök–zincir birleşimi holdout** kullanır.
18 kompozisyon hedefinin tamamı farklı köklerle train'de kullanılabilir; 14 zincirin
tamamen yasaklanması kaldırılmıştır. Katalogdaki 76 hedef plan için kullanılabilir;
slotların %30'u kompozisyon, %70'i diğer hedeflerdir (küçük pilotlarda yuvarlama).
Korunan aynı kritik kök + aynı zincir hedefi birleşimi yasaktır. Aynı kökün başka
zincir/hedefte, aynı zincirin başka kökte kullanılması serbesttir; ayrı lemma-holdout
kökleri ve metin/kopya kontrolleri korunur. Yerel filtre kritik lemma ve yüzey-kök
heuristiğiyle, morfoloji judge'ı tüm metin üzerinden çiftleri denetler. Ek metadata'sı
olmayan adaylar korunan family'nin zincir hedefi altında muhafazakâr biçimde indekslenir;
bu tam morfolojik çözümleme veya tüm yüzey varyantları için eksiksiz koruma değildir.
Eski snapshot/planlar yeni politikayla devam ettirilmez; yeniden prepare gerekir.
Ham test metinleri LLM'lere gönderilmez. Snapshot hazırlama ve kaynak checksum kontrolü
yerelde test dosyalarını okur; bu, tamamen test dosyasına erişimsiz bir hat değildir.
Bu değişiklik sealed test dosyalarını/split'lerini değiştirmez; eski composition_holdout
etiketleri tarihsel plan metadata'sıdır. Bu train ile görülen zincirler için paper'da
"unseen-chain zero-shot" iddiası kurulmaz. Kök–zincir grupları ayrıca doğrulanıp raporlanır.
Kota train'e aittir; testin aday sayısı veya split'i bu kod tarafından değiştirilmez.
Şu an bu hat ayrı validation üretmez. Yalnız açıkça development olarak ayrılmış veriyle
ayar yapılmalıdır; bütün 600 sealed ise eğitim deneyinden önce ayrıca bağımsız validation
planlanmalıdır. Sealed örneklerle hiperparametre seçilmez.

## Komutlar (repo kökünden)

İnsan kontrolü bitmeden maliyet/süre denemesi için `prepare --pilot` kullanılabilir.
Sızıntı filtreleri ve iki judge değişmez; insan onayı uydurulmaz. Manifest ve export
`purpose=pilot_only`, `eligible_for_final_train=false` taşır. Pilot çıktısı final
train'e eklenmez; final üretim için onaylı kaynakla ayrı run gerekir.

### Eski pilot ölçümü (18 Eylül 2026)

Pilot veri klasörleri ve yerel SQLite/cache kayıtları kullanıcı isteğiyle silinmiştir;
bu bölüm yalnız geçmiş koşunun ölçüm özetidir, mevcut veri dosyası veya kalite onayı değildir.
Beş kabul için toplam 208,49 saniye / $0,050287 harcandı; düzeltmeler ve iki
tükenen slotun maliyeti dahildir. Gemini 3.8 Flash çağrılarının tamamı Flex idi.
Bu küçük pilotun doğrusal 1.000-kabul tahmini yaklaşık $10,06 ve sıralı 11,58 saattir;
fiyat/sağlayıcı/ret oranına bağlıdır, garanti veya kalite onayı değildir.
Okumada katılımcı/zaman/yer kaymaları bulundu: otomatik kabul edilmiş bu pilot
final train'e alınmaz. Büyük üretimden önce positive anlam koruması ve morfolojik
negatiflerin hedef dışı içerik değişimleri güçlendirilmelidir.
19 Eylül'de yapılan sonraki v5 pilotu da v6 sözleşmesinden öncedir ve final train'e
uygun değildir. V6 kalite/hız ölçümü yeni run-id ile ayrıca yapılmalıdır.

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

### Train-v7 kalite ve paralellik

`family_workers=3`: en fazla üç family eşzamanlı ilerler; her biri ayrı generator
çağrısıdır. Her family'nin iki judge'ı paralel çalışır (en fazla altı judge isteği).
Global çağrı bütçesi kilitle paylaşılır; `--limit` kadar yeni kabul sınırı aşılmaz.
SQLite event'leri slot kimliğini korur. Kabulden hemen önce kilitli duplicate kontrolü
tekrarlanır; eşzamanlı benzer çıktılar birlikte kabul edilmez. Run/global üretim
kilitleri korunur. Dalga en fazla üç slot içerir; yavaş slot bir sonraki dalgayı bekletir.

Generator `event_frame` içinde katılımcılar, nesne, olay, yer, zaman ve sonucu bir kez
yazar. `context_sentences` ve sıfır tabanlı `critical_position` Python'a aktarılır;
aday başına yalnız kritik cümle, lemma/sözcük ve morph negatiflerde `morph_change`
(feature/from/to) üretilir. Ortak bağlam tüm adaylarla uyumlu, nötr olmalıdır.
Üretici `text` tekrarı yazmaz; Python çıktı ve patch sonrasında metni yeniden kurar.

Her judge dört aday için kısa `candidate_checks` ve metinden kanıt döndürür.
Semantik judge altı olay alanını doğrudan query/aday metinlerinden karşılaştırır;
generator'ın event_frame'i ve gold etiketi bu judge'a verilmez. `relevant_ids` ile
alan sonuçları çelişirse veya eksik/null alan varsa otomatik kabul edilmez.
Morfoloji judge target_valid/natural/content_preserved alanlarını kontrol eder;
genel pass yazsa bile false kontrol sonucu fail'e dönüştürülür. Üreticinin
morph_change açıklaması kanıt sayılmaz. Bunlar model değerlendirmeleridir; gerçek
semantik doğruluk ayrıca pilotla ölçülmelidir.

Üretim sırası query → anlamı koruyan positive → positive'dan iki morfolojik karşıt
→ ayrı içerik negatifi şeklindedir. Query–positive kopyası yerel filtreyle engellenir.
Positive ve iki morph-hard aynı nötr bağlam cümlelerini kullanır; yerel filtre bağlam
değişimini reddeder. Strict modda positive–morph_1 aynı lemma ve kritik sözcük dışında
aynı normalize cümle şablonunu kullanır. Diğer modlarda ifade çeşitliliği korunur.
Bu kontroller zaman/katılımcı/anlam eşdeğerliğini tek başına kanıtlamaz.

Semantik judge etiketleri görmeden ilgili adayı seçer; Python amaçlanan positive ile
karşılaştırır. Morfoloji judge slotları görür, fakat bunları doğru kabul etmez: positive
hedefi, her morph-hard'ın hedef işlev farkını, doğallığını ve hedef dışı içerik değişimini
denetler. Böylece içerik negatifi ile yanlış üretilmiş morph-hard ayrılır. Bu tur kör
relevance oylaması değildir. İki judge mevcut çağrılarında bu kontrolleri yapar;
ek judge veya ek zorunlu API turu eklenmemiştir. Generator low/Flex olarak kalır.
Eski pilot kalite onayı sayılmaz; yeni kurallar gerçek pilotla ayrıca ölçülmelidir.
Kod sözleşmesi değiştiğinden eski run sessizce devam etmez, yeni prepare gerekir.

- İkisi de **pass** → kabul; confidence tek başına ret nedeni değildir.
- Herhangi biri somut adaya bağlı **fail, confidence ≥80** → yalnız ilgili adayı düzelt.
- Hata bildiren düşük güvenli karar, abstain, eksik/çelişkili rapor → sınırlı tekrar, otomatik kabul yok.
- **En fazla 2 judge turu / 1 düzeltme**, ardından ret.
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
Hedef-lemma holdout yalnız query/aday kritik lemma metadata'sında ve kritik sözcük
yüzeyinde aranır (4+ karakter kökler için prefix; kısa köklerde exact). Yan bağlam
sözcükleri yasak değildir: bu katı corpus-wide lemma-disjointlik iddiası değildir.
Kök listesi yalnız testin lemma_holdout grubundan alınır. Bu tarama gerçek morfolojik
çözümleyici değildir; yanlış ret veya kaçırma olabilir. Lemma/şablon/zincirin gerçek
metindeki kullanımı ayrıca GLM'e denetletilir. IDs'yi farklı yazmak tek başına disjointlik
kanıtı değildir; tamamen farklı kelimeli anlamsal kopyaları bu filtreler garantiyle yakalamaz.
**"Sıfır leakage garantisi" iddiası yoktur.**

Schema, korunan metin/kökler ve doğru kritik cümle tipi zorunludur. Cümle sayısı,
uzunluk oranı üretim hedefidir, ret filtresi değildir. Strict positive–morph_1 için
aynı lemma ve hedef sözcük dışındaki aynı şablon yerel filtreyle zorunludur;
diğer modlarda tek-token edit zorunlu değildir.
Human-review/uyarı kuyruğu yoktur. Generator lemma açıklamaları
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

Gemini 3.8 Flash low/Flex; GPT-5.6 Luna low/Flex; GLM 5.3 Flash low.
`exclude:true` reasoning'i kapatmaz, görünür yanıttan çıkarır. `sort:price` en ucuz uygun
sağlayıcıyı önceliklendirir; fallback açık. Generator için milyon token başına $0.75 giriş /
$3.75 çıkış; Luna için $0.10 giriş / $0.60 çıkış tavanı vardır. Flex ile async Batch
farklıdır; bu hat senkron normal API isteğini düşük öncelikli Flex servisinde çalıştırır.
Desteklenmeyen parametre sessizce atılmasın diye `require_parameters:true` kullanılır.

Mantıksal çağrı başına en fazla 3 teknik deneme vardır; `length` token bütçesini en fazla
4 katına çıkarabilir. `--max-calls 30` dolayısıyla en fazla 90 sağlayıcı denemesidir,
30 dolar veya 30 token değildir. Kaynak kod/API canlı pilotta test edilmeden süre veya
kabul oranı taahhüt edilmez. Anahtar hiçbir rapora yazılmaz.

Referans: https://openrouter.ai/docs/guides/routing/provider-selection
