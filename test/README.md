# Turkish Morph Retrieval Test — v3.9.0

Bu dizin Türkçe encoder'ların küçük fakat anlam değiştiren morfolojik farkları ayırt edip
etmediğini ölçen benchmark'ı üretir ve değerlendirir. Train sistemi [`../train/`](../train/)
altında ayrıdır; test üretiminde eski JSON cümleleri few-shot örneği olarak kullanılmaz.

## Veri tasarımı

| Özellik | Development | Final test | Toplam |
|---|---:|---:|---:|
| Contrast family/query | 100 | 500 | 600 |
| Aday/family | 11 | 11 | 11 |
| Aday pasaj | 1.100 | 5.500 | 6.600 |
| Strict minimal pair | 25 | 125 | 150 |
| Controlled diverse | 45 | 225 | 270 |
| Natural retrieval | 30 | 150 | 180 |
| Generator A/B | 50/50 | 250/250 | 300/300 |

Her family tam olarak şunları içerir:

- 1 query
- 1 positive/gold
- 8 hard negative
- 2 easy negative
- Otomatik binary qrels: gold `1`, diğer 10 aday `0`

Benchmark üç tamamlayıcı family modu kullanır:

- `%25 strict_minimal`: gold ile `hard_01` yalnız hedef morfolojik biçimde ayrışır.
- `%45 controlled_diverse`: aynı olay ve morfolojik karşıtlık korunur; gold ile hard'lar farklı
  doğal sözdizimleri kullanabilir.
- `%30 natural_retrieval`: query, gold ve negatifler doğal olarak farklı kurulabilir; gold bilgi
  ihtiyacını karşılayan tek passage, hard'lar çekici fakat gerçekten irrelevant kayıtlardır.

Query ifadesi `%50 morph_explicit` ve `%50 semantic_paraphrase` olarak dengelenir. İlk grupta
hedef anlam query'de çekimli biçimle görünür; ikinci grupta farklı sözcük/sözdizimiyle anlatılır.
Query–gold lexical ilişki ayrıca `%30 high`, `%40 medium`, `%30 low` bandına planlanır. Bunlar
“şu kadar benzemek zorunda” kuralı değil, yüzey ve semantik çeşitlilik kotalarıdır.

Uzunluk dağılımı:

- Query: `%75` bir cümle, `%25` iki cümle.
- Pasaj: `%30/%30/%30/%10` oranında 1/2/3/4 cümle.
- Ortak bağlam cümleleri family içindeki 11 adayda aynıdır.
- Yalnız kritik cümle değişir; kritik cümlenin pasaj konumu dengelenir.

Generalizasyon dağılımı:

- `%40 standard`
- `%20 lemma_holdout`
- `%20 template_holdout`
- `%20 composition_holdout`
- Yaklaşık `%20` ek `domain_shift` etiketi

Çoklu generator'lar run'a özel SQLite dataset memory üzerinden koordine edilir. Registry atomik
slot rezervasyonu, kabul edilen morfoloji/semantik metadata'sı ve aggregate coverage tutar. Prompt
yalnız sayımları ve tekrar edilmemesi gereken lemma/anlatı etiketlerini görür; eski test cümleleri
memory üzerinden few-shot olarak sızmaz. Tasarım ve import akışı:
[`DATASET_MEMORY.md`](DATASET_MEMORY.md).

Farklı bilgisayarlarda çalışan ekip üyeleri aynı SQLite dosyasını Git'e koymaz. Kabul edilmiş
aralıklar `test/data/final_shards/` altında ayrı JSONL dosyaları olarak paylaşılır;
`range-run` bunları yerel accepted state ve SQLite memory'sine yeniden işler, önceki sıranın
tamamlandığını doğrular, yalnız istenen yeni 1-based inclusive aralığı üretir ve manifestli shard'a
çıkarır. Boşluk, çakışma ve üretim sözleşmesi değişikliği reddedilir. Komutlar ve ajanlara verilecek
operasyon promptu:
[`COLLABORATIVE_GENERATION.md`](COLLABORATIVE_GENERATION.md).

`development` model/ayar seçimi içindir. `sealed_test`, kararlar tamamlandıktan sonra yalnız final
sonuç için kullanılır; kapalı tutulması gereken şey veri metni değil, model seçerken final gold
sonuçlarına tekrar tekrar bakmamaktır.

## Tek üretim akışı

```text
config + taxonomy
        ↓
600 dengeli ve sabit kota slotu
        ↓
Codex CLI generator (300) + Claude Code CLI generator (300)
        ↓
strict JSON + deterministic QC
        ↓
özellik-kör semantic judge (kör ve deterministik karıştırılmış candidate sırası)
        ↓
ayrı model ailesinden feature-aware morphology judge
        ↓
yüksek güvenli açık judge hatası varsa aynı slot için otomatik refill
        ↓
600 kabul edilmiş family → kör insan final review → freeze
```

Deterministic QC, dataset-memory ihlali veya bir judge'ın en az 85 güvenle somut semantic ya da
morphology aday hatası bulması aynı sabit slotta onarım üretir. Somut uyarıyla birleşen 85-altı
güven, `abstain`/`unclear` veya uyarı olmasa bile 60-altı güven human-review önceliğidir. Temiz
60–84 kararı yalnız QC notudur. İnsan incelemesi üretim döngüsünde değildir: 600 otomatik kabul edilmiş
family tamamlandıktan sonra, freeze öncesinde yapılır. İnsan redleri ilgili slotu yeniden açar.
Bu öncelik koşullarını taşıdığı hâlde otomatik kabul edilen family'ler
`qc.human_review_priority=true` ve dahili neden listesiyle kaydedilir. Review manifestinde bunlar
önce sıralanır; etiket reviewer kararını veya körlüğünü değiştirmez.

Codex ve Claude generator'ları kayıtlı CLI abonelik oturumlarını kullanır; OpenRouter API key yalnız
semantic ve morphology judge çağrılarında kullanılır. Dört rol farklı model ailesinden olmalıdır.
Legacy train Gemini ile üretildiği için `google/*` rollerde varsayılan olarak yasaktır. CLI ve
OpenRouter için requested/actual model, request hash ve kullanım provenance'a yazılır. OpenRouter
judge'larında `data_collection=deny`, `zdr=true` ve structured output zorunludur.

Provider-facing üretim JSON'u bilerek küçüktür. LLM yalnız query, ortak context, kritik
lemma/query sözcüğü ve her aday için `candidate_slot + critical_sentence + critical_word`
yazar. Rol, subtype, `morph_relation`, qrels, edit script, feature açıklaması ve kimlikler
güvenilir plandan Python tarafından eklenir. Böylece model onlarca sabit metadata alanını her
family'de tekrar üretmez. Yerel repair çağrısı yalnız judge'ın işaretlediği candidate slotlarının
iki metin alanını döndürür; Python bu yamayı eski JSON'a uygular ve diğer family alanlarını teknik
olarak kilitli tutar. Family-geneli hatalarda kontrollü geniş onarım yapar.

Easy negatifler rastgele konu dışı cümleler değildir. Query ile aynı domain/register içinde
kalır; mümkünse kişi, kurum, yer veya konu ipuçlarından birini paylaşır, fakat farklı bir olay
ve bilgi ihtiyacı anlattığı için cevabı vermez. Örneğin hastanın annesini tanıyamaması sorgusunda
meteoroloji haberi yerine aynı psikiyatristin başka bir hastaya randevu vermesi uygun easy'dir.
Python bunları `same_domain_off_intent` olarak etiketler. Corpus kapısı bir easy'nin başka bir
family'nin gold'uyla birebir veya yüksek fuzzy benzerlikle çakışmasını reddeder; benzersiz
`semantic_frame_id` kuralı da aynı cevabın farklı family'lerde tasarlanmasını engeller.

## Hibrit family tasarımı ve örnekler

### 1. Strict minimal

Bu dilim nedensel morfoloji testidir. Query ile gold aynı cümle olmak zorunda değildir; minimal
çift gold ile ana morph hard arasındadır.

```text
Query:
Selin seyahatlerinde aktarmalı seçeneklerden uzak durur.

GOLD:
Bilet alırken Selin aktarmalı uçuşları seçmez.

hard_01 — minimal_morph_negative:
Bilet alırken Selin aktarmalı uçuşları seçer.
```

600 family'sinin 150'sinde positive ile `hard_01` aynı kritik lemmayı, sözdizimsel şablonu,
token sırasını ve olayı korur; yalnız hedef eki veya ek zincirini taşıyan kritik sözcükte ayrışır.
`edit_script` bu değişimi Python tarafında kaydeder ve validator iki kritik cümlenin sözcük
iskeletini karşılaştırır.

### 2. Controlled diverse

Bu dilimde gold ve negatifler aynı cümle kalıbına zorlanmaz; aynı bilgi ihtiyacı ve planlanan
morfolojik karşıtlık açık kalır.

```text
Query:
Selin seyahatlerinde aktarma yapmak istemediği için doğrudan uçmayı tercih ediyor.

GOLD:
Uçuş ararken Selin'in önceliği doğrudan seferlerdir; aktarmalı seçenekleri genellikle değerlendirmez.

Morphological hard:
Son yolculuğunda Selin doğrudan sefer bulamadığı için aktarmalı bir uçuşu değerlendirdi.

Semantic hard:
Selin aktarmalı uçuşları daha ekonomik bulur ve doğrudan seferleri nadiren tercih eder.
```

### 3. Natural retrieval

Query, gold ve negatifler bağımsız doğal anlatımlar kullanabilir. Relevance sözcük benzerliğiyle
değil, passage'ın query'deki bilgi ihtiyacını gerçekten karşılamasıyla belirlenir. Bu dilimde
`lexical_retrieval_trap` ve `semantic_retrieval_hard` da kullanılır. Hard'lar gold'un cümle
kalıbını kopyalamak zorunda değildir ve false-negative kontrolü özellikle önemlidir.

Her üç modda da query'nin yalnız bir eşanlamlı sözcüğü değiştirilerek gold yapılması yasaktır.
Validator token edit/sequence kontrolüyle yakın kopyayı yakalar. Gold'un hard'lara karşı
sistematik lexical avantajı ayrıca karşılaştırmalı olarak ölçülür.

Tasarım dayanakları: [Contrast Sets](https://aclanthology.org/2020.findings-emnlp.117/)
küçük fakat anlamlı kontrollü müdahaleleri; [NevIR](https://aclanthology.org/2024.eacl-long.139/)
minimal belge karşıtlıklarını retrieval içinde; [MIRACL](https://aclanthology.org/2023.tacl-1.63/)
doğal query/passage ve bağımsız relevance yargılarını; [DuReader-Retrieval](https://aclanthology.org/2022.emnlp-main.357/)
ise lexical/syntactic mismatch ile pooled false-negative kontrolünü örnekler. Bu benchmark bu
nedenle yalnız minimal veya yalnız doğal veri yerine iki yaklaşımı hibrit kullanır.

## Lexical artefakt kontrolü

Generator positive ile hard kritik cümlelerini family moduna uygun lexical zorlukta yazar.
Strict/controlled/natural modlarında sırasıyla en az 4/3/2 hard aynı olay ve içerik ipuçlarını
korur. Validator:

- Sırasıyla en az 4/3/2 hard'ın query-word overlap'ının gold kadar yüksek olmasını,
- Gold–hard median overlap farkının modlara göre en fazla `0.15/0.25/0.35` olmasını,
- Sırasıyla en az 4/3/2 hard'ın query içerik köklerinin en az `%60`ını korumasını ister.

Gold sürekli en yüksek veya sürekli en düşük overlap'a itilmez; aksi durumda düz ya da ters
artefakt oluşur. Kaliteyi geçmeyen family aynı dengeli slot için taze örnekle değiştirilir. Freeze
öncesinde word-overlap `R@1 ≤ 0.20`, character 3-gram ve BM25 `R@1 ≤ 0.30` kapıları uygulanır.
Ucuz baseline eşitlikleri candidate ID ile bozulmaz; eşit skorlu adayların bütün olası sıraları
üzerinden beklenen, tie-aware metrik hesaplanır.

## Fenomenler ve hard negatifler

Kod 6 macro grup altında 76 hedef taşır: 58 single ve 18 composition/chain.

- Hâl, konum ve yön
- Çoğul, iyelik ve fiilde kişi-sayı uyumu (`V.AGR`)
- Zaman, görünüş, kip ve kanıtsallık
- Olumsuzluk, çatı ve valency
- Ulaç, sıfat-fiil, türetim ve allomorph
- Ek-zinciri composition

Yeni kapsam: isim cümlesi olumsuzluğu (`COP.NEG`), ek-fiil TAM (`COP.TAM`), soru parçacığı
odağı (`Q.PART.SCOPE`), `-mA/-DIK` adlaştırma ayrımı (`NMLZ.MA_VS_DIK`), `-DIK` yapısında
genitif–iyelik bağı (`REL.GEN.POSS`) ve `kendi` zamirinde kişi/sayı-gönderim (`ANAPHOR.AGR`).
`Q.PART.SCOPE` doğal bir odak sorusu üretir ve parçacığın yerini değiştirmeyi gerektirdiğinden,
token sırasını sabit tutan 150-family strict minimal-pair slice'ına atanmaz.

Oflazer çizgisindeki yapılandırılmış Türkçe morfolojisinden alınan beş ek hedef:

- `MORPH.CONTEXT_AMBIG`: aynı yüzey biçiminin bağlamla seçilen farklı lemma/POS/çekim analizi.
- `DERIV.IG_CHAIN`: kök POS → türetim sınırları/IG'ler → final POS zinciri.
- `CASE.ROLE.FRAME`: hâl işaretleriyle agent, patient/theme, goal/recipient ve source ayrımı.
- `SUSP.AFFIX`: koordinasyonda yalnız son eşlenikte yüzeyleşen ekin doğru kapsamı.
- `MWE.MORPH`: destek fiilli, kalıplaşmış ve tekrarlı yapılarda çok-tokenlı morfoloji.

`MORPH.CONTEXT_AMBIG`, `SUSP.AFFIX` ve `MWE.MORPH` tek-token editine indirgenmez ve bu nedenle
strict minimal-pair kotasına
atanmaz. Yeni üretim şeması `participant_bindings` ile query–gold olayındaki her agent, patient,
theme, goal, source, causer ve diğer rolü somut katılımcıya bağlar. Bu alan generator niyet
metadatasıdır; semantik judge gold/rol etiketlerini görmeden metinden bağımsız doğrulama yapar.

Bu tasarımın dilbilimsel dayanakları: Oflazer'in
[iki seviyeli Türkçe morfolojisi](https://aclanthology.org/E93-1066/),
[bağlamsal morfolojik belirsizlik giderme](https://aclanthology.org/C00-1042/),
[morfoloji–sözdizim arayüzü](https://aclanthology.org/P06-1020/),
[IG tabanlı dependency parsing](https://aclanthology.org/J08-3003/) ve
[Türkçede MWE–morfoloji entegrasyonu](https://aclanthology.org/W04-0409/) çalışmalarıdır.

Sekiz hard'ın işlevsel bileşimi:

1. Bir ana morphology hard (`minimal_morph_negative` yalnız strict modda;
   diğerlerinde `controlled_morph_negative`)
2. İki morphology-adjacent hard
3. İki semantic/participant/time hard
4. Bir lexical veya yakın-paraphrase trap
5. İki feature'a/moda uyarlanmış hard

Strict ve controlled modlarında son iki hard fenomene göre seçilir:

- Hâl/fiil uyumu/çatı: `argument_role_reversal`, `morph_distractor`
- Ad çoğulluğu/iyelik: `noun_possessor_number_trap`, `morph_distractor`
- TAM/türetim: `scope_attachment_trap`, `morph_distractor`
- Composition: `partial_chain_negative`, `scope_attachment_trap`
- Allomorph: `allomorph_form_function_trap`, `morph_distractor`

Natural modda son iki slot `semantic_retrieval_hard` ve `morph_distractor` olur. Bu kayıtlar
gold'un cümle kalıbını kopyalamak zorunda değildir.

Geçerli allomorph negative değildir. Örneğin `-da/-de/-ta/-te` aynı bulunma işlevinin yüzey
biçimleri olabilir; `şubesinde` ile `şubesinden` ise LOC/ABL anlam karşıtlığıdır.

## Kayıt ve qrels yapısı

Sadeleştirilmiş internal family:

```json
{
  "family_id": "family_raw_00042_xxx",
  "split": "sealed_test",
  "query": "Ekip raporu zamanında tamamlamadı.",
  "gold_id": "family_raw_00042_xxx_c03",
  "qrels": {
    "family_raw_00042_xxx_c01": 0,
    "family_raw_00042_xxx_c02": 0,
    "family_raw_00042_xxx_c03": 1
  },
  "candidates": [
    {
      "id": "family_raw_00042_xxx_c03",
      "role": "positive",
      "relevance": 1,
      "text": "Ekip raporu vaktinde bitirmedi."
    }
  ]
}
```

Qrels, retrieval cevap anahtarıdır: `query_id + candidate_id + relevance`.

- `1`: query'yi karşılayan tek gold.
- `0`: aynı family için üretilmiş ve LLM judge tarafından yanlış olduğu doğrulanmış negatif.

Kontrollü deney kendi 11 adayını kullanır. Full-corpus retrieval'da her query'nin tasarım gereği
tek gold'u vardır; diğer family'ler farklı semantic frame taşır. Exact/fuzzy cross-family kopya,
easy→başka-family-gold çakışması ve frame tekrar kontrollerinden geçen diğer belgeler
nonrelevant kabul edilerek sealed testteki
5.500 belgenin tamamı sıralanır.

Paper iki sonucu birlikte ana değerlendirme olarak raporlar: kontrollü 11-aday contrast retrieval
küçük morfolojik farkı izole eder; full-corpus retrieval gold'un büyük ortak corpus içinde bulunup
bulunamadığını ölçer. Full-corpus sonuç yalnız tanısal yardımcı analiz değildir.

## Otomatik kalite kontrolleri

Family düzeyi:

- Tam `1 positive + 8 hard + 2 easy`
- Eksiksiz ve benzersiz candidate slot/ID'leri
- Query/pasaj için planlanan cümle sayısı
- `Q.PART.SCOPE` için query ve 11 kritik aday cümlesinin tamamı soru biçiminde; diğer
  75 fenomen için query ve kritik aday cümlelerinin tamamı düz bildirim biçiminde
- Kritik sözcüğün gerçekten metinde bulunması
- Strict minimal-pair iskeleti ve `edit_script` uyumu
- Allomorph/function ayrımı
- Candidate uzunluk dengesi ve gold-length bias
- Positive ile hard kritik cümlelerinde dengeli query-word overlap
- En az dört hard'ın gold kadar lexical overlap ve query içeriği taşıması
- Tek ve doğru gold qrels
- Özellik-kör semantic judge ile benzersiz positive, doğallık ve iç tutarlılık
- Ayrı feature-aware morphology judge ile ek işlevi, kapsam ve allomorf kontrolü
- Karıştırılmış candidate sırası, confidence ve abstain kaydı
- Judge başarısızlığında sabit slotta otomatik refill
- 600 accepted family tamamlandıktan sonra freeze öncesi kör insan review
- Her negatif için query'yi doğrulama/paraphrase etme riski ve her aday için iç tutarlılık

Corpus/freeze düzeyi:

- Exact ve fuzzy query tekrarları
- Cross-family candidate tekrarları
- Cross-family query–candidate yakın kopyaları
- Aşırı kullanılan soyut cümle şablonları
- Generator'a göre tekrarlanan başlangıç kalıpları
- Lemma/template/composition/domain leakage
- Candidate ve kritik-cümle konum bias'ı
- Tie-aware word-overlap, character 3-gram ve BM25 freeze eşikleri
- Train üretildikten sonra exact/fuzzy train–test leakage

Opsiyonel [Stanza](https://stanfordnlp.github.io/stanza/pipeline.html) audit'i kritik lemma ve UD
`UFeats` bilgisini kontrol eder. `morph_explicit` query'lerde hedef özelliğin kritik query
sözcüğünde bulunmasını da denetler. Bu gerçek morfem segmentasyonu değildir; audit uyarıları otomatik
gold değiştirmez.

Opsiyonel güçlü adjudicator `config.json > generation > judges > adjudicator` altında
`enabled=true` ve ayrı bir model ailesi verilerek açılır. Kararı otomatik gold değiştirmez; tavsiyesi
final insan manifestine eklenir. `review-export` semantic ve morphology görünümlerini ayırır;
`review-apply` aynı reviewer'ın tekrarını saymaz ve gerekli bağımsız reviewer çoğunluğu oluşmadan
slotu accept/reject durumuna geçirmez.

## Evaluation

Kontrollü 11-aday metrikleri:

- `Recall@1/3`, `MRR@10`, `nDCG@10`
- `hard_only_mrr@10`, `hard_only_ndcg@10`
- `pairwise_hard_accuracy`
- `pairwise_morph_hard_accuracy`
- `pairwise_semantic_hard_accuracy`
- `all_hard_family_consistency`
- `minimal_margin`, `hardest_hard_margin`, `hardest_negative_margin`

Full-corpus retrieval metrikleri:

- `Recall@1/3/10/50`, `MRR@10`, `nDCG@10`
- Her sealed query bütün 5.500 test belgesini sıralar.

Artefakt kontrolleri:

- Longest candidate / most tokens / candidate position
- Tie-aware character 3-gram / word overlap / BM25
- Query'siz candidate-only char-TFIDF
- Kritik sözcük silme
- `prefix5` suffix-reduction kontrolü; gerçek lemma/kök analizi değildir

İstatistikler query-level bootstrap `%95 CI`, paired bootstrap, approximate randomization,
McNemar, Holm düzeltmesi ve slice sonuçlarını içerir. Tekil 76 fenomen küçük örnekli tanısal
tablodur; ana rapor macro/layer/objective ve morph-hard/semantic-hard düzeyindedir.

## Komutlar

```bash
# API'siz regresyon testi ve plan
python3 -m test self-test
python3 -m test plan --run-id test_v39_final
python3 -m test memory-report --run-id test_v39_final

# Mevcut train metadata'sını tekrar önleme hafızasına ekle
python3 -m test memory-ingest --run-id test_v39_final --input TRAIN.json \
  --source train_current --split train

# 300 Codex CLI + 300 Claude Code CLI; judge'lar OpenRouter
codex login status
claude  # ilk açılışta /login ile Claude abonelik hesabını seç
export OPENROUTER_API_KEY="..."
# Opsiyonel override; varsayılanlar gpt-5.6-sol ve claude-opus-5 (ikisi de medium).
# export TEST_CODEX_GENERATOR_MODEL="gpt-5.6-sol"
# export TEST_CLAUDE_GENERATOR_MODEL="claude-opus-5"

# 1-slot ücretli smoke test; varsayılan judge'lar DeepSeek V4 Flash + GLM 5.3 Flash'tır.
python3 -m test generate --run-id smoke_v39 --limit 1 --workers 1

# İstenirse modeller environment ile override edilebilir.
export TEST_SEMANTIC_JUDGE_MODEL="deepseek/deepseek-v4-flash-0731"
export TEST_MORPHOLOGY_JUDGE_MODEL="z-ai/glm-5.3-flash"
python3 -m test generate --run-id test_v39_final

# GLM 5.3 Flash: zorunlu low reasoning; reasoning metni response'a eklenmez.

# 600 accepted family tamamlanınca freeze öncesi kör insan review
python3 -m test review-export --run-id test_v39_final
python3 -m test review-apply --run-id test_v39_final --input decisions.jsonl
python3 -m test judge-report --run-id test_v39_final
python3 -m test generate --run-id test_v39_final  # yalnız insanın reddettiği slotları refill eder

# Tamamlanan 100/500 kotasını doğrula, freeze et ve qrels export et
python3 -m test finalize --run-id test_v39_final

# Train üretildikten sonra leakage audit
python3 -m test audit-leakage --test TEST.json --train TRAIN.json

# Opsiyonel lemma/UFeats audit
python3 -m test morph-audit --input TEST.json --output morph_audit.json --download-model

```

Eski ayrı preview üretim motoru final üretim öncesinde kaldırılmıştır. Pilot ve 600-family üretimi
aynı nihai pipeline'ı kullanır. Generator çağrıları aynı modele ait üç bağımsız family'yi tek
structured-output isteğinde üretir; deterministic QC, iki bağımsız judge, sorunlu slot onarımı ve
SQLite kaydı family bazında kalır. Yeni veriler yalnız
`runs/<run-id>/` altındaki plan, SQLite registry, accepted/rejected kayıtları ve provider
provenance'ı üzerinden ilerler. Küçük notebook resmî shard'ların ilk beş family'sini kullanır;
ayrı bir eski pilot JSON tutulmaz. Paper release final insan kontrolü ve freeze sonrasında çıkar.

## Ana dosyalar

| Dosya | Görev |
|---|---|
| `config.json` | Sayılar, dağılımlar, eşikler ve modeller |
| `taxonomy.py` | Fenomen ve hard-negative kataloğu |
| `planner.py` | 600 dengeli slot ve iki-generator için 300/300 atama |
| `schema.py`, `prompts.py` | Strict üretim/judge sözleşmesi |
| `providers.py` | Codex/Claude CLI generator ve OpenRouter judge adapter'ları |
| `pipeline.py` | Generate → QC → semantic/morph judges → refill → checkpoint |
| `review.py` | 600 kabulden sonra kör final-review manifesti ve slot reopen/accept |
| `judge_report.py` | Order stability, escalation ve insan-kalibrasyon raporu |
| `validators.py` | Family/corpus/duplicate/leakage kontrolleri ve qrels |
| `selection.py` | Tam 100/500 dağılım ve generator kotalarını doğrulama |
| `exports.py` | Freeze, blind/internal JSON, BEIR ve qrels |
| `morphology.py` | Opsiyonel Stanza audit'i |
| `evaluation.py` | Metrikler, baseline, ablation ve istatistik |
| `notebooks/morph_baseline_eval_pilot5_colab.ipynb` | Resmî shard'lardan beş-family / 55-belge hızlı test |
| `notebooks/morph_baseline_eval_600_colab.ipynb` | 100 dev + 500 final paper değerlendirmesi |

Freeze çıktıları:

```text
test/runs/<run_id>/
├── plan.json
├── accepted.jsonl
├── rejected.jsonl
├── generation_report.json
├── release/
│   ├── morph_dev_v3.9.0.json
│   ├── morph_test_blind_v3.9.0.json
│   ├── artifact_audit.json
│   └── freeze_manifest.json
└── private/
    ├── morph_test_internal_v3.9.0.json
    ├── private_qrels.jsonl
    ├── beir_test_qrels.tsv
    └── train_exclusion_holdouts.json
```
