# Test benchmark — kısa sürüm

- 600 family: 100 development + 500 final test.
- Her family: 1 query + 1 gold + 8 hard negative + 2 easy negative.
- Easy'ler aynı domain/ortamda farklı olayı anlatan `same_domain_off_intent` adaylardır; rastgele
  konu dışı değildir ve başka bir family'nin gold'uyla exact/fuzzy çakışamaz.
- Family modları: 150 strict minimal + 270 controlled diverse + 180 natural retrieval.
- Query anlatımı: 300 morph-explicit + 300 semantic-paraphrase.
- Query–gold lexical bandı: 180 high + 240 medium + 180 low.
- Generator'lar: 300 Codex CLI + 300 Claude Code CLI; API key yalnız iki OpenRouter judge içindir.
- Query: `%75` 1 cümle, `%25` 2 cümle.
- Pasaj: `%30/%30/%30/%10` oranında 1/2/3/4 cümle.
- 76 fenomen, 6 macro grup; morph-hard ve semantic-hard ayrı raporlanır.
- Oflazer-informed yeni beşli: `MORPH.CONTEXT_AMBIG`, `DERIV.IG_CHAIN`,
  `CASE.ROLE.FRAME`, `SUSP.AFFIX`, `MWE.MORPH`.

Tek akış:

```text
600 dengeli slot → 300 Codex + 300 Claude → deterministic QC → semantic judge (kör/karıştırılmış sıra)
→ morphology judge → başarısız slotu refill → 600 accepted → insan final review → freeze
```

Bir family kontrolden kalırsa fenomeni, split'i, generator'ı ve uzunluk kotası değişmez; yalnız
örnek yeniden üretilir. Üç refill turu yetmezse aynı komut yeniden çalıştırıldığında kaldığı yerden
devam eder. Amaç fazladan 1.800 örnek yazmak değil, tam 600 kabul edilmiş family elde etmektir.

Çoklu generator koordinasyonu `dataset_memory.sqlite3` üzerinden yapılır. Registry slotları atomik
rezerve eder; kabul edilen fenomen/lemma/anlatı metadata'sından aggregate coverage ve kaçınılacak
etiketler üretir. Önceki test cümleleri generator promptuna verilmez. Ayrıntı:
[`DATASET_MEMORY.md`](DATASET_MEMORY.md).

Farklı bilgisayarlarda SQLite paylaşılmaz. Git'te tutulan JSONL shard + manifest çiftleri her
kişinin yerel accepted state ve SQLite memory'sine alınır. Kullanıcı 1-based inclusive sıradaki
aralığı `range-run --producer codex --from 51 --to 120` biçiminde verir; sync, sözleşme kontrolü,
üretim ve manifestli shard export otomatik yapılır. Boşluk veya çakışma reddedilir.
Komutlar ve hazır ajan talimatı:
[`COLLABORATIVE_GENERATION.md`](COLLABORATIVE_GENERATION.md).

Strict modda gold–`hard_01` yalnız hedef biçimde ayrışır. Controlled modda gold ve hard farklı
doğal sözdizimi kullanabilir. Natural modda query, gold ve negatifler bağımsız yazılabilir; gold
bilgi ihtiyacını karşılayan tek passage'dır. Lexical kapılar moda göre 4/3/2 içerik-koruyan hard
ister ve word-overlap/char-3gram/BM25 sonuçlarını tie-aware hesaplar.

LLM küçük bir JSON üretir: query, ortak context, kritik lemma/sözcük, query–gold için somut
katılımcı/rol bağları ve aday başına yalnız
`candidate_slot + critical_sentence + critical_word`. Rol, subtype, morph relation, qrels,
edit script ve kimlikler Python tarafından eklenir.

Semantic judge hedef özelliği görmeden açık ikinci-gold, katılımcı rolleri, tutarlılık ve doğallık
hatalarını kontrol eder. Morphology judge hedef özellik, bozuk çekim, allomorf, hâl/çatı rolleri,
IG/türetim sınırları ve çok-tokenlı morfolojik kapsamı kontrol eder. Yalnız güveni en az
85 olan semantic veya morphology somut aday hatası aynı slotun iki metin alanını onartır; Python
diğer family alanlarını değiştirmeden korur. Somut uyarıyla birleşen
85-altı güven, `abstain`/`unclear` veya uyarı olmasa bile 60-altı güven human-review önceliğidir.
Temiz 60–84 kararı yalnız QC notudur. Human review üretim
sırasında değil, 600 kabul edilmiş family tamamlandıktan
sonra freeze öncesinde uygulanır; insanın reddettiği slotlar yeniden üretilir.
Bu öncelik koşullarını taşıyan kabul edilmiş family'ler
`human_review_priority=true` ile işaretlenir ve final-review manifestinde önce gösterilir.

Eski ayrı preview üretim kodu final üretim öncesinde kaldırılmıştır. Pilot ve ana üretim aynı
nihai pipeline'ı kullanır; ilk üretim çağrıları üç family'lik batch, bütün QC/judge/refill ve SQLite
kayıtları family bazındadır. Resmî shard'ların ilk beş family'si
`notebooks/morph_baseline_eval_pilot5_colab.ipynb` ile hızlıca değerlendirilir; ayrı bir eski pilot
JSON tutulmaz. 600-family paper notebook'u bundan ayrıdır.

Qrels family oluşturulurken hazırdır:

```text
gold candidate = 1
diğer 10 generated negative = 0
```

Kontrollü 11-aday metrikleri `Recall@1/3`, MRR@10 ve nDCG@10'dur. Full-corpus retrieval, her
sealed query için farklı semantic frame'lerdeki 5.500 belgenin tamamını sıralar ve
`Recall@1/3/10/50`, MRR@10, nDCG@10 raporlar.

Paper'ın iki ana sonucu birlikte sunulur: kontrollü contrast retrieval morfolojik ayrımı,
full-corpus retrieval ise aynı gold'u büyük ortak corpus içinde bulma başarısını ölçer. İkincisi
yalnız tanısal bir tablo değildir.

Komutlar:

```bash
python3 -m test self-test
python3 -m test plan --run-id test_v39_final
python3 -m test memory-report --run-id test_v39_final

export OPENROUTER_API_KEY="..."
# Opsiyonel override; varsayılanlar gpt-5.6-sol ve claude-opus-5 (medium).
# export TEST_CODEX_GENERATOR_MODEL="gpt-5.6-sol"
# export TEST_CLAUDE_GENERATOR_MODEL="claude-opus-5"
# Opsiyonel override; varsayılanlar aşağıdaki ücretli ve ZDR uyumlu modellerdir.
export TEST_SEMANTIC_JUDGE_MODEL="deepseek/deepseek-v4-flash-0731"
export TEST_MORPHOLOGY_JUDGE_MODEL="z-ai/glm-5.3-flash"
python3 -m test generate --run-id test_v39_final
# GLM 5.3 Flash low reasoning kullanır; reasoning metni saklanmaz.
python3 -m test review-export --run-id test_v39_final  # 600 accepted family hazır olunca
python3 -m test review-apply --run-id test_v39_final --input decisions.jsonl
python3 -m test judge-report --run-id test_v39_final
python3 -m test generate --run-id test_v39_final  # yalnız insanın reddettiği slotları refill eder
python3 -m test finalize --run-id test_v39_final
```

Detaylar: [`README.md`](README.md). Final Colab:
[`notebooks/morph_baseline_eval_600_colab.ipynb`](notebooks/morph_baseline_eval_600_colab.ipynb).
