# Dataset memory ve coverage registry

Çoklu generator'lar geçmiş test cümlelerini prompta tekrar yüklemez. Her run için
`test/runs/<run-id>/dataset_memory.sqlite3` oluşturulur ve yalnız yapılandırılmış durum tutulur:

- değiştirilemez slot sözleşmesi ve atomik worker rezervasyonu,
- morfolojik fenomen, ek/yüzey biçimleri ve kritik lemma,
- anlatı, olay, katılımcı rolleri, polarity, zaman ve kapsam etiketleri,
- split, generalization bucket, domain, register, template ve generator provenance,
- lifecycle olayları ile aggregate coverage sayımları.

Generator yalnız coverage sayımlarını ve daha önce kullanılan lemma/anlatı etiketlerini görür;
önceki query veya aday metinleri prompta konmaz. Böylece prompt maliyeti büyümez ve eski test
örneklerinin istemeden few-shot olarak kopyalanması önlenir.

## Akış

```text
plan → slot reserve → aggregate memory → generate → deterministic QC → cascade judge
     → accepted metadata commit; judge reddinde refill → 600 accepted → insan final review → freeze
```

SQLite `WAL` ve `BEGIN IMMEDIATE` rezervasyonu aynı slotun iki workera verilmesini engeller. JSONL
çıktıları veri teslim formatı olmaya devam eder; registry koordinasyon ve denetim katmanıdır.

## Komutlar

Plan oluşturulduğunda registry de hazırlanır:

```bash
python3 -m test plan --run-id test_v39_final
python3 -m test memory-report --run-id test_v39_final
```

Train/dev metadata'sını test üretiminden önce hafızaya eklemek için:

```bash
python3 -m test memory-ingest \
  --run-id test_v39_final \
  --input train/data_morph_v2/morph_train_v2.2.json \
  --source train_v2_2 \
  --split train
```

`memory-ingest` ham cümleleri generation context'e taşımaz; yalnız kompakt metadata ve tekrar
önleme sinyalleri kullanılır. `test/runs/` private/yerel olduğundan SQLite dosyası commitlenmez.
Import edilen train/dev metadata'sı ayrıca bucket sözleşmesini uygular: `lemma_holdout` kritik lemma,
`template_holdout` template ve `composition_holdout` tam ek zinciri external veride görülmüşse family
judge çağrısından önce reddedilip aynı slotta yeniden üretilir.

## Etiket sözleşmesi

Her yeni family aşağıdaki küçük `semantic_profile` nesnesini üretir:

```json
{
  "narrative_tag": "banka_para_transferi",
  "event_type": "para_cekme",
  "participant_roles": ["agent", "source", "theme"],
  "participant_bindings": [
    {"role": "agent", "participant": "Selin"},
    {"role": "source", "participant": "şube"},
    {"role": "theme", "participant": "para"}
  ],
  "polarity": "affirmative",
  "temporal_frame": "past",
  "scope_target": "predicate"
}
```

Etiketler ASCII `snake_case` olmalı; özel ad veya ham cümle içermemelidir. Yalnız
`participant_bindings.participant`, query–gold olayındaki somut katılımcıyı kısa biçimde yazar;
rol listesiyle birebir uyuşur. Morfoloji metadata'sı
generator beyanından değil, güvenilir plandaki feature taksonomisinden eklenir. Semantic profil
generator niyetidir. Cascade judge reddi aynı slotta otomatik refill üretir. İnsan review yalnız
600 kabul edilmiş family tamamlandıktan sonra uygulanır; insan reject verirse slot yeniden üretim
için `rejected` durumuna açılır.
