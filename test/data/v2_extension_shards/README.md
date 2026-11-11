# Test V2 extension shards

Bu klasör, insan-kontrollü V1 test setini **600'den 1.200 family'ye** genişletmek
içindir. V1'in 600 kaydı korunur; V2 `extension_v40` planı bunun üzerine 600 yeni
family ekler.

## Üretici sırası ve dosya adları

Dosya numaraları V1'in ardından devam eden **genel family sırasıdır**; V2 planının
`raw_00xxx` slot ID'leriyle aynı şey değildir:

- Codex: **601–900** (300 family), shard adları `codex_601_615.jsonl` biçiminde.
- Claude: **901–1200** (300 family), shard adları `claude_901_915.jsonl` biçiminde.

Claude için V2'de yeni numaralandırma 1'den başlamaz. Üretim planında `generator_a`
Codex'e, `generator_b` Claude'a karşılık gelir; her biri `extension_v40` planındaki
300 kendi slotunu alır. Genel numara yalnız paylaşılabilir dosya adında kullanılır.

## Üretim sözleşmesi

- Run: `extension_v40` (`test/config_v2_extension.json`)
- Generator: Codex GPT-5.6 Sol / Claude Opus 5, medium reasoning
- Semantic judge: DeepSeek V4 Flash
- Morphology judge: GLM-5.3 Flash, low reasoning
- Final insan kontrolü: henüz tamamlanmadı.

Extension memory'sine V1 için `test/data/morph_test_600_sealed.json`,
`prior_test_release` / `sealed_test` kaynağıyla alındı. Train metadata'sı bu run'ın
memory'sine ingest edilmedi. Run'ın kendi üretilmiş metadata'sı ayrıca tutuluyor.

## Senkronizasyon öncesi veri uyarısı

Codex shard'larının mevcut kopyasında 300 satır var ama yalnızca **298 benzersiz
slot ID** bulunuyor: `raw_01001` ve `raw_01003` hem `codex_795_800.jsonl` hem
`codex_801_900.jsonl` içinde, farklı içerikle yer alıyor; `raw_00947` ve
`raw_00991` ise yok. Bu kayıtları sessizce atlayıp üretimi 300 tamamlanmış gibi
raporlamayın. Önce doğru sürümler belirlenip shard kapsamı düzeltilmeli; sonra
shared-sync ve benzersizlik kontrolleri yeniden çalıştırılmalı. Mevcut shard'lar
bu notla değiştirilmiş sayılmaz.
