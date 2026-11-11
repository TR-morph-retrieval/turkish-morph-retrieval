# Test V2 extension shards

Bu klasör, insan-kontrollü V1 test setini **600'den 1.200 family'ye** genişletmek
içindir. V1'in 600 kaydı korunur; V2 `extension_v40` planı bunun üzerine 600 yeni
family ekler.

## Üretici sırası ve dosya adları

Dosya numaraları V1'in ardından devam eden **genel family sırasıdır**; V2 planının
`raw_00xxx` slot ID'leriyle aynı şey değildir:

- Codex: **601–900** (300 family), shard adları `codex_601_615.jsonl` biçiminde.
- Claude: **901–1200** (300 family), shard adları `claude_901_915.jsonl` biçiminde.

Önceden 616–705 olarak adlandırılmış altı Claude shard'ı, içerikleri değiştirilmeden
916–1005 adlarına taşındı. Bu 90 satırın tamamı plandaki Claude slot sırasıyla
eşleşiyor ve provenance'da istenen/gerçek model `claude-opus-5-5` görünüyor.
İlk 901–915 aralığı henüz bu depoda yok; ayrıca tamamlanmalıdır. Sonradan
eklenen ve planın 1006–1020 sırasına denk gelen Claude shard'ı da doğru genel
numaraya taşındı.

Claude için V2'de yeni numaralandırma 1'den başlamaz. Üretim planında `generator_a`
Codex'e, `generator_b` Claude'a karşılık gelir; her biri aynı 600-slotluk plandaki
300 kendi slotunu alır. Genel numara yalnız paylaşılabilir dosya adında kullanılır.

## Üretim sözleşmesi

- Codex'in mevcut V2 kayıtları: `extension_v40` / `test/config_v2_extension.json`
  (Codex GPT-5.6 Sol; o run'ın henüz kullanılmamış Claude kolunda Opus 5 kayıtlıdır).
- Yeni Claude V2 üretimi: `extension_v41_opus55` /
  `test/config_v2_extension_opus55.json` (Claude Code `claude-opus-5-5`, medium
  reasoning). Eski config/run manifesti geriye dönük değiştirilmez.
- Semantic judge: DeepSeek V4 Flash
- Morphology judge: GLM-5.3 Flash, low reasoning
- Final insan kontrolü: henüz tamamlanmadı.

Yerel `extension_v40` memory'sinde 600 V1 kayıt `prior_test_release` / `sealed_test`
etiketiyle bulunuyor; train metadata'sı bulunmuyor. SQLite kaynak dosya yolunu
tutmuyor, fakat kayıtlı metadata'nın 600/600'ü
`test/data/final_test/morph_test_600_human_reviewed.json` ile birebir eşleşiyor
(eski `test/data/morph_test_600_sealed.json` ile 589/600). Yeniden kurulumda
insan kontrollü dosya kullanılır. Run'ın kendi üretilmiş metadata'sı ayrıca tutulur.

Opus 5.5 için yeni run açılmasının nedeni, mevcut manifestin generator modelini ve
config hash'ini sabitlemesidir. Modeli aynı run içinde değiştirmek sözleşme hatası
verir. Yeni config yalnız Claude modelini değiştirir; slot planı ve Codex modeli
aynı kalır. Codex shard'ları aşağıdaki çakışma giderildikten sonra yeni run'a
senkronlanmalı, ardından Claude kolu üretilmelidir. Eski shard'ların provenance'ı
değiştirilmez; yeni Claude kayıtlarının gerçek model kimliği provenance'da
`claude-opus-5-5` olarak doğrulanmalıdır.

## Codex çakışma çözümü ve kalan eksikler

`raw_01001` ve `raw_01003` için plandaki doğru Codex sırası 801–802'dir;
`codex_801_900.jsonl` içindeki iki kayıt korundu. Yanlışlıkla eski
`codex_795_800.jsonl` dosyasına eklenmiş farklı metinli kopyalar aktif
shard'dan çıkarıldı. Önceki sürümler Git geçmişindeki `5af02d9` commit'inin
`codex_795_800.jsonl` dosyasında (5–6. satırlar) geri alınabilir. İçerik
aralıkları doğrulanarak `codex_774_794.jsonl` → `codex_775_795.jsonl`, kalan
`codex_795_800.jsonl` → `codex_797_800.jsonl` olarak yeniden adlandırıldı.

Codex genel sıra 774 (`raw_00947`) ve 796 (`raw_00991`) aynı `extension_v40`
GPT-5.6 Sol üretim/hakem hattında tamamlanıp tekli shard'lar olarak eklendi.
Codex tarafında artık **300 satır / 300 benzersiz slot** var. Diğer erken dönem Codex dosyalarının bazı adları
(642–658) içerik sırasıyla birebir eşleşmese de slot ID'leri tekildir;
`shared-sync` gerçek plan/slot ID üzerinden doğrulama yapar.

Opus 5.5 run'ında `shared-sync` 300 Codex + 105 Claude = 405 kayıtla geçmiştir.
Kalan 195 slotun tamamı Claude tarafındadır. Bu, son üretimin tamamlandığı anlamına
gelmez; insan kontrolü ayrıca yapılacaktır.

## Claude üretimine hazırlık

Codex çakışması ve eksik iki slot giderildi; her makinede
aynı config ve run ID kullanılır:

```bash
python3 -m test --config test/config_v2_extension_opus55.json plan --run-id extension_v41_opus55
python3 -m test --config test/config_v2_extension_opus55.json memory-ingest --run-id extension_v41_opus55 --input test/data/final_test/morph_test_600_human_reviewed.json --source prior_test_release --split sealed_test
python3 -m test --config test/config_v2_extension_opus55.json shared-sync --run-id extension_v41_opus55 --input-dir test/data/v2_extension_shards
python3 -m test --config test/config_v2_extension_opus55.json generate --run-id extension_v41_opus55 --generator-id generator_b --offset 0 --limit 15
python3 -m test --config test/config_v2_extension_opus55.json shard-export --run-id extension_v41_opus55 --generator-id generator_b --offset 0 --limit 15 --output test/data/v2_extension_shards/claude_901_915.jsonl
```

Her batch'te Claude sıra başlangıcı `N` ise `--offset N-901` ve çıktı adı
`claude_N_(N+14).jsonl` olur. Mevcut 916–1020 shard'ları yeniden üretilmez.
İlk eksik 901–915 tamamlandıktan sonraki boş aralık 1021–1035'tir
(`--offset 120`). Her batch'ten önce güncel shard'lar pull/sync edilir.
`range-run` komutu eski V1
`final_shards` sözleşmesine bağlıdır; bu V2 uzantısı için yukarıdaki açık
`generate` + `shard-export` akışı kullanılır.
