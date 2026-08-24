# Sıralı ortak test üretimi

SQLite her bilgisayarda yereldir ve Git'e gönderilmez. Ortak gerçek kaynak
`test/data/final_shards/` altındaki JSONL shard + manifest çiftleridir. Her yeni üretici önce
Git'teki önceki shard'ları doğrular, yerel SQLite'a senkronlar ve yalnız sıradaki kesintisiz aralığı
üretir.

## Kullanıcıya görünen aralık

Aralıklar 1'den başlar ve iki uç da dahildir. Örneğin:

- İlk Codex üretimi: `1–50` → 50 family.
- Sonraki Codex üretimi: `51–120` → 70 family.
- `50–120` reddedilir; 50. family önceki shard'da bulunduğu için çakışır.

Codex ve Claude'un ayrı 300'er slotluk sırası vardır. Kod bunları plandaki `generator_a` ve
`generator_b` slotlarına dönüştürür; kullanıcı offset hesaplamaz. Claude sırası, Codex'in 300
family'si tamamlanıp pushlanmadan açılamaz; böylece Claude üretimi bütün Codex metadata'sını görür.

## Tek komut

```bash
git pull --ff-only
python3 -m test self-test
python3 -m test range-show --producer codex --from 51 --to 120
python3 -m test range-run  --producer codex --from 51 --to 120
```

`range-run`:

1. Bütün shard + manifest çiftlerini ve SHA-256 değerlerini doğrular.
2. Codex/Claude sıralarında boşluk veya çakışma olmadığını kontrol eder.
3. Önceki accepted family'leri yerel SQLite memory'ye aktarır.
4. Yerel config, plan, prompt, generator/judge ve pipeline kaynak sözleşmesini önceki shard'larla
   karşılaştırır.
5. Yalnız istenen yeni aralığı QC + iki judge + refill akışıyla üretir.
6. Tamamlanınca örneğin `codex_051_120.jsonl` ve
   `codex_051_120.jsonl.manifest.json` dosyalarını otomatik oluşturur.

Komut yarıda kalırsa aynı aralıkla yeniden çalıştırılır; SQLite accepted slotları atlar ve yalnız
eksikleri tamamlar. Shard yazıldıktan hemen sonra süreç kesilip manifest yazılamadıysa, aynı
makinedeki accepted state doğrulanarak manifest güvenli biçimde yeniden oluşturulur.

## Commit ve push

```bash
git add test/data/final_shards/codex_051_120.jsonl \
        test/data/final_shards/codex_051_120.jsonl.manifest.json
git commit -m "Add Codex test shard 51-120"
git push
```

Bir sonraki kişi önce bu commit'i pull eder. Böylece farklı makinelerde SQLite paylaşılmasa bile
her yeni DB önceki verilerden yeniden kurulur ve yeni generation context daha önce kullanılan
lemma, template, semantic frame ve anlatı metadata'sını görür. Ham test cümleleri generator'a
few-shot olarak verilmez.

## Codex'e verilecek prompt

```text
Bu repoda Codex sırasındaki 51–120 family aralığını üret ve pushla.

Önce git status ile çalışma ağacını kontrol et; ilgisiz yerel değişiklik varsa üzerine yazmadan dur
ve bildir. Temizse git pull --ff-only çalıştır. test/COLLABORATIVE_GENERATION.md dosyasını tamamen
oku; ardından self-test ve range-show çalıştır. Sonra yalnız range-run komutunu kullan. Prompt,
config, schema ve üretim kodunu değiştirme. QC, DeepSeek semantic judge, GLM morphology judge ve
refill akışını tamamla. Üretim biterse JSONL shard ile eşleşen manifesti ve ranges-report sonucunu
doğrula. .env, API key, test/runs, cache ve SQLite'ı commit etme. Yalnız oluşan shard + manifest
dosyalarını commit edip main'e pushla; sonunda süreyi, accepted/refill sayılarını, judge maliyetini
ve commit hash'ini söyle.
```

Claude için yalnız “Codex sırası” yerine “Claude sırası” denir; pipeline doğru generator'ı seçer.

## Güvenlik garantileri

- Yeni aralık, mevcut son numaranın tam bir sonrasından başlamazsa reddedilir.
- Aynı slot iki farklı içerikle gelirse `shared-sync` durur.
- Shard veya manifestten biri eksikse üretim başlamaz.
- Shard hash'i, family sayısı ya da tam slot listesi değişmişse manifest doğrulaması kalır.
- Cross-family exact/fuzzy kopyalar birleşimde yeniden kontrol edilir.
- Config, plan, prompt, modeller veya pipeline kodu değişmiş shard'lar aynı benchmark'a karışamaz.
- Her generator sırası en fazla 300 family olabilir.
