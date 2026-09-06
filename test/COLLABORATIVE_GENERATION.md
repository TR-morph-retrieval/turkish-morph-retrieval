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
`generator_b` slotlarına dönüştürür; kullanıcı offset hesaplamaz. İki üretici dönüşümlü batch'lerle
ilerleyebilir: örneğin Codex `1–50`, Claude `1–50`, Codex `51–100`. Her batch'ten sonra shard ve
manifest pushlanır; sıradaki kişi pull ettiği için önceki bütün metadata yerel SQLite memory'ye
aktarılır. Bilimsel olarak gereksiz olan “önce Codex'in 300'ü bitsin” kilidi yoktur.

## Tek komut

### Judge token sınırında kesilirse

`finish_reason=length` bir judge kararı değildir; JSON tamamlanmış görünse bile kabul edilmez.
OpenRouter adapter aynı model/prompt ile bütçeyi en fazla iki kez artırır
(2600 → 5200 → 10400); hâlâ kesiliyorsa başarısız döner. QC eşikleri ve reasoning ayarı korunur.
Her yanıtın bütçesi, sağlayıcısı, finish_reason ve kullanımı cache içindeki
`*.attempts.jsonl` kaydında; başarılı isteğin denemeleri family provenance içindeki
`usage.transport_attempts` alanında tutulur. Buradaki iç kullanım kayıtlarını toplam maliyete
ikinci kez eklemeyin; dış usage son başarılı yanıtı temsil eder.

`transport_compatibility.json` yalnız bu onaylı kod sürümünün tam hash eşlemesini içerir.
Eski shard sözleşmesi korunurken gerçek çalıştırma hash'leri yeni family provenance'ında
ve yerel `transport_execution.json` içinde kaydedilir. Başka kod/config/prompt değişiklikleri
uyumluluk istisnasından yararlanamaz. Bu, bütün örneklerin aynı token bütçesiyle judge edildiği
anlamına gelmez; etkin bütçeler provenance'dan raporlanmalıdır.

Claude 136–150 tamamlanmamışsa arkadaşınız güncel kodu pull ettikten sonra aynı
`range-run --producer claude --from 136 --to 150` komutunu çalıştırmalıdır.
Yerelde accepted olan kayıtlar korunur; eksikler tamamlanmadan shard yayınlanmaz.

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

## Codex 300 tamamlandıktan sonra Claude üretimi

Mevcut resmî durumda Codex sırası `1–300` tamamlanmıştır. İkinci üretici yalnız Claude sırasını
`1–300` arasında, 15-family shard'lar hâlinde üretir:

```text
1–15, 16–30, 31–45, 46–60, 61–75,
76–90, 91–105, 106–120, 121–135, 136–150,
151–165, 166–180, 181–195, 196–210, 211–225,
226–240, 241–255, 256–270, 271–285, 286–300
```

İlk aralıktan önce `git pull --ff-only`, `self-test`, `ranges-report` ve
`range-show --producer claude --from 1 --to 15` çalıştırılır. Beklenen başlangıç kapsamı
`Codex=300, Claude=0` olmalıdır. Her aralık için yalnız şu producer kullanılır:

```bash
python3 -m test range-run --producer claude --from 1 --to 15
```

Sonraki aralıklarda `--from/--to` yukarıdaki sıraya göre değiştirilir. Tam 15 family accepted
olmadan shard pushlanmaz. Rejected/eksik slot varsa aynı komut yeniden çalıştırılır; accepted
slotlar korunur ve yalnız eksikler refill edilir. Her tamamlanan aralıktan sonra ilgili
`claude_XXX_YYY.jsonl` + manifest çifti commit edilip `main` branch'ine pushlanır. Push başarılı
olmadan sonraki aralığa geçilmez.

Claude yalnız generator'dır. Deterministic QC, DeepSeek semantic judge ve GLM morphology judge
mevcut pipeline tarafından aynen uygulanır. İnsan review, toplam `Codex=300 + Claude=300 = 600`
accepted family tamamlandıktan sonra ayrıca başlatılır. `.env` repo kökünden otomatik okunur;
`.env`, `test/runs/`, cache ve SQLite Git'e eklenmez.

Üreticiler aynı anda çalıştırılmamalıdır. Her batch tamamlanıp pushlandıktan sonra sıradaki kişi
`git pull --ff-only` ile başlamalıdır. Kod her üreticinin kendi 1–300 sırasındaki boşluk ve
çakışmaları engeller; Git üzerinde sırayla çalışma ise iki farklı makinenin aynı eski memory
anlık görüntüsünden üretim yapmasını önler.

## Güvenlik garantileri

- Yeni aralık, mevcut son numaranın tam bir sonrasından başlamazsa reddedilir.
- Aynı slot iki farklı içerikle gelirse `shared-sync` durur.
- Shard veya manifestten biri eksikse üretim başlamaz.
- Shard hash'i, family sayısı ya da tam slot listesi değişmişse manifest doğrulaması kalır.
- Cross-family exact/fuzzy kopyalar birleşimde yeniden kontrol edilir.
- Config, plan, prompt, modeller veya pipeline kodu değişmiş shard'lar aynı benchmark'a karışamaz.
- Her generator sırası en fazla 300 family olabilir.
