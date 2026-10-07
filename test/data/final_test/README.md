# İnsan kontrollü final test parçaları

Bu klasör, insan kontrolünden sonra düzeltilmiş test parçalarını birleştirme öncesinde
ayrı ve izlenebilir biçimde tutar.

## Murat — Codex 151–300

- `murat_codex_151_300.jsonl`: değerlendirmede kullanılacak 150 düzeltilmiş family.
- `murat_codex_151_300.json`: aynı family'leri kaynak sıraları ve paket metadata'sıyla taşır.
- `murat_codex_151_300.html`: insan inceleme arayüzünün teslim edilen kopyasıdır.

Kontrol sonucu, 150 benzersiz family'nin ve her family'deki `1 positive + 8 hard + 2 easy`
aday yapısının korunduğunu doğruladı. Kaynak review paketine kıyasla 49 family'nin
metin veya anotasyon alanlarında düzeltme vardır. JSONL SHA-256:
`310aced8981475a2c72c5b292efbf68ae999bcd10f06b60f02b2292e5bfc86b1`.

HTML seçimleri tarayıcının `localStorage` alanında tutulduğu için bu üç dosya bağımsız
bir karar günlüğü içermez. Ayrı bir `murat_review.json` gelirse aynı klasörde saklanmalıdır.

## Emir — Codex 1–150

- `emir_codex_001_150_FINAL.json`: insan kontrolü sonrası düzeltilmiş 150 family'yi
  paket metadata'sıyla birlikte taşır.

Kontrol sonucu 150 benzersiz family, her family için `1 positive + 8 hard + 2 easy`
aday yapısı ve geçerli gold kimliği doğrulandı. Kaynak review paketine göre 80 family'de
metin veya anotasyon düzeltmesi vardır. Dosya SHA-256:
`d4f9815361ae4982dd98536bcc1d4190f34aef992d49477c8a42e02497a2ec47`.

## Burak — Claude 1–150

- `burak_claude_001_150.fixed.jsonl`: insan kontrolü sonrası düzeltilmiş 150 family.

Kontrol sonucu 150 benzersiz family, her family için `1 positive + 8 hard + 2 easy`
aday yapısı ve geçerli gold kimliği doğrulandı. Kaynak review paketine göre 24 family'de
düzeltme vardır. Dosya SHA-256:
`f07c6110533e796268e99754d81b3cb6596f455222b8fa145279ee52839983b7`.

## Kuzey — Claude 151–300

- `kuzey_claude_151_300.fixed.jsonl`: insan kontrolü sonrası düzeltilmiş 150 family.

Kontrol sonucu 150 benzersiz family, her family için `1 positive + 8 hard + 2 easy`
aday yapısı ve geçerli gold kimliği doğrulandı. Kaynak review paketine göre 40 family'de
düzeltme vardır. Dosya SHA-256:
`ee0169912d9046852fece85dc773483eb5d14deeb94e8724d563880f3b601e71`.

Hazır insan kontrollü toplam: **600 family**
(Emir 150 + Murat 150 + Burak 150 + Kuzey 150).

## Birleşik final dosya

`morph_test_600_human_reviewed.json`, dört insan-kontrollü parçanın özgün 600-family
sırası korunarak birleştirilmiş kanonik değerlendirme dosyasıdır. Dosyanın `curation`
alanında kaynak dosyalar, reviewer/producer bilgileri ve SHA-256 değerleri bulunur.

- 600 benzersiz family
- 300 Claude + 300 Codex
- 600 positive + 4.800 hard negative + 1.200 easy negative
- SHA-256: `53ef9cd3aec930c0d897210922be288e7b97270c9d2d346caee5eb51fd7d139e`
