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
