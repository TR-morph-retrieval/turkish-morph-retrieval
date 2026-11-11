# 5K train üretim sahipliği

## Güncel devir kararı

- İlk 2.820 slot birden fazla üreticinin tamamlanmış shardlarından oluşur; kesin
  kaynak bilgisi ilgili shard manifestlerindedir.
- Kalan `train_002821`–`train_005000` aralığının tamamının üretim sorumlusu
  **Burak**'tır.
- Arda tarafından daha önce üretilip judge/policy hattını geçen beş kısmi kayıt
  `train/data/handoff/arda_2821_2830_judged_partial.jsonl` dosyasında Burak'a
  devredilmiştir.
- Handoff dosyası final shard değildir; Burak ilgili 20'lik final shardı
  oluştururken bu kayıtları aynı slot kimlikleriyle kullanabilir.

## Sözleşme notu

`assignments.json` ve run planı 5.000 slotu kapsar. 2.821–5.000 arasındaki
20'lik blokların tamamı Burak'a atanmıştır. Önceki shardların üretici/provenance
bilgisi değiştirilmemiştir.

Burak her tamamlanan 20-family shardı doğrulayıp ayrı commit olarak `main`
dalına gönderir. Aynı slot başka bir üretici tarafından yeniden üretilmez.
