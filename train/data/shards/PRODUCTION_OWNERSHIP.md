# 5K train üretim sahipliği

## Güncel devir kararı

- Arda'nın üretim sorumluluğu `train_002820` ile sona ermiştir.
- `train_002821`–`train_005000` aralığının üretim sorumlusu **Burak**'tır.
- Arda tarafından daha önce üretilip judge/policy hattını geçen beş kısmi kayıt
  `train/data/handoff/arda_2821_2830_judged_partial.jsonl` dosyasında Burak'a
  devredilmiştir.
- Handoff dosyası final shard değildir; Burak ilgili 20'lik final shardı
  oluştururken bu kayıtları aynı slot kimlikleriyle kullanabilir.

## Sözleşme notu

Mevcut `assignments.json` ve yerel run planı 3.000 slotla sınırlıdır. Bu nedenle
dosyadaki resmî bloklar şimdilik 2.821–3.000 aralığını Burak'a atar. Plan 5.000'e
genişletildiğinde 3.001–5.000 blokları da Burak adına oluşturulmalıdır; yeni bir
üretici atanmadıkça bu sahiplik değiştirilmez.

Burak her tamamlanan 20-family shardı doğrulayıp ayrı commit olarak `main` dalına
gönderir. Aynı slot başka bir üretici tarafından yeniden üretilmez.
