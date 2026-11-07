# Train üretim devri

Arda'nın Gemini/Antigravity üretimi 2.820 numaralı slotta sona erdi. Üretim
sahipliği 2.821–3.000 aralığında Burak'a devredildi.

`arda_2821_2830_judged_partial.jsonl`, Arda'nın daha önce Gemini ile oluşturduğu
2.821–2.830 taslaklarından iki bağımsız LLM judge ve policy gate'i geçen kayıtları
korur. Bu dosya final shard değildir ve train loader tarafından okunmaz. Burak,
2.821–2.840 final shardını hazırlarken bu kayıtları yeniden üretmek yerine aynı
slotlarda kullanabilir:

- `train_002821`: accept
- `train_002822`: accept
- `train_002826`: accept
- `train_002828`: human_review (`low_confidence_pass`)
- `train_002830`: accept

Judge tarafından repair/regenerate/retry istenen 2.823, 2.824, 2.825, 2.827 ve
2.829 final veriye aktarılmamıştır. Bunlar Burak tarafından yeniden üretilmelidir.
