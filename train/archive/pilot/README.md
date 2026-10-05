# Pilot train arşivi

Bu klasördeki kayıtlar yalnız üretim hattı, maliyet, süre ve LoRA pilot deneyleri için
üretilmiştir. Final train verisine katılmaz.

- `data/`: geçmiş kümülatif pilot JSONL snapshot'ları.
- `runs/`: yerel SQLite, cache ve provider kayıtları; Git tarafından izlenmez.

Gerçek train üretimi `train1250` run-id'siyle temiz `train/runs/` alanında yapılır ve
paylaşılacak shard'lar `train/data/shards/` altında tutulur.
