# Final train verisi

Bu klasör yalnız doğrulanmış production shard'larının nihai birleştirilmiş çıktısı içindir.
Pilot veriler burada kullanılmaz. Beklenen birleşik dosya adı `train1250.jsonl`'dir.

Üreticiler önce kendi aralıklarını `train/data/shards/` altında paylaşır; checksum, aralık,
config ve kod sözleşmesi doğrulandıktan sonra tek final dosya oluşturulur.
