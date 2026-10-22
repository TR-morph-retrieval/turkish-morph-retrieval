# MorphTR 600 human-reviewed evaluation — v3.9.1

Bu klasör, insan kontrolünden geçmiş 600-family dondurulmuş benchmark üzerinde
22 Ekim 2026 tarihinde üretilen değerlendirme çıktılarını içerir.

## Provenance

- Dataset: `test/data/final_test/morph_test_600_human_reviewed.json`
- Dataset SHA-256: `53ef9cd3aec930c0d897210922be288e7b97270c9d2d346caee5eb51fd7d139e`
- Evaluation code SHA-256: `ba845f5efc059869f2880ac4d70ef561e6ec7a0ab6ddb10112b91499e467fc05`
- Evaluation API: `3.3`
- Koşunun Git commit'i: `3f97e3a2c17cea8e6a2b687e95f7889cc159061f`
- Seed: `42`; bootstrap: `10.000`

## Tamamlanan encoderlar

Sekiz encoder başarıyla tamamlandı. Kontrollü 11-aday değerlendirmesinde en yüksek
Recall@1, MRR@10 ve nDCG@10 değerlerini BGE-M3 verdi:

| Model | Recall@1 | Recall@3 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BGE-M3 | 0.533 | 0.877 | 0.711 | 0.783 |
| E5-large | 0.473 | 0.847 | 0.669 | 0.752 |
| TR-MTEB-FT-110M | 0.405 | 0.853 | 0.633 | 0.725 |
| GTE-multilingual-base | 0.415 | 0.812 | 0.628 | 0.720 |
| Qwen3-Embedding-8B | 0.357 | 0.767 | 0.582 | 0.684 |
| Jina-Embeddings-v3 | 0.335 | 0.762 | 0.572 | 0.677 |
| ModernBERT-TR | 0.310 | 0.767 | 0.558 | 0.667 |
| Mursit-Base-TR | 0.198 | 0.625 | 0.458 | 0.590 |

## Tamamlanmayan encoderlar

- `Snowflake/snowflake-arctic-embed-m-v2.0`: runtime'da `xformers` kurulu olmadığı için yüklenemedi.
- `google/embeddinggemma-300m`: gated model erişimi/Hugging Face kimlik doğrulaması olmadığı için yüklenemedi.

Bu iki hata `model_errors.json` içinde tam traceback ile korunur. Sonuç cache'leri
yeniden üretilebilir ve yaklaşık 19 MB olduğu için Git'e eklenmemiştir; rapor CSV'leri,
per-query çıktılar, metadata ve hata kayıtları eksiksiz saklanmıştır.
