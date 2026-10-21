# Test V2 extension shards

Bu klasör, insan-kontrollü 600-family V1 test setine eklenecek **200 family**
uzantısını içerir (`codex_601_800.jsonl` aralıkları).

- Run: `extension_v40`
- Generator: GPT-5.6 Sol, medium reasoning
- Semantic judge: DeepSeek V4 Flash
- Morphology judge: GLM-5.3 Flash, low reasoning
- Durum: generator, deterministik kontroller, iki LLM judge ve policy gate geçmiş;
  final insan kontrolü henüz uygulanmamıştır.

Tüm JSONL dosyaları birlikte 200 benzersiz `slot_id` içerir. DB'deki accepted
kümesiyle birebir eşleşir. İnsan kontrolü tamamlandıktan sonra V1 + V2 toplamı
800 family olarak dondurulacaktır.
