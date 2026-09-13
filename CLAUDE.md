# Repository guidance

## Scope

Bu repo Türkçe morphology-aware retrieval projesidir. Yeni çalışma iki net sınır taşır:

- `train/`: legacy model-selection ve Gemini train/dev veri üretimi. Mevcut çıktılar/provenance
  korunur; yeni test kodu buradan örnek metin okumaz.
- `test/`: v3 paper test benchmark'ının tek yetkili generator/review/freeze paketi.

## Test invariants

- 600 sealed final family (validation ayrımı sonraya bırakılmıştır).
- Family başına 1 positive + 8 hard + 2 easy; toplam 11 aday.
- Query %75/%25 (1/2 cümle); pasaj %30/%30/%30/%10 (1/2/3/4 cümle).
- Generator ve blind judge farklı OpenRouter model aileleri.
- Valid allomorph negatif olamaz; allomorph invariance ve morpheme sensitivity ayrıdır.
- Gerçek test örneği generation promptunda few-shot olamaz.
- Reviewer'lar gold/role/subtype görmez. Çözülmemiş disagreement freeze'i engeller.
- `test/runs/` private qrels ve reviewer dosyaları içerdiği için gitignore'dadır.
- Full-corpus testte yalnız own-family gold qrels yeterli değildir; pooled human judgments gerekir.

## Commands

```bash
python3 -m test self-test
python3 -m test plan --run-id test_v3
python3 -m test generate --run-id test_v3_pilot --limit 30
python3 -m test prepare-review --run-id test_v3
python3 -m test merge-reviews --run-id test_v3
python3 -m test finalize --run-id test_v3
```

Legacy train komutları `train/README.md` içindedir ve `train/` çalışma dizininden çalıştırılır.

## Safety and data handling

- API anahtarları yalnız ortam değişkenlerinden okunur; repo veya manifestlere yazılmaz.
- Cache kimliği provider/model/system/prompt/schema/temperature/pipeline sürümünü içerir.
- Test text/qrels değişikliği sessiz in-place edit değil, yeni dataset sürümü gerektirir.
- Türkçe case normalization için `test.validators.tr_lower` kullanılır; düz `str.lower()` ile
  `İ` combining-dot hatası üretmeyin.
