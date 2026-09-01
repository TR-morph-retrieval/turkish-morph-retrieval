# Morphology-Aware Contrastive Fine-Tuning for Turkish Retrieval

Türkçe eklerin taşıdığı anlamı retrieval embedding'lerinde korumayı hedefleyen dual-encoder
araştırma projesi. Repo artık veri yaşam döngüsünü iki bağımsız parçaya ayırır:

| Dizin | Amaç |
|---|---|
| [`train/`](train/) | Eski Gemini train/dev generator'ı, model-selection araçları ve v2.0–v2.2 JSON geçmişi. Bu bölüm korunmuş legacy sistemdir. |
| [`test/`](test/) | 100 development + 500 final test; iki generator, bağımsız LLM judge, otomatik QC/qrels/freeze. |
| [`test/notebooks/morph_baseline_eval_pilot5_colab.ipynb`](test/notebooks/morph_baseline_eval_pilot5_colab.ipynb) | Resmî shard'ların ilk beş family'sini hızlıca değerlendiren küçük notebook. |
| [`test/notebooks/morph_baseline_eval_300_colab.ipynb`](test/notebooks/morph_baseline_eval_300_colab.ipynb) | İlk 300 Codex family'si için 50 dev + 250 sealed ara değerlendirmesi. |
| [`test/notebooks/morph_baseline_eval_600_colab.ipynb`](test/notebooks/morph_baseline_eval_600_colab.ipynb) | 100 development + 500 final için eksiksiz paper değerlendirmesi. |

Eski 50-family JSON ve önceki veri sürümü
[`train/legacy_test_data/`](train/legacy_test_data/) altında provenance amacıyla saklanır. Yeni test
generator'ı bu metinleri few-shot olarak kullanmaz.

## Yeni test kararı

- Toplam **600 bağımsız contrast family**: 100 development + 500 final test.
- Her family: 1 query, **1 positive + 8 hard negative + 2 easy negative**.
- Query uzunluğu: **%75 bir, %25 iki cümle**; tek ve açık bir bilgi ihtiyacı.
- Aday pasaj uzunluğu: **%30/%30/%30/%10 oranında 1/2/3/4 cümle**. Kritik cümle konumu
  dengelenir; aynı family'deki diğer tam bağlam cümleleri bütün adaylarda aynıdır.
- Standard, lemma-holdout, template-holdout ve compositional-holdout dilimleri.
- Allomorph invariance ile anlam değiştiren morfem karşıtlığı ayrı objective'lerdir.
- 6 macro grup altında 76 fenomen; bağlamsal morfolojik belirsizlik, IG/türetim zinciri,
  hâl-rol çerçevesi, askıda ekleme ve çok sözcüklü morfoloji dahil.
- 150 strict minimal-pair family; iki generator finalde 300+300 dengelenir.
- 600 dengeli slot → deterministic QC → farklı model ailesinden blind judge. Kalan örnek aynı
  slotun kotasını koruyan taze üretimle hemen değiştirilir → hash/manifest ile freeze.
- Qrels family oluşturulurken hazırdır: gold `1`, aynı family'deki 10 negatif `0`; full-corpus
  retrieval farklı semantic frame'lerdeki bütün test belgelerini ortak corpus olarak sıralar.

Ayrıntılı şema, uyarlanabilir hard-negative taksonomisi ve komutlar:
[`test/README.md`](test/README.md). Daha kısa başlangıç özeti:
[`test/README_SHORT.md`](test/README_SHORT.md).

## Hızlı doğrulama

Yeni test kodu planlama/QC tarafında yalnız Python standart kütüphanesini kullanır:

```bash
python3 -m test self-test
python3 -m test plan --run-id test_v39_final
```

API üretimi için:

```bash
export OPENROUTER_API_KEY="..."
# Opsiyonel override; varsayılanlar gpt-5.6-sol ve claude-opus-5 (medium).
# export TEST_CODEX_GENERATOR_MODEL="gpt-5.6-sol"
# export TEST_CLAUDE_GENERATOR_MODEL="claude-opus-5"
export TEST_SEMANTIC_JUDGE_MODEL="deepseek/deepseek-v4-flash-0731"
export TEST_MORPHOLOGY_JUDGE_MODEL="z-ai/glm-5.3-flash"
python3 -m test generate --run-id test_v39_final
```

GLM 5.3 Flash morphology judge zorunlu `low` reasoning ile çalışır; reasoning metni çıktıya eklenmez.

Generator'lar Codex ve Claude abonelik CLI'larını, bağımsız judge'lar OpenRouter API'sini kullanır.
Model kimlikleri, prompt/config sürümü, request hash'leri, token kullanımı ve git commit'i run
manifestinde tutulur.

Legacy train sistemi için önce `cd train`, ardından [`train/README.md`](train/README.md) içindeki
komutları kullanın.

## Paper değerlendirme katmanları

Paper iki tamamlayıcı ana benchmark sonucu raporlar; full-corpus retrieval yalnız tanısal bir
ek değildir:

1. Her query'nin kendi 11 adayı üzerinde `Recall@1/3`, MRR@10 ve nDCG@10.
2. Ortak 5.500 test dokümanında full-corpus `Recall@1/3/10/50`, MRR@10 ve nDCG@10.
3. Genel retrieval yeteneğinin korunması için harici Turkish-BEIR/TR-MTEB sonuçları.

## Lisans

MIT — [`LICENSE`](LICENSE).
