# E5-large LoRA — 2.000-family MorphTR eğitimi

Bu klasör, 2.000 MorphTR train family'siyle eğitilen `intfloat/multilingual-e5-large`
LoRA koşusunun MorphTR sonuçlarını içerir. Haricî TabiBench sonuçları bu kayda dahil
edilmemiştir.

## Deney yapısı

- Train: 2.000 family (`accepted` ve `human_review` dahil)
- Augmentation: 2.000 ileri + deterministik 1.000 ters yön = 3.000 örnek
- Development: insan-kontrollü V1'den feature-dengeli 100 family
- Final test: insan-kontrollü V1'in kalan 500 family'si
- Model: `intfloat/multilingual-e5-large`
- Seed: 42
- Epoch: 5; en iyi epoch: 5
- Batch size: 32; learning rate: `1e-5`; warmup ratio: 0.1
- LoRA: rank 32, alpha 64, dropout 0.05
- Eğitilebilir parametre: 14.221.312 / 574.111.744 (%2,4771)
- Train loss: 1,340212; süre: 1.084,9 saniye (yaklaşık 18,1 dakika)
- Train source SHA-256: `04404581dab262b9f726e37d2d4637ecfdb60ef5e9941f545c8773b8f219518c`
- Kaynak çalıştırılmış notebook SHA-256: `741ff50f771ebf036f1a148b302f346d4ddb6c7f72588a4b7d674dac9ce50819`
- Yapısal gövdesi olmadığı için dışlanan kayıtlar: `train_001410`, `train_001412`

## Temel sonuç

| Split | Metrik | Zero-shot | LoRA | Mutlak fark |
|---|---|---:|---:|---:|
| Validation 100 | Recall@1 | 0,470 | 0,550 | +0,080 |
| Validation 100 | Recall@3 | 0,790 | 0,840 | +0,050 |
| Validation 100 | MRR@10 | 0,652123 | 0,700040 | +0,047917 |
| Validation 100 | nDCG@10 | 0,737652 | 0,773669 | +0,036017 |
| Final test 500 | Recall@1 | 0,440 | 0,474 | +0,034 |
| Final test 500 | Recall@3 | 0,814 | 0,888 | +0,074 |
| Final test 500 | MRR@10 | 0,641316 | 0,677847 | +0,036531 |
| Final test 500 | nDCG@10 | 0,730071 | 0,758545 | +0,028474 |

Final testte bütün temel sıralama metrikleri yükselmiştir. Özellikle Recall@3'teki
7,4 puanlık ve morfolojik hard pairwise doğruluktaki 4,33 puanlık artış, eğitimin
morfolojik karşıtlıkları ilk sıralarda ayırt etmeyi iyileştirdiğini gösterir.

Dosyalar:

- `morphtr_metrics.csv`: zero-shot ve LoRA MorphTR metrikleri
- `validation_curve.csv`: epoch bazında development eğrisi
- `run_metadata.json`: model, veri ve hiperparametre provenance'ı

