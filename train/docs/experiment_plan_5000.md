# MorphTR 5K train ve ölçekleme deney planı

## Sabit hedef

- Final train havuzu: **5.000 temiz ve benzersiz family**
- `train_002821`–`train_004500` üretim sorumlusu: **Burak**
- `train_004501`–`train_005000` üretim sorumlusu: **Murat**
- Final değerlendirme havuzu: **1.200 insan-kontrollü family**
  - Development: 200 family
  - Sealed test: 1.000 family
- Test verisi train üretimine, filtre kararlarına veya hiperparametre seçimine sızdırılmaz.

## Train veri dondurma

İnsan kontrolü ve düzeltmeler tamamlandıktan sonra veri
`train5000_final.jsonl` adıyla dondurulur. Manifestte kaynak shardlar ve SHA-256
değerleri, family/fenomen/üretici dağılımları, insan kontrolü özeti ve leakage
denetimi sonucu tutulur.

## Ölçekleme deneyi

Aynı dondurulmuş 5K havuzdan fenomen ve üretici dağılımını koruyan,
deterministik ve iç içe alt kümeler seçilir: **1K ⊂ 2.5K ⊂ 5K**.

| Koşul | Train family | Amaç |
|---|---:|---|
| Small | 1.000 | Düşük veri rejimi |
| Medium | 2.500 | Güçlü temel |
| Main | 5.000 | Paper ana modeli |

## Simetri deneyi

Scaling curve başlangıçta bütün boyutlarda aynı **%50 kısmi simetri** ayarıyla
çalıştırılır:

| Train family | İleri çift | Ters çift | Toplam çift |
|---|---:|---:|---:|
| 1.000 | 1.000 | 500 | 1.500 |
| 2.500 | 2.500 | 1.250 | 3.750 |
| 5.000 | 5.000 | 2.500 | 7.500 |

Simetrinin katkısı yalnız 2.5K üzerinde `%0`, `%25`, `%50` ve gerekirse `%100`
ablation ile ölçülür. MorphTR kazanımı ve genel retrieval korunumu birlikte
değerlendirilir. En iyi oran seçildikten sonra ana 5K modeli o oranla eğitilir.

## Raporlama

- Checkpoint seçimi yalnız 200-family development üzerinde yapılır.
- 1.000-family sealed test, model ve hiperparametre kararı tamamlandıktan sonra açılır.
- Ana sonuçlar mümkünse üç seed için ortalama ve standart sapmayla verilir.
- Base ve LoRA modelleri hem MorphTR hem genel Türkçe retrieval görevlerinde karşılaştırılır.
- Temel ablationlar: veri ölçeği, simetri oranı ve insan-düzeltilmiş veri katkısıdır.
