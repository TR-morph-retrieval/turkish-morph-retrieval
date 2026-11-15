# 2.500 Train Slotu İnsan İncelemesi (Human Review)

İlk 2.500 train slotu içindeki 968 `human_review` adayı üç incelemeci arasında dağıtılmıştır:
- **Burak**: 323 slot (`burak_train_review.zip`)
- **Emir**: 322 slot (`emir_train_review.zip`)
- **Murat**: 323 slot (`murat_train_review.zip`)

Kaynak SHA-256: `afb81c71ca505bc1c12f51127a434a4ea6e896d9ebb724d5870f6cff9f1b620d`

## Murat İnceleme Sonuçları (323 / 323 Tamamlandı)

- **Pass**: 269 family (%83.3) — Morfolojik ve semantik gereksinimleri karşılayan, doğal Türkçe kullanıma uygun kayıtlar.
- **Düzelttim (fix)**: 53 family (%16.4) — Yazım, biçimbirim sınırları, ek ayrımı veya aday metinleri MorphTR sözleşmesine uygun biçimde düzeltildi ve `validate_family` ile doğrulandı.
- **Reject**: 1 family (%0.3) — `train_001062` (`RECP`): Query kökü (`anlaş-`) ile aday kökleri (`uzlaş-`) arasındaki uyumsuzluk ve Türkçede bulunmayan yapay `uzladılar` sözcüğü nedeniyle reddedildi.

### Teslim Edilen Dosyalar

- `murat_train_human_review.json`: 323 inceleme kararını içeren nihai JSON çıktısı.
- `murat_train_review_completed.zip`: Karar JSON'ı, form verileri işlenmiş HTML arayüzü ve görev atama dosyasını içeren arşiv.
