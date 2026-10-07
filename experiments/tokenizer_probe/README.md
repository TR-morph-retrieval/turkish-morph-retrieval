# Tokenizer probe: mE5-large + morfem sınırları

`purpose=tokenizer_probe_only`. 600 sealed family bu deneyde eğitim/seçim için kullanılır (5-katlı döngüsel
train/val/test); çıktılar sealed-test sonucu değildir ve paper'da öyle raporlanmaz. `test/` ve `train/` değişmez.

Soru: TurkishTokenizer ya da Morpheus morfem sınırları, mE5-large'a embedding'e dokunmadan verilirse
(sınır işareti + `Split` pre-tokenizer yaması) küçük-veri LoRA'da fark yaratır mı? Bu, tam tokenizer değiştirme
ve sürekli ön-eğitim rejimini yanıtlamaz.

## Sıra
1. `python3 -m experiments.tokenizer_probe.split`: bölme + `artifacts/phenomenon_coverage.csv` (GPU yok).
2. Segmentasyon önbelleği (izole venv: `python3 -m venv .venv-seg`, `pip install turkish-tokenizer huggingface_hub torch numpy`,
   `git clone https://github.com/lonewolf-rd/TurkishMorpheus .vendor/TurkishMorpheus`):
   `.venv-seg/bin/python -m experiments.tokenizer_probe.segment_cache --tokenizer tt|mph`
3. `python3 -m experiments.tokenizer_probe.audit` (`.venv-tok`: `tokenizers`, `transformers`): tokenizasyon denetimi.
4. Colab: `tokenizer_probe_colab.ipynb` (smoke, zero-shot, 15 LoRA koşusu, rapor).
5. Testler: `python3 -m unittest discover -s experiments/tokenizer_probe -p 'test_*.py'` (tokenizer testleri `.venv-tok` ile çalışır).

## Notlar
- `artifacts/` sealed veriden türediği için git'e girmez (`.gitignore`); Colab için Drive'a `MyDrive/tokenizer_probe_artifacts` olarak kopyalanır.
- Sealed JSON hash-pinli; 3 ailedeki bozuk karakter (mojibake, 113 metin) yalnız bellekte onarılır.
- TurkishTokenizer token'ları kanonik (`-lar` = ler/lar); yüzey sınırları token öneklerinin decode'undan çıkarılır, uyuşmayan
  kelimeler (yaklaşık %5) segmentsiz kalır. Allomorf kanonikleştirme bu deneyde test edilmez.
- Kontrol kolları (`shift_*`, `rand_*`) morfem kollarından yaklaşık iki kat fazla parçalanma üretir (denetim tablosu);
  morfolojiye atıf için daha sıkı bir kontrol (eşit uzunluk artışlı SentencePiece örneklemesi) Aşama C'de gerekebilir.
- Karar kuralı `report.py` içinde ön-kayıtlıdır (ΔMRR@10 ±0,03 bandı, lemma-kümeli bootstrap, Holm-McNemar).
