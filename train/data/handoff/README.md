# Kuzey train devri (1001–1250)

`kuzey_partial_1001_1100.jsonl`: kuzey aralığında kabul edilmiş **69 family**
(30 `accept`, 39 `human_review`). Satırlar `shard-export` çıktısıyla birebir aynı
envelope'lardır. Run `train1250`, contract `a2dbeeec…`, config `train-v26-drift-aware-review`;
generator OpenRouter üzerinden Gemini 3.8 Flash low (`workflow.py run`). Maliyet $6,14.

Dosya `shards/` dışında duruyor: 20'lik aralıklar eksik olduğu için `shard-export` ve
`validate_shards` bunu shard olarak kabul etmez. Notebook da yalnız `train/data/shards/`
okuduğu için bu satırlar eğitime iki kez girmez.

- Eksik slotlar (31 adet, 1001–1100 içinde): manifest'teki `missing_slot_ids`.
- 1101–1250 hiç başlamadı.

## Devam etmek için

Kendi `train1250` run'ına (2250'ye uzatılmış plan) bu satırları al. Bu, `shard-sync`'in
yaptığı güncellemenin aynısı:

```python
import json, sqlite3
db = sqlite3.connect('train/runs/train1250/state.sqlite3')
for line in open('train/data/handoff/kuzey_partial_1001_1100.jsonl', encoding='utf-8'):
    row = json.loads(line)
    db.execute("UPDATE jobs SET status='accepted', result=? WHERE id=? AND status!='accepted'",
               (json.dumps(row, ensure_ascii=False), row['slot_id']))
db.commit()
```

Sonra eksikleri üret ve her 20'lik aralığı `--producer kuzey` ile çıkar:

```bash
python3 train/workflow.py run --run-id train1250 --from-index 1001 --to-index 1100 --limit 31 --max-calls 600
python3 train/workflow.py shard-export --run-id train1250 --producer kuzey --from-index 1001 --to-index 1020 \
  --output train/data/shards/kuzey_1001_1020.jsonl   # 1021–1040 … 1241–1250 için aynı şekilde
```

Beş shard da `shards/` altına girince bu `handoff/` klasörünü silmek yeterli.

## Not

- Zor slotlar: `train_001009` (ALLO.DAT, natural_retrieval) ve `train_001018` (EVID.COND.NEG)
  27+ denemede kabul edilmedi.
- `workflow.py` içindeki kabul öncesi kilitli kontrol `guard.check()` sonucunun tamamına
  bakıyor. Bu yüzden policy'nin `human_review` olarak kabul ettiği, yalnız soft quality
  uyarısı taşıyan family'ler commit anında reddediliyor (`concurrent_duplicate_or_guard_failure`).
  1018'de bu üç kez oldu. `agent_generator.py` yolunda bu kontrol yok. Kod değişikliği
  contract'ı değiştireceği için dokunulmadı.
