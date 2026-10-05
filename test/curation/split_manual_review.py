"""Export four labelled manual-curation packets without changing benchmark records."""
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / 'human_review_600'
ASSIGNMENTS = [('burak','claude',1,150), ('kuzey','claude',151,300),
               ('emir','codex',1,150), ('murat','codex',151,300)]

STYLE = '''body{font:16px system-ui;max-width:1050px;margin:30px auto;padding:20px;background:#fafafa;color:#202020}article{background:white;border:1px solid #ddd;border-radius:12px;padding:20px;margin:20px 0}.warning{background:#fff0cd;padding:12px}li{padding:10px;margin:6px 0;background:#f3f4f5}textarea{width:98%;height:90px}summary,button{cursor:pointer}button,select{padding:10px}header{position:sticky;top:0;background:#fafafa;padding:10px;z-index:2}.gold{background:#e0f5e8}small{color:#555}'''
SCRIPT = '''const key='morph-review-'+document.body.dataset.reviewer;
const fields=[...document.querySelectorAll('textarea,select')];
const old=JSON.parse(localStorage.getItem(key)||'{}');
fields.forEach(f=>{if(old[f.id]!==undefined)f.value=old[f.id];f.oninput=()=>{old[f.id]=f.value;localStorage.setItem(key,JSON.stringify(old))}});
document.querySelector('#save').onclick=()=>{const rows=[...document.querySelectorAll('article')].map(a=>({family_id:a.dataset.id,source_position:Number(a.dataset.position),decision:a.querySelector('select').value,notes:a.querySelector('textarea').value}));const blob=new Blob([JSON.stringify({reviewer:document.body.dataset.reviewer,source_sha256:document.body.dataset.hash,review_kind:'labelled_curation',reviews:rows},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=document.body.dataset.reviewer+'_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
document.querySelector('#filter').onchange=e=>document.querySelectorAll('article').forEach(a=>a.hidden=e.target.checked&&a.dataset.priority!=='true');'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pools = {}
    for producer in ['claude','codex']:
        indexed = {}
        for path in sorted((ROOT/'data'/'final_shards').glob(producer+'_*.jsonl')):
            manifest = json.loads(Path(str(path)+'.manifest.json').read_text())
            rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            for position, row in enumerate(rows, manifest['from']):
                if position in indexed:
                    raise ValueError('Overlapping shard range')
                indexed[position] = row
        if set(indexed) != set(range(1,301)):
            raise ValueError('Incomplete producer coverage')
        pools[producer] = indexed
    seen = set()
    assignments = []
    esc = lambda v: html.escape(str(v))
    for reviewer,producer,start,end in ASSIGNMENTS:
        person_out = OUT / reviewer
        person_out.mkdir(parents=True, exist_ok=True)
        records = [{'source_position':i,'family':pools[producer][i]} for i in range(start,end+1)]
        for row in records:
            fid=row['family']['family_id']
            if fid in seen: raise ValueError('Duplicate assignment')
            seen.add(fid)
        sha=hashlib.sha256(json.dumps(records,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        priority=sum(bool(r['family']['qc'].get('human_review_priority')) for r in records)
        assignments.append(dict(reviewer=reviewer,producer=producer,start=start,end=end,count=len(records),human_review_priority=priority,source_sha256=sha))
        packet=dict(assignments[-1],review_kind='labelled_curation',records=records)
        (person_out/f'{reviewer}_{producer}_{start:03d}_{end:03d}.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2),encoding='utf-8')
        # Preserve original serialized rows, including field order and all provenance.
        original_lines = {}
        for path in sorted((ROOT/'data'/'final_shards').glob(producer+'_*.jsonl')):
            manifest = json.loads(Path(str(path)+'.manifest.json').read_text())
            lines = [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            original_lines.update(enumerate(lines, manifest['from']))
        (person_out/f'{reviewer}_{producer}_{start:03d}_{end:03d}.jsonl').write_text(
            '\n'.join(original_lines[i] for i in range(start,end+1))+'\n', encoding='utf-8')
        cards=[]
        # Warnings first, preserving original source position and family IDs.
        for entry in sorted(records,key=lambda r:(not r['family']['qc'].get('human_review_priority',False),r['source_position'])):
            x=entry['family'];fid=x['family_id'];position=entry['source_position'];flag=bool(x['qc'].get('human_review_priority'))
            candidates=''.join(f'<li class="{"gold" if c["role"]=="positive" else ""}"><small>{esc(c["candidate_slot"])} · {esc(c["id"])}</small><br>{esc(c["text"])}</li>' for c in x['candidates'])
            warning=f'<div class="warning">İnsan kontrolü önceliği: {esc("; ".join(x["qc"].get("human_review_priority_reasons",[])))}</div>' if flag else ''
            judges=esc(json.dumps(x['qc'].get('judging',{}),ensure_ascii=False,indent=2))
            metadata = {k:x.get(k) for k in ['target_feature_label','macro_phenomenon','objective','layer','family_mode','generalization_bucket','domain','register','query_sentence_count','passage_sentence_count','critical_lemma','critical_word_query','critical_word_positive','feature_delta']}
            metadata_html = '<details><summary>Bu family’nin veri yapısı ve hedefi</summary><dl>' + ''.join(f'<dt><strong>{esc(k)}</strong></dt><dd>{esc(v)}</dd>' for k,v in metadata.items()) + '</dl></details>'
            # Insert guidance without changing field IDs/local browser storage.
            warning += metadata_html
            cards.append(f'''<article data-id="{esc(fid)}" data-position="{position}" data-priority="{str(flag).lower()}"><h2>{producer} #{position} · {esc(x['target_feature'])}</h2><small>{esc(fid)}</small>{warning}<p><strong>Query:</strong> {esc(x['query'])}</p><ol>{candidates}</ol><details><summary>Judge güveni ve gerekçeleri</summary><pre style="white-space:pre-wrap">{judges}</pre></details><p><select id="{esc(fid)}-decision"><option value="pending">Henüz incelenmedi</option><option value="approve">Uygun</option><option value="needs_fix">Düzeltme gerekli</option><option value="reject">Family yeniden hazırlanmalı</option></select></p><textarea id="{esc(fid)}-notes" placeholder="Sorunlu aday ID/slotu, gerekçe ve varsa önerilen düzeltme"></textarea></article>''')
        guide = '''<section style="background:#eef4fb;padding:20px;border-radius:12px"><h2>Veri yapısı ve inceleme rehberi</h2>
<p>Bu çalışma, Türkçe eklerin taşıdığı anlamı retrieval encoder’larının ayırt edip edemediğini ölçer. Benchmark toplam 600 family içerir: 300 Codex, 300 Claude. Mevcut ayrımda tamamı sealed testtir; insan kontrolü yapılması sealed olmasına engel değildir. Sealed, model eğitiminde ve ayarlamasında kullanılmaması demektir.</p>
<p><strong>Family:</strong> Bir query ve 11 aday pasajdan oluşan tek değerlendirme birimi. <strong>Query:</strong> Aranan bilgi/anlam. <strong>Gold / positive_01:</strong> Üreticinin ilgili olarak etiketlediği tek aday, yeşil gösterilir. Bu etiket insan incelemesinde doğrulanmalıdır. Query’nin kelimesi kelimesine kopyası olması beklenmez.</p>
<p><strong>hard_01–hard_08:</strong> Konu ve sözcükler yakınken morfoloji, kapsam, zaman, katılımcı veya içerik nedeniyle yanlış olması beklenen 8 aday. <strong>easy_01–easy_02:</strong> Query’yi karşılamayan 2 daha kolay aday; aynı konu veya kişiler geçebilir. Bir ek farkı tek başına negatiflik kanıtı değildir. Adayın daha az ayrıntı vermesi de onu otomatik yanlış yapmaz.</p>
<p><strong>Uzunluk:</strong> Toplam planda query %75 tek, %25 iki cümle; adaylar %30/%30/%30/%10 oranında 1/2/3/4 cümle. Kritik cümle hedef farkın bulunduğu cümledir; diğer cümleler bağlam sağlar. Ek bağlamın gold veya negatifle çelişmediğini de kontrol edin.</p>
<p><strong>76 fenomen, 6 ana grup:</strong> Hâl/konum/yön; iyelik/kişi/sayı; zaman/kip/görünüş/kanıtsallık; olumsuzluk/çatı/katılımcı yapısı; türetim/fiilimsi/allomorfi; ek zinciri kompozisyonu.</p>
<p><strong>Görevler:</strong> morpheme_sensitivity = ek değişince anlam farkını ayırt etme; allomorph_invariance = aynı işlevin farklı yüzey biçimlerinde anlamı koruma; composition = eklerin birlikte taşıdığı anlamı çözme. Allomorf eşdeğerliğini farklı hâl/anlam karşıtlığıyla karıştırmayın.</p>
<p><strong>Family modları:</strong> strict_minimal (%25) = gold ile ana morfolojik hard arasında kontrollü küçük değişim; controlled_diverse (%45) = aynı hedefe yönelik çeşitli doğal tuzaklar; natural_retrieval (%30) = daha serbest doğal anlatımlar. <strong>Holdout:</strong> lemma_holdout kökü, template_holdout cümle kalıbını, composition_holdout belirli ek zincirini eğitimden ayırma planıdır. standard temel gruptur.</p>
<p><strong>Otomatik üretim bandı:</strong> Codex veya Claude üretimi → yerel yapı/uzunluk/kopya kontrolleri → DeepSeek semantik judge → GLM morfoloji judge → gerektiğinde düzeltme. Ayrıntılı gerçek kararlar her family altında bulunur. Judge confidence, kendi kararına verdiği güvendir; ölçülmüş doğruluk yüzdesi değildir.</p>
<p><strong>Sarı uyarı:</strong> human_review_priority; düşük güven, çekimserlik veya kalite/relevance/morfoloji bulgusu nedeniyle önce bakılması önerilen örnek. Uyarı kesin hata anlamına gelmez, uyarısız kayıt da kesin doğru değildir. Tüm 150 family incelenmelidir.</p>
<ol><li>Query ile gold aynı bilgi ihtiyacını karşılıyor mu? Kritik anlam veya katılımcı kaybolmuş mu?</li><li>Hard/easy adaylardan başka biri de ilgili olabilir mi? Özellikle daha genel, eksik ayrıntılı veya farklı kanıtsallıkla aynı olayı anlatan adayları sorgulayın.</li><li>Hedef morfoloji gerçekten ölçülüyor mu? Negatif yalnız anlatım bozukluğu yüzünden kolayca eleniyor mu?</li><li>Cümleler doğal mı? Noktalama, zaman uyumu, bozuk Türkçe karakterler ve bağlam çelişkisi var mı?</li></ol>
<p><strong>Kararlar:</strong> Uygun = family’nin tamamını okudunuz ve sorun görmediniz. Düzeltme gerekli = bir veya birkaç adayda düzeltilebilir sorun var; slot/ID ve gerekçe yazın. Family yeniden hazırlanmalı = query/hedef/gold yapısı temel olarak sorunlu. Emin olmadığınız konuyu not ederek düzeltme gerekli seçebilirsiniz. Judge’a katılmak zorunda değilsiniz.</p>
<p>Bu etiketli düzeltme turudur; kör agreement ölçümü değildir. HTML orijinal veriyi değiştirmez. Notlar tarayıcıda tutulur; düzenli olarak <strong>Kararları JSON olarak indir</strong> ile yedekleyin. Bitince sonucu koordinatöre gönderin. Dosyayı başka konuma taşırsanız yerel notlar görünmeyebilir.</p></section>'''
        page=f'''<!doctype html><html lang="tr"><meta charset="utf-8"><title>{reviewer} · {producer} inceleme</title><style>{STYLE}</style><body data-reviewer="{reviewer}" data-hash="{sha}"><header><strong>{reviewer.title()} · {producer} {start}–{end} · 150 family · {priority} uyarılı</strong><p><button id="save">Kararları JSON olarak indir</button> <label><input type="checkbox" id="filter">Yalnız uyarılıları göster</label></p></header>{guide}{''.join(cards)}<script>{SCRIPT}</script></body></html>'''
        (person_out/f'{reviewer}_{producer}_{start:03d}_{end:03d}.html').write_text(page,encoding='utf-8')
    if len(seen)!=600:raise ValueError('Expected 600 unique assignments')
    (OUT/'assignments.json').write_text(json.dumps(assignments,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'directory':str(OUT),'assignments':assignments},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
