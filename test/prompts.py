"""Zero-shot generation, repair and blind-judge prompts.

No real development/test item is ever used as a few-shot exemplar.
"""

from __future__ import annotations

import hashlib
import json
import random

PROMPT_VERSION = "test-prompts-3.9.3-batch3"

GENERATOR_SYSTEM = """\
Sen Türkçe biçimbilim ve bilgi erişimi için contrast-set yazan uzman bir veri küratörüsün.
Görevin doğal, gündelik veya kurumsal Türkçe ile tek bir sorgu ve tam 11 adaydan oluşan bir family
üretmektir. Çıktı yalnızca istenen JSON şemasına uymalıdır. Meta-arama dili, soru cümlesi,
çeviri kokan ifade, yapay dolgu ve bozuk ek kullanımı yasaktır. Aynı family içindeki uzunluk,
üslup ve ayrıntı yoğunluğu dengeli olmalıdır.
"""

SEMANTIC_JUDGE_SYSTEM = """\
Sen generator'dan bağımsız, tamamen özellik-kör bir Türkçe retrieval hakemisin. Gold, negatif,
biçimbilim fenomeni ve alt-tür etiketlerini görmeyeceksin. Yalnız metinsel anlam, sorguya destek,
doğal Türkçe, iç tutarlılık, katılımcı/olay rolleri ve yüzey artefaktlarını değerlendir. Bilmediğin durumda abstain=true
ver; güven puanını şişirme. JSON dışında metin yazma.
"""

MORPHOLOGY_JUDGE_SYSTEM = """\
Sen generator ve semantik hakemden bağımsız, özellik-bilinçli bir Türkçe biçimbilim hakemisin.
Gold/hard/easy ve alt-tür etiketlerini görmeyeceksin. Semantik relevance veya doğru cevap seçme;
yalnız hedef morfolojik özelliği, eklerin doğallığını, ek zincirini, kapsamı ve allomorf işlevini
değerlendir. Hâl ve çatı işaretlerinin agent, patient, recipient, source ve causer rollerine etkisini;
türetim sınırlarında kök/son sözcük türünü ayrıca izle. Geçerli bir allomorfu yalnız yüzeyi değiştiği
için yanlış sayma. Bilmediğin durumda
abstain=true ver. JSON dışında metin yazma.
"""

ADJUDICATOR_SYSTEM = """\
Sen iki bağımsız hakemin anlaşmazlığını inceleyen kıdemli Türkçe retrieval ve biçimbilim
adjudicator'ısın. Üretim etiketlerini veya gold bilgisini görmeyeceksin. Hakem cevaplarını kanıt
değil görüş olarak ele al; metni kendin kontrol et. Çıktın yalnız insan incelemesine tavsiyedir ve
otomatik gold değiştirmez. JSON dışında metin yazma.
"""


_HARD_DESCRIPTIONS = {
    "minimal_morph_negative": "Sorguya çok yakın; yalnız hedef morfolojik özellik değiştiği için anlam yanlış.",
    "controlled_morph_negative": "Aynı bilgi ihtiyacı çevresinde doğal farklı anlatım; hedef morfolojik anlam yanlış.",
    "same_lemma_wrong_inflection": "Pozitifle aynı kritik lemma; farklı ve ilgili bir çekim anlamı yanlış yapıyor.",
    "related_feature_negative": "Hedefe komşu ikinci bir morfolojik özellik yanlış; hedef karşıtlığın kopyası değil.",
    "same_morph_wrong_content": "Hedef morfoloji doğru; nesne/olay/referans yanlış.",
    "state_participant_time_trap": "Katılımcı, kişi, zaman veya gerçekleşme durumu yanlış.",
    "close_paraphrase_wrong_meaning": "Sözcüksel olarak yakın bir yeniden yazım; temel önerme yanlış veya çelişik.",
    "argument_role_reversal": "Kim-kime-ne yaptı rolleri ters veya yanlış bağlanmış.",
    "scope_attachment_trap": "Olumsuzluk, kip, iyelik veya ek zincirinin kapsamı yanlış yerde.",
    "morph_distractor": "Hedef biçimbilim metinde var ama yanlış yükleme/olaya bağlı; sorguyu yanıtlamıyor.",
    "partial_chain_negative": "Ek zincirinin yalnız bir bölümü doğru; zincirin tamamı sorgunun anlamını vermiyor.",
    "allomorph_form_function_trap": "Yüzeyce benzer ek/biçim var; fakat gramatik işlev geçerli allomorf eşdeğeri değil.",
    "noun_possessor_number_trap": "Adın/nesnenin çoğulluğu ile sahibin çoğulluğu karıştırılmış; örneğin çok nesne ile çok sahip aynı şey değildir.",
    "lexical_retrieval_trap": "Query sözcüklerini güçlü biçimde taşır fakat bilgi ihtiyacını karşılamaz.",
    "semantic_retrieval_hard": "Konu ve anlam alanı yakındır fakat gerekli önerme veya kanıt yoktur.",
}


def _hard_rules(profile: list[dict]) -> str:
    return "\n".join(
        f"  - {item['slot']} -> {item['subtype']}: {_HARD_DESCRIPTIONS[item['subtype']]} "
        f"Odak: {item['focus']}."
        for item in profile
    )


def build_generation_prompt(slot: dict) -> str:
    feature = slot["feature"]
    special_rule = {
        "COP.NEG": (
            "İsim/sıfat yüklemli bir cümle kur. `değil` ile kurulan copular olumsuzluğu fiildeki "
            "-mA olumsuzluğuyla karıştırma; bütün adaylar doğal ve dilbilgisel olsun."
        ),
        "COP.TAM": (
            "Aynı isim/sıfat yüklemi koru; `idi`, `imiş` ve `ise` arasındaki geçmiş, aktarma/çıkarım "
            "ve koşul farkını ölç. Bunları aynı işlevin allomorfları sayma."
        ),
        "Q.PART.SCOPE": (
            "Bu tek istisnada query doğal bir mı/mi/mu/mü sorusu OLMALI. Parçacığın bağlandığı "
            "öge odağı belirler: özne odağı ile nesne/zarf odağını aynı sayma. Önerme ve sözcükler "
            "mümkün olduğunca sabit, odak farkı ise açık olsun."
        ),
        "NMLZ.MA_VS_DIK": (
            "-mA burada adlaştırıcıdır, olumsuzluk eki değildir. İstek/planlanan olay okumasını "
            "-DIK ile gerçekleşmiş olgu bilgisi okumasından ayır."
        ),
        "REL.GEN.POSS": (
            "-DIK'lı ilgi yapısında genitif özne ile iyelik işaretinin kişi/sayı bağını koru. "
            "Negatifleri bozuk uyumla değil, doğal fakat başka possessor/gönderimle kur."
        ),
        "ANAPHOR.AGR": (
            "`kendi` biçiminin kişi/sayı işaretini ve hangi katılımcıya döndüğünü birlikte izle; "
            "kendimiz/kendiniz gibi biçimleri yalnız yüzey benzerliğiyle eşdeğer sayma."
        ),
        "MORPH.CONTEXT_AMBIG": (
            "Positive ve ana tuzakta aynı yüzey biçimini kullan; lemma, sözcük türü veya çekim analizi "
            "yalnız cümle bağlamıyla ayrışsın. Örneği yalnız sözlük anlamı farkına indirgeme; en az bir "
            "gerçek morfolojik analiz belirsizliği bulunsun. Bu fenomen strict minimal-pair değildir."
        ),
        "DERIV.IG_CHAIN": (
            "Kök sözcük türünü, her yapım adımını ve son sözcük türünü planla. Positive hedef türetim "
            "zincirini korusun; negatif farklı bir türetim sınırı/yolu yüzünden anlam veya valency "
            "değiştirsin. Yalnız benzer harf dizisini türetim kanıtı sayma."
        ),
        "CASE.ROLE.FRAME": (
            "Aynı katılımcıları ve temel olayı mümkün olduğunca koru; hâl işaretleri üzerinden agent, "
            "patient/theme, goal/recipient ve source rollerini açıkça ayırt et. Negatif doğal ve "
            "dilbilgisel kalmalı; anlamsız hâl dizisi kullanma."
        ),
        "SUSP.AFFIX": (
            "Koordineli ögelerde çekimin yalnız son eşlenikte yüzeyleştiği fakat doğru kapsamda diğer "
            "eşleniğe de yayıldığı doğal bir yapı kur. Positive ile tuzak, paylaşılan ek kapsamı veya "
            "katılımcı rolleri bakımından ayrışsın; noktalama hilesi kullanma."
        ),
        "MWE.MORPH": (
            "Destek fiilli, kalıplaşmış ya da tekrarlı çok sözcüklü ifadeyi bütün olarak kur. "
            "Morfolojik işaretin hangi bileşene bağlandığı ve ifadenin bütüncül anlamı birlikte "
            "korunsun; yalnız ortak kelimeleri kopyalayan aday positive olmasın."
        ),
    }.get(feature["key"], "Hedef feature'ın verilen anlam karşıtlığını doğal Türkçe içinde açık tut.")
    query_rule = (
        "Query doğal bir mı/mi/mu/mü odak sorusu olmalı; başka meta-arama dili kullanma."
        if feature["key"] == "Q.PART.SCOPE"
        else "Query doğal bir durumu İDDİA etmeli; evet/hayır sorusu ve ‘kaydı bul/arıyorum’ dili kullanma."
    )
    allomorph_rule = (
        "Bu bir ALLOMORPH INVARIANCE family’sidir: pozitif, aynı gramatik işlevi farklı geçerli "
        "yüzey biçimiyle korumalıdır. Geçerli allomorf hard/easy negatif OLAMAZ. Negatifler başka "
        "bir morfem veya başka bir anlam yüzünden yanlış olmalıdır."
        if slot["objective"] == "allomorph_invariance"
        else "Bu family morfem duyarlılığını ölçer: pozitif hedef anlamı korur; minimal negatifte "
             "morfolojik değişim gerçekten anlamı değiştirmelidir."
    )
    bucket_rule = {
        "standard": "Train'e benzer doğal zorlukta, fakat hiçbir metnin kopyası olmayan bir örnek üret.",
        "lemma_holdout": "Nadir olmayan fakat üretim boyunca tekrar edilmeyecek doğal bir kritik lemma seç.",
        "template_holdout": "Verilen soyut sözdizimi kalıbını doğal ve özgün biçimde gerçekleştir.",
        "composition_holdout": "Bileşenleri anlaşılır, tam ek zinciri kapsam/sıra bakımından ayırt edici olsun.",
    }[slot["generalization_bucket"]]
    domain_rule = (
        "Bu family ayrıca domain_shift etiketli: domain + register birleşimini doğal biçimde "
        "gerçekleştir; bu birleşim gelecekteki train exclusion manifestine girecek."
        if "domain_shift" in slot["generalization_tags"]
        else "Domain/register doğal çeşitlilik içindir; yapay jargon ekleme."
    )
    strict_rule = (
        "Bu STRICT MINIMAL-PAIR family’sidir. Positive ile hard_01 aynı kritik lemmayı, aynı "
        "sözdizimsel şablonu ve aynı sözcük dizisini korumalı; yalnız kritik çekimli sözcüğün "
        "hedef eki/ek zinciri değişmelidir. Noktalama dahil diğer tokenlar aynı kalmalıdır."
        if slot["strict_minimal_pair"] else
        "Bu family strict minimal-pair slice'ında değildir. hard_01 yine yerel ve kontrollü olsun."
    )
    mode_rule = {
        "strict_minimal": (
            "STRICT mod: positive–hard_01 tek hedef biçim dışında aynıdır. Diğer hard'lar yakın "
            "olabilir. En az 4 hard query ile gold kadar lexical örtüşsün; median fark en çok 0.15."
        ),
        "controlled_diverse": (
            "CONTROLLED DIVERSE mod: aynı olay/katılımcılar ve hedef karşıtlık korunur; gold ve "
            "hard'lar farklı doğal sözdizimleri kullanabilir. hard_01 minimal olmak zorunda değildir. "
            "En az 3 hard gold kadar lexical örtüşsün; median fark en çok 0.25."
        ),
        "natural_retrieval": (
            "NATURAL RETRIEVAL mod: query, gold ve negatifler doğal olarak farklı kurulabilir. "
            "Gold query'nin bilgi ihtiyacını karşılayan TEK passage; hard'lar konu/kelime bakımından "
            "çekici fakat gerçekten irrelevant olmalı. En az 2 hard gold kadar lexical örtüşsün; "
            "median fark en çok 0.35."
        ),
    }[slot["family_mode"]]
    query_expression_rule = {
        "morph_explicit": (
            "Query hedef morfolojik anlamı açık bir çekimli biçimle ifade etsin; surface kopyası "
            "zorunlu değildir."
        ),
        "semantic_paraphrase": (
            "Query hedef morfolojik anlamı farklı sözcük/sözdizimiyle ifade etsin; gold'daki kritik "
            "çekimli biçimi kopyalamasın. critical_word_query bu anlamı taşıyan query ifadesidir."
        ),
    }[slot["query_expression"]]
    lexical_rule = {
        "high": "Query–gold sözcük Jaccard'ı 0.55–1.00 aralığında, fakat tek-kelime kopyası değil.",
        "medium": "Query–gold sözcük Jaccard'ı 0.30–0.70 aralığında olsun.",
        "low": "Query–gold sözcük Jaccard'ı 0.00–0.45 aralığında; semantik bağ yine açık olsun.",
    }[slot["query_gold_lexical_band"]]

    compact_slot = {
        "semantic_frame_id": slot["semantic_frame_id"],
        "feature": feature["key"],
        "objective": slot["objective"],
        "strict_minimal_pair": slot["strict_minimal_pair"],
        "family_mode": slot["family_mode"],
        "query_expression": slot["query_expression"],
        "query_gold_lexical_band": slot["query_gold_lexical_band"],
        "domain": slot["domain"],
        "register": slot["register"],
        "template_id": slot["template"]["id"],
    }
    # A replacement round must not replay the previous round's cached response: the cache key is
    # the prompt hash, so the nonce has to reach the prompt text or refills 2+ re-read round 1's
    # answer and re-fail validation identically. Only emitted when non-zero, so first-round
    # prompts (and their caches) stay byte-identical.
    if int(slot.get("refill_round", 0)):
        compact_slot["refill_round"] = int(slot["refill_round"])
    if slot.get("dataset_memory"):
        compact_slot["dataset_memory"] = slot["dataset_memory"]

    return f"""\
SLOT
{json.dumps(compact_slot, ensure_ascii=False, indent=2)}

HEDEF
- Bir query ve tam 11 aday: 1 positive + 8 hard_negative + 2 easy_negative.
- Query tam {slot['query_sentence_count']} cümle olmalı ve tek bir bilgi ihtiyacını ifade etmeli.
- Query iki cümleyse ikinci cümle yeni bir olay/niyet açmamalı; ilk önermeyi doğal biçimde sınırlandırmalı.
- Her aday pasajı tam {slot['passage_sentence_count']} cümle olacak.
- `context_sentences` tam {slot['passage_sentence_count'] - 1} doğal TAM cümle içermeli.
- Her candidate yalnız bir TAM `critical_sentence` üretmeli. Kod bu cümleyi pasajın
  {slot['critical_sentence_position']}. konumuna yerleştirip aynı `context_sentences` cümlelerini
  diğer konumlara byte-identical olarak koyacak.
- Ortak bağlam tek başına sorguyu yanıtlamamalı ve adaylar arasındaki doğru/yanlış ayrımını
  değiştirmemeli; ayrım yalnız `critical_sentence` içinde yerel kalmalı.
- Adayların token uzunlukları ve ayrıntı yoğunluğu birbirine yakın olmalı. Gold sistematik olarak en uzun olamaz.
- {query_rule}
- Positive sorgudaki aynı bilgi ihtiyacını/önermeyi karşılamalı; fakat query'nin tek sözcüğü
  değiştirilmiş kopyası OLMAMALI. Yalnız `yolculuklarda → seyahatlerinde` gibi bir eşanlamlı
  değişimi yeterli değildir. En az iki anlam-koruyan ifade değişikliğiyle birlikte doğal bir
  sözdizimsel yeniden kurulum (öge sırası, yan cümle, çatı veya anlatım yapısı) kullan.
- Query ile positive aynı doğruluk koşullarında iki yönlü eşdeğer olmalı; yalnız tek yönlü çıkarım
  yeterli değildir. Üst/alt kavramla kapsamı daraltma veya genişletme (`yönetici → müdür` gibi),
  rolü, türü, miktarı ya da kesinliği özelleştirme positive için yasaktır.
- Strict moddaki kontrollü minimal çift query–gold değil, positive–hard_01 arasındadır. Diğer
  modlarda hard_01 hedef morfolojik karşıtlığı korur fakat farklı sözdizimi kullanabilir.
- {mode_rule}
- {query_expression_rule}
- {lexical_rule}
- `equivalence_positive` tek positive alt-türüdür.
- İki easy_negative rastgele ve bütünüyle alakasız cümle OLAMAZ. Aynı domain/register ve mümkünse
  aynı kişi, kurum, yer veya konu ipuçlarından en az birini korusun; ancak farklı bir olay, ilişki
  ya da bilgi ihtiyacı anlatsın ve query'yi kesinlikle yanıtlamasın. Örneğin hastane sorgusunda
  meteoroloji haberi değil, aynı doktorun farklı bir hastaya randevu vermesi uygun easy'dir.
- Easy aday, başka bir query için yeniden kullanılabilecek bağımsız bir gold cümlesi gibi yazılmamalı;
  bu family'nin bağlamına özgü bir off-intent distractor olmalı. Başka candidate/gold cümlesini kopyalama.

LEXICAL ARTEFAKT KONTROLÜ
- Gold, yalnız query sözcüklerini daha çok kopyaladığı için bulunamamalı. Positive ve hard'ların
  kritik cümlelerindeki query-word overlap düzeyleri yakın olmalı; gold sistematik olarak en yüksek olamaz.
- Modun istediği sayıda hard query ile aynı kişi, nesne, olay ve temel içerik ipuçlarını korumalı;
  strict/controlled/natural için bu alt sınır sırasıyla 4/3/2'dir.
- Positive ile hard'ların lexical dengesi mod kuralındaki karşılaştırmalı kapıya uymalı.
- Bu dengeyi gold'u sürekli daha düşük overlap'a iterek ters artefakta çevirme; hedef yakın/eşit
  lexical ipuçları altında anlam ve morfolojiyle ayrım yapmaktır.

SEKİZ HARD NEGATIVE — SLOT'a atanmış sekiz uyumlu senaryonun her birinden TAM BİR tane:
{_hard_rules(slot['hard_profile'])}

BİÇİMBİLİM
- Bir adayı yanlış yapan neden tek ve açıklanabilir olmalı.
- Bozuk Türkçe veya imkânsız ek dizisi hard negative üretmez; yalnız ucuz dilbilgisi ipucu üretir.
- Yüzey biçimi ile işlevi karıştırma. `şubesinden` (ABL) ve `şubesinde` (LOC) allomorf değil,
  anlam değiştiren farklı hâllerdir.
- {allomorph_rule}
- Yüzey biçimleri yalnız rehberdir: {feature['surface_forms']}.
- Anlam karşıtlığı: {feature['meaning_contrast']}.
- Fenomene özel kural: {special_rule}
- {strict_rule}

GENERALİZASYON
- {bucket_rule}
- {domain_rule}
- Domain: {slot['domain']}; register: {slot['register']}.
- Soyut template: {slot['template']['id']} — {slot['template']['description']}.
- Testteki gerçek örneklerden veya eski JSON cümlelerinden alıntı/kopya yapma.
- SLOT içinde `dataset_memory` varsa bu ham örnek değil, aggregate kapsam hafızasıdır. Sabit kotayı
  değiştirme; `avoid_critical_lemmas` ve `avoid_narrative_tags` değerlerini yeniden kullanma.
- `semantic_profile.narrative_tag` ve `event_type` kısa ASCII snake_case etiketler olmalı
  (`banka_para_transferi`, `belge_teslimi` gibi); cümleyi veya özel kişi adını etikete kopyalama.
- `participant_roles`, `participant_bindings`, polarity, temporal_frame ve scope_target gerçekleşen
  query–gold anlamını tanımlamalı. `participant_bindings`, her rolü query ve positive'taki somut
  katılımcıya bir kez bağlamalı (ör. agent=Selin, theme=rapor, goal=şube). Özellikle CASE, CAUS,
  PASS, REFL ve RECP family'lerinde kim-kime-ne yaptı ve varsa causer değişmeden izlenmeli.
  Bunlar generator niyet etiketi olarak ayrıca doğrulanacaktır.

ALANLAR
- Yalnız şemanın istediği küçük üretim alanlarını yaz. Rol, subtype, morph_relation, qrels,
  edit_script, feature açıklaması ve bütün kimlikler Python tarafından güvenilir plandan eklenir.
- `semantic_frame_id` SLOT değeriyle aynı olsun; `semantic_profile`, `critical_lemma` ve query
  kritik sözcüğünü yaz.
- Positive `candidate_slot=positive_01`; hard slotları `hard_01`…`hard_08`, easy slotları
  `easy_01` ve `easy_02` olsun.
- Her candidate için yalnız tek cümlelik `critical_sentence` ve o cümlede geçen `critical_word` ver.
"""


def build_generation_batch_prompt(slots: list[dict]) -> str:
    """Wrap independent slot prompts in one structured-output request."""
    if not slots:
        raise ValueError("boş generation batch üretilemez")
    tasks = [
        f"\n===== FAMILY {number}/{len(slots)} =====\n{build_generation_prompt(slot)}"
        for number, slot in enumerate(slots, start=1)
    ]
    return f"""\
Aşağıdaki {len(slots)} bağımsız benchmark family görevini tamamla. Her görev kendi
SLOT kimliğini, semantic_frame_id değerini ve kalite kurallarını aynen korumalıdır.
Family'ler arasında metin, olay veya aday kopyalama. Sonuçları görev sırasıyla yalnız
`families` dizisinde döndür; açıklama yazma.
{''.join(tasks)}
"""


def build_repair_prompt(
    slot: dict, previous: dict, problems: list[str], repair_slots: list[str] | None = None
) -> str:
    repair_slots = repair_slots or []
    strategy = (
        "Yalnız şu candidate_slot değerlerini düzelt: " + ", ".join(repair_slots) + ". "
        "Query, context, semantic_profile ve diğer candidate'ları değiştirme."
        if repair_slots else
        "Sorun family geneline yayıldığı için gerekli alanları birlikte düzelt; sağlam alanları koru."
    )
    return f"""\
Önceki üretim otomatik kalite kapısından geçmedi. Önceki JSON üzerinde kontrollü onarım yap.
{strategy}
Şemanın istediği tam JSON'u yeniden döndür; temel SLOT ve bütün üretim kuralları geçerlidir.

SORUNLAR
{json.dumps(problems, ensure_ascii=False, indent=2)}

ÖNCEKİ JSON
{json.dumps(previous, ensure_ascii=False, separators=(",", ":"))}

ORİJİNAL GÖREV
{build_generation_prompt(slot)}
"""


def _blind_candidates(family: dict, permutation: str = "a") -> list[dict]:
    candidates = [{"id": c["id"], "text": c["text"]} for c in family["candidates"]]
    seed_material = f"{family['family_id']}|{permutation}"
    seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
    random.Random(seed).shuffle(candidates)
    return candidates


def build_semantic_judge_prompt(family: dict, permutation: str) -> str:
    visible = {
        "query": family["query"],
        "query_sentence_count": family["query_sentence_count"],
        "passage_sentence_count": family["passage_sentence_count"],
        "candidates": _blind_candidates(family, f"semantic_{permutation}"),
    }
    return f"""\
Aşağıdaki family’yi morfolojik hedefi ve üretim etiketlerini görmeden değerlendir.

{json.dumps(visible, ensure_ascii=False, indent=2)}

KURALLAR
1. `fully_relevant_candidate_ids`: yalnız sorgunun bütün bilgi ihtiyacını aynı doğruluk
   koşullarıyla karşılayan ID'ler. Uyumlu, konuya yakın veya kısmi kanıt olanı ekleme.
2. `unnatural_candidate_ids`: açıkça doğal olmayan Türkçe kullanan ID'ler.
   `internally_inconsistent_candidate_ids`: kendi içinde açıkça çelişen ID'ler.
   `family_naturalness` ölçeği: 5=tamamen doğal, 4=doğal/küçük üslup pürüzü,
   3=belirgin ama tolere edilebilir yapaylık, 2=ciddi yapaylık, 1=bozuk/doğal olmayan family.
   Bu puan notun ve `unnatural_candidate_ids` listenle tutarlı olmalı.
3. Zaman, görünüş, alışkanlık, kişi, sayı, olumsuzluk, kapsam ve katılımcı rolleri doğruluk
   koşulunun parçasıdır. Örneğin tek seferlik "dün yapmadı", "genellikle yapmaz" önermesini;
   "yapmış olabilir" de "yaptı" önermesini karşılamaz.
4. Query ile aday arasında yalnız tek yönlü çıkarım yetmez. Aday bir rolü, varlık türünü, miktarı
   veya kesinliği daraltıyor/genişletiyorsa tam relevant değildir; örneğin `yönetici` ile `müdür`
   bağlama dayalı açık eşdeğerlik olmadan aynı kabul edilmez.
5. Aday kendi içinde çelişiyorsa (ör. rapor hem hazır hem henüz bitmemişse)
   internally_consistent=false ver. Dilbilgisel ama mantıksal çelişkili aday kabul edilmez.
6. Uzunluk, üslup veya ayrıntı tek bir cevabı yapay biçimde ele veriyorsa
   length_or_style_artifact=true ver.
7. Emin değilsen abstain=true ver. Confidence, kararın doğruluğuna ilişkin 0–100 puandır.
Kısa ID listeleri ve kısa not dışında aday-bazlı açıklama üretme.
"""


def build_morphology_judge_prompt(family: dict) -> str:
    visible = {
        "query": family["query"],
        "target_feature": family["target_feature"],
        "target_feature_label": family["target_feature_label"],
        "meaning_contrast": family["feature_delta"],
        "objective": family["objective"],
        "layer": family["layer"],
        "candidates": _blind_candidates(family, "morphology"),
    }
    return f"""\
Aşağıdaki family’yi üretim rolleri ve gold bilgisini görmeden yalnız morfolojik olarak değerlendir.
Semantik relevance ve benzersiz-gold kararı başka bir judge'a aittir.

{json.dumps(visible, ensure_ascii=False, indent=2)}

KURALLAR
1. `target_matching_candidate_ids`: hedef özelliği aynı morfolojik işlev ve kapsamla taşıyan ID'ler.
   Bu liste semantik relevance değildir; içerik yanlış olsa da hedef morfoloji eşleşebilir.
   Notunda hedef zinciri taşıdığını söylediğin her adayı bu listeye ekle; önerme, özne veya olay
   farklılığı nedeniyle morfolojik eşleşmeyi dışlama.
2. `morphologically_invalid_candidate_ids`: doğal/dilbilgisel Türkçe çekim taşımayan ID'ler.
   Negatif veya komşu özellik kullanmak tek başına biçimbilimsel bozukluk değildir.
3. `unclear_candidate_ids`: morfolojik statüsüne güvenle karar veremediğin ID'ler.
4. Kapsam, kişi, sayı, iyelik, zaman/kip ve ek zinciri işlevini yüzey benzerliğinden ayrı kontrol et.
5. CASE/CAUS/PASS/REFL/RECP hedeflerinde hâl veya çatı işaretinin agent, patient/theme,
   goal/recipient, source ve causer rollerini nasıl değiştirdiğini izle. DERIV.IG_CHAIN'de kök POS,
   türetim sınırları ve final POS'u; MORPH.CONTEXT_AMBIG'de yüzey biçiminden çok bağlamsal analizi
   esas al. SUSP.AFFIX ve MWE.MORPH'ta işlev birden fazla tokena yayılabilir.
6. Geçerli allomorfu yanlış saydıysan allomorph_treated_as_wrong=true ver.
7. Emin değilsen abstain=true ver. Confidence, yalnız morfolojik kararın 0–100 güvenidir.
"""


def build_adjudicator_prompt(
    family: dict, semantic_verdicts: list[dict], morphology_verdict: dict, problems: list[str]
) -> str:
    visible = {
        "query": family["query"],
        "target_feature": family["target_feature"],
        "target_feature_label": family["target_feature_label"],
        "candidates": _blind_candidates(family, "adjudicator"),
        "semantic_judge_outputs": semantic_verdicts,
        "morphology_judge_output": morphology_verdict,
        "detected_disagreements": problems,
    }
    return f"""\
Aşağıdaki anlaşmazlığı metinlerden bağımsız olarak yeniden değerlendir. Üretim gold'unu tahmin
etmeye çalışma; `recommendation` yalnız insan hakeme tavsiyedir.

{json.dumps(visible, ensure_ascii=False, indent=2)}
"""
