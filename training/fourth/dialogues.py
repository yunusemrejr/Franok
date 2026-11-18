"""Exact two- and three-turn dialogues from code templates, English and Turkish. Every answer is computed from the state the
user described, never written by a teacher model.

Kinds: object tracking across two turns, one-word label choices with the options given, clarifying questions for vague
requests, and multi-step counting and unit arithmetic with "just the number" follow-ups.
Pools avoid the objects and phrasings of the held-out conversation files; filter.py still checks every user turn against them.
"""

import random
import re

NAMES = {
    "en": ["Sam", "Maya", "Leo", "Nora", "Ben", "Ella", "Omar", "Ruth", "Jack", "Ivy", "Tara", "Felix", "Hugo", "Lena", "Dev", "Rosa", "Theo", "Mira"],
    "tr": ["Ali", "Ayşe", "Mert", "Elif", "Can", "Zeynep", "Emre", "Deniz", "Selin", "Burak", "Cem", "Gül", "Hakan", "Ece", "Kaan", "Pınar", "Tolga", "Yağmur"],
}

# (English, Turkish nominative, Turkish accusative)
THINGS = [
    ("key", "anahtar", "anahtarı"), ("wallet", "cüzdan", "cüzdanı"), ("umbrella", "şemsiye", "şemsiyeyi"),
    ("passport", "pasaport", "pasaportu"), ("notebook", "defter", "defteri"), ("remote control", "kumanda", "kumandayı"),
    ("torch", "el feneri", "el fenerini"), ("scarf", "atkı", "atkıyı"), ("hat", "şapka", "şapkayı"),
    ("mug", "kupa", "kupayı"), ("book", "kitap", "kitabı"), ("watch", "saat", "saati"), ("bottle", "şişe", "şişeyi"),
    ("map", "harita", "haritayı"), ("towel", "havlu", "havluyu"), ("comb", "tarak", "tarağı"),
    ("toy car", "oyuncak araba", "oyuncak arabayı"), ("ruler", "cetvel", "cetveli"), ("eraser", "silgi", "silgiyi"),
    ("candle", "mum", "mumu"), ("jacket", "ceket", "ceketi"), ("screwdriver", "tornavida", "tornavidayı"),
    ("camera", "fotoğraf makinesi", "fotoğraf makinesini"), ("lunchbox", "beslenme kabı", "beslenme kabını"),
    ("hammer", "çekiç", "çekici"), ("sponge", "sünger", "süngeri"), ("brush", "fırça", "fırçayı"),
    ("phone charger", "şarj aleti", "şarj aletini"),
]
# English "at" and "to" phrases, Turkish locative and dative.
PLACES = [
    dict(at="in the cupboard", to="into the cupboard", loc="dolapta", dat="dolaba"),
    dict(at="on the coffee table", to="onto the coffee table", loc="sehpada", dat="sehpaya"),
    dict(at="in the bag", to="into the bag", loc="çantada", dat="çantaya"),
    dict(at="in the kitchen", to="into the kitchen", loc="mutfakta", dat="mutfağa"),
    dict(at="in the garage", to="into the garage", loc="garajda", dat="garaja"),
    dict(at="on the balcony", to="out onto the balcony", loc="balkonda", dat="balkona"),
    dict(at="in the hall", to="into the hall", loc="koridorda", dat="koridora"),
    dict(at="in the wardrobe", to="into the wardrobe", loc="gardırobda", dat="gardıroba"),
    dict(at="in the car", to="into the car", loc="arabada", dat="arabaya"),
    dict(at="in the basket", to="into the basket", loc="sepette", dat="sepete"),
    dict(at="on the windowsill", to="onto the windowsill", loc="pencere kenarında", dat="pencere kenarına"),
    dict(at="in the fridge", to="into the fridge", loc="buzdolabında", dat="buzdolabına"),
    dict(at="on the sofa", to="onto the sofa", loc="koltukta", dat="koltuğa"),
    dict(at="in the box", to="into the box", loc="kutuda", dat="kutuya"),
    dict(at="under the bed", to="under the bed", loc="yatağın altında", dat="yatağın altına"),
    dict(at="in the toolbox", to="into the toolbox", loc="alet çantasında", dat="alet çantasına"),
    dict(at="in the bathroom", to="into the bathroom", loc="banyoda", dat="banyoya"),
    dict(at="in the living room", to="into the living room", loc="oturma odasında", dat="oturma odasına"),
    dict(at="in the backpack", to="into the backpack", loc="sırt çantasında", dat="sırt çantasına"),
    dict(at="on the bookcase", to="onto the bookcase", loc="kitaplıkta", dat="kitaplığa"),
    dict(at="by the door", to="next to the door", loc="kapının yanında", dat="kapının yanına"),
]

# Items that are counted: (English singular, plural, Turkish, verbs that remove some: (English past, Turkish "I ... them" form)).
COUNTED = [
    ("marble", "marbles", "misket", [("lost", "kaybettim"), ("gave away", "verdim")]),
    ("sticker", "stickers", "çıkartma", [("gave away", "verdim"), ("used", "kullandım")]),
    ("button", "buttons", "düğme", [("lost", "kaybettim"), ("used", "kullandım")]),
    ("bead", "beads", "boncuk", [("lost", "kaybettim"), ("used", "kullandım")]),
    ("egg", "eggs", "yumurta", [("used", "kullandım"), ("broke", "kırdım")]),
    ("coin", "coins", "bozuk para", [("spent", "harcadım"), ("lost", "kaybettim")]),
    ("card", "cards", "kart", [("gave away", "verdim"), ("lost", "kaybettim")]),
    ("nail", "nails", "çivi", [("used", "kullandım"), ("lost", "kaybettim")]),
    ("balloon", "balloons", "balon", [("popped", "patlattım"), ("gave away", "verdim")]),
    ("sweet", "sweets", "şeker", [("ate", "yedim"), ("gave away", "verdim")]),
    ("screw", "screws", "vida", [("used", "kullandım"), ("lost", "kaybettim")]),
    ("cup", "cups", "bardak", [("broke", "kırdım"), ("gave away", "verdim")]),
    ("tile", "tiles", "fayans", [("used", "kullandım"), ("broke", "kırdım")]),
    ("postcard", "postcards", "kartpostal", [("sent", "gönderdim"), ("gave away", "verdim")]),
    ("lemon", "lemons", "limon", [("used", "kullandım"), ("gave away", "verdim")]),
    ("sock", "socks", "çorap", [("gave away", "verdim"), ("lost", "kaybettim")]),
]
# (English singular, plural, Turkish nominative, Turkish locative)
CONTAINERS = [
    ("box", "boxes", "kutu", "kutuda"), ("packet", "packets", "paket", "pakette"), ("bag", "bags", "torba", "torbada"),
    ("crate", "crates", "kasa", "kasada"), ("tray", "trays", "tepsi", "tepside"), ("jar", "jars", "kavanoz", "kavanozda"),
    ("basket", "baskets", "sepet", "sepette"), ("tin", "tins", "teneke kutu", "teneke kutuda"),
]
LENGTHS = [("rope", "ip"), ("ribbon", "kurdele"), ("pipe", "boru"), ("cable", "kablo"), ("plank", "tahta"), ("hose", "hortum"), ("curtain", "perde")]
ONLY_NUMBER = {
    "en": ["Only the number, please.", "Number only.", "No words, only the number.", "Answer with the number alone.", "Digits only.", "Reply with a single number."],
    "tr": ["Yalnızca sayı olsun.", "Sadece rakamla yaz.", "Cevabı sadece sayı olarak ver.", "Açıklama yok, sadece sayı.", "Tek bir sayı yaz.", "Sadece sayı."],
}

CATEGORIES = {
    "animal": dict(en=["cat", "horse", "duck", "sheep", "frog", "fox", "goat", "bee", "camel", "owl", "turtle", "donkey", "bear", "lion"],
                   tr=["kedi", "at", "ördek", "koyun", "kurbağa", "tilki", "keçi", "arı", "deve", "baykuş", "kaplumbağa", "eşek", "ayı", "aslan"],
                   en_name="an animal", tr_name="hayvan"),
    "fruit": dict(en=["banana", "cherry", "grape", "lemon", "peach", "melon", "orange", "plum", "fig", "strawberry"],
                  tr=["muz", "kiraz", "üzüm", "limon", "şeftali", "kavun", "portakal", "erik", "nar", "çilek"],
                  en_name="a fruit", tr_name="meyve"),
    "tool": dict(en=["saw", "spade", "drill", "pliers", "wrench", "axe", "chisel", "rake"],
                 tr=["testere", "kürek", "matkap", "pense", "İngiliz anahtarı", "balta", "keski", "tırmık"],
                 en_name="a tool", tr_name="alet"),
    "vehicle": dict(en=["train", "bicycle", "boat", "truck", "taxi", "tractor", "ferry", "motorbike"],
                    tr=["tren", "bisiklet", "tekne", "kamyon", "taksi", "traktör", "vapur", "motosiklet"],
                    en_name="a vehicle", tr_name="taşıt"),
    "furniture": dict(en=["chair", "wardrobe", "bed", "bookcase", "stool", "armchair", "bench"],
                      tr=["sandalye", "gardırop", "yatak", "kitaplık", "tabure", "koltuk", "bank"],
                      en_name="a piece of furniture", tr_name="mobilya"),
    "clothing": dict(en=["shirt", "sock", "coat", "skirt", "glove", "trousers", "sweater"],
                     tr=["gömlek", "çorap", "mont", "etek", "eldiven", "pantolon", "kazak"],
                     en_name="a piece of clothing", tr_name="giysi"),
    "colour": dict(en=["red", "yellow", "purple", "pink", "black", "brown", "grey", "white"],
                   tr=["kırmızı", "sarı", "mor", "pembe", "siyah", "kahverengi", "gri", "beyaz"],
                   en_name="a colour", tr_name="renk"),
    "instrument": dict(en=["guitar", "drum", "flute", "violin", "piano", "trumpet"],
                       tr=["gitar", "davul", "flüt", "keman", "piyano", "trompet"],
                       en_name="a musical instrument", tr_name="çalgı"),
    "vegetable": dict(en=["carrot", "onion", "potato", "pepper", "cucumber", "cabbage", "leek"],
                      tr=["havuç", "soğan", "patates", "biber", "salatalık", "lahana", "pırasa"],
                      en_name="a vegetable", tr_name="sebze"),
    "body part": dict(en=["elbow", "knee", "shoulder", "ankle", "wrist", "thumb", "heel"],
                      tr=["dirsek", "diz", "omuz", "bilek", "başparmak", "topuk", "kaş"],
                      en_name="a part of the body", tr_name="vücut organı"),
}
WORDS = dict(
    en=["garden", "window", "pencil case", "river", "mountain", "orange", "kitchen", "bridge", "island", "blanket", "dolphin", "market", "candle", "forest", "button", "ladder", "pillow", "rocket", "village", "sandwich"],
    tr=["bahçe", "pencere", "nehir", "dağ", "mutfak", "köprü", "ada", "battaniye", "yunus", "pazar", "mum", "orman", "düğme", "merdiven", "yastık", "roket", "köy", "sandviç", "okul", "çiçek"],
)
EN_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# Turkish ablative forms, so "X comes after Y" needs no suffix rules.
TR_DAYS = [("Pazartesi", "Pazartesi'den"), ("Salı", "Salı'dan"), ("Çarşamba", "Çarşamba'dan"), ("Perşembe", "Perşembe'den"),
           ("Cuma", "Cuma'dan"), ("Cumartesi", "Cumartesi'den"), ("Pazar", "Pazar'dan")]
EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
TR_MONTHS = [("Ocak", "Ocak'tan"), ("Şubat", "Şubat'tan"), ("Mart", "Mart'tan"), ("Nisan", "Nisan'dan"), ("Mayıs", "Mayıs'tan"),
             ("Haziran", "Haziran'dan"), ("Temmuz", "Temmuz'dan"), ("Ağustos", "Ağustos'tan"), ("Eylül", "Eylül'den"),
             ("Ekim", "Ekim'den"), ("Kasım", "Kasım'dan"), ("Aralık", "Aralık'tan")]
# (English sentence with {n}, Turkish sentence with {n}, true number, smallest and largest wrong number)
FACTS = [
    ("A spider has {n} legs.", "Bir örümceğin {n} bacağı vardır.", 8, 4, 12),
    ("A triangle has {n} sides.", "Üçgenin {n} kenarı vardır.", 3, 2, 8),
    ("A year has {n} months.", "Bir yılda {n} ay vardır.", 12, 6, 20),
    ("A day has {n} hours.", "Bir günde {n} saat vardır.", 24, 12, 48),
    ("A square has {n} equal sides.", "Karenin {n} eşit kenarı vardır.", 4, 2, 8),
    ("A bicycle has {n} wheels.", "Bisikletin {n} tekerleği vardır.", 2, 3, 6),
    ("One hand has {n} fingers.", "Bir elde {n} parmak vardır.", 5, 2, 10),
    ("A minute has {n} seconds.", "Bir dakikada {n} saniye vardır.", 60, 30, 100),
    ("A dozen means {n}.", "Bir düzine {n} demektir.", 12, 6, 24),
    ("A metre has {n} centimetres.", "Bir metrede {n} santimetre vardır.", 100, 10, 1000),
    ("A kilogram is {n} grams.", "Bir kilogram {n} gramdır.", 1000, 100, 10000),
    ("A cat has {n} legs.", "Bir kedinin {n} bacağı vardır.", 4, 2, 8),
    ("A cube has {n} faces.", "Bir küpün {n} yüzü vardır.", 6, 4, 12),
    ("An hour has {n} minutes.", "Bir saatte {n} dakika vardır.", 60, 30, 100),
    ("A quarter of an hour is {n} minutes.", "Çeyrek saat {n} dakikadır.", 15, 10, 30),
    ("A week has {n} days.", "Bir haftada {n} gün vardır.", 7, 5, 10),
    ("A litre has {n} millilitres.", "Bir litrede {n} mililitre vardır.", 1000, 100, 10000),
    ("A leap year has {n} days.", "Artık yılda {n} gün vardır.", 366, 300, 400),
]
EN_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
EN_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
TR_ONES = "sıfır bir iki üç dört beş altı yedi sekiz dokuz".split()
TR_TENS = "_ on yirmi otuz kırk elli altmış yetmiş seksen doksan".split()


def number_word(n, lang):
    """Words for 0 to 999 and the round numbers the facts use."""
    if lang == "en":
        if n < 20:
            return EN_ONES[n]
        if n < 100:
            return EN_TENS[n // 10] + ("-" + EN_ONES[n % 10] if n % 10 else "")
        head = EN_ONES[n // 100] + " hundred"
        return head + (" " + number_word(n % 100, lang) if n % 100 else "")
    if n < 10:
        return TR_ONES[n]
    if n < 100:
        return TR_TENS[n // 10] + (" " + TR_ONES[n % 10] if n % 10 else "")
    head = "yüz" if n // 100 == 1 else TR_ONES[n // 100] + " yüz"
    return head + (" " + number_word(n % 100, lang) if n % 100 else "")


def mi(word):
    """Turkish question particle that follows `word`, with vowel harmony."""
    vowels = [c for c in word.lower() if c in "aeıioöuü"]
    last = vowels[-1] if vowels else "e"
    return word + " " + {"a": "mı", "ı": "mı", "e": "mi", "i": "mi", "o": "mu", "u": "mu", "ö": "mü", "ü": "mü"}[last]


def cap(text):
    return text[:1].upper() + text[1:]


def pick(rng, options):
    return options[rng.randrange(len(options))]


def end_with(lang, rng):
    return pick(rng, ONLY_NUMBER[lang])


# ---------------------------------------------------------------- object tracking

def tracking(rng, lang):
    (a_en, a_tn, a_ta), (b_en, b_tn, b_ta) = rng.sample(THINGS, 2)
    p1, p2, p3 = rng.sample(PLACES, 3)
    names = {"A": (a_en, a_tn, a_ta), "B": (b_en, b_tn, b_ta)}
    where = {"A": p1, "B": p2}
    n = pick(rng, NAMES[lang])
    asked = pick(rng, ["A", "B"])
    reverse = rng.random() < 0.18
    form = rng.randrange(4)
    A, B = names["A"], names["B"]
    if lang == "en":
        if reverse:
            q1 = pick(rng, [f"The {A[0]} is {p1['at']} and the {B[0]} is {p2['at']}. Which one is {where[asked]['at']}?",
                            f"Right now the {A[0]} is {p1['at']}, while the {B[0]} is {p2['at']}. What is {where[asked]['at']}?"])
            a1 = f"The {names[asked][0]}."
        else:
            q1 = [f"The {A[0]} is {p1['at']} and the {B[0]} is {p2['at']}. Where is the {names[asked][0]}?",
                  f"I keep the {A[0]} {p1['at']} and the {B[0]} {p2['at']}. Where should I look for the {names[asked][0]}?",
                  f"{n} left the {A[0]} {p1['at']} and the {B[0]} {p2['at']}. Where did {n} leave the {names[asked][0]}?",
                  f"Quick check: {A[0]} {p1['at']}, {B[0]} {p2['at']}. Where is the {names[asked][0]}?"][form]
            a1 = pick(rng, [f"The {names[asked][0]} is {where[asked]['at']}.", f"{cap(where[asked]['at'])}."])
    else:
        if reverse:
            q1 = pick(rng, [f"{cap(A[1])} {p1['loc']}, {B[1]} {p2['loc']}. Hangisi {where[asked]['loc']}?",
                            f"Şu an {A[1]} {p1['loc']}, {B[1]} {p2['loc']}. {cap(where[asked]['loc'])} hangisi var?"])
            a1 = f"{cap(names[asked][1])}."
        else:
            q1 = [f"{cap(A[1])} {p1['loc']}, {B[1]} {p2['loc']}. {cap(names[asked][1])} nerede?",
                  f"{cap(A[2])} {p1['loc']}, {B[2]} {p2['loc']} tutuyorum. {cap(names[asked][1])} nerede?",
                  f"{n} {A[2]} {p1['loc']}, {B[2]} {p2['loc']} bıraktı. {cap(names[asked][1])} nerede?",
                  f"Kısa bir kontrol: {A[1]} {p1['loc']}, {B[1]} {p2['loc']}. {cap(names[asked][1])} nerede?"][form]
            a1 = pick(rng, [f"{cap(names[asked][1])} {where[asked]['loc']}.", f"{cap(where[asked]['loc'])}."])
    # second turn: a move or a swap, then a question about one thing or both
    previous = dict(where)
    op = pick(rng, ["move", "move", "swap", "move_other"])
    mover = pick(rng, ["A", "B"])
    other = "B" if mover == "A" else "A"
    if op == "swap":
        where = {"A": previous["B"], "B": previous["A"]}
    else:
        where[mover] = p3
    target = pick(rng, ["A", "B", "A", "B", "both"])
    if op == "move_other":
        target = other
    status = lambda k: "still" if where[k] == previous[k] else "now"
    if lang == "en":
        if op == "swap":
            move = pick(rng, ["Then I swapped the two.", "We switched their places.", "After that, I exchanged the two of them."])
        else:
            m = names[mover][0]
            move = pick(rng, [f"I moved the {m} {p3['to']}.", f"Then the {m} went {p3['to']}.", f"After that, I carried the {m} {p3['to']}.",
                              f"Later {n} took the {m} {p3['to']}."])
        if target == "both":
            q2 = f"{move} Where are the {A[0]} and the {B[0]} now?"
            a2 = f"The {A[0]} is {where['A']['at']} and the {B[0]} is {where['B']['at']}."
        else:
            t = names[target][0]
            q2 = pick(rng, [f"{move} Where is the {t} now?", f"{move} And the {t}, where is it?", f"{move} Where can I find the {t}?"])
            a2 = pick(rng, [f"The {t} is {status(target)} {where[target]['at']}.", f"{cap(where[target]['at'])}."])
    else:
        if op == "swap":
            move = pick(rng, ["Sonra ikisinin yerini değiştirdim.", "İkisinin yerlerini birbiriyle değiştirdik.", "Ardından yerlerini çaprazladım."])
        else:
            m = names[mover]
            move = pick(rng, [f"{cap(m[2])} {p3['dat']} taşıdım.", f"Sonra {m[2]} {p3['dat']} götürdüm.", f"Ardından {m[2]} {p3['dat']} koydum.",
                              f"{n} {m[2]} {p3['dat']} taşıdı."])
        if target == "both":
            q2 = f"{move} {cap(A[1])} ile {B[1]} şimdi nerede?"
            a2 = f"{cap(A[1])} {where['A']['loc']}, {B[1]} {where['B']['loc']}."
        else:
            t = names[target][1]
            q2 = pick(rng, [f"{move} {cap(t)} şimdi nerede?", f"{move} Peki {t} nerede?", f"{move} {cap(t)} nerede duruyor?"])
            word = "hâlâ" if status(target) == "still" else "şimdi"
            a2 = pick(rng, [f"{cap(t)} {word} {where[target]['loc']}.", f"{cap(where[target]['loc'])}."])
    return [(q1, a1), (q2, a2)]


# ---------------------------------------------------------------- one-word labels

def sentence_pool(rng, lang):
    """A fact (statement), a question and a request about the same freshly drawn content."""
    thing = pick(rng, THINGS)
    place = pick(rng, PLACES)
    name = pick(rng, NAMES[lang])
    adj = pick(rng, [("heavy", "ağır"), ("old", "eski"), ("empty", "boş"), ("new", "yeni"), ("clean", "temiz"), ("broken", "bozuk"),
                     ("warm", "sıcak"), ("small", "küçük"), ("light", "hafif"), ("dusty", "tozlu"), ("loud", "gürültülü")])
    hour = rng.randint(6, 21)
    n = rng.randint(2, 19)
    item = pick(rng, COUNTED)
    kind = rng.randrange(5)
    if lang == "en":
        t = thing[0]
        sets = [
            (f"The {t} is {adj[0]}.", f"Is the {t} {adj[0]}?", f"Please check whether the {t} is {adj[0]}."),
            (f"The {t} is {place['at']}.", f"Is the {t} {place['at']}?", f"Please put the {t} {place['to']}."),
            (f"{name} reads every day.", f"Does {name} read every day?", f"{name}, please read it out loud."),
            (f"There are {n} {item[1]} {place['at']}.", f"Are there {n} {item[1]} {place['at']}?", f"Please count how many {item[1]} there are {place['at']}."),
            (f"The shop opens at {hour}:00.", f"Does the shop open at {hour}:00?", f"Please open the shop at {hour}:00."),
        ]
    else:
        t = thing[1]
        sets = [
            (f"{cap(t)} {adj[1]}.", f"{cap(mi(t + ' ' + adj[1]))}?", f"Lütfen {thing[2]} kontrol et."),
            (f"{cap(t)} {place['loc']}.", f"{cap(t)} {mi(place['loc'])}?", f"{cap(thing[2])} {place['dat']} koyar mısın?"),
            (f"{name} her gün okuyor.", f"{name} her gün okuyor mu?", f"{name}, lütfen sesli oku."),
            (f"{cap(place['loc'])} {n} {item[2]} var.", f"{cap(place['loc'])} {n} {item[2]} var mı?", f"Lütfen {place['loc']} kaç {item[2]} olduğunu say."),
            (f"Mağaza saat {tr_hour(hour)} açılıyor.", f"Mağaza saat {mi(tr_hour(hour))} açılıyor?", f"Lütfen mağazayı saat {tr_hour(hour)} aç."),
        ]
    s, q, r = sets[kind]
    return dict(fact=s, question=q, request=r)


def tr_hour(h):
    """Turkish 'at <hour>' (saat + locative), for 6 to 21."""
    forms = {6: "altıda", 7: "yedide", 8: "sekizde", 9: "dokuzda", 10: "onda", 11: "on birde", 12: "on ikide", 13: "on üçte",
             14: "on dörtte", 15: "on beşte", 16: "on altıda", 17: "on yedide", 18: "on sekizde", 19: "on dokuzda", 20: "yirmide", 21: "yirmi birde"}
    return forms[h]


def follow(lang, rng, options):
    return pick(rng, {
        "en": ["And this one? “{s}” Same choices.", "What about “{s}”? Same options.", "Next: “{s}” Same question.", "Now “{s}”"],
        "tr": ["Bu da? “{s}” Aynı seçenekler.", "Peki “{s}” cümlesi? Aynı seçenekler.", "Sıradaki: “{s}” Aynı soru.", "Şimdi “{s}”"],
    }[lang])


def label_kind(rng, lang):
    kind = pick(rng, ["qs", "qs", "rf", "rf", "three", "tf", "tf", "tf", "tf", "category", "category", "odd", "number", "ask"])
    return globals()["label_" + kind](rng, lang)


def label_qs(rng, lang):
    a, b = sentence_pool(rng, lang), sentence_pool(rng, lang)
    order = rng.sample(["question", "fact"], 2)
    pairs = [(a[order[0]], order[0]), (b[order[1]], order[1])]
    names = dict(en=dict(question="question", fact="statement"), tr=dict(question="soru", fact="bilgi"))[lang]
    if lang == "en":
        ask = pick(rng, ["Label the sentence “{s}” with one word: question or statement.",
                         "Is the following a question or a statement? “{s}” Give only the label.",
                         "Sort this sentence as question or statement, one word only: “{s}”",
                         "Tell me whether “{s}” asks something or states something. Use the word question or statement."])
    else:
        ask = pick(rng, ["“{s}” cümlesini tek kelimeyle etiketle: soru ya da bilgi.",
                         "Aşağıdaki cümle soru mu, bilgi mi? “{s}” Sadece etiketi yaz.",
                         "Şu cümleyi soru ya da bilgi olarak ayır, tek kelime: “{s}”",
                         "“{s}” bir şey soruyor mu yoksa bir şey mi söylüyor? Soru ya da bilgi yaz."])
    turns = [(ask.format(s=pairs[0][0]), cap(names[pairs[0][1]]) + ".")]
    turns.append((follow(lang, rng, None).format(s=pairs[1][0]), cap(names[pairs[1][1]]) + "."))
    return turns


def label_rf(rng, lang):
    a, b = sentence_pool(rng, lang), sentence_pool(rng, lang)
    order = rng.sample(["request", "fact"], 2)
    pairs = [(a[order[0]], order[0]), (b[order[1]], order[1])]
    names = dict(en=dict(request="request", fact="fact"), tr=dict(request=pick(rng, ["rica", "istek"]), fact="bilgi"))[lang]
    if lang == "en":
        ask = pick(rng, ["Does “{s}” ask someone to do something, or just report something? Use the word request or fact.",
                         "Label “{s}” with one word: request or fact.",
                         "Sort this sentence into request or fact, label only: “{s}”"])
    else:
        ask = pick(rng, [f"“{{s}}” birinden bir şey mi istiyor, yoksa bir şey mi bildiriyor? {cap(names['request'])} ya da bilgi yaz.",
                         f"“{{s}}” cümlesini tek kelimeyle etiketle: {names['request']} ya da bilgi.",
                         f"Bu cümleyi {names['request']} ya da bilgi olarak ayır, sadece etiket: “{{s}}”"])
    turns = [(ask.format(s=pairs[0][0]), cap(names[pairs[0][1]]) + ".")]
    turns.append((follow(lang, rng, None).format(s=pairs[1][0]), cap(names[pairs[1][1]]) + "."))
    return turns


def label_three(rng, lang):
    kinds = ["question", "request", "fact"]
    a, b = sentence_pool(rng, lang), sentence_pool(rng, lang)
    k1, k2 = rng.sample(kinds, 2)
    names = dict(en=dict(question="question", request="request", fact="fact"), tr=dict(question="soru", request="rica", fact="bilgi"))[lang]
    if lang == "en":
        ask = pick(rng, ["Is “{s}” a question, a request or a fact? Reply with one word.", "Choose one label for “{s}”: question, request or fact."])
    else:
        ask = pick(rng, ["“{s}” bir soru mu, rica mı, bilgi mi? Tek kelimeyle yaz.", "“{s}” cümlesine tek etiket seç: soru, rica ya da bilgi."])
    return [(ask.format(s=a[k1]), cap(names[k1]) + "."), (follow(lang, rng, None).format(s=b[k2]), cap(names[k2]) + ".")]


def tf_item(rng, lang):
    """One statement with a computed truth value: returns (sentence, is_true)."""
    kind = rng.randrange(7)
    truth = rng.random() < 0.5
    if kind == 0:
        a, b = rng.randint(3, 15), rng.randint(3, 15)
        c = a * b if truth else a * b + pick(rng, [-3, -2, -1, 1, 2, 3, 10, -10])
        return (f"{a} times {b} is {c}." if lang == "en" else f"{a} çarpı {b}, {c} eder."), truth
    if kind == 1:
        a, b = rng.randint(20, 99), rng.randint(20, 99)
        c = a + b if truth else a + b + pick(rng, [-10, -1, 1, 10, 2, -2])
        return (f"{a} plus {b} is {c}." if lang == "en" else f"{a} artı {b}, {c} eder."), truth
    if kind == 2:
        a, b = rng.sample(range(10, 500), 2)
        big, small = max(a, b), min(a, b)
        x, y = (big, small) if truth else (small, big)
        return (f"{x} is bigger than {y}." if lang == "en" else f"{x} sayısı {y} sayısından büyüktür."), truth
    if kind == 3:
        n = rng.randint(11, 98)
        even = n % 2 == 0
        said_even = rng.random() < 0.5
        return ((f"{n} is an {'even' if said_even else 'odd'} number." if lang == "en" else f"{n} {'çift' if said_even else 'tek'} bir sayıdır."), even == said_even)
    if kind == 4:
        i = rng.randrange(7)
        j = (i + 1) % 7 if truth else (i + rng.choice([2, 3, 4, 5])) % 7
        if lang == "en":
            return f"{EN_DAYS[j]} comes right after {EN_DAYS[i]}.", truth
        return f"{TR_DAYS[i][1]} hemen sonra {TR_DAYS[j][0]} gelir.", truth
    if kind == 5:
        en, tr, true_n, lo, hi = pick(rng, FACTS)
        n = true_n
        while not truth and n == true_n:
            n = rng.randint(lo, hi)
        text = (en if lang == "en" else tr)
        digits = rng.random() < 0.5 or n >= 100
        return text.format(n=n if digits else number_word(n, lang)), n == true_n
    word = pick(rng, WORDS[lang])
    letters = len(word.replace(" ", ""))
    n = letters if truth else letters + pick(rng, [-2, -1, 1, 2])
    return (f"The word “{word}” has {n} letters." if lang == "en" else f"“{word}” kelimesinde {n} harf var."), n == letters


def label_tf(rng, lang):
    names = dict(en=("True", "False"), tr=("Doğru", "Yanlış"))[lang]
    turns = []
    for i in range(2):
        s, ok = tf_item(rng, lang)
        if i == 0:
            q = pick(rng, {
                "en": ["True or false: {s} Answer with one word.", "Is this statement correct? “{s}” Say true or false.", "“{s}” Is that true or false? One word only."],
                "tr": ["Doğru mu yanlış mı: {s} Tek kelimeyle cevap ver.", "Bu cümle doğru mu? “{s}” Doğru ya da yanlış de.", "“{s}” Doğru mu, yanlış mı? Sadece tek kelime."],
            }[lang]).format(s=s)
        else:
            q = pick(rng, {
                "en": ["And this one: {s} Same two words.", "Next one: “{s}” True or false?", "What about: {s}"],
                "tr": ["Bu da: {s} Aynı iki kelime.", "Sıradaki: “{s}” Doğru mu yanlış mı?", "Peki bu: {s}"],
            }[lang]).format(s=s)
        turns.append((q, names[0 if ok else 1] + "."))
    return turns


def label_category(rng, lang):
    turns = []
    cats = rng.sample(sorted(CATEGORIES), 2)
    for i, cat in enumerate(cats):
        c = CATEGORIES[cat]
        others = [x for k, v in CATEGORIES.items() if k != cat for x in v[lang]]
        member, other = pick(rng, c[lang]), pick(rng, others)
        pair = [member, other]
        rng.shuffle(pair)
        if lang == "en":
            text = pick(rng, [f"Which of these is {c['en_name']}: {pair[0]} or {pair[1]}? One word.",
                              f"Pick {c['en_name']} from these two: {pair[0]}, {pair[1]}. Reply with that word only.",
                              f"{cap(pair[0])} or {pair[1]}: which one is {c['en_name']}? Just the word."])
            if i:
                text = pick(rng, [f"Same question for {pair[0]} and {pair[1]}, but now {c['en_name']}.", f"Now {c['en_name']}: {pair[0]} or {pair[1]}?"])
        else:
            text = pick(rng, [f"Bunlardan hangisi {c['tr_name']}: {mi(pair[0])}, {mi(pair[1])}? Tek kelime yaz.",
                              f"{cap(pair[0])} ile {pair[1]} arasından {c['tr_name']} olanı seç. Sadece o kelimeyi yaz.",
                              f"Hangisi {c['tr_name']}: {mi(pair[0])}, {mi(pair[1])}? Yalnızca kelimeyi yaz."])
            if i:
                text = pick(rng, [f"Aynı soru, ama şimdi {c['tr_name']}: {mi(pair[0])}, {mi(pair[1])}?", f"Şimdi {c['tr_name']} olan hangisi: {mi(pair[0])}, {mi(pair[1])}?"])
        turns.append((text, cap(member) + "."))
    return turns


def label_odd(rng, lang):
    turns = []
    for i in range(2):
        cat, wrong = rng.sample(sorted(CATEGORIES), 2)
        group = rng.sample(CATEGORIES[cat][lang], 2)
        odd = pick(rng, CATEGORIES[wrong][lang])
        words = group + [odd]
        rng.shuffle(words)
        if lang == "en":
            text = f"Which word does not belong with the others: {words[0]}, {words[1]}, {words[2]}? One word." if not i else f"And here: {', '.join(words)}. Which one is the odd one?"
        else:
            text = f"Hangi kelime diğerlerine uymuyor: {words[0]}, {words[1]}, {words[2]}? Tek kelime." if not i else f"Peki bunlarda: {', '.join(words)}. Hangisi farklı?"
        turns.append((text, cap(odd) + "."))
    return turns


def label_number(rng, lang):
    """Comparisons with one-word or yes/no answers, computed."""
    turns = []
    for i in range(2):
        kind = rng.randrange(3)
        if kind == 0:
            a, b = rng.sample(range(10, 999), 2)
            ans = max(a, b)
            q = f"Which is the larger number, {a} or {b}? {end_with(lang, rng)}" if lang == "en" else f"{a} ile {b} arasında hangisi daha büyük? {end_with(lang, rng)}"
            turns.append((q, str(ans)))
        elif kind == 1:
            n = rng.randint(11, 98)
            q = f"Is {n} odd or even? One word." if lang == "en" else f"{n} tek mi çift mi? Tek kelime yaz."
            turns.append((q, ("Even." if n % 2 == 0 else "Odd.") if lang == "en" else ("Çift." if n % 2 == 0 else "Tek.")))
        else:
            a, b = rng.randint(5, 60), rng.randint(5, 60)
            q = f"Is {a} greater than {b}? Yes or no." if lang == "en" else f"{a} sayısı {b} sayısından büyük mü? Evet ya da hayır."
            turns.append((q, ("Yes." if a > b else "No.") if lang == "en" else ("Evet." if a > b else "Hayır.")))
    return turns


def label_ask(rng, lang):
    """Whether a request can be answered as it stands or needs a question first. Options are given."""
    vague = {
        "en": ["Add them up.", "Convert that for me.", "How long does it take?", "Which one is bigger?", "Split it equally.", "How much will it cost?",
               "Send it to them tonight.", "Make it shorter.", "Cancel the other one.", "Put it back."],
        "tr": ["Bunları topla.", "Bunu benim için çevir.", "Ne kadar sürer?", "Hangisi daha büyük?", "Bunu eşit paylaştır.", "Kaça mal olur?",
               "Bunu bu akşam onlara gönder.", "Bunu kısalt.", "Ötekini iptal et.", "Onu geri koy."],
    }[lang]
    a, b, n = rng.randint(12, 60), rng.randint(12, 60), rng.randint(2, 9)
    clear = {
        "en": [f"What is {a} plus {b}?", f"How many minutes are in {n} hours?", f"Which is bigger, {a} or {b + 100}?", f"What is {n} times {a}?",
               f"How many days are in {n} weeks?", f"What is half of {2 * a}?", f"How many centimetres are in {n} metres?"],
        "tr": [f"{a} artı {b} kaç eder?", f"{n} saatte kaç dakika vardır?", f"{a} ile {b + 100} arasında hangisi büyük?", f"{n} çarpı {a} kaç eder?",
               f"{n} haftada kaç gün vardır?", f"{2 * a} sayısının yarısı kaç eder?", f"{n} metre kaç santimetre eder?"],
    }[lang]
    turns = []
    picks = [("ask", pick(rng, vague)), ("answer", pick(rng, clear))]
    rng.shuffle(picks)
    labels = dict(en=dict(ask="Ask", answer="Answer"), tr=dict(ask="Sor", answer="Cevapla"))[lang]
    for i, (label, text) in enumerate(picks):
        if lang == "en":
            q = (f"A user writes: “{text}” Should the assistant ask a question first or answer right away? Reply with one word: ask or answer."
                 if not i else f"And for “{text}”? Same choice.")
        else:
            q = (f"Bir kullanıcı şunu yazıyor: “{text}” Asistan önce soru mu sormalı, yoksa hemen cevap mı vermeli? Tek kelimeyle yaz: sor ya da cevapla."
                 if not i else f"Peki “{text}” için? Aynı seçim.")
        turns.append((q, labels[label] + "."))
    return turns


# ---------------------------------------------------------------- clarifying questions

def clarify(rng, lang):
    scenario = rng.randrange(6)
    pre = pick(rng, {
        "en": ["", "", "Hey, ", "Okay, ", "Quick thing: ", "Sorry to bother you, but ", "Right, ", "One more thing: "],
        "tr": ["", "", "Selam, ", "Tamam, ", "Küçük bir şey: ", "Rahatsız ediyorum ama ", "Peki, ", "Bir şey daha: "],
    }[lang])
    low = lambda s: s if not pre or s.startswith("I ") else s[0].lower() + s[1:]
    if scenario == 0:
        a, b, c = rng.randint(3, 40), rng.randint(3, 40), rng.randint(3, 40)
        if lang == "en":
            return [(pre + low(pick(rng, ["Add them up.", "Can you add them together?", "I need the total."])), pick(rng, ["Which numbers should I add?", "Which numbers do you mean?"])),
                    (pick(rng, [f"{a}, {b} and {c}.", f"It is {a}, {b} and {c}."]), f"{a} plus {b} plus {c} is {a + b + c}.")]
        return [(pre + low(pick(rng, ["Bunları topla.", "Şunları toplar mısın?", "Toplamı lazım."])), pick(rng, ["Hangi sayıları toplayayım?", "Hangi sayıları kastediyorsun?"])),
                (pick(rng, [f"{a}, {b} ve {c}.", f"{a}, {b} ile {c}."]), f"{a} artı {b} artı {c}, {a + b + c} eder.")]
    if scenario == 1:
        units = [("kilometres", "metres", 1000, "kilometreyi", "metreye"), ("metres", "centimetres", 100, "metreyi", "santimetreye"),
                 ("kilograms", "grams", 1000, "kilogramı", "grama"), ("hours", "minutes", 60, "saati", "dakikaya"),
                 ("litres", "millilitres", 1000, "litreyi", "mililitreye"), ("days", "hours", 24, "günü", "saate")]
        s_en, d_en, f, s_tr, d_tr = pick(rng, units)
        n = rng.randint(2, 15)
        if lang == "en":
            return [(pre + low(pick(rng, ["Convert it for me.", "Can you change the unit?", "I need it in a different unit."])), "What should I convert, and into which unit?"),
                    (f"{n} {s_en} into {d_en}.", f"{n} {s_en} is {n * f} {d_en}.")]
        return _convert_tr(pre, low, rng, n, s_tr, d_tr, f)
    if scenario == 2:
        a, b = rng.sample(range(10, 999), 2)
        if lang == "en":
            return [(pre + low(pick(rng, ["Which one is bigger?", "Tell me which is larger.", "Which is the larger one?"])), "Which two numbers do you want me to compare?"),
                    (pick(rng, [f"{a} and {b}.", f"{a} versus {b}."]), f"{max(a, b)} is bigger.")]
        return [(pre + low(pick(rng, ["Hangisi daha büyük?", "Hangisinin daha büyük olduğunu söyler misin?", "Büyük olan hangisi?"])), "Hangi iki sayıyı karşılaştırayım?"),
                (pick(rng, [f"{a} ve {b}.", f"{a} ile {b}."]), f"{max(a, b)} daha büyük.")]
    if scenario == 3:
        v = pick(rng, [20, 30, 40, 50, 60, 80, 90])
        h = rng.randint(2, 6)
        d = v * h
        if lang == "en":
            return [(pre + low(pick(rng, ["How long will it take?", "How much time do I need?", "When will I get there?"])), "I need two numbers: the distance and your speed."),
                    (pick(rng, [f"{d} kilometres at {v} kilometres per hour.", f"The distance is {d} km and I go {v} km per hour."]), f"{d} divided by {v} is {h}, so it takes {h} hours.")]
        return [(pre + low(pick(rng, ["Ne kadar sürer?", "Ne kadar zamana ihtiyacım var?", "Ne zaman varırım?"])), "İki sayıya ihtiyacım var: mesafe ve hızın."),
                (pick(rng, [f"{d} kilometre, saatte {v} kilometre.", f"Mesafe {d} km, ben saatte {v} km gidiyorum."]), f"{d} bölü {v}, {h} eder. Yani {h} saat sürer.")]
    if scenario == 4:
        n, p = rng.randint(2, 12), rng.choice([4, 5, 6, 8, 9, 12, 15, 20, 25])
        if lang == "en":
            return [(pre + low(pick(rng, ["How much will it cost?", "What is the total price?", "Tell me the price."])), "How many do you want, and what does one cost?"),
                    (pick(rng, [f"{n} of them, {p} lira each.", f"I want {n}, and one costs {p} lira."]), f"{n} times {p} is {n * p} lira.")]
        return [(pre + low(pick(rng, ["Kaça gelir?", "Toplam fiyat ne olur?", "Fiyatı söyler misin?"])), "Kaç tane istiyorsun, tanesi kaç lira?"),
                (pick(rng, [f"{n} tane, tanesi {p} lira.", f"{n} tane istiyorum, biri {p} lira."]), f"{n} çarpı {p}, {n * p} lira eder.")]
    people = rng.choice([2, 3, 4, 5, 6, 8])
    each = rng.randint(2, 9)
    total = people * each
    if lang == "en":
        return [(pre + low(pick(rng, ["Split it equally.", "Share it out fairly.", "Divide it for us."])), "How many things are there, and how many people?"),
                (pick(rng, [f"{total} sweets for {people} people.", f"There are {total} sweets and {people} of us."]), f"Each person gets {each} sweets.")]
    return [(pre + low(pick(rng, ["Bunu eşit paylaştır.", "Adil bir şekilde böl.", "Bizim için paylaştırır mısın?"])), "Kaç tane var, kaç kişisiniz?"),
            (pick(rng, [f"{people} kişiye {total} şeker.", f"{total} şeker var, {people} kişiyiz."]), f"Kişi başına {each} şeker düşer.")]


def _convert_tr(pre, low, rng, n, s_tr, d_tr, f):
    words = {"kilometreyi": "kilometre", "metreyi": "metre", "kilogramı": "kilogram", "saati": "saat", "litreyi": "litre", "günü": "gün"}
    dest = {"metreye": "metre", "santimetreye": "santimetre", "grama": "gram", "dakikaya": "dakika", "mililitreye": "mililitre", "saate": "saat"}
    return [(pre + low(pick(rng, ["Bunu benim için çevir.", "Birimi değiştirir misin?", "Başka bir birimle lazım."])), "Neyi hangi birime çevireyim?"),
            (f"{n} {s_tr} {d_tr}.", f"{n} {words[s_tr]}, {n * f} {dest[d_tr]} eder.")]


# ---------------------------------------------------------------- counting and units

def counting(rng, lang):
    kind = rng.randrange(5)
    return [stock, lengths, letters, running, duration][kind](rng, lang)


def stock(rng, lang):
    en_s, en_p, tr, verbs = pick(rng, COUNTED)
    c_s, c_p, c_tr, c_loc = pick(rng, CONTAINERS)
    per, boxes = rng.randint(4, 24), rng.randint(2, 9)
    total = per * boxes
    add = rng.random() < 0.3
    delta = rng.randint(3, max(4, total // 3))
    final = total + delta if add else total - delta
    verb = pick(rng, verbs)
    third = rng.random() < 0.25
    divisors = [d for d in (2, 3, 4, 5, 6) if final % d == 0 and final // d > 0]
    num = end_with(lang, rng)
    if lang == "en":
        q1 = f"A {c_s} holds {per} {en_p}. How many {en_p} are in {boxes} {c_p}?"
        a1 = f"{boxes} {c_p} hold {total} {en_p}."
        q2 = (f"I found {delta} more. How many do I have now? {num}" if add else f"Then I {verb[0]} {delta} of them. How many are left? {num}")
        turns = [(q1, a1), (q2, str(final))]
        if third and divisors:
            d = pick(rng, divisors)
            turns.append((f"Share them equally among {d} people. How many does each person get? {end_with(lang, rng)}", str(final // d)))
        return turns
    q1 = f"Bir {c_tr} içinde {per} {tr} var. {boxes} {c_loc} kaç {tr} olur?"
    a1 = f"{boxes} {c_loc} {total} {tr} olur."
    q2 = (f"{delta} tane daha buldum. Şimdi kaç tane oldu? {num}" if add else f"Sonra {delta} tanesini {verb[1]}. Kaç tane kaldı? {num}")
    turns = [(q1, a1), (q2, str(final))]
    if third and divisors:
        d = pick(rng, divisors)
        turns.append((f"Bunları {d} kişiye eşit paylaştır. Kişi başına kaç tane düşer? {end_with(lang, rng)}", str(final // d)))
    return turns


def lengths(rng, lang):
    kind = rng.randrange(4)
    num = end_with(lang, rng)
    if kind == 0:
        thing = pick(rng, LENGTHS)
        m, cut = rng.randint(2, 9), rng.randint(10, 90)
        total = m * 100
        if lang == "en":
            return [(f"A {thing[0]} is {m} metres long. How many centimetres is that?", f"It is {total} centimetres."),
                    (f"I cut off {cut} centimetres. How long is it now, in centimetres? {num}", str(total - cut))]
        return [(f"Bir {thing[1]} {m} metre uzunluğunda. Kaç santimetre eder?", f"{total} santimetre eder."),
                (f"{cut} santimetresini kestim. Şimdi kaç santimetre kaldı? {num}", str(total - cut))]
    if kind == 1:
        h, seen = rng.randint(2, 4), rng.choice([25, 40, 55, 70, 85, 100])
        total = h * 60
        what = pick(rng, [("film", "film"), ("concert", "konser"), ("lesson", "ders"), ("match", "maç")])
        if lang == "en":
            return [(f"A {what[0]} lasts {h} hours. How many minutes is that?", f"That is {total} minutes."),
                    (f"I have already watched {seen} minutes. How many minutes are left? {num}", str(total - seen))]
        return [(f"Bir {what[1]} {h} saat sürüyor. Kaç dakika eder?", f"{total} dakika eder."),
                (f"{seen} dakikasını izledim. Kaç dakikası kaldı? {num}", str(total - seen))]
    if kind == 2:
        kg, used = rng.randint(2, 9), rng.choice([150, 250, 300, 450, 600, 750])
        total = kg * 1000
        food = pick(rng, [("flour", "un"), ("rice", "pirinç"), ("sugar", "şeker"), ("lentils", "mercimek")])
        if lang == "en":
            return [(f"A bag of {food[0]} weighs {kg} kilograms. How many grams is that?", f"It is {total} grams."),
                    (f"I used {used} grams. How many grams are left? {num}", str(total - used))]
        return [(f"Bir çuval {food[1]} {kg} kilogram geliyor. Kaç gram eder?", f"{total} gram eder."),
                (f"{used} gramını kullandım. Kaç gram kaldı? {num}", str(total - used))]
    litres, poured = rng.randint(2, 6), rng.choice([200, 350, 500, 650, 800])
    total = litres * 1000
    if lang == "en":
        return [(f"A tank holds {litres} litres. How many millilitres is that?", f"It is {total} millilitres."),
                (f"I poured out {poured} millilitres. How many millilitres remain? {num}", str(total - poured))]
    return [(f"Bir depoya {litres} litre su sığıyor. Kaç mililitre eder?", f"{total} mililitre eder."),
            (f"{poured} mililitresini döktüm. Kaç mililitre kaldı? {num}", str(total - poured))]


def letters(rng, lang):
    word = pick(rng, WORDS[lang])
    num = end_with(lang, rng)
    count = len(word.replace(" ", ""))
    vowels = sum(c in ("aeiou" if lang == "en" else "aeıioöuü") for c in word.lower())
    if lang == "en":
        return [(f"How many letters are in “{word}”? {num}", str(count)),
                (pick(rng, ["And how many of those are vowels? ", "Now count only the vowels. "]) + end_with(lang, rng), str(vowels))]
    return [(f"“{word}” kelimesinde kaç harf var? {num}", str(count)),
            (pick(rng, ["Bunların kaçı ünlü? ", "Şimdi sadece ünlüleri say. "]) + end_with(lang, rng), str(vowels))]


def running(rng, lang):
    start, gain, loss = rng.randint(10, 60), rng.randint(5, 40), rng.randint(3, 30)
    mid = start + gain
    if loss >= mid:
        loss = mid - 1
    item = pick(rng, COUNTED)
    num1, num2 = end_with(lang, rng), end_with(lang, rng)
    if lang == "en":
        return [(f"I have {start} {item[1]}. A friend gives me {gain} more. How many do I have now? {num1}", str(mid)),
                (f"Then I {pick(rng, item[3])[0]} {loss}. How many remain? {num2}", str(mid - loss))]
    return [(f"Elimde {start} {item[2]} var. Bir arkadaşım {gain} tane daha veriyor. Şimdi kaç tane oldu? {num1}", str(mid)),
            (f"Sonra {loss} tanesini {pick(rng, item[3])[1]}. Kaç tane kaldı? {num2}", str(mid - loss))]


def duration(rng, lang):
    h, m = rng.randint(7, 20), rng.choice([0, 5, 10, 15, 20, 30, 35, 40, 45, 50])
    d = rng.choice([45, 60, 75, 90, 105, 120, 135])
    later = rng.choice([10, 15, 20, 30])
    what = pick(rng, [("film", "film"), ("class", "ders"), ("concert", "konser"), ("workshop", "atölye")])
    end = h * 60 + m + d
    end2 = end + later
    f = lambda t: f"{t // 60 % 24}:{t % 60:02d}"
    g = lambda t: f"{t // 60 % 24}.{t % 60:02d}"
    if lang == "en":
        return [(f"A {what[0]} starts at {f(h * 60 + m)} and lasts {d} minutes. At what time does it end?", f"It ends at {f(end)}."),
                (f"It starts {later} minutes later instead. What time does it end then? Just the time.", f(end2))]
    return [(f"Bir {what[1]} {g(h * 60 + m)} gibi başlıyor ve {d} dakika sürüyor. Saat kaçta biter?", f"Saat {g(end)} gibi biter."),
            (f"Bunun yerine {later} dakika geç başlıyor. O zaman saat kaçta biter? Sadece saati yaz.", g(end2))]


# ---------------------------------------------------------------- driver

KINDS = [("tracking", tracking, 0.23), ("label", label_kind, 0.40), ("clarify", clarify, 0.09), ("counting", counting, 0.28)]


def generate(count, seed=82828, prefix="p"):
    rng = random.Random(seed)
    out, seen = [], set()
    kinds = [k for k, _, _ in KINDS]
    weights = [w for _, _, w in KINDS]
    attempts = 0
    while len(out) < count:
        attempts += 1
        if attempts > count * 20:
            raise RuntimeError("Dialogue templates are too repetitive for the requested count")
        i = len(out)
        lang = "en" if i % 2 == 0 else "tr"
        name = rng.choices(kinds, weights)[0]
        turns = dict((k, f) for k, f, _ in KINDS)[name](rng, lang)
        key = re.sub(r"\W+", " ", " ".join(u + " " + a for u, a in turns).lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(id=f"{prefix}{i:05d}", lang=lang, source="programmatic", kind=name,
                        turns=[dict(user=u, assistant=a) for u, a in turns]))
    return out


if __name__ == "__main__":
    import json
    import sys

    for c in generate(int(sys.argv[1]) if len(sys.argv) > 1 else 12):
        print(json.dumps(c, ensure_ascii=False))
