"""Arithmetic conversations from code templates. Every answer is computed, never written by the teacher."""

import random

NAMES = {
    "en": ["Sam", "Maya", "Leo", "Nora", "Ben", "Ella", "Omar", "Ruth", "Jack", "Ivy"],
    "tr": ["Ali", "Ayşe", "Mert", "Elif", "Can", "Zeynep", "Emre", "Deniz", "Selin", "Burak"],
}
GENITIVE = {
    "Ali": "Ali'nin", "Ayşe": "Ayşe'nin", "Mert": "Mert'in", "Elif": "Elif'in", "Can": "Can'ın",
    "Zeynep": "Zeynep'in", "Emre": "Emre'nin", "Deniz": "Deniz'in", "Selin": "Selin'in", "Burak": "Burak'ın",
}
ITEMS = {
    "en": [("notebook", "notebooks"), ("pen", "pens"), ("apple", "apples"), ("ticket", "tickets"),
           ("bottle", "bottles"), ("bun", "buns"), ("stamp", "stamps"), ("candle", "candles")],
    "tr": ["defter", "kalem", "elma", "bilet", "şişe", "simit", "pul", "mum"],
}
SAY = {"+": ("plus", "artı"), "-": ("minus", "eksi"), "*": ("times", "çarpı"), "/": ("divided by", "bölü")}


def apply(op, a, b):
    return {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op]


def direct(rng, lang):
    op = rng.choice("+-*/")
    if op == "+":
        a, b = rng.randint(11, 99), rng.randint(11, 99)
    elif op == "-":
        a, b = rng.randint(30, 150), rng.randint(5, 29)
    elif op == "*":
        a, b = rng.randint(3, 12), rng.randint(3, 15)
    else:
        b = rng.randint(2, 12)
        a = b * rng.randint(2, 15)
    c = apply(op, a, b)
    en, tr = SAY[op]
    if lang == "en":
        asks = rng.choice([f"What is {a} {en} {b}?", f"Quick one: {a} {op} {b}?", f"How much is {a} {en} {b}?"])
        first = f"{a} {en} {b} is {c}."
    else:
        asks = rng.choice([f"{a} {tr} {b} kaç eder?", f"Hızlı bir soru: {a} {op} {b}?", f"{a} {tr} {b} kaçtır?"])
        first = f"{a} {tr} {b}, {c} eder."
    op2 = rng.choice("+-")
    d = rng.randint(2, 20)
    if op2 == "-" and c <= d:
        op2 = "+"
    e = apply(op2, c, d)
    en2, tr2 = SAY[op2]
    if lang == "en":
        more = rng.choice([f"Now {en2} {d}.", f"And then {en2} {d}?", f"What if I {'add' if op2 == '+' else 'take away'} {d} from that?"])
        second = f"{c} {en2} {d} is {e}."
    else:
        more = rng.choice(
            [f"Şimdi buna {d} ekle." if op2 == "+" else f"Şimdi bundan {d} çıkar.",
             f"Bu sonuca {d} eklesem kaç olur?" if op2 == "+" else f"Bu sonuçtan {d} çıkarsam kaç olur?"]
        )
        second = f"{c} {tr2} {d}, {e} eder."
    return [(asks, first), (more, second)]


def shopping(rng, lang):
    p1, p2 = rng.randint(4, 25), rng.randint(2, 15)
    n1, n2 = rng.randint(2, 6), rng.randint(2, 6)
    t1, t2 = p1 * n1, p2 * n2
    total = t1 + t2
    wallet = total + rng.randint(5, 60)
    left = wallet - total
    if lang == "en":
        (s1, pl1), (s2, pl2) = rng.sample(ITEMS["en"], 2)
        q = f"A {s1} costs {p1} lira and a {s2} costs {p2} lira. How much do {n1} {pl1} and {n2} {pl2} cost together?"
        a = f"The {pl1} cost {t1} lira and the {pl2} cost {t2} lira, so the total is {total} lira."
        q2 = f"I have {wallet} lira. How much is left after buying them?"
        a2 = f"You will have {left} lira left."
    else:
        i1, i2 = rng.sample(ITEMS["tr"], 2)
        q = f"Bir {i1} {p1} lira, bir {i2} {p2} lira. {n1} {i1} ile {n2} {i2} birlikte kaç lira tutar?"
        a = f"{n1} {i1} {t1} lira, {n2} {i2} {t2} lira eder. Toplam {total} lira."
        q2 = f"Cebimde {wallet} lira var. Bunları alınca ne kadar param kalır?"
        a2 = f"{left} lira paran kalır."
    return [(q, a), (q2, a2)]


def sharing(rng, lang):
    kids = rng.choice([2, 3, 4, 5, 6])
    each = rng.randint(2, 9)
    total = kids * each
    other = rng.choice([k for k in (2, 3, 4, 5, 6) if total % k == 0 and k != kids] or [kids])
    name = rng.choice(NAMES[lang])
    if lang == "en":
        q = f"{name} has {total} cookies to share equally among {kids} friends. How many does each friend get?"
        a = f"Each friend gets {each} cookies."
        q2 = f"What if there were {other} friends instead?"
        a2 = f"Then each friend would get {total // other} cookies."
    else:
        q = f"{GENITIVE[name]} {total} kurabiyesi var ve bunları {kids} arkadaşına eşit paylaştıracak. Her arkadaşa kaç kurabiye düşer?"
        a = f"Her arkadaşa {each} kurabiye düşer."
        q2 = f"Peki arkadaş sayısı {other} olsaydı?"
        a2 = f"O zaman her arkadaşa {total // other} kurabiye düşerdi."
    return [(q, a), (q2, a2)]


def clock(rng, lang):
    h, m = rng.randint(6, 19), rng.choice([0, 10, 15, 20, 30, 40, 45, 50])
    d = rng.choice([25, 35, 40, 50, 55, 70, 85])
    late = rng.choice([10, 15, 20])
    end = h * 60 + m + d
    after = end + late
    fmt = lambda t: f"{t // 60 % 24}:{t % 60:02d}"
    if lang == "en":
        q = f"A bus leaves at {h}:{m:02d} and the trip takes {d} minutes. What time does it arrive?"
        a = f"It arrives at {fmt(end)}."
        q2 = f"And if it leaves {late} minutes late?"
        a2 = f"Then it arrives at {fmt(after)}."
    else:
        q = f"Otobüs {h}.{m:02d} gibi kalkıyor ve yolculuk {d} dakika sürüyor. Varış saati ne olur?"
        a = f"Varış saati {end // 60 % 24}.{end % 60:02d} olur."
        q2 = f"Peki {late} dakika geç kalkarsa?"
        a2 = f"O zaman varış saati {after // 60 % 24}.{after % 60:02d} olur."
    return [(q, a), (q2, a2)]


def recipe(rng, lang):
    base_people = rng.choice([2, 4])
    per = rng.choice([50, 100, 150, 200]) // (base_people // 2) if base_people == 4 else rng.choice([50, 100, 150])
    amount = per * (base_people // 2) if base_people == 4 else per
    unit_people = amount / base_people
    target = rng.choice([p for p in (2, 3, 4, 6, 8, 10) if p != base_people])
    need = unit_people * target
    other = rng.choice([p for p in (2, 3, 4, 6, 8, 10) if p not in (base_people, target)])
    need2 = unit_people * other
    f = lambda x: str(int(x)) if x == int(x) else str(round(x, 1))
    if lang == "en":
        q = f"A recipe uses {amount} g of flour for {base_people} people. How much flour is needed for {target} people?"
        a = f"You need {f(need)} g of flour."
        q2 = f"And for {other} people?"
        a2 = f"For {other} people you need {f(need2)} g."
    else:
        q = f"Bir tarif {base_people} kişi için {amount} gram un istiyor. {target} kişi için ne kadar un gerekir?"
        a = f"{f(need)} gram un gerekir."
        q2 = f"Peki {other} kişi için?"
        a2 = f"{other} kişi için {f(need2)} gram gerekir."
    return [(q, a), (q2, a2)]


def convert(rng, lang):
    kind = rng.choice(["h", "m", "kg"])
    n = rng.randint(2, 9)
    if kind == "h":
        r, s = n * 60, "minutes"
        en = (f"How many minutes are in {n} hours?", f"There are {r} minutes in {n} hours.")
        tr = (f"{n} saatte kaç dakika vardır?", f"{n} saatte {r} dakika vardır.")
    elif kind == "m":
        r = n * 100
        en = (f"How many centimetres are in {n} metres?", f"There are {r} centimetres in {n} metres.")
        tr = (f"{n} metre kaç santimetre eder?", f"{n} metre {r} santimetre eder.")
    else:
        r = n * 1000
        en = (f"How many grams are in {n} kilograms?", f"There are {r} grams in {n} kilograms.")
        tr = (f"{n} kilogram kaç gram eder?", f"{n} kilogram {r} gram eder.")
    m = rng.randint(2, 9)
    while m == n:
        m = rng.randint(2, 9)
    r2 = m * (60 if kind == "h" else 100 if kind == "m" else 1000)
    if lang == "en":
        q2 = f"And {m}?"
        a2 = f"{m} is {r2}."
        return [en, (q2, a2)]
    return [tr, (f"Peki {m}?", f"{m} için {r2} eder.")]


def percent(rng, lang):
    price = rng.choice([40, 60, 80, 120, 150, 200, 240])
    pc = rng.choice([10, 20, 25, 50])
    cut = price * pc // 100
    final = price - cut
    pc2 = rng.choice([p for p in (10, 20, 25, 50) if p != pc])
    final2 = price - price * pc2 // 100
    if lang == "en":
        q = f"A jacket costs {price} lira and has a {pc} percent discount. What is the new price?"
        a = f"The discount is {cut} lira, so the new price is {final} lira."
        q2 = f"And with a {pc2} percent discount?"
        a2 = f"Then the price is {final2} lira."
    else:
        q = f"Bir ceket {price} lira ve yüzde {pc} indirimli. Yeni fiyat ne olur?"
        a = f"İndirim {cut} lira, yani yeni fiyat {final} lira olur."
        q2 = f"Peki indirim yüzde {pc2} olsaydı?"
        a2 = f"O zaman fiyat {final2} lira olurdu."
    return [(q, a), (q2, a2)]


def just_number(rng, lang):
    a, b = rng.randint(12, 90), rng.randint(12, 90)
    c = rng.randint(2, 9)
    ans1, ans2 = a + b, (a + b) * c
    if lang == "en":
        return [(f"What is {a} plus {b}? Just the number.", f"{ans1}"),
                (f"Now multiply that by {c}. Just the number.", f"{ans2}")]
    return [(f"{a} artı {b} kaç eder? Sadece sayıyı yaz.", f"{ans1}"),
            (f"Şimdi bunu {c} ile çarp. Sadece sayıyı yaz.", f"{ans2}")]


def speed(rng, lang):
    v, h, h2 = rng.choice([4, 5, 6, 12, 15, 20, 40, 50, 60, 80]), rng.randint(2, 9), rng.randint(2, 9)
    if h2 == h:
        h2 += 1
    if lang == "en":
        return [(f"A cyclist rides {v} km every hour. How far does the cyclist go in {h} hours?", f"The cyclist goes {v * h} km in {h} hours."),
                (f"And in {h2} hours?", f"In {h2} hours the cyclist goes {v * h2} km.")]
    return [(f"Bir bisikletçi saatte {v} km gidiyor. {h} saatte kaç km gider?", f"{h} saatte {v * h} km gider."),
            (f"Peki {h2} saatte?", f"{h2} saatte {v * h2} km gider.")]


TEMPLATES = [direct, shopping, sharing, clock, recipe, convert, percent, just_number, speed, direct, shopping]


def generate(count, seed=71717):
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < count:
        i = len(out)
        lang = "en" if i % 2 == 0 else "tr"
        turns = rng.choice(TEMPLATES)(rng, lang)
        if turns[0][0] in seen:
            continue
        seen.add(turns[0][0])
        out.append(
            dict(id=f"a{i:05d}", lang=lang, source="arithmetic",
                 turns=[dict(user=u, assistant=a) for u, a in turns])
        )
    return out


if __name__ == "__main__":
    import json
    for c in generate(14):
        print(json.dumps(c, ensure_ascii=False))
