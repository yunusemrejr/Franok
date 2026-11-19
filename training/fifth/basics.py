"""Short dialogues about calendar facts, counting, small sums and everyday knowledge.

Rows have the format of training.fifth.identity, so the filter and the trainer read them unchanged.
A question is left out when it is close to a held-out test turn, and a sum is left out when a test
turn contains the same pair of numbers, so the tests keep asking for things this set never showed.
"""

import argparse
import json
import random
import re
from pathlib import Path

from training.fifth import filter as held

ASCII = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")

DAYS = dict(
    en=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    tr=["pazartesi", "salı", "çarşamba", "perşembe", "cuma", "cumartesi", "pazar"],
)
DAYS_FROM = ["pazartesiden", "salıdan", "çarşambadan", "perşembeden", "cumadan", "cumartesiden", "pazardan"]
MONTHS = dict(
    en=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    tr=["ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık"],
)
MONTHS_FROM = ["ocaktan", "şubattan", "marttan", "nisandan", "mayıstan", "hazirandan", "temmuzdan", "ağustostan", "eylülden", "ekimden", "kasımdan", "aralıktan"]
SEASONS = dict(en=["spring", "summer", "autumn", "winter"], tr=["ilkbahar", "yaz", "sonbahar", "kış"])
SEASONS_FROM = ["ilkbahardan", "yazdan", "sonbahardan", "kıştan"]

CAPITALS = dict(
    en=[("Turkey", "Ankara"), ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Japan", "Tokyo"),
        ("Spain", "Madrid"), ("England", "London"), ("Russia", "Moscow"), ("Egypt", "Cairo"), ("Greece", "Athens")],
    tr=[("türkiye'nin", "Ankara'dır"), ("fransa'nın", "Paris'tir"), ("almanya'nın", "Berlin'dir"), ("italya'nın", "Roma'dır"),
        ("japonya'nın", "Tokyo'dur"), ("ispanya'nın", "Madrid'dir"), ("ingiltere'nin", "Londra'dır"), ("rusya'nın", "Moskova'dır"),
        ("mısır'ın", "Kahire'dir"), ("yunanistan'ın", "Atina'dır")],
)
OPPOSITES = dict(
    en=[("hot", "cold"), ("big", "small"), ("up", "down"), ("fast", "slow"), ("day", "night"), ("open", "closed"),
        ("happy", "sad"), ("early", "late"), ("light", "dark"), ("wet", "dry")],
    tr=[("sıcak", "soğuktur"), ("soğuk", "sıcaktır"), ("büyük", "küçüktür"), ("küçük", "büyüktür"), ("hızlı", "yavaştır"),
        ("yavaş", "hızlıdır"), ("gece", "gündüzdür"), ("gündüz", "gecedir"), ("açık", "kapalıdır"), ("kapalı", "açıktır"),
        ("mutlu", "üzgündür"), ("üzgün", "mutludur"), ("erken", "geçtir"), ("ıslak", "kurudur"), ("kuru", "ıslaktır"),
        ("aydınlık", "karanlıktır"), ("karanlık", "aydınlıktır")],
)

FACTS = dict(
    en=[
        (["how many days are in a week?", "how many days does a week have?"], ["A week has seven days.", "There are seven days in a week."]),
        (["how many months are in a year?", "how many months does a year have?"], ["A year has twelve months.", "There are twelve months in a year."]),
        (["how many hours are in a day?", "how many hours does a day have?"], ["A day has 24 hours.", "There are 24 hours in a day."]),
        (["how many minutes are in an hour?", "how many minutes is one hour?"], ["An hour has 60 minutes.", "There are 60 minutes in an hour."]),
        (["how many seconds are in a minute?", "how many seconds is one minute?"], ["A minute has 60 seconds.", "There are 60 seconds in a minute."]),
        (["how many days are in a year?", "how many days does a year have?"], ["A year has 365 days, and a leap year has 366."]),
        (["how many weeks are in a year?"], ["A year has about 52 weeks."]),
        (["how many seasons are there?", "what are the seasons?"], ["There are four seasons: spring, summer, autumn and winter."]),
        (["which days are the weekend?", "what days make up the weekend?"], ["The weekend is Saturday and Sunday."]),
        (["what is the first month of the year?", "which month comes first in the year?"], ["January is the first month of the year."]),
        (["what is the last month of the year?"], ["December is the last month of the year."]),
        (["which season is the coldest?"], ["Winter is the coldest season."]),
        (["which season is the hottest?"], ["Summer is the hottest season."]),
        (["how many continents are there?"], ["There are seven continents."]),
        (["how many planets are in the solar system?"], ["There are eight planets in the solar system."]),
        (["which planet do we live on?"], ["We live on Earth."]),
        (["what is the biggest planet?", "which planet is the largest?"], ["Jupiter is the biggest planet in the solar system."]),
        (["how many letters are in the english alphabet?"], ["The English alphabet has 26 letters."]),
        (["how many sides does a triangle have?"], ["A triangle has three sides."]),
        (["how many sides does a square have?"], ["A square has four equal sides."]),
        (["how many legs does a spider have?"], ["A spider has eight legs."]),
        (["how many legs does a cat have?"], ["A cat has four legs."]),
        (["how many legs does a bird have?"], ["A bird has two legs."]),
        (["what color is the sky?", "what colour is the sky?"], ["On a clear day the sky is blue."]),
        (["what color is grass?"], ["Grass is green."]),
        (["what color is snow?"], ["Snow is white."]),
        (["what do bees make?"], ["Bees make honey."]),
        (["what do cows give us?"], ["Cows give milk."]),
        (["where does the sun rise?"], ["The sun rises in the east and sets in the west."]),
        (["where does the sun set?"], ["The sun sets in the west."]),
        (["at what temperature does water freeze?"], ["Water freezes at 0 degrees Celsius."]),
        (["at what temperature does water boil?"], ["Water boils at 100 degrees Celsius at sea level."]),
        (["what is the largest ocean?"], ["The Pacific is the largest ocean."]),
        (["what is the tallest mountain in the world?"], ["Mount Everest is the tallest mountain in the world."]),
        (["what is the largest animal?"], ["The blue whale is the largest animal."]),
        (["what is the fastest land animal?"], ["The cheetah is the fastest land animal."]),
        (["how many is a dozen?"], ["A dozen is twelve."]),
        (["how many colors are in a rainbow?"], ["A rainbow has seven colors."]),
        (["which is bigger, an elephant or a mouse?"], ["An elephant is much bigger than a mouse."]),
        (["what is frozen water called?"], ["Frozen water is called ice."]),
        (["how many fingers are on one hand?"], ["One hand has five fingers."]),
    ],
    tr=[
        (["bir haftada kaç gün var?", "bir hafta kaç gündür?"], ["Bir haftada yedi gün vardır."]),
        (["bir yılda kaç ay var?", "bir yıl kaç aydır?"], ["Bir yılda on iki ay vardır."]),
        (["bir günde kaç saat var?", "bir gün kaç saattir?"], ["Bir günde 24 saat vardır."]),
        (["bir saatte kaç dakika var?", "bir saat kaç dakikadır?"], ["Bir saatte 60 dakika vardır."]),
        (["bir dakikada kaç saniye var?", "bir dakika kaç saniyedir?"], ["Bir dakikada 60 saniye vardır."]),
        (["bir yılda kaç gün var?", "bir yıl kaç gündür?"], ["Bir yılda 365 gün vardır, artık yıllarda 366."]),
        (["bir yılda kaç hafta var?"], ["Bir yılda yaklaşık 52 hafta vardır."]),
        (["kaç mevsim var?", "mevsimler nelerdir?"], ["Dört mevsim vardır: ilkbahar, yaz, sonbahar ve kış."]),
        (["hafta sonu hangi günlerdir?"], ["Hafta sonu cumartesi ve pazardır."]),
        (["yılın ilk ayı hangisidir?", "yılın ilk ayı ne?"], ["Yılın ilk ayı ocaktır."]),
        (["yılın son ayı hangisidir?"], ["Yılın son ayı aralıktır."]),
        (["en soğuk mevsim hangisi?"], ["En soğuk mevsim kıştır."]),
        (["en sıcak mevsim hangisi?"], ["En sıcak mevsim yazdır."]),
        (["dünyada kaç kıta var?", "kaç kıta var?"], ["Yedi kıta vardır."]),
        (["güneş sisteminde kaç gezegen var?"], ["Güneş sisteminde sekiz gezegen vardır."]),
        (["hangi gezegende yaşıyoruz?"], ["Dünya'da yaşıyoruz."]),
        (["en büyük gezegen hangisi?"], ["Güneş sisteminin en büyük gezegeni Jüpiter'dir."]),
        (["türk alfabesinde kaç harf var?"], ["Türk alfabesinde 29 harf vardır."]),
        (["üçgenin kaç kenarı var?"], ["Üçgenin üç kenarı vardır."]),
        (["karenin kaç kenarı var?"], ["Karenin dört eşit kenarı vardır."]),
        (["örümceğin kaç bacağı var?"], ["Örümceğin sekiz bacağı vardır."]),
        (["kedinin kaç bacağı var?"], ["Kedinin dört bacağı vardır."]),
        (["kuşun kaç bacağı var?"], ["Kuşun iki bacağı vardır."]),
        (["gökyüzü ne renk?", "gökyüzü ne renktir?"], ["Açık bir günde gökyüzü mavidir."]),
        (["çimen ne renk?"], ["Çimen yeşildir."]),
        (["kar ne renk?"], ["Kar beyazdır."]),
        (["arılar ne üretir?"], ["Arılar bal yapar."]),
        (["inekler bize ne verir?"], ["İnekler süt verir."]),
        (["güneş nereden doğar?"], ["Güneş doğudan doğar, batıdan batar."]),
        (["güneş nereden batar?"], ["Güneş batıdan batar."]),
        (["su kaç derecede donar?"], ["Su 0 santigrat derecede donar."]),
        (["su kaç derecede kaynar?"], ["Su deniz seviyesinde 100 santigrat derecede kaynar."]),
        (["en büyük okyanus hangisi?"], ["En büyük okyanus Büyük Okyanus'tur, Pasifik de denir."]),
        (["dünyanın en yüksek dağı hangisi?"], ["Dünyanın en yüksek dağı Everest'tir."]),
        (["en büyük hayvan hangisi?"], ["En büyük hayvan mavi balinadır."]),
        (["karada en hızlı koşan hayvan hangisi?"], ["Karada en hızlı koşan hayvan çitadır."]),
        (["bir düzine kaç tane eder?"], ["Bir düzine on iki tanedir."]),
        (["gökkuşağında kaç renk var?"], ["Gökkuşağında yedi renk vardır."]),
        (["fil mi büyük fare mi?"], ["Fil fareden çok daha büyüktür."]),
        (["donmuş suya ne denir?"], ["Donmuş suya buz denir."]),
        (["bir elde kaç parmak var?"], ["Bir elde beş parmak vardır."]),
    ],
)

SUMS = dict(
    en={
        "+": (["what's {a} plus {b}?", "what is {a} + {b}?", "{a}+{b}", "{a} + {b} = ?", "how much is {a} plus {b}?", "{a} plus {b}?"],
              ["{a} plus {b} is {c}.", "That's {c}.", "{a} + {b} = {c}."]),
        "-": (["what's {a} minus {b}?", "what is {a} - {b}?", "{a} - {b}", "{a} minus {b}?", "how much is {a} minus {b}?"],
              ["{a} minus {b} is {c}.", "That's {c}.", "{a} - {b} = {c}."]),
        "x": (["what's {a} times {b}?", "what is {a} x {b}?", "{a} * {b}", "{a} times {b}?", "what is {a} multiplied by {b}?"],
              ["{a} times {b} is {c}.", "That's {c}.", "{a} x {b} = {c}."]),
        "/": (["what's {a} divided by {b}?", "what is {a} / {b}?", "{a} divided by {b}?"],
              ["{a} divided by {b} is {c}.", "That's {c}."]),
    },
    tr={
        "+": (["{a} artı {b} kaç eder?", "{a} + {b} kaç?", "{a}+{b}", "{a} artı {b}?", "{a} ile {b} toplanınca kaç eder?"],
              ["{a} artı {b}, {c} eder.", "{c} eder.", "{a} + {b} = {c}."]),
        "-": (["{a} eksi {b} kaç eder?", "{a} - {b} kaç?", "{a} - {b}", "{a} eksi {b}?"],
              ["{a} eksi {b}, {c} eder.", "{c} eder.", "{a} - {b} = {c}."]),
        "x": (["{a} çarpı {b} kaç eder?", "{a} x {b} kaç?", "{a} * {b}", "{a} kere {b} kaç eder?"],
              ["{a} çarpı {b}, {c} eder.", "{c} eder.", "{a} x {b} = {c}."]),
        "/": (["{a} bölü {b} kaç eder?", "{a} / {b} kaç?", "{a} bölü {b}?"],
              ["{a} bölü {b}, {c} eder.", "{c} eder."]),
    },
)
LINK = dict(en=["and ", "ok, ", "one more: ", "next, "], tr=["peki ", "tamam, ", "bir de ", "şimdi "])


def cap(text, lang):
    head = text[0]
    if lang == "tr":
        head = {"i": "İ", "ı": "I"}.get(head, head.upper())
    else:
        head = head.upper()
    return head + text[1:]


def sum_turn(lang, rng):
    op = rng.choices("+-x/", weights=(4, 3, 3, 1))[0]
    if op == "+":
        a, b = rng.randint(0, 60), rng.randint(0, 60)
        c = a + b
    elif op == "-":
        a = rng.randint(1, 99)
        b = rng.randint(0, a)
        c = a - b
    elif op == "x":
        a, b = rng.randint(1, 12), rng.randint(1, 12)
        c = a * b
    else:
        b, c = rng.randint(1, 12), rng.randint(1, 12)
        a = b * c
    asks, answers = SUMS[lang][op]
    return rng.choice(asks).format(a=a, b=b), rng.choice(answers).format(a=a, b=b, c=c), {a, b}


def number_turn(lang, rng):
    kind = rng.choice(("after", "before", "half", "double"))
    n = rng.randint(1, 99)
    if kind == "half":
        n = 2 * rng.randint(1, 50)
    table = dict(
        en=dict(
            after=("what number comes after {n}?", "{m} comes after {n}.", n + 1),
            before=("what number comes before {n}?", "{m} comes before {n}.", n - 1),
            half=("what is half of {n}?", "Half of {n} is {m}.", n // 2),
            double=("what is double {n}?", "Double {n} is {m}.", 2 * n),
        ),
        tr=dict(
            after=("{n} sayısından sonra hangi sayı gelir?", "{n} sayısından sonra {m} gelir.", n + 1),
            before=("{n} sayısından önce hangi sayı gelir?", "{n} sayısından önce {m} gelir.", n - 1),
            half=("{n} sayısının yarısı kaç eder?", "{n} sayısının yarısı {m} eder.", n // 2),
            double=("{n} sayısının iki katı kaç eder?", "{n} sayısının iki katı {m} eder.", 2 * n),
        ),
    )
    ask, answer, m = table[lang][kind]
    return ask.format(n=n), answer.format(n=n, m=m), set()


def order_turn(lang, rng):
    names, froms, unit = rng.choice((
        (DAYS, DAYS_FROM, dict(en="day", tr="gün")),
        (MONTHS, MONTHS_FROM, dict(en="month", tr="ay")),
        (SEASONS, SEASONS_FROM, dict(en="season", tr="mevsim")),
    ))
    i = rng.randrange(len(names[lang]))
    after = rng.random() < 0.6
    j = (i + (1 if after else -1)) % len(names[lang])
    here, there, u = names[lang][i], names[lang][j], unit[lang]
    if lang == "en":
        word = "after" if after else "before"
        ask = rng.choice((f"what {u} comes {word} {here}?", f"which {u} is {word} {here}?", f"what's the {u} {word} {here}?"))
        answer = rng.choice((f"{cap(there, lang)} comes {word} {here}.", f"The {u} {word} {here} is {there}."))
    else:
        word = "sonra" if after else "önce"
        ask = rng.choice((f"{froms[i]} {word} hangi {u} gelir?", f"{froms[i]} {word}ki {u} hangisi?"))
        answer = f"{cap(froms[i], lang)} {word} {there} gelir."
    return ask, answer, (here.lower() if lang == "en" else froms[i], word)


def fact_turn(lang, rng):
    r = rng.random()
    if r < 0.14:
        place, city = rng.choice(CAPITALS[lang])
        if lang == "en":
            return rng.choice((f"what is the capital of {place}?", f"what's the capital of {place}?")), f"The capital of {place} is {city}.", set()
        return f"{place} başkenti neresi?", f"{cap(place, lang)} başkenti {city}.", set()
    if r < 0.26:
        if lang == "en":
            a, b = rng.choice(OPPOSITES[lang])
            if rng.random() < 0.5:
                a, b = b, a
            return f"what is the opposite of {a}?", f"The opposite of {a} is {b}.", set()
        a, b = rng.choice(OPPOSITES[lang])
        return rng.choice((f"{a} kelimesinin zıt anlamlısı nedir?", f"{a} kelimesinin zıttı ne?")), f"{cap(a, lang)} kelimesinin zıt anlamlısı {b}.", set()
    asks, answers = rng.choice(FACTS[lang])
    return rng.choice(asks), rng.choice(answers), set()


def typed(text, lang, rng, first):
    if not first and rng.random() < 0.35:
        text = rng.choice(LINK[lang]) + text
    if lang == "en" and rng.random() < 0.45:
        text = text.lower()
    elif rng.random() < 0.5:
        text = cap(text, lang)
    if rng.random() < 0.3:
        text = text.rstrip("?").rstrip()
    if lang == "tr" and rng.random() < 0.15:
        text = text.translate(ASCII)
    return text


def held_out():
    tests, _ = held.load_tests()
    pairs = set()
    for text, _, _ in tests:
        numbers = [int(n) for n in re.findall(r"\d+", text)]
        pairs |= {frozenset((a, b)) for a in numbers for b in numbers if a != b}
    return tests, {n for n, _, _ in tests}, pairs


def near_test(user, tests, exact):
    if held.normal(user) in exact:
        return True
    words = held.content(user)
    return any(len(tc) > 1 and held.jaccard(words, tc) >= 0.3 for _, tc, _ in tests)


def dialogues(count, seed=31, prefix="k"):
    rng = random.Random(seed)
    tests, exact, pairs = held_out()
    makers = (sum_turn, number_turn, order_turn, fact_turn)
    seen, rows, skipped = set(), [], 0
    while len(rows) < count:
        lang = "en" if len(rows) % 2 == 0 else "tr"
        turns = []
        for k in range(rng.choices((1, 2, 3), weights=(35, 40, 25))[0]):
            ask, answer, numbers = rng.choices(makers, weights=(38, 8, 20, 34))[0](lang, rng)
            asked = isinstance(numbers, tuple) and any(all(part in text.split() for part in numbers) for text in exact)
            if asked or (isinstance(numbers, set) and frozenset(numbers) in pairs) or near_test(ask, tests, exact):
                skipped += 1
                continue
            turns.append((typed(ask, lang, rng, first=not turns), answer))
        key = tuple(turns)
        if not turns or key in seen or len({a for _, a in turns}) < len(turns):
            continue
        seen.add(key)
        rows.append(
            dict(
                id=f"{prefix}{len(rows):06d}",
                lang=lang,
                source="identity",
                kind="basics",
                turns=[dict(user=u, assistant=a) for u, a in turns],
                split="dev" if len(rows) % 25 == 24 else "train",
            )
        )
    return rows, skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows, skipped = dialogues(a.count, a.seed)
    with a.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(len(rows), "->", a.out, "questions left out as close to a test turn:", skipped)


if __name__ == "__main__":
    main()
