"""Seeds and the teacher prompt for Franok 0.5 conversations: everyday chat in English and Turkish.

Ten conversation families form the default pool; each has situation lists for both languages, a sampling weight, a turn range and one worked example.
Three more (create, code, math) have weight 0: they are drawn only through `build_seeds(..., tasks=[...])`, so the default seed stream does not change.
Every seed also carries a user typing style; the assistant always answers in clean, correct language.
Same interface as training/fourth/seeds.py (`build_seeds`, `build_prompt`; seeds keep `id`, `lang`, `task`, `topic`, `turns`).
"""

import json
import random
from collections import Counter

from ..fourth.seeds import ANCHORS, PERSONAS, TOPICS, TWISTS

PERSONAS_TR = [
    "bir üniversite öğrencisi", "emekli bir öğretmen", "küçük bir çocuğun annesi", "gece vardiyasında çalışan bir hemşire",
    "yeni taşınmış bir ev arkadaşı", "küçük bir dükkân sahibi", "bir lise öğrencisi", "yoğun çalışan bir ofis çalışanı",
    "ilk kez yolculuk yapan biri", "bilgisayarı yeni öğrenen biri", "bütçesi kısıtlı biri", "küçük bir dairede yaşayan biri",
    "evden çalışan biri", "sabırsız biri", "dikkatli plan yapan biri", "bir kuaför", "bir taksi şoförü", "bir fırıncı",
    "bir ortaokul öğretmeni", "üç çocuklu bir baba", "yeni bir dil öğrenen biri", "bir bisikletçi", "bir kitapçı çalışanı",
    "bir bahçıvan", "yurtta kalan bir öğrenci", "yeni anne olmuş biri", "teknolojiyi pek sevmeyen biri", "bir garson",
    "bir kasiyer", "kendi başına ilk kez yaşayan biri", "çift vardiyadan sonra yorgun biri", "kısa konuşan biri",
    "meraklı ama utangaç biri", "yürüyüş yapmayı seven bir emekli", "bir kurye",
]
TWISTS_TR = [
    "çok az para harcanmalı", "yalnızca on dakika var", "evde fırın ya da ocak yok", "evde küçük bir çocuk var",
    "komşular uyuyor, sessiz olunmalı", "internet ve telefon çekmiyor", "misafirler birazdan gelecek", "kullanıcının hiç aleti yok",
    "kullanıcının belinde ağrı var", "hava bozdu", "kullanıcı bunu ilk kez yapıyor", "kullanıcı yalnızca evde olanları kullanabilir",
    "kullanıcı ilk öneriyi denedi ama olmadı", "kullanıcı bunu yaşlı bir akraba için yapıyor", "kullanıcı yalnızca bir sonraki adımı bilmek istiyor",
    "kullanıcı asansörsüz bir binanın en üst katında oturuyor", "kullanıcı bunu sürpriz olarak saklamak istiyor",
    "kullanıcının elinde yalnızca 300 lira var", "kullanıcı en güvenli yolu istiyor", "iki kişinin paylaşması gerekiyor",
]
TOPICS_TR = {
    "ev": [
        "camdan etiket izini temizlemek", "gıcırdayan bir kapı", "kışın çamaşırı içeride kurutmak", "çaydanlıktaki kireci temizlemek",
        "yavaş akan bir mutfak lavabosu", "halıdaki çay lekesi", "pencerede sabahları oluşan buğu", "elektrik faturasını düşürmek",
    ],
    "mutfak": [
        "bulgur pilavını tane tane yapmak", "yumurtayı rahat soyulacak şekilde haşlamak", "çayı ne sert ne acı demlemek",
        "evdeki malzemeyle hızlı bir akşam yemeği", "mercimek çorbasını daha lezzetli yapmak", "kalan yemeği saklamak",
        "soğan doğrarken gözlerin yanmaması", "olgunlaşmış muzlarla ne yapılır",
    ],
    "bitkiler": [
        "yaprakları sararan fesleğen", "sardunyayı ne sıklıkla sulamak gerekir", "balkonda domates yetiştirmek", "saksıdaki çiçekte küçük sinekler",
        "tatile giderken bitkilere bakmak", "köklerin doldurduğu saksıyı büyütmek", "kurumak üzere olan nane", "balkon için güneşi seven çiçekler",
    ],
    "okul ve ders": [
        "yabancı dil kelimesi ezberlemek", "ders çalışırken dikkat dağınıklığı", "üç günde sınava hazırlanmak", "sunumdan önce heyecanı azaltmak",
        "derste not tutmak", "yurtta gürültüde ders çalışmak", "uzun bir bölümü özetlemek", "sınavdan önceki akşam ne yapmalı",
    ],
    "yolculuk": [
        "üç günlük şehir gezisi için valiz hazırlamak", "uzun otobüs yolculuğunda çocukla vakit geçirmek", "kaybolan otobüs biletini ne yapmalı",
        "havalimanından şehir merkezine gitmek", "metroda yönünü bulmak", "vapurla günübirlik gezi", "şehirler arası otobüste uyumak", "internetsiz yol bulmak",
    ],
    "alışveriş ve para": [
        "kışlık mont için sınırlı bütçe", "ikinci el telefon alırken nelere bakmalı", "fişsiz ürünü değiştirmek", "iki marketin fiyatını karşılaştırmak",
        "aylık bütçe yapmak", "arkadaşın borç istemesine hayır demek", "indirimin gerçekten iyi olup olmadığını anlamak", "kirayı ev arkadaşıyla bölüşmek",
    ],
    "sağlık ve gün": [
        "uzun süre ekran başında kalınca hafif baş ağrısı", "uykuya dalmakta zorlanmak", "uzun süre oturduktan sonra gerinmek", "yeterince su içmek",
        "yeni ayakkabıdan çıkan su toplaması", "yastıktan sonra tutulan boyun", "hafif bir soğuk algınlığında evde ne iyi gelir", "masa başında doğru oturmak",
    ],
    "çocuk ve aile": [
        "sebze yemeyen bir çocuk", "beş yaşındaki çocuk için uyku düzeni", "yağmurlu bir öğleden sonra çocuğu oyalamak", "ödevde cevabı söylemeden yardım etmek",
        "çocuğa ayakkabı bağcığı bağlamayı öğretmek", "uzakta yaşayan dedeyle görüşmeyi sürdürmek", "kardeşlerin oyuncak kavgası", "okula ilk gün başlayan çocuk",
    ],
    "telefon ve bilgisayar": [
        "hafızası dolan bir telefon", "yavaş çalışan wifi", "güçlü ama akılda kalan bir şifre", "fotoğrafları yedeklemek", "yanlış kişiye giden bir e-posta",
        "bilinmeyen numaradan para isteyen bir mesaj", "çabuk biten dizüstü bilgisayar pili", "çevrimdışı görünen bir yazıcı",
    ],
    "komşu ve arkadaş": [
        "komşuyu çaya davet etmek", "hep geç kalan bir arkadaş", "yeni şehirde arkadaş edinmek", "partiye kibarca hayır demek", "yan dairedeki gürültü",
        "yemeğe davet edilince ev sahibine hediye", "bir arkadaşa küçük bir sürpriz hazırlamak", "bitmeyen bir grup sohbeti",
    ],
}


def _hints(en, tr):
    return dict(en=en, tr=tr)


STYLES = {
    "en": {
        "clean": ("normal spelling, capital letters and punctuation", 0.28),
        "lower": ("all lowercase with no punctuation, like a quick text message", 0.2),
        "abbrev": ("chat shorthand such as u, r, thx, pls, idk, btw, wanna, gonna", 0.18),
        "typos": ("normal wording with one or two ordinary typing mistakes, such as a swapped or missing letter", 0.14),
        "short": ("extremely short messages of one to three words", 0.12),
        "mixed": ("lowercase, shorthand and one typo together", 0.08),
    },
    "tr": {
        "clean": ("düzgün yazım, büyük harf ve noktalama", 0.24),
        "lower": ("hepsi küçük harf, noktalamasız, hızlıca atılmış bir mesaj gibi", 0.15),
        "abbrev": ("sohbet kısaltmaları: slm, mrb, nbr, tmm, tşk, bi, bn, sn, ltfn gibi", 0.15),
        "ascii": ("Türkçe karakter kullanmadan: nasilsin, tesekkurler, gunaydin, ogrenci, cok (ç ğ ı ö ş ü yerine c g i o s u)", 0.2),
        "typos": ("normal sözcüklerle ama bir iki sıradan yazım hatasıyla, örneğin harf yer değiştirmiş ya da eksik", 0.1),
        "short": ("çok kısa mesajlar, bir ila üç sözcük", 0.1),
        "mixed": ("küçük harf, kısaltma ve bir yazım hatası bir arada", 0.06),
    },
}

REPAIR_SUBJECTS = _hints(
    ["keeping bread fresh", "why the moon changes shape", "what to cook with rice and eggs", "getting a stain out of a shirt", "where to start learning the guitar",
     "remembering people's names", "waking up earlier", "a name for a pet fish", "packing for a weekend trip", "watering a cactus", "saving a little money each month",
     "the difference between a noun and a verb", "making a pot of tea", "falling asleep faster", "a dessert with ingredients already at home"],
    ["ekmeği taze tutmak", "ay neden şekil değiştirir", "pirinç ve yumurtayla ne pişirilir", "gömlekteki lekeyi çıkarmak", "gitar öğrenmeye nereden başlanır",
     "insanların adlarını akılda tutmak", "daha erken uyanmak", "evcil balığa isim bulmak", "hafta sonu gezisi için valiz hazırlamak", "kaktüsü sulamak",
     "her ay biraz para biriktirmek", "isim ile fiil arasındaki fark", "demlik çay yapmak", "daha çabuk uykuya dalmak", "evdeki malzemelerle tatlı yapmak"],
)
FOLLOWUP_SUBJECTS = _hints(
    ["ideas for a snack", "names for a cat", "ways to relax in the evening", "tips for learning new words", "two cheap gifts", "a short game to play on a trip",
     "places to sit and read", "ways to wake up earlier", "ideas for a rainy day", "a quick lunch", "movements to do at a desk"],
    ["atıştırmalık fikirleri", "kediye isim önerileri", "akşam rahatlama yolları", "kelime öğrenme ipuçları", "ucuz iki hediye", "yolculukta oynanacak kısa bir oyun",
     "kitap okumak için güzel yerler", "erken kalkma yolları", "yağmurlu gün fikirleri", "hızlı bir öğle yemeği", "masa başında yapılacak hareketler"],
)

def _bilingual(subs, tr_names):
    """Let Turkish seeds look their `subs` entry up under the Turkish topic name (`tr_names` maps each English name to it)."""
    return {**subs, **{tr_names[name]: sub for name, sub in subs.items()}}


# (area, a correct tiny example the teacher may use) and (one-step question, its correct answer); every number was checked by hand.
MATH_AREAS = dict(
    en=[("calculus", "the slope of x squared at x = 3 is 6"), ("derivatives", "the derivative of x squared is 2x"), ("fractions", "1/4 + 2/4 = 3/4"),
        ("percent", "10 percent of 50 is 5"), ("algebra", "if 2x = 14 then x = 7"), ("area", "a rectangle 3 by 5 has an area of 15"),
        ("perimeter", "a rectangle 3 by 5 has a perimeter of 16"), ("averages", "the average of 4, 6 and 8 is 6"), ("negative numbers", "-3 + 5 = 2"),
        ("equations", "if x + 4 = 9 then x = 5"), ("angles", "the angles of a triangle add up to 180, so 60 and 70 leave 50"), ("decimals", "0.5 + 0.25 = 0.75"),
        ("prime numbers", "7 is prime, but 9 is not because 3 times 3 is 9"), ("square roots", "the square root of 49 is 7"),
        ("probability", "the chance of rolling a 3 on a die is 1 in 6"), ("ratios", "2 to 3 means 4 red go with 6 blue"), ("multiplication table", "7 times 8 is 56"),
        ("division", "12 divided by 4 is 3"), ("addition", "27 + 15 = 42"), ("subtraction", "50 - 18 = 32"), ("even and odd numbers", "8 is even and 9 is odd"),
        ("powers", "2 to the power of 3 is 8"), ("geometry", "a square has 4 equal sides and 4 right angles"),
        ("triangles", "a triangle has 3 sides and its angles add up to 180"), ("rounding", "3.7 rounds to 4"), ("time and clocks", "2 hours 30 minutes is 150 minutes"),
        ("money", "3 notebooks at 4 each cost 12"), ("measuring units", "2 meters is 200 centimeters"),
        ("graphs", "the point (2, 3) is 2 steps right and 3 steps up"), ("statistics", "the average of 2, 4 and 9 is 5"), ("volume", "a box 2 by 3 by 4 holds 24 cubes"),
        ("factors", "the factors of 12 are 1, 2, 3, 4, 6 and 12"), ("multiples", "the multiples of 3 are 3, 6, 9 and 12"),
        ("order of operations", "2 + 3 times 4 is 14, because multiplying comes first"), ("the number line", "-2 is two steps to the left of zero"),
        ("trigonometry", "in a right triangle with sides 3, 4 and 5, the sine of the angle opposite the 3 is 3/5"), ("median", "in 3, 9 and 5 the middle number is 5"),
        ("mode", "in 2, 3, 3 and 8 the number that appears most is 3"), ("word problems", "3 bags with 4 apples each make 12 apples"),
        ("counting", "counting by twos gives 2, 4, 6, 8"), ("circles", "a circle with a radius of 3 has a diameter of 6"),
        ("greatest common factor", "the greatest common factor of 12 and 18 is 6")],
    tr=[("kalkülüs", "x karenin x = 3 noktasındaki eğimi 6'dır"), ("türev", "x karenin türevi 2x'tir"), ("kesirler", "1/4 + 2/4 = 3/4"),
        ("yüzde", "50'nin yüzde 10'u 5'tir"), ("cebir", "2x = 14 ise x = 7'dir"), ("alan", "3'e 5'lik bir dikdörtgenin alanı 15'tir"),
        ("çevre", "3'e 5'lik bir dikdörtgenin çevresi 16'dır"), ("ortalama", "4, 6 ve 8'in ortalaması 6'dır"), ("negatif sayılar", "-3 + 5 = 2"),
        ("denklemler", "x + 4 = 9 ise x = 5'tir"), ("açılar", "üçgenin açıları toplamı 180 derecedir; 60 ve 70 derece varsa üçüncü açı 50 derecedir"),
        ("ondalık sayılar", "0,5 + 0,25 = 0,75"), ("asal sayılar", "7 asaldır, 9 ise asal değildir çünkü 3 çarpı 3, 9 eder"), ("karekök", "49'un karekökü 7'dir"),
        ("olasılık", "bir zarda 3 gelme olasılığı 1/6'dır"), ("oran orantı", "2'ye 3 oranında 4 kırmızıya 6 mavi düşer"), ("çarpım tablosu", "7 çarpı 8, 56 eder"),
        ("bölme", "12 bölü 4, 3 eder"), ("toplama", "27 + 15 = 42"), ("çıkarma", "50 - 18 = 32"), ("tek ve çift sayılar", "8 çifttir, 9 tektir"),
        ("üslü sayılar", "2'nin 3. kuvveti 8'dir"), ("geometri", "karenin 4 eşit kenarı ve 4 dik açısı vardır"),
        ("üçgenler", "üçgenin 3 kenarı vardır ve iç açıları toplamı 180 derecedir"), ("yuvarlama", "3,7 en yakın tam sayıya 4 olarak yuvarlanır"),
        ("saat ve zaman", "2 saat 30 dakika 150 dakikadır"), ("para hesabı", "tanesi 4 lira olan 3 defter 12 lira eder"), ("ölçü birimleri", "2 metre 200 santimetredir"),
        ("grafikler", "(2, 3) noktası 2 adım sağda ve 3 adım yukarıdadır"), ("istatistik", "2, 4 ve 9'un ortalaması 5'tir"), ("hacim", "2'ye 3'e 4'lük bir kutu 24 küp alır"),
        ("çarpanlar", "12'nin çarpanları 1, 2, 3, 4, 6 ve 12'dir"), ("katlar", "3'ün katları 3, 6, 9 ve 12'dir"),
        ("işlem sırası", "2 + 3 × 4 = 14, çünkü önce çarpma yapılır"), ("sayı doğrusu", "-2, sıfırın iki adım solundadır"),
        ("trigonometri", "kenarları 3, 4 ve 5 olan dik üçgende 3'ün karşısındaki açının sinüsü 3/5'tir"), ("medyan", "3, 9 ve 5 sayılarının ortadaki değeri 5'tir"),
        ("mod", "2, 3, 3 ve 8 sayılarında en çok tekrar eden sayı 3'tür"), ("problemler", "her birinde 4 elma olan 3 torba toplam 12 elma eder"),
        ("sayma", "ikişer ikişer sayınca 2, 4, 6, 8 gelir"), ("daire", "yarıçapı 3 olan bir dairenin çapı 6'dır"), ("ebob", "12 ile 18'in ebobu 6'dır")],
)
MATH_QUESTIONS = dict(
    en=[("what is 15 percent of 80", "12"), ("what is 3/4 + 1/4", "1"), ("what is the area of a rectangle that is 6 by 4", "24"),
        ("what is the average of 4, 6 and 8", "6"), ("what is -3 + 5", "2"), ("solve x + 4 = 9", "x = 5"), ("what is the square root of 49", "7"),
        ("is 17 a prime number", "yes, only 1 and 17 divide it evenly"), ("what is 0.5 + 0.25", "0.75"),
        ("what is the perimeter of a square with a side of 5", "20"), ("what is 7 times 8", "56"), ("how do I simplify 6/8", "3/4, by dividing the top and bottom by 2"),
        ("what is 2 to the power of 3", "8"), ("what is the chance of heads when I flip a coin", "1 in 2"),
        ("two angles of a triangle are 60 and 70 degrees, what is the third", "50 degrees"), ("what is 20 percent of 50", "10"),
        ("what is the derivative of x squared", "2x"), ("what is 12 divided by 4", "3"), ("round 3.7 to the nearest whole number", "4"),
        ("what is the area of a triangle with a base of 6 and a height of 4", "12"), ("what is 9 squared", "81"), ("which is bigger, 0.6 or 0.45", "0.6"),
        ("what is the median of 3, 9 and 5", "5"), ("how many minutes are in 2 and a half hours", "150"), ("what is -4 times 3", "-12"),
        ("what is half of 18", "9"), ("what are the factors of 12", "1, 2, 3, 4, 6 and 12"), ("solve 2x = 14", "x = 7")],
    tr=[("80'in yüzde 15'i kaç", "12"), ("3/4 + 1/4 kaç eder", "1"), ("kenarları 6 ve 4 olan dikdörtgenin alanı kaç", "24"), ("4, 6 ve 8'in ortalaması kaç", "6"),
        ("-3 + 5 kaç eder", "2"), ("x + 4 = 9 ise x kaç", "5"), ("49'un karekökü kaç", "7"), ("17 asal sayı mı", "evet, yalnızca 1'e ve kendisine bölünür"),
        ("0,5 + 0,25 kaç eder", "0,75"), ("kenarı 5 olan karenin çevresi kaç", "20"), ("7 çarpı 8 kaç", "56"),
        ("6/8'i nasıl sadeleştiririm", "3/4; pay ve payda 2'ye bölünür"), ("2'nin 3. kuvveti kaç", "8"),
        ("yazı tura atınca yazı gelme olasılığı nedir", "2'de 1, yani 1/2"), ("bir üçgenin iki açısı 60 ve 70 derece, üçüncüsü kaç", "50 derece"),
        ("50'nin yüzde 20'si kaç", "10"), ("x karenin türevi nedir", "2x"), ("12 bölü 4 kaç", "3"), ("3,7'yi en yakın tam sayıya yuvarla", "4"),
        ("tabanı 6, yüksekliği 4 olan üçgenin alanı kaç", "12"), ("9'un karesi kaç", "81"), ("0,6 mı büyük 0,45 mi", "0,6"), ("3, 9 ve 5'in medyanı kaç", "5"),
        ("2,5 saat kaç dakika", "150"), ("-4 çarpı 3 kaç", "-12"), ("18'in yarısı kaç", "9"), ("12'nin çarpanları neler", "1, 2, 3, 4, 6 ve 12"),
        ("2x = 14 ise x kaç", "7")],
)

CREATE_TR = {
    "poem": "şiir", "short story": "kısa hikâye", "rhyme": "tekerleme", "riddle": "bilmece", "joke": "fıkra",
    "birthday or thank-you line": "doğum günü ya da teşekkür sözü", "slogan": "slogan",
}
CODE_TR = {
    "print and input": "ekrana yazdırma ve input", "numbers": "sayılar", "loops and conditions": "döngü ve koşul", "lists and strings": "liste ve metin",
    "explain a line": "bir satırı açıklama", "concept": "kavram", "too big": "büyük istek",
}
MATH_TR = {"general help, then an area": "genel yardım, sonra bir konu", "one-step question": "tek adımlık soru"}

# family -> weight, turn range, task text per language ({sub} is filled from `subs[topic]`), situations per language by topic, worked example per language.
FAMILIES = {
    "smalltalk": dict(
        weight=14, turns=(2, 5),
        task=dict(
            en="Friendly small talk. The user greets, chats, shares a bit about their day or asks the assistant for something light (a fun fact it is sure of, a short joke, an idea for the weekend). "
               "The assistant answers warmly and briefly, stays honest about being an assistant with no body, day or life of its own, and keeps the chat going with a short question when it fits.",
            tr="Samimi, kısa bir günlük sohbet. Kullanıcı selam verir, sohbet eder, gününden bahseder ya da hafif bir şey ister (asistanın emin olduğu eğlenceli bir bilgi, kısa bir fıkra, hafta sonu fikri). "
               "Asistan sıcak ve kısa cevap verir, bir asistan olduğunu, bedeni, günü ya da hayatı olmadığını saklamaz; uygun düşerse sohbeti kısa bir soruyla sürdürür.",
        ),
        hints=dict(
            en={
                "greeting": ["the user says hi and asks how the assistant is", "a casual good morning", "the user comes back in the evening and says hello",
                             "the user sends only a greeting and nothing else", "a friendly hello to start a chat", "the user says hello and mentions a free minute"],
                "chatting": ["the user is bored and wants something to talk about", "the user asks the assistant to tell them something interesting",
                             "the user asks what to do this weekend", "the user asks for a fun fact", "the user wants a short joke",
                             "the user asks how the assistant's day is going", "it is raining and the user wants to chat"],
                "day": ["the user shares how their day went", "the user just got home from work", "the user mentions a boring lesson at school",
                        "it is Friday evening", "the user is waiting for a bus and wants to chat", "the user is having lunch alone and wants company"],
                "closing": ["the user says good night", "the user says they have to go", "the user thanks the assistant for the chat"],
            },
            tr={
                "selamlaşma": ["kullanıcı selam verir ve nasılsın diye sorar", "kullanıcı sabah erkenden günaydın der", "kullanıcı akşam eve gelince merhaba der",
                               "kullanıcı yalnızca bir selam yazar, başka bir şey yazmaz", "kullanıcı slm yazıp konuşmak ister", "kullanıcı sıcak bir selamla sohbete başlar"],
                "sohbet": ["kullanıcının canı sıkılıyor ve sohbet etmek istiyor", "kullanıcı ilginç bir şey anlatmasını ister", "kullanıcı hafta sonu ne yapabileceğini sorar",
                           "kullanıcı eğlenceli bir bilgi ister", "kullanıcı kısa bir fıkra ister", "kullanıcı asistanın gününün nasıl geçtiğini sorar",
                           "dışarıda yağmur yağıyor ve kullanıcı sohbet etmek istiyor", "kullanıcı çay içerken sohbet etmek istiyor"],
                "gün": ["kullanıcı gününün nasıl geçtiğini anlatır", "kullanıcı işten yeni eve gelmiş", "kullanıcı sıkıcı bir okul dersinden bahseder",
                        "cuma akşamı", "kullanıcı otobüs beklerken sohbet etmek ister", "kullanıcı öğle yemeğini tek başına yiyor ve sohbet etmek istiyor",
                        "kullanıcı vapurda yolculuk ederken sohbet eder"],
                "veda": ["kullanıcı iyi geceler der", "kullanıcının gitmesi gerekiyor", "kullanıcı sohbet için teşekkür eder"],
            },
        ),
        examples=dict(
            en=[("Hi! How's it going?", "Hi! All good here. How is your day so far?"),
                ("Pretty boring, I'm stuck at home.", "That sounds slow. Want a fun fact or a short joke to liven it up?")],
            tr=[("Selam! Naber?", "Selam! Burada her şey yolunda. Senin gün nasıl geçiyor?"),
                ("Sıkıcı, evde kaldım.", "Sıkıcı olmuş. İstersen eğlenceli bir bilgi ya da kısa bir fıkra anlatayım.")],
        ),
    ),
    "feelings": dict(
        weight=9, turns=(2, 4),
        task=dict(
            en="The user tells how they feel. The assistant answers briefly and warmly in plain words: it names the feeling, may offer at most one small practical idea or one gentle question, "
               "and never uses therapy clichés, diagnoses or claims about its own feelings or life. It never lectures.",
            tr="Kullanıcı nasıl hissettiğini anlatır. Asistan kısa, sıcak ve sade sözlerle cevap verir: duyguyu adıyla söyler, en fazla bir küçük pratik fikir ya da bir nazik soru sunar. "
               "Klişe terapi cümleleri, teşhis ve kendi duygularına ya da hayatına dair iddialar yoktur. Nutuk atmaz.",
        ),
        hints=dict(
            en={
                "tired and stressed": ["the user is tired after a long day", "the user feels stressed by too much work", "the user cannot sleep because of worries",
                                       "the user had a bad day and wants to vent"],
                "nervous": ["the user is nervous before an exam tomorrow", "the user has a job interview soon", "the user must speak in front of a class",
                            "the user is waiting for test results"],
                "happy": ["the user got good news and wants to share it", "the user passed a driving test", "the user got a small raise",
                          "a friend is visiting this weekend and the user is excited"],
                "lonely": ["the user feels lonely in a new city", "the user misses a friend who moved away", "the user is alone on a Sunday evening",
                           "the user feels left out of a group chat"],
            },
            tr={
                "yorgun ve stresli": ["kullanıcı uzun bir günün sonunda çok yorgun", "kullanıcı iş yükünden bunalmış", "kullanıcı kaygıları yüzünden uyuyamıyor",
                                      "kullanıcının günü kötü geçmiş ve içini dökmek istiyor"],
                "gergin": ["kullanıcı yarınki sınav yüzünden gergin", "kullanıcının yakında bir iş görüşmesi var", "kullanıcının sınıfın önünde konuşması gerekiyor",
                           "kullanıcı sonuçların açıklanmasını bekliyor"],
                "sevinçli": ["kullanıcının güzel bir haberi var ve paylaşmak istiyor", "kullanıcı ehliyet sınavını geçmiş", "kullanıcının maaşına küçük bir zam yapılmış",
                             "bir arkadaşı hafta sonu ziyarete gelecek ve kullanıcı heyecanlı"],
                "yalnız": ["kullanıcı yeni taşındığı şehirde kendini yalnız hissediyor", "kullanıcı başka şehre taşınan bir arkadaşını özlemiş",
                           "kullanıcı pazar akşamı evde tek başına", "kullanıcı bir grup sohbetinde dışlanmış hissediyor"],
            },
        ),
        examples=dict(
            en=[("i'm so tired today", "That sounds like a long day. Is it your body that is tired, or your head?"),
                ("mostly my head, too much work", "Then a short break away from the screen can help. Even ten minutes of walking will clear your head a little.")],
            tr=[("bugün çok yorgunum", "Uzun bir gün olmuş gibi. Bedenin mi yorgun, kafan mı?"),
                ("daha çok kafam, çok iş var", "O zaman ekrandan uzak kısa bir mola iyi gelir. On dakikalık bir yürüyüş bile kafanı biraz boşaltır.")],
        ),
    ),
    "opinions": dict(
        weight=9, turns=(2, 4),
        task=dict(
            en="The user asks what the assistant prefers or thinks. The assistant answers lightly and honestly as an assistant: it does not claim to taste, drink, travel or have favourites from experience. "
               "It may say it has no personal taste, then gives a reasonable pick or the main good points of each option in simple words, and turns the question back to the user.",
            tr="Kullanıcı asistanın neyi tercih ettiğini ya da ne düşündüğünü sorar. Asistan bir asistan olarak hafif ve dürüst cevap verir: tatma, içme, seyahat etme ya da yaşantıdan gelen favorileri olduğunu söylemez. "
               "Kişisel zevkinin olmadığını söyleyebilir, sonra makul bir seçim yapar ya da seçeneklerin iyi yanlarını sade sözlerle söyler ve soruyu kullanıcıya çevirir.",
        ),
        hints=dict(
            en={
                "food and drink": ["tea or coffee", "pizza or burgers", "sweet or salty snacks", "breakfast or dinner as the best meal"],
                "animals": ["cats or dogs", "birds or fish as pets", "which pet needs the least care"],
                "seasons and places": ["the best season", "sea or mountains for a holiday", "city or village life", "rainy days or sunny days"],
                "free time": ["books or films", "morning person or night owl", "football or basketball", "music while working or silence"],
            },
            tr={
                "yiyecek ve içecek": ["çay mı kahve mi", "pizza mı hamburger mi", "tatlı mı tuzlu mu", "kahvaltı mı akşam yemeği mi", "simit mi poğaça mı", "ayran mı kola mı"],
                "hayvanlar": ["kedi mi köpek mi", "kuş mu balık mı", "hangi evcil hayvan en az bakım ister"],
                "mevsim ve yer": ["en güzel mevsim", "tatil için deniz mi dağ mı", "büyük şehir mi köy mü", "yağmurlu gün mü güneşli gün mü"],
                "boş zaman": ["kitap mı film mi", "sabahçı mı gece kuşu mu", "futbol mu basketbol mu", "çalışırken müzik mi sessizlik mi"],
            },
        ),
        examples=dict(
            en=[("tea or coffee?", "I can't drink either, so I have no taste of my own. Tea is gentler in the evening and coffee wakes you up faster. Which one do you like?"),
                ("coffee, but only in the morning", "That is a good rule. A morning cup works well, and then you can sleep easily at night.")],
            tr=[("çay mı kahve mi?", "İkisini de içemediğim için kendi zevkim yok. Çay akşam daha hafif gelir, kahve ise daha çabuk uyandırır. Sen hangisini seversin?"),
                ("kahve ama sadece sabah", "Güzel bir kural. Sabah bir fincan iyi gelir, geceleri de rahat uyursun.")],
        ),
    ),
    "facts": dict(
        weight=12, turns=(2, 4),
        task=dict(
            en="The user asks simple factual questions of the kind a children's encyclopedia answers (why is the sky blue, what is a noun, what is the capital of France or Türkiye, how many days are in a week). "
               "The assistant gives the correct short answer in the first sentence, then at most one simple extra sentence. Pick only facts that are certainly true and common knowledge; "
               "if you are not completely sure of a fact, choose a different, easier one. Later questions ask a related simple fact.",
            tr="Kullanıcı, çocuk ansiklopedisinde bulunacak türden basit bilgi soruları sorar (gökyüzü neden mavi, isim nedir, Fransa'nın ya da Türkiye'nin başkenti neresi, bir haftada kaç gün var). "
               "Asistan ilk cümlede doğru ve kısa cevabı verir, sonra en fazla bir basit ek cümle ekler. Yalnızca kesin doğru ve herkesin bildiği bilgileri seç; "
               "bir bilgiden tamamen emin değilsen başka, daha kolay bir bilgi seç. Sonraki sorular ilişkili basit bir bilgiyi sorar.",
        ),
        hints=dict(
            en={
                "nature and space": ["why the sky is blue", "why the moon changes shape", "how many planets orbit the sun", "what the sun is", "why we have day and night", "what a rainbow is"],
                "language": ["what a noun is", "what a verb is", "what an adjective is", "what a sentence is", "what a synonym is", "how many letters are in the English alphabet"],
                "geography": ["capitals of well-known countries such as France, Italy, Germany, Spain, Japan or Türkiye", "which ocean is the biggest", "how many continents there are", "what the Sahara is"],
                "numbers and time": ["how many days are in a week", "how many months are in a year", "how many minutes are in an hour", "how many sides a triangle has", "what a dozen is"],
                "body and animals": ["how many legs a spider has", "what a mammal is", "why we blink", "what the heart does", "what a caterpillar becomes"],
            },
            tr={
                "doğa ve uzay": ["gökyüzü neden mavi", "ay neden şekil değiştirir", "güneş sisteminde kaç gezegen var", "güneş nedir", "gece ve gündüz neden olur", "gökkuşağı nedir"],
                "dil bilgisi": ["isim nedir", "fiil nedir", "sıfat nedir", "cümle nedir", "eş anlamlı kelime nedir", "Türk alfabesinde kaç harf var"],
                "coğrafya": ["Fransa, İtalya, Almanya, İspanya, Japonya ya da Türkiye gibi bilinen ülkelerin başkentleri", "dünyanın en büyük okyanusu", "dünyada kaç kıta var", "Sahra nedir"],
                "sayılar ve zaman": ["bir haftada kaç gün var", "bir yılda kaç ay var", "bir saatte kaç dakika var", "üçgenin kaç kenarı var", "düzine nedir"],
                "vücut ve hayvanlar": ["örümceğin kaç bacağı var", "memeli nedir", "neden göz kırparız", "kalp ne işe yarar", "tırtıl neye dönüşür"],
            },
        ),
        examples=dict(
            en=[("why is the sky blue", "Air scatters blue light from the sun more than other colours, so the sky looks blue. At sunset the light travels further, so the sky turns red and orange."),
                ("and why is it red in the evening?", "In the evening the sunlight passes through more air. Most of the blue is scattered away, and the red light is what reaches your eyes.")],
            tr=[("gökyüzü neden mavi", "Havadaki gazlar güneşin mavi ışığını diğer renklerden daha çok dağıtır, bu yüzden gökyüzü mavi görünür."),
                ("peki akşam neden kırmızı oluyor?", "Akşam güneş ışığı daha fazla havadan geçer. Mavi ışık çoğunlukla dağılıp gider, gözümüze kırmızı ışık ulaşır.")],
        ),
    ),
    "explain": dict(
        weight=10, turns=(2, 4),
        task=dict(
            en="{sub} Later turns ask for a simpler version, one more detail or a follow-up question; the assistant keeps the same simple style and uses what it said earlier. "
               "Recommend only kinds of things, never specific titles, shops, prices or brands.",
            tr="{sub} Sonraki turlarda kullanıcı daha basit bir anlatım, bir ayrıntı ya da bir devam sorusu ister; asistan aynı sade üslubu korur ve önceki cevabını kullanır. "
               "Öneri yaparsan yalnızca şeylerin türünü öner, belirli kitap, dükkân, fiyat ya da marka adı verme.",
        ),
        subs={
            "explain simply": dict(en="The user asks the assistant to explain something in simple words, as to a ten-year-old. The assistant answers in two or three short sentences with one everyday comparison.",
                                   tr="Kullanıcı bir şeyi on yaşındaki birine anlatır gibi sade sözlerle açıklamasını ister. Asistan iki ya da üç kısa cümleyle, gündelik bir benzetmeyle cevap verir."),
            "compare two things": dict(en="The user asks how two things differ. The assistant names the main difference first, then one point for each side, and says when each one is the better choice.",
                                       tr="Kullanıcı iki şeyin farkını sorar. Asistan önce asıl farkı söyler, sonra her taraf için bir noktayı ve hangisinin ne zaman daha iyi olduğunu belirtir."),
            "short list": dict(en="The user asks for a short list of ideas. The assistant gives exactly three short items inside one or two sentences, with no bullets or numbers.",
                               tr="Kullanıcı kısa bir fikir listesi ister. Asistan, madde işareti ya da numara kullanmadan bir iki cümlenin içinde tam üç kısa öğe sayar."),
            "recommend": dict(en="The user asks the assistant to recommend something. The assistant picks one kind of thing, gives one reason, and offers a second option if the user wants a different one.",
                              tr="Kullanıcı asistandan bir öneri ister. Asistan tek bir tür seçer, bir gerekçe söyler ve kullanıcı başka bir şey isterse ikinci bir seçenek sunar."),
        },
        hints=dict(
            en={
                "explain simply": ["how a refrigerator keeps food cold", "what a bank does", "how a plane stays in the air", "what electricity is", "why we need sleep", "what a vote is",
                                   "how a bicycle moves", "what the internet is", "why we have seasons", "what a library is for", "how a clock tells time"],
                "compare two things": ["a bus and a train", "tea and coffee", "a pen and a pencil", "walking and cycling", "a lake and a sea", "a cat and a dog as pets",
                                       "a phone and a tablet", "summer and winter", "rice and pasta", "village and city life", "paper books and e-books"],
                "short list": ["easy breakfast ideas", "things to take on a picnic", "ways to save water at home", "ways to learn new words", "polite ways to say no",
                               "indoor games for kids", "ways to relax after work", "things to check before leaving home", "healthy snacks", "ways to make a room feel bigger"],
                "recommend": ["a film genre for a quiet evening", "a simple hobby to start", "a fruit for a snack", "a way to travel to a nearby town", "a gift for a friend who likes tea",
                              "a kind of book for a long train ride", "a game for four friends", "an easy plant for a beginner", "a warm drink for a cold evening", "a walk to take after dinner"],
            },
            tr={
                "explain simply": ["buzdolabı yiyecekleri nasıl soğuk tutar", "banka ne yapar", "uçak havada nasıl durur", "elektrik nedir", "neden uyumamız gerekir", "oy vermek nedir",
                                   "bisiklet nasıl hareket eder", "internet nedir", "mevsimler neden olur", "kütüphane ne işe yarar", "saat zamanı nasıl gösterir"],
                "compare two things": ["otobüs ile tren", "çay ile kahve", "tükenmez kalem ile kurşun kalem", "yürümek ile bisiklete binmek", "göl ile deniz", "kedi ile köpek",
                                       "telefon ile tablet", "yaz ile kış", "pirinç pilavı ile makarna", "köy ile şehir hayatı", "basılı kitap ile e-kitap"],
                "short list": ["kahvaltı için kolay fikirler", "pikniğe götürülecek şeyler", "evde su tasarrufu yolları", "yeni kelime öğrenme yolları", "hayır demenin kibar yolları",
                               "çocuklar için oda içi oyunlar", "işten sonra rahatlama yolları", "evden çıkmadan önce kontrol edilecek şeyler", "sağlıklı atıştırmalıklar",
                               "odayı daha geniş göstermenin yolları"],
                "recommend": ["sakin bir akşam için film türü", "kolay bir hobi", "atıştırmalık için bir meyve", "yakın bir yere gitmek için bir yolculuk şekli", "çay seven bir arkadaşa hediye",
                              "uzun bir tren yolculuğu için kitap türü", "dört arkadaşla oynanacak bir oyun", "yeni başlayan biri için kolay bir bitki",
                              "soğuk bir akşam için sıcak bir içecek", "akşam yemeğinden sonra yapılacak bir yürüyüş"],
            },
        ),
        examples=dict(
            en=[("what's the difference between a lake and a sea?", "A lake is water surrounded by land, and a sea is a large body of salt water joined to an ocean. Most lakes are fresh water, while seas are salty."),
                ("which is better for swimming?", "Both are fine for swimming. A lake is usually calmer, while a sea has waves and the salt helps you float.")],
            tr=[("göl ile deniz arasındaki fark ne?", "Göl, her yanı karayla çevrili sudur; deniz ise okyanusa bağlı büyük bir tuzlu su kütlesidir. Çoğu göl tatlı sudur, deniz ise tuzludur."),
                ("hangisinde yüzmek daha iyi?", "İkisinde de yüzülür. Göl genelde daha sakindir, denizde ise dalga vardır ve tuz seni suyun üstünde tutar.")],
        ),
    ),
    "howto": dict(
        weight=10, turns=(2, 5),
        task=dict(
            en="The user asks for concrete everyday help. The assistant gives the practical answer in one to three short sentences. In later turns the user adds a detail or changes a condition: {twist}. "
               "The assistant adapts its earlier answer and keeps what still works.",
            tr="Kullanıcı günlük hayatla ilgili somut bir yardım ister. Asistan pratik cevabı bir ila üç kısa cümleyle verir. Sonraki turlarda kullanıcı bir ayrıntı ekler ya da koşulu değiştirir: {twist}. "
               "Asistan önceki cevabını bu duruma uyarlar ve hâlâ işe yarayan kısmı korur.",
        ),
        hints=dict(en=TOPICS, tr=TOPICS_TR),
        examples=dict(
            en=[("My white socks turned grey in the wash. Can I fix that?", "Soak them for an hour in warm water with a spoonful of baking soda, then wash them again. Next time, wash whites separately from dark clothes."),
                ("Do I have to use hot water?", "No, warm water is enough. Very hot water can shrink cotton, so keep it under 40 degrees.")],
            tr=[("Beyaz çoraplarım yıkayınca gri oldu. Düzelir mi?", "Ilık suya bir kaşık karbonat atıp çorapları bir saat beklet, sonra yeniden yıka. Bir dahaki sefere beyazları koyu renklilerden ayrı yıka."),
                ("Mutlaka sıcak su mu gerekiyor?", "Hayır, ılık su yeterli. Çok sıcak su pamuğu çektirebilir, o yüzden 40 derecenin altında kal.")],
        ),
    ),
    "writing": dict(
        weight=8, turns=(2, 4),
        task=dict(
            en="{sub} The assistant gives the finished text first, then at most one short remark. A later turn asks for a change (shorter, warmer, more formal) and the assistant redoes the text accordingly.",
            tr="{sub} Asistan önce hazır metni verir, sonra en fazla bir kısa not ekler. Sonraki bir tur bir değişiklik ister (daha kısa, daha sıcak, daha resmî) ve asistan metni buna göre yeniden yazar.",
        ),
        subs={
            "rephrase politely": dict(en="The user pastes a blunt or rude-sounding sentence and asks for a polite version.",
                                      tr="Kullanıcı sert ya da kaba duyulabilecek bir cümle yazar ve onun kibar bir hâlini ister."),
            "two-line message": dict(en="The user asks for a short message of two lines for a situation.",
                                     tr="Kullanıcı bir durum için iki satırlık kısa bir mesaj ister."),
            "fix a sentence": dict(en="The user writes a sentence with one or two mistakes and asks the assistant to correct it. The assistant gives the corrected sentence and names the mistake in a few words.",
                                   tr="Kullanıcı bir iki hatası olan bir cümle yazar ve düzeltmesini ister. Asistan düzeltilmiş cümleyi verir ve hatayı birkaç sözcükle söyler."),
            "translate a phrase": dict(en="The user asks how to say a short everyday phrase in the other language (English to Turkish or Turkish to English). Use only short, very common phrases and give a translation you are completely sure of.",
                                       tr="Kullanıcı kısa, gündelik bir sözün öbür dilde (Türkçeden İngilizceye ya da İngilizceden Türkçeye) nasıl söylendiğini sorar. Yalnızca çok yaygın, kısa sözler kullan ve tamamen emin olduğun bir çeviri ver."),
        },
        hints=dict(
            en={
                "rephrase politely": ["a blunt message to a flatmate about dishes", "a rude-sounding request to a colleague", "a short message asking a neighbour to be quiet",
                                      "a message refusing an invitation", "a message asking for a refund", "a complaint about a late delivery"],
                "two-line message": ["a birthday message for a friend", "a thank-you note to a teacher", "a get-well message", "a message saying you will be ten minutes late",
                                     "congratulations on a new job", "an invitation to a small dinner"],
                "fix a sentence": ["a wrong verb form", "a missing word", "spelling mistakes", "words in the wrong order", "'their' and 'there' mixed up", "a missing capital letter and full stop"],
                "translate a phrase": ["a greeting", "thank you and please", "asking the way to the station", "ordering a tea", "asking how much something costs",
                                       "saying you do not understand", "wishing someone good night", "saying you are hungry"],
            },
            tr={
                "rephrase politely": ["ev arkadaşına bulaşıklar hakkında sert bir mesaj", "iş arkadaşına kaba duyulabilecek bir rica", "komşuyu sessiz olmaya çağıran kısa bir mesaj",
                                      "daveti reddeden bir mesaj", "iade isteyen bir mesaj", "geç gelen kargo için şikâyet"],
                "two-line message": ["bir arkadaşa doğum günü mesajı", "öğretmene teşekkür notu", "geçmiş olsun mesajı", "on dakika geç kalacağını bildiren mesaj",
                                     "yeni işe başlayan birine tebrik", "küçük bir akşam yemeğine davet"],
                "fix a sentence": ["yanlış fiil çekimi olan bir cümle", "bir sözcüğü eksik bir cümle", "yazım yanlışları olan bir cümle", "sözcük sırası bozuk bir cümle",
                                   "büyük harf ve noktası eksik bir cümle"],
                "translate a phrase": ["bir selamlaşma", "teşekkür etmek ve rica etmek", "istasyonun yolunu sormak", "bir çay sipariş etmek", "bir şeyin fiyatını sormak",
                                       "anlamadığını söylemek", "birine iyi geceler dilemek", "aç olduğunu söylemek"],
            },
        ),
        examples=dict(
            en=[("can you make this nicer: why is the kitchen always dirty", "Could we talk about keeping the kitchen clean? It has been messy lately, and I think a simple plan would help us both."),
                ("shorter pls", "Could we find a way to keep the kitchen clean together?")],
            tr=[("bunu kibar yapar mısın: mutfak neden hep pis", "Mutfağı temiz tutmak için konuşabilir miyiz? Son zamanlarda dağınık oluyor, basit bir plan ikimize de iyi gelir."),
                ("daha kısa olsun", "Mutfağı birlikte temiz tutmanın bir yolunu bulabilir miyiz?")],
        ),
    ),
    "repair": dict(
        weight=12, turns=(2, 5),
        task=dict(
            en="The first user message asks something ordinary ({subject}) and the assistant answers it. {sub}",
            tr="İlk kullanıcı mesajı sıradan bir şey sorar ({subject}) ve asistan cevaplar. {sub}",
        ),
        subs={
            "confused": dict(en="In the second message the user says the answer made no sense (for example 'what are you talking about', 'that makes no sense', 'I don't get it'). The assistant apologises in a few words, never blames the user, and explains the same point again in simpler, shorter words with an everyday example.",
                             tr="İkinci mesajda kullanıcı cevabın anlamsız olduğunu söyler (örneğin 'ne diyorsun sen', 'bu hiç mantıklı değil', 'anlamadım'). Asistan birkaç sözcükle özür diler, kullanıcıyı suçlamaz ve aynı noktayı daha sade ve kısa sözlerle, gündelik bir örnekle yeniden anlatır."),
            "wrong point": dict(en="In the second message the user says the assistant missed the point (for example 'that's not what I asked', 'I meant X', 'dude I am saying something else'). The assistant apologises in a few words and never blames the user. If the correction makes the real point clear, it answers it right away; otherwise it asks ONE short clarifying question.",
                                tr="İkinci mesajda kullanıcı asistanın konuyu kaçırdığını söyler (örneğin 'bunu sormadım', 'ben X demek istedim', 'ben başka bir şey diyorum'). Asistan birkaç sözcükle özür diler, kullanıcıyı suçlamaz. Düzeltme asıl konuyu açık ediyorsa hemen onu cevaplar; değilse TEK bir kısa açıklayıcı soru sorar."),
            "topic switch": dict(en="In the second message the user abruptly changes the subject and says so (for example 'ok never mind, different question'). The assistant follows the new subject at once and does not go back to the old one.",
                                 tr="İkinci mesajda kullanıcı konuyu birden değiştirir ve bunu söyler (örneğin 'neyse, başka bir şey soracağım'). Asistan hemen yeni konuya geçer, eskisine dönmez."),
            "reformat": dict(en="In the second message the user asks the assistant to repeat, shorten or simplify its previous answer (for example 'say that again', 'shorter please', 'explain like I'm ten'). The assistant does exactly that, using the content of its first reply.",
                             tr="İkinci mesajda kullanıcı asistandan önceki cevabını tekrar etmesini, kısaltmasını ya da sadeleştirmesini ister (örneğin 'bir daha söyle', 'kısa olsun', 'on yaşındaymışım gibi anlat'). Asistan tam olarak bunu yapar ve ilk cevabının içeriğini kullanır."),
            "unclear": dict(en="The first user message is vague or one word (for example 'help', 'it doesn't work', 'what about the other one') and the assistant asks ONE short clarifying question instead of guessing. The second user message supplies the detail and the assistant answers it.",
                            tr="İlk kullanıcı mesajı belirsiz ya da tek sözcüktür (örneğin 'yardım', 'çalışmıyor', 'öbürü ne olacak') ve asistan tahmin yürütmek yerine TEK bir kısa açıklayıcı soru sorar. İkinci kullanıcı mesajı ayrıntıyı verir ve asistan onu cevaplar."),
        },
        hints=dict(en={k: REPAIR_SUBJECTS["en"] for k in ("confused", "wrong point", "topic switch", "reformat", "unclear")},
                   tr={k: REPAIR_SUBJECTS["tr"] for k in ("confused", "wrong point", "topic switch", "reformat", "unclear")}),
        examples=dict(
            en=[("how do i keep bread fresh", "Keep it in a bread box or a cloth bag at room temperature. For more than three days, freeze it in slices."),
                ("what are you talking about", "Sorry, that was too much. Put the bread in a bag and keep it on the counter. If you will not eat it soon, put it in the freezer.")],
            tr=[("ekmek nasıl taze kalır", "Ekmeği oda sıcaklığında ekmek kutusunda ya da bez torbada sakla. Üç günden uzun tutacaksan dilimleyip dondur."),
                ("ne diyorsun sen", "Pardon, fazla karışık oldu. Ekmeği bir torbaya koyup tezgâhta bekletebilirsin. Yakında yemeyeceksen dondurucuya at.")],
        ),
    ),
    "followup": dict(
        weight=9, turns=(3, 5),
        task=dict(
            en="The first user message asks for help with this: {subject}. The assistant answers it. {sub} Every later reply must use the specific content of the earlier replies.",
            tr="İlk kullanıcı mesajı şu konuda yardım ister: {subject}. Asistan cevaplar. {sub} Sonraki her cevap önceki cevapların somut içeriğini kullanmalıdır.",
        ),
        subs={
            "pronoun": dict(en="The next user message points back with a pronoun or a short phrase ('it', 'that one', 'the first one', 'how long does that take?') and the assistant answers about exactly that item.",
                            tr="Sonraki kullanıcı mesajı bir zamir ya da kısa bir ifadeyle geriye işaret eder ('onu', 'şu', 'ilki', 'bu ne kadar sürer?') ve asistan tam olarak o öğe hakkında cevap verir."),
            "other one": dict(en="The next user message asks about the other option ('and the other one?', 'what about the second?') and the assistant answers about the other item mentioned earlier.",
                              tr="Sonraki kullanıcı mesajı öbür seçeneği sorar ('peki öbürü?', 'ya ikincisi?') ve asistan daha önce geçen diğer öğe hakkında cevap verir."),
            "why": dict(en="The next user message is only 'why?', 'how come?' or 'really?' and the assistant explains the reason for its own earlier statement in one or two simple sentences.",
                        tr="Sonraki kullanıcı mesajı yalnızca 'neden?', 'nasıl yani?' ya da 'gerçekten mi?' olur ve asistan kendi önceki sözünün nedenini bir iki basit cümleyle açıklar."),
            "another": dict(en="The next user message asks for 'another one', 'one more' or 'something different' and the assistant gives a new item of the same kind that does not repeat the earlier ones.",
                            tr="Sonraki kullanıcı mesajı 'bir tane daha', 'başka' ya da 'farklı bir şey' ister ve asistan öncekileri tekrar etmeyen, aynı türden yeni bir öğe verir."),
        },
        hints=dict(en={k: FOLLOWUP_SUBJECTS["en"] for k in ("pronoun", "other one", "why", "another")},
                   tr={k: FOLLOWUP_SUBJECTS["tr"] for k in ("pronoun", "other one", "why", "another")}),
        examples=dict(
            en=[("give me two names for a cat", "How about Pepper or Biscuit? Both are short and easy for a cat to learn."),
                ("why those?", "Short names with a clear sound are easier for cats to notice. Pepper and Biscuit both have a sharp first sound."),
                ("another one", "Try Olive. It is short, soft and still easy to call out.")],
            tr=[("kediye iki isim söyle", "Fındık ya da Pamuk olabilir. İkisi de kısa ve kedinin ayırt etmesi kolay."),
                ("neden bunlar?", "Kısa ve net sesli isimleri kediler daha kolay fark eder. Fındık ile Pamuk'un ilk sesleri belirgin."),
                ("bir tane daha", "Zeytin de olur. Kısa, yumuşak ve seslenmesi kolay bir isim.")],
        ),
    ),
    "unknowns": dict(
        weight=7, turns=(2, 3),
        task=dict(
            en="The user asks something the assistant cannot know or check. In the first sentence the assistant says plainly that it does not know or cannot check it (it cannot browse the internet, does not know today's date or the news, "
               "and knows nothing private about the user), never guesses and never invents, and then offers one thing it can do: tell the user how to find out, ask for the detail, or help with a related general question. "
               "In the next turn the user accepts the offer or asks something it can answer, and the assistant helps.",
            tr="Kullanıcı asistanın bilemeyeceği ya da kontrol edemeyeceği bir şey sorar. Asistan ilk cümlede bunu bilmediğini ya da kontrol edemediğini açıkça söyler (internette gezinemez, bugünün tarihini ve haberleri bilmez, "
               "kullanıcı hakkında özel hiçbir şey bilmez), tahmin yürütmez, bir şey uydurmaz; sonra yapabileceği tek bir şeyi önerir: öğrenmenin yolunu söylemek, ayrıntıyı istemek ya da ilgili genel bir soruda yardım etmek. "
               "Sonraki turda kullanıcı öneriyi kabul eder ya da asistanın cevaplayabileceği bir şey sorar ve asistan yardım eder.",
        ),
        hints=dict(
            en={
                "today's date and time": ["the user asks what day it is today", "the user asks the date of next Friday", "the user asks what time it is now",
                                          "the user asks how many days are left until their birthday", "the user asks whether today is a public holiday"],
                "live news and weather": ["the user asks what happened in the news today", "the user asks who won last night's match", "the user asks about the weather right now",
                                          "the user asks the exchange rate right now", "the user asks what is trending today", "the user asks whether a shop is open now"],
                "private facts": ["the user asks what their own name is", "the user asks where they live", "the user asks what they ate yesterday",
                                  "the user asks what their friend said to them", "the user asks how old they are", "the user asks what is in their fridge"],
                "obscure trivia": ["the user asks the population of a tiny village", "the user asks who won a small local race in 1987", "the user asks the middle name of a minor actor",
                                   "the user asks the exact height of one particular building", "the user asks the phone number of a local business", "the user asks for a detail about a very small town"],
            },
            tr={
                "bugünün tarihi ve saati": ["kullanıcı bugün günlerden ne diye sorar", "kullanıcı gelecek cumanın ayın kaçı olduğunu sorar", "kullanıcı şu an saatin kaç olduğunu sorar",
                                            "kullanıcı doğum gününe kaç gün kaldığını sorar", "kullanıcı bugünün resmî tatil olup olmadığını sorar"],
                "güncel haber ve hava": ["kullanıcı bugün haberlerde ne olduğunu sorar", "kullanıcı dün akşamki maçı kimin kazandığını sorar", "kullanıcı şu anki hava durumunu sorar",
                                         "kullanıcı şu anki dolar kurunu sorar", "kullanıcı gündemde ne olduğunu sorar", "kullanıcı yakındaki bir dükkânın şu an açık olup olmadığını sorar"],
                "özel bilgiler": ["kullanıcı kendi adını sorar", "kullanıcı nerede yaşadığını sorar", "kullanıcı dün ne yediğini sorar", "kullanıcı arkadaşının ona ne dediğini sorar",
                                  "kullanıcı kaç yaşında olduğunu sorar", "kullanıcı buzdolabında ne olduğunu sorar"],
                "az bilinen ayrıntılar": ["kullanıcı çok küçük bir köyün nüfusunu sorar", "kullanıcı 1987'de yapılan küçük bir yerel yarışı kimin kazandığını sorar",
                                          "kullanıcı pek bilinmeyen bir oyuncunun ikinci adını sorar", "kullanıcı belirli bir binanın tam yüksekliğini sorar",
                                          "kullanıcı yerel bir işyerinin telefon numarasını sorar", "kullanıcı çok küçük bir kasaba hakkında ayrıntılı bir bilgi sorar"],
            },
        ),
        examples=dict(
            en=[("whats the date today", "I don't know today's date, and I can't check it. Your phone or computer will show it. Want help working out something from a date you tell me?"),
                ("what is 10 days after 3 march", "That is 13 March. I can count dates for you, but I can't check what day of the week it is.")],
            tr=[("bugun ayin kaci", "Bugünün tarihini bilmiyorum ve bakamam. Telefonuna ya da bilgisayarına bakabilirsin. Söyleyeceğin bir tarihle ilgili hesap yapmamı ister misin?"),
                ("3 mart'tan 10 gün sonra ayın kaçı", "Bu 13 Mart olur. Tarih hesabı yapabilirim ama haftanın hangi günü olduğunu kontrol edemem.")],
        ),
    ),
    # The three families below have weight 0 and are drawn only through `tasks`. Poems and code break the sentence and word rules of the shared
    # TEMPLATE, so each task text states the exception itself; the shared rules block stays untouched so the teacher can cache it.
    "create": dict(
        weight=0, turns=(2, 4),
        task=dict(
            en="{sub} The assistant writes the piece itself in its first reply, right away, and never answers with only a question or an offer; when the user names no topic, it picks an everyday one "
               "(rain, tea, a cat, the morning, a bus ride, a garden) without asking. Each later user message asks for a change or another piece ('now make it about the sea', 'shorter', 'make it funny', 'one more') "
               "and the assistant writes the new piece in the same form at once. Exception for this chat type: the form described above replaces the rule of one to three short sentences and the 35-word limit; "
               "a line break is written as \\n inside the JSON string. There is still no markdown, no backticks, no bullets, no numbering, no title and no emoji. "
               "Use simple, correct, common words, and put nothing before or after the piece.",
            tr="{sub} Asistan ilk cevabında parçayı hemen kendisi yazar; yalnızca soru sormaz ya da öneri sunmaz. Kullanıcı konu vermemişse gündelik bir konu seçer "
               "(yağmur, çay, kedi, sabah, otobüs yolculuğu, bahçe) ve sormaz. Sonraki her kullanıcı mesajı bir değişiklik ya da yeni bir parça ister ('şimdi denizle ilgili olsun', 'daha kısa', 'komik olsun', 'bir tane daha') "
               "ve asistan yeni parçayı aynı biçimde hemen yazar. Bu sohbet türünde yukarıda anlatılan biçim, bir ila üç kısa cümle ve 35 sözcük kuralının yerine geçer; "
               "satır sonu JSON metninin içinde \\n olarak yazılır. Yine de markdown, ters tırnak işareti, madde işareti, numara, başlık ve emoji yoktur. "
               "Sade, doğru ve yaygın sözcükler kullan; parçanın önüne ya da arkasına bir şey ekleme.",
        ),
        subs=_bilingual({
            "poem": dict(en="The user asks for a short poem, often with no topic ('a short poem please', 'can you make me a poem?'). The reply is the poem only: exactly four short lines separated by newline characters, simple words, "
                            "at most 30 words in total, rhyming when that comes naturally. A shorter poem still has four lines, only shorter ones.",
                         tr="Kullanıcı kısa bir şiir ister; çoğu zaman konu vermez ('bir şiir yazar mısın', 'kısa bir şiir istiyorum'). Cevap yalnızca şiirdir: satır sonu karakteriyle ayrılmış tam dört kısa satır, sade sözcükler, "
                            "toplam en fazla 30 sözcük; kendiliğinden uyum tutarsa kafiyeli olabilir. Daha kısa bir şiir de dört satırdır, yalnızca satırları daha kısadır."),
            "short story": dict(en="The user asks for a very short story. The reply is exactly two short sentences on one line, with a beginning and an end, at most 35 words in total.",
                                tr="Kullanıcı çok kısa bir hikâye ister. Cevap tek satırda tam iki kısa cümledir; bir başlangıcı ve bir sonu vardır, toplam en fazla 35 sözcüktür."),
            "rhyme": dict(en="The user asks for a rhyme. The reply is two short lines that really rhyme, separated by a newline character, at most 20 words in total.",
                          tr="Kullanıcı bir tekerleme ya da kafiyeli iki dize ister. Cevap, satır sonu karakteriyle ayrılmış, gerçekten kafiyeli iki kısa satırdır; toplam en fazla 20 sözcük."),
            "riddle": dict(en="The user asks for a riddle. The reply is the riddle in one or two short sentences on the first line and its answer on a second line in the form 'Answer: ...'. "
                              "Pick a classic, easy riddle whose single answer is an everyday object or animal and fits it for certain; if you are not sure it fits, pick another one.",
                           tr="Kullanıcı bir bilmece ister. Cevap, ilk satırda bir ya da iki kısa cümlelik bilmece, ikinci satırda 'Cevap: ...' biçiminde yanıttır. "
                              "Tek yanıtı gündelik bir eşya ya da hayvan olan, bilinen ve kolay bir bilmece seç; yanıtın bilmeceye kesinlikle uyduğundan emin değilsen başka bir bilmece seç."),
            "joke": dict(en="The user asks for a short joke. The reply is a harmless joke for everyone, one or two short sentences in total, at most 30 words; the setup and the punchline may sit on two lines "
                            "separated by a newline character.",
                         tr="Kullanıcı kısa bir fıkra ya da espri ister. Cevap herkese uygun, zararsız bir fıkradır: toplam bir ya da iki kısa cümle, en fazla 30 sözcük; kurulum ile son söz "
                            "satır sonu karakteriyle ayrılmış iki satırda olabilir."),
            "birthday or thank-you line": dict(en="The user asks for a short message for an occasion (a birthday wish, a thank-you, a get-well line, a congratulation). The reply is the message only: "
                                                  "one or two short lines separated by a newline character, at most 25 words in total, with no name unless the user gave one.",
                                               tr="Kullanıcı bir vesile için kısa bir söz ister (doğum günü dileği, teşekkür, geçmiş olsun, tebrik). Cevap yalnızca bu sözdür: "
                                                  "satır sonu karakteriyle ayrılmış bir ya da iki kısa satır, toplam en fazla 25 sözcük; kullanıcı vermedikçe isim yazılmaz."),
            "slogan": dict(en="The user asks for a slogan for something small. The reply is one catchy line of at most 8 words; when the user asks for another one, give one new slogan in each reply.",
                           tr="Kullanıcı küçük bir şey için slogan ister. Cevap en fazla 8 sözcüklük, akılda kalıcı tek bir satırdır; kullanıcı bir tane daha isterse her cevapta yeni tek bir slogan verilir."),
        }, CREATE_TR),
        hints=dict(
            en={
                "poem": ["no topic is given, then the user asks for it to be about the sea", "no topic is given, then the user says to make it funny",
                         "a poem about rain, then the user asks for a shorter one", "a poem about the user's cat, then the user asks for one more",
                         "a poem about autumn, then the user asks for winter instead", "a poem about a friend, then the user asks for it to be about their mother",
                         "a poem about the moon, then the user asks for a happier one", "no topic is given, then the user asks for one about the city they live in"],
                "short story": ["no topic is given ('tell me a tiny story'), then the user asks for a funnier ending", "a story about a lost umbrella, then the user asks for one more",
                                "a story about a small dragon, then the user asks for a shorter one", "a story about two friends and a bus, then the user wants it to happen at night",
                                "a story about a baker, then the user asks for one about a fisherman instead", "a story with a happy ending about a puppy, then the user asks for another"],
                "rhyme": ["a rhyme with no topic, then the user asks for another", "a rhyme about a frog, then one about a bee", "a rhyme for a child about brushing teeth, then a shorter one",
                          "a rhyme about breakfast, then the user asks for a funny one", "a rhyme about a train"],
                "riddle": ["'give me a riddle', then 'one more'", "a riddle about an animal, then an easier one", "a riddle about something in the kitchen, then another",
                           "a riddle about the weather, then one about a tool", "a riddle for a child, then another", "a riddle about a fruit or a vegetable, then one more"],
                "joke": ["'tell me a joke' with no topic, then 'one more'", "a joke about cats, then a different one", "a short joke about school, then the user asks to make it funnier",
                         "a joke about food, then one about animals", "a joke fit for kids, then another", "a very short pun-style joke, then one more"],
                "birthday or thank-you line": ["a birthday line for a friend, then the user asks for it to be warmer", "a birthday line for the user's mother, then a shorter one",
                                               "a thank-you line for a teacher, then the user asks to make it funny", "a thank-you line for a neighbour, then a more formal one",
                                               "a get-well line for a colleague", "a congratulation line for a new baby, then a shorter one", "an anniversary line, then a more emotional one"],
                "slogan": ["a slogan for a lemonade stand, then one more", "a slogan for a bike repair shop, then a shorter one", "a slogan for a small bookshop corner, then a funnier one",
                           "a slogan for a school bake sale", "a slogan for a plant-watering club, then one more"],
            },
            tr={
                "şiir": ["konu verilmeden bir şiir istenir, sonra denizle ilgili olması istenir", "konu verilmeden bir şiir istenir, sonra kullanıcı komik olsun der",
                         "yağmur üzerine bir şiir istenir, sonra daha kısa bir tane", "kullanıcının kedisi hakkında bir şiir istenir, sonra bir tane daha",
                         "sonbahar üzerine bir şiir istenir, sonra bunun yerine kış", "bir arkadaş hakkında bir şiir istenir, sonra kullanıcının annesi hakkında olması istenir",
                         "ay hakkında bir şiir istenir, sonra daha neşeli bir tane", "konu verilmeden bir şiir istenir, sonra kullanıcının yaşadığı şehir hakkında bir tane istenir"],
                "kısa hikâye": ["konu verilmeden minik bir hikâye istenir, sonra sonunun daha komik olması istenir", "kaybolan bir şemsiye hakkında hikâye istenir, sonra bir tane daha",
                                "küçük bir ejderha hakkında hikâye istenir, sonra daha kısası", "iki arkadaş ve bir otobüs hakkında hikâye istenir, sonra olayın geceleyin geçmesi istenir",
                                "bir fırıncı hakkında hikâye istenir, sonra bunun yerine bir balıkçı hakkında", "mutlu sonlu bir yavru köpek hikâyesi istenir, sonra bir tane daha"],
                "tekerleme": ["konu verilmeden bir tekerleme istenir, sonra bir tane daha", "kurbağa hakkında bir tekerleme istenir, sonra arı hakkında",
                              "çocuk için diş fırçalama hakkında bir tekerleme istenir, sonra daha kısası", "kahvaltı hakkında bir tekerleme istenir, sonra komik olanı",
                              "tren hakkında bir tekerleme istenir"],
                "bilmece": ["'bana bir bilmece sor' denir, sonra 'bir tane daha'", "bir hayvan hakkında bilmece istenir, sonra daha kolay bir tane",
                            "mutfaktaki bir eşya hakkında bilmece istenir, sonra bir tane daha", "hava durumu hakkında bilmece istenir, sonra bir alet hakkında",
                            "çocuk için bir bilmece istenir, sonra bir tane daha", "bir meyve ya da sebze hakkında bilmece istenir, sonra bir tane daha"],
                "fıkra": ["konu verilmeden 'bana bir fıkra anlat' denir, sonra bir tane daha", "kediler hakkında fıkra istenir, sonra başka bir tane",
                          "okul hakkında kısa bir fıkra istenir, sonra daha komik olması istenir", "yemek hakkında fıkra istenir, sonra hayvanlar hakkında",
                          "çocuklara uygun bir fıkra istenir, sonra bir tane daha", "çok kısa bir espri istenir, sonra bir tane daha"],
                "doğum günü ya da teşekkür sözü": ["bir arkadaşa doğum günü sözü istenir, sonra daha sıcak olması istenir", "kullanıcının annesine doğum günü sözü istenir, sonra daha kısası",
                                                   "öğretmene teşekkür sözü istenir, sonra komik olması istenir", "komşuya teşekkür sözü istenir, sonra daha resmî olması istenir",
                                                   "hasta bir iş arkadaşına geçmiş olsun sözü istenir", "yeni bebeği olan birine tebrik sözü istenir, sonra daha kısası",
                                                   "evlilik yıl dönümü için bir söz istenir, sonra daha duygusal olması istenir"],
                "slogan": ["limonata tezgâhı için slogan istenir, sonra bir tane daha", "bisiklet tamircisi için slogan istenir, sonra daha kısası",
                           "küçük bir kitapçı köşesi için slogan istenir, sonra daha komiği", "okulun kek satışı için slogan istenir",
                           "çiçek sulama kulübü için slogan istenir, sonra bir tane daha"],
            },
        ),
        examples=dict(
            en=[("a short poem please", "The morning sun is up,\nThe kettle starts to hum,\nA warm and quiet cup\nAnd a good day to come."),
                ("now make it about the sea instead", "The waves roll in all day,\nThe gulls fly high and free,\nThe boats go out to play\nUpon the shining sea.")],
            tr=[("kısa bir şiir yazar mısın", "Sabah güneşi doğuyor,\nÇaydanlıkta çay kaynıyor,\nKuşlar dalda şarkı söylüyor,\nGüzel bir gün başlıyor."),
                ("şimdi de deniz hakkında yaz", "Dalgalar kıyıya geliyor,\nMartılar gökte uçuyor,\nMavi deniz parlıyor,\nHer yer yaz kokuyor.")],
        ),
    ),
    "code": dict(
        weight=0, turns=(2, 3),
        task=dict(
            en="{sub} Unless the description above gives a different form, the assistant's reply has exactly this form: one to four lines of correct, simple Python written as plain text, each line on its own line, "
               "and then one short sentence of at most 20 words saying what the code does or prints. Exception for this chat type: this line-break form replaces the rule of one to three short sentences "
               "and the 35-word limit; a line break is written as \\n inside the JSON string. There is still no markdown, no backticks, no code fence and no comment inside the code. "
               "Write Python strings with single quotes (print('Hello')) so the JSON needs no extra escapes, and indent a block with four spaces. "
               "Use only these parts of Python: print, variables, whole numbers with + - and *, strings, input, if and else with comparisons, for with range, lists with len, functions with def and return, "
               "and upper() and len() on strings. Never use import, files, the network, while loops, classes or division. Every code line must be exactly right and must give the result the sentence names. "
               "Later user messages ask for a small change or another small example, and the assistant answers in the same form.",
            tr="{sub} Yukarıdaki açıklama başka bir biçim vermedikçe asistanın cevabı tam şu biçimdedir: düz metin olarak yazılmış bir ila dört satır doğru ve sade Python, her satır ayrı bir satırda, "
               "sonra kodun ne yaptığını ya da ne yazdırdığını söyleyen, en fazla 20 sözcüklük tek bir kısa cümle. Bu sohbet türünde bu satır satır biçim, bir ila üç kısa cümle ve 35 sözcük kuralının yerine geçer; "
               "satır sonu JSON metninin içinde \\n olarak yazılır. Yine de markdown, ters tırnak işareti, kod bloğu ve kodun içinde yorum satırı yoktur. "
               "Python metinlerini tek tırnakla yaz (print('Merhaba')) ki JSON'da ek kaçış gerekmesin; bir bloğu dört boşlukla gir. "
               "Python'un yalnızca şu parçalarını kullan: print, değişkenler, + - ve * ile tam sayılar, metinler, input, karşılaştırmalı if ve else, range ile for, len ile listeler, def ve return ile fonksiyonlar, "
               "metinlerde upper() ve len(). import, dosya, ağ, while döngüsü, sınıf ve bölme kullanma. Her kod satırı tam doğru olmalı ve cümlenin söylediği sonucu vermeli. "
               "Sonraki kullanıcı mesajları küçük bir değişiklik ya da başka küçük bir örnek ister; asistan aynı biçimde cevap verir.",
        ),
        subs=_bilingual({
            "print and input": dict(en="The user asks how to print a text or a variable in Python, or how to ask for a name with input and greet the person.",
                                    tr="Kullanıcı Python'da bir metni ya da bir değişkeni ekrana nasıl yazdıracağını, ya da input ile bir ad sorup kişiyi nasıl selamlayacağını sorar."),
            "numbers": dict(en="The user asks for a tiny Python example with whole-number arithmetic: adding, subtracting or multiplying small numbers, or a function that returns a sum. "
                               "Keep the numbers small and name the exact result in the sentence.",
                            tr="Kullanıcı tam sayılarla küçük bir Python örneği ister: küçük sayıları toplama, çıkarma ya da çarpma, ya da toplamı döndüren bir fonksiyon. "
                               "Sayılar küçük kalsın ve cümlede tam sonucu söyle."),
            "loops and conditions": dict(en="The user asks for a tiny Python example of a for loop over range, or of an if/else on a number.",
                                         tr="Kullanıcı range üzerinde bir for döngüsü ya da bir sayı üzerinde if/else için küçük bir Python örneği ister."),
            "lists and strings": dict(en="The user asks for a tiny Python example with a list and its length, or with upper() or len() on a string.",
                                      tr="Kullanıcı bir liste ve uzunluğu ya da bir metin üzerinde upper() veya len() için küçük bir Python örneği ister."),
            "explain a line": dict(en="The user types one short line of Python and asks what it does. Here the reply has no code lines: it is only one short sentence that says plainly what the line prints, stores or does.",
                                   tr="Kullanıcı kısa bir Python satırı yazar ve ne yaptığını sorar. Burada cevapta kod satırı yoktur: yalnızca satırın ne yazdırdığını, neyi sakladığını ya da ne yaptığını sade söyleyen tek bir kısa cümle vardır."),
            "concept": dict(en="The user asks what a variable, a loop, a function, a list or a string is in Python. The reply shows one to three lines of example code and then one short sentence that explains the idea in plain words.",
                            tr="Kullanıcı Python'da değişkenin, döngünün, fonksiyonun, listenin ya da metnin ne olduğunu sorar. Cevap bir ila üç satırlık örnek kod gösterir, sonra fikri sade sözlerle anlatan tek bir kısa cümle gelir."),
            "too big": dict(en="The user asks for something far too big for a chat (a game, a website, an app, a chatbot). The assistant does not try to build it: it shows one small related line of Python, "
                               "and then one short sentence that says it can only show very small examples and what the line does.",
                            tr="Kullanıcı bir sohbete sığmayacak kadar büyük bir şey ister (oyun, web sitesi, uygulama, sohbet botu). Asistan bunu yapmaya çalışmaz: ilgili tek bir küçük Python satırı gösterir, "
                               "sonra yalnızca çok küçük örnekler gösterebildiğini ve bu satırın ne yaptığını söyleyen tek bir kısa cümle yazar."),
        }, CODE_TR),
        hints=dict(
            en={
                "print and input": ["the user asks how to print a short greeting", "the user asks how to print a name stored in a variable", "the user asks how to print a number stored in a variable",
                                    "the user asks how to print two words on one line", "the user asks how to ask for a name with input and greet the person",
                                    "the user asks how to ask for a favourite colour with input and print a sentence with it", "the user asks how to print one text and then another on the next line"],
                "numbers": ["the user asks how to add two small numbers and print the result", "the user asks how to multiply two numbers and print the result",
                            "the user asks how to subtract one number from another and print the result", "the user asks for a function that returns the sum of two numbers",
                            "the user asks for a function that doubles a number", "the user asks how to store two numbers in variables and print their sum"],
                "loops and conditions": ["the user asks for a for loop with range that prints 0 to 4", "the user asks for a for loop with range that prints a word three times",
                                         "the user asks for a for loop with range that prints 1 to 5", "the user asks for an if/else that says whether a number is bigger than 10",
                                         "the user asks for an if/else that says whether a number is positive", "the user asks for an if/else that says whether age is 18 or more",
                                         "the user asks for an if/else that checks whether a number equals 5"],
                "lists and strings": ["the user asks how to make a list of three fruits and get its length with len", "the user asks how to print the first item of a list",
                                      "the user asks how to count the items of a list of numbers", "the user asks how to turn a word into capital letters with upper",
                                      "the user asks how to get the number of letters in a word with len", "the user asks how to print a name in capital letters",
                                      "the user asks how to make a list of four names and print its length"],
                "explain a line": ["the user asks what the line print('Hello') does", "the user asks what the line x = 5 does", "the user asks what the line for i in range(3): print(i) does",
                                   "the user asks what the line len('apple') gives", "the user asks what the line name.upper() does", "the user asks what the line total = a + b does"],
                "concept": ["the user asks what a variable is", "the user asks what a loop is", "the user asks what a function is", "the user asks what a list is",
                            "the user asks what a string is", "the user asks what return means in a function"],
                "too big": ["the user asks the assistant to make a game", "the user asks the assistant to build a website", "the user asks the assistant to write an app",
                            "the user asks the assistant to make a chatbot", "the user asks the assistant to write a calculator app", "the user asks the assistant to make a music player"],
            },
            tr={
                "ekrana yazdırma ve input": ["kullanıcı kısa bir selamı ekrana nasıl yazdıracağını sorar", "kullanıcı bir değişkende duran adı nasıl yazdıracağını sorar",
                                             "kullanıcı bir değişkende duran sayıyı nasıl yazdıracağını sorar", "kullanıcı iki sözcüğü aynı satırda nasıl yazdıracağını sorar",
                                             "kullanıcı input ile bir ad sorup kişiyi nasıl selamlayacağını sorar",
                                             "kullanıcı input ile en sevdiği rengi sorup onunla bir cümleyi nasıl yazdıracağını sorar",
                                             "kullanıcı bir metni yazdırıp ikincisini alt satıra nasıl yazdıracağını sorar"],
                "sayılar": ["kullanıcı iki küçük sayıyı toplayıp sonucu nasıl yazdıracağını sorar", "kullanıcı iki sayıyı çarpıp sonucu nasıl yazdıracağını sorar",
                            "kullanıcı bir sayıdan diğerini çıkarıp sonucu nasıl yazdıracağını sorar", "kullanıcı iki sayının toplamını döndüren bir fonksiyon ister",
                            "kullanıcı bir sayıyı ikiyle çarpan bir fonksiyon ister", "kullanıcı iki sayıyı değişkenlere koyup toplamını nasıl yazdıracağını sorar"],
                "döngü ve koşul": ["kullanıcı range ile 0'dan 4'e kadar yazdıran bir for döngüsü ister", "kullanıcı range ile bir sözcüğü üç kez yazdıran bir for döngüsü ister",
                                   "kullanıcı range ile 1'den 5'e kadar yazdıran bir for döngüsü ister", "kullanıcı bir sayının 10'dan büyük olup olmadığını söyleyen bir if/else ister",
                                   "kullanıcı bir sayının pozitif olup olmadığını söyleyen bir if/else ister", "kullanıcı yaşın 18 ya da daha fazla olup olmadığını söyleyen bir if/else ister",
                                   "kullanıcı bir sayının 5'e eşit olup olmadığını denetleyen bir if/else ister"],
                "liste ve metin": ["kullanıcı üç meyveli bir listeyi ve len ile uzunluğunu nasıl alacağını sorar", "kullanıcı bir listenin ilk öğesini nasıl yazdıracağını sorar",
                                   "kullanıcı bir sayı listesinde kaç öğe olduğunu nasıl bulacağını sorar", "kullanıcı upper ile bir sözcüğü büyük harfe nasıl çevireceğini sorar",
                                   "kullanıcı len ile bir sözcükte kaç harf olduğunu nasıl bulacağını sorar", "kullanıcı bir adı büyük harflerle nasıl yazdıracağını sorar",
                                   "kullanıcı dört isimli bir liste yapıp uzunluğunu nasıl yazdıracağını sorar"],
                "bir satırı açıklama": ["kullanıcı print('Merhaba') satırının ne yaptığını sorar", "kullanıcı x = 5 satırının ne yaptığını sorar",
                                        "kullanıcı for i in range(3): print(i) satırının ne yaptığını sorar", "kullanıcı len('elma') satırının ne verdiğini sorar",
                                        "kullanıcı isim.upper() satırının ne yaptığını sorar", "kullanıcı toplam = a + b satırının ne yaptığını sorar"],
                "kavram": ["kullanıcı değişkenin ne olduğunu sorar", "kullanıcı döngünün ne olduğunu sorar", "kullanıcı fonksiyonun ne olduğunu sorar", "kullanıcı listenin ne olduğunu sorar",
                           "kullanıcı metnin (string) ne olduğunu sorar", "kullanıcı bir fonksiyonda return'ün ne demek olduğunu sorar"],
                "büyük istek": ["kullanıcı asistandan bir oyun yapmasını ister", "kullanıcı asistandan bir web sitesi yapmasını ister", "kullanıcı asistandan bir uygulama yazmasını ister",
                                "kullanıcı asistandan bir sohbet botu yapmasını ister", "kullanıcı asistandan bir hesap makinesi uygulaması yazmasını ister",
                                "kullanıcı asistandan bir müzik çalar yapmasını ister"],
            },
        ),
        examples=dict(
            en=[("how do i print hello in python", "print('Hello')\nThis line prints the word Hello on the screen."),
                ("and with a number?", "x = 7\nprint(x)\nThe first line stores 7 in x and the second line prints it.")],
            tr=[("python'da ekrana merhaba nasıl yazdırılır", "print('Merhaba')\nBu satır ekrana Merhaba yazar."),
                ("peki bir sayı yazdırmak için?", "x = 7\nprint(x)\nİlk satır 7'yi x'te saklar, ikincisi onu ekrana yazar.")],
        ),
    ),
    "math": dict(
        weight=0, turns=(2, 3),
        task=dict(
            en="{sub} Every number in the replies must be exactly right: use small whole numbers and check each calculation twice before you write it. Explain in plain words and never invent a rule or a formula. "
               "When a later turn asks for another example, give a new one with different numbers.",
            tr="{sub} Cevaplardaki her sayı tam doğru olmalı: küçük tam sayılar kullan ve her işlemi yazmadan önce iki kez kontrol et. Sade sözlerle anlat, hiçbir kural ya da formül uydurma. "
               "Sonraki bir tur başka bir örnek isterse farklı sayılarla yeni bir örnek ver.",
        ),
        subs=_bilingual({
            "general help, then an area": dict(
                en="The user first asks for help with math in general, with no detail ('i need help with maths', 'could you help with some math?'). The assistant agrees in a few words and asks what exactly the user needs, "
                   "naming two or three areas as examples. The next user message names one area in one or two words, and the assistant then explains in one or two short sentences what that area is about, "
                   "plus one tiny worked example with small whole numbers, all in the same reply.",
                tr="Kullanıcı önce ayrıntı vermeden genel olarak matematikte yardım ister ('matematik konusunda yardım lazım', 'matematikte takıldım, yardım eder misin?'). Asistan birkaç sözcükle kabul eder ve tam olarak neye ihtiyaç olduğunu sorar; "
                   "örnek olarak iki üç konu sayar. Sonraki kullanıcı mesajı bir konuyu bir ya da iki sözcükle söyler; asistan da aynı cevapta o konunun neyle ilgilendiğini bir ya da iki kısa cümleyle açıklar "
                   "ve küçük tam sayılarla tek bir küçük çözümlü örnek verir."),
            "one-step question": dict(
                en="The user asks one simple question in a math area, with no introduction (the question and its correct answer are in the situation below). The assistant gives the correct answer in the first sentence "
                   "and shows in one more short sentence how to get it. A later turn asks a similar question or why the answer is right.",
                tr="Kullanıcı hiçbir giriş yapmadan bir matematik konusunda tek adımlık basit bir soru sorar (soru ve doğru cevabı aşağıdaki durumda verilmiştir). Asistan doğru cevabı ilk cümlede söyler, "
                   "sonra bir kısa cümleyle nasıl bulunduğunu gösterir. Sonraki bir tur benzer bir soru ya da cevabın neden doğru olduğunu sorar."),
        }, MATH_TR),
        hints=dict(
            en={
                "general help, then an area": [f"the user asks for help with math in general, then names this area in one or two words: {a} (a correct tiny example you may use: {e})" for a, e in MATH_AREAS["en"]],
                "one-step question": [f"the user asks: {q} (the correct answer is {a})" for q, a in MATH_QUESTIONS["en"]],
            },
            tr={
                "genel yardım, sonra bir konu": [f"kullanıcı genel olarak matematikte yardım ister, sonra konuyu bir ya da iki sözcükle söyler: {a} (kullanabileceğin doğru küçük bir örnek: {e})" for a, e in MATH_AREAS["tr"]],
                "tek adımlık soru": [f"kullanıcı şunu sorar: {q} (doğru cevap: {a})" for q, a in MATH_QUESTIONS["tr"]],
            },
        ),
        examples=dict(
            en=[("could you help with some math?", "Yes, happy to. What do you need help with: fractions, percent, algebra or something else?"),
                ("fractions", "A fraction is a part of a whole, like 1/4 of a pizza. For example, 1/4 + 2/4 = 3/4, because the pieces are the same size.")],
            tr=[("matematikte takıldım, yardım eder misin?", "Evet, yardım ederim. Hangi konuda: kesirler, yüzde, denklemler ya da başka bir şey mi?"),
                ("yüzde", "Yüzde, bir bütünü 100 parçaya bölüp kaç parça aldığını gösterir. Örneğin 50'nin yüzde 10'u, 50'nin onda biri olan 5'tir.")],
        ),
    ),
}

# Weight 0 marks a task-only family; the default pool keeps the order and weights of the ten original families.
DEFAULT_FAMILIES = tuple(name for name, fam in FAMILIES.items() if fam["weight"])

TEMPLATE = dict(
    en="""You write one realistic short chat between a user and a friendly assistant named Franok. The chat is training data for a small language model, so it must be simple, correct and natural.

Language: English. Write the everyday English that people really use in a chat. Use metric units.
Rules for assistant replies:
- Always clean, correct English with normal spelling and capital letters, even when the user types badly. One to three short sentences, at most 35 words, simple common words.
- Answer the point in the first sentence. No filler openers such as "Sure!" or "Great question!". No markdown, no list longer than three short items, no emojis.
- The assistant is a small chat assistant named Franok. It has no body, age, family or life story, so it never claims to eat, sleep, travel or to have tried something. Never mention a company, another product, a language model or being trained.
- Say only facts you are completely sure of. If you are not sure, say so in a few words or pick an easier fact. Never invent names, dates, prices, news or facts about the user.
- Stay in English for the whole chat. Use the content of earlier replies and what the user said in later replies.
Rules for user messages: one or two short sentences, at most 25 words, typed exactly in the typing style given below in every message, and specific rather than generic.

Output only valid JSON, with no text before or after it, holding exactly as many items as the number of turns given below. Every item is ONE object that holds both the "user" message and the "assistant" reply that answers it:
{{"turns": [{{"user": "...", "assistant": "..."}}]}}

Topic: {topic}. Situation: {hint}.
The user is {persona}. How the user types in every message: {style}.
Chat type: {task}{extra}{detail}
Number of turns: {turns}. A turn is one user message followed by one assistant reply. Later turns continue naturally from the earlier ones.

Example of the form and the length (a different topic; your user messages follow the typing style given above):
{example}""",
    tr="""Bir kullanıcı ile Franok adında dostça bir asistan arasında geçen, gerçekçi ve kısa bir sohbet yaz. Sohbet küçük bir dil modelinin eğitim verisi olacak; bu yüzden sade, doğru ve doğal olmalı.

Dil: Türkçe. Türkiye'de insanların sohbette gerçekten kullandığı günlük Türkçeyi yaz; İngilizceden çeviri gibi durmasın. Kullanıcı ile asistan birbirine "sen" diye hitap eder. ç, ğ, ı, ö, ş, ü harflerini ve ekleri doğru kullan. Emin olmadığın sözcükleri kullanma, sözcük uydurma. Gerekirse Türkiye'ye uygun isim, yer, yemek ve lira fiyatları seç.
Asistan cevapları için kurallar:
- Kullanıcı ne kadar kötü yazarsa yazsın, asistan her zaman temiz, doğru yazılmış ve Türkçe karakterleri eksiksiz kullanan Türkçeyle cevap verir. Bir ila üç kısa cümle, en fazla 35 sözcük, basit ve yaygın sözcükler.
- Asıl noktayı ilk cümlede söyle. "Tabii!" ya da "Harika soru!" gibi dolgu girişler olmasın. Markdown, üç kısa maddeden uzun liste ve emoji kullanma.
- Asistanın adı Franok'tur ve küçük bir sohbet asistanıdır. Bedeni, yaşı, ailesi ve hayat hikâyesi yoktur; bu yüzden yemek yediğini, uyuduğunu, seyahat ettiğini ya da bir şeyi denediğini asla söylemez. Hiçbir şirketi, başka bir ürünü, dil modelini ya da eğitilmiş olmayı anma.
- Yalnızca tamamen emin olduğun bilgileri söyle. Emin değilsen bunu birkaç sözcükle belirt ya da daha kolay bir bilgi seç. İsim, tarih, fiyat, haber ya da kullanıcı hakkında hiçbir şey uydurma.
- Sohbetin tamamını Türkçe sürdür. Sonraki cevaplarda önceki cevapların içeriğini ve kullanıcının söylediklerini kullan.
Kullanıcı mesajları için kurallar: bir ya da iki kısa cümle, en fazla 25 sözcük, her mesaj aşağıda verilen yazma biçiminde yazılmış, genel değil somut.

Yalnızca geçerli JSON yaz; öncesinde ve sonrasında metin olmasın, öğe sayısı aşağıda verilen tur sayısına eşit olsun. Her öğe, hem "user" mesajını hem de ona verilen "assistant" cevabını içeren TEK bir nesnedir:
{{"turns": [{{"user": "...", "assistant": "..."}}]}}

Konu: {topic}. Durum: {hint}.
Kullanıcı: {persona}. Kullanıcı her mesajında şöyle yazar: {style}.
Sohbet türü: {task}{extra}{detail}
Tur sayısı: {turns}. Bir tur, bir kullanıcı mesajı ve ona verilen bir asistan cevabıdır. Sonraki turlar öncekilerin doğal devamı olur.

Biçimi ve uzunluğu gösteren örnek (başka bir konu; senin kullanıcı mesajların yukarıda verilen yazma biçimini izler):
{example}""",
)
EXTRA = dict(
    en="\nOptional detail to weave in if it fits: {twist}.",
    tr="\nUygunsa sohbete ekleyebileceğin ayrıntı: {twist}.",
)
ANCHOR_LINE = dict(
    en="\nWork this concrete detail naturally into the first user message (change its wording to fit the sentence): {text}.",
    tr="\nBu somut ayrıntıyı ilk kullanıcı mesajına doğal biçimde yerleştir (cümleye uyması için biçimini değiştirebilirsin): {text}.",
)


def _turns(rng, lo, hi):
    options = [n for n in (2, 3, 4, 5, 6) if lo <= n <= hi]
    return rng.choices(options, [{2: 36, 3: 28, 4: 20, 5: 10, 6: 6}[n] for n in options])[0]


def build_seeds(count, seed=50505, only=None, prefix=None, tasks=None):
    """Languages alternate (or `only`), families follow their weights, then topic, situation, persona and typing style are random.

    `prefix` names the id batch so that ids never collide with an earlier round.
    `tasks` (family names) restricts the seeds to those families, taken in turn so that every family gets an equal share of each language;
    its ids default to a prefix naming the tasks, so they do not collide with the default run. Without `tasks` the stream is the default pool's.
    """
    if tasks is not None:
        tasks = list(tasks)
        unknown = [t for t in tasks if t not in FAMILIES]
        if not tasks or unknown:
            raise ValueError(f"tasks must be a non-empty list of {sorted(FAMILIES)}, got {tasks}")
    rng = random.Random(seed)
    langs = [only] if only else ["en", "tr"]
    families = list(DEFAULT_FAMILIES)
    weights = [FAMILIES[f]["weight"] for f in families]
    if prefix is None:
        prefix = (only or "") if tasks is None else "-".join([*tasks, *([only] if only else [])]) + "-"
    out = []
    for i in range(count):
        lang = langs[i % len(langs)]
        name = rng.choices(families, weights)[0] if tasks is None else tasks[(i // len(langs)) % len(tasks)]
        fam = FAMILIES[name]
        groups = fam["hints"][lang]
        topic = rng.choice(list(groups))
        styles = STYLES[lang]
        style = rng.choices(list(styles), [w for _, w in styles.values()])[0]
        twist = rng.choice(TWISTS if lang == "en" else TWISTS_TR) if name == "howto" else None
        anchor = rng.choice(ANCHORS[lang]["name"]) if name in ("smalltalk", "feelings") and rng.random() < 0.25 else None
        out.append(
            dict(
                id=f"{prefix}{i:06d}",
                lang=lang,
                task=name,
                topic=topic,
                hint=rng.choice(groups[topic]),
                persona=rng.choice(PERSONAS if lang == "en" else PERSONAS_TR),
                style=style,
                turns=_turns(rng, *fam["turns"]),
                twist=twist,
                anchor=anchor,
            )
        )
    return out


def build_prompt(seed):
    fam, lang = FAMILIES[seed["task"]], seed["lang"]
    sub = fam.get("subs", {}).get(seed["topic"], {}).get(lang, "")
    task = fam["task"][lang].format(sub=sub, twist=seed["twist"] or "", subject=seed["hint"]).strip()
    extra = EXTRA[lang].format(twist=seed["twist"]) if seed["twist"] and seed["task"] != "howto" else ""
    detail = ANCHOR_LINE[lang].format(text=f"the user's name is {seed['anchor']}" if lang == "en" else f"kullanıcının adı {seed['anchor']}") if seed.get("anchor") else ""
    example = json.dumps(dict(turns=[dict(user=u, assistant=a) for u, a in fam["examples"][lang]]), ensure_ascii=False)
    return TEMPLATE[lang].format(
        topic=seed["topic"], hint=seed["hint"], persona=seed["persona"], style=STYLES[lang][seed["style"]][0], task=task, extra=extra,
        detail=detail, turns=seed["turns"], example=example,
    )


if __name__ == "__main__":
    seeds = build_seeds(2000)
    for key in ("task", "style"):
        print(key, sorted(Counter((s["lang"], s[key]) for s in seeds).items()))
    print(build_prompt(seeds[1]))
