"""Original MIT-licensed sample dialogues about names, capabilities and changing plans."""

import json, random
from pathlib import Path

OPENERS = {
    "en": [
        ("Hi!", "Hi! What would you like to talk about?"),
        ("Hello", "Hello! How can I help?"),
        ("Hey there.", "Hey! What's on your mind?"),
        ("Good morning!", "Good morning! How's your day going?"),
        ("Are you there?", "I'm here. What can I help you with?"),
        ("Do you have a name?", "My name is Franok."),
        ("Who are you?", "I'm Franok, a small English and Turkish language model."),
        ("What languages do you speak?", "I can chat in English and Turkish."),
        ("Can we talk for a minute?", "Of course. What's on your mind?"),
        (
            "I'm having a rough day.",
            "That sounds difficult. Do you want to talk about what happened?",
        ),
        ("Can you see my screen?", "I can only read what you send in this chat."),
        ("Can I ask something?", "Of course. What's your question?"),
        (
            "I don't understand your answer.",
            "Which part would you like me to explain differently?",
        ),
        (
            "This is confusing.",
            "Let's take it one step at a time. What part is confusing?",
        ),
        ("Thank you!", "You're welcome!"),
        ("See you later.", "See you later!"),
        (
            "I’m just looking for someone to chat with.",
            "Sure. What have you been up to today?",
        ),
    ],
    "tr": [
        ("Merhaba!", "Merhaba! Ne hakkında konuşmak istersin?"),
        ("Selam", "Selam! Sana nasıl yardımcı olabilirim?"),
        ("Nasılsın?", "Sohbete hazırım. Sen nasılsın?"),
        ("Günaydın!", "Günaydın! Günün nasıl gidiyor?"),
        ("Orada mısın?", "Buradayım. Nasıl yardımcı olabilirim?"),
        ("Adın ne?", "Benim adım Franok."),
        (
            "Sen kimsin?",
            "Ben Franok, Türkçe ve İngilizce konuşabilen küçük bir dil modeliyim.",
        ),
        ("Hangi dilleri biliyorsun?", "Türkçe ve İngilizce konuşabiliyorum."),
        ("Biraz sohbet edebilir miyiz?", "Tabii. Aklında ne var?"),
        (
            "Bugün pek iyi geçmedi.",
            "Bunu duyduğuma üzüldüm. Neler olduğunu anlatmak ister misin?",
        ),
        (
            "Ekranımı görebiliyor musun?",
            "Yalnızca bu sohbette yazdıklarını okuyabiliyorum.",
        ),
        ("Bir şey sorabilir miyim?", "Elbette, sorunu dinliyorum."),
        ("Cevabını anlamadım.", "Hangi kısmı farklı bir şekilde açıklamamı istersin?"),
        ("Kafam karıştı.", "Birlikte adım adım gidelim. Hangi kısmı karışık geldi?"),
        ("Çok teşekkürler!", "Rica ederim!"),
        ("Sonra görüşürüz.", "Görüşmek üzere!"),
        ("Sadece sohbet etmek istiyorum.", "Olur. Bugün neler yaptın?"),
    ],
}
NAMES = "Duru Efe Arda Suna Cem Lara Deniz Mira Ada Aylin Ekin İpek Batu Nehir Mert Sena Tuna Leyla Ece Can Selin Mina Derin Robin Nova Alex Lena Tom Rosa Roy Erin Inez Sam Maya".split()


def dialogues(seed, count):
    rng = random.Random(seed)
    for i in range(count):
        lang = ("en", "tr")[i % 2]
        en = lang == "en"
        name, new = rng.sample(NAMES, 2)
        hours = sorted(rng.sample(range(8, 21), 2))
        t, u = [f"{h:02}:{rng.choice([0, 15, 30, 45]):02}" for h in hours]
        hello, hi = rng.choice(OPENERS[lang])
        turns = [hello, hi]
        task = rng.randrange(6)
        if task == 0:
            turns += (
                [
                    f"My name is {name}.",
                    f"Nice to meet you, {name}.",
                    "What did I say my name was?",
                    f"You said your name was {name}.",
                ]
                if en
                else [
                    f"Benim adım {name}.",
                    f"Tanıştığımıza memnun oldum, {name}.",
                    "Adımın ne olduğunu hatırlıyor musun?",
                    f"Adının {name} olduğunu söyledin.",
                ]
            )
        elif task == 1:
            turns += (
                [
                    f"Call me {name}.",
                    f"Sure, {name}.",
                    f"Actually, call me {new}.",
                    f"Of course, {new}.",
                    "What should you call me now?",
                    new + ".",
                ]
                if en
                else [
                    f"Bana {name} diye hitap et.",
                    f"Tamam, {name}.",
                    f"Vazgeçtim, bana {new} de.",
                    f"Tabii, {new}.",
                    "Şimdi bana nasıl hitap edeceksin?",
                    new + ".",
                ]
            )
        elif task == 2:
            turns += (
                [
                    f"My train leaves at {t}.",
                    f"Got it, your train leaves at {t}.",
                    f"Correction: it's {u}.",
                    f"Thanks for the update. The train leaves at {u}.",
                    "What time does it leave?",
                    u + ".",
                ]
                if en
                else [
                    f"Trenim saat {t}’te kalkıyor.",
                    f"Trenin kalkış saati {t}.",
                    f"Düzelteyim, yeni saat {u}.",
                    f"Tamam, güncel kalkış saati {u}.",
                    "Trenin son kalkış saati ne?",
                    u + ".",
                ]
            )
        elif task == 3:
            place = rng.choice(
                [
                    "the kitchen",
                    "the office",
                    "the hallway",
                    "the cupboard",
                    "the studio",
                ]
                if en
                else ["mutfak", "ofis", "antre", "dolap", "atölye"]
            )
            thing = rng.choice(
                ["notebook", "phone", "folder", "calendar", "key"]
                if en
                else ["defter", "telefon", "dosya", "takvim", "anahtar"]
            )
            turns += (
                [
                    f"The {thing} is in {place}.",
                    f"I'll keep that in mind.",
                    "Where did I leave it?",
                    f"You left the {thing} in {place}.",
                ]
                if en
                else [
                    f"{thing.capitalize()} için ayırdığım yer: {place}.",
                    "Tamam, aklımda.",
                    "Ona nerede yer ayırmıştım?",
                    f"Söylediğin yer {place}.",
                ]
            )
        elif task == 4:
            turns += (
                [
                    f"Help me invite {name} to coffee.",
                    f"Hi {name}, would you like to grab a coffee?",
                    f"Add the time, {t}.",
                    f"Hi {name}, would you like to grab a coffee at {t}?",
                ]
                if en
                else [
                    f"{name} için kısa bir kahve daveti yaz.",
                    f"Merhaba {name}, birlikte kahve içmek ister misin?",
                    f"Saati de ekle: {t}.",
                    f"Merhaba {name}, saat {t} civarında birlikte kahve içelim mi?",
                ]
            )
        else:
            turns = (
                [
                    f"I have a question about {name}.",
                    "What would you like to know?",
                    f"What's their favourite colour?",
                    "You haven't mentioned their favourite colour. Do you know it?",
                ]
                if en
                else [
                    f"{name} hakkında bir sorum var.",
                    "Ne öğrenmek istiyorsun?",
                    "En sevdiği renk hangisi?",
                    "En sevdiği rengi henüz söylemedin. Biliyor musun?",
                ]
            )
        yield dict(
            language=lang,
            messages=[
                dict(role=("user", "assistant")[j % 2], content=s)
                for j, s in enumerate(turns)
            ],
        )


def main():
    out = Path("work/tasks")
    out.mkdir(parents=True, exist_ok=True)
    seen = set()
    for s, n, seed in [("validation", 384, 36573), ("train", 8192, 77545)]:
        with (out / ("practice-" + s + ".jsonl")).open("w") as f:
            for item in dialogues(seed, n):
                key = json.dumps(item, ensure_ascii=False)
                if key not in seen:
                    seen.add(key)
                    f.write(key + "\n")


if __name__ == "__main__":
    main()
