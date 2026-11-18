"""Programmatic seeds and the teacher prompt for Franok 0.4 conversations."""

import json
import random
from collections import Counter

TOPICS = {
    "home": [
        "removing sticky label residue from a glass jar",
        "a door that squeaks when it opens",
        "keeping a small bathroom smelling fresh",
        "drying clothes indoors in winter",
        "hanging a shelf without drilling into the wall",
        "a mattress that feels too warm at night",
        "getting a red wine stain out of a cotton tablecloth",
        "organizing a tiny entrance hall",
        "a slow-draining kitchen sink",
        "noise coming from the flat upstairs",
        "cleaning a kettle with limescale",
        "choosing a rug for a living room",
        "a window that fogs up every morning",
        "storing winter blankets in summer",
        "a drawer that keeps getting stuck",
        "lowering the electricity bill at home",
    ],
    "cooking": [
        "cooking rice that is not sticky",
        "a quick dinner with eggs and spinach",
        "boiling an egg with a soft yolk",
        "a vegetarian meal for guests",
        "storing leftover soup",
        "baking bread without a mixer",
        "what to do with overripe bananas",
        "making tea strong but not bitter",
        "a breakfast that takes five minutes",
        "replacing a missing ingredient such as butter or milk",
        "cooking dry lentils",
        "a child's lunchbox that stays fresh",
        "reheating pasta without drying it out",
        "cutting an onion without tears",
        "a simple dessert with three ingredients",
        "how much salt a pot of soup needs",
    ],
    "plants": [
        "a basil plant with yellow leaves",
        "how often to water a cactus",
        "a balcony plant that gets afternoon sun",
        "repotting a plant whose roots fill the pot",
        "tiny flies in the soil of a houseplant",
        "growing tomatoes in a pot",
        "a plant that drops leaves after being moved",
        "low-light plants for a bedroom",
        "when to prune a rose bush",
        "saving a dried-out mint plant",
        "starting seeds on a windowsill",
        "keeping plants alive during a holiday",
        "brown tips on a spider plant",
        "choosing soil for succulents",
        "a lemon tree in a pot",
        "flowers that attract bees",
    ],
    "study": [
        "memorizing vocabulary for a language class",
        "staying focused during an hour of reading",
        "preparing for an exam in three days",
        "taking notes during a lecture",
        "a daily study routine",
        "understanding a hard maths topic",
        "writing the opening of a short essay",
        "calming exam nerves",
        "studying with a noisy flatmate",
        "learning from mistakes on a test",
        "balancing a part-time job and classes",
        "a good way to review before sleep",
        "summarising a long chapter",
        "choosing between two elective courses",
        "learning to touch-type",
        "preparing a five-minute presentation",
    ],
    "travel": [
        "packing for a three-day city trip",
        "getting from an airport to the city centre",
        "a long bus journey with a child",
        "what to do after losing a train ticket",
        "visiting a museum with limited time",
        "staying safe while travelling alone",
        "travelling with hand luggage only",
        "tiredness after a long flight",
        "a day trip by ferry",
        "choosing between a train and a bus",
        "finding a cheap place to stay",
        "a rainy day on holiday",
        "finding the way without mobile data",
        "exchanging a small amount of foreign money",
        "asking for directions politely in a new city",
        "choosing a seat on a long flight",
    ],
    "shopping": [
        "choosing between two similar phones",
        "buying a winter coat on a small budget",
        "checking whether a second-hand bicycle is worth buying",
        "returning an item without a receipt",
        "comparing prices at two supermarkets",
        "a gift for a friend who likes reading",
        "choosing running shoes",
        "picking fruit that is ripe today",
        "avoiding impulse purchases",
        "choosing a laptop bag",
        "buying a fan for a hot room",
        "a gift for a grandmother",
        "buying or renting a tool you need once",
        "judging the quality of a jacket in a shop",
        "reading the label of a cleaning product",
        "writing a weekly grocery list",
    ],
    "weather": [
        "what to wear on a windy, cold morning",
        "whether to carry an umbrella on a cloudy day",
        "why the sky turns red at sunset",
        "staying cool during a heat wave",
        "driving safely in fog",
        "what a forecast of 70 percent rain means",
        "drying laundry on a humid day",
        "protecting plants from frost",
        "why it feels colder than the thermometer says",
        "preparing the home for a thunderstorm",
        "packing for a rainy weekend",
        "sunburn on a cloudy day",
        "snow on the road in the morning",
        "why hail happens",
        "reading simple weather symbols",
        "planning a picnic when the weather keeps changing",
    ],
    "pets": [
        "a cat that scratches the sofa",
        "walking a dog twice a day",
        "feeding a puppy",
        "a goldfish tank that turns cloudy",
        "introducing a new cat to the home",
        "a dog that barks when left alone",
        "keeping a pet cool in summer",
        "a hamster that sleeps all day",
        "choosing a pet for a small flat",
        "trimming a dog's nails",
        "a cat that refuses new food",
        "bird seed for a balcony feeder",
        "travelling with a pet in a car",
        "the first week with a rescue dog",
        "keeping pet hair off clothes",
        "daily care for a rabbit",
    ],
    "simple science": [
        "why ice floats on water",
        "why the moon changes shape",
        "how a magnet works in simple words",
        "why lightning is seen before thunder is heard",
        "what makes bread rise",
        "why metal feels colder than wood",
        "why leaves change colour in autumn",
        "how plants make food from light",
        "why the sea is salty",
        "how rainbows form",
        "why people yawn",
        "how a thermometer works",
        "what causes the tides",
        "why a rubbed balloon sticks to a wall",
        "why the sky is blue",
        "why a spoon looks bent in a glass of water",
    ],
    "time and schedules": [
        "planning a morning routine",
        "fitting a 30-minute walk into a busy day",
        "converting a meeting time between two cities",
        "arriving on time for a 9:30 appointment",
        "splitting a weekend between chores and rest",
        "a bus transfer with ten minutes between buses",
        "a weekly study timetable",
        "how many hours of sleep fit between 23:30 and 06:45",
        "scheduling a call across two time zones",
        "a routine for remembering to water plants",
        "finding a free hour among three tasks",
        "deciding what to do first from a to-do list",
        "how long a 240-kilometre trip takes at 80 km/h",
        "planning breaks during a long work day",
        "going to bed earlier",
        "what day it will be ten days from now",
    ],
    "money basics": [
        "saving a little each month",
        "splitting a restaurant bill fairly",
        "what an interest rate means",
        "a simple monthly budget for a student",
        "paying off a small debt first",
        "the difference between a debit card and a credit card",
        "tracking spending with a notebook",
        "saying no when a friend asks to borrow money",
        "whether a discount is really a good deal",
        "building a small emergency fund",
        "what inflation means for prices",
        "sharing rent with a flatmate",
        "keeping receipts for a warranty",
        "paying in cash or by card",
        "buying now or waiting for a sale",
        "teaching a child about pocket money",
    ],
    "writing help": [
        "a polite email asking for a deadline extension",
        "a short thank-you note to a neighbour",
        "a text message declining an invitation kindly",
        "a two-line birthday message for a colleague",
        "making a sentence sound more formal",
        "apologizing for a late reply",
        "a short introduction for a CV",
        "a complaint to a shop about a broken item",
        "shortening a long message",
        "a title for a school project on recycling",
        "a friendly reminder about an unpaid share of the rent",
        "a caption for a holiday photo",
        "a message asking a teacher for feedback",
        "a note to the landlord about a leaking tap",
        "rewriting a blunt sentence so it sounds kinder",
        "a short speech to welcome guests",
    ],
    "everyday health": [
        "a mild headache after a long day at a screen",
        "trouble falling asleep",
        "stretching after sitting all day",
        "drinking enough water",
        "a small cut on a finger",
        "a stiff neck after sleeping badly",
        "a sore throat that started yesterday",
        "eating well on a tight schedule",
        "tired eyes from reading",
        "a short walking routine for a beginner",
        "a blister from new shoes",
        "what helps at home with a mild cold",
        "sitting posture at a desk",
        "a nosebleed that has just started",
        "healthy snacks for work",
        "feeling dizzy after standing up fast",
    ],
}

TOPICS.update(
    {
        "phones and computers": [
            "a phone that runs out of storage",
            "a laptop fan that gets loud",
            "choosing a strong password you can remember",
            "a video call where the sound keeps cutting out",
            "backing up photos before a trip",
            "a slow Wi-Fi connection in one room",
            "an email that went to the wrong person",
            "cleaning a keyboard with sticky keys",
            "a phone screen with a small crack",
            "a printer that says it is offline",
            "organising files in folders",
            "a text message from an unknown number asking for money",
            "setting up a new phone for a grandparent",
            "a laptop battery that drains quickly",
        ],
        "work and office": [
            "a meeting that always runs over time",
            "asking a manager for a day off",
            "a noisy open-plan office",
            "a to-do list that never gets shorter",
            "starting a new job next Monday",
            "writing a short status update for the team",
            "a colleague who interrupts often",
            "eating lunch at the desk every day",
            "preparing for a job interview tomorrow",
            "a long commute and tired mornings",
            "saying no to extra work politely",
            "a shared calendar full of clashes",
            "learning a new spreadsheet formula",
            "taking short breaks during a busy shift",
        ],
        "neighbours and social life": [
            "inviting a neighbour for tea",
            "a friend who is always late",
            "making friends in a new city",
            "declining a party invitation",
            "a loud celebration next door",
            "a gift for a host when visiting for dinner",
            "meeting a friend's family for the first time",
            "a group chat that never stops buzzing",
            "returning a borrowed book late",
            "a birthday that was almost forgotten",
            "a shy child at a family gathering",
            "keeping in touch with a friend who moved away",
            "a neighbour's dog that barks all afternoon",
            "planning a small surprise for a friend",
        ],
        "hobbies and games": [
            "starting to learn the guitar",
            "a jigsaw puzzle with a missing piece",
            "choosing a first board game for four people",
            "taking better photos with a phone",
            "knitting a first scarf",
            "a chess opening for a beginner",
            "keeping a simple diary",
            "drawing a face that looks right",
            "a long walk with a bird guide",
            "building a small model from a kit",
            "a card game to play on a train",
            "practising a song for ten minutes a day",
            "fishing for the first time",
            "joining a local running group",
        ],
        "city life and transport": [
            "finding the right bus stop",
            "topping up a travel card",
            "cycling to work in traffic",
            "parking in a crowded street",
            "a train that is delayed by an hour",
            "crossing a big city by metro",
            "a taxi fare that seems too high",
            "walking home at night",
            "a missed stop on a long bus line",
            "carrying heavy shopping on public transport",
            "a ferry timetable with two lines",
            "sharing a ride with a colleague",
            "a rainy morning commute",
            "learning the metro map of a new city",
        ],
        "children and family": [
            "a toddler who refuses to eat vegetables",
            "a bedtime routine for a five-year-old",
            "a child who is afraid of the dark",
            "keeping children busy on a rainy afternoon",
            "a teenager who stays in their room",
            "helping with homework without giving the answer",
            "a grandparent who lives far away",
            "a sibling quarrel over a toy",
            "teaching a child to tie shoelaces",
            "a family dinner with picky eaters",
            "a first day at a new school",
            "saying goodbye at nursery without tears",
            "a baby who wakes up at night",
            "chores that a seven-year-old can do",
        ],
        "bikes and cars": [
            "a bicycle tyre that goes flat",
            "a bicycle chain that keeps slipping",
            "checking the oil level in a car",
            "a car that will not start on a cold morning",
            "a warning light on the dashboard",
            "buying a used bicycle helmet",
            "packing a car for a long drive",
            "a windscreen that fogs up inside",
            "a squeaking brake on a bicycle",
            "keeping a car clean inside",
            "choosing a bicycle lock",
            "a long drive with a sleepy driver",
            "checking tyre pressure",
            "learning to ride a bicycle as an adult",
        ],
        "small repairs": [
            "a button that fell off a shirt",
            "a leaking tap",
            "a hole in a pair of socks",
            "a zip that will not close",
            "a wobbly chair leg",
            "a light bulb that keeps burning out",
            "a scratch on a wooden table",
            "a torn page in a book",
            "a loose door handle",
            "a bicycle bell that stopped ringing",
            "a pot with a loose handle",
            "a ripped seam on a bag",
            "a curtain rail that is coming off",
            "a remote control that stopped working",
        ],
        "learning languages": [
            "remembering new words in another language",
            "speaking with a stranger in a language you are learning",
            "a pronunciation that feels impossible",
            "choosing between an app and a class",
            "reading a first book in another language",
            "mixing up two similar words",
            "a short daily listening habit",
            "writing a message in a language you are still learning",
            "the difference between two similar verbs",
            "keeping going after a month without progress",
            "a language exchange with a pen friend",
            "watching films with subtitles to learn",
            "learning numbers and dates",
            "polite phrases for a trip abroad",
        ],
        "feelings and motivation": [
            "feeling nervous before speaking in public",
            "starting a task that feels too big",
            "a bad day that is hard to shake off",
            "feeling lonely on a Sunday evening",
            "being proud of a small success",
            "comparing yourself with others online",
            "waiting for important news",
            "getting back on track after missing a week",
            "feeling tired but not sleepy",
            "a friend who is going through a hard time",
            "making a decision between two good options",
            "keeping a New Year plan going",
            "being criticised at work",
            "needing a quiet hour alone",
        ],
        "geography and nature": [
            "why deserts are cold at night",
            "how rivers reach the sea",
            "what a time zone is",
            "why mountains have snow on top",
            "how a compass finds north",
            "what a rainforest is",
            "why islands have their own animals",
            "what the equator is",
            "how long a day is in summer and winter",
            "why some lakes are salty",
            "how bees make honey",
            "why birds fly south for winter",
            "what a volcano is",
            "how caves form",
        ],
    }
)

PERSONAS = [
    "a university student", "a retired teacher", "a parent of a toddler",
    "a night-shift nurse", "a new flatmate", "a small-shop owner", "a teenager",
    "a busy office worker", "a first-time traveller", "a complete beginner",
    "someone on a tight budget", "someone living in a tiny flat",
    "a freelance designer", "a bus driver", "a grandmother",
    "someone who just moved to a new city", "a high-school student",
    "a delivery rider", "someone working from home", "a shy person",
    "an impatient person", "a careful planner", "a hairdresser", "a taxi driver",
    "a baker", "a nurse's assistant", "a secondary-school teacher", "a plumber",
    "a first-year engineering student", "a pensioner who likes walking", "a farmer's daughter",
    "a waiter", "a call-centre worker", "a father of three", "a person learning a second language",
    "a cyclist", "a bookshop assistant", "a gardener", "a student sharing a room",
    "a new parent", "a warehouse worker", "a tour guide", "a person who dislikes technology",
    "a curious ten-year-old with a parent nearby", "a person recovering from a cold",
    "a volunteer at a food bank", "a cashier", "a train conductor", "an athlete in training",
    "a person who is always in a hurry", "a thoughtful person who writes full sentences",
    "a person living alone for the first time", "an exhausted person after a double shift",
]
STYLES = [
    "short and casual", "polite, in full sentences", "terse, with no greeting",
    "a little worried", "curious and chatty but under 30 words",
    "direct and practical",
    "lower case with no punctuation, like a quick text message",
    "friendly and slightly humorous",
    "formal and businesslike",
    "unsure, with a small apology",
    "very short, three to six words",
    "detailed, giving one useful fact about the situation",
    "a little annoyed",
    "enthusiastic but vague",
]
TWISTS = [
    "it must cost almost nothing",
    "only ten minutes are available",
    "there is no oven or stove",
    "someone in the household is allergic to a specific, named common ingredient",
    "the space is very small",
    "a young child will be around",
    "it has to be quiet because someone is sleeping",
    "there is no internet or phone signal",
    "guests are arriving soon",
    "the user has no tools at all",
    "the user has a bad back",
    "it has to be done outdoors",
    "the user dislikes the main item of the first suggestion",
    "the user needs a much shorter version",
    "the weather has turned bad",
    "it needs to last a whole week",
    "the user can use only what is already at home",
    "the user has just ten euros or about 300 lira",
    "the user is in a different city from the person involved",
    "the user tried the first idea and it failed",
    "the user is doing it for an older relative",
    "the user is in a hurry and wants only the next step",
    "the user must keep it secret as a surprise",
    "the user is doing this for the first time",
    "it has to be reusable rather than thrown away",
    "the user lives on the top floor with no lift",
    "the user wants the safest option, not the fastest",
    "two people have to share it",
    "it must work for a person who cannot read small print",
    "the user has no car",
]
# Concrete details that make each first message different from the next.
ANCHORS = {
    "en": {
        "name": [
            "Sam", "Priya", "Tomasz", "Hana", "Marco", "Aisha", "Keiko", "Lucas", "Fatima", "Oliver",
            "Chloe", "Dmitri", "Grace", "Yusuf", "Anna", "Mateo", "Zara", "Henrik", "Leila", "Jonas",
            "Ines", "Rahul", "Mina", "Victor", "Elsa", "Kofi", "Nadia", "Pavel", "Bea", "Ravi",
        ],
        "object": [
            "a green backpack", "a thermos flask", "a pair of old boots", "a wool scarf", "a ceramic mug",
            "a folding chair", "a wooden spoon", "a bicycle pump", "a desk lamp", "a plastic crate",
            "a reading light", "a set of keys on a blue ring", "a cast-iron pan", "a small radio",
            "a cotton shirt", "a metal water bottle", "a hand mirror", "a paper map", "a clay pot",
            "a tin of paint", "an umbrella with a bent rib", "a sleeping bag", "a laundry basket",
            "a glass bowl", "a pair of headphones", "a shoebox", "a rubber doormat", "a bedside clock",
            "a garden hose", "a wall calendar", "a hand towel", "a lunch bag", "a leather belt",
            "a set of coasters", "a toolbox", "an extension cable", "a picnic blanket", "a jar of honey",
        ],
        "number": [
            "{n} euros", "{n} minutes", "{n} kilometres", "{n} days", "{n} people", "{n} pages",
            "the number {n}", "about {n} grams", "{n} degrees", "{n} litres",
        ],
        "place": [
            "a small town near the sea", "a flat on the fourth floor", "a village with one shop",
            "a student dormitory", "a house with a small garden", "a basement flat", "a hostel",
            "an apartment above a bakery", "a shared house", "a farm in the hills", "a flat near a station",
            "a house by a river", "a quiet street with trees", "a market town", "a flat on a busy road",
        ],
    },
    "tr": {
        "name": [
            "Ayşe", "Mehmet", "Zeynep", "Burak", "Elif", "Kemal", "Selin", "Onur", "Derya", "Emre",
            "Büşra", "Cem", "Gül", "Hakan", "İpek", "Murat", "Nazlı", "Okan", "Pınar", "Serkan",
            "Tuğba", "Volkan", "Yasemin", "Ahmet", "Berna", "Ceren", "Doruk", "Esra", "Furkan", "Hülya",
        ],
        "object": [
            "yeşil bir sırt çantası", "bir termos", "eski bir çift bot", "yünlü bir atkı", "seramik bir kupa",
            "katlanır bir sandalye", "bir tahta kaşık", "bir bisiklet pompası", "bir masa lambası", "plastik bir kasa",
            "bir okuma lambası", "mavi halkalı bir anahtarlık", "bir dökme demir tava", "küçük bir radyo",
            "pamuklu bir gömlek", "metal bir su matarası", "bir el aynası", "kâğıt bir harita", "bir toprak çömlek",
            "bir kutu boya", "teli eğri bir şemsiye", "bir uyku tulumu", "bir çamaşır sepeti",
            "cam bir kase", "bir çift kulaklık", "bir ayakkabı kutusu", "kauçuk bir paspas", "bir başucu saati",
            "bir bahçe hortumu", "bir duvar takvimi", "bir el havlusu", "bir beslenme çantası", "deri bir kemer",
            "bir takım bardak altlığı", "bir alet çantası", "bir uzatma kablosu", "bir piknik örtüsü", "bir kavanoz bal",
        ],
        "number": [
            "{n} lira", "{n} dakika", "{n} kilometre", "{n} gün", "{n} kişi", "{n} sayfa",
            "{n} sayısı", "yaklaşık {n} gram", "{n} derece", "{n} litre",
        ],
        "place": [
            "denize yakın küçük bir kasaba", "dördüncü kattaki bir daire", "tek marketi olan bir köy",
            "bir öğrenci yurdu", "küçük bahçeli bir ev", "bodrum katında bir daire", "bir pansiyon",
            "fırının üstündeki bir daire", "paylaşılan bir ev", "dağlardaki bir çiftlik", "istasyona yakın bir daire",
            "nehir kenarında bir ev", "ağaçlı sessiz bir sokak", "küçük bir ilçe", "işlek bir caddedeki daire",
        ],
    },
}
OPENERS = [
    "a direct question", "a short description of the problem with no question mark",
    "a request that starts with a polite phrase", "a statement of what the user already tried, then a question",
    "a question that begins with a question word", "a message that starts with the reason for asking",
    "a very short question", "a request for a specific number of options or steps",
    "a worry stated in one sentence", "a question that mentions a time or a deadline",
]
LABEL_DOMAINS = [
    "animals", "geography", "counting and numbers", "days, months and the calendar", "the human body",
    "foods and kitchen items", "materials such as wood, glass or metal", "colours and shapes",
    "simple physics such as floating, heat and magnets", "letters, words and spelling",
    "coins, prices and change", "plants and trees", "sports and games", "musical instruments",
    "weather and seasons", "tools and what they are for", "transport and vehicles", "time on a clock",
    "family words", "rooms of a house and furniture",
]
LABEL_FORMATS = [
    "answer yes or no", "pick one of two named options", "answer with only a number",
    "answer with only a category word", "answer true or false", "answer with one word",
    "answer with the name of one thing", "pick the odd one out from three words",
]
TASKS = {
    "practical": (
        "The user asks for a concrete everyday piece of help. The second user message is a related practical "
        "question that adds a new detail."
    ),
    "clarify": (
        "Turn 1: the user message is vague or misses key information, and the assistant reply is exactly one short "
        "clarifying question (not an answer). Turn 2: the user supplies the missing details and the assistant reply "
        "answers directly using them. With three turns, the third turn is a follow-up question that builds on that answer."
    ),
    "change": (
        "The user asks for help, then in the second message adds or changes a preference or constraint: {twist}. "
        "The second assistant reply adapts the first answer to that change and keeps what still works."
    ),
    "label": (
        "The user asks general-knowledge or common-sense questions that have one correct, checkable answer that is the same "
        "for everybody (yes or no, one of two options, a number, a category) and tells the assistant to answer in that format. "
        "Use the situation only as a loose theme: if it would need the user's own facts, local opening hours, prices, travel "
        "times or an opinion, ask about a well-known general fact on the same theme instead. The assistant answers with only "
        "that word, number or short phrase, and it must be true. The second question uses the same format and builds on the first."
    ),
    "polite": (
        "The user is courteous: greets, thanks, apologizes for bothering, or asks politely. The assistant answers warmly "
        "and briefly, then helps with the actual request in the second turn."
    ),
    "refer": (
        "The second user message points back to the first assistant reply (for example: that second tip, make it shorter, "
        "explain that word, which one did you say first). The second reply must use the specific content of the first reply."
    ),
}
LANGUAGES = {
    "en": (
        "English",
        "Write natural, everyday English. Use metric units.",
    ),
    "tr": (
        "Turkish",
        "Write the whole chat in natural, everyday Turkish as people in Turkey speak it. Do not translate from English: "
        "choose names, places, foods, prices in lira and habits that fit Turkey when the situation allows. "
        "The user and assistant address each other informally (sen). Use ç, ğ, ı, ö, ş, ü and Turkish suffixes correctly. "
        "Keep sentences short and grammatical, use only Turkish words you are sure of, and do not invent words or advice.",
    ),
}
EXAMPLES = {
    ("en", "practical"): [
        ("My white socks turned grey in the wash. Can I fix that?",
         "Soak them for an hour in warm water with a spoonful of baking soda, then wash them again. Next time, wash whites separately from dark clothes."),
        ("Do I have to use hot water?",
         "No, warm water is enough. Very hot water can shrink cotton, so keep it under 40 degrees."),
    ],
    ("en", "clarify"): [
        ("Can you help me choose a gift?",
         "Of course. Who is it for, and roughly how much do you want to spend?"),
        ("It's for my brother, he turns 12, and I can spend about 20 pounds.",
         "A small building kit or a sketchbook with coloured pencils would fit. Both are fun at that age and stay within your budget."),
    ],
    ("en", "change"): [
        ("What can I cook for dinner with potatoes and onions?",
         "Try a potato and onion omelette. Slice both thinly, fry them until soft, then pour beaten eggs over them and cook until set."),
        ("I don't have any eggs. What else can I do?",
         "Fry the same potatoes and onions with a little paprika and serve them with bread. It is simple and filling."),
    ],
    ("en", "label"): [
        ("Is a whale a fish or a mammal? Answer in one word.", "Mammal."),
        ("And a dolphin? One word again.", "Mammal."),
    ],
    ("en", "polite"): [
        ("Good morning! Sorry to bother you so early.",
         "Good morning! It is no bother at all. What can I help you with?"),
        ("Thank you. How can I keep bread fresh for longer?",
         "Keep it in a bread box or a cloth bag at room temperature. If you need more than three days, freeze it in slices."),
    ],
    ("en", "refer"): [
        ("Give me two ways to remember where I put my keys.",
         "Always put them in the same bowl by the door. Or say out loud where you put them each time."),
        ("Can you make the second one shorter, something I can repeat?",
         "Say to yourself: \"Keys on the shelf.\""),
    ],
    ("tr", "practical"): [
        ("Beyaz çoraplarım yıkayınca gri oldu. Düzelir mi?",
         "Ilık suya bir kaşık karbonat atıp çorapları bir saat beklet, sonra yeniden yıka. Bir dahaki sefere beyazları koyu renklilerden ayrı yıka."),
        ("Mutlaka sıcak su mu gerekiyor?",
         "Hayır, ılık su yeterli. Çok sıcak su pamuğu çektirebilir, o yüzden 40 derecenin altında kal."),
    ],
    ("tr", "clarify"): [
        ("Bana bir hediye seçmede yardım eder misin?",
         "Tabii. Kim için ve yaklaşık ne kadar harcamak istiyorsun?"),
        ("Abim için, 12 yaşına giriyor, 500 liraya kadar harcayabilirim.",
         "Küçük bir yapı seti ya da renkli kalemli bir çizim defteri iyi olur. İkisi de bu yaşta eğlenceli ve bütçene uyar."),
    ],
    ("tr", "change"): [
        ("Evde patates ve soğan var. Akşam yemeği için ne pişirebilirim?",
         "Patatesli soğanlı bir omlet yapabilirsin. İkisini ince dilimleyip yumuşayana kadar kavur, üstüne çırpılmış yumurta dök ve pişir."),
        ("Yumurta yok. Başka ne yapabilirim?",
         "Aynı patates ve soğanı biraz pul biberle kavur, ekmekle ye. Basit ve doyurucu olur."),
    ],
    ("tr", "label"): [
        ("Balina balık mı, memeli mi? Tek kelimeyle yaz.", "Memeli."),
        ("Peki yunus? Yine tek kelime.", "Memeli."),
    ],
    ("tr", "polite"): [
        ("Günaydın! Bu kadar erken rahatsız ettiğim için kusura bakma.",
         "Günaydın! Hiç rahatsız etmedin. Sana nasıl yardımcı olabilirim?"),
        ("Sağ ol. Ekmeği daha uzun süre taze tutmak için ne yapmalıyım?",
         "Ekmeği oda sıcaklığında ekmek kutusunda ya da bez torbada sakla. Üç günden uzun tutacaksan dilimleyip dondur."),
    ],
    ("tr", "refer"): [
        ("Anahtarlarımı nereye koyduğumu hatırlamak için iki yol söyle.",
         "Hep kapının yanındaki aynı kaseye koy. Ya da her bıraktığında nereye koyduğunu yüksek sesle söyle."),
        ("İkincisini kısaltır mısın? Kendi kendime tekrar edebileceğim bir şey olsun.",
         "Şunu söyle: \"Anahtar rafta.\""),
    ],
}


def anchor(rng, lang):
    """One concrete detail (a name, an object, a number or a place) for the first user message."""
    kind = rng.choices(["name", "object", "number", "place"], weights=[3, 4, 3, 2])[0]
    pick = rng.choice(ANCHORS[lang][kind])
    if kind == "number":
        n = rng.choice([rng.randint(2, 20), rng.randint(20, 99), rng.choice([100, 150, 200, 250, 500])])
        pick = pick.format(n=n)
    return dict(kind=kind, text=pick)


def build_seeds(count, seed=40404, only=None, prefix=None):
    """Balanced across language, topic and task type; everything else is random. `only` restricts the language.

    Every seed also carries a concrete detail and a message form so first messages differ even when the topic repeats.
    `prefix` names the id batch; round 2 uses "r2" so ids never collide with the round-1 conversations.
    """
    rng = random.Random(seed)
    cells = [
        (lang, topic, task)
        for lang in LANGUAGES
        if only in (None, lang)
        for topic in TOPICS
        for task in TASKS
    ]
    rng.shuffle(cells)
    prefix = only or "" if prefix is None else prefix
    out = []
    for i in range(count):
        lang, topic, task = cells[i % len(cells)]
        if i % len(cells) == len(cells) - 1:
            rng.shuffle(cells)
        out.append(
            dict(
                id=f"{prefix}{i:06d}",
                lang=lang,
                topic=topic,
                task=task,
                hint=rng.choice(TOPICS[topic]),
                persona=rng.choice(PERSONAS),
                style=rng.choice(STYLES),
                twist=rng.choice(TWISTS)
                if task == "change" or (task in ("practical", "polite") and rng.random() < 0.2)
                else None,
                turns=3 if rng.random() < 0.15 else 2,
                anchor=anchor(rng, lang) if rng.random() < 0.7 else None,
                opener=rng.choice(OPENERS),
                domain=rng.choice(LABEL_DOMAINS) if task == "label" else None,
                format=rng.choice(LABEL_FORMATS) if task == "label" else None,
            )
        )
    return out


def build_prompt(seed):
    language, note = LANGUAGES[seed["lang"]]
    task = TASKS[seed["task"]].format(twist=seed["twist"] or "")
    extra = ""
    if seed["twist"] and seed["task"] != "change":
        extra = f"\nOptional detail to weave in if it fits: {seed['twist']}."
    detail = ""
    if seed.get("anchor") and seed["task"] in ("practical", "change", "clarify", "refer"):
        detail += f"\nWork this concrete detail naturally into the first user message (change its wording to fit the sentence): {seed['anchor']['text']}."
    if seed.get("opener") and seed["task"] in ("practical", "change", "refer"):
        detail += f"\nThe first user message is {seed['opener']}."
    if seed.get("domain"):
        detail += f"\nQuestion domain (replaces the topic above): {seed['domain']}. Answer format: {seed['format']}. Pick an unusual question in that domain, not the first one that comes to mind."
    shown = [
        dict(user=u, assistant=a) for u, a in EXAMPLES[seed["lang"], seed["task"]]
    ]
    return f"""You write one realistic short chat between a user and a helpful assistant. The chat is training data for a small language model.

Language: {language}. {note}
Topic: {seed['topic']}. Situation: {seed['hint']}.
The user is {seed['persona']}. User writing style: {seed['style']}.
Chat type: {task}{extra}{detail}
Number of turns: {seed['turns']}. A turn is one user message followed by one assistant reply. Later turns continue naturally from the earlier ones.

Rules for assistant replies:
- One to three short sentences, at most 45 words. Simple, common words.
- Answer the question directly in the first sentence. No filler openers such as "Sure!" or "Great question!".
- Obey every instruction the user gives about the format or length of the answer.
- No lists, no markdown, no emojis. Never mention being an AI, a model or a company.
- Be accurate, practical and safe. Say only facts you are sure of; if unsure, pick a simpler fact or advice. For health topics give only everyday self-care and say when to see a doctor in a few words. No diagnoses or doses.
- If the request cannot be answered without a missing detail, ask one short question instead of guessing.
- In later turns, use the content of the earlier replies and the new information the user gave.
Rules for user messages: one or two sentences, at most 30 words, written the way this user would really type. Each chat must be specific and different from a typical example.

Output only valid JSON with exactly {seed['turns']} items, with no text before or after it. Every item is ONE object that holds both the "user" message and the "assistant" reply that answers it:
{{"turns": [{{"user": "...", "assistant": "..."}}]}}

Example of the form and the length (a different topic):
{json.dumps(dict(turns=shown), ensure_ascii=False)}"""


if __name__ == "__main__":
    seeds = build_seeds(24000)
    print(Counter((s["lang"], s["task"]) for s in seeds).most_common(3))
    print(build_prompt(seeds[1]))
