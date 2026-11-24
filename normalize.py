import re
import unicodedata

RELAXED_FORMS = {"person_name", "email_url", "address", "org_brand", "place", "technical_term"}

ARTIFACT_RE = re.compile(r"<\|[^|]*\|>|\[[^\]]{0,40}\]|[\u2047\ufffd\u3002\u3001\uff0c\u300c\u300d\u300a\u300b]"
                         r"|[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]")


def strip_artifacts(s):
    return ARTIFACT_RE.sub(" ", str(s))


ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9}
TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90}
SCALES = {"thousand": 1000, "lakh": 10 ** 5, "lakhs": 10 ** 5, "lac": 10 ** 5, "lacs": 10 ** 5, "million": 10 ** 6,
          "crore": 10 ** 7, "crores": 10 ** 7, "billion": 10 ** 9}
ORD_ONES = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
            "ninth": 9}
ORD_TEENS = {"tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
             "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19}
ORD_TENS = {"twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50, "sixtieth": 60, "seventieth": 70,
            "eightieth": 80, "ninetieth": 90}
NUMBER_WORDS = set(ONES) | set(TEENS) | set(TENS) | set(SCALES) | {"hundred"} | set(ORD_ONES) | set(ORD_TEENS) \
    | set(ORD_TENS)

def variant_key(s):
    return re.sub(r"[^a-z0-9@._/\-]", "", unicodedata.normalize("NFKD", str(s).lower()))


def same_entity(text, entity, form):
    raise NotImplementedError("scoring is map-only")
