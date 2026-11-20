"""Text normalization shared by training and inference."""
import re

# Kazakh-specific letters -> closest Russian letter (people often type without them)
KAZ = str.maketrans({"қ": "к", "ң": "н", "ғ": "г", "ү": "у", "ұ": "у", "ә": "а",
                     "і": "и", "ө": "о", "һ": "х", "ё": "е", "й": "и"})
# Latin -> Cyrillic, used only when the whole comment is written in Latin
LAT = str.maketrans({"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у",
                     "k": "к", "m": "м", "t": "т", "b": "б", "h": "х", "g": "г", "s": "с",
                     "i": "и", "l": "л", "n": "н", "d": "д", "z": "з", "q": "к", "j": "ж",
                     "u": "у", "r": "р", "v": "в", "f": "ф", "w": "в"})


def normalize(text: str) -> str:
    """Make spelling variants look the same.
    Example: "ҚОТАААҚ" -> "котак", "kotak bas" -> "котак бас"."""

    t = str(text).lower()
    t = t.translate(KAZ)
    if re.search(r"[a-z]", t) and not re.search(r"[а-я]", t):
        t = t.translate(LAT)
    t = re.sub(r"(.)\1{2,}", r"\1", t)   # "ааааа" -> "а"
    t = re.sub(r"\s+", " ", t).strip()
    return t


def group_key(text: str) -> str:
    """Key for near-duplicate grouping: normalized letters only (no digits, spaces, punctuation, emoji).
    Example: "Керемет!!! 👍" and "керемет" get the same key."""

    return re.sub(r"[^a-zа-я]", "", normalize(text))
