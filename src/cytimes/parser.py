# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from __future__ import annotations

# Cython imports
import cython
from cython.cimports.libc import math  # type: ignore
from cython.cimports.libc.stdlib import strtoll  # type: ignore
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cpython.time import localtime  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_READ_CHAR as str_read  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_GET_LENGTH as str_len  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_FromOrdinal as str_chr  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_Replace as str_replace  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_Contains as str_contains  # type: ignore
from cython.cimports.cpython.set import PySet_Add as set_add  # type: ignore
from cython.cimports.cpython.set import PySet_Discard as set_discard  # type: ignore
from cython.cimports.cpython.set import PySet_Contains as set_contains  # type: ignore
from cython.cimports.cpython.dict import PyDict_SetItem as dict_setitem  # type: ignore
from cython.cimports.cpython.dict import PyDict_GetItem as dict_getitem  # type: ignore
from cython.cimports.cpython.dict import PyDict_DelItem as dict_delitem  # type: ignore
from cython.cimports.cpython.dict import PyDict_Contains as dict_contains  # type: ignore
from cython.cimports.cpython.list import PyList_GET_SIZE as list_len  # type: ignore
from cython.cimports.cpython.list import PyList_GET_ITEM as list_getitem  # type: ignore
from cython.cimports.cpython.list import PyList_SET_ITEM as list_setitem  # type: ignore
from cython.cimports.cytimes import typeref, utils  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
import datetime
from dateutil.parser._parser import parserinfo
from cytimes import typeref, utils, errors

__all__ = ["timelex", "parse", "parse_dtobj", "Configs", "Parser"]

# Constants -----------------------------------------------------------------------------------
# . default configs
# fmt: off
CONFIG_PERTAIN: set[str] = {"of"}
CONFIG_JUMP: set[str] = {
    " ", ".", ",", ";", "-", "/", "'",
    "at", "on", "and", "ad", "m", "t", "of",
    "st", "nd", "rd", "th", "年" ,"月", "日"
}
CONFIG_UTC: set[str] = {
    "utc", "gmt", "z"
}
CONFIG_TZ: dict[str, int] = {
    "utc": 0,          # UTC
    "gmt": 0,          # Greenwich Mean Time
}
CONFIG_MONTH: dict[str, int] = {
    # EN(a)   # EN             # DE            # FR            # IT            # ES             # PT            # NL            # SE            #PL                 # TR          # CN       # Special
    "jan": 1,  "january": 1,   "januar": 1,    "janvier": 1,   "gennaio": 1,   "enero": 1,      "janeiro": 1,   "januari": 1,   "januari": 1,   "stycznia": 1,      "ocak": 1,    "一月": 1,
    "feb": 2,  "february": 2,  "februar": 2,   "février": 2,   "febbraio": 2,  "febrero": 2,    "fevereiro": 2, "februari": 2,  "februari": 2,  "lutego": 2,        "şubat": 2,   "二月": 2,  "febr": 2,
    "mar": 3,  "march": 3,     "märz": 3,      "mars": 3,      "marzo": 3,     "marzo": 3,      "março": 3,     "maart": 3,     "mars": 3,      "marca": 3,         "mart": 3,    "三月": 3,
    "apr": 4,  "april": 4,     "april": 4,     "avril": 4,     "aprile": 4,    "abril": 4,      "abril": 4,     "april": 4,     "april": 4,     "kwietnia": 4,      "nisan": 4,   "四月": 4,
    "may": 5,  "may": 5,       "mai": 5,       "mai": 5,       "maggio": 5,    "mayo": 5,       "maio": 5,      "mei": 5,       "maj": 5,       "maja": 5,          "mayıs": 5,   "五月": 5,
    "jun": 6,  "june": 6,      "juni": 6,      "juin": 6,      "giugno": 6,    "junio": 6,      "junho": 6,     "juni": 6,      "juni": 6,      "czerwca": 6,       "haziran": 5, "六月": 6,
    "jul": 7,  "july": 7,      "juli": 7,      "juillet": 7,   "luglio": 7,    "julio": 7,      "julho": 7,     "juli": 7,      "juli": 7,      "lipca": 7,         "temmuz": 7,  "七月": 7,
    "aug": 8,  "august": 8,    "august": 8,    "août": 8,      "agosto": 8,    "agosto": 8,     "agosto": 8,    "augustus": 8,  "augusti": 8,   "sierpnia": 8,      "ağustos": 8, "八月": 8,
    "sep": 9,  "september": 9, "september": 9, "septembre": 9, "settembre": 9, "septiembre": 9, "setembro": 9,  "september": 9, "september": 9, "września": 9,      "eylül": 9,   "九月": 9,  "sept": 9,
    "oct": 10, "october": 10,  "oktober": 10,  "octobre": 10,  "ottobre": 10,  "octubre": 10,   "outubro": 10,  "oktober": 10,  "oktober": 10,  "października": 10, "ekim": 10,   "十月": 10,
    "nov": 11, "november": 11, "november": 11, "novembre": 11, "novembre": 11, "noviembre": 11, "novembro": 11, "november": 11, "november": 11, "listopada": 11,    "kasım": 11,  "十一月": 11,
    "dec": 12, "december": 12, "dezember": 12, "décembre": 12, "dicembre": 12, "diciembre": 12, "dezembro": 12, "december": 12, "december": 12, "grudnia": 12,      "aralık": 12, "十二月": 12
}
CONFIG_WEEKDAY: dict[str, int] = {
    # EN(a)   # EN            # DE             # FR           # IT            # ES            # NL            # SE          # PL               # TR            # CN        # CN(a)
    "mon": 0, "monday": 0,    "montag": 0,     "lundi": 0,    "lunedì": 0,    "lunes": 0,     "maandag": 0,   "måndag": 0,  "poniedziałek": 0, "pazartesi": 0, "星期一": 0, "周一": 0,
    "tue": 1, "tuesday": 1,   "dienstag": 1,   "mardi": 1,    "martedì": 1,   "martes": 1,    "dinsdag": 1,   "tisdag": 1,  "wtorek": 1,       "salı": 1,      "星期二": 1, "周二": 1,
    "wed": 2, "wednesday": 2, "mittwoch": 2,   "mercredi": 2, "mercoledì": 2, "miércoles": 2, "woensdag": 2,  "onsdag": 2,  "środa": 2,        "çarşamba": 2,  "星期三": 2, "周三": 2,
    "thu": 3, "thursday": 3,  "donnerstag": 3, "jeudi": 3,    "giovedì": 3,   "jueves": 3,    "donderdag": 3, "torsdag": 3, "czwartek": 3,     "perşembe": 3,  "星期四": 3, "周四": 3,
    "fri": 4, "friday": 4,    "freitag": 4,    "vendredi": 4, "venerdì": 4,   "viernes": 4,   "vrijdag": 4,   "fredag": 4,  "piątek": 4,       "cuma": 4,      "星期五": 4, "周五": 4,
    "sat": 5, "saturday": 5,  "samstag": 5,    "samedi": 5,   "sabato": 5,    "sábado": 5,    "zaterdag": 5,  "lördag": 5,  "sobota": 5,       "cumartesi": 5, "星期六": 5, "周六": 5,
    "sun": 6, "sunday": 6,    "sonntag": 6,    "dimanche": 6, "domenica": 6,  "domingo": 6,   "zondag": 6,    "söndag": 6,  "niedziela": 6,    "pazar": 6,     "星期日": 6, "周日": 6
}
CONFIG_HMS_FLAG: dict[str, int] = {
    # EN(a)   # EN         # # DE          # FR           IT            # ES           # PT           # NL           # SE           # PL          # TR            # CN
    "h": 0,   "hour": 0,    "stunde": 0,   "heure": 0,    "ora": 0,     "hora": 0,     "hora": 0,     "uur": 0,      "timme": 0,    "godzina": 0, "saat": 0,      "时": 0,
    "hr": 0,  "hours": 0,   "stunden": 0,  "heures": 0,   "ore": 0,     "horas": 0,    "horas": 0,    "uren": 0,     "timmar": 0,   "godziny": 0, "saatler": 0,   "小时": 0,
    "m": 1,   "minute": 1,  "minute": 1,   "minute": 1,   "minuto": 1,  "minuto": 1,   "minuto": 1,   "minuut": 1,   "minut": 1,    "minuta": 1,  "dakika": 1,    "分": 1,
    "min": 1, "minutes": 1, "minuten": 1,  "minutes": 1,  "minuti": 1,  "minutos": 1,  "minutos": 1,  "minuten": 1,  "minuter": 1,  "minuty": 1,  "dakikalar": 1, "分钟": 1,
    "s": 2,   "second": 2,  "sekunde": 2,  "seconde": 2,  "secondo": 2, "segundo": 2,  "segundo": 2,  "seconde": 2,  "sekund": 2,   "sekunda": 2, "saniye": 2,    "秒": 2,
    "sec": 2, "seconds": 2, "sekunden": 2, "secondes": 2, "secondi": 2, "segundos": 2, "segundos": 2, "seconden": 2, "sekunder": 2, "sekundy": 2, "saniyeler": 2,
                                                                                                                                    "godzin": 0,                                           
}
CONFIG_AMPM_FLAG: dict[str, int] = {
    # EN(a)  # EN(a)  #EN             # DE             # IT             # ES         # PT        # NL          # SE              # PL             # TR          # CN
    "a": 0,  "am": 0, "morning": 0,   "morgen": 0,     "mattina": 0,    "mañana": 0, "manhã": 0, "ochtend": 0, "morgon": 0,      "rano": 0,       "sabah": 0,   "上午": 0,
    "p": 1,  "pm": 1, "afternoon": 1, "nachmittag": 1, "pomeriggio": 1, "tarde": 1,  "tarde": 1, "middag": 1,  "eftermiddag": 1, "popołudnie": 1, "öğleden": 1, "下午": 1
}
# fmt: on


# Timelex -------------------------------------------------------------------------------------
@cython.ccall
@cython.wraparound(True)
def timelex(dtstr: str) -> list[str]:
    """This function breaks the time string into lexical units (tokens).
    Lexical units are demarcated by changes in the character set, so any
    continuous string of letters or number is considered one unit `<'list[str]'>`.
    """
    return _timelex(dtstr, 0, 0)


@cython.cfunc
@cython.inline(True)
@cython.wraparound(True)
def _timelex(
    dtstr: str,
    start: cython.Py_ssize_t = 0,
    size: cython.Py_ssize_t = 0,
) -> list[str]:
    """(cfunc) This function breaks the time string into lexical units
    (tokens). Lexical units are demarcated by changes in the character
    set, so any continuous string of letters or number is considered
    one unit `<'list[str]'>`.

    :param dtstr `<'str'>`: The datetime string to be broken into lexical tokens.
    :param start `<'int'>`: The starting index for the 'dtstr' string, defaults to `0`.
    :param size `<'int'>`: The size of the 'dtstr' string, defaults to `0`.
        If 'size <= 0', the function measure the size of the 'dtstr' string internal.
    """
    # Validate index & size
    if size <= 0:
        size = str_len(dtstr)
    if start < 0:
        start = 0
    if size == 0 or start >= size:
        return []  # exit: eof

    # Setup
    tokens: list[str] = []
    idx: cython.Py_ssize_t = start
    cur_ch: cython.Py_UCS4
    tmp_ch: cython.Py_UCS4
    while idx < size and (tmp_ch := str_read(dtstr, idx)) in (" ", 0):
        idx += 1

    # Main string loop -------------------------------------------------------
    while idx < size:
        token: str = None
        state: cython.int = 0

        # Nested token loop - - - - - - - - - - - - - - - - - - - - - - - - -
        while idx < size:
            # Read new charactor for the current token
            if tmp_ch == 0:
                idx += 1
                if idx == size:
                    # Reached end of the string:
                    # 1. exit nested token loop
                    # 2. main loop also ends
                    break
                while idx < size and (cur_ch := str_read(dtstr, idx)) == 0:  # null
                    idx += 1
                if cur_ch == 0:
                    # No more valid charactors:
                    # 1. exit nested token loop
                    # 2. main loop also ends
                    break
            # Use cached charactor for the current token
            else:
                cur_ch, tmp_ch = tmp_ch, 0

            # . state 0: start of token
            if state == 0:
                token = str_chr(cur_ch)
                if cur_ch.isalpha():
                    state = 1  # alpha token
                elif cur_ch.isdigit():
                    state = 2  # digit token
                else:
                    # exit nested token loop:
                    # "single charactor token"
                    break

            # . state 1: alpha token
            elif state == 1:
                if cur_ch.isalpha():
                    token += str_chr(cur_ch)
                elif cur_ch == ".":
                    token += "."
                    state = 3  # alpha[.]
                else:
                    # 1. cache for the next token
                    # 2. exit nested token loop
                    tmp_ch = cur_ch
                    break

            # . state 2: digit token
            elif state == 2:
                if cur_ch.isdigit():
                    token += str_chr(cur_ch)
                elif cur_ch == ".":
                    token += "."
                    state = 4  # digit[.]
                elif cur_ch == "," and str_len(token) >= 2:
                    token += ","
                    state = 4  # digit[,]
                else:
                    # 1. cache for the next token
                    # 2. exit nested token loop
                    tmp_ch = cur_ch
                    break

            # . state 3: alpha[.]
            elif state == 3:
                if cur_ch == ".":
                    token += "."
                elif cur_ch.isalpha():
                    token += str_chr(cur_ch)
                elif cur_ch.isdigit() and token[-1] == ".":
                    token += str_chr(cur_ch)
                    state = 4  # digit[.]
                else:
                    # 1. cache for the next token
                    # 2. exit nested token loop
                    tmp_ch = cur_ch
                    break

            # . state 4: digit[.,]
            elif state == 4:
                if cur_ch == ".":
                    token += "."
                elif cur_ch.isdigit():
                    token += str_chr(cur_ch)
                elif cur_ch.isalpha() and token[-1] == ".":
                    token += str_chr(cur_ch)
                    state = 3  # alpha[.]
                else:
                    # 1. cache for the next token
                    # 2. exit nested token loop
                    tmp_ch = cur_ch
                    break
            # Nested token loop ----------------------------------------------

        # Further handle token contains "." / ","
        if state in (3, 4):
            # . token with extra "." / ","
            if state == 3 or token[-1] in ".," or utils.str_count(token, ".") > 1:
                tok: str = None
                for i in range(str_len(token)):
                    ch: cython.Py_UCS4 = str_read(token, i)
                    if ch == ".":
                        if tok is not None:
                            tokens.append(tok)
                            tok = None
                        tokens.append(".")
                    elif ch == ",":
                        if tok is not None:
                            tokens.append(tok)
                            tok = None
                        tokens.append(",")
                    elif tok is not None:
                        tok += str_chr(ch)
                    else:
                        tok = str_chr(ch)
                if tok is not None:
                    tokens.append(tok)
            # . digit token only contains ","
            else:
                if state == 4 and not str_contains(token, "."):
                    token = str_replace(token, ",", ".", -1)
                tokens.append(token)

        # Complete token
        elif token is not None:
            tokens.append(token)

        # None means: eof
        else:
            break
        # Main tokens loop ---------------------------------------------------

    # Return the time lexical tokens
    return tokens


# Configs -------------------------------------------------------------------------------------
@cython.cclass
class Configs:
    """Represents the configuration for the Parser."""

    # Settings
    _year1st: cython.bint
    _day1st: cython.bint
    _pertain: set[str]
    _jump: set[str]
    _utc: set[str]
    _tz: dict[str, int]
    _month: dict[str, int]
    _weekday: dict[str, int]
    _hms_flag: dict[str, int]
    _ampm_flag: dict[str, int]
    # Internal
    _words: set[str]

    def __init__(
        self,
        year1st: cython.bint = False,
        day1st: cython.bint = False,
    ) -> None:
        """The configuration for the Parser.

        :param year1st `<'bool'>`: Interpret the first ambiguous Y/M/D value as year, defaults to `False`.
        :param day1st `<'bool'>`: Interpret the first ambiguous Y/M/D values as day, defaults to `False`.

        ### Ambiguous Y/M/D
        Both the 'year1st' & 'day1st' arguments works together to determine how
        to interpret ambiguous Y/M/D values. The 'year1st' argument has higher
        priority than the 'day1st' argument.

        #### In case when all three values are ambiguous (e.g. `01/05/09`):
        - If 'year1st=False' & 'day1st=False', interprets as: `2009-01-05` (M/D/Y).
        - If 'year1st=False' & 'day1st=True', interprets as: `2009-05-01` (D/M/Y).
        - If 'year1st=True' & 'day1st=False', interprets as: `2001-05-09` (Y/M/D).
        - If 'year1st=True' & 'day1st=True', interprets as: `2001-09-05` (Y/D/M).

        #### In case when the 'year' value is known (e.g. `32/01/05`):
        - If 'day1st=False', interpretes as: `2032-01-05` (Y/M/D).
        - If 'day1st=True', interpretes as: `2032-05-01` (Y/D/M).

        #### In case when only one value is ambiguous (e.g. `32/01/20`):
        - The Parser should automatically figure out the correct Y/M/D order,
          and both 'year1st' & 'day1st' arguments are ignored.

        ### Settings
        Configs provides 'add_*()', 'rem_*()' and 'set_*()' methods to modify the following settings:
        - pertain: Words to recognize as pertain, e.g: `'of'`
        - jump: Words that should be skipped, e.g: `'and'`, `'at'`, `'on'`
        - utc: Words to recognize as UTC timezone, e.g: `'utc'`, `'gmt'`
        - tz: Words to recognize as timezone and the corresponding utc offset in seconds, e.g: `'est'`, `'pst'`
        - month: Words to recognize as month, e.g: `'january'`, `'february'`
        - weekday: Words to recognize as weekday, e.g: `'monday'`, `'tuesday'`
        - hms_flag: Words to recognize as as H/M/S flag, e.g: `'hour'`, `'minute'`
        - ampm_flag: Words to recognizes as AM/PM flag, e.g: `'am'`, `'pm'`
        """
        # Settings
        self._year1st = year1st
        self._day1st = day1st
        self._pertain = CONFIG_PERTAIN
        self._jump = CONFIG_JUMP
        self._utc = CONFIG_UTC
        self._tz = CONFIG_TZ
        self._month = CONFIG_MONTH
        self._weekday = CONFIG_WEEKDAY
        self._hms_flag = CONFIG_HMS_FLAG
        self._ampm_flag = CONFIG_AMPM_FLAG
        # Keywords
        self._construct()

    @classmethod
    def from_parserinfo(cls, info: parserinfo) -> Configs:
        """Create 'Configs' from an existing 'dateutil.parser.parserinfo'.

        ### Example
        >>> from cytimes import Configs
        >>> from dateutil.parser import parserinfo
        >>> info = parserinfo()
        >>> cfg = Configs.from_parserinfo(info)
        """
        # Validate perserinfo
        if not isinstance(info, typeref.PARSERINFO):
            raise errors.InvalidParserInfo(
                "<'%s'> Support import from <'dateutil.parser.parserinfo'>, "
                "instead got %s." % (cls.__name__, type(info))
            )

        # Import settings
        cfg = Configs(year1st=info.yearfirst, day1st=info.dayfirst)
        cfg._pertain = set(info.PERTAIN)
        cfg._jump = set(info.JUMP)
        cfg._utc = set(info.UTCZONE)
        cfg._tz = info.TZOFFSET
        cfg._month = {w: i + 1 for i, wds in enumerate(info.MONTHS) for w in wds}
        cfg._weekday = {w: i for i, wds in enumerate(info.WEEKDAYS) for w in wds}
        cfg._hms_flag = {w: i for i, wds in enumerate(info.HMS) for w in wds}
        cfg._ampm_flag = {w: i for i, wds in enumerate(info.AMPM) for w in wds}
        # Reconstruct words
        cfg._construct()
        # Return Configs
        return cfg

    # Property --------------------------------------------------------
    @property
    def year1st(self) -> bool:
        """Whether to interpret the first ambiguous
        Y/M/D values as year `<'bool'>`."""
        return self._year1st

    @property
    def day1st(self) -> bool:
        """Whether to interpret the first ambiguous
        Y/M/D values as day `<'bool'>`."""
        return self._day1st

    # Pertain ---------------------------------------------------------
    @property
    def pertain(self) -> set[str]:
        """The words to recognize as pertain `<'set[str]'>`."""
        return self._pertain

    def add_pertain(self, *words: str) -> None:
        """Add words that should be recognized as pertain.

        ### Example
        >>> cfg.add_pertain("of", ...)
        """
        for word in words:
            set_add(self._pertain, self._validate_word("pertain", word))

    def rem_pertain(self, *words: str) -> None:
        """Remove existing pertain words from Configs.

        ### Example
        >>> cfg.rem_pertain("of", ...)
        """
        for word in words:
            set_discard(self._pertain, word)
            set_discard(self._words, word)

    def set_pertain(self, *words: str) -> None:
        """Set the words that should be recognized as pertain.

        - If 'words' is given, replace all existing pertain words
          with the specified 'words'.
        - If 'words' is empty, resets to the default pertain words.

        ### Example
        >>> cfg.set_pertain("of", ...)
        """
        if words:
            self._pertain = set(words)
        else:
            self._pertain = CONFIG_PERTAIN
        self._construct()

    # Jump ------------------------------------------------------------
    @property
    def jump(self) -> set[str]:
        """The words that should be skipped `<'set[str]'>`."""
        return self._jump

    def add_jump(self, *words: str) -> None:
        """Add words that should be skipped.

        ### Example
        >>> cfg.add_jump("at", "on", ...)
        """
        for word in words:
            set_add(self._jump, self._validate_word("jump", word))

    def rem_jump(self, *words: str) -> None:
        """Remove jump words from the Configs.

        ### Example
        >>> cfg.rem_jump("at", "on", ...)
        """
        for word in words:
            set_discard(self._jump, word)
            set_discard(self._words, word)

    def set_jump(self, *words: str) -> None:
        """Set the words that should be skipped.

        - If 'words' is given, replace all existing jump words
          with the specified 'words'.
        - If 'words' is empty, resets to the default jump words.

        ### Example
        >>> cfg.set_jump("at", "on", ...)
        """
        if words:
            self._jump = set(words)
        else:
            self._jump = CONFIG_JUMP
        self._construct()

    # UTC -------------------------------------------------------------
    @property
    def utc(self) -> set[str]:
        """The words to recognize as UTC timezone `<'set[str]'>`."""
        return self._utc

    def add_utc(self, *words: str) -> None:
        """Add words that should be recognized as UTC timezone.

        ### Example
        >>> cfg.add_utc("utc", "gmt", "z", ...)
        """
        for word in words:
            set_add(self._utc, self._validate_word("utc", word))

    def rem_utc(self, *words: str) -> None:
        """Remove UTC timezone words from the Configs.

        ### Example
        >>> cfg.rem_utc("utc", "gmt", "z", ...)
        """
        for word in words:
            set_discard(self._utc, word)
            set_discard(self._words, word)

    def set_utc(self, *words: str) -> None:
        """Set the words that should be recognized as UTC timezone.

        - If 'words' is given, replace all existing UTC timezone words
          with the specified 'words'.
        - If 'words' is empty, resets to the default UTC timezone words.

        ### Example
        >>> cfg.set_utc("utc", "gmt", "z", ...)
        """
        if words:
            self._utc = set(words)
        else:
            self._utc = CONFIG_UTC
        self._construct()

    # Timezone --------------------------------------------------------
    @property
    def tz(self) -> dict[str, int]:
        """The words to recognize as a timezone `<'dict[str, int]'>`

        Where keys are the timezone words and values are the
        corresponding utc offset in seconds.
        """
        return self._tz

    def add_tz(
        self,
        word: str,
        hour: int = 0,
        minute: int = 0,
        seconds: int = 0,
    ) -> None:
        """Add word that should be recognized as a timezone
        with the corresponding utcoffset offset.

        ### Example
        >>> cfg.add_tz("est", hour=-5)
        """
        dict_setitem(
            self._tz,
            self._validate_word("tz", word),
            self._validate_value_tzoffset(hour * 3_600 + minute * 60 + seconds),
        )

    def rem_tz(self, *words: str) -> None:
        """Remove timezone words from the Configs.

        ### Example
        >>> cfg.rem_tz("est", "edt", ...
        """
        for word in words:
            try:
                dict_delitem(self._tz, word)
            except KeyError:
                pass
            set_discard(self._words, word)

    def set_tz(self, **words_and_offsets: int) -> None:
        """Set the words that should be recognized as timezone
        and the corresponding utcoffsets.

        - If 'words_and_offsets' is given, replace all existing timezone words
          with the specified 'words_and_offsets'.
        - If 'words_and_offsets' is empty, resets to the default timezone words.

        ### Example
        >>> cfg.set_tz(**{"est": -18000, "edt": -14400, ... })
        """
        if words_and_offsets:
            self._tz = words_and_offsets
        else:
            self._tz = CONFIG_TZ
        self._construct()

    # Month -----------------------------------------------------------
    @property
    def month(self) -> dict[str, int]:
        """The words to recognize as month `<'dict[str, int]'>`.

        Where keys are the words and values are the corresonding
        month number: 1(Jan)...12(Dec).
        """
        return self._month

    def add_month(self, month: int, *words: str) -> None:
        """Add words that should be recognized as one specific
        month: 1(Jan)...12(Dec).

        ### Example
        >>> cfg.add_month(1, "jan", "january", ...)
        """
        month = self._validate_value_month(month)
        for word in words:
            dict_setitem(self._month, self._validate_word("month", word), month)

    def rem_month(self, *words: str) -> None:
        """Remove month words from the Configs.

        ### Example
        >>> cfg.rem_month("jan", "january", ...)
        """
        for word in words:
            try:
                dict_delitem(self._month, word)
            except KeyError:
                pass
            set_discard(self._words, word)

    def set_month(self, **words_and_months: int) -> None:
        """Set the words that should be recognized as month.

        - If 'words_and_months' is given, replace all existing month words
          with the specified 'words_and_months'.
        - If 'words_and_months' is empty, resets to the default month words.

        ### Example
        >>> cfg.set_month(
                **{
                    "jan": 1, "january": 1,
                    "feb": 2, "february": 2,
                    ...
                }
            )
        """
        if words_and_months:
            self._month = words_and_months
        else:
            self._month = CONFIG_MONTH
        self._construct()

    # Weekday ---------------------------------------------------------
    @property
    def weekday(self) -> dict[str, int]:
        """The words to recognizes as weekday `<'dict[str, int]'>`.

        Where keys are the words and values are the corresponding
        weekday number: 0(Monday)...6(Sunday).
        """
        return self._weekday

    def add_weekday(self, weekday: int, *words: str) -> None:
        """Add words that should be recognized as one specific
        weekday: 0(Monday)...6(Sunday).

        ### Example
        >>> cfg.add_weekday(0, "mon", "monday", ...)
        """
        weekday = self._validate_value_weekday(weekday)
        for word in words:
            dict_setitem(self._weekday, self._validate_word("weekday", word), weekday)

    def rem_weekday(self, *words: str) -> None:
        """Remove weekday words from the Configs.

        ### Example
        >>> cfg.rem_weekday("mon", "monday", ...)
        """
        for word in words:
            try:
                dict_delitem(self._weekday, word)
            except KeyError:
                pass
            set_discard(self._words, word)

    def set_weekday(self, **words_and_weekdays: int) -> None:
        """Set the words that should be recognized as weekday.

        - If 'words_and_weekdays' is given, replace all existing weekday words
          with the specified 'words_and_weekdays'.
        - If 'words_and_weekdays' is empty, resets to the default weekday words.

        ### Example
        >>> cfg.set_weekday(
                **{
                    "mon": 0, "monday": 0,
                    "tue": 1, "tuesday": 1,
                    ...
                }
            )
        """
        if words_and_weekdays:
            self._weekday = words_and_weekdays
        else:
            self._weekday = CONFIG_WEEKDAY
        self._construct()

    # HMS flag -------------------------------------------------------
    @property
    def hms_flag(self) -> dict[str, int]:
        """The words to recognize as H/M/S flag `<'dict[str, int]'>`.

        Where keys are the words and values are the corresponding
        flag number: 0(hour), 1(minute), 2(second).
        """
        return self._hms_flag

    def add_hms_flag(self, hms_flag: int, *words: str) -> None:
        """Add words that should be recognized as one specific
        H/M/S flag: 0(hour), 1(minute), 2(second).

        ### Example
        >>> cfg.add_hms_flag(0, "h", "hour", "hours", ...)
        """
        hms_flag = self._validate_value_hms_flag(hms_flag)
        for word in words:
            dict_setitem(
                self._hms_flag, self._validate_word("hms_flag", word), hms_flag
            )

    def rem_hms_flag(self, *words: str) -> None:
        """Remove H/M/S flag words from the Configs.

        ### Example
        >>> cfg.rem_hms_flag("h", "hour", "hours", ...)
        """
        for word in words:
            try:
                dict_delitem(self._hms_flag, word)
            except KeyError:
                pass
            set_discard(self._words, word)

    def set_hms_flag(self, **words_and_hmsflags: int) -> None:
        """Set the words that should be recognized as H/M/S flag.

        - If 'words_and_hmsflags' is given, replace all existing H/M/S flag words
          with the specified 'words_and_hmsflags'.
        - If 'words_and_hmsflags' is empty, resets to the default H/M/S flag words.

        ### Example
        >>> cfg.set_hms_flag(
                **{
                    "h": 0, "hour": 0, "hours": 0, ...
                    "m": 1, "minute": 1, "minutes": 1, ...
                    "s": 2, "second": 2, "seconds": 2, ...
                }
            )
        """
        if words_and_hmsflags:
            self._hms_flag = words_and_hmsflags
        else:
            self._hms_flag = CONFIG_HMS_FLAG
        self._construct()

    # AM/PM flag -----------------------------------------------------
    @property
    def ampm_flag(self) -> dict[str, int]:
        """The words to recognize as AM/PM flag `<'dict[str, int]'>`.

        Where keys are the words and values are the corresponding
        flag number: 0(AM), 1(PM).
        """
        return self._ampm_flag

    def add_ampm_flag(self, ampm_flag: int, *words: str) -> None:
        """Add words that should be recognized as one specific
        AM/PM flag: 0(AM), 1(PM).

        ### Example
        >>> cfg.add_ampm(0, "am", "a.m.", ...)
        """
        ampm_flag = self._validate_value_ampm_flag(ampm_flag)
        for word in words:
            dict_setitem(
                self._ampm_flag, self._validate_word("ampm_flag", word), ampm_flag
            )

    def rem_ampm_flag(self, *words: str) -> None:
        """Remove AM/PM flag words from the Configs.

        ### Example
        >>> cfg.rem_ampm("am", "a.m.", ...)
        """
        for word in words:
            try:
                dict_delitem(self._ampm_flag, word)
            except KeyError:
                pass
            set_discard(self._words, word)

    def set_ampm(self, **words_and_ampmflags: int) -> None:
        """Set the words that should be recognized as AM/PM flag.

        - If 'words_and_ampm' is given, replace all existing AM/PM flag words
          with the specified 'words_and_ampm'.
        - If 'words_and_ampm' is empty, resets to the default AM/PM flag words.

        ### Example
        >>> cfg.set_ampm(
                **{
                    "am": 0, "a.m.": 0, ...
                    "pm": 1, "p.m.": 1, ...
                }
            )
        """
        if words_and_ampmflags:
            self._ampm_flag = words_and_ampmflags
        else:
            self._ampm_flag = CONFIG_AMPM_FLAG
        self._construct()

    # Internal --------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _construct(self) -> cython.bint:
        """(cfunc) Construct the the Configs."""
        # Reset words
        self._words = set()
        # fmt: off
        # . month
        self._month = {
            self._validate_word("month", word):
            self._validate_value_month(value)
            for word, value in self._month.items()
        }
        # . weekday
        self._weekday = {
            self._validate_word("weekday", word):
            self._validate_value_weekday(value)
            for word, value in self._weekday.items()
        }
        # . hms flag
        self._hms_flag = {
            self._validate_word("hms_flag", word):
            self._validate_value_hms_flag(value)
            for word, value in self._hms_flag.items()
        }
        # . ampm flag
        self._ampm_flag = {
            self._validate_word("ampm_flag", word):
            self._validate_value_ampm_flag(value)
            for word, value in self._ampm_flag.items()
        }
        # . utc
        self._utc = {
            self._validate_word("utc", word)
            for word in self._utc }
        # . timezone
        self._tz = {
            self._validate_word("tz", word):
            self._validate_value_tzoffset(value)
            for word, value in self._tz.items()
        }
        # . pertain
        self._pertain = {
            self._validate_word("pertain", word)
            for word in self._pertain }
        # . jump
        self._jump = {
            self._validate_word("jump", word)
            for word in self._jump }
        # fmt: on
        # Finished
        return True

    @cython.cfunc
    @cython.inline(True)
    def _validate_word(self, setting: str, word: object) -> str:
        """(cfunc) Validate if the 'word' conflicts with
        exsiting words in the Configs `<'str'>`."""
        # Validate type
        if type(word) is not str:
            raise errors.InvalidConfigsValue(
                "<'%s'> The word for 'Configs.%s' must be a string, instead got %s."
                % (self.__class__.__name__, setting, type(word))
            )
        w: str = word
        w = w.lower()

        # Check if the word is conflicting with other words
        if set_contains(self._words, w):
            conflict: str = None
            # . exclude jump words, since it has the
            # . lowest priority and maximum freedom
            if set_contains(self._jump, w):
                pass
            # . month
            elif dict_contains(self._month, w):
                if setting != "month":
                    conflict = "month"
            # . weekday
            elif dict_contains(self._weekday, w):
                if setting != "weekday":
                    conflict = "weekday"
            # . hms flag
            elif dict_contains(self._hms_flag, w):
                if setting != "hms_flag":
                    conflict = "hms_flag"
            # . ampm flag
            elif dict_contains(self._ampm_flag, w):
                if setting != "ampm_flag":
                    conflict = "ampm_flag"
            # . utc
            elif set_contains(self._utc, w):
                if setting != "utc" and setting != "tz":
                    conflict = "utc"
            # . timezone
            elif dict_contains(self._tz, w):
                if setting != "tz":
                    conflict = "tz"
            # . pertain
            elif set_contains(self._pertain, w):
                if setting != "pertain":
                    conflict = "pertain"
            # . raise error
            if conflict is not None:
                raise errors.InvalidConfigsValue(
                    "<'%s'> The word '%s' for 'Configs.%s' conflicts "
                    "with exsiting words in 'Configs.%s'."
                    % (self.__class__.__name__, w, setting, conflict)
                )
        else:
            set_add(self._words, w)

        # Return the word
        return w

    @cython.cfunc
    @cython.inline(True)
    def _validate_value(
        self,
        setting: str,
        value: object,
        min: cython.int,
        max: cython.int,
    ) -> object:
        """(cfunc) Validate if the 'value' of a word is in
        the valid range (min...max) `<'int'>`."""
        if type(value) is not int:
            raise errors.InvalidConfigsValue(
                "<'%s'> The value for 'Configs.%s' must be an integer, instead got %s."
                % (self.__class__.__name__, setting, type(value))
            )
        v: cython.longlong = value
        if not min <= v <= max:
            raise errors.InvalidConfigsValue(
                "<'%s'> The value for 'Configs.%s' must between %d...%d, instead got %d."
                % (self.__class__.__name__, setting, min, max, v)
            )
        return value

    @cython.cfunc
    @cython.inline(True)
    def _validate_value_month(self, value: object) -> object:
        """(cfunc) Validate if the 'value' of a month is in
        the valid range (1...12) `<'int'>`."""
        return self._validate_value("month", value, 1, 12)

    @cython.cfunc
    @cython.inline(True)
    def _validate_value_weekday(self, value: object) -> object:
        """(cfunc) Validate if the 'value' of a weekday is in
        the valid range (0...6) `<'int'>`."""
        return self._validate_value("weekday", value, 0, 6)

    @cython.cfunc
    @cython.inline(True)
    def _validate_value_hms_flag(self, value: object) -> object:
        """(cfunc) Validate if the 'value' of a H/M/S flag is in
        the valid range (0...2) `<'int'>`."""
        return self._validate_value("hms_flag", value, 0, 2)

    @cython.cfunc
    @cython.inline(True)
    def _validate_value_ampm_flag(self, value: object) -> object:
        """(cfunc) Validate if the 'value' of a AM/PM flag is in
        the valid range (0...1) `<'int'>`."""
        return self._validate_value("ampm_flag", value, 0, 1)

    @cython.cfunc
    @cython.inline(True)
    def _validate_value_tzoffset(self, value: object) -> object:
        """(cfunc) Validate if the 'value' of a timezone offset is in
        the valid range (-86_340...86_340) `<'int'>`."""
        return self._validate_value("tz", value, -86_340, 86_340)

    # Special methods -------------------------------------------------
    def __repr__(self) -> str:
        reprs: list = [
            "*year1st* = %s" % self._year1st,
            "*day1st* = %s" % self._day1st,
            "*pertain* = %s" % sorted(self._pertain),
            "*jump* = %s" % sorted(self._jump),
            "*utc* = %s" % sorted(self._utc),
            "*tz* = %s" % self._tz,
            "*month* = %s" % self._month,
            "*weekday* = %s" % self._weekday,
            "*hms* = %s" % self._hms_flag,
            "*ampm* = %s" % self._ampm_flag,
        ]
        return "<%s (\n  %s\n)>" % (self.__class__.__name__, ",\n  ".join(reprs))


# Parser --------------------------------------------------------------------------------------
@cython.cclass
class Result:
    """Represents the datetime result from the Parser."""

    # Y/M/D
    _ymd: cython.int[3]
    _idx: cython.int
    _yidx: cython.int
    _midx: cython.int
    _didx: cython.int
    # Values
    year: cython.int
    month: cython.int
    day: cython.int
    weekday: cython.int
    hour: cython.int
    minute: cython.int
    second: cython.int
    microsecond: cython.int
    ampm: cython.int
    tzoffset: cython.int
    century_specified: cython.bint

    def __cinit__(self) -> None:
        """The datetime result from the Parser."""
        self.reset()

    # Y/M/D -----------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def set_ymd_int(self, value: cython.longlong, label: cython.int) -> cython.bint:
        """(cfunc) Set one Y/M/D value from an integer `<'bool'>`.

        :param value `<'int'>`: One of the Y/M/D value.
        :param label `<'int'>`: The label for the value:
            - label=0: unknown
            - label=1: year
            - label=2: month
            - label=3: day
        :returns `<'bool'`>: `False` if all slots (max 3) are fully populated.
        """
        # Y/M/D slots already fully populated
        if self._idx >= 2:
            return False

        # Validate Y/M/D 'value'
        if value > 9_999:
            raise ValueError("invalid Y/M/D value '%d'." % value)
        elif value > 31:
            label = 1
            if value > 99:
                self.century_specified = True
        elif value < 0:
            raise ValueError("invalid Y/M/D value '%d'." % value)

        # Set Y/M/D value & index
        return self._set_ymd(value, label)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def set_ymd_str(self, value: str, label: cython.int) -> cython.bint:
        """(cfunc) Set one Y/M/D value from a string `<'bool'>`.

        :param value `<'str'>`: One of the Y/M/D value.
        :param label `<'int'>`: The label for the value:
            - label=0: unknown
            - label=1: year
            - label=2: month
            - label=3: day
        :returns `<'bool'`>: `False` if all slots (max 3) are fully populated.
        """
        # Y/M/D slots already fully populated
        if self._idx >= 2:
            return False

        # Validate Y/M/D 'value'
        try:
            num: cython.longlong = int(value)
        except Exception as err:
            raise ValueError("invalid Y/M/D value '%s'." % value) from err
        if num > 9_999:
            raise ValueError("invalid Y/M/D value '%s'." % value)
        elif num > 31:
            label = 1
            if num > 99:
                self.century_specified = True
        elif num < 0:
            raise ValueError("invalid Y/M/D value '%s'." % value)
        elif str_len(value) > 2:
            self.century_specified = True
            label = 1

        # Set Y/M/D value & index
        return self._set_ymd(num, label)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_ymd(self, value: cython.int, label: cython.int) -> cython.bint:
        """(cfunc) Set one Y/M/D value & its index `<'bool'>`

        #### This method is for internal use only.

        :param value `<'int'>`: One of the Y/M/D value.
        :param label `<'int'>`: The label for the value:
            - label=0: unknown
            - label=1: year
            - label=2: month
            - label=3: day
        """
        # Set Y/M/D value
        self._idx += 1
        self._ymd[self._idx] = value
        if label == 0:
            return True

        # Set Y/M/D label index
        if label == 1:
            # . year index
            if self._yidx == -1:
                self._yidx = self._idx
        elif label == 2:
            # . month index
            if self._midx == -1:
                self._midx = self._idx
        elif label == 3:
            # . day index
            if self._didx == -1:
                self._didx = self._idx
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def ymd_populated(self) -> cython.int:
        """(cfunc) Get the number of Y/M/D slots that have already
        been populated with, expects value between 0...3 `<'int'>`.
        """
        return self._idx + 1

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def ymd_labeled(self) -> cython.int:
        """(cfunc) Get the number of Y/M/D slots that have already
        been labeled (solved), expects value between 0...3 `<'int'>`.
        """
        count: cython.int = 0
        if self._yidx != -1:
            count += 1
        if self._midx != -1:
            count += 1
        if self._didx != -1:
            count += 1
        return count

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def could_be_day(self, value: cython.longlong) -> cython.bint:
        """(cfunc) Determine if an integer could be the day,
        based on the current Y/M/D slot values `<'bool'>`.
        """
        # Day slot already populated
        if self._didx != -1:
            return False
        # Month is known & value is in range
        if self._midx == -1:
            return 1 <= value <= 31
        # Year is known & value is in range
        month = self._ymd[self._midx]
        if self._yidx == -1:
            return 1 <= value <= utils.days_in_month(2000, month)
        # Both Y/M are known & value is in range
        year = self._ymd[self._yidx]
        return 1 <= value <= utils.days_in_month(year, month)

    # Values ----------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def prepare(self, year1st: cython.bint, day1st: cython.bint) -> cython.bint:
        """(cfunc) Prepare the result (solve unlabeld and ambiguous Y/M/D).
        Returns `True` if the result is valid `<'bool'>`.

        #### Must call this method before accessing the result values.
        """
        populated: cython.int = self.ymd_populated()
        labeled: cython.int = self.ymd_labeled()
        yidx: cython.int = self._yidx
        midx: cython.int = self._midx
        didx: cython.int = self._didx
        v0: cython.int = self._ymd[0]
        v1: cython.int = self._ymd[1]
        v2: cython.int = self._ymd[2]

        # All Y/M/D have been solved already
        if populated == labeled > 0:
            self.year = self._ymd[yidx] if yidx != -1 else -1
            self.month = self._ymd[midx] if midx != -1 else -1
            self.day = self._ymd[didx] if didx != -1 else -1

        # Has only one Y/M/D value
        elif populated == 1:
            if midx != -1:  # month labeled
                self.month = v0
            elif v0 > 31:  # must be year
                self.year = v0
            else:  # probably day
                self.day = v0

        # Have two Y/M/D values
        elif populated == 2:
            # . month labeled
            if midx != -1:
                if midx == 0:
                    if v1 > 31 or (v0 == 2 and v1 > 29):
                        self.month, self.year = v0, v1
                    else:  # probably day: Jan-01
                        self.month, self.day = v0, v1
                else:
                    if v0 > 31 or (v1 == 2 and v0 > 29):
                        self.year, self.month = v0, v1
                    else:  # probably day: 01-Jan
                        self.day, self.month = v0, v1
            # . month not labeled
            elif v0 > 31 or (v1 == 2 and v0 > 29):  # 99-Feb
                self.year, self.month = v0, v1
            elif v1 > 31 or (v0 == 2 and v1 > 29):  # Feb-99
                self.month, self.year = v0, v1
            elif day1st and 1 <= v1 <= 12:  # 01-Jan
                self.day, self.month = v0, v1
            else:  # Jan-01
                self.month, self.day = v0, v1

        # Have three Y/M/D values
        elif populated == 3:
            # . lack one label
            if labeled == 2:
                if midx != -1:  # month labeled
                    self.month = self._ymd[midx]
                    if yidx != -1:  # year labeled
                        self.year = self._ymd[yidx]
                        self.day = self._ymd[3 - yidx - midx]
                    else:  # day labeled
                        self.day = self._ymd[didx]
                        self.year = self._ymd[3 - midx - didx]
                elif yidx != -1:  # year labeled
                    self.year = self._ymd[yidx]
                    if midx != -1:  # month labeled
                        self.month = self._ymd[midx]
                        self.day = self._ymd[3 - yidx - midx]
                    else:  # day labeled
                        self.day = self._ymd[didx]
                        self.month = self._ymd[3 - yidx - didx]
                else:  # day labeled
                    self.day = self._ymd[didx]
                    if yidx != -1:  # year labeled
                        self.year = self._ymd[yidx]
                        self.month = self._ymd[3 - yidx - didx]
                    else:  # month labeled
                        self.month = self._ymd[midx]
                        self.year = self._ymd[3 - midx - didx]
            # . lack more than one labels (guess)
            elif midx == 0:
                if v1 > 31:  # Apr-2003-25
                    self.month, self.year, self.day = v0, v1, v2
                else:  # Apr-25-2003
                    self.month, self.day, self.year = v0, v1, v2
            elif midx == 1:
                if v0 > 31 or (year1st and 0 < v2 <= 31):  # 99-Jan-01
                    self.year, self.month, self.day = v0, v1, v2
                else:  # 01-Jan-99
                    self.day, self.month, self.year = v0, v1, v2
            elif midx == 2:
                if v1 > 31:  # 01-99-Jan
                    self.day, self.year, self.month = v0, v1, v2
                else:  # 99-01-Jan
                    self.year, self.day, self.month = v0, v1, v2
            else:
                if v0 > 31 or yidx == 0 or (year1st and 0 < v1 <= 12 and 0 < v2 <= 31):
                    if day1st and 0 < v2 <= 12:  # 99-01-Jan
                        self.year, self.day, self.month = (v0, v1, v2)
                    else:  # 99-Jan-01
                        self.year, self.month, self.day = (v0, v1, v2)
                elif v0 > 12 or (day1st and 0 < v1 <= 12):  # 01-Jan-99
                    self.day, self.month, self.year = (v0, v1, v2)
                else:  # Jan-01-99
                    self.month, self.day, self.year = (v0, v1, v2)

        # Swap month & day (if necessary)
        if self.month > 12 and 1 <= self.day <= 12:
            self.month, self.day = self.day, self.month

        # Adjust year to current century (if necessary)
        if not self.century_specified and 0 <= self.year < 100:
            year_now: cython.int = localtime().tm_year
            year: cython.int = self.year + year_now // 100 * 100
            # . too far into the future
            if year >= year_now + 50:
                year -= 100
            # . too distance from the now
            elif year < year_now - 50:
                year += 100
            self.year = year

        # Check validity
        return self.valid()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def valid(self) -> cython.bint:
        """(cfunc) Check if the result is valid
        (contains any datetime values) `<'bool'>`.
        """
        return (
            self.year != -1
            or self.month != -1
            or self.day != -1
            or self.hour != -1
            or self.minute != -1
            or self.second != -1
            or self.microsecond != -1
            or self.weekday != -1
            or self.tzoffset != -100_000
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def reset(self) -> cython.bint:
        """(cfunc) Reset all results."""
        # Y/M/D
        self._ymd = [-1, -1, -1]
        self._idx = -1
        self._yidx = -1
        self._midx = -1
        self._didx = -1
        # Result
        self.year = -1
        self.month = -1
        self.day = -1
        self.weekday = -1
        self.hour = -1
        self.minute = -1
        self.second = -1
        self.microsecond = -1
        self.ampm = -1
        #: tzoffset must between -86340...86340,
        #: -100_000 represents no tzoffset.
        self.tzoffset = -100_000
        self.century_specified = False

    # Special methods -------------------------------------------------
    def __repr__(self) -> str:
        reprs: list[str] = []

        # Datetime results
        # . year
        if self.year != -1:
            reprs.append("year=%d" % self.year)
        elif self._yidx != -1:
            reprs.append("year=%d" % self._ymd[self._yidx])
        # . month
        if self.month != -1:
            reprs.append("month=%d" % self.month)
        elif self._midx != -1:
            reprs.append("month=%d" % self._ymd[self._midx])
        # . day
        if self.day != -1:
            reprs.append("day=%d" % self.day)
        elif self._didx != -1:
            reprs.append("day=%d" % self._ymd[self._didx])
        # . rest
        if self.weekday != -1:
            reprs.append("weekday=%d" % self.weekday)
        if self.hour != -1:
            reprs.append("hour=%d" % self.hour)
        if self.minute != -1:
            reprs.append("minute=%d" % self.minute)
        if self.second != -1:
            reprs.append("second=%d" % self.second)
        if self.microsecond != -1:
            reprs.append("microsecond=%d" % self.microsecond)
        if self.ampm != -1:
            reprs.append("ampm=%d" % self.ampm)
        if self.tzoffset != -100_000:
            reprs.append("tzoffset=%d" % self.tzoffset)

        # Construct
        return "<%s (%s)>" % (self.__class__.__name__, ", ".join(reprs))

    def __bool__(self) -> bool:
        return self.valid()


@cython.cclass
class Parser:
    """Represents the datetime Parser."""

    # Settings
    _ignoretz: cython.bint
    # Configs
    _year1st: cython.bint
    _day1st: cython.bint
    _pertain: set[str]
    _jump: set[str]
    _utc: set[str]
    _tz: dict[str, int]
    _month: dict[str, int]
    _weekday: dict[str, int]
    _hms_flag: dict[str, int]
    _ampm_flag: dict[str, int]
    # Result
    _res: Result
    # Process
    _idx: cython.Py_ssize_t
    _size: cython.Py_ssize_t
    _tokens: list[str]
    _tk1: str
    _tk2: str
    _tk3: str
    _tk4: str

    def __init__(self, cfg: Configs = None) -> None:
        """The datetime Parser.

        :param cfg `<'Configs/None'>`: The configuration for the Parser, defaults to `None`.

        For more information about `<'Configs'>`, please refer to the Configs class.
        """
        # Configs: default
        if cfg is None:
            self._year1st = False
            self._day1st = False
            self._pertain = CONFIG_PERTAIN
            self._jump = CONFIG_JUMP
            self._utc = CONFIG_UTC
            self._tz = CONFIG_TZ
            self._month = CONFIG_MONTH
            self._weekday = CONFIG_WEEKDAY
            self._hms_flag = CONFIG_HMS_FLAG
            self._ampm_flag = CONFIG_AMPM_FLAG
        # Configs: load
        else:
            self._year1st = cfg._year1st
            self._day1st = cfg._day1st
            self._pertain = cfg._pertain
            self._jump = cfg._jump
            self._utc = cfg._utc
            self._tz = cfg._tz
            self._month = cfg._month
            self._weekday = cfg._weekday
            self._hms_flag = cfg._hms_flag
            self._ampm_flag = cfg._ampm_flag
        # Result
        self._res = Result()

    # Parse --------------------------------------------------------------------------------
    @cython.ccall
    def parse(
        self,
        dtstr: str,
        default: datetime.date | datetime.datetime | None = None,
        year1st: bool | None = None,
        day1st: bool | None = None,
        ignoretz: cython.bint = False,
        isoformat: cython.bint = True,
    ) -> datetime.datetime:
        """Parse the datetime string into `<'datetime.datetime'>`.

        :param dtstr `<'str'>`: The string that contains datetime information.

        :param default `<'datetime/date/None'>`: Default value to fill in missing date fields, defaults to `None`.
            - `<'date/datetime'>` If the parser fails to extract Y/M/D from the string,
               use the passed-in 'default' to fill in the missing fields.
            - If `None`, raises `PaserBuildError` if any Y/M/D fields is missing.

        :param year1st `<'bool/None'>`: Interpret the first ambiguous Y/M/D value as year, defaults to `None`.
            If 'year1st=None', use `cfg.year1st` if 'cfg' is specified; otherwise, defaults to `False`.

        :param day1st `<'bool/None'>`: Interpret the first ambiguous Y/M/D values as day, defaults to `None`.
            If 'day1st=None', use `cfg.day1st` if 'cfg' is specified; otherwise, defaults to `False`.

        :param ignoretz `<'bool'>`: Whether to ignore timezone information, defaults to `False`.
            - `True`: Ignores any timezone information and returns a timezone-naive datetime (increases parser performance).
            - `False`: Processes timezone information and generates a timezone-aware datetime if matched by `cfg.utc` & `cfg.tz`.

        :param isoformat `<'bool'>`: Whether try to parse 'dtstr' as ISO format, defaults to `True`.
            - `True`: First tries to process 'dtstr' as ISO format; if failed, falls back
                to token parsing (best performance for most strings).
            - `False`: Only processes 'dtstr' through token parsing (increases performance
                if 'dtstr' is not ISO format).

        ## Ambiguous Y/M/D
        Both 'year1st' and 'day1st' determine how to interpret ambiguous
        Y/M/D values. 'year1st' has higher priority.

        #### When all three values are ambiguous (e.g. `01/05/09`):
        - If 'year1st=False' & 'day1st=False': interprets as `2009-01-05` (M/D/Y).
        - If 'year1st=False' & 'day1st=True': interprets as `2009-05-01` (D/M/Y).
        - If 'year1st=True' & 'day1st=False': interprets as `2001-05-09` (Y/M/D).
        - If 'year1st=True' & 'day1st=True': interprets as `2001-09-05` (Y/D/M).

        #### When the 'year' value is known (e.g. `32/01/05`):
        - If 'day1st=False': interpretes as `2032-01-05` (Y/M/D).
        - If 'day1st=True': interpretes as `2032-05-01` (Y/D/M).

        #### When only one value is ambiguous (e.g. `32/01/20`):
        - The parser automatically determines the correct order; 'year1st' and 'day1st' are ignored.
        """
        # Settings
        self._ignoretz = ignoretz

        # Process
        self._process(dtstr, isoformat)

        # Prepare
        if not self._res.prepare(
            self._year1st if year1st is None else bool(year1st),
            self._day1st if day1st is None else bool(day1st),
        ):
            raise errors.ParserFailedError(
                "<'%s'> Failed to parse: '%s'.\n"
                "Error: cannot recognize any tokens as datetime components."
                % (self.__class__.__name__, dtstr)
            )

        # Build
        return self._build(dtstr, default)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _process(self, dtstr: str, isoformat: cython.bint) -> cython.bint:
        """(cfunc) Process the datetime string.

        :raise `ParserFailedError`: if any error occurs during the process.
        """
        # Reset result & index & size
        self._res.reset()
        self._idx, self._size = 0, str_len(dtstr)

        # Process
        try:
            if isoformat:
                return self._process_iso_format(dtstr)
            else:
                return self._process_timelex_tokens(dtstr)
        except MemoryError:
            raise
        except Exception as err:
            raise errors.ParserFailedError(
                "<'%s'> Failed to parse: '%s'.\nError: %s"
                % (self.__class__.__name__, dtstr, err)
            ) from err

    # Build --------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _build(self, dtstr: str, default: object) -> datetime.datetime:
        """(cfunc) Build datetime from the processed result `<'datetime.datetime'>`.

        :raise `ParserBuildError`: if failed to build datetime with the parsed result.
        """
        try:
            # Ignore timezone
            if self._ignoretz:
                return self._generate_dt(default, None)

            # Timezone-naive
            offset: cython.int = self._res.tzoffset
            if offset == -100_000:
                return self._generate_dt(default, None)

            # Timezone-aware
            else:
                tz = utils.UTC if offset == 0 else utils.tz_new(0, 0, offset)
                return self._generate_dt(default, tz)
        except Exception as err:
            raise errors.ParserBuildError(
                "<'%s'> Failed to build datetime from: '%s'.\nResult: %s\nErrors: %s"
                % (self.__class__.__name__, dtstr, self._res, err)
            ) from err

    @cython.cfunc
    @cython.inline(True)
    def _generate_dt(self, default: object, tz: object) -> datetime.datetime:
        """(cfunc) Generate datetime from the processed result `<'datetime.datetime'>`."""
        # Check default
        has_default: cython.bint = utils.is_date(default)

        # . year
        if self._res.year != -1:
            yy = self._res.year
        elif has_default:
            yy = datetime.datetime_year(default)
        else:
            raise ValueError("lack of 'year' value.")
        # . month
        if self._res.month != -1:
            mm = self._res.month
        elif has_default:
            mm = datetime.datetime_month(default)
        else:
            raise ValueError("lack of 'month' value.")
        # . day
        if self._res.day != -1:
            dd = self._res.day
        elif has_default:
            dd = datetime.datetime_day(default)
        else:
            raise ValueError("lack of 'day' value.")
        if dd > 28:
            dd = min(dd, utils.days_in_month(yy, mm))
        # . weekday
        if self._res.weekday != -1:
            wkd: cython.int = utils.ymd_weekday(yy, mm, dd)
            if wkd != self._res.weekday:
                _ymd = utils.ymd_fr_ordinal(
                    utils.ymd_to_ordinal(yy, mm, dd) + self._res.weekday - wkd
                )
                yy, mm, dd = _ymd.year, _ymd.month, _ymd.day
        # . hour
        hh = self._res.hour if self._res.hour != -1 else 0
        # . minute
        mi = self._res.minute if self._res.minute != -1 else 0
        # . second
        ss = self._res.second if self._res.second != -1 else 0
        # . microsecond
        us = self._res.microsecond if self._res.microsecond != -1 else 0

        # Generate datetime
        return datetime.datetime_new(yy, mm, dd, hh, mi, ss, us, tz, 0)

    # ISO format ---------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _process_iso_format(self, dtstr: str) -> cython.bint:
        """(cfunc) Process 'dtstr' as ISO format.

        Automatically fallback to timelex tokens processor
        when iso format processor failed.
        """
        # Parse date components
        if not self._parse_iso_date(dtstr):
            # ISO format date parser failed,
            # fallback to timelex tokens processor.
            return self._process_timelex_tokens(dtstr)

        # Parse time components
        if not self._parse_iso_time(dtstr):
            # ISO format time parser failed,
            # fallback to timelex tokens processor.
            return self._process_timelex_tokens(dtstr)

        # Success
        return True

    # . parser - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_iso_date(self, dtstr: str) -> cython.bint:
        """(cfunc) Parse the date components of the ISO format `<'bool'>`.

        Returns `True` if successfully process the date components
        Y/M/D values and updates the current 'self._idx' position.
        Else `False` if the start of the 'dtstr' does not comply
        with ISO format, 'self._idx' stays at `0`.
        """
        # ISO format size must be >= 7: YYYYWww or YYYYDDD
        size: cython.Py_ssize_t = self._size
        if size < 7:
            return False  # exit: invalid

        # ISO format always starts with year (YYYY)
        year = utils.parse_isoyear(dtstr, 0, size)
        if year == -1:
            return False  # exit: invalid

        # Parse components
        # . YYYY[-]
        ch4: cython.Py_UCS4 = str_read(dtstr, 4)
        idx: cython.Py_ssize_t
        if utils.is_isodate_sep(ch4):
            if size < 8:
                #: For ISO format with "-" seperator
                #: The minimum size should be 8.
                return False  # exit: invalid

            # . YYYY-MM[-]
            ch7: cython.Py_ssize_t = str_read(dtstr, 7)
            if utils.is_isodate_sep(ch7):
                # . parse month: YYYY-[MM]
                month = utils.parse_isomonth(dtstr, 5, size)
                if month == -1:
                    return False  # exit: invalid
                # . parse day: YYYY-MM-[DD]
                day = utils.parse_isoday(dtstr, 8, size)
                if day == -1:
                    return False  # exit: invalid
                idx = 10

            # YYYY-[W]
            elif utils.is_isoweek_sep(str_read(dtstr, 5)):
                # . parse week: YYYY-W[ww]
                week = utils.parse_isoweek(dtstr, 6, size)
                if week == -1:
                    return False  # exit: invalid
                # . parse weekday: YYYY-Www-[D]
                if size > 9 and utils.is_isodate_sep(str_read(dtstr, 8)):
                    weekday = utils.parse_isoweekday(dtstr, 9, size)
                    if weekday == -1:
                        return False  # exit: invalid
                    idx = 10
                else:
                    idx, weekday = 8, 1
                # . calculate MM/DD
                _ymd = utils.ymd_fr_isocalendar(year, week, weekday)
                year, month, day = _ymd.year, _ymd.month, _ymd.day

            # . YYYY-DD[D]
            elif utils.is_ascii_digit(ch7):
                # . parse days of the year: YYYY-[DDD]
                days = utils.parse_isoyearday(dtstr, 5, size)
                if days == -1:
                    return False  # exit: invalid
                # Calculate MM/DD
                _ymd = utils.ymd_fr_days_of_year(year, days)
                month, day = _ymd.month, _ymd.day
                idx = 8

            # . Invalid ISO format
            else:
                return False  # exit: invalid

        # . YYYY[W]
        elif utils.is_isoweek_sep(ch4):
            # . parse week: YYYYW[ww]
            week = utils.parse_isoweek(dtstr, 5, size)
            if week == -1:
                return False  # exit: invalid
            # . parse weekday
            if size > 7:
                # . YYYYWww-[D]
                if size > 8 and utils.is_isodate_sep(str_read(dtstr, 7)):
                    weekday = utils.parse_isoweekday(dtstr, 8, size)
                    if weekday == -1:
                        idx, weekday = 7, 1
                    else:
                        idx = 9
                # . YYYYWww[D]
                else:
                    weekday = utils.parse_isoweekday(dtstr, 7, size)
                    if weekday == -1:
                        idx, weekday = 7, 1
                    else:
                        idx = 8
            else:
                idx, weekday = 7, 1
            # . calculate MM/DD
            _ymd = utils.ymd_fr_isocalendar(year, week, weekday)
            year, month, day = _ymd.year, _ymd.month, _ymd.day

        # . YYYY[D]
        elif utils.is_ascii_digit(ch4):
            # . YYYYMMD[D]
            if size > 7 and utils.is_ascii_digit(str_read(dtstr, 7)):
                # . parse month: YYYY[MM]
                month = utils.parse_isomonth(dtstr, 4, size)
                if month == -1:
                    return False  # exit: invalid
                # . parse day: YYYYMM[DD]
                day = utils.parse_isoday(dtstr, 6, size)
                if day == -1:
                    return False  # exit: invalid
                idx = 8
            # . YYYYDDD
            else:
                # . parse days of the year: YYYY[DDD]
                days = utils.parse_isoyearday(dtstr, 4, size)
                if days == -1:
                    return False  # exit: invalid
                # Calculate MM/DD
                _ymd = utils.ymd_fr_days_of_year(year, days)
                month, day = _ymd.month, _ymd.day
                idx = 7

        # . Invalid ISO format
        else:
            return False

        # Set index & Y/M/D
        self._idx = idx
        self._res.century_specified = True
        self._res.set_ymd_int(year, 1)
        self._res.set_ymd_int(month, 2)
        self._res.set_ymd_int(day, 3)
        return True  # exit: complete

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_iso_time(self, dtstr: str) -> cython.bint:
        """(cfunc) Parse the time components of the ISO format `<'bool'>`.

        This method should only be called after '_parse_iso_date()'
        successfully parsed the ISO format date components.

        Return `True` if successfully parsed the ISO format time
        components of the 'dtstr' and reach the end of the string.
        Else `False` if failed to parse the ISO format time components.
        The 'self._idx' stays at where the date components ends,
        and the remaining should fall back to timelex tokens
        processr for further handling.
        """
        # Validate
        idx: cython.Py_ssize_t = self._idx
        size: cython.Py_ssize_t = self._size
        if idx == size:
            # eof: no time component
            return True
        if idx < 7 or size - idx < 7:
            #: The minimum time component is [HHMMSS],
            #: adding iso seperator "T", the size
            #: of the 'dtstr' should have at least
            #: 7 more extra charactors.
            return False  # fallback
        if not utils.is_iso_sep(str_read(dtstr, idx)):
            #: The charactor right after date components
            #: should be either "T" or " ".
            return False  # fallback

        # Parse HH:MM:SS / HHMMSS
        # . hour: ...T[HH]
        hour = utils.parse_isohour(dtstr, idx + 1, size)
        if hour == -1:
            return False  # fallback
        # . with seperator: ...THH[:]
        if utils.is_isotime_sep(str_read(dtstr, idx + 3)):
            # . THH:MM:SS requires at least 9 charactors.
            if size - idx < 9:
                return False  # fallback
            # . minute: ...THH:[MM]
            minute = utils.parse_isominute(dtstr, idx + 4, size)
            if minute == -1:
                return False  # fallback
            # . second: ...THH:MM:[SS]
            second = utils.parse_isosecond(dtstr, idx + 7, size)
            if second == -1:
                return False  # fallback
            # . set H/M/S & index
            idx += 9
            self._res.hour = hour
            self._res.minute = minute
            self._res.second = second
            # . eof: ...THH:MM:SS
            if idx == size:
                return True  # success
        # . without seperator: ...THH[]
        else:
            # . minute: ...THH[MM]
            minute = utils.parse_isominute(dtstr, idx + 3, size)
            if minute == -1:
                return False  # fallback
            # . second: ...THHMM[SS]
            second = utils.parse_isosecond(dtstr, idx + 5, size)
            if second == -1:
                return False  # fallback
            # . set H/M/S & index
            idx += 7
            self._res.hour = hour
            self._res.minute = minute
            self._res.second = second
            # . eof: ...THHMMSS
            if idx == size:
                return True  # success

        # Parse fraction: ...THH:MM:SS[.SSS]
        if size - idx > 1 and str_read(dtstr, idx) in (".", ","):
            idx += 1  # skip: [.]
            buffer: cython.char[7]
            ch: cython.Py_UCS4
            digits: cython.Py_ssize_t = 0
            # . parse fraction values
            while idx < size and digits < 6:
                ch = str_read(dtstr, idx)
                if not utils.is_ascii_digit(ch):
                    break
                buffer[digits] = ch
                idx += 1
                digits += 1
            # . compensate missing digits
            if digits < 6:
                if digits == 0:
                    self._idx = idx - 1  # update index
                    return self._parse_iso_extra(dtstr)  # fallback
                ch = "0"
                for i in range(digits, 6):
                    buffer[i] = ch
            # . convert to integer
            buffer[6] = 0  # null-term
            us = strtoll(buffer, cython.NULL, 10)
            self._res.microsecond = us
            # . skip extra digits & spaces
            while idx < size and (
                utils.is_ascii_digit(ch := str_read(dtstr, idx)) or ch == " "
            ):
                idx += 1
            # . eof: ...THH:MM:SS.SSS[SSS]
            if idx == size:
                return True  # success
        # Skip spaces
        else:
            while idx < size and str_read(dtstr, idx) == " ":
                idx += 1
            # . eof: ...THH:MM:SS
            if idx == size:
                return True  # success

        # Parse UTC: ...THH:MM:SS(.SSSSSS)[+/-HH:MM]
        if size - idx > 4:
            # . offset sign
            ch: cython.Py_UCS4 = str_read(dtstr, idx)
            if ch == "+":
                sign: cython.int = 1
            elif ch == "-":
                sign: cython.int = -1
            else:
                self._idx = idx  # update index
                return self._parse_iso_extra(dtstr)  # fallback
            # . offset hour
            hh: cython.int = utils.parse_isohour(dtstr, idx + 1, size)
            if hh == -1:
                self._idx = idx  # update index
                return self._parse_iso_extra(dtstr)  # fallback
            # . offset minute
            ch: cython.Py_UCS4 = str_read(dtstr, idx + 3)
            if utils.is_isotime_sep(ch):
                # . +/-HH:[MM]
                mm: cython.int = utils.parse_isominute(dtstr, idx + 4, size)
                if mm == -1:
                    self._idx = idx  # update index
                    return self._parse_iso_extra(dtstr)  # fallback
                idx += 6  # skip: [+/-HH:MM]
            else:
                # . +/-HH[MM]
                mm: cython.int = utils.parse_isominute(dtstr, idx + 3, size)
                if mm == -1:
                    self._idx = idx  # update index
                    return self._parse_iso_extra(dtstr)  # fallback
                idx += 5  # skip: [+/-HHMM]
            # . set timezone offset
            self._res.tzoffset = sign * (hh * 3_600 + mm * 60)
            # . eof: ...THH:MM:SS(.SSSSSS)[+/-HH:MM]
            if idx == size:
                return True  # success

        # Parse UTC: ...THH:MM:SS(.SSSSSS)[Z]
        if (ex := size - idx) > 0 and str_read(dtstr, idx) in ("z", "Z"):
            # . eof: THH:MM:SS(.SSSSSS)[Z]
            if ex == 1:
                self._res.tzoffset = 0
                return True  # success
            # . space: ...THH:MM:SS(.SSSSSS)[Z][space]
            if ex > 1 and str_read(dtstr, idx + 1) == " ":
                self._res.tzoffset = 0
                idx += 2  # skip: [Z][space]
                if idx == size:
                    return True  # success

        # Still have extra characters
        self._idx = idx  # update index
        return self._parse_iso_extra(dtstr)  # fallback

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_iso_extra(self, dtstr: str) -> cython.bint:
        """(cfunc) Parser the extra charactors after ISO format `<'bool'>`

        This method dedicates to parse the extra charactors left after
        ISO time parser successfully extracted 'HH:MM:SS[.f]', but failed
        to handle the remains. Since the remaining charactors can only
        represents 'weekday', 'AM/PM' flag and 'timezone' information,
        this method is basically a lite version of timelex token parser
        excluding 'numeric' and 'month' parsing capabilities.
        """
        # Reset index, size & tokens
        tokens = _timelex(dtstr.lower(), self._idx, self._size)
        self._tokens = tokens
        self._idx, self._size = 0, list_len(tokens)

        # Parse tokens
        while (token := self._token(self._idx)) is not None:
            # . reset cache
            self._reset_tokens()
            # . weekday token
            if self._parse_token_weekday(token):
                pass
            # . am/pm token
            elif self._parse_token_ampm_flag(token):
                pass
            # . tzname token
            elif self._parse_token_tzname(token):
                pass
            # . tzoffset token
            elif self._parse_token_tzoffset(token):
                pass
            # . next token
            self._idx += 1

        # Success
        self._reset_tokens()
        self._tokens = None
        return True

    # Timelex tokens -----------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _process_timelex_tokens(self, dtstr: str) -> cython.bint:
        """(cfunc) Process 'dtstr' through timelex token approach."""
        # Reset index, size & tokens
        tokens = _timelex(dtstr.lower(), self._idx, self._size)
        self._tokens = tokens
        self._idx, self._size = 0, list_len(tokens)

        # Parse tokens
        while (token := self._token(self._idx)) is not None:
            # . reset cache
            self._reset_tokens()
            # . numeric token
            if self._parse_token_numeric(token):
                pass
            # . month token
            elif self._parse_token_month(token):
                pass
            # . weekday token
            elif self._parse_token_weekday(token):
                pass
            # . am/pm token
            elif self._parse_token_ampm_flag(token):
                pass
            # . tzname token
            elif self._parse_token_tzname(token):
                pass
            # . tzoffset token
            elif self._parse_token_tzoffset(token):
                pass
            # . next token
            self._idx += 1

        # Success
        self._reset_tokens()
        self._tokens = None
        return True

    # . parser - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_token_numeric(self, token: str) -> cython.bint:
        """(cfunc) Parse as a 'numeric' token `<'bool'>`.

        Returns `True` if the token represents a 'numeric'
        value and processed successfully, else `False`.
        """
        # Validate if is a numeric token
        try:
            t_val: cython.double = float(token)
        except ValueError:
            return False  # exit: not a numeric token
        if not math.isfinite(t_val):
            #: 'inf' or 'nan' should not be treated
            #: as numeric values, but aplha instead.
            return False  # exit: not a numeric token
        t_len: cython.Py_ssize_t = str_len(token)
        t_int: cython.int

        # . (Y/M/D)HH[MM]
        if (
            t_len in (2, 4)  # HHMM or HHMM
            and self._res.ymd_populated() == 3  # Y/M/D solved
            and self._res.hour == -1  # hour not set
            and (
                self._idx + 2 > self._size  # has 2 more tokens
                or (
                    (t1 := self._token1()) is not None  # next token exists
                    and t1 != ":"  # next token is not seperator
                    and self._token_to_hms_flag(t1) == -1  # next token is not hms flag
                )
            )
        ):
            # . (Y/M/D)[HH]
            self._res.hour = utils.parse_isohour(token, 0, t_len)
            if t_len == 4:
                # . (Y/M/D)HH[MM]
                self._res.minute = utils.parse_isominute(token, 2, t_len)
            return True  # exit

        # . YYMMDD / HHMMSS
        if t_len == 6 and not str_contains(token, "."):
            # . [YYMMDD]
            if self._res.ymd_populated() == 0:
                self._res.set_ymd_int(utils.slice_to_uint(token, 0, 2), 0)
                self._res.set_ymd_int(utils.slice_to_uint(token, 2, 2), 0)
                self._res.set_ymd_int(utils.slice_to_uint(token, 4, 2), 0)
            # . [HHMMSS]
            else:
                self._res.hour = utils.parse_isohour(token, 0, t_len)
                self._res.minute = utils.parse_isominute(token, 2, t_len)
                self._res.second = utils.parse_isosecond(token, 4, t_len)
            return True  # exit

        # . HHMMSS.[us]
        if t_len > 6 and str_read(token, 6) == ".":
            # . [HHMMSS]
            self._res.hour = utils.parse_isohour(token, 0, t_len)
            self._res.minute = utils.parse_isominute(token, 2, t_len)
            self._res.second = utils.parse_isosecond(token, 4, t_len)
            # . [us]
            if t_len > 7:
                self._res.microsecond = utils.parse_isofraction(token, 7, t_len)
            return True  # exit

        # . YYYYMMDD[HHMM[SS]]
        if t_len in (8, 12, 14):
            # . [YYYYMMDD]
            self._res.set_ymd_int(utils.parse_isoyear(token, 0, t_len), 1)
            self._res.set_ymd_int(utils.slice_to_uint(token, 4, 2), 0)
            self._res.set_ymd_int(utils.slice_to_uint(token, 6, 2), 0)
            # . YYYYMMDD[HHMM]
            if t_len > 8:
                self._res.hour = utils.parse_isohour(token, 8, t_len)
                self._res.minute = utils.parse_isominute(token, 10, t_len)
                # . YYYYMMDDHHMM[SS]
                if t_len > 12:
                    self._res.second = utils.parse_isosecond(token, 12, t_len)
            return True  # exit

        # . HH[ ]h or MM[ ]m or SS[.ss][ ]s
        if self._parse_token_hmsf(t_val):
            return True  # exit

        # . HH:MM[:SS[.ss]]
        if (tk2 := self._token2()) is not None and self._token1() == ":":
            # . [HH:MM]
            self._res.hour = int(t_val)
            self._set_mmss(float(tk2))
            # . [:SS[.ss]]
            if (tk4 := self._token4()) is not None and self._token3() == ":":
                self._set_ssff(float(tk4))
                self._idx += 2  # skip 'tk3' & 'tk4'
            self._idx += 2  # skip 'tk1' & 'tk2'
            return True  # exit

        # . YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
        if (tk1 := self._token1()) is not None and tk1 in ("-", "/", "."):
            # 1st Y/M/D value
            self._res.set_ymd_int(int(t_val), 0)

            # 2nd Y/M/D value
            if tk2 is not None and not self._is_token_jump(tk2):
                try:
                    t_int = int(tk2)
                    # 01-[01]-01
                    self._res.set_ymd_int(t_int, 0)
                except ValueError:
                    # 01-[Jan]-01
                    t_int = self._token_to_month(tk2)
                    if t_int != -1:
                        self._res.set_ymd_int(t_int, 2)  # must be month
                    else:
                        self._res.set_ymd_str(tk2, 0)

                # 3rd Y/M/D value
                if (tk4 := self._token4()) is not None and self._token3() == tk1:
                    try:
                        t_int: cython.int = int(tk4)
                        # 01-01-[01]
                        self._res.set_ymd_int(t_int, 0)
                    except ValueError:
                        # 01-01-[Jan]
                        t_int: cython.int = self._token_to_month(tk4)
                        if t_int != -1:
                            self._res.set_ymd_int(t_int, 2)  # must be month
                        else:
                            self._res.set_ymd_str(tk4, 0)

                    self._idx += 2  # skip 'tk3' & 'tk4'
                self._idx += 1  # skip 'tk2'
            self._idx += 1  # skip 'tk1'
            return True  # exit

        # "hour AM" or YY|MM|DD
        if self._idx + 1 >= self._size or self._is_token_jump(tk1):
            # . 12 AM
            if tk2 is not None and (ampm_flag := self._token_to_ampm_flag(tk2)) != -1:
                self._res.hour = self._adj_hour_ampm(int(t_val), ampm_flag)
                self._idx += 1  # skip 'tk2'
            # . YY|MM|DD
            else:
                self._res.set_ymd_int(int(t_val), 0)
            self._idx += 1  # skip 'tk1'
            return True  # exit

        # "hourAM"
        t_int = int(t_val)
        if 0 <= t_int < 24 and (ampm_flag := self._token_to_ampm_flag(tk1)) != -1:
            self._res.hour = self._adj_hour_ampm(t_int, ampm_flag)
            self._idx += 1  # skip 'tk1'
            return True  # exit

        # Probably is 'day'
        if self._res.could_be_day(t_int):
            self._res.set_ymd_int(t_int, 0)
            return True  # exit

        # Exit
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_token_month(self, token: str) -> cython.bint:
        """(cfunc) Parse as a 'month' token `<'bool'>`.

        Returns `True` if the token represents a 'month'
        value and processed successfully, else False.
        """
        # Validate if is a 'month' token
        mm: cython.int = self._token_to_month(token)
        if mm == -1:
            return False  # exit: not a 'month' token
        self._res.set_ymd_int(mm, 2)

        # Try parse 'year' & 'day'
        if (tk2 := self._token2()) is not None:
            # . Jan-[01]
            if (tk1 := self._token1()) in ("-", "/", "."):
                self._res.set_ymd_str(tk2, 0)
                # . Jan-01-[01]
                if (tk4 := self._token4()) is not None and self._token3() == tk1:
                    self._res.set_ymd_str(tk4, 0)

                    self._idx += 2  # skip 'tk4' & 'tk3'
                self._idx += 2  # skip 'tk2' & 'tk1'

            # . Jan of [01] | 01 is clearly year
            elif (
                (tk4 := self._token4()) is not None
                and self._is_token_pertain(tk2)
                and self._token3() == " "
            ):
                try:
                    yy: cython.int = int(tk4)
                    self._res.set_ymd_int(yy, 1)
                    self._idx += 4  # skip 'tk1' to 'tk4'
                except ValueError:
                    pass  # wrong guess

        # Exit
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_token_weekday(self, token: str) -> cython.bint:
        """(cfunc) Parse as a 'weekday' token `<'bool'>`.

        Returns `True` if the token represents a 'weekday'
        value and processed successfully, else `False`.
        """
        # Validate if is a 'weekday' token
        wd: cython.int = self._token_to_weekday(token)

        if wd == -1:
            return False  # exit: not a 'weekday' token

        # Set result
        self._res.weekday = wd
        return True  # exit

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_token_hmsf(self, t_val: cython.double) -> cython.bint:
        """(cfunc) Parse the token value as one of the H/M/S/f components `<'bool'>`.

        Returns `True` if the token value represents one of the
        H/M/S/f components and processed successfully, else `False`.
        """
        hms_flag: cython.int
        # Forward
        if (tk1 := self._token1()) is not None and (
            hms_flag := self._token_to_hms_flag(tk1)
        ) != -1:
            # There is an "h", "m", or "s" label following this token.
            # We take assign the upcoming label to the current token.
            # e.g. the "12" in 12h"
            self._set_hmsf(t_val, hms_flag)
            self._idx += 1  # skip 'tk1'
            return True

        # Backward
        if (tk1_b := self._token(self._idx - 1)) is not None and (
            hms_flag := self._token_to_hms_flag(tk1_b)
        ) != -1:
            # There is a "h", "m", or "s" preceding this token. Since neither
            # of the previous cases was hit, there is no label following this
            # token, so we use the previous label.
            # e.g. the "04" in "12h04"
            # looking backwards, flag increase 1
            self._set_hmsf(t_val, hms_flag + 1)
            return True

        # Not HMSF token
        return False

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_token_ampm_flag(self, token: str) -> cython.bint:
        """(cfunc) Parse as an 'AM/PM' flag token `<'bool'>`.

        Returns `True` if the token represents an `AM/PM` flag
        and processed successfully, else `False`.
        """
        # AM/PM flag already set
        if self._res.ampm != -1:
            return False  # exit

        # Missing hour / Not a 12 hour clock
        hour: cython.int = self._res.hour
        if not 0 <= hour <= 12:
            return False  # exit

        # Validate if is an 'AM/PM' token
        ampm_flag: cython.int = self._token_to_ampm_flag(token)
        if ampm_flag == -1:
            return False  # exit: not an 'AM/PM' token

        # Adjust hour according to AM/PM
        self._res.hour = self._adj_hour_ampm(hour, ampm_flag)
        self._res.ampm = ampm_flag
        return True  # eixt

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_token_tzname(self, token: str) -> cython.bint:
        """(cfunc) Parse as a 'tzname' token `<'bool'>`.

        Returns `True` if the token represents a timezone
        name and processed successfully, else `False`.
        """
        # Check if need to parse tzname
        if (
            self._ignoretz  # ignore timezone
            or self._res.hour == -1  # hour not set
            or self._res.tzoffset != -100_000  # tzoffset exits
        ):
            return False  # exit: tzname no needed

        # Validate if the token could be a tzname
        # . utc timezone
        if self._is_token_utc(token):
            self._res.tzoffset = 0
        # . timezone name length must between 3-5
        if not 3 <= str_len(token) <= 5:
            return False  # exit: not tzname
        # . timezone name must be ASCCI alpha
        for ch in token:
            if not utils.is_ascii_alpha_lower(ch):
                return False  # exit: not tzname
            self._res.tzoffset = self._token_to_tzoffset(token)

        # Check for something like GMT+3, or BRST+3. Notice
        # that it doesn't mean "I am 3 hours after GMT", but
        # "my time +3 is GMT". If found, we reverse the logic
        # so that tzoffset parsing code will get it right.
        if (tk1 := self._token1()) is not None:
            if tk1 == "+":
                list_setitem(self._tokens, self._idx + 1, "-")
            elif tk1 == "-":
                list_setitem(self._tokens, self._idx + 1, "+")
            else:
                return True  # exit
            # Reset tzoffset
            self._res.tzoffset = -100_000  # means None

        return True  # exit

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_token_tzoffset(self, token: str) -> cython.bint:
        """(cfunc) Parse as a 'tzoffset' token `<'bool'>`.

        Returns `True` if the token represents a timezone
        offset components and processed successfully, else
        `False`.
        """
        # Check if need to parse tzoffset
        if (
            self._res.hour == -1  # hour not set
            or self._res.tzoffset != -100_000  # tzoffset exits
        ):
            return False  # exit: not 'tzoffset' token

        # Validate if the token is 'tzoffset' component
        if token == "+":
            sign: cython.int = 1
        elif token == "-":
            sign: cython.int = -1
        else:
            return False  # eixt: not 'tzoffset' token

        # Validate next token
        if (tk1 := self._token1()) is None:
            return False  # exit: not 'tzoffset' token
        try:
            offset: cython.longlong = int(tk1)
        except ValueError:
            return False  # exit: not 'tzoffset' token

        # Calculate offset
        t_len: cython.Py_ssize_t = str_len(tk1)
        if t_len == 4:
            # . +/-[0300]
            hh: cython.int = utils.parse_isohour(tk1, 0, t_len)
            mm: cython.int = utils.parse_isominute(tk1, 2, t_len)
            offset = sign * (hh * 3_600 + mm * 60)
        elif (tk3 := self._token3()) is not None and self._token2() == ":":
            try:
                # . +/-[03]:00
                mm: cython.int = int(tk3)
                offset = sign * (offset * 3_600 + mm * 60)
                self._idx += 2  # skip 'tk2' & 'tk3'
            except ValueError:
                # . +/-[03] / Invalid
                offset = sign * (offset * 3_600) if t_len <= 2 else -100_000
        else:
            # . +/-[03] / Invalid
            offset = sign * (offset * 3_600) if t_len <= 2 else -100_000

        # Validate offset
        if not -86_340 <= offset <= 86_340:
            raise ValueError(
                "invalid timezone offset %d (seconds), "
                "must between -86340 and 86340." % (offset)
            )
        self._res.tzoffset = offset

        # Exit
        self._idx += 1  # skip 'tk1'
        return True  # exit

    # . tokens - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    def _token(self, idx: cython.Py_ssize_t) -> str:
        """(cfunc) Get token by index `<'str/None'>`.

        #### Returns `None` if index out of range.
        """
        if 0 <= idx < self._size:
            return cython.cast(str, list_getitem(self._tokens, idx))
        else:
            return None

    @cython.cfunc
    @cython.inline(True)
    def _token1(self) -> str:
        """(cfunc) Get the next (index+1) token `<'str/None'>`.

        #### Returns `None` if index out of range.
        """
        if self._tk1 is None:
            self._tk1 = self._token(self._idx + 1)
        return self._tk1

    @cython.cfunc
    @cython.inline(True)
    def _token2(self) -> str:
        """(cfunc) Get the next (index+2) token `<'str/None'>`.

        #### Returns `None` if index out of range.
        """
        if self._tk2 is None:
            self._tk2 = self._token(self._idx + 2)
        return self._tk2

    @cython.cfunc
    @cython.inline(True)
    def _token3(self) -> str:
        """(cfunc) Get the next (index+3) token `<'str/None'>`.

        #### Returns `None` if index out of range.
        """
        if self._tk3 is None:
            self._tk3 = self._token(self._idx + 3)
        return self._tk3

    @cython.cfunc
    @cython.inline(True)
    def _token4(self) -> str:
        """(cfunc) Get the next (index+4) token `<'str/None'>`.

        #### Returns `None` if index out of range.
        """
        if self._tk4 is None:
            self._tk4 = self._token(self._idx + 4)
        return self._tk4

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _reset_tokens(self) -> cython.bint:
        """(cfunc) Reset cached token1...token4."""
        self._tk1 = None
        self._tk2 = None
        self._tk3 = None
        self._tk4 = None
        return True

    # . utils - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_hmsf(self, t_val: cython.double, hms_flag: cython.int) -> cython.bint:
        """(cfunc) Set the 'HH:MM:SS.f' result based on
        the given token value and 'hms_flag'."""
        if hms_flag == 0:  # flag: hour
            return self._set_hhmm(t_val)
        if hms_flag == 1:  # flag: minute
            return self._set_mmss(t_val)
        if hms_flag == 2:  # flag: second
            return self._set_ssff(t_val)
        return False

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_hhmm(self, t_val: cython.double) -> cython.bint:
        """(cfunc) Set 'HH:MM' result based on the given token value."""
        self._res.hour = int(t_val)
        if rem := t_val % 1:
            self._res.minute = math.lround(rem * 60)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_mmss(self, t_val: cython.double) -> cython.bint:
        """(cfunc) Set 'MM:SS' result based on the given token value."""
        self._res.minute = int(t_val)
        if rem := t_val % 1:
            self._res.second = math.lround(rem * 60)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_ssff(self, t_val: cython.double) -> cython.bint:
        """(cfunc) Set 'SS.f' result based on the given token value."""
        self._res.second = int(t_val)
        if rem := t_val % 1:
            self._res.microsecond = math.lround(rem * 1_000_000)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _adj_hour_ampm(self, hour: cython.int, ampm_flag: cython.int) -> cython.int:
        """(cfunc) Adjust 'hour' value based on the given AM/PM flag `<'int'>`."""
        if hour < 12:
            if ampm_flag == 1:
                hour += 12
            return max(0, hour)
        elif hour == 12 and ampm_flag == 0:
            return 0
        else:
            return hour

    # . configs - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_token_pertain(self, token: object) -> cython.bint:
        """(cfunc) Check if the token should be
        recognized as a pertain `<'bool'>`.
        """
        return set_contains(self._pertain, token)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_token_jump(self, token: object) -> cython.bint:
        """(cfunc) Check if the token should be
        recognized as a jump word `<'bool'>`.
        """
        return set_contains(self._jump, token)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_token_utc(self, token: object) -> cython.bint:
        """(cfunc) Check if the token should be
        recognized as an UTC timezone `<'bool'>`."""
        return set_contains(self._utc, token)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-2, check=False)
    def _token_to_month(self, token: object) -> cython.int:
        """(cfunc) Try to convert the token to month number `<'int'>`.

        Returns the month number (1-12) if token matched
        with 'month' settings in Configs, else -1.
        """
        val = dict_getitem(self._month, token)
        if val == cython.NULL:
            return -1
        return cython.cast(object, val)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-2, check=False)
    def _token_to_weekday(self, token: object) -> cython.int:
        """(cfunc) Try to convert the token to weekday number `<'int'>`.

        Returns the weekday value (0-6) if token matched
        with 'weekday' settings in Configs, else -1.
        """
        val = dict_getitem(self._weekday, token)
        if val == cython.NULL:
            return -1
        return cython.cast(object, val)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-200_000, check=False)
    def _token_to_tzoffset(self, token: object) -> cython.int:
        """(cfunc) Try to convert the token to timezone offset in seconds `<'int'>`.

        Returns the timezone offset if token matched with
        the 'tz' settings in Configs, else -100_000.
        """
        val = dict_getitem(self._tz, token)
        if val == cython.NULL:
            return -100_000
        return cython.cast(object, val)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-2, check=False)
    def _token_to_hms_flag(self, token: object) -> cython.int:
        """(cfunc) Try to convert the token to hms flag `<'int'>`.

        Returns the hms flag (0=hour, 1=minute, 2=second) if the token
        matched with 'hms_flag' settings in Configs, else -1.
        """
        val = dict_getitem(self._hms_flag, token)
        if val == cython.NULL:
            return -1
        return cython.cast(object, val)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-2, check=False)
    def _token_to_ampm_flag(self, token: object) -> cython.int:
        """(cfunc) Try to convert the token to AM/PM flag `<'int'>`.

        Returns the AM/PM flag (0=am, 1=pm) if the token
        matched with 'ampm_flag' settings in Configs, else -1.
        """
        val = dict_getitem(self._ampm_flag, token)
        if val == cython.NULL:
            return -1
        return cython.cast(object, val)


_default_parser: Parser = Parser()


@cython.ccall
def parse(
    dtstr: str,
    default: datetime.date | datetime.datetime | None = None,
    year1st: bool | None = None,
    day1st: bool | None = None,
    ignoretz: cython.bint = False,
    isoformat: cython.bint = True,
    cfg: Configs = None,
) -> datetime.datetime:
    """Parse the datetime string into `<'datetime.datetime'>`.

    :param dtstr `<'str'>`: The string that contains datetime information.

    :param default `<'datetime/date/None'>`: Default value to fill in missing date fields, defaults to `None`.
        - `<'date/datetime'>` If the parser fails to extract Y/M/D from the string,
            use the passed-in 'default' to fill in the missing fields.
        - If `None`, raises `PaserBuildError` if any Y/M/D fields is missing.

    :param year1st `<'bool/None'>`: Interpret the first ambiguous Y/M/D value as year, defaults to `None`.
        If 'year1st=None', use `cfg.year1st` if 'cfg' is specified; otherwise, defaults to `False`.

    :param day1st `<'bool/None'>`: Interpret the first ambiguous Y/M/D values as day, defaults to `None`.
        If 'day1st=None', use `cfg.day1st` if 'cfg' is specified; otherwise, defaults to `False`.

    :param ignoretz `<'bool'>`: Whether to ignore timezone information, defaults to `False`.
        - `True`: Ignores any timezone information and returns a timezone-naive datetime (increases parser performance).
        - `False`: Processes timezone information and generates a timezone-aware datetime if matched by `cfg.utc` & `cfg.tz`.

    :param isoformat `<'bool'>`: Whether try to parse 'dtstr' as ISO format, defaults to `True`.
        - `True`: First tries to process 'dtstr' as ISO format; if failed, falls back
            to token parsing (best performance for most strings).
        - `False`: Only processes 'dtstr' through token parsing (increases performance
            if 'dtstr' is not ISO format).

    ## Ambiguous Y/M/D
    Both 'year1st' and 'day1st' determine how to interpret ambiguous
    Y/M/D values. 'year1st' has higher priority.

    #### When all three values are ambiguous (e.g. `01/05/09`):
    - If 'year1st=False' & 'day1st=False': interprets as `2009-01-05` (M/D/Y).
    - If 'year1st=False' & 'day1st=True': interprets as `2009-05-01` (D/M/Y).
    - If 'year1st=True' & 'day1st=False': interprets as `2001-05-09` (Y/M/D).
    - If 'year1st=True' & 'day1st=True': interprets as `2001-09-05` (Y/D/M).

    #### When the 'year' value is known (e.g. `32/01/05`):
    - If 'day1st=False': interpretes as `2032-01-05` (Y/M/D).
    - If 'day1st=True': interpretes as `2032-05-01` (Y/D/M).

    #### When only one value is ambiguous (e.g. `32/01/20`):
    - The parser automatically determines the correct order; 'year1st' and 'day1st' are ignored.
    """
    if cfg is None:
        return _default_parser.parse(
            dtstr, default, year1st, day1st, ignoretz, isoformat
        )
    return Parser(cfg).parse(dtstr, default, year1st, day1st, ignoretz, isoformat)


@cython.ccall
def parse_dtobj(
    dtobj: object,
    default: datetime.date | datetime.datetime | None = None,
    year1st: bool | None = None,
    day1st: bool | None = None,
    ignoretz: cython.bint = False,
    isoformat: cython.bint = True,
    cfg: Configs = None,
) -> datetime.datetime:
    """Parse from a datetime-like object in to datetime `<'datetime.datetime'>`.

    :param dtobj `<'object'>`: Datetime-like object:
        - `<'str'>` A datetime string containing datetime information.
        - `<'datetime.datetime'>` An instance of `datetime.datetime`.
        - `<'datetime.date'>` An instance of `datetime.date` (time fields set to 0).
        - `<'int/float'>` Numeric value treated as total seconds since Unix Epoch.
        - `<'np.datetime64'>` Resolution above microseconds ('us') will be discarded.
        - `<'None'>` Return the current local datetime.

    ## Praser Parameters
    #### Parameters below only take effect when 'dtobj' is of type `<'str'>`.

    :param default `<'datetime/date/None'>`: Default value to fill in missing date fields, defaults to `None`.
        - `<'date/datetime'>` If the parser fails to extract Y/M/D from the string,
          use the passed-in 'default' to fill in the missing fields.
        - If `None`, raises `PaserBuildError` if any Y/M/D fields is missing.

    :param year1st `<'bool/None'>`: Interpret the first ambiguous Y/M/D value as year, defaults to `None`.
        If 'year1st=None', use `cfg.year1st` if 'cfg' is specified; otherwise, defaults to `False`.

    :param day1st `<'bool/None'>`: Interpret the first ambiguous Y/M/D values as day, defaults to `None`.
        If 'day1st=None', use `cfg.day1st` if 'cfg' is specified; otherwise, defaults to `False`.

    :param ignoretz `<'bool'>`: Whether to ignore timezone information, defaults to `False`.
        - `True`: Ignores any timezone information and returns a timezone-naive datetime (increases parser performance).
        - `False`: Processes timezone information and generates a timezone-aware datetime if matched by `cfg.utc` & `cfg.tz`.

    :param isoformat `<'bool'>`: Whether try to parse 'dtstr' as ISO format, defaults to `True`.
        - `True`: First tries to process 'dtstr' as ISO format; if failed, falls back
          to token parsing (best performance for most strings).
        - `False`: Only processes 'dtstr' through token parsing (increases performance
          if 'dtstr' is not ISO format).

    :param cfg `<'Configs/None'>`: Custom parser configurations, defaults to `None`.

    ## Ambiguous Y/M/D
    Both 'year1st' and 'day1st' determine how to interpret ambiguous
    Y/M/D values. 'year1st' has higher priority.

    #### When all three values are ambiguous (e.g. `01/05/09`):
    - If 'year1st=False' & 'day1st=False': interprets as `2009-01-05` (M/D/Y).
    - If 'year1st=False' & 'day1st=True': interprets as `2009-05-01` (D/M/Y).
    - If 'year1st=True' & 'day1st=False': interprets as `2001-05-09` (Y/M/D).
    - If 'year1st=True' & 'day1st=True': interprets as `2001-09-05` (Y/D/M).

    #### When the 'year' value is known (e.g. `32/01/05`):
    - If 'day1st=False': interpretes as `2032-01-05` (Y/M/D).
    - If 'day1st=True': interpretes as `2032-05-01` (Y/D/M).

    #### When only one value is ambiguous (e.g. `32/01/20`):
    - The parser automatically determines the correct order; 'year1st' and 'day1st' are ignored.
    """
    # . datetime string
    if isinstance(dtobj, str):
        return parse(dtobj, default, year1st, day1st, ignoretz, isoformat, cfg)
    # . datetime.datetime
    if utils.is_dt(dtobj):
        return dtobj
    # . datetime.date
    if utils.is_date(dtobj):
        return utils.dt_fr_date(dtobj, None)
    # . numeric
    if isinstance(dtobj, (int, float)):
        return utils.dt_fr_seconds(dtobj, None)
    # . np.datetime64
    if utils.is_dt64(dtobj):
        return utils.dt64_to_dt(dtobj, None)
    # . None
    if dtobj is None:
        return utils.dt_now(None)
    # . invalid
    raise errors.ParserFailedError("unsupported 'dtobj' type %s." % type(dtobj))
