# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from __future__ import annotations

# Cython imports
import cython
from cython.cimports.libc import math  # type: ignore
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_READ_CHAR as str_loc  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_GET_LENGTH as str_len  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_FindChar as str_findc  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_Replace as str_replace  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_Substring as str_substr  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_Contains as str_contains  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_FromOrdinal as str_fr_ucs4  # type: ignore
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
from cython.cimports.cytimes import cytime, cydatetime as cydt  # type: ignore

np.import_array()
datetime.import_datetime()

# Python imports
import datetime, time
from dateutil.parser._parser import parserinfo
from cytimes import errors, cydatetime as cydt

__all__ = ["Config", "Parser"]

# Constants -----------------------------------------------------------------------------------
# . default config
# fmt: off
CONFIG_PERTAIN: set[str] = {"of"}
CONFIG_JUMP: set[str] = {
    " ", ".", ",", ";", "-", "/", "'",
    "at", "on", "and", "ad", "m", "t", "of", 
    "st", "nd", "rd", "th", "年" ,"月", "日" }
CONFIG_UTC: set[str] = { 
    "utc", "gmt", "z" }
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
    "dec": 12, "december": 12, "dezember": 12, "décembre": 12, "dicembre": 12, "diciembre": 12, "dezembro": 12, "december": 12, "december": 12, "grudnia": 12,      "aralık": 12, "十二月": 12 }
CONFIG_WEEKDAY: dict[str, int] = {
    # EN(a)   # EN            # DE             # FR           # IT            # ES            # NL            # SE          # PL               # TR            # CN        # CN(a)
    "mon": 0, "monday": 0,    "montag": 0,     "lundi": 0,    "lunedì": 0,    "lunes": 0,     "maandag": 0,   "måndag": 0,  "poniedziałek": 0, "pazartesi": 0, "星期一": 0, "周一": 0,
    "tue": 1, "tuesday": 1,   "dienstag": 1,   "mardi": 1,    "martedì": 1,   "martes": 1,    "dinsdag": 1,   "tisdag": 1,  "wtorek": 1,       "salı": 1,      "星期二": 1, "周二": 1,
    "wed": 2, "wednesday": 2, "mittwoch": 2,   "mercredi": 2, "mercoledì": 2, "miércoles": 2, "woensdag": 2,  "onsdag": 2,  "środa": 2,        "çarşamba": 2,  "星期三": 2, "周三": 2,
    "thu": 3, "thursday": 3,  "donnerstag": 3, "jeudi": 3,    "giovedì": 3,   "jueves": 3,    "donderdag": 3, "torsdag": 3, "czwartek": 3,     "perşembe": 3,  "星期四": 3, "周四": 3,
    "fri": 4, "friday": 4,    "freitag": 4,    "vendredi": 4, "venerdì": 4,   "viernes": 4,   "vrijdag": 4,   "fredag": 4,  "piątek": 4,       "cuma": 4,      "星期五": 4, "周五": 4,
    "sat": 5, "saturday": 5,  "samstag": 5,    "samedi": 5,   "sabato": 5,    "sábado": 5,    "zaterdag": 5,  "lördag": 5,  "sobota": 5,       "cumartesi": 5, "星期六": 5, "周六": 5,
    "sun": 6, "sunday": 6,    "sonntag": 6,    "dimanche": 6, "domenica": 6,  "domingo": 6,   "zondag": 6,    "söndag": 6,  "niedziela": 6,    "pazar": 6,     "星期日": 6, "周日": 6 }
CONFIG_HMS: dict[str, int] = {
    # EN(a)   # EN         # # DE          # FR           IT            # ES           # PT           # NL           # SE           # PL          # TR            # CN
    "h": 0,   "hour": 0,    "stunde": 0,   "heure": 0,    "ora": 0,     "hora": 0,     "hora": 0,     "uur": 0,      "timme": 0,    "godzina": 0, "saat": 0,      "时": 0,
    "hr": 0,  "hours": 0,   "stunden": 0,  "heures": 0,   "ore": 0,     "horas": 0,    "horas": 0,    "uren": 0,     "timmar": 0,   "godziny": 0, "saatler": 0,   "小时": 0,
    "m": 1,   "minute": 1,  "minute": 1,   "minute": 1,   "minuto": 1,  "minuto": 1,   "minuto": 1,   "minuut": 1,   "minut": 1,    "minuta": 1,  "dakika": 1,    "分": 1,
    "min": 1, "minutes": 1, "minuten": 1,  "minutes": 1,  "minuti": 1,  "minutos": 1,  "minutos": 1,  "minuten": 1,  "minuter": 1,  "minuty": 1,  "dakikalar": 1, "分钟": 1,
    "s": 2,   "second": 2,  "sekunde": 2,  "seconde": 2,  "secondo": 2, "segundo": 2,  "segundo": 2,  "seconde": 2,  "sekund": 2,   "sekunda": 2, "saniye": 2,    "秒": 2,
    "sec": 2, "seconds": 2, "sekunden": 2, "secondes": 2, "secondi": 2, "segundos": 2, "segundos": 2, "seconden": 2, "sekunder": 2, "sekundy": 2, "saniyeler": 2, 
                                                                                                                                    "godzin": 0,                                                            
    
}
CONFIG_AMPM: dict[str, int] = {
    # EN(a)  # EN(a)  #EN             # DE             # IT             # ES         # PT        # NL          # SE              # PL             # TR          # CN
    "a": 0,  "am": 0, "morning": 0,   "morgen": 0,     "mattina": 0,    "mañana": 0, "manhã": 0, "ochtend": 0, "morgon": 0,      "rano": 0,       "sabah": 0,   "上午": 0,
    "p": 1,  "pm": 1, "afternoon": 1, "nachmittag": 1, "pomeriggio": 1, "tarde": 1,  "tarde": 1, "middag": 1,  "eftermiddag": 1, "popołudnie": 1, "öğleden": 1, "下午": 1 }
CONFIG_TZINFO: dict[str, int] = {
    "utc": 0, # Universal Time Coordinate
    "pst": -8 * 3_600, # Pacific Standard Time
    "cet":  1 * 3_600, # Central European Time
}
# fmt: on
# . datetime
US_FRACTION_CORRECTION: cython.uint[5] = [100000, 10000, 1000, 100, 10]
# . timezone
TIMEZONE_NAME_LOCAL: set[str] = set(time.tzname)
# . charactor
CHAR_NULL: cython.Py_UCS4 = 0  # '' null
CHAR_SPACE: cython.Py_UCS4 = 32  # " "
CHAR_PLUS: cython.Py_UCS4 = 43  # "+"
CHAR_COMMA: cython.Py_UCS4 = 44  # ","
CHAR_DASH: cython.Py_UCS4 = 45  # "-"
CHAR_PERIOD: cython.Py_UCS4 = 46  # "."
CHAR_SLASH: cython.Py_UCS4 = 47  # "/"
CHAR_COLON: cython.Py_UCS4 = 58  # ":"
CHAR_LOWER_T: cython.Py_UCS4 = 116  # "t"
CHAR_LOWER_W: cython.Py_UCS4 = 119  # "w"
CHAR_LOWER_Z: cython.Py_UCS4 = 122  # "z"
# . type
TP_PARSERINFO: object = parserinfo


# ISO Format ----------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_iso_sep(char: cython.Py_UCS4) -> cython.bint:
    """Check if the charactor is the separator for ISO format
    between date & time ("T" or " ") `<bool>`"""
    return char == CHAR_SPACE or char == CHAR_LOWER_T


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_isodate_sep(char: cython.Py_UCS4) -> cython.bint:
    """Check if a charactor is the separator for ISO format
    date part ("-") `<bool>`"""
    return char == CHAR_DASH or char == CHAR_SLASH


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_isoweek_sep(char: cython.Py_UCS4) -> cython.bint:
    """Check if a charactor is the separator for ISO format
    week ("W") identifier `<bool>`"""
    return char == CHAR_LOWER_W


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_isotime_sep(char: cython.Py_UCS4) -> cython.bint:
    """Check if a charactor is the separator for ISO format
    time part (":") `<bool>`"""
    return char == CHAR_COLON


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def parse_us_fraction(us_frac_str: str, us_frac_len: cython.uint = 0) -> cython.uint:
    """Parse the microsecond fraction string ('0001' or '000001'),
    and automatically adjust the microsecond value based on the
    fraction length `<int>`."""
    # Validate 'us_str'
    if us_frac_len == 0:
        us_frac_len = str_len(us_frac_str)
    if us_frac_len < 1:
        raise ValueError("microsecond fraction is empty: %s." % repr(us_frac_str))
    if us_frac_len > 6:
        us_frac_len = 6
        us_frac_str = us_frac_str[0:6]
    # Parse microsecond
    try:
        val: cython.uint = int(us_frac_str)
    except Exception as err:
        raise ValueError(
            "Invalid microsecond fraction: %s." % repr(us_frac_str)
        ) from err
    # Adjust fraction
    if us_frac_len < 6:
        val *= US_FRACTION_CORRECTION[us_frac_len - 1]
    return val


# Timelex -------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_ascii_digit(char: cython.Py_UCS4) -> cython.bint:
    """Check if the charactor is an ASCII digit number `<bool>`"""
    return 48 <= char <= 57


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_ascii_alpha(char: cython.Py_UCS4) -> cython.bint:
    return is_ascii_alpha_upper(char) or is_ascii_alpha_lower(char)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_ascii_alpha_upper(char: cython.Py_UCS4) -> cython.bint:
    """Check if the charactor is an ASCII [A-Z] in uppercase `<bool>`."""
    return 65 <= char <= 90


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_ascii_alpha_lower(char: cython.Py_UCS4) -> cython.bint:
    """Check if the charactor is an ASCII [a-z] in lowercase `<bool>`."""
    return 97 <= char <= 122


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def str_count(s: str, char: str) -> cython.uint:
    """Count the number of occurrences of a character in a string `<int>`.
    Equivalent to `s.count(char)`."""
    return s.count(char)


@cython.cfunc
@cython.inline(True)
@cython.wraparound(True)
def parse_timelex(dtstr: str, length: cython.uint = 0) -> list[str]:
    """This function breaks the time string into lexical units (tokens),
    which can be parsed by the Parser. Lexical units are demarcated by
    changes in the character set, so any continuous string of letters or
    number is considered one unit `list[str]>`."""
    # Validate dtstr
    if dtstr is None:
        raise errors.InvalidDatetimeStrError(
            "Only support 'dtstr' as a string, instead "
            "got: {} {}.".format(type(dtstr), dtstr)
        )
    if length == 0:
        length = str_len(dtstr)
    tokens: list[str] = []
    index: cython.int = -1
    max_index: cython.int = length - 1
    temp_char: cython.Py_UCS4 = CHAR_NULL  # '' null

    # Main string loop
    while index <= max_index:
        curr_char: cython.Py_UCS4
        has_alpha: cython.bint = False
        token_state: cython.int = 0
        token: str = None

        # Nested token loop
        while index <= max_index:
            # Retrieve the charactor for the current token.
            if temp_char == CHAR_NULL:
                index += 1
                if index > max_index:
                    # Reached end of the string:
                    # 1. exit the nested token loop.
                    # 2. main loop will also be stopped.
                    break
                while (curr_char := str_loc(dtstr, index)) == CHAR_NULL:
                    index += 1
                    if index > max_index:
                        break
                if not curr_char:
                    # No more valid charactor:
                    # 1. exit the nested token loop.
                    # 2. main loop will also be stopped.
                    break
            # Retrieve the cached charactor for the next token.
            else:
                curr_char, temp_char = temp_char, CHAR_NULL

            # Token state 0: the 1st charactor of the token.
            if token_state == 0:
                # . assign the 1st charactor to the currnet token.
                token = str_fr_ucs4(curr_char)
                if curr_char.isalpha():
                    token_state = 1  # alpha token
                elif curr_char.isdigit():
                    token_state = 2  # digit token
                elif curr_char.isspace():
                    token = " "
                    break  # exit token loop: space token
                else:
                    break  # exit token loop: single charactor token

            # Token state 1: alpha token
            elif token_state == 1:
                has_alpha = True  # mark the token contains alpha.
                if curr_char.isalpha():
                    token += str_fr_ucs4(curr_char)
                elif curr_char == CHAR_PERIOD:  # "."
                    token += "."
                    token_state = 3  # alpha token w/t "."
                else:
                    # exit token loop, and cache the
                    # charactor for the next token.
                    temp_char = curr_char
                    break

            # Token state 2: digit token
            elif token_state == 2:
                if curr_char.isdigit():
                    token += str_fr_ucs4(curr_char)
                elif curr_char == CHAR_PERIOD:  # "."
                    token += "."
                    token_state = 4  # digit token w/t "."
                elif curr_char == CHAR_COMMA and str_len(token) >= 2:  # ","
                    token += ","
                    token_state = 4  # digit token w/t ","
                else:
                    # exit token loop, and cache the
                    # charactor for the next token.
                    temp_char = curr_char
                    break

            # Token state 3: alpha token w/t "."
            elif token_state == 3:
                has_alpha = True  # mark the token contains alpha.
                if curr_char == CHAR_PERIOD or curr_char.isalpha():
                    token += str_fr_ucs4(curr_char)
                elif curr_char.isdigit() and token[-1] == ".":
                    token += str_fr_ucs4(curr_char)
                    token_state = 4  # digit token w/t "."
                else:
                    # exit token loop, and cache the
                    # charactor for the next token.
                    temp_char = curr_char
                    break

            # Token state 4: digit token w/t "."
            elif token_state == 4:
                if curr_char == CHAR_PERIOD or curr_char.isdigit():
                    token += str_fr_ucs4(curr_char)
                elif curr_char.isalpha() and token[-1] == ".":
                    token += str_fr_ucs4(curr_char)
                    token_state = 3  # alpha token w/t "."
                else:
                    # exit token loop, and cache the
                    # charactor for the next token.
                    temp_char = curr_char
                    break

        # Further handle token with "." / ","
        if 3 <= token_state <= 4:
            if has_alpha or token[-1] in ".," or str_count(token, ".") > 1:
                tok: str = None
                for i in range(str_len(token)):
                    chr: cython.Py_UCS4 = str_loc(token, i)
                    if chr == CHAR_PERIOD:  # "."
                        if tok is not None:
                            tokens.append(tok)
                            tok = None
                        tokens.append(".")
                    elif chr == CHAR_COMMA:  # ","
                        if tok is not None:
                            tokens.append(tok)
                            tok = None
                        tokens.append(",")
                    elif tok is not None:
                        tok += str_fr_ucs4(chr)
                    else:
                        tok = str_fr_ucs4(chr)
                if tok is not None:
                    tokens.append(tok)
            else:
                if token_state == 4 and not str_contains(token, "."):
                    token = str_replace(token, ",", ".", -1)
                tokens.append(token)

        # Token that is None means the end of the charactor set.
        elif token is not None:
            tokens.append(token)

    # Return the time lexical tokens
    return tokens


# Result --------------------------------------------------------------------------------------
@cython.cclass
class Result:
    """Represents the parsed result form the Parser."""

    # Y/M/D
    _ymd: cython.int[3]
    _ymd_idx: cython.int
    _ymd_yidx: cython.int
    _ymd_midx: cython.int
    _ymd_didx: cython.int
    # Result
    year: cython.int
    month: cython.int
    day: cython.int
    weekday: cython.int
    hour: cython.int
    minute: cython.int
    second: cython.int
    microsecond: cython.int
    ampm: cython.int
    tzname: str
    tzoffset: cython.int
    _century_specified: cython.bint

    def __init__(self) -> None:
        """The parsed result form the Parser."""
        # Y/M/D
        self._ymd = [-1, -1, -1]
        self._ymd_idx = -1
        self._ymd_yidx = -1
        self._ymd_midx = -1
        self._ymd_didx = -1
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
        self.tzname = None
        self.tzoffset = -100_000
        self._century_specified = False

    # Y/M/D -----------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def append_ymd(self, value: object, label: cython.uint) -> cython.bint:
        """Append a Y/M/D value. Returns True if append successfully,
        else False if Y/M/D slots (max 3) are full `<bool>`.

        :param value `<str/int>`: A Y/M/D value.
        :param label `<int>`: The label for the value:
            - label=0: unknown
            - label=1: year
            - label=2: month
            - label=3: day
        """
        # Validate the value
        if self._ymd_idx >= 2:
            return False  # exit: Y/M/D slots are full
        try:
            val: cython.int = int(value)
        except Exception as err:
            raise ValueError(
                "Invalid Y/M/D value to append to the Result: %s" % repr(value)
            ) from err

        # Pre-determine the year label
        if val >= 100 or (isinstance(value, str) and str_len(value) > 2):
            self._century_specified = True
            label = 1  # year label

        # Set & label Y/M/D value
        self._set_ymd(val)
        if label == 0:
            pass
        elif label == 1 and self._ymd_yidx == -1:
            self._ymd_yidx = self._ymd_idx
        elif label == 2 and self._ymd_midx == -1:
            self._ymd_midx = self._ymd_idx
        elif label == 3 and self._ymd_didx == -1:
            self._ymd_didx = self._ymd_idx
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def ymd_values(self) -> cython.uint:
        """Access the number of Y/M/D values
        that have been stored in the Result `<int>`."""
        return self._ymd_idx + 1

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def could_be_day(self, value: cython.int) -> cython.bint:
        """Determine if a time integer could be the
        day of the date `<bool>`."""
        # Day value already set.
        if self._ymd_didx >= 0:
            return False
        # Month value not set & value in range.
        elif self._ymd_midx == -1:
            return 1 <= value <= 31
        # Year value not set & value in range.
        elif self._ymd_yidx == -1:
            month = self._ymd[self._ymd_midx]
            max_days: cython.int = cydt.days_in_month(2000, month)
            return 1 <= value <= max_days
        # Both Y&M value are set & value in range.
        else:
            year = self._ymd[self._ymd_yidx]
            month = self._ymd[self._ymd_midx]
            max_days: cython.int = cydt.days_in_month(year, month)
            return 1 <= value <= max_days

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_ymd(self, value: cython.int) -> cython.bint:
        """(Internal) Set the Y/M/D value. Returns True
        if the Y/M/D value has been set, else False
        if Y/M/D slots (max 3) are full `<bool>`."""
        if self._ymd_idx < 2:
            self._ymd_idx += 1
            self._ymd[self._ymd_idx] = value
            return True
        else:
            return False

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _labeled_ymd(self) -> cython.uint:
        """(Internal) Get the number of Y/M/D values
        that have been labeled (solved) `<int>`."""
        count: cython.int = 0
        if self._ymd_yidx >= 0:
            count += 1
        if self._ymd_midx >= 0:
            count += 1
        if self._ymd_didx >= 0:
            count += 1
        return count

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _resolve_ymd(self, day1st: cython.bint, year1st: cython.bint) -> cython.bint:
        """(Internal) Try the best to sort out which Y/M/D member
        is year, month & day based on the give Y/M/D result `<bool>`."""
        ymd_values: cython.int = self.ymd_values()
        ymd_labeled: cython.int = self._labeled_ymd()

        # All Y/M/D member have been solved already.
        if ymd_values == ymd_labeled > 0:
            self.year = self._ymd[self._ymd_yidx] if self._ymd_yidx >= 0 else -1
            self.month = self._ymd[self._ymd_midx] if self._ymd_midx >= 0 else -1
            self.day = self._ymd[self._ymd_didx] if self._ymd_didx >= 0 else -1

        # Only has one Y/M/D member.
        elif ymd_values == 1:
            if self._ymd_midx >= 0:  # labeled as month
                self.month = self._ymd[0]
            elif self._ymd[0] > 31:  # out of day range
                self.year = self._ymd[0]
            else:
                self.day = self._ymd[0]

        # Have two Y/M/D members.
        elif ymd_values == 2:
            # . with month label
            if self._ymd_midx >= 0:
                if self._ymd_midx == 0:
                    self.month = self._ymd[0]
                    if self._ymd[1] > 31:
                        # Jan-99
                        self.year = self._ymd[1]
                    else:
                        # Jan-01
                        self.day = self._ymd[1]
                else:
                    self.month = self._ymd[1]
                    if self._ymd[0] > 31:
                        # 99-Jan
                        self.year = self._ymd[0]
                    else:
                        # 01-Jan
                        self.day = self._ymd[0]
            # . without month label
            elif self._ymd[0] > 31:
                # 99-Jan
                self.year, self.month = self._ymd[0], self._ymd[1]
            elif self._ymd[1] > 31:
                # Jan-99
                self.month, self.year = self._ymd[0], self._ymd[1]
            elif day1st and 1 <= self._ymd[1] <= 12:
                # 01-Jan
                self.day, self.month = self._ymd[0], self._ymd[1]
            else:
                # Jan-01
                self.month, self.day = self._ymd[0], self._ymd[1]

        # Have all three Y/M/D member
        elif ymd_values == 3:
            # . lack of one label
            if ymd_labeled == 2:
                if self._ymd_yidx >= 0:  # year is labeled
                    self.year = self._ymd[self._ymd_yidx]
                    if self._ymd_midx >= 0:  # month is labeled
                        self.month = self._ymd[self._ymd_midx]
                        self.day = self._ymd[3 - self._ymd_yidx - self._ymd_midx]
                    else:  # day is labeled
                        self.day = self._ymd[self._ymd_didx]
                        self.month = self._ymd[3 - self._ymd_yidx - self._ymd_didx]
                elif self._ymd_midx >= 0:  # month is labeled
                    self.month = self._ymd[self._ymd_midx]
                    if self._ymd_yidx >= 0:  # year is labeled
                        self.year = self._ymd[self._ymd_yidx]
                        self.day = self._ymd[3 - self._ymd_yidx - self._ymd_midx]
                    else:  # day is labeled
                        self.day = self._ymd[self._ymd_didx]
                        self.year = self._ymd[3 - self._ymd_midx - self._ymd_didx]
                else:  # day is labeled
                    self.day = self._ymd[self._ymd_didx]
                    if self._ymd_yidx >= 0:  # year is labeled
                        self.year = self._ymd[self._ymd_yidx]
                        self.month = self._ymd[3 - self._ymd_yidx - self._ymd_didx]
                    else:  # month is labeled
                        self.month = self._ymd[self._ymd_midx]
                        self.year = self._ymd[3 - self._ymd_midx - self._ymd_didx]
            # . lack more than one labels
            elif self._ymd_midx == 0:
                if self._ymd[1] > 31:
                    # Apr-2003-25
                    self.month, self.year, self.day = (
                        self._ymd[0],
                        self._ymd[1],
                        self._ymd[2],
                    )
                else:
                    # Apr-25-2003
                    self.month, self.day, self.year = (
                        self._ymd[0],
                        self._ymd[1],
                        self._ymd[2],
                    )
            elif self._ymd_midx == 1:
                if self._ymd[0] > 31 or (year1st and 0 < self._ymd[2] <= 31):
                    # 99-Jan-01
                    self.year, self.month, self.day = (
                        self._ymd[0],
                        self._ymd[1],
                        self._ymd[2],
                    )
                else:
                    # 01-Jan-99
                    self.day, self.month, self.year = (
                        self._ymd[0],
                        self._ymd[1],
                        self._ymd[2],
                    )
            elif self._ymd_midx == 2:
                if self._ymd[1] > 31:
                    # 01-99-Jan
                    self.day, self.year, self.month = (
                        self._ymd[0],
                        self._ymd[1],
                        self._ymd[2],
                    )
                else:
                    # 99-01-Jan
                    self.year, self.day, self.month = (
                        self._ymd[0],
                        self._ymd[1],
                        self._ymd[2],
                    )
            else:
                if (
                    self._ymd[0] > 31
                    or self._ymd_yidx == 0
                    or (year1st and 0 < self._ymd[1] <= 12 and 0 < self._ymd[2] <= 31)
                ):
                    if day1st and 0 < self._ymd[2] <= 12:
                        # 99-01-Jan
                        self.year, self.day, self.month = (
                            self._ymd[0],
                            self._ymd[1],
                            self._ymd[2],
                        )
                    else:
                        # 99-Jan-01
                        self.year, self.month, self.day = (
                            self._ymd[0],
                            self._ymd[1],
                            self._ymd[2],
                        )
                elif self._ymd[0] > 12 or (day1st and 0 < self._ymd[1] <= 12):
                    # 01-Jan-99
                    self.day, self.month, self.year = (
                        self._ymd[0],
                        self._ymd[1],
                        self._ymd[2],
                    )
                else:
                    # Jan-01-99
                    self.month, self.day, self.year = (
                        self._ymd[0],
                        self._ymd[1],
                        self._ymd[2],
                    )

        # Swap month & day (if necessary)
        if self.month > 12 and 1 <= self.day <= 12:
            self.month, self.day = self.day, self.month

        # Adjust year to current century (if necessary)
        if 0 <= self.year < 100 and not self._century_specified:
            curr_year: cython.int = cytime.localtime().tm_year
            century: cython.int = curr_year // 100 * 100
            self.year += century
            # . too far into the future
            if self.year >= curr_year + 50:
                self.year -= 100
            # . too distance from the past
            elif self.year < curr_year - 50:
                self.year += 100

        # Finished
        return True

    # Result ----------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def prepare(self, day1st: cython.bint, year1st: cython.bint) -> cython.bint:
        """Prepare the parsed datetime result. Must call this
        method before accessing the result values `<bool>`.
        """
        self._resolve_ymd(day1st, year1st)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def is_valid(self) -> cython.bint:
        """Check if the result is valid (contains any parsed values) `<bool>`."""
        return (
            self.year != -1
            or self.month != -1
            or self.day != -1
            or self.hour != -1
            or self.minute != -1
            or self.second != -1
            or self.microsecond != -1
            or self.weekday != -1
            or self.ampm != -1
            or self.tzname is not None
            or self.tzoffset != -100_000
        )

    # Special methods -------------------------------------------------
    def __repr__(self) -> str:
        # Representations
        reprs: list[str] = []

        if self.year != -1:
            reprs.append("year=%d" % self.year)
        if self.month != -1:
            reprs.append("month=%d" % self.month)
        if self.day != -1:
            reprs.append("day=%d" % self.day)
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
        if self.tzname is not None:
            reprs.append("tzname='%s'" % self.tzname)
        if self.tzoffset != -100_000:
            reprs.append("tzoffset=%d" % self.tzoffset)

        # Construct
        return "<%s (%s)>" % (self.__class__.__name__, ", ".join(reprs))

    def __bool__(self) -> bool:
        return self.is_valid()


# Config --------------------------------------------------------------------------------------
@cython.cclass
class Config:
    """Represents the configurations of the Parser."""

    # Settings
    _day1st: cython.bint
    _year1st: cython.bint
    _pertain: set[str]
    _jump: set[str]
    _utc: set[str]
    _month: dict[str, int]
    _weekday: dict[str, int]
    _hms: dict[str, int]
    _ampm: dict[str, int]
    _tzinfo: dict[str, int]
    # Keywords
    _keywords: set[str]

    def __init__(self, day1st: bool = False, year1st: bool = False) -> None:
        """The configurations of the Parser.

        ### Ambiguous Y/M/D
        :param day1st `<bool>`: Whether to interpret first ambiguous date values as day. Defaults to `False`.
        :param year1st `<bool>`: Whether to interpret first the ambiguous date value as year. Defaults to `False`.

        Both the 'day1st' & 'year1st' arguments works together to determine how
        to interpret ambiguous Y/M/D values.

        In the case when all three values are ambiguous (e.g. `01/05/09`):
        - If 'day1st=True' and 'year1st=True', the date will be interpreted as `'Y/D/M'`.
        - If 'day1st=False' and 'year1st=True', the date will be interpreted as `'Y/M/D'`.
        - If 'day1st=True' and 'year1st=False', the date will be interpreted as `'D/M/Y'`.
        - If 'day1st=False' and 'year1st=False', the date will be interpreted as `'M/D/Y'`.

        In the case when the year value is clear (e.g. `2010/01/05` or `99/01/05`):
        - If 'day1st=True', the date will be interpreted as `'Y/D/M'`.
        - If 'day1st=False', the date will be interpreted as `'Y/M/D'`.

        In the case when only one value is ambiguous (e.g. `01/20/2010` or `01/20/99`):
        - There is no need to set 'day1st' or 'year1st', the date should be interpreted correctly.

        ### Configurations
        Besides the 'day1st' & 'year1st' arguments, Config also provides 'add_xxx()',
        'rem_xxx()' and 'reset_xxx()' methods to modify the following settings:
        - pertain: Words that should be recognized as pertain, e.g: `'of'`.
        - jump: Words that should be skipped, e.g: `'and'`, `'at'`, `'on'`.
        - utc: Words that should be recognized as UTC timezone, e.g: `'utc'`, `'gmt'`.
        - tzinfo: Words that should be recognized as timezone and constrcted
          with the specified timezone offset in seconds, e.g: `'est'`, `'pst'`.
        - month: Words that should be recognized as month, e.g: `'january'`, `'february'`.
        - weekday: Words that should be recognized as weekday, e.g: `'monday'`, `'tuesday'`.
        - hms: Words that should be recognized as HH/MM/SS, e.g: `'hour'`, `'minute'`.
        - ampm: Words that should be recognized as AM/PM, e.g: `'am'`, `'pm'`.

        ### Import Settings
        For user who wants to use an existing '<dateutil.parser.parserinfo>'
        settings, Config provides the 'import_parserinfo' method to bridge
        the compatibility with the 'dateutil' libaray.

        >>> from cytimes import Config
        >>> from dateutil.parser import parserinfo
        >>> info = parserinfo()
            cfg = Config()
            cfg.import_parserinfo(info)
        """
        # Settings
        self._day1st = bool(day1st)
        self._year1st = bool(year1st)
        self._pertain = CONFIG_PERTAIN
        self._jump = CONFIG_JUMP
        self._utc = CONFIG_UTC
        self._month = CONFIG_MONTH
        self._weekday = CONFIG_WEEKDAY
        self._hms = CONFIG_HMS
        self._ampm = CONFIG_AMPM
        self._tzinfo = CONFIG_TZINFO
        # Keywords
        self._construct_keywords()

    # Ambiguous -------------------------------------------------------
    @property
    def day1st(self) -> bool:
        """Whether to interpret first ambiguous date values as day `<bool>`."""
        return self._day1st

    @property
    def year1st(self) -> bool:
        """Whether to interpret first ambiguous date values as year `<bool>`."""
        return self._year1st

    # Pertain ---------------------------------------------------------
    @property
    def pertain(self) -> set[str]:
        """The words that should be recognized as pertain `<set[str]>`.

        ### Example
        >>> cfg.pertain
        >>> {"of"}
        """
        return self._pertain

    def add_pertain(self, *words: str) -> None:
        """Add words that should be recognized as pertain.

        ### Example
        >>> cfg.add_pertain("of", ...)
        """
        for word in words:
            word = self._validate_keyword("pertain", word)
            set_add(self._pertain, word)

    def rem_pertain(self, *words: str) -> None:
        """Remove words that should be recognized as pertain.

        ### Example
        >>> cfg.rem_pertain("of", ...)
        """
        for word in words:
            set_discard(self._pertain, word)
            set_discard(self._keywords, word)

    def reset_pertain(self, *words: str) -> None:
        """Reset the words that should be recognized as pertain.
        If 'words' are not specified, resets to default setting.

        ### Example
        >>> cfg.reset_pertain("of", ...)
        """
        if words:
            self._pertain = set(words)
        else:
            self._pertain = CONFIG_PERTAIN
        self._construct_keywords()

    # Jump ------------------------------------------------------------
    @property
    def jump(self) -> set[str]:
        """The words that should be skipped `<set[str]>`.

        ### Example
        >>> cfg.jump
        >>> {
                " ", ".", ",", ";", "-", "/", "'",
                "at", "on", "and", "ad", "t", "st",
                "nd", "rd", "th"
            }
        """
        return self._jump

    def add_jump(self, *words: str) -> None:
        """Add words that should be skipped.

        ### Example
        >>> cfg.add_jump("at", "on", ...)
        """
        for word in words:
            word = self._validate_keyword("jump", word)
            set_add(self._jump, word)

    def rem_jump(self, *words: str) -> None:
        """Remove words that should be skipped.

        ### Example
        >>> cfg.rem_jump("at", "on", ...)
        """
        for word in words:
            set_discard(self._jump, word)
            set_discard(self._keywords, word)

    def reset_jump(self, *words: str) -> None:
        """Reset the words that should be skipped.
        If 'words' are not specified, resets to default setting.

        ### Example
        >>> cfg.reset_jump("at", "on", ...)
        """
        if words:
            self._jump = set(words)
        else:
            self._jump = CONFIG_JUMP
        self._construct_keywords()

    # UTC -------------------------------------------------------------
    @property
    def utc(self) -> set[str]:
        """The words that should be recognized as UTC timezone `<set[str]>`.

        ### Example
        >>> cfg.utc
        >>> {"utc", "gmt", "z"}
        """
        return self._utc

    def add_utc(self, *words: str) -> None:
        """Add words that should be recognized as UTC timezone.

        ### Example
        >>> cfg.add_utc("utc", "gmt", "z", ...)
        """
        for word in words:
            word = self._validate_keyword("utc", word)
            set_add(self._utc, word)

    def rem_utc(self, *words: str) -> None:
        """Remove words that should be recognized as UTC timezone.

        ### Example
        >>> cfg.rem_utc("utc", "gmt", "z", ...)
        """
        for word in words:
            set_discard(self._utc, word)
            set_discard(self._keywords, word)

    def reset_utc(self, *words: str) -> None:
        """Reset the words that should be recognized as UTC timezone.
        If 'words' are not specified, resets to default setting.

        ### Example
        >>> cfg.reset_utc("utc", "gmt", "z", ...)
        """
        if words:
            self._utc = set(words)
        else:
            self._utc = CONFIG_UTC
        self._construct_keywords()

    # Month -----------------------------------------------------------
    @property
    def month(self) -> dict[str, int]:
        """The words that should be recognized as month `<dict[str, int]>`.

        ### Example
        >>> cfg.month
        >>> {
                "january": 1,
                "jan": 1,
                "february": 2,
                "feb": 2,
                ...
            }
        """
        return self._month

    def add_month(self, month: int, *words: str) -> None:
        """Add words that should be recognized as a specific month.

        ### Example
        >>> cfg.add_month(1, "january", "jan", ...)
        """
        month = self._validate_value("month", month, 1, 12)
        for word in words:
            word = self._validate_keyword("month", word)
            dict_setitem(self._month, word, month)

    def rem_month(self, *words: str) -> None:
        """Remove words that should be recognized as month.

        ### Example
        >>> cfg.rem_month("january", "jan", ...)
        """
        for word in words:
            try:
                dict_delitem(self._month, word)
            except KeyError:
                pass
            set_discard(self._keywords, word)

    def reset_month(self, **word_n_month: int) -> None:
        """Reset the words that should be recognized as month.
        If 'word_n_month' are not specified, resets to default
        setting.

        ### Example
        >>> cfg.reset_month(
                january=1, jan=1,
                february=2, feb=2,
                ...
            )
        """
        if word_n_month:
            self._month = word_n_month
        else:
            self._month = CONFIG_MONTH
        self._construct_keywords()

    # Weekday ---------------------------------------------------------
    @property
    def weekday(self) -> dict[str, int]:
        """The words that should be recognized as weekday,
        where 0=Monday...6=Sunday. `<dict[str, int]>`.

        ### Example
        >>> cfg.weekday
        >>> {
                "monday": 0,
                "mon": 0,
                "tuesday": 1,
                "tue": 1,
                ...
            }
        """
        return self._weekday

    def add_weekday(self, weekday: int, *words: str) -> None:
        """Add words that should be recognized as a specific
        weekday, where 0=Monday...6=Sunday.

        ### Example
        >>> cfg.add_weekday(0, "monday", "mon", ...)
        """
        weekday = self._validate_value("weekday", weekday, 0, 6)
        for word in words:
            word = self._validate_keyword("weekday", word)
            dict_setitem(self._weekday, word, weekday)

    def rem_weekday(self, *words: str) -> None:
        """Remove words that should be recognized as weekday.

        ### Example
        >>> cfg.rem_weekday("monday", "mon", ...)
        """
        for word in words:
            try:
                dict_delitem(self._weekday, word)
            except KeyError:
                pass
            set_discard(self._keywords, word)

    def reset_weekday(self, **word_n_weekday: int) -> None:
        """Reset the words that should be recognized as weekday,
        where 0=Monday...6=Sunday. If 'word_n_weekday' are not
        specified, resets to default setting.

        ### Example
        >>> cfg.reset_weekday(
                monday=0, mon=0,
                tuesday=1, tue=1,
                ...
            )
        """
        if word_n_weekday:
            self._weekday = word_n_weekday
        else:
            self._weekday = CONFIG_WEEKDAY
        self._construct_keywords()

    # HMS ------------------------------------------------------------
    @property
    def hms(self) -> dict[str, int]:
        """The words that should be recognized as HH/MM/SS,
        where 0=hour, 1=minute, 2=second. `<dict[str, int]>`.

        ### Example
        >>> cfg.hms
        >>> {
                "hour": 0,
                "minute": 1,
                "second": 2,
                ...
            }
        """
        return self._hms

    def add_hms(self, hms: int, *words: str) -> None:
        """Add words that should be recognized as HH/MM/SS,
        where 0=hour, 1=minute, 2=second.

        ### Example
        >>> cfg.add_hms(0, "hour", "hours", "h", ...)
        """
        hms = self._validate_value("hms", hms, 0, 2)
        for word in words:
            word = self._validate_keyword("hms", word)
            dict_setitem(self._hms, word, hms)

    def rem_hms(self, *words: str) -> None:
        """Remove words that should be recognized as HH/MM/SS.

        ### Example
        >>> cfg.rem_hms("hour", "hours", "h", ...)
        """
        for word in words:
            try:
                dict_delitem(self._hms, word)
            except KeyError:
                pass
            set_discard(self._keywords, word)

    def reset_hms(self, **word_n_hms: int) -> None:
        """Reset the words that should be recognized as HH/MM/SS,
        where 0=hour, 1=minute, 2=second. If 'word_n_hms' are not
        specified, resets to default setting.

        ### Example
        >>> cfg.reset_hms(
                hour=0, hours=0, h=0,
                minute=1, minutes=1, min=1,
                second=2, seconds=2, sec=2,
                ...
            )
        """
        if word_n_hms:
            self._hms = word_n_hms
        else:
            self._hms = CONFIG_HMS
        self._construct_keywords()

    # AM/PM ----------------------------------------------------------
    @property
    def ampm(self) -> dict[str, int]:
        """The words that should be recognized as AM/PM,
        where 0=AM and 1=PM. `<dict[str, int]>`.

        ### Example
        >>> cfg.ampm
        >>> {
                "am": 0,
                "pm": 1,
                ...
            }
        """
        return self._ampm

    def add_ampm(self, ampm: int, *words: str) -> None:
        """Add words that should be recognized as AM/PM,
        where 0=AM and 1=PM.

        ### Example
        >>> cfg.add_ampm(0, "am", "a.m.", ...)
        """
        ampm = self._validate_value("ampm", ampm, 0, 1)
        for word in words:
            word = self._validate_keyword("ampm", word)
            dict_setitem(self._ampm, word, ampm)

    def rem_ampm(self, *words: str) -> None:
        """Remove words that should be recognized as AM/PM.

        ### Example
        >>> cfg.rem_ampm("am", "a.m.", ...)
        """
        for word in words:
            try:
                dict_delitem(self._ampm, word)
            except KeyError:
                pass
            set_discard(self._keywords, word)

    def reset_ampm(self, **word_n_ampm: int) -> None:
        """Reset the words that should be recognized as AM/PM,
        where 0=AM and 1=PM. If 'word_n_ampm' are not specified,
        resets to default setting.

        ### Example
        >>> cfg.reset_ampm(
                am=0, pm=1,
                **{"a.m."=0, "p.m."=1}
                ...
            )
        """
        if word_n_ampm:
            self._ampm = word_n_ampm
        else:
            self._ampm = CONFIG_AMPM
        self._construct_keywords()

    # Tzinfo ----------------------------------------------------------
    @property
    def tzinfo(self) -> dict[str, int]:
        """The words that should be recognized as a timezone
        and constrcted with the specified timezone offset
        in seconds `<dict[str, int]>`.

        ### Example
        >>> cfg.tzinfo
        >>> {
                'est': -18000,
                'edt': -14400,
                ...
            }
        """
        return self._tzinfo

    def add_tzinfo(self, word: str, hour: int = 0, minute: int = 0) -> None:
        """Add word that should be recognized as a timezone
        and the corresponding timezone offset (hours & minutes).

        ### Example
        >>> cfg.add_tzinfo("est", -5)
        """
        # Validate hour & minute
        if not isinstance(hour, int):
            raise errors.InvalidConfigValue(
                "<{}>\nTimezone offset 'hour' must "
                "be an integer, instead got: {} {}".format(
                    self.__class__.__name__, type(hour), repr(hour)
                )
            )
        if not isinstance(minute, int):
            raise errors.InvalidConfigValue(
                "<{}>\nTimezone offset 'minute' must "
                "be an integer, instead got: {} {}".format(
                    self.__class__.__name__, type(minute), repr(minute)
                )
            )
        tzoffset = hour * 3_600 + minute * 60
        dict_setitem(
            self._tzinfo,
            self._validate_keyword("tzinfo", word),
            self._validate_value("tzinfo", tzoffset, -86_340, 86_340),
        )

    def rem_tzinfo(self, *words: str) -> None:
        """Remove words that should be recognized as timezone.

        ### Example
        >>> cfg.rem_tzinfo("est", "edt", ...
        """
        for word in words:
            try:
                dict_delitem(self._tzinfo, word)
            except KeyError:
                pass
            set_discard(self._keywords, word)

    def reset_tzinfo(self, **word_n_tzoffset: int) -> None:
        """Reset the words that should be recognized as
        timezones and the corresponding timezone offsets
        in seconds. If 'word_n_tzoffset' are not specified,
        resets to default setting.

        ### Example
        >>> cfg.reset_tzinfo(
                est=-18000,
                edt=-14400,
                ...
            )
        """
        if word_n_tzoffset:
            self._tzinfo = word_n_tzoffset
        else:
            self._tzinfo = CONFIG_TZINFO
        self._construct_keywords()

    # Import ----------------------------------------------------------
    def import_parserinfo(self, info: parserinfo) -> None:
        """Import settings from a `<dateutil.parser.parserinfo>` instance.

        This method is designed to bridge the compatibility with
        `dateutil` libaray. After import, current settings of the
        Config will be overwirtten by the specified `parserinfo`.

        ### Example
        >>> from cytimes import Config
        >>> from dateutil.parser import parserinfo
        >>> info = parserinfo()
            cfg = Config()
            cfg.import_parserinfo(info)
        """
        # Validate perserinfo
        if not isinstance(info, TP_PARSERINFO):
            raise errors.InvalidParserInfo(
                "<{}>\nConfig can only import "
                "'dateutil.parser.parserinfo', instead got: "
                "{} {}.".format(self.__class__.__name__, type(info), repr(info))
            )

        # Import settings
        self._day1st = info.dayfirst
        self._year1st = info.yearfirst
        self._pertain = set(info.PERTAIN)
        self._jump = set(info.JUMP)
        self._utc = set(info.UTCZONE)
        self._month = {w: i + 1 for i, wds in enumerate(info.MONTHS) for w in wds}
        self._weekday = {w: i for i, wds in enumerate(info.WEEKDAYS) for w in wds}
        self._hms = {w: i for i, wds in enumerate(info.HMS) for w in wds}
        self._ampm = {w: i for i, wds in enumerate(info.AMPM) for w in wds}
        self._tzinfo = info.TZOFFSET

        # Reconstruct keywords
        self._construct_keywords()

    # Validate --------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _construct_keywords(self) -> cython.bint:
        """(Internal) Construct the the Config keywords.
        Called when the Config is initialized or reset."""
        # Reset the keywords
        self._keywords = set()
        # fmt: off
        # . month
        self._month = {
            self._validate_keyword("month", word):
            self._validate_value("month", value, 1, 12)
            for word, value in self._month.items()
        }
        # . weekday
        self._weekday = {
            self._validate_keyword("weekday", word):
            self._validate_value("weekday", value, 0, 6)
            for word, value in self._weekday.items()
        }
        # . hms
        self._hms = {
            self._validate_keyword("hms", word):
            self._validate_value("hms", value, 0, 2)
            for word, value in self._hms.items()
        }
        # . ampm
        self._ampm = {
            self._validate_keyword("ampm", word):
            self._validate_value("ampm", value, 0, 1)
            for word, value in self._ampm.items()
        }
        # . tzinfo
        self._tzinfo = {
            self._validate_keyword("tzinfo", word):
            self._validate_value("tzinfo", value, -86_340, 86_340)
            for word, value in self._tzinfo.items()
        }
        # . utc
        self._utc = {
            self._validate_keyword("utc", word)
            for word in self._utc }
        
        # . pertain
        self._pertain = {
            self._validate_keyword("pertain", word)
            for word in self._pertain }
        # . jump
        self._jump = {
            self._validate_keyword("jump", word)
            for word in self._jump }
        # fmt: on
        # Finished
        return True

    @cython.cfunc
    @cython.inline(True)
    def _validate_keyword(self, setting: str, word: object) -> object:
        """(Internal) Validate if there are conflicting
        (duplicated) keywords exsit in the settings, which
        could lead to undesirable parsing result."""
        # Validate the keyword type
        if not isinstance(word, str):
            raise errors.InvalidConfigKeyword(
                "<{}>\nThe keyword for [{}] must be a string, "
                "instead got: {} {}".format(
                    self.__class__.__name__, setting, type(word), repr(word)
                )
            )
        word = word.lower()

        # Validate if the keyword is conflicting with other keywords.
        # Skip jump keywords, because jump has the freedom to contain
        # any desired words.
        if set_contains(self._keywords, word) and not set_contains(self._jump, word):
            # locate conflicting keywords
            conflict: str = None
            # Skip jump conflict keywords, jump has
            # the freedom to be duplicated with other
            # settings.
            if dict_contains(self._month, word):
                if setting != "month":
                    conflict = "month"
            elif dict_contains(self._weekday, word):
                if setting != "weekday":
                    conflict = "weekday"
            elif dict_contains(self._hms, word):
                if setting != "hms":
                    conflict = "hms"
            elif dict_contains(self._ampm, word):
                if setting != "ampm":
                    conflict = "ampm"
            elif dict_contains(self._tzinfo, word):
                if setting != "tzinfo":
                    conflict = "tzinfo"
            elif set_contains(self._utc, word):
                if setting != "utc":
                    conflict = "utc"
            elif set_contains(self._pertain, word):
                if setting != "pertain":
                    conflict = "pertain"
            # raise keyword error
            if conflict is not None:
                raise errors.InvalidConfigKeyword(
                    "<{}>\nThe keyword '{}' for [{}] is conflicting "
                    "with keywords in [{}] of the Parser Config.".format(
                        self.__class__.__name__, word, setting, conflict
                    )
                )
        else:
            set_add(self._keywords, word)

        # Return the validated keyword
        return word

    @cython.cfunc
    @cython.inline(True)
    def _validate_value(
        self,
        setting: str,
        value: object,
        min: cython.int,
        max: cython.int,
    ) -> object:
        """(Internal) Validate if the setting keyword has
        a valid value that is within the required range."""
        if not isinstance(value, int):
            raise errors.InvalidConfigValue(
                "<{}>\nThe value for [{}] must be an "
                "integer, instead got: {} {}.".format(
                    self.__class__.__name__, setting, type(value), repr(value)
                )
            )
        val: cython.int = value
        if not min <= val <= max:
            raise errors.InvalidConfigValue(
                "<{}>\nThe value for [{}] must be within "
                "the range of {}..{}, instead got: {}.".format(
                    self.__class__.__name__, setting, min, max, value
                )
            )
        return value

    # Special methods -------------------------------------------------
    def __repr__(self) -> str:
        # Representations
        reprs: list = [
            "day1st=%s" % self._day1st,
            "year1st=%s" % self._year1st,
            "pertain=%s" % sorted(self._pertain),
            "jump=%s" % sorted(self._jump),
            "utc=%s" % sorted(self._utc),
            "month=%s" % self._month,
            "weekday=%s" % self._weekday,
            "hms=%s" % self._hms,
            "ampm=%s" % self._ampm,
            "tzinfo=%s" % self._tzinfo,
        ]
        # Construct
        return "<%s (\n  %s\n)>" % (self.__class__.__name__, ",\n  ".join(reprs))


# Parser --------------------------------------------------------------------------------------
@cython.cclass
class Parser:
    """Represents the datetime Parser."""

    # Config
    _day1st: cython.bint
    _year1st: cython.bint
    _ignoretz: cython.bint
    _fuzzy: cython.bint
    _pertain: set[str]
    _jump: set[str]
    _utc: set[str]
    _month: dict[str, int]
    _weekday: dict[str, int]
    _hms: dict[str, int]
    _ampm: dict[str, int]
    _tzinfo: dict[str, int]
    # Result
    _result: Result
    # Process
    _dtstr: str
    _dtstr_len: cython.uint
    _isodate_type: cython.uint
    _tokens: list[str]
    _tokens_count: cython.int
    _index: cython.int
    _token_r1: str
    _token_r2: str
    _token_r3: str
    _token_r4: str

    def __init__(self, cfg: Config = None) -> None:
        """The datetime Parser.

        :param cfg `<Config/None>`: The configurations of the Parser. Defaults to `None`.
        """
        # Load specifed config
        if cfg is not None:
            self._day1st = cfg._day1st
            self._year1st = cfg._year1st
            self._pertain = cfg._pertain
            self._jump = cfg._jump
            self._utc = cfg._utc
            self._month = cfg._month
            self._weekday = cfg._weekday
            self._hms = cfg._hms
            self._ampm = cfg._ampm
            self._tzinfo = cfg._tzinfo
        # Load default config
        else:
            self._day1st = False
            self._year1st = False
            self._pertain = CONFIG_PERTAIN
            self._jump = CONFIG_JUMP
            self._utc = CONFIG_UTC
            self._month = CONFIG_MONTH
            self._weekday = CONFIG_WEEKDAY
            self._hms = CONFIG_HMS
            self._ampm = CONFIG_AMPM
            self._tzinfo = CONFIG_TZINFO

    # Parsing ------------------------------------------------------------------------------
    @cython.ccall
    def parse(
        self,
        dtstr: str,
        default: datetime.datetime | datetime.date | None = None,
        day1st: bool | None = None,
        year1st: bool | None = None,
        ignoretz: bool = False,
        fuzzy: bool = False,
    ) -> datetime.datetime:
        """Parse a string contains date & time information into `<datetime>`.

        ### Time String & Default
        :param dtstr `<str>`: The string that contains date & time information.
        :param default `<datetime/date>`: The default to fill-in missing datetime elements. Defaults to `None`.
        - `None`: If parser failed to extract Y/M/D values from the string,
           the date of '1970-01-01' will be used to fill-in the missing year,
           month & day values.
        - `<date>`: If parser failed to extract Y/M/D values from the string,
           the give `date` will be used to fill-in the missing year, month &
           day values.
        - `<datetime>`: If parser failed to extract datetime elements from
           the string, the given `datetime` will be used to fill-in the
           missing year, month, day, hour, minute, second and microsecond.

        ### Ambiguous Y/M/D
        :param day1st `<bool>`: Whether to interpret first ambiguous date values as day. Defaults to `None`.
        :param year1st `<bool>`: Whether to interpret first the ambiguous date value as year. Defaults to `None`.

        Both the 'day1st' & 'year1st' arguments works together to determine how
        to interpret ambiguous Y/M/D values. If not provided (set to `None`),
        defaults to the 'day1st' & 'year1st' settings of the Parser `<Config>`.

        In the case when all three values are ambiguous (e.g. `01/05/09`):
        - If 'day1st=True' and 'year1st=True', the date will be interpreted as `'Y/D/M'`.
        - If 'day1st=False' and 'year1st=True', the date will be interpreted as `'Y/M/D'`.
        - If 'day1st=True' and 'year1st=False', the date will be interpreted as `'D/M/Y'`.
        - If 'day1st=False' and 'year1st=False', the date will be interpreted as `'M/D/Y'`.

        In the case when the year value is clear (e.g. `2010/01/05` or `99/01/05`):
        - If 'day1st=True', the date will be interpreted as `'Y/D/M'`.
        - If 'day1st=False', the date will be interpreted as `'Y/M/D'`.

        In the case when only one value is ambiguous (e.g. `01/20/2010` or `01/20/99`):
        - There is no need to set 'day1st' or 'year1st', the date should be interpreted correctly.

        ### Timezone
        :param ignoretz `<bool>`: Whether to ignore timezone information. Defaults to `False`.
        - `True`: Parser ignores any timezone information and only returns
           timezone-naive datetime. Setting to `True` can further increase
           parser performance.
        - `False`: Parser will try to process the timzone information in
           the string, and generate a timezone-aware datetime if timezone
           has been matched by the Parser `<Config>` settings: 'utc' & 'tzinfo'.

        ### Complex Time String
        :param fuzzy `<bool>`: Whether to allow fuzzy parsing. Defaults to `False`.
        - `True`: Parser will increase its flexibility on tokens when parsing
           complex (sentence like) time string, such as:
            * 'On June 8th, 2020, I am going to be the first man on Mars' => <2020-06-08 00:00:00>
            * 'Meet me at the AM/PM on Sunset at 3:00 PM on December 3rd, 2003' => <2003-12-03 15:00:00>
        - `False`: A stricter parsing rule will be applied and complex time
           string can lead to parser failure. However, this mode should be
           able to handle most of the time strings.

        ### Exception
        :raise `cyParserError`: If failed to parse the given `dtstr`.

        ### Return
        :return: `<datetime>` The parsed datetime object.
        """
        # Settings
        if day1st is not None:
            self._day1st = bool(day1st)
        if year1st is not None:
            self._year1st = bool(year1st)
        self._ignoretz = bool(ignoretz)
        self._fuzzy = bool(fuzzy)

        # Validate 'dtstr'
        if dtstr is None:
            raise errors.InvalidDatetimeStrError(
                "Only support 'dtstr' as a string, instead "
                "got: {} {}.".format(type(dtstr), dtstr)
            )
        self._dtstr = dtstr.lower()
        self._dtstr_len = str_len(dtstr)

        # Parsing
        try:
            # ISO format
            if self._process_iso():
                return self._build(default)  # exit: success

            # Core process
            if self._process_core():
                return self._build(default)  # exit: success

        except errors.cyParserError as err:
            err.add_note("-> Unable to parse: %s." % repr(self._dtstr))
            raise err
        except Exception as err:
            raise errors.cyParserFailedError(
                "<{}>\nUnable to parse the datetime string: {}.\n"
                "Error: {}".format(self.__class__.__name__, repr(self._dtstr), err)
            ) from err
        raise errors.cyParserFailedError(
            "<{}>\nUnable to parse the datetime string: "
            "{}.".format(self.__class__.__name__, repr(self._dtstr))
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _process_iso(self) -> cython.bint:
        """(Internal) The iso format process to parse the 'dtstr' `<bool>`."""
        # Find isoformat datetime separator
        self._isodate_type = 0
        sep_loc: cython.uint = self._find_isoformat_sep()
        if sep_loc == 0:
            return False  # exit: not isoformat

        # Parse iso format
        if sep_loc == self._dtstr_len:
            # . parse date component
            self._result = Result()
            if not self._parse_isoformat_date(self._dtstr, sep_loc):
                return False  # exit: not isoformat
        else:
            # . verify datetime seperator (' ' or 'T')
            if not is_iso_sep(self._get_char(sep_loc)):
                return False  # exit: not isoformat
            # . parse date component
            self._result = Result()
            if not self._parse_isoformat_date(
                str_substr(self._dtstr, 0, sep_loc), sep_loc
            ):
                return False  # exit: not isoformat
            # . parse time component
            tstr: str = str_substr(self._dtstr, sep_loc + 1, self._dtstr_len)
            tstr_len: cython.uint = self._dtstr_len - sep_loc - 1
            if not self._parse_isoformat_time(tstr, tstr_len):
                return False  # exit: not isoformat

        # Prepare result
        self._result.prepare(self._day1st, self._year1st)
        return self._result.is_valid()  # exit: success/fail

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _process_core(self) -> cython.bint:
        """(Internal) The core process to parse the 'dtstr' `<bool>`."""
        # Convert dtstr to tokens
        self._tokens = parse_timelex(self._dtstr, self._dtstr_len)
        self._tokens_count = list_len(self._tokens)
        self._index = 0
        self._result = Result()

        # Parse token
        while self._index < self._tokens_count:
            # . access token
            token = self._get_token(self._index)
            # . reset tokens
            self._token_r1 = None
            self._token_r2 = None
            self._token_r3 = None
            self._token_r4 = None

            # . numeric token
            if self._parse_numeric_token(token):
                self._index += 1
            # . month token
            elif self._parse_month_token(token):
                self._index += 1
            # . weekday token
            elif self._parse_weekday_token(token):
                self._index += 1
            # . am/pm token
            elif self._parse_ampm_token(token):
                self._index += 1
            # . tzname token
            elif self._parse_tzname_token(token):
                self._index += 1
            # . tzoffset token
            elif self._parse_tzoffset_token(token):
                self._index += 1
            # . jump token
            elif self._is_token_jump(token):
                self._index += 1
            # . fuzzy parsing
            elif self._fuzzy:
                self._index += 1
            # . failed to parse
            else:
                raise errors.InvalidTokenError(
                    "<{}>\nFailed to parse: {}.\n"
                    "If this is a complex (sentence like) time string, "
                    "try set 'fuzzy=True' to increase parser flexibility.".format(
                        self.__class__.__name__, repr(self._dtstr)
                    )
                )

        # Prepare result
        self._result.prepare(self._day1st, self._year1st)
        return self._result.is_valid()  # exit: success/fail

    @cython.cfunc
    @cython.inline(True)
    def _build(self, default: object) -> datetime.datetime:
        """(Internal) Build the `datetime` based on the
        parsed result `<datetime>`."""
        # Timeonze-naive
        if self._ignoretz:
            return self._build_datetime(default, None)

        # Timezone-aware
        tzname: str = self._result.tzname
        offset: cython.int = self._result.tzoffset
        # . local timezone (handle ambiguous time)
        if tzname is not None and set_contains(TIMEZONE_NAME_LOCAL, tzname):
            # Build with local tzinfo
            dt = self._build_datetime(default, None)
            dt = cydt.dt_replace_tzinfo(dt, cydt.gen_tzinfo_local(dt))
            # Handle ambiguous local datetime
            dt = self._handle_ambiguous_time(dt, tzname)
            # Adjust for winter GMT zones parsed in the UK
            if dt.tzname() != tzname and tzname == "UTC":
                dt = cydt.dt_replace_tzinfo(dt, cydt.UTC)
            return dt  # exit: finished
        # . utc (tzoffset == 0)
        elif offset == 0:
            tzinfo = cydt.UTC
        # . other timezone
        elif offset != -100_000:
            if tzname == "UTC":  # utc
                tzinfo = cydt.gen_tzinfo(offset)
            else:  # Custom timezone name
                tzinfo = cydt.gen_tzinfo(offset, tzname)
        # . timezone-naive
        else:
            tzinfo = None
        # Build with tzinfo
        return self._build_datetime(default, tzinfo)

    @cython.cfunc
    @cython.inline(True)
    def _build_datetime(self, default: object, tzinfo: object) -> datetime.datetime:
        """(Internal) Build the `<datetime>`."""
        # Check if valid default is given
        if cydt.is_date(default):
            if cydt.is_dt(default):
                default_mode: cython.uint = 2  # default is datetime
            else:
                default_mode: cython.uint = 1  # default is date
        else:
            default_mode: cython.uint = 0  # default is local time

        # . year
        if self._result.year > 0:
            year: cython.uint = self._result.year
        elif default_mode > 0:
            year: cython.uint = datetime.PyDateTime_GET_YEAR(default)
        else:
            year: cython.uint = 1970

        # . month
        if self._result.month > 0:
            month: cython.uint = self._result.month
        elif default_mode > 0:
            month: cython.uint = datetime.PyDateTime_GET_MONTH(default)
        else:
            month: cython.uint = 1

        # . day
        if self._result.day > 0:
            day: cython.uint = self._result.day
        elif default_mode > 0:
            day: cython.uint = datetime.PyDateTime_GET_DAY(default)
        else:
            day: cython.uint = 1
        if day > 28:
            day = min(day, cydt.days_in_month(year, month))

        # . hour
        if self._result.hour >= 0:
            hour: cython.uint = self._result.hour
        elif default_mode == 2:
            hour: cython.uint = datetime.PyDateTime_DATE_GET_HOUR(default)
        else:
            hour: cython.uint = 0

        # . minute
        if self._result.minute >= 0:
            minute: cython.uint = self._result.minute
        elif default_mode == 2:
            minute: cython.uint = datetime.PyDateTime_DATE_GET_MINUTE(default)
        else:
            minute: cython.uint = 0

        # . second
        if self._result.second >= 0:
            second: cython.uint = self._result.second
        elif default_mode == 2:
            second: cython.uint = datetime.PyDateTime_DATE_GET_SECOND(default)
        else:
            second: cython.uint = 0

        # . microsecond
        if self._result.microsecond >= 0:
            microsecond: cython.uint = self._result.microsecond
        elif default_mode == 2:
            microsecond: cython.uint = datetime.PyDateTime_DATE_GET_MICROSECOND(default)
        else:
            microsecond: cython.uint = 0

        # Generate datetime
        dt: datetime.datetime = cydt.gen_dt(
            year, month, day, hour, minute, second, microsecond, tzinfo, 0
        )

        # Adjust weekday
        if 0 <= self._result.weekday <= 6:
            dt = cydt.dt_adj_weekday(dt, self._result.weekday)

        # Return datetime
        return dt

    @cython.cfunc
    @cython.inline(True)
    def _handle_ambiguous_time(
        self,
        dt: datetime.datetime,
        tzname: str,
    ) -> datetime.datetime:
        """(Internal) Handle ambiguous time `<datetime>`."""
        if dt.tzname() != tzname:
            new_dt: datetime.datetime = cydt.dt_replace_fold(dt, 1)
            if new_dt.tzname() == tzname:
                return new_dt
        return dt

    # ISO format ---------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _find_isoformat_sep(self) -> cython.uint:
        """(Internal) This function tries to find the date & time separator
        (" " or "T") location for an ISO format datetime string. Returns 0 if
        the string is certainly not under isoformat, else the possible separator
        location of the string `<int>`.

        Also it sets the 'iso_datetype' to enable iso date parser to perform
        quick value parsing.

        Meaning for iso_datetype:
        - 0: Not isoformat
        - 1: YYYY-MM-DD
        - 2: YYYYMMDD
        - 3: YYYY-Www-D
        - 4: YYYYWwwD
        - 5: YYYY-Www
        - 6: YYYYWww
        - 7: YYYY-DDD
        - 8: YYYYDDD
        """
        # ISO format string length must be >= 7.
        if self._dtstr_len < 7:
            return 0  # exit: not isoformat

        # Find datetime separator
        char4: cython.Py_UCS4 = self._get_char(4)
        # YYYY[-]
        if is_isodate_sep(char4):
            if self._dtstr_len < 8:
                return 0  # exit: not isoformat
            # YYYY-[W]
            char5: cython.Py_UCS4 = self._get_char(5)
            if is_isoweek_sep(char5):
                # YYYY-Www[-]
                if self._dtstr_len > 8 and is_isodate_sep(self._get_char(8)):
                    if self._dtstr_len == 9:  # [YYYY-Www-]
                        return 0  # exit: not isoformat
                    elif is_ascii_digit(self._get_char(10)):
                        self._isodate_type = 5
                        return 8  # exit: [YYYY-Www]
                    else:
                        self._isodate_type = 3
                        return 10  # exit: [YYYY-Www-D]
                else:
                    self._isodate_type = 5
                    return 8  # exit: [YYYY-Www]
            # YYYY-[M]
            elif is_ascii_digit(char5):
                char7: cython.Py_UCS4 = self._get_char(7)
                if self._dtstr_len >= 10 and is_isodate_sep(char7):
                    self._isodate_type = 1
                    return 10  # exit: [YYYY-MM-DD]
                elif is_ascii_digit(char7):
                    self._isodate_type = 7
                    return 8  # exit: [YYYY-DDD]

        # YYYY[W]
        elif is_isoweek_sep(char4):
            # YYYYWw[w]
            if is_ascii_digit(self._get_char(6)):
                if is_ascii_digit(self._get_char(7)):
                    self._isodate_type = 4
                    return 8  # exit: [YYYYWwwD]
                else:
                    self._isodate_type = 6
                    return 7  # exit: [YYYYWww]

        # YYYY[D]
        elif is_ascii_digit(char4):
            # YYYYDD[D] / YYYYMM[D]
            if is_ascii_digit(self._get_char(6)):
                if is_ascii_digit(self._get_char(7)):
                    self._isodate_type = 2
                    return 8  # exit: [YYYYMMDD]
                else:
                    self._isodate_type = 8
                    return 7  # exit: [YYYYDDD]

        # Invalid Isoformat
        return 0  # exit: not isoformat

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_isoformat_date(self, dstr: str, length: cython.uint) -> cython.bint:
        """(Internal) Parse the date components of the isoformat string `<bool>`.

        Meaning for iso_datetype:
        - 0: Not isoformat
        - 1: YYYY-MM-DD
        - 2: YYYYMMDD
        - 3: YYYY-Www-D
        - 4: YYYYWwwD
        - 5: YYYY-Www
        - 6: YYYYWww
        - 7: YYYY-DDD
        - 8: YYYYDDD
        """
        # Validate year
        try:
            year: cython.uint = int(dstr[0:4])
        except Exception:
            return False  # exit: invalid year
        if not 1 <= year <= 9_999:
            return False  # exit: invalid year
        self._result.append_ymd(year, 1)

        # Parse month & day
        if self._isodate_type <= 2:
            mth_str: str
            day_str: str
            # . YYYY-MM-DD
            if self._isodate_type == 1 and length == 10:
                mth_str = dstr[5:7]
                day_str = dstr[8:10]
            # . YYYYMMDD
            elif self._isodate_type == 2 and length == 8:
                mth_str = dstr[4:6]
                day_str = dstr[6:8]
            # . Invalid
            else:
                return False  # exit: not isoformat
            # . Validate month
            try:
                month: cython.uint = int(mth_str)
            except Exception:
                return False  # exit: invalid month
            if not 1 <= month <= 12:
                return False  # exit: invalid month
            # . Validate day
            try:
                day: cython.uint = int(day_str)
            except Exception:
                return False  # exit: invalid day
            if not 1 <= day <= cydt.days_in_month(year, month):
                return False
            # Append 'month' & 'day'
            self._result.append_ymd(month, 2)
            self._result.append_ymd(day, 3)
            return True  # exit: success

        # Parse week & [weekday]
        if self._isodate_type <= 6:
            week_str: str
            wkdy_str: str
            # . YYYY-Www-D
            if self._isodate_type == 3 and length == 10:
                week_str = dstr[6:8]
                wkdy_str = dstr[9:10]
            # . YYYYWwwD
            elif self._isodate_type == 4 and length == 8:
                week_str = dstr[5:7]
                wkdy_str = dstr[7:8]
            # . YYYY-Www
            elif self._isodate_type == 5 and length == 8:
                week_str = dstr[6:8]
                wkdy_str = None
            # . YYYYWww
            elif self._isodate_type == 6 and length == 7:
                week_str = dstr[5:7]
                wkdy_str = None
            # . Invalid
            else:
                return False  # exit: not isoformat
            # . Validate week
            try:
                week: cython.uint = int(week_str)
            except Exception:
                return False  # exit: invalid week
            if not 1 <= week <= 53:
                return False  # exit: invalid week
            # . Validate weekday
            if wkdy_str is not None:
                try:
                    weekday: cython.uint = int(wkdy_str)
                except Exception:
                    return False  # exit: invalid weekday
                if not 1 <= weekday <= 7:
                    return False  # exit: invalid weekday
            else:
                weekday: cython.uint = 1
            # . Convert to Y/M/D
            ymd = cydt.isocalendar_to_ymd(year, week, weekday)
            self._result.append_ymd(ymd.month, 2)
            self._result.append_ymd(ymd.day, 3)
            return True  # exit: success

        # Parse days of year
        if self._isodate_type <= 8:
            days_str: str
            # . YYYY-DDD
            if self._isodate_type == 7 and length == 8:
                days_str = dstr[5:8]
            # . YYYYDDD
            elif self._isodate_type == 8 and length == 7:
                days_str = dstr[4:7]
            # . Invalid
            else:
                return False  # exit: not isoformat
            # . Validate days
            try:
                days: cython.uint = int(days_str)
            except Exception:
                return False  # exit: invalid days
            if not 1 <= days <= 366:
                return False  # exit: invalid days
            # . Convert to Y/M/D
            ymd = cydt.days_of_year_to_ymd(year, days)
            self._result.append_ymd(ymd.month, 2)
            self._result.append_ymd(ymd.day, 3)
            return True

        # . Invalid isoformat
        return False

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_isoformat_time(self, tstr: str, length: cython.uint) -> cython.bint:
        """(Internal) Parse the time components of the isoformat string `<bool>`."""
        # Validate 'tstr'
        if length < 2:
            return False  # exit: isoformat time to short [HH].

        # Search for isoformat timezone
        if self._ignoretz:
            tz_pos: cython.uint = 0
        else:
            tz_pos: cython.uint = self._find_isoformat_tz(tstr, length)

        # Parse HMS.f (without iso timezone)
        if tz_pos == 0:
            return self._parse_isoformat_hms(tstr, length)  # exit: success/fail

        # Parse HMS.f (with iso timezone)
        hms_len: cython.uint = tz_pos - 1
        if not self._parse_isoformat_hms(tstr[0:hms_len], hms_len):
            return False  # exit: invalid time component

        # UTC timzeone
        tz_sep: cython.Py_UCS4 = str_loc(tstr, tz_pos - 1)
        if tz_pos == length and tz_sep == CHAR_LOWER_Z:
            self._result.tzoffset = 0
            return True  # exit: success

        # Parse timezone
        pos_ed: cython.uint = tz_pos + 2
        # . parse tz hour
        if pos_ed > length:
            return False  # exit: incomplete tzoffset
        try:
            hour: cython.int = int(tstr[tz_pos:pos_ed])
        except Exception:
            return False  # exit: invalid tzoffset
        # . parse tz minute
        if pos_ed < length:
            nchar: cython.Py_UCS4 = str_loc(tstr, pos_ed)
            if is_isotime_sep(nchar):
                tz_pos += 3
                pos_ed += 3
            else:
                tz_pos += 2
                pos_ed += 2
            if pos_ed > length:
                return False  # exit: incomplete tzoffset
            try:
                minute: cython.int = int(tstr[tz_pos:pos_ed])
            except Exception:
                return False  # exit: invalid tzoffset
        else:
            minute: cython.int = 0
        # . parse tz sign
        tzsign: cython.int = 1 if tz_sep == CHAR_PLUS else -1
        # . calculate tzoffset
        offset: cython.int = tzsign * (hour * 3_600 + minute * 60)
        if self._result.tzoffset != -100_000:
            self._result.tzoffset = self._result.tzoffset - offset
        else:
            self._result.tzoffset = offset
        return True  # exit: success

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_isoformat_hms(self, tstr: str, length: cython.uint) -> cython.bint:
        """(Internal) Parse the HMS.f components of the isoformat string `<bool>`."""
        # Parse HMS
        comps: cython.int[4] = [0, 0, 0, 0]
        pos: cython.uint = 0
        idx: cython.uint
        nchar: cython.Py_UCS4
        has_sep: cython.bint
        for idx in range(0, 3):
            # . validate component
            if (length - pos) < 2:
                return False  # exit: incomplete HMS
            # . parse component
            try:
                val: cython.int = int(tstr[pos : pos + 2])
            except Exception:
                return False  # exit: invalid HMS
            comps[idx] = val
            # . validate time seperator
            pos += 2
            nchar = str_loc(tstr, pos) if pos < length else 0
            if nchar == 0 or idx >= 2:
                break
            if idx == 0:
                has_sep = is_isotime_sep(nchar)
            if has_sep and not is_isotime_sep(nchar):
                return False  # exit: invalid HMS seperator
            pos += has_sep

        # Parse microsecond / [possible] timezone name
        if pos < length:
            # Validate microsecond component
            nchar = str_loc(tstr, pos)
            if nchar == CHAR_PERIOD or nchar == CHAR_COMMA:  # separator [.,]
                pos += 1
                # . search for microsecond digits
                pos_ed: cython.uint = pos
                while pos_ed < length:
                    if not is_ascii_digit(str_loc(tstr, pos_ed)):
                        break
                    pos_ed += 1
                if pos == pos_ed:
                    return False  # exit: imcomplete microsecond
                # . parse microsecond component
                try:
                    val = parse_us_fraction(tstr[pos:pos_ed], pos_ed - pos)
                except Exception:
                    return False  # exit: invalid microsecond
                comps[3] = val
                pos = pos_ed

            # Parse [possible] timezone name
            if not self._ignoretz and pos < length:
                # . search for first alpha
                while pos < length:
                    if is_ascii_alpha_lower(str_loc(tstr, pos)):
                        break
                    pos += 1
                # . parse timezone name
                if length - pos >= 3:
                    offset = self._token_to_tzoffset(tstr[pos:length])
                    if offset == -100_000:
                        return False  # exit: invalid timezone name
                    self._result.tzoffset = offset

        # Append HMS
        self._result.hour = comps[0]
        self._result.minute = comps[1]
        self._result.second = comps[2]
        self._result.microsecond = comps[3]
        return True  # exit: success

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _find_isoformat_tz(self, tstr: str, length: cython.uint) -> cython.uint:
        """(Internal) Find the location of possible isoformat UTC timezone
        in 'tstr'. Equivalent to re.search('[+-Z]', tstr). Returns 0 if
        not found `<int>`."""
        loc: cython.Py_ssize_t = str_findc(tstr, CHAR_PLUS, 0, length, 1)
        if loc >= 0:
            return loc + 1
        loc = str_findc(tstr, CHAR_DASH, 0, length, 1)
        if loc >= 0:
            return loc + 1
        loc = str_findc(tstr, CHAR_LOWER_Z, 0, length, 1)
        if loc >= 0:
            return loc + 1
        return 0

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _get_char(self, index: cython.uint) -> cython.Py_UCS4:
        """(Internal) Get character of the 'dtstr' by index `<Py_UCS4>`.
        Returns 0 if the index is out of range."""
        if index < self._dtstr_len:
            return str_loc(self._dtstr, index)
        else:
            return 0

    # Numeric token ------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_numeric_token(self, token: str) -> cython.bint:
        """(Internal) Parse the 'numeric' token. Returns
        True if the token represents numeric value and
        processed successfully, else False `<bool>`."""
        # Convert token to numeric value
        try:
            tok_val: cython.double = float(token)
        except ValueError:
            return False  # exit: not a numeric token
        if not math.isfinite(tok_val):
            return False  # exit: invalid infinite value ('inf')

        # Token info
        tok_len: cython.int = str_len(token)
        token_r1: str = self._get_token_r1()

        # (19990101T)23[59]
        if (
            self._result.ymd_values() == 3
            and self._result.hour == -1
            and (tok_len == 2 or tok_len == 4)
            and (
                self._index + 1 >= self._tokens_count
                or (
                    token_r1 is not None
                    and token_r1 != ":"
                    and self._token_to_hms(token_r1) == -1
                )
            )
        ):
            self._result.hour = int(token[0:2])
            if tok_len == 4:
                self._result.minute = int(token[2:4])
            return True  # exit

        # YYMMDD or HHMMSS[.ss]
        if tok_len == 6 or (tok_len > 6 and str_loc(token, 6) == CHAR_PERIOD):
            if self._result.ymd_values() == 0 and not "." in token:
                self._result.append_ymd(token[0:2], 0)
                self._result.append_ymd(token[2:4], 0)
                self._result.append_ymd(token[4:tok_len], 0)
            else:
                # 19990101T235959[.59]
                self._result.hour = int(token[0:2])
                self._result.minute = int(token[2:4])
                self._set_second_and_us(token[4:tok_len])
            return True  # exit

        # YYYYMMDD
        if tok_len == 8 or tok_len == 12 or tok_len == 14:
            self._result.append_ymd(token[0:4], 1)
            self._result.append_ymd(token[4:6], 0)
            self._result.append_ymd(token[6:8], 0)
            if tok_len > 8:
                self._result.hour = int(token[8:10])
                self._result.minute = int(token[10:12])
                if tok_len > 12:
                    self._result.second = int(token[12:14])
            return True  # exit

        # HH[ ]h or MM[ ]m or SS[.ss][ ]s
        if self._parse_hms_token(token, tok_val):
            return True  # exit

        # HH:MM[:SS[.ss]]
        token_r2: str = self._get_token_r2()
        if token_r2 is not None and token_r1 == ":":
            # . HH:MM
            self._result.hour = int(tok_val)
            minute = self._covnert_numeric_token(token_r2)
            self._set_minite_and_second(minute)
            # . SS:[.ss]
            token_r4: str = self._get_token_r4()
            if token_r4 is not None and self._get_token_r3() == ":":
                self._set_second_and_us(token_r4)
                self._index += 2  # skip SS.ss
            self._index += 2  # skip HH:MM
            return True  # exit

        # YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
        if token_r1 is not None and (
            token_r1 == "-" or token_r1 == "/" or token_r1 == "."
        ):
            # 1st Y/M/D value
            self._result.append_ymd(token, 0)

            # 2nd Y/M/D value
            if token_r2 is not None and not self._is_token_jump(token_r2):
                try:
                    # 01-01[-01]
                    month = int(token_r2)
                    self._result.append_ymd(month, 0)
                except ValueError:
                    # 01-Jan[-01]
                    month = self._token_to_month(token_r2)
                    if month != -1:
                        self._result.append_ymd(month, 2)
                    else:
                        self._result.append_ymd(token_r2, 0)

                # 3rd Y/M/D value
                token_r4: str = self._get_token_r4()
                if token_r4 is not None and self._get_token_r3() == token_r1:  # sep
                    month = self._token_to_month(token_r4)
                    if month != -1:
                        self._result.append_ymd(month, 2)
                    else:
                        self._result.append_ymd(token_r4, 0)
                    self._index += 2  # skip 3rd Y/M/D
                self._index += 1  # skip 2nd Y/M/D
            self._index += 1  # skip 1st Y/M/D
            return True  # exit

        # "hour AM" or year|month|day
        if self._index + 1 >= self._tokens_count or self._is_token_jump(token_r1):
            if token_r2 is not None and (ampm := self._token_to_ampm(token_r2)) != -1:
                # 12 AM
                self._result.hour = self._adjust_ampm_hour(int(tok_val), ampm)
                self._index += 1  # skip AMPM
            else:
                # Year, month or day
                self._result.append_ymd(token, 0)
            self._index += 1  # skip token r1
            return True  # exit

        # "hourAM"
        if 0 <= tok_val < 24 and (ampm := self._token_to_ampm(token_r1)) != -1:
            self._result.hour = self._adjust_ampm_hour(int(tok_val), ampm)
            self._index += 1  # skip token r1
            return True  # exit

        # Possible is day
        if self._result.could_be_day(int(tok_val)):
            self._result.append_ymd(token, 0)
            return True  # exit

        # Invalid
        if not self._fuzzy:
            raise errors.InvalidTokenError(
                "<{}>\nFailed to handle the numeric token: "
                "{}".format(self.__class__.__name__, repr(token))
            )

        return True

    @cython.cfunc
    @cython.inline(True)
    def _covnert_numeric_token(self, token: str) -> cython.double:
        """(Internal) Try to convert numeric token to `<float>`."""
        try:
            tok_val: cython.double = float(token)
            if not math.isfinite(tok_val):
                raise ValueError("Token does not represent a finite value.")
        except ValueError as err:
            raise errors.InvalidNumericToken(
                "<{}>\nFailed to convert numeric token {} to float: "
                "{}".format(self.__class__.__name__, repr(token), err)
            ) from err
        return tok_val

    # Month token --------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_month_token(self, token: str) -> cython.bint:
        """(Internal) Parse the 'month' token. Returns
        True if the token represents month value and
        processed successfully, else False `<bool>`."""
        # Validate if is month token
        month = self._token_to_month(token)
        if month == -1:
            return False  # exit: not a month token
        # Append month
        self._result.append_ymd(month, 2)

        # Try to get year & day
        token_r2: str = self._get_token_r2()
        if token_r2 is not None:
            token_r1: str = self._get_token_r1()
            if token_r1 == "-" or token_r1 == "/":
                # Jan-01[-99?] uncertain
                self._result.append_ymd(token_r2, 0)
                token_r4: str = self._get_token_r4()
                if token_r4 is not None and self._get_token_r3() == token_r1:  # sep
                    # Jan-01-99 confirmed
                    self._result.append_ymd(token_r4, 0)
                    self._index += 2  # skip token r3 & r4
                self._index += 2  # skip token r1 & r2
                return True  # exit

            # Jan of 01. In this case, 01 is clearly year
            token_r4: str = self._get_token_r4()
            if (
                token_r4 is not None
                and self._is_token_pertain(token_r2)
                and self._get_token_r3() == " "
            ):
                try:
                    year = int(token_r4)
                    self._result.append_ymd(year, 1)
                except ValueError:
                    pass  # wrong guess
                self._index += 4  # skip token r1 - r4
                return True  # exit

        # Finished
        return True  # exit

    # Weekday token ------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_weekday_token(self, token: str) -> cython.bint:
        """(Internal) Parse the 'weekday' token. Returns
        True if the token represents weekday value and
        processed successfully, else False `<bool>`."""
        # Validate if is weekday token
        weekday = self._token_to_weekday(token)
        if weekday == -1:
            return False  # exit: not a weekday token

        # Set parse result
        self._result.weekday = weekday
        return True  # exit

    # HMS token ----------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_hms_token(self, token: str, value: cython.double) -> cython.bint:
        """(Internal) Parse the 'hms' token. Returns
        True if the token represents hms value and
        processed successfully, else False `<bool>`."""
        # Pre-binding
        hms: cython.int

        # Looking forwards
        token_r1 = self._get_token_r1()
        if token_r1 is not None:
            # There is an "h", "m", or "s" label following this token.
            # We take assign the upcoming label to the current token.
            # e.g. the "12" in 12h"
            if (hms := self._token_to_hms(token_r1)) != -1:
                self._set_hms_result(token, value, hms)
                self._index += 1  # skip token r1
                return True

            # There is a space and then an "h", "m", or "s" label.
            # e.g. the "12" in "12 h"
            token_r2 = self._get_token_r2()
            if (
                token_r2 is not None
                and token_r1 == " "
                and (hms := self._token_to_hms(token_r2)) != -1
            ):
                self._set_hms_result(token, value, hms)
                self._index += 2  # skip token r1 & r2
                return True

        # Looking backwords
        if self._index > 0:
            token_l1: str = self._get_token(self._index - 1)
            # There is a "h", "m", or "s" preceding this token. Since neither
            # of the previous cases was hit, there is no label following this
            # token, so we use the previous label.
            # e.g. the "04" in "12h04"
            if (hms := self._token_to_hms(token_l1)) != -1:
                # looking backwards, hms increment one.
                self._set_hms_result(token, value, hms + 1)
                return True

            # If we are looking at the final token, we allow for a
            # backward-looking check to skip over a space.
            # TODO: Are we sure this is the right condition here?
            token_l2 = self._get_token(self._index - 2)
            if (
                token_l2 is not None
                and token_l1 == " "
                and (hms := self._token_to_hms(token_l2)) != -1
            ):
                # looking backwards, hms increment one.
                self._set_hms_result(token, value, hms + 1)
                return True

        # Not HMS token
        return False

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_hms_result(
        self,
        token: str,
        value: cython.double,
        hms: cython.int,
    ) -> cython.bint:
        """(Internal) Set the HH:MM:SS result."""
        if hms == 0:
            self._set_hour_and_minite(value)
        elif hms == 1:
            self._set_minite_and_second(value)
        elif hms == 2:
            self._set_second_and_us(token)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_hour_and_minite(self, value: cython.double) -> cython.bint:
        """(Internal) Set the hour & minite result."""
        self._result.hour = int(value)
        if rem := value % 1:
            self._result.minute = int(rem * 60)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_minite_and_second(self, value: cython.double) -> cython.bint:
        """(Internal) Set the minite & second result."""
        self._result.minute = int(value)
        if rem := value % 1:
            self._result.second = int(rem * 60)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_second_and_us(self, token: str) -> cython.bint:
        """(Internal) Set the second & microsecond result."""
        if "." in token:
            toks: list = token.split(".")
            sec = cython.cast(str, list_getitem(toks, 0))
            self._result.second = int(sec)
            us = cython.cast(str, list_getitem(toks, 1))
            self._result.microsecond = parse_us_fraction(us, 0)
        else:
            self._result.second = int(token)
        return True

    # AM/PM token --------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_ampm_token(self, token: str) -> cython.bint:
        """(Internal) Parse the 'am/pm' token. Returns
        True if the token represents am/pm value and
        processed successfully, else False `<bool>`."""
        # Validate if is am/pm token
        if self._result.ampm != -1:
            return False  # exit: AM/PM flag already set
        ampm = self._token_to_ampm(token)
        if ampm == -1:
            return False  # exit: not an ampm token

        # If AM/PM is found, but hour is not.
        hour: cython.int = self._result.hour
        if hour == -1:
            if self._fuzzy:
                return False  # exit
            raise errors.InvalidTokenError(
                "<{}>\nMissing hour value for the am/pm token: "
                "{}.".format(self.__class__.__name__, repr(token))
            )

        # If AM/PM is found, but hour is not a 12 hour clock
        if not 0 <= hour <= 12:
            if self._fuzzy:
                return False  # exit
            raise errors.InvalidTokenError(
                "<{}>\nInvalid hour value [{}] for the am/pm token: "
                "{}.".format(self.__class__.__name__, hour, repr(token))
            )

        # Adjust & set ampm
        self._result.hour = self._adjust_ampm_hour(hour, ampm)
        self._result.ampm = ampm
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _adjust_ampm_hour(self, hour: cython.int, ampm: cython.int) -> cython.uint:
        """(Internal) Adjust the hour according to the AM/PM flag."""
        if hour < 12:
            if ampm == 1:
                hour += 12
            return max(0, hour)
        elif hour == 12 and ampm == 0:
            return 0
        else:
            return hour

    # Tzname token -------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_tzname_token(self, token: str) -> cython.bint:
        """(Internal) Parse the 'tzname' token. Returns
        True if the token represents timezone name and
        processed successfully, else False `<bool>`."""
        # Validate if is timezome name
        if (
            self._ignoretz  # ignore timezone
            or self._result.hour == -1  # hour not set yet
            or self._result.tzname is not None  # tzname already set
            or self._result.tzoffset != -100_000  # tzoffset already set
            or (is_tz := self._could_be_tzname(token)) == 0  # not tzname
        ):
            return False  # exit: not tzname

        # Set tzname & tzoffset
        if is_tz == 1:
            # UTC timezone
            self._result.tzname = "UTC"
            self._result.tzoffset = 0
        else:
            # Token as timezone name
            self._result.tzname = token.upper()
            self._result.tzoffset = self._token_to_tzoffset(token)

        # Check for something like GMT+3, or BRST+3. Notice
        # that it doesn't mean "I am 3 hours after GMT", but
        # "my time +3 is GMT". If found, we reverse the
        # logic so that tzoffset parsing code will get it
        # right.
        token_r1: str = self._get_token_r1()
        if token_r1 is not None:
            if token_r1 == "+":
                list_setitem(self._tokens, self._index + 1, "-")
            elif token_r1 == "-":
                list_setitem(self._tokens, self._index + 1, "+")
            else:
                return True  # exit
            # Reset tzoffset
            self._result.tzoffset = -100_000
        return True  # exit

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _could_be_tzname(self, token: str) -> cython.uint:
        """(Internal) Check if a token could be a timezone
        name `<int>`. Returns 0 if not tzname, 1 if is UTC
        and 2 if the token is the timezone name itself.
        """
        # Invalid token
        if token is None:
            return 0  # exit: not tzname

        # Could be an UTC timezone
        if self._is_token_utc(token):
            return 1  # exit: token is UTC

        # Timezone name must be ASCII [a-z] & 3-5 length
        if not 3 <= str_len(token) <= 5:
            return 0  # exit: not tzname
        char: cython.Py_UCS4
        for char in token:
            if not is_ascii_alpha_lower(char):
                return 0  # exit: not tzname
        return 2  # exit: could be tzname

    # Tzoffset token -----------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_tzoffset_token(self, token: str) -> cython.bint:
        """(Internal) Parse the 'tzoffset' token. Returns
        True if the token represents timezone offset and
        processed successfully, else False `<bool>`."""
        # Validate if is timezome tzoffset
        if (
            self._ignoretz  # ignore timezone
            or self._result.hour == -1  # hour not set yet
            or self._result.tzoffset != -100_000  # tzoffset already set
        ):
            return False

        # Validate current token
        if token == "+":
            sign: cython.int = 1
        elif token == "-":
            sign: cython.int = -1
        else:
            return False  # exit: not tzoffset

        # Validate next token
        token_r1 = self._get_token_r1()
        if token_r1 is None:
            return False  # exit: not tzoffset
        try:
            offset: cython.int = int(token_r1)
        except ValueError:
            return False  # exit: not tzoffset

        # Calculate & set tzoffset
        offset = self._calculate_tzoffset(token_r1, sign, offset)
        if not -86_340 <= offset <= 86_340:
            raise errors.InvalidTokenError(
                "<{}>\nFailed to parse timezone offset from tokens: "
                "{}".format(self.__class__.__name__, [repr(token), repr(token_r1)])
            )
        self._result.tzoffset = offset

        # Look for a timezone name between parenthesis
        if self._result.tzname is None:
            # No more tokens
            token_r2 = self._get_token_r2()
            if token_r2 is None:
                pass

            # -0300(BRST) # w/o space
            elif token_r2 == "(":
                token_r3 = self._get_token_r3()  # BRST
                # fmt: off
                if (
                    (is_tz := self._could_be_tzname(token_r3)) > 0 
                    and self._get_token_r4() == ")"
                ):
                # fmt: on
                    self._result.tzname = "UTC" if is_tz == 1 else token_r3.upper()
                    self._index += 3  # skip token r2 - r4

            # -0300 (BRST) # with space
            elif self._is_token_jump(token_r2) and self._get_token_r3() == "(":
                token_r4 = self._get_token_r4()
                # fmt: off
                if (
                    (is_tz := self._could_be_tzname(token_r4)) > 0 
                    and self._get_token(self._index + 5) == ")"
                ):
                # fmt: on
                    self._result.tzname = "UTC" if is_tz == 1 else token_r4.upper()
                    self._index += 4  # skip token r2 - r5

        # Finished
        self._index += 1  # skip token r1
        return True  # exit

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _calculate_tzoffset(
        self,
        token_r1: str,
        sign: cython.int,
        offset: cython.int,
    ) -> cython.int:
        """(Internal) Calculate the timezone offset
        in seconds `<int>`."""
        # Prase tzoffset
        tok_len: cython.int = str_len(token_r1)
        # -0300
        if tok_len == 4:
            h_offset: cython.int = int(token_r1[0:2])
            m_offset: cython.int = int(token_r1[2:4])
            return sign * (h_offset * 3_600 + m_offset * 60)

        # -03:00
        token_r3: str = self._get_token_r3()
        if token_r3 is not None and self._get_token_r2() == ":":
            try:
                m_offset: cython.int = int(token_r3)
                self._index += 2  # skip token r2 & r3
                return sign * (offset * 3_600 + m_offset * 60)
            except ValueError:
                pass  # wrong guess

        # -[0]3
        if tok_len <= 2:
            return sign * (offset * 3_600)

        # Invalid
        return -100_000

    # Get token ----------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _get_token(self, index: cython.int) -> str:
        """(Internal) Get the token by index `<str>`."""
        if 0 <= index < self._tokens_count:
            return cython.cast(str, list_getitem(self._tokens, index))
        else:
            return None

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _get_token_r1(self) -> str:
        """(Internal) Get the next (+1) token `<str>`."""
        if self._token_r1 is None:
            return self._get_token(self._index + 1)
        else:
            return self._token_r1

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _get_token_r2(self) -> str:
        """(Internal) Get the next (+2) token `<str>`."""
        if self._token_r2 is None:
            return self._get_token(self._index + 2)
        else:
            return self._token_r2

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _get_token_r3(self) -> str:
        """(Internal) Get the next (+3) token `<str>`."""
        if self._token_r3 is None:
            return self._get_token(self._index + 3)
        else:
            return self._token_r3

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _get_token_r4(self) -> str:
        """(Internal) Get the next (+4) token `<str>`."""
        if self._token_r4 is None:
            return self._get_token(self._index + 4)
        else:
            return self._token_r4

    # Config -------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_token_pertain(self, token: object) -> cython.bint:
        """(Internal) Check if the given token should be
        recognized as a pertain `<bool>`."""
        return set_contains(self._pertain, token)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_token_jump(self, token: object) -> cython.bint:
        """(Internal) Check if the given token should be
        recognized as a jump word `<bool>`."""
        return set_contains(self._jump, token)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_token_utc(self, token: object) -> cython.bint:
        """(Internal) Check if the given token should be
        recognized as an UTC timezone `<bool>`."""
        return set_contains(self._utc, token)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _token_to_month(self, token: object) -> cython.int:
        """(Internal) Try to convert token to month `<int>`.
        Returns the month value (1-12) if token matched
        with 'month' settings in Configs, else -1.
        """
        val = dict_getitem(self._month, token)
        if val == cython.NULL:
            return -1
        else:
            return cython.cast(object, val)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _token_to_weekday(self, token: object) -> cython.int:
        """(Internal) Try to convert token to weekday `<int>`.
        Returns the weekday value (0-6) if token matched
        with 'weekday' settings in Configs, else -1.
        """
        val = dict_getitem(self._weekday, token)
        if val == cython.NULL:
            return -1
        else:
            return cython.cast(object, val)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _token_to_hms(self, token: object) -> cython.int:
        """(Internal) Try to convert token to hms `<int>`.
        Returns the hms value (0=hour, 1=minute, 2=second) if
        token matched with 'hms' settings in Configs, else -1.
        """
        val = dict_getitem(self._hms, token)
        if val == cython.NULL:
            return -1
        else:
            return cython.cast(object, val)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _token_to_ampm(self, token: object) -> cython.int:
        """(Internal) Try to convert token to ampm `<int>`.
        Returns the ampm value (0=am, 1=pm) if token
        matched with 'ampm' settings in Configs, else -1.
        """
        val = dict_getitem(self._ampm, token)
        if val == cython.NULL:
            return -1
        else:
            return cython.cast(object, val)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _token_to_tzoffset(self, token: object) -> cython.int:
        """(Internal) Try to convert token to tzoffset `<int>`.
        Returns the tzoffset in seconds if token matched
        with the 'tzinfo' settings in Configs, else -100_000.
        """
        val = dict_getitem(self._tzinfo, token)
        if val == cython.NULL:
            return -100_000
        else:
            return cython.cast(object, val)
