# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from __future__ import annotations

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.libc.math import isfinite  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cpython.time import localtime  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_READ_CHAR as read_char  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_GET_LENGTH as str_len  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_FindChar as str_findc  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_Replace as str_replace  # type: ignore
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
from cython.cimports.cytimes import cydatetime as cydt, typeref  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
import datetime, time
from dateutil.parser._parser import parserinfo
from cytimes import cydatetime as cydt, typeref, errors

__all__ = ["Config", "Parser"]

# Constants -----------------------------------------------------------------------------------
# . default config
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
    "pst": -8 * 3_600, # Pacific Standard Time
    "cet":  1 * 3_600, # Central European Time
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
    "p": 1,  "pm": 1, "afternoon": 1, "nachmittag": 1, "pomeriggio": 1, "tarde": 1,  "tarde": 1, "middag": 1,  "eftermiddag": 1, "popołudnie": 1, "öğleden": 1, "下午": 1
}
# fmt: on
# . datetime
US_FRACTION_CORRECTION: cython.uint[5] = [100000, 10000, 1000, 100, 10]
# . timezone
TIMEZONE_NAME_LOCAL: set[str] = set(time.tzname)


# Timelex -------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.wraparound(True)
def timelex(dtstr: str, length: cython.Py_ssize_t = 0) -> list[str]:
    """This function breaks the time string into lexical units (tokens),
    which can be parsed by the Parser. Lexical units are demarcated by
    changes in the character set, so any continuous string of letters or
    number is considered one unit `<'list[str]'>`."""
    # Setup
    if length <= 0:
        length = str_len(dtstr)
    index: cython.Py_ssize_t = -1
    temp_ch: cython.Py_UCS4 = 0  # '' null
    tokens: list[str] = []

    # Main string loop -------------------------------------------------------
    while index < length:
        token: str = None
        token_state: cython.int = 0
        alpha_token: cython.bint = False
        curr_ch: cython.Py_UCS4

        # Nested token loop - - - - - - - - - - - - - - - - - - - - - - - - -
        while index < length:
            # Retrieve the charactor for the current token.
            if temp_ch == 0:
                if (index := index + 1) >= length:
                    # Reached end of the string:
                    # 1. exit the nested token loop.
                    # 2. main loop will also be stopped.
                    break  # exit token loop
                while (curr_ch := read_char(dtstr, index)) == 0:
                    if (index := index + 1) >= length:
                        break
                if curr_ch == 0:
                    # No more valid charactor:
                    # 1. exit the nested token loop.
                    # 2. main loop will also be stopped.
                    break  # exit token loop
            # Retrieve the temp charactor as the next token.
            else:
                curr_ch, temp_ch = temp_ch, 0

            # Token state 0: the 1st charactor of the token.
            if token_state == 0:
                # . assign the 1st charactor to the currnet token.
                token = str_fr_ucs4(curr_ch)
                if curr_ch.isalpha():
                    token_state = 1  # alpha token
                elif curr_ch.isdigit():
                    token_state = 2  # digit token
                else:
                    break  # exit token loop: single charactor token

            # Token state 1: alpha token
            elif token_state == 1:
                alpha_token = True  # mark the token contains alpha.
                if curr_ch.isalpha():
                    token += str_fr_ucs4(curr_ch)
                elif curr_ch == ".":  # "."
                    token += "."
                    token_state = 3  # alpha token w/t "."
                else:
                    # exit token loop, and cache the
                    # charactor for the next token.
                    temp_ch = curr_ch
                    break

            # Token state 2: digit token
            elif token_state == 2:
                if curr_ch.isdigit():
                    token += str_fr_ucs4(curr_ch)
                elif curr_ch == ".":  # "."
                    token += "."
                    token_state = 4  # digit token w/t "."
                elif curr_ch == "," and str_len(token) >= 2:  # ","
                    token += ","
                    token_state = 4  # digit token w/t ","
                else:
                    # exit token loop, and cache the
                    # charactor for the next token.
                    temp_ch = curr_ch
                    break

            # Token state 3: alpha token w/t "."
            elif token_state == 3:
                alpha_token = True  # mark the token contains alpha.
                if curr_ch == "." or curr_ch.isalpha():
                    token += str_fr_ucs4(curr_ch)
                elif curr_ch.isdigit() and token[-1] == ".":
                    token += str_fr_ucs4(curr_ch)
                    token_state = 4  # digit token w/t "."
                else:
                    # exit token loop, and cache the
                    # charactor for the next token.
                    temp_ch = curr_ch
                    break

            # Token state 4: digit token w/t "."
            elif token_state == 4:
                if curr_ch == "." or curr_ch.isdigit():
                    token += str_fr_ucs4(curr_ch)
                elif curr_ch.isalpha() and token[-1] == ".":
                    token += str_fr_ucs4(curr_ch)
                    token_state = 3  # alpha token w/t "."
                else:
                    # exit token loop, and cache the
                    # charactor for the next token.
                    temp_ch = curr_ch
                    break
            # Nested token loop - - - - - - - - - - - - - - - - - - - - - - -

        # Further handle token contains "." / ","
        if token_state == 3 or token_state == 4:
            if alpha_token or token[-1] in ".," or str_count(token, ".") > 1:  # type: ignore
                tok: str = None
                for i in range(str_len(token)):
                    ch: cython.Py_UCS4 = read_char(token, i)
                    if ch == ".":  # "."
                        if tok is not None:
                            tokens.append(tok)
                            tok = None
                        tokens.append(".")
                    elif ch == ",":  # ","
                        if tok is not None:
                            tokens.append(tok)
                            tok = None
                        tokens.append(",")
                    elif tok is not None:
                        tok += str_fr_ucs4(ch)
                    else:
                        tok = str_fr_ucs4(ch)
                if tok is not None:
                    tokens.append(tok)
            else:
                if token_state == 4 and not str_contains(token, "."):
                    token = str_replace(token, ",", ".", -1)
                tokens.append(token)

        # Token that is None means the end of the charactor set.
        elif token is None:
            break
        else:
            tokens.append(token)
        # Main string loop ---------------------------------------------------

    # Return the time lexical tokens
    return tokens


# Result --------------------------------------------------------------------------------------
@cython.cclass
class Result:
    """Represents the result for the Parser."""

    # Y/M/D
    _ymd: cython.int[3]
    _idx: cython.int
    _yidx: cython.int
    _midx: cython.int
    _didx: cython.int
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
    century_specified: cython.bint

    def __cinit__(self) -> None:
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
        self.tzoffset = -100_000
        self.century_specified = False

    def __init__(self) -> None:
        """The result for the Parser."""
        # Result
        self.tzname = None

    # Y/M/D -----------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def append(self, value: int | str, label: cython.uint) -> cython.bint:
        """(cfunc) Append a Y/M/D value. Returns `False` if all
        slots (max 3) are fully populated `<'bool'>`.

        :param value `<'str/int'>`: A Y/M/D value.
        :param label `<'int'>`: The label for the value:
            - label=0: unknown
            - label=1: year
            - label=2: month
            - label=3: day
        """
        # Validate the value.
        if self._idx >= 2:
            return False  # exit: Y/M/D slots are full
        try:
            num: cython.int = int(value)
        except Exception as err:
            raise ValueError("Invalid Y/M/D value: %s" % repr(value)) from err
        if num < 0:
            raise ValueError("Invalid Y/M/D value: %s" % repr(value))

        # Pre-determine the year label.
        if num >= 100 or str_len(value) > 2:
            self.century_specified = True
            label = 1  # year label

        # Set & label Y/M/D value.
        self._idx += 1
        self._ymd[self._idx] = num  # populate slot
        if label == 0:
            pass
        elif label == 1 and self._yidx == -1:
            self._yidx = self._idx  # assign year index
        elif label == 2 and self._midx == -1:
            self._midx = self._idx  # assign month index
        elif label == 3 and self._didx == -1:
            self._didx = self._idx  # assign day index
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def populated(self) -> cython.uint:
        """(cfunc) Get the number of Y/M/D slots that have already
        been populated with values `<'int'>`."""
        return self._idx + 1

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def labeled(self) -> cython.uint:
        """(cfunc) Get the number of Y/M/D values that
        have already been labeled (solved) `<'int'>`."""
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
    def could_be_day(self, value: cython.int) -> cython.bint:
        """(cfunc) Determine if an integer 'value' could be the
        day of the date `<'bool'>`."""
        # Day value already populated.
        if self._didx != -1:
            return False
        # Month value not populated & value is in range.
        if self._midx == -1:
            return 1 <= value <= 31
        # Year value not populated & value is in range.
        month = self._ymd[self._midx]
        if self._yidx == -1:
            return 1 <= value <= cydt.days_in_month(2000, month)
        # Both Y&M values are populated & value in range.
        year = self._ymd[self._yidx]
        return 1 <= value <= cydt.days_in_month(year, month)

    # Result ----------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def prepare(self, day1st: cython.bint, year1st: cython.bint) -> cython.bint:
        """Prepare the parsed datetime result. Must call this
        method before accessing the result values `<'bool'>`.
        """
        populated: cython.uint = self.populated()
        labeled: cython.uint = self.labeled()

        # All Y/M/D have been solved already.
        if populated == labeled > 0:
            self.year = self._ymd[self._yidx] if self._yidx != -1 else -1
            self.month = self._ymd[self._midx] if self._midx != -1 else -1
            self.day = self._ymd[self._didx] if self._didx != -1 else -1

        # Only has one Y/M/D value.
        elif populated == 1:
            v0: cython.int = self._ymd[0]
            if self._midx != -1:  # month labeled
                self.month = v0
            elif v0 > 31:  # has to be year
                self.year = v0
            else:  # probably day
                self.day = v0

        # Have two Y/M/D values.
        elif populated == 2:
            v0: cython.int = self._ymd[0]
            v1: cython.int = self._ymd[1]
            # . month labeled
            if self._midx != -1:
                if self._midx == 0:
                    self.month = v0
                    if v1 > 31:  # has to be year: Jan-99
                        self.year = v1
                    else:  # probably day: Jan-01
                        self.day = v1
                else:
                    self.month = v0
                    if v1 > 31:  # has to be year: 99-Jan
                        self.year = v1
                    else:  # probably day: 01-Jan
                        self.day = v1
            # . month not labeled
            elif v0 > 31:  # 99-Jan
                self.year, self.month = v0, v1
            elif v1 > 31:  # Jan-99
                self.month, self.year = v0, v1
            elif day1st and 1 <= v1 <= 12:  # 01-Jan
                self.day, self.month = v0, v1
            else:  # Jan-01
                self.month, self.day = v0, v1

        # Have three Y/M/D values.
        elif populated == 3:
            # . lack of one label
            if labeled == 2:
                if self._midx != -1:  # month labeled
                    self.month = self._ymd[self._midx]
                    if self._yidx != -1:  # year labeled
                        self.year = self._ymd[self._yidx]
                        self.day = self._ymd[3 - self._yidx - self._midx]
                    else:  # day labeled
                        self.day = self._ymd[self._didx]
                        self.year = self._ymd[3 - self._midx - self._didx]
                elif self._yidx != -1:  # year labeled
                    self.year = self._ymd[self._yidx]
                    if self._midx != -1:  # month labeled
                        self.month = self._ymd[self._midx]
                        self.day = self._ymd[3 - self._yidx - self._midx]
                    else:  # day labeled
                        self.day = self._ymd[self._didx]
                        self.month = self._ymd[3 - self._yidx - self._didx]
                else:  # day labeled
                    self.day = self._ymd[self._didx]
                    if self._yidx != -1:  # year labeled
                        self.year = self._ymd[self._yidx]
                        self.month = self._ymd[3 - self._yidx - self._didx]
                    else:  # month labeled
                        self.month = self._ymd[self._midx]
                        self.year = self._ymd[3 - self._midx - self._didx]
            # . lack more than one labels (guess)
            elif self._midx == 0:
                v0: cython.int = self._ymd[0]
                v1: cython.int = self._ymd[1]
                v2: cython.int = self._ymd[2]
                if v1 > 31:  # Apr-2003-25
                    self.month, self.year, self.day = v0, v1, v2
                else:  # Apr-25-2003
                    self.month, self.day, self.year = v0, v1, v2
            elif self._midx == 1:
                v0: cython.int = self._ymd[0]
                v1: cython.int = self._ymd[1]
                v2: cython.int = self._ymd[2]
                if v0 > 31 or (year1st and 0 < v2 <= 31):  # 99-Jan-01
                    self.year, self.month, self.day = v0, v1, v2
                else:  # 01-Jan-99
                    self.day, self.month, self.year = v0, v1, v2
            elif self._midx == 2:
                v0: cython.int = self._ymd[0]
                v1: cython.int = self._ymd[1]
                v2: cython.int = self._ymd[2]
                if v1 > 31:  # 01-99-Jan
                    self.day, self.year, self.month = v0, v1, v2
                else:  # 99-01-Jan
                    self.year, self.day, self.month = v0, v1, v2
            else:
                v0: cython.int = self._ymd[0]
                v1: cython.int = self._ymd[1]
                v2: cython.int = self._ymd[2]
                if (
                    v0 > 31
                    or self._yidx == 0
                    or (year1st and 0 < v1 <= 12 and 0 < v2 <= 31)
                ):
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
        if 0 <= self.year < 100 and not self.century_specified:
            curr_year: cython.int = localtime().tm_year
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

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def valid(self) -> cython.bint:
        """Check if the result is valid (contains any parsed values) `<'bool'>`."""
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
        return "<%s(%s)>" % (self.__class__.__name__, ", ".join(reprs))

    def __bool__(self) -> bool:
        return self.valid()


# Config --------------------------------------------------------------------------------------
@cython.cclass
class Config:
    """Represents the configurations for the <'Parser'>."""

    # Settings
    _day1st: cython.bint
    _year1st: cython.bint
    _pertain: set[str]
    _jump: set[str]
    _utc: set[str]
    _tz: dict[str, int]
    _month: dict[str, int]
    _weekday: dict[str, int]
    _hms: dict[str, int]
    _ampm: dict[str, int]
    # Words
    _words: set[str]

    @classmethod
    def from_parserinfo(cls, info: parserinfo) -> Config:
        """Import settings from dateutil.parser.parserinfo `<'Config'>`.

        ### Example
        >>> from cytimes import Config
        >>> from dateutil.parser import parserinfo
        >>> info = parserinfo()
            cfg = Config.from_parserinfo(info)
        """
        # Validate perserinfo
        if not isinstance(info, typeref.PARSERINFO):
            raise errors.InvalidParserInfo(
                "<'%s'>\nConfig can only import from <'dateutil.parser.parserinfo'>, "
                "instead got: %s %s." % (cls.__name__, type(info), repr(info))
            )

        # Import settings
        cfg = Config(day1st=info.dayfirst, year1st=info.yearfirst)
        cfg._pertain = set(info.PERTAIN)
        cfg._jump = set(info.JUMP)
        cfg._utc = set(info.UTCZONE)
        cfg._tz = info.TZOFFSET
        cfg._month = {w: i + 1 for i, wds in enumerate(info.MONTHS) for w in wds}
        cfg._weekday = {w: i for i, wds in enumerate(info.WEEKDAYS) for w in wds}
        cfg._hms = {w: i for i, wds in enumerate(info.HMS) for w in wds}
        cfg._ampm = {w: i for i, wds in enumerate(info.AMPM) for w in wds}
        # Reconstruct words
        cfg._construct_words()
        # Return Config
        return cfg

    def __init__(self, day1st: bool = False, year1st: bool = False) -> None:
        """The configurations for the <'Parser'>.

        ### Ambiguous Y/M/D
        :param day1st `<'bool'>`: Whether to interpret first ambiguous date values as day. Defaults to `False`.
        :param year1st `<'bool'>`: Whether to interpret first the ambiguous date value as year. Defaults to `False`.

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

        In the case when only one value is ambiguous (e.g. `01/20/2010` or `01/20/99`),
        arguments 'day1st' or 'year1st' takes no effect.

        ### Configurations
        Besides the 'day1st' & 'year1st' arguments, Config also provides 'add_xxx()',
        'rem_xxx()' and 'set_xxx()' methods to modify the following settings:
        - pertain: Words that should be recognized as pertain, e.g: `'of'`.
        - jump: Words that should be skipped, e.g: `'and'`, `'at'`, `'on'`.
        - utc: Words that should be recognized as UTC timezone, e.g: `'utc'`, `'gmt'`.
        - tz: Words that should be recognized as timezone and constrcted
          with the specified timezone offset in seconds, e.g: `'est'`, `'pst'`.
        - month: Words that should be recognized as month, e.g: `'january'`, `'february'`.
        - weekday: Words that should be recognized as weekday, e.g: `'monday'`, `'tuesday'`.
        - hms: Words that should be recognized as H/M/S, e.g: `'hour'`, `'minute'`.
        - ampm: Words that should be recognized as AM/PM, e.g: `'am'`, `'pm'`.

        ### Import Settings from `dateutil.parser.parserinfo`
        For user who wants to use an existing '<'dateutil.parser.parserinfo'>'
        settings, Config provides the 'from_parserinfo' method to bridge
        the compatibility with the 'dateutil' libaray.

        >>> from cytimes import Config
        >>> from dateutil.parser import parserinfo
        >>> info = parserinfo()
            cfg = Config.from_parserinfo(info)
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
        self._tz = CONFIG_TZ
        # Keywords
        self._construct_words()

    # Ambiguous -------------------------------------------------------
    @property
    def day1st(self) -> bool:
        """Whether to interpret first ambiguous Y/M/D values as day `<'bool'>`."""
        return self._day1st

    @property
    def year1st(self) -> bool:
        """Whether to interpret first ambiguous Y/M/D values as year `<'bool'>`."""
        return self._year1st

    # Pertain ---------------------------------------------------------
    @property
    def pertain(self) -> set[str]:
        """The words that should be recognized as pertain `<'set[str]'>`.

        ### Example
        >>> cfg.pertain
        >>> {"of"}
        """
        return self._pertain

    def add_pertain(self, *words: str) -> None:
        """Add 'words' that should be recognized as pertain.

        ### Example
        >>> cfg.add_pertain("of", ...)
        """
        for word in words:
            word = self._validate_word("pertain", word)
            set_add(self._pertain, word)

    def rem_pertain(self, *words: str) -> None:
        """Remove 'words' that should `NOT` be recognized as pertain.

        ### Example
        >>> cfg.rem_pertain("of", ...)
        """
        for word in words:
            set_discard(self._pertain, word)
            set_discard(self._words, word)

    def set_pertain(self, *words: str) -> None:
        """Set the 'words' that should be recognized as pertain.
        - Replaces the current pertain words with the specified 'words'.
        - If 'words' are not provided, resets to the default pertain words.

        ### Example
        >>> cfg.set_pertain("of", ...)
        """
        if words:
            self._pertain = set(words)
        else:
            self._pertain = CONFIG_PERTAIN
        self._construct_words()

    # Jump ------------------------------------------------------------
    @property
    def jump(self) -> set[str]:
        """The words that should be skipped `<'set[str]'>`.

        ### Example
        >>> cfg.jump
        >>> {
                " ", ".", ",", ";", "-", "/", "'",
                "at", "on", "and", "ad", "m", "t", "of",
                "st", "nd", "rd", "th", "年" ,"月", "日"
            }
        """
        return self._jump

    def add_jump(self, *words: str) -> None:
        """Add 'words' that should be skipped.

        ### Example
        >>> cfg.add_jump("at", "on", ...)
        """
        for word in words:
            word = self._validate_word("jump", word)
            set_add(self._jump, word)

    def rem_jump(self, *words: str) -> None:
        """Remove 'words' that should `NOT` be skipped.

        ### Example
        >>> cfg.rem_jump("at", "on", ...)
        """
        for word in words:
            set_discard(self._jump, word)
            set_discard(self._words, word)

    def set_jump(self, *words: str) -> None:
        """Set the 'words' that should be skipped.
        - Replaces the current skip words with the specified 'words'.
        - If 'words' are not provided, resets to the default skip words.

        ### Example
        >>> cfg.set_jump("at", "on", ...)
        """
        if words:
            self._jump = set(words)
        else:
            self._jump = CONFIG_JUMP
        self._construct_words()

    # UTC -------------------------------------------------------------
    @property
    def utc(self) -> set[str]:
        """The words that should be recognized as UTC timezone `<'set[str]'>`.

        ### Example
        >>> cfg.utc
        >>> {"utc", "gmt", "z"}
        """
        return self._utc

    def add_utc(self, *words: str) -> None:
        """Add 'words' that should be recognized as UTC timezone.

        ### Example
        >>> cfg.add_utc("utc", "gmt", "z", ...)
        """
        for word in words:
            word = self._validate_word("utc", word)
            set_add(self._utc, word)

    def rem_utc(self, *words: str) -> None:
        """Remove 'words' that should `NOT` be recognized as UTC timezone.

        ### Example
        >>> cfg.rem_utc("utc", "gmt", "z", ...)
        """
        for word in words:
            set_discard(self._utc, word)
            set_discard(self._words, word)

    def set_utc(self, *words: str) -> None:
        """Set the 'words' that should be recognized as UTC timezone.
        - Replaces the current UTC timezone words with the specified 'words'.
        - If 'words' are not provided, resets to the default UTC timezone words.

        ### Example
        >>> cfg.set_utc("utc", "gmt", "z", ...)
        """
        if words:
            self._utc = set(words)
        else:
            self._utc = CONFIG_UTC
        self._construct_words()

    # Timezone --------------------------------------------------------
    @property
    def tz(self) -> dict[str, int]:
        """The words that should be recognized as a timezone
        and constrcted with the specified timezone offset
        in seconds `<'dict[str, int]'>`.

        ### Example
        >>> cfg.tz
        >>> {'pst': -28800, 'cet': 3600, ...}
        """
        return self._tz

    def add_tz(self, word: str, hour: int = 0, minute: int = 0) -> None:
        """Add 'word' that should be recognized as a timezone
        and the corresponding timezone offset (hours & minutes).

        ### Example
        >>> cfg.add_tz("est", hour=-5)
        """
        tzoffset = hour * 3_600 + minute * 60
        dict_setitem(
            self._tz,
            self._validate_word("tz", word),
            self._validate_value("tz", tzoffset, -86_340, 86_340),
        )

    def rem_tz(self, *words: str) -> None:
        """Remove 'words' that should `NOT` be recognized as timezone.

        ### Example
        >>> cfg.rem_tz("est", "edt", ...
        """
        for word in words:
            try:
                dict_delitem(self._tz, word)
            except KeyError:
                pass
            set_discard(self._words, word)

    def set_tz(self, **word_n_tzoffset: int) -> None:
        """Set the 'words' that should be recognized as timezone.
        - Replaces the current timezone words with the specified 'words'.
        - If 'words' are not provided, resets to the default timezone words.

        ### Example
        >>> cfg.set_tz(est=-18000, edt=-14400, ...)
        """
        if word_n_tzoffset:
            self._tz = word_n_tzoffset
        else:
            self._tz = CONFIG_TZ
        self._construct_words()

    # Month -----------------------------------------------------------
    @property
    def month(self) -> dict[str, int]:
        """The words that should be recognized as month `<'dict[str, int]'>`.

        Where keys are the words and values are the month number.

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
        """Add 'words' that should be recognized as a specific month.

        ### Example
        >>> cfg.add_month(1, "january", "jan", ...)
        """
        month = self._validate_value("month", month, 1, 12)
        for word in words:
            word = self._validate_word("month", word)
            dict_setitem(self._month, word, month)

    def rem_month(self, *words: str) -> None:
        """Remove 'words' that should `NOT` be recognized as month.

        ### Example
        >>> cfg.rem_month("january", "jan", ...)
        """
        for word in words:
            try:
                dict_delitem(self._month, word)
            except KeyError:
                pass
            set_discard(self._words, word)

    def set_month(self, **word_n_month: int) -> None:
        """Set the 'words' that should be recognized as month.
        - Replaces the current month words with the specified 'words'.
        - If 'words' are not provided, resets to the default month words.

        ### Example
        >>> cfg.set_month(
                january=1, jan=1,
                february=2, feb=2,
                ...
            )
        """
        if word_n_month:
            self._month = word_n_month
        else:
            self._month = CONFIG_MONTH
        self._construct_words()

    # Weekday ---------------------------------------------------------
    @property
    def weekday(self) -> dict[str, int]:
        """The words that should be recognized as weekday `<'dict[str, int]'>`.

        Where keys are the words and values are the weekday
        number (0=Monday...6=Sunday),

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
        """Add 'words' that should be recognized as a specific weekday,
        where 0=Monday...6=Sunday.

        ### Example
        >>> cfg.add_weekday(0, "monday", "mon", ...)
        """
        weekday = self._validate_value("weekday", weekday, 0, 6)
        for word in words:
            word = self._validate_word("weekday", word)
            dict_setitem(self._weekday, word, weekday)

    def rem_weekday(self, *words: str) -> None:
        """Remove 'words' that should `NOT` be recognized as weekday.

        ### Example
        >>> cfg.rem_weekday("monday", "mon", ...)
        """
        for word in words:
            try:
                dict_delitem(self._weekday, word)
            except KeyError:
                pass
            set_discard(self._words, word)

    def set_weekday(self, **word_n_weekday: int) -> None:
        """Set the 'words' that should be recognized as weekday.
        - Replaces the current weekday words with the specified 'words'.
        - If 'words' are not provided, resets to the default weekday words.

        ### Example
        >>> cfg.set_weekday(
                monday=0, mon=0,
                tuesday=1, tue=1,
                ...
            )
        """
        if word_n_weekday:
            self._weekday = word_n_weekday
        else:
            self._weekday = CONFIG_WEEKDAY
        self._construct_words()

    # HMS ------------------------------------------------------------
    @property
    def hms(self) -> dict[str, int]:
        """The words that should be recognized as H/M/S `<'dict[str, int]'>`.

        Where keys are the words and values are the time
        unit (0=hour, 1=minute, 2=second).

        ### Example
        >>> cfg.hms
        >>> {"hour": 0, "minute": 1, "second": 2, ...}
        """
        return self._hms

    def add_hms(self, hms: int, *words: str) -> None:
        """Add 'words' that should be recognized as H/M/S,
        where 0=hour, 1=minute, 2=second.

        ### Example
        >>> cfg.add_hms(0, "hour", "hours", "h", ...)
        """
        hms = self._validate_value("hms", hms, 0, 2)
        for word in words:
            word = self._validate_word("hms", word)
            dict_setitem(self._hms, word, hms)

    def rem_hms(self, *words: str) -> None:
        """Remove 'words' that should `NOT` be recognized as H/M/S.

        ### Example
        >>> cfg.rem_hms("hour", "hours", "h", ...)
        """
        for word in words:
            try:
                dict_delitem(self._hms, word)
            except KeyError:
                pass
            set_discard(self._words, word)

    def set_hms(self, **word_n_hms: int) -> None:
        """Set the 'words' that should be recognized as H/M/S.
        - Replaces the current H/M/S words with the specified 'words'.
        - If 'words' are not provided, resets to the default H/M/S words.

        ### Example
        >>> cfg.set_hms(
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
        self._construct_words()

    # AM/PM ----------------------------------------------------------
    @property
    def ampm(self) -> dict[str, int]:
        """The words that should be recognized as AM/PM `<'dict[str, int]'>`.
        Where keys are the words and values are the AM/PM flag (0=AM, 1=PM).

        ### Example
        >>> cfg.ampm
        >>> {"am": 0, "pm": 1, ...}
        """
        return self._ampm

    def add_ampm(self, ampm: int, *words: str) -> None:
        """Add 'words' that should be recognized as AM/PM,
        where 0=AM and 1=PM.

        ### Example
        >>> cfg.add_ampm(0, "am", "a.m.", ...)
        """
        ampm = self._validate_value("ampm", ampm, 0, 1)
        for word in words:
            word = self._validate_word("ampm", word)
            dict_setitem(self._ampm, word, ampm)

    def rem_ampm(self, *words: str) -> None:
        """Remove 'words' that should `NOT` be recognized as AM/PM.

        ### Example
        >>> cfg.rem_ampm("am", "a.m.", ...)
        """
        for word in words:
            try:
                dict_delitem(self._ampm, word)
            except KeyError:
                pass
            set_discard(self._words, word)

    def set_ampm(self, **word_n_ampm: int) -> None:
        """Set the 'words' that should be recognized as AM/PM.
        - Replaces the current AM/PM words with the specified 'words'.
        - If 'words' are not provided, resets to the default AM/PM words.

        ### Example
        >>> cfg.set_ampm(
                am=0, pm=1,
                **{"a.m."=0, "p.m."=1}
                ...
            )
        """
        if word_n_ampm:
            self._ampm = word_n_ampm
        else:
            self._ampm = CONFIG_AMPM
        self._construct_words()

    # Validate --------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _construct_words(self) -> cython.bint:
        """(cfunc) Construct the the Config words `<'bool'>`."""
        # Reset the words
        self._words = set()
        # fmt: off
        # . month
        self._month = {
            self._validate_word("month", word):
            self._validate_value("month", value, 1, 12)
            for word, value in self._month.items()
        }
        # . weekday
        self._weekday = {
            self._validate_word("weekday", word):
            self._validate_value("weekday", value, 0, 6)
            for word, value in self._weekday.items()
        }
        # . hms
        self._hms = {
            self._validate_word("hms", word):
            self._validate_value("hms", value, 0, 2)
            for word, value in self._hms.items()
        }
        # . ampm
        self._ampm = {
            self._validate_word("ampm", word):
            self._validate_value("ampm", value, 0, 1)
            for word, value in self._ampm.items()
        }
        # . utc
        self._utc = {
            self._validate_word("utc", word)
            for word in self._utc }
        # . timezone
        self._tz = {
            self._validate_word("tz", word):
            self._validate_value("tz", value, -86_340, 86_340)
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
    def _validate_word(self, setting: str, word: object) -> object:
        """(cfunc) Validate if the word conflicts (duplicated) with the
        exsiting words in the settings, which could lead to unexpected
        parsing result `<'str'>`."""
        # Validate the word type
        try:
            w: str = word
        except Exception as err:
            raise errors.InvalidConfigWord(
                "<'%s'>\nThe 'word' for [Config.%s] must be <'str'>, instead got: %s %s"
                % (self.__class__.__name__, setting, type(word), repr(word))
            ) from err
        w = w.lower()

        # Validate if the word is conflicting with other words.
        # Skip jump words, because jump has the maximum freedom.
        if set_contains(self._words, w) and not set_contains(self._jump, w):
            conf_setting: str = None
            if dict_contains(self._month, w):
                if setting != "month":
                    conf_setting = "month"
            elif dict_contains(self._weekday, w):
                if setting != "weekday":
                    conf_setting = "weekday"
            elif dict_contains(self._hms, w):
                if setting != "hms":
                    conf_setting = "hms"
            elif dict_contains(self._ampm, w):
                if setting != "ampm":
                    conf_setting = "ampm"
            elif set_contains(self._utc, w):
                if setting != "utc" and setting != "tz":
                    conf_setting = "utc"
            elif dict_contains(self._tz, w):
                if setting != "tz":
                    conf_setting = "tz"
            elif set_contains(self._pertain, w):
                if setting != "pertain":
                    conf_setting = "pertain"
            # raise error
            if conf_setting is not None:
                raise errors.InvalidConfigWord(
                    "<'%s'>\nThe word '%s' for [Config.%s] conflicts "
                    "(duplicated) with words in [Config.%s]."
                    % (self.__class__.__name__, w, setting, conf_setting)
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
        """(cfunc) Validate if the value of a word is in
        the valid range (min...max) `<'int'>`."""
        try:
            v: cython.int = value
        except Exception as err:
            raise errors.InvalidConfigValue(
                "<'%s'>\nThe value for [Config.%s] must be <'int'> instead got: %s %s."
                % (self.__class__.__name__, setting, type(value), repr(value))
            ) from err
        if not min <= v <= max:
            raise errors.InvalidConfigValue(
                "<'%s'>\nThe value for [Config.%s] must between %d...%d, instead got: %s."
                % (self.__class__.__name__, setting, min, max, value)
            )
        return value

    # Special methods -------------------------------------------------
    def __repr__(self) -> str:
        reprs: list = [
            "day1st=%s" % self._day1st,
            "year1st=%s" % self._year1st,
            "pertain=%s" % sorted(self._pertain),
            "jump=%s" % sorted(self._jump),
            "utc=%s" % sorted(self._utc),
            "tz=%s" % self._tz,
            "month=%s" % self._month,
            "weekday=%s" % self._weekday,
            "hms=%s" % self._hms,
            "ampm=%s" % self._ampm,
        ]
        return "<%s(\n  *%s\n)>" % (self.__class__.__name__, ",\n  *".join(reprs))


# Parser --------------------------------------------------------------------------------------
@cython.cclass
class Parser:
    """Represents the datetime Parser."""

    # Config
    _day1st: cython.bint
    _year1st: cython.bint
    _ignoretz: cython.bint
    _pertain: set[str]
    _jump: set[str]
    _utc: set[str]
    _tz: dict[str, int]
    _month: dict[str, int]
    _weekday: dict[str, int]
    _hms: dict[str, int]
    _ampm: dict[str, int]
    # Result
    _result: Result
    # Process
    _iso_type: cython.uint
    _tokens: list[str]
    _token_count: cython.Py_ssize_t
    _index: cython.Py_ssize_t
    _token1: str
    _token2: str
    _token3: str
    _token4: str

    def __init__(self, cfg: Config = None) -> None:
        """The datetime Parser.

        :param cfg `<'Config/None'>`: The configurations for the Parser. Defaults to `None`.

        For more information about `<'Config'>`, please refer to the Config class.
        """
        # Load specifed config
        if cfg is not None:
            self._day1st = cfg._day1st
            self._year1st = cfg._year1st
            self._pertain = cfg._pertain
            self._jump = cfg._jump
            self._utc = cfg._utc
            self._tz = cfg._tz
            self._month = cfg._month
            self._weekday = cfg._weekday
            self._hms = cfg._hms
            self._ampm = cfg._ampm
        # Load default config
        else:
            self._day1st = False
            self._year1st = False
            self._pertain = CONFIG_PERTAIN
            self._jump = CONFIG_JUMP
            self._utc = CONFIG_UTC
            self._tz = CONFIG_TZ
            self._month = CONFIG_MONTH
            self._weekday = CONFIG_WEEKDAY
            self._hms = CONFIG_HMS
            self._ampm = CONFIG_AMPM

    # Parse --------------------------------------------------------------------------------
    @cython.ccall
    def parse(
        self,
        dtstr: str,
        default: datetime.datetime | datetime.date | None = None,
        day1st: bool | None = None,
        year1st: bool | None = None,
        ignoretz: bool = False,
        isoformat: bool = True,
    ) -> datetime.datetime:
        """Parse the datetime string into `<'datetime'>`.

        ### Time String & Default
        :param dtstr `<'str'>`: The string that contains datetime information.
        :param default `<'datetime/date'>`: The default to fill-in missing datetime values. Defaults to `None`.
        - `<'date'>`: If parser failed to extract Y/M/D values from the string,
           the give 'date' will be used to fill-in the missing Y/M/D values.
        - `<'datetime'>`: If parser failed to extract datetime elements from
           the string, the given 'datetime' will be used to fill-in the
           missing Y/M/D & H/M/S.f values.
        - `None`: raise `cyParserBuildError` if Y/M/D values are missing.

        ### Ambiguous Y/M/D
        :param day1st `<'bool'>`: Whether to interpret first ambiguous date values as day. Defaults to `None`.
        :param year1st `<'bool'>`: Whether to interpret first the ambiguous date value as year. Defaults to `None`.

        Both the 'day1st' & 'year1st' arguments works together to determine how
        to interpret ambiguous Y/M/D values. If not provided (set to `None`),
        defaults to the 'day1st' & 'year1st' settings from Parser `<'Config'>`.

        In the case when all three values are ambiguous (e.g. `01/05/09`):
        - If 'day1st=True' and 'year1st=True', the date will be interpreted as `'Y/D/M'`.
        - If 'day1st=False' and 'year1st=True', the date will be interpreted as `'Y/M/D'`.
        - If 'day1st=True' and 'year1st=False', the date will be interpreted as `'D/M/Y'`.
        - If 'day1st=False' and 'year1st=False', the date will be interpreted as `'M/D/Y'`.

        In the case when the year value is clear (e.g. `2010/01/05` or `99/01/05`):
        - If 'day1st=True', the date will be interpreted as `'Y/D/M'`.
        - If 'day1st=False', the date will be interpreted as `'Y/M/D'`.

        In the case when only one value is ambiguous (e.g. `01/20/2010` or `01/20/99`),
        arguments 'day1st' or 'year1st' takes no effect.

        ### Timezone
        :param ignoretz `<'bool'>`: Whether to ignore timezone information. Defaults to `False`.
        - `True`: Parser ignores any timezone information and only returns
           timezone-naive datetime. Setting to `True` can increase parser
           performance.
        - `False`: Parser will try to process the timzone information in
           the string, and generate a timezone-aware datetime if timezone
           has been matched by the Parser `<'Config'>` settings: 'utc' & 'tz'.

        ### ISO Format
        :param isoformat `<'bool'>`: Whether the 'dtstr' is in ISO format. Defaults to `True`.
        - `True`: Parser will first try to process the 'dtstr' in ISO format.
           If failed, then process the 'dtstr' through timelex tokens.
        - `False`: Parser will only process the 'dtstr' through timelex tokens.
           If the 'dtstr' is confirmed to not be in ISO format, setting to
           `False` can increase parser performance.
        """
        # Setup
        if day1st is not None:
            self._day1st = bool(day1st)
        if year1st is not None:
            self._year1st = bool(year1st)
        self._ignoretz = ignoretz

        # Process
        self._process(dtstr, bool(isoformat))
        return self._build(dtstr, default)

    # Process ------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _process(self, dtstr: str, isoformat: cython.bint) -> cython.bint:
        """Process the datetime string into `<'bool'>`."""
        # Lowercase
        dtstr = dtstr.lower()

        # Process
        try:
            # . process ISO format
            if isoformat and self._process_isoformat(dtstr):
                return True  # exit: success
            # . process as tokens
            if self._process_tokens(dtstr):
                return True  # exit: success
        # Process Error
        except MemoryError:
            raise
        except errors.cyParserError as err:
            err.add_note("-> Unable to parse: '%s'." % dtstr)
            raise err
        except Exception as err:
            raise errors.cyParserFailedError(
                "<'%s'>\nUnable to parse: '%s'.\nError: %s"
                % (self.__class__.__name__, dtstr, err)
            ) from err

        # Process Failed
        raise errors.cyParserFailedError(
            "<'%s'>\nUnable to parse: '%s'." % (self.__class__.__name__, dtstr)
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _process_isoformat(self, dtstr: str) -> cython.bint:
        """(cfunc) Process 'dtstr' as ISO format `<'bool'>`."""
        # Find ISO format datetime separator
        length: cython.Py_ssize_t = str_len(dtstr)
        sep_loc: cython.Py_ssize_t = self._find_iso_sep(dtstr, length)
        if sep_loc == 0:
            return False  # exit: not ISO format

        # Parse iso format
        if sep_loc == length:
            # . process date component
            self._result = Result()
            if not self._parse_iso_date(dtstr, sep_loc):
                return False  # exit: not ISO format
        else:
            # . verify datetime seperator (' ' or 'T')
            if not is_iso_sep(read_char(dtstr, sep_loc)):  # type: ignore
                return False  # exit: not ISO format
            # . process date component
            self._result = Result()
            if not self._parse_iso_date(dtstr[0:sep_loc], sep_loc):
                return False  # exit: not ISO format
            # . process time component
            tstr: str = dtstr[sep_loc + 1 : length]
            tstr_len: cython.Py_ssize_t = length - sep_loc - 1
            if not self._parse_iso_time(tstr, tstr_len):
                return False  # exit: not ISO format

        # Prepare result
        self._result.prepare(self._day1st, self._year1st)
        return self._result.valid()  # exit: success/fail

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _process_tokens(self, dtstr: str) -> cython.bint:
        """(cfunc) Process 'dtstr' through timelex tokens `<'bool'>`."""
        # Parse tokens
        self._parse_tokens(dtstr, str_len(dtstr))
        self._result = Result()

        # Parse token
        while self._index < self._token_count:
            # . access token
            token = cython.cast(str, list_getitem(self._tokens, self._index))
            self._reset_tokens()
            # . numeric token
            if self._parse_token_numeric(token):
                self._index += 1
            # . month token
            elif self._parse_token_month(token):
                self._index += 1
            # . weekday token
            elif self._parse_token_weekday(token):
                self._index += 1
            # . am/pm token
            elif self._parse_token_ampm(token):
                self._index += 1
            # . tzname token
            elif self._parse_token_tzname(token):
                self._index += 1
            # . tzoffset token
            elif self._parse_token_tzoffset(token):
                self._index += 1
            # . jump token
            elif self._is_token_jump(token):
                self._index += 1
            # . fuzzy parsing
            else:
                self._index += 1

        # Prepare result
        self._result.prepare(self._day1st, self._year1st)
        return self._result.valid()  # exit: success/fail

    # Build --------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _build(self, dtstr: str, default: object) -> datetime.datetime:
        """(cfunc) Build form process `Result` `<'datetime'>`."""
        # Timeonze-naive
        if self._ignoretz:
            return self._build_datetime(dtstr, default, None)

        # Timezone-aware
        tzname: str = self._result.tzname
        offset: cython.int = self._result.tzoffset
        # . local timezone (handle ambiguous time)
        if tzname is not None and set_contains(TIMEZONE_NAME_LOCAL, tzname):
            # Build with local timezone
            dt = self._build_datetime(dtstr, default, None)
            dt = cydt.dt_replace_tzinfo(dt, cydt.gen_tzinfo_local(dt))
            # Handle ambiguous local datetime
            dt = self._handle_ambiguous_time(dt, tzname)
            # Adjust for winter GMT zones parsed in the UK
            if dt.tzname() != tzname and tzname == "UTC":
                dt = cydt.dt_replace_tzinfo(dt, cydt.UTC)
            return dt  # exit: finished
        # . utc (tzoffset == 0)
        elif offset == 0:
            tz = cydt.UTC
        # . other timezone
        elif offset != -100_000:
            tz = cydt.gen_tzinfo(offset)
        # . timezone-naive
        else:
            tz = None
        # Build with timezone
        return self._build_datetime(dtstr, default, tz)

    @cython.cfunc
    @cython.inline(True)
    def _build_datetime(
        self,
        dtstr: str,
        default: object,
        tz: object,
    ) -> datetime.datetime:
        """(Internal) Build the `<'datetime'>`."""
        # Set build mode
        mode: cython.uint
        if datetime.PyDate_Check(default):
            # Use the <'datetime.datetime'> to fill-in
            # the missing Y/M/D & H:M:S.f values.
            if datetime.PyDateTime_Check(default):
                mode = 2
            # Use the <'datetime.date'> to fill-in
            # the missing Y/M/D values.
            else:
                mode = 1
        else:
            # Raise error if Y/M/D values are missing.
            mode = 0

        # Year
        year: cython.uint
        if self._result.year != -1:
            year = self._result.year
        elif mode > 0:
            year = datetime.PyDateTime_GET_YEAR(default)
        else:
            raise errors.cyParserBuildError(
                "<'%s'>\nUnable to build datetime: lack of 'year' from '%s'."
                % (self.__class__.__name__, dtstr)
            )

        # Month
        month: cython.uint
        if self._result.month != -1:
            month = self._result.month
        elif mode > 0:
            month = datetime.PyDateTime_GET_MONTH(default)
        else:
            raise errors.cyParserBuildError(
                "<'%s'>\nUnable to build datetime: lack of 'month' from '%s'."
                % (self.__class__.__name__, dtstr)
            )

        # Day
        day: cython.uint
        if self._result.day != -1:
            day = self._result.day
        elif mode > 0:
            day = datetime.PyDateTime_GET_DAY(default)
        else:
            raise errors.cyParserBuildError(
                "<'%s'>\nUnable to build datetime: lack of 'day' from for '%s'."
                % (self.__class__.__name__, dtstr)
            )

        # Hour
        hour: cython.uint
        if self._result.hour != -1:
            hour = self._result.hour
        elif mode == 2:
            hour = datetime.PyDateTime_DATE_GET_HOUR(default)
        else:
            hour = 0

        # Minute
        minute: cython.uint
        if self._result.minute != -1:
            minute = self._result.minute
        elif mode == 2:
            minute = datetime.PyDateTime_DATE_GET_MINUTE(default)
        else:
            minute = 0

        # Second
        second: cython.uint
        if self._result.second != -1:
            second = self._result.second
        elif mode == 2:
            second = datetime.PyDateTime_DATE_GET_SECOND(default)
        else:
            second = 0

        # Microsecond
        if self._result.microsecond != -1:
            microsecond = self._result.microsecond
        elif mode == 2:
            microsecond = datetime.PyDateTime_DATE_GET_MICROSECOND(default)
        else:
            microsecond = 0

        # Generate datetime
        try:
            dt: datetime.datetime = datetime.datetime_new(
                year, month, day, hour, minute, second, microsecond, tz, 0
            )
        except Exception as err:
            raise errors.cyParserBuildError(
                "<'%s'>\nUnable to build datetime: %s for '%s'."
                % (self.__class__.__name__, err, dtstr)
            ) from err

        # Adjust weekday
        if self._result.weekday != -1:
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
        """(cfunc) Handle ambiguous time `<'datetime'>`."""
        if dt.tzname() != tzname:
            new_dt: datetime.datetime = cydt.dt_replace_fold(dt, 1)
            if new_dt.tzname() == tzname:
                return new_dt
        return dt

    # ISO format ---------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_iso_date(self, dstr: str, length: cython.Py_ssize_t) -> cython.bint:
        """(cfunc) Process the date components of the ISO format `<'bool'>`.

        'iso_type' meaning:
        - 0: Not ISO format
        - 1: "YYYY-MM-DD"
        - 2: "YYYYMMDD"
        - 3: "YYYY-Www-D"
        - 4: "YYYYWwwD"
        - 5: "YYYY-Www"
        - 6: "YYYYWww"
        - 7: "YYYY-DDD"
        - 8: "YYYYDDD"
        """
        # Process year
        try:
            year: cython.uint = slice_to_int(dstr, 0, 4)  # type: ignore
        except Exception:
            return False  # exit: invalid year
        if not 1 <= year <= 9_999:
            return False  # exit: invalid year
        self._result.append(year, 1)
        self._result.century_specified = True

        # Process YYYY-MM-DD & YYYYMMDD
        i1_s: cython.Py_ssize_t
        i1_e: cython.Py_ssize_t
        i2_s: cython.Py_ssize_t
        i2_e: cython.Py_ssize_t
        if self._iso_type <= 2:
            if self._iso_type == 1 and length == 10:  # YYYY-MM-DD
                i1_s, i1_e, i2_s, i2_e = 5, 7, 8, 10
                i2_s, i2_e = 8, 10
            elif self._iso_type == 2 and length == 8:  # YYYYMMDD
                i1_s, i1_e, i2_s, i2_e = 4, 6, 6, 8
            else:
                return False  # exit: not ISO format
            # . validate month
            try:
                month: cython.uint = slice_to_int(dstr, i1_s, i1_e)  # type: ignore
            except Exception:
                return False  # exit: invalid month
            if not 1 <= month <= 12:
                return False  # exit: invalid month
            # . validate day
            try:
                day: cython.uint = slice_to_int(dstr, i2_s, i2_e)  # type: ignore
            except Exception:
                return False  # exit: invalid day
            if not 1 <= day <= cydt.days_in_month(year, month):
                return False  # exit: invalid day
            # . append 'month' & 'day'
            self._result.append(month, 2)
            self._result.append(day, 3)
            return True  # exit: success

        # Process YYYY-Www-D, YYYYWwwD, YYYY-Www, YYYYWww
        if self._iso_type <= 6:
            if self._iso_type == 3 and length == 10:  # YYYY-Www-D
                i1_s, i1_e, i2_s, i2_e = 6, 8, 9, 10
            elif self._iso_type == 4 and length == 8:  # YYYYWwwD
                i1_s, i1_e, i2_s, i2_e = 5, 7, 7, 8
            elif self._iso_type == 5 and length == 8:  # YYYY-Www
                i1_s, i1_e, i2_s, i2_e = 6, 8, 0, 0
            elif self._iso_type == 6 and length == 7:  # YYYYWww
                i1_s, i1_e, i2_s, i2_e = 5, 7, 0, 0
            else:
                return False  # exit: not ISO format
            # . validate week
            try:
                week: cython.uint = slice_to_int(dstr, i1_s, i1_e)  # type: ignore
            except Exception:
                return False  # exit: invalid week
            if not 1 <= week <= 53:
                return False  # exit: invalid week
            # . validate weekday
            if i2_s != 0:
                try:
                    weekday: cython.uint = slice_to_int(dstr, i2_s, i2_e)  # type: ignore
                except Exception:
                    return False  # exit: invalid weekday
                if not 1 <= weekday <= 7:
                    return False  # exit: invalid weekday
            else:
                weekday: cython.uint = 1
            # . append 'month' & 'day'
            ymd = cydt.isocalendar_to_ymd(year, week, weekday)
            self._result.append(ymd.month, 2)
            self._result.append(ymd.day, 3)
            return True  # exit: success

        # Process YYYY-DDD, YYYYDDD
        if self._iso_type <= 8:
            if self._iso_type == 7 and length == 8:  # YYYY-DDD
                i2_s, i2_e = 5, 8
            elif self._iso_type == 8 and length == 7:  # YYYYDDD
                i2_s, i2_e = 4, 7
            else:
                return False  # exit: not ISO format
            # . validate days
            try:
                days: cython.uint = slice_to_int(dstr, i2_s, i2_e)  # type: ignore
            except Exception:
                return False  # exit: invalid days of year
            if not 1 <= days <= 366:
                return False  # exit: invalid days of year
            # . append 'month' & 'day'
            ymd = cydt.days_of_year_to_ymd(year, days)
            self._result.append(ymd.month, 2)
            self._result.append(ymd.day, 3)
            return True

        # . Not ISO format
        return False

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_iso_time(self, tstr: str, length: cython.Py_ssize_t) -> cython.bint:
        """(cfunc) Process the time components of the ISO format `<'bool'>`."""
        # Validate 'tstr'
        if length < 2:
            return False  # exit: ISO format time to short [HH].

        # Find ISO format timezone position
        if self._ignoretz:
            pos: cython.Py_ssize_t = 0
        else:
            pos: cython.Py_ssize_t = self._find_iso_utc(tstr, length)

        # Process HMS.f (w/o ISO format timezone)
        if pos == 0:
            return self._parse_iso_hms(tstr, length)  # exit: success/fail

        # Process HMS.f (w/t ISO format timezone)
        hms_len: cython.Py_ssize_t = pos - 1
        if not self._parse_iso_hms(tstr[0:hms_len], hms_len):
            return False  # exit: invalid time component

        # Process z(UTC) timzeone
        sep: cython.Py_UCS4 = read_char(tstr, pos - 1)
        if sep == "z":
            # . HMS ends with 'z'
            if pos == length:
                self._result.tzoffset = 0
                return True  # exit: success
            # . HMS followed by 'z', ' ' and extra characters
            if length > pos + 1 and read_char(tstr, pos) == " ":
                self._result.tzoffset = 0
                return self._parse_iso_extra(tstr[pos:length], length - pos, True)

        # Process numeric timezone
        loc: cython.Py_ssize_t = pos + 2
        if loc > length:
            return False  # exit: incomplete offset
        # . parse timezone hour
        try:
            hour: cython.uint = slice_to_int(tstr, pos, loc)  # type: ignore
        except Exception:
            return False  # exit: invalid offset
        # . parse timezone minute
        minute: cython.uint
        if loc < length:
            ch: cython.Py_UCS4 = read_char(tstr, loc)
            if is_isotime_sep(ch):  # type: ignore
                pos += 3
                loc += 3
            else:
                pos += 2
                loc += 2
            if loc > length:
                return False  # exit: incomplete offset
            try:
                minute = slice_to_int(tstr, pos, loc)  # type: ignore
            except Exception:
                return False  # exit: invalid offset
        else:
            minute = 0
        # . calculate offset
        sign: cython.int = 1 if sep == "+" else -1
        offset: cython.int = sign * (hour * 3_600 + minute * 60)
        if self._result.tzoffset != -100_000:
            # [original tzoffset] + [offset] => [UTC offset]
            offset = self._result.tzoffset - offset
        # . validate offset
        if not -86_340 <= offset <= 86_340:
            raise errors.InvalidTokenError(
                "<'%s'>\nInvalid timezone offset: '%ds'."
                % (self.__class__.__name__, offset)
            )
        self._result.tzoffset = offset

        # Parse extra characters
        if loc < length:
            return self._parse_iso_extra(tstr[loc:length], length - loc, True)

        # Finished
        return True  # exit: success

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_iso_hms(self, tstr: str, length: cython.Py_ssize_t) -> cython.bint:
        """(cfunc) Parse the HMS.f components of the ISO format `<'bool'>`."""
        # Process HMS
        hmsf: cython.uint[4] = [0, 0, 0, 0]
        pos: cython.Py_ssize_t = 0
        idx: cython.Py_ssize_t
        sep: cython.bint
        val: cython.uint
        ch: cython.Py_UCS4
        for idx in range(0, 3):
            # . validate component
            if (length - pos) < 2:
                return False  # exit: incomplete HMS.f
            # . parse component
            try:
                val = slice_to_int(tstr, pos, pos + 2)  # type: ignore
            except Exception:
                return False  # exit: invalid HMS
            hmsf[idx] = val
            # . validate time seperator
            pos += 2
            ch = read_char(tstr, pos) if pos < length else 0
            if ch == 0 or idx >= 2:
                break
            if idx == 0:
                sep = is_isotime_sep(ch)  # type: ignore
            if sep and not is_isotime_sep(ch):  # type: ignore
                return False  # exit: invalid HMS seperator
            pos += sep

        # Process microsecond & (possible) timezone
        if pos < length:
            # Validate microsecond component
            if (ch := read_char(tstr, pos)) in ".,":  # separator [.,]
                pos += 1
                # . search for microsecond
                loc: cython.Py_ssize_t = pos
                while loc < length and is_ascii_digit(read_char(tstr, loc)):  # type: ignore
                    loc += 1
                if pos == loc:
                    return False  # exit: incomplete microsecond
                # . microsecond component
                try:
                    val = token_to_microsecond(tstr[pos:loc], loc - pos)  # type: ignore
                except Exception:
                    return False  # exit: invalid microsecond
                hmsf[3] = val
                pos = loc

            # Extra characters (timezone)
            if pos < length:
                # Append HMS
                self._result.hour = hmsf[0]
                self._result.minute = hmsf[1]
                self._result.second = hmsf[2]
                self._result.microsecond = hmsf[3]
                # Parse extra characters
                return self._parse_iso_extra(
                    tstr[pos:length], length - pos, self._ignoretz
                )

        # Append HMS
        self._result.hour = hmsf[0]
        self._result.minute = hmsf[1]
        self._result.second = hmsf[2]
        self._result.microsecond = hmsf[3]
        return True  # exit: success

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_iso_extra(
        self,
        estr: str,
        length: cython.Py_ssize_t,
        ignoretz: cython.bint,
    ) -> cython.bint:
        """(cfunc) Parse the extra components of the ISO format `<'int'>`.
        Extra components limits to AM/PM & timezone name.
        """
        # Parse tokens
        self._parse_tokens(estr, length)

        # Process tokens
        while self._index < self._token_count:
            # . access token
            token = cython.cast(str, list_getitem(self._tokens, self._index))
            self._reset_tokens()
            # . am/pm token
            if self._parse_token_ampm(token):
                self._index += 1
            # . tzname token
            elif not ignoretz and self._parse_token_tzname(token):
                self._index += 1
            # . jump token
            else:
                self._index += 1

        # Finished
        return True  # exit: success

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _find_iso_sep(self, dtstr: str, length: cython.Py_ssize_t) -> cython.Py_ssize_t:
        """(cfunc) This function tries to find the datetime separator
        `"T"` or `" "` position in the datetime string. Returns `0` if
        the string is certainly not ISO format, else the possible
        separator position `<'int'>`.

        This function also sets the 'iso_type':
        - 0: Not ISO format
        - 1: "YYYY-MM-DD"
        - 2: "YYYYMMDD"
        - 3: "YYYY-Www-D"
        - 4: "YYYYWwwD"
        - 5: "YYYY-Www"
        - 6: "YYYYWww"
        - 7: "YYYY-DDD"
        - 8: "YYYYDDD"
        """
        # ISO format string length must be >= 7.
        if length < 7:
            self._iso_type = 0
            return 0  # exit: not ISO format: minimum YYYYWww or YYYYDDD

        # Find datetime separator
        char4: cython.Py_UCS4 = read_char(dtstr, 4)
        # YYYY[-]
        if is_isodate_sep(char4):  # type: ignore
            if length < 8:
                self._iso_type = 0
                return 0  # exit: not ISO format, minimum YYYY-MM-DD
            # YYYY-[W]
            char5: cython.Py_UCS4 = read_char(dtstr, 5)
            if is_isoweek_sep(char5):  # type: ignore
                # YYYY-Www[-]
                if length > 8 and is_isodate_sep(read_char(dtstr, 8)):  # type: ignore
                    if length == 9:  # [YYYY-Www-]
                        self._iso_type = 0
                        return 0  # exit: not ISO format
                    elif is_ascii_digit(read_char(dtstr, 10)):  # type: ignore
                        self._iso_type = 5
                        return 8  # exit: [YYYY-Www]
                    else:
                        self._iso_type = 3
                        return 10  # exit: [YYYY-Www-D]
                else:
                    self._iso_type = 5
                    return 8  # exit: [YYYY-Www]
            # YYYY-[M]
            elif is_ascii_digit(char5):  # type: ignore
                char7: cython.Py_UCS4 = read_char(dtstr, 7)
                if length >= 10 and is_isodate_sep(char7):  # type: ignore
                    self._iso_type = 1
                    return 10  # exit: [YYYY-MM-DD]
                elif is_ascii_digit(char7):  # type: ignore
                    self._iso_type = 7
                    return 8  # exit: [YYYY-DDD]

        # YYYY[W]
        elif is_isoweek_sep(char4):  # type: ignore
            # YYYYWw[w]
            if is_ascii_digit(read_char(dtstr, 6)):  # type: ignore
                if is_ascii_digit(read_char(dtstr, 7)):  # type: ignore
                    self._iso_type = 4
                    return 8  # exit: [YYYYWwwD]
                else:
                    self._iso_type = 6
                    return 7  # exit: [YYYYWww]

        # YYYY[D]
        elif is_ascii_digit(char4):  # type: ignore
            # YYYYDD[D] / YYYYMM[D]
            if is_ascii_digit(read_char(dtstr, 6)):  # type: ignore
                if is_ascii_digit(read_char(dtstr, 7)):  # type: ignore
                    self._iso_type = 2
                    return 8  # exit: [YYYYMMDD]
                else:
                    self._iso_type = 8
                    return 7  # exit: [YYYYDDD]

        # Invalid ISO format
        return 0  # exit: not ISO format

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _find_iso_utc(self, tstr: str, length: cython.Py_ssize_t) -> cython.Py_ssize_t:
        """(cfunc) This function tries to find the ISO format UTC timezone
        identifier `"+"`, `"-"` or `"z"` position in the time string.
        Returns `0` if UTC identifier is not found `<'int'>`."""
        loc: cython.Py_ssize_t
        # Find '+' position
        if (loc := str_findc(tstr, "+", 0, length, 1)) >= 0:
            return loc + 1
        # Find '-' position
        if (loc := str_findc(tstr, "-", 0, length, 1)) >= 0:
            return loc + 1
        # Find 'z' position
        if (loc := str_findc(tstr, "z", 0, length, 1)) >= 0:
            return loc + 1
        # UTC not found
        return 0

    # Tokens -------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_token_numeric(self, token: str) -> cython.bint:
        """(cfunc) Parse the 'numeric' token. Returns
        True if the token represents numeric value and
        processed successfully, else False `<'bool'>`."""
        # Convert token to numeric value
        try:
            t_num: cython.double = float(token)
        except ValueError:
            return False  # exit: not a numeric token
        if not isfinite(t_num):
            return False  # exit: invalid infinite value ('inf')

        # Get token length
        t_len: cython.Py_ssize_t = str_len(token)

        # (19990101T)23[59]
        t1: str = self._access_token1()  # next (+1) token
        if (
            self._result.populated() == 3  # Y/M/D all populated
            and self._result.hour == -1  # hour not determined
            and (t_len == 2 or t_len == 4)  # '23' or '2359'
            and (
                self._index + 1 >= self._token_count  # next (+2) token exists
                or (
                    t1 is not None  # next (+1) token exists
                    and t1 != ":"  # next (+1) token is not ':'
                    and self._token_to_hms(t1) == -1  # next (+1) token is not hms
                )
            )
        ):
            self._result.hour = slice_to_int(token, 0, 2)  # type: ignore
            if t_len == 4:
                self._result.minute = slice_to_int(token, 2, 4)  # type: ignore
            return True  # exit

        # YYMMDD or HHMMSS[.ss]
        if t_len == 6 or (t_len > 6 and read_char(token, 6) == "."):
            if self._result.populated() == 0 and not "." in token:
                self._result.append(token[0:2], 0)
                self._result.append(token[2:4], 0)
                self._result.append(token[4:t_len], 0)
            else:
                # 19990101T235959[.59]
                self._result.hour = slice_to_int(token, 0, 2)  # type: ignore
                self._result.minute = slice_to_int(token, 2, 4)  # type: ignore
                self._set_sf(token[4:t_len])
            return True  # exit

        # YYYYMMDD[HHMMSS]
        if t_len == 8 or t_len == 12 or t_len == 14:
            self._result.append(token[0:4], 1)
            self._result.append(token[4:6], 0)
            self._result.append(token[6:8], 0)
            if t_len > 8:
                self._result.hour = slice_to_int(token, 8, 10)  # type: ignore
                self._result.minute = slice_to_int(token, 10, 12)  # type: ignore
                if t_len > 12:
                    self._result.second = slice_to_int(token, 12, 14)  # type: ignore
            return True  # exit

        # HH[ ]h or MM[ ]m or SS[.ss][ ]s
        if self._parse_token_hms(token, t_num):
            return True  # exit

        # HH:MM[:SS[.ss]]
        t2: str = self._access_token2()
        if t2 is not None and t1 == ":":
            # . HH:MM
            self._result.hour = int(t_num)
            self._set_ms(token_to_float(t2))  # type: ignore
            # . SS:[.ss]
            t4: str = self._access_token4()
            if t4 is not None and self._access_token3() == ":":
                self._set_sf(t4)
                self._index += 2  # skip SS.ss
            self._index += 2  # skip HH:MM
            return True  # exit

        # YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
        if t1 is not None and t1 in ("-", "/", "."):
            # 1st Y/M/D value
            self._result.append(token, 0)

            # 2nd Y/M/D value
            if t2 is not None and not self._is_token_jump(t2):
                try:
                    # 01-01[-01]
                    month = int(t2)
                    self._result.append(month, 0)
                except ValueError:
                    # 01-Jan[-01]
                    month = self._token_to_month(t2)
                    if month != -1:
                        self._result.append(month, 2)
                    else:
                        self._result.append(t2, 0)

                # 3rd Y/M/D value
                t4: str = self._access_token4()
                if t4 is not None and self._access_token3() == t1:  # sep
                    month = self._token_to_month(t4)
                    if month != -1:
                        self._result.append(month, 2)
                    else:
                        self._result.append(t4, 0)
                    self._index += 2  # skip 3rd Y/M/D
                self._index += 1  # skip 2nd Y/M/D
            self._index += 1  # skip 1st Y/M/D
            return True  # exit

        # "hour AM" or year|month|day
        if self._index + 1 >= self._token_count or self._is_token_jump(t1):
            if t2 is not None and (ampm := self._token_to_ampm(t2)) != -1:
                # 12 AM
                self._result.hour = self._adjust_ampm(int(t_num), ampm)
                self._index += 1  # skip AMPM
            else:
                # Year, month or day
                self._result.append(token, 0)
            self._index += 1  # skip token t1
            return True  # exit

        # "hourAM"
        if 0 <= t_num < 24 and (ampm := self._token_to_ampm(t1)) != -1:
            self._result.hour = self._adjust_ampm(int(t_num), ampm)
            self._index += 1  # skip token t1
            return True  # exit

        # Possible is day
        if self._result.could_be_day(int(t_num)):
            self._result.append(token, 0)
            return True  # exit

        # Exit
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_token_month(self, token: str) -> cython.bint:
        """(cfunc) Parse the 'month' token. Returns
        True if the token represents month value and
        processed successfully, else False `<'bool'>`."""
        # Validate if is month token
        month = self._token_to_month(token)
        if month == -1:
            return False  # exit: not a month token
        self._result.append(month, 2)  # append month

        # Try to parse year & day
        t2: str = self._access_token2()
        if t2 is not None:
            t1: str = self._access_token1()
            if t1 in ("-", "/"):
                # Jan-01[-99?] uncertain
                self._result.append(t2, 0)
                t4: str = self._access_token4()
                if t4 is not None and self._access_token3() == t1:  # sep
                    # Jan-01-99 confirmed
                    self._result.append(t4, 0)
                    self._index += 2  # skip token t3 & t4
                self._index += 2  # skip token t1 & t2
                return True  # exit

            # Jan of 01. In this case, 01 is clearly year
            t4: str = self._access_token4()
            if (
                t4 is not None
                and self._is_token_pertain(t2)
                and self._access_token3() == " "
            ):
                try:
                    year = int(t4)
                    self._result.append(year, 1)
                except ValueError:
                    pass  # wrong guess
                self._index += 4  # skip token t1 - t4
                return True  # exit

        # Finished
        return True  # exit

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_token_weekday(self, token: str) -> cython.bint:
        """(cfunc) Parse the 'weekday' token. Returns
        True if the token represents weekday value and
        processed successfully, else False `<'bool'>`."""
        # Validate if is weekday token
        wkd = self._token_to_weekday(token)
        if wkd == -1:
            return False  # exit: not a weekday token
        # Set parse result
        else:
            self._result.weekday = wkd
            return True  # exit

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_token_hms(self, token: str, t_num: cython.double) -> cython.bint:
        """(cfunc) Parse the 'hms' token. Returns
        True if the token represents hms value and
        processed successfully, else False `<'bool'>`."""
        hms: cython.int
        # Looking forwards
        t1: str = self._access_token1()
        if t1 is not None:
            # There is an "h", "m", or "s" label following this token.
            # We take assign the upcoming label to the current token.
            # e.g. the "12" in 12h"
            if (hms := self._token_to_hms(t1)) != -1:
                self._set_hms(token, t_num, hms)
                self._index += 1  # skip token t1
                return True

            # There is a space and then an "h", "m", or "s" label.
            # e.g. the "12" in "12 h"
            t2: str = self._access_token2()
            if t2 is not None and t1 == " " and (hms := self._token_to_hms(t2)) != -1:
                self._set_hms(token, t_num, hms)
                self._index += 2  # skip token t1 & t2
                return True

        # Looking backwords
        t1: str = self._access_token(self._index - 1)  # previous (-1) token
        if t1 is not None:
            # There is a "h", "m", or "s" preceding this token. Since neither
            # of the previous cases was hit, there is no label following this
            # token, so we use the previous label.
            # e.g. the "04" in "12h04"
            if (hms := self._token_to_hms(t1)) != -1:
                # looking backwards, hms increment one.
                self._set_hms(token, t_num, hms + 1)
                return True

            # If we are looking at the final token, we allow for a
            # backward-looking check to skip over a space.
            t2: str = self._access_token(self._index - 2)  # previous (-2) token
            if t2 is not None and t1 == " " and (hms := self._token_to_hms(t2)) != -1:
                # looking backwards, hms increment one.
                self._set_hms(token, t_num, hms + 1)
                return True

        # Not HMS token
        return False

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_token_ampm(self, token: str) -> cython.bint:
        """(cfunc) Parse the 'am/pm' token. Returns
        True if the token represents am/pm value and
        processed successfully, else False `<'bool'>`."""
        # AM/PM token already set
        if self._result.ampm != -1:
            return False  # exit

        # Missing hour value
        hour: cython.int = self._result.hour
        if hour == -1:
            return False  # exit

        # Validate AM/PM token
        ampm: cython.int = self._token_to_ampm(token)
        if ampm == -1:
            return False  # exit: not an ampm token

        # If AM/PM is found, but hour is not a 12 hour clock
        if not 0 <= hour <= 12:
            return False  # exit

        # Adjust hour to the correct AM/PM
        self._result.hour = self._adjust_ampm(hour, ampm)
        self._result.ampm = ampm
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_token_tzname(self, token: str) -> cython.bint:
        """(cfunc) Parse the 'tzname' token. Returns
        True if the token represents timezone name and
        processed successfully, else False `<'bool'>`."""
        # Validate if is timezome name
        if (
            self._ignoretz  # ignore timezone
            or self._result.hour == -1  # hour not set yet
            or self._result.tzoffset != -100_000  # tzoffset already set
            or self._result.tzname is not None  # tzname already set
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
        t1: str = self._access_token1()
        if t1 is not None:
            if t1 == "+":
                list_setitem(self._tokens, self._index + 1, "-")
            elif t1 == "-":
                list_setitem(self._tokens, self._index + 1, "+")
            else:
                return True  # exit
            # Reset tzoffset
            self._result.tzoffset = -100_000  # reset to default (None)
        return True  # exit

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_token_tzoffset(self, token: str) -> cython.bint:
        """(cfunc) Parse the 'tzoffset' token. Returns
        True if the token represents timezone offset and
        processed successfully, else False `<'bool'>`."""
        # Validate if is timezome offset
        if (
            self._ignoretz  # ignore timezone
            or self._result.hour == -1  # hour not set yet
            or self._result.tzoffset != -100_000  # offset already set
        ):
            return False

        # Validate offset identifier
        if token == "+":
            sign: cython.int = 1
        elif token == "-":
            sign: cython.int = -1
        else:
            return False  # exit: not offset

        # Validate next token
        t1: str = self._access_token1()
        if t1 is None:
            return False  # exit: not offset
        try:
            offset: cython.int = int(t1)
        except ValueError:
            return False  # exit: not offset

        # Calculate offset
        t_len: cython.Py_ssize_t = str_len(t1)
        if t_len == 4:
            # . -0300 => sign: "-" | t1: "0300"
            hour: cython.int = token_to_int(t1[0:2])  # type: ignore
            min: cython.int = token_to_int(t1[2:4])  # type: ignore
            offset = sign * (hour * 3_600 + min * 60)
        elif self._access_token2() == ":" and (t3 := self._access_token3()) is not None:
            # . -03:00 => sign: "-" | t1: "03" | t2: ":" | t3: "00"
            try:
                min: cython.int = int(t3)
                offset = sign * (offset * 3_600 + min * 60)
                self._index += 2  # skip token t2 & t3
            except ValueError:
                # -[0]3 => sign: "-" | t1: "[0]3"
                offset = sign * (offset * 3_600) if t_len <= 2 else -100_000
        else:
            # . -[0]3 => sign: "-" | t1: "[0]3"
            offset = sign * (offset * 3_600) if t_len <= 2 else -100_000

        # Validate offset
        if not -86_340 <= offset <= 86_340:
            raise errors.InvalidTokenError(
                "<'%s'>\nInvalid timezone offset token: '%s'."
                % (self.__class__.__name__, t1)
            )
        self._result.tzoffset = offset

        # Search for a timezone name between parenthesis
        if self._result.tzname is None:
            # No more tokens
            t2: str = self._access_token2()
            if t2 is None:
                pass
            # -0300(BRST) w/o space
            elif t2 == "(" and self._access_token4() == ")":
                t3: str = self._access_token3()
                if (is_tz := self._could_be_tzname(t3)) != 0:
                    self._result.tzname = "UTC" if is_tz == 1 else t3.upper()
                    self._index += 3  # skip token t2 - t4
            # -0300 (BRST) w/t space
            elif (
                self._is_token_jump(t2)
                and self._access_token3() == "("
                and self._access_token(self._index + 5) == ")"
            ):
                t4: str = self._access_token4()
                if (is_tz := self._could_be_tzname(t4)) != 0:
                    self._result.tzname = "UTC" if is_tz == 1 else t4.upper()
                    self._index += 4  # skip token t2 - t5

        # Finished
        self._index += 1  # skip token t1
        return True  # exit

    # Tokens utils -------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_hms(
        self,
        token: str,
        t_num: cython.double,
        hms: cython.int,
    ) -> cython.bint:
        """(cfunc) Set the values for HH:MM:SS.f `'<bool'>`."""
        if hms == 0:
            self._set_hm(t_num)
        elif hms == 1:
            self._set_ms(t_num)
        elif hms == 2:
            self._set_sf(token)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_hm(self, t_num: cython.double) -> cython.bint:
        """(cfunc) Set the values for 'hour' & 'minite' `<'bool'>`."""
        self._result.hour = int(t_num)
        if rem := t_num % 1:
            self._result.minute = int(rem * 60)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_ms(self, t_num: cython.double) -> cython.bint:
        """(cfunc) Set the values for 'minite' & 'second' `<'bool'>`."""
        self._result.minute = int(t_num)
        if rem := t_num % 1:
            self._result.second = int(rem * 60)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _set_sf(self, token: str) -> cython.bint:
        """(cfunc) Set the values for 'second' & 'microsecond (fraction)' `<'bool'>`."""
        # Find fraction seperater
        t_len: cython.Py_ssize_t = str_len(token)
        f_sep: cython.Py_ssize_t = str_findc(token, ".", 0, t_len, 1)
        # Contains microsecond
        if f_sep >= 0:
            self._result.second = token_to_int(token[0:f_sep])  # type: ignore
            self._result.microsecond = token_to_microsecond(token[f_sep + 1 : t_len], 0)  # type: ignore
        # Exclusive for second
        else:
            self._result.second = token_to_int(token)  # type: ignore
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _adjust_ampm(self, hour: cython.int, ampm: cython.int) -> cython.uint:
        """(cfunc) Adjust the 'hour' value according to the AM/PM flag `<'int'>`."""
        if hour < 12:
            if ampm == 1:
                hour += 12
            return max(0, hour)
        elif hour == 12 and ampm == 0:
            return 0
        else:
            return hour

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _could_be_tzname(self, token: str) -> cython.uint:
        """(cfunc) Check if a token could be a timezone
        name `<'int'>`. Returns 0 if not tzname, 1 if is UTC
        and 2 if the token is the timezone name.
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
        for ch in token:
            if not is_ascii_alpha_lower(ch):  # type: ignore
                return 0  # exit: not tzname
        return 2  # exit: possibily tzname

    # Tokens access ------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_tokens(self, dtstr: str, length: cython.Py_ssize_t) -> cython.bint:
        """(cfunc) Parse the Timelex tokens from the datetime string `<'bool'>`."""
        self._tokens = timelex(dtstr, length)
        self._token_count = list_len(self._tokens)
        self._index = 0
        return True

    @cython.cfunc
    @cython.inline(True)
    def _access_token(self, index: cython.Py_ssize_t) -> str:
        """(cfunc) Access the token by index `<'str'>`."""
        if 0 <= index < self._token_count:
            return cython.cast(str, list_getitem(self._tokens, index))
        else:
            return None

    @cython.cfunc
    @cython.inline(True)
    def _access_token1(self) -> str:
        """(cfunc) Access the next (+1) token `<'str'>`."""
        if self._token1 is None:
            self._token1 = self._access_token(self._index + 1)
        return self._token1

    @cython.cfunc
    @cython.inline(True)
    def _access_token2(self) -> str:
        """(cfunc) Access the next (+2) token `<'str'>`."""
        if self._token2 is None:
            self._token2 = self._access_token(self._index + 2)
        return self._token2

    @cython.cfunc
    @cython.inline(True)
    def _access_token3(self) -> str:
        """(cfunc) Access the next (+3) token `<'str'>`."""
        if self._token3 is None:
            self._token3 = self._access_token(self._index + 3)
        return self._token3

    @cython.cfunc
    @cython.inline(True)
    def _access_token4(self) -> str:
        """(cfunc) Access the next (+4) token `<'str'>`."""
        if self._token4 is None:
            self._token4 = self._access_token(self._index + 4)
        return self._token4

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _reset_tokens(self) -> cython.bint:
        """(cfunc) Reset all cached tokens."""
        self._token1 = None
        self._token2 = None
        self._token3 = None
        self._token4 = None
        return 1

    # Config -------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_token_pertain(self, token: object) -> cython.bint:
        """(cfunc) Check if the given token should be
        recognized as a pertain `<'bool'>`."""
        return set_contains(self._pertain, token)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_token_jump(self, token: object) -> cython.bint:
        """(cfunc) Check if the given token should be
        recognized as a jump word `<'bool'>`."""
        return set_contains(self._jump, token)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_token_utc(self, token: object) -> cython.bint:
        """(cfunc) Check if the given token should be
        recognized as an UTC timezone `<'bool'>`."""
        return set_contains(self._utc, token)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-2, check=False)
    def _token_to_month(self, token: object) -> cython.int:
        """(cfunc) Try to convert token to month `<'int'>`.
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
    @cython.exceptval(-2, check=False)
    def _token_to_weekday(self, token: object) -> cython.int:
        """(cfunc) Try to convert token to weekday `<'int'>`.
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
    @cython.exceptval(-2, check=False)
    def _token_to_hms(self, token: object) -> cython.int:
        """(cfunc) Try to convert token to hms `<'int'>`.
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
    @cython.exceptval(-2, check=False)
    def _token_to_ampm(self, token: object) -> cython.int:
        """(cfunc) Try to convert token to ampm `<'int'>`.
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
    @cython.exceptval(-100001, check=False)
    def _token_to_tzoffset(self, token: object) -> cython.int:
        """(cfunc) Try to convert token to tzoffset `<'int'>`.
        Returns the timezone offset in seconds if token matched
        with the 'tz' settings in Configs, else -100_000.
        """
        val = dict_getitem(self._tz, token)
        if val == cython.NULL:
            return -100_000
        else:
            return cython.cast(object, val)
