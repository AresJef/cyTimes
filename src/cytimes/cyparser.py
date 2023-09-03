# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from __future__ import annotations

# Cython imports
import cython
from cython.cimports.cpython.unicode import PyUnicode_Check as is_unicode  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_READ_CHAR as uni_loc  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_GET_LENGTH as str_len  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_Contains as str_contains  # type: ignore
from cython.cimports.cpython.unicode import Py_UNICODE_ISALPHA as uni_isalpha  # type: ignore
from cython.cimports.cpython.unicode import Py_UNICODE_ISDIGIT as uni_isdigit  # type: ignore
from cython.cimports.cpython.unicode import Py_UNICODE_ISSPACE as uni_isspace  # type: ignore
from cython.cimports.cpython.set import PySet_New as gen_set  # type: ignore
from cython.cimports.cpython.set import PySet_Add as set_add  # type: ignore
from cython.cimports.cpython.dict import PyDict_Copy as copy_dict  # type: ignore
from cython.cimports.cpython.dict import PyDict_SetItem as dict_set  # type: ignore
from cython.cimports.cpython.dict import PyDict_DelItem as dict_del  # type: ignore
from cython.cimports.cpython.list import PyList_Append as list_append  # type: ignore
from cython.cimports.cpython.list import PyList_GET_SIZE as list_len_noexc  # type: ignore
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cytimes import cytime, cymath  # type: ignore
from cython.cimports.cytimes import cydatetime as cydt  # type: ignore
from cython.cimports.cytimes.cytimedelta import cytimedelta  # type: ignore

np.import_array()
datetime.import_datetime()

# Python imports
import datetime, time, re
from typing import Union, Type
from dateutil.tz import tz as dateutil_tz
from dateutil.parser._parser import parserinfo
from cytimes.cytimedelta import cytimedelta
from cytimes import cymath, cydatetime as cydt


__all__ = ["ParserInfo", "Parser", "parse", "TimeLex"]


# Unicode -------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def uni_isdot(char: cython.int) -> cython.bint:
    return char == 46


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def uni_iscomma(char: cython.int) -> cython.bint:
    return char == 44


# TimeLex -------------------------------------------------------------------------------------
# # Fractional seconds are sometimes split by a comma
SPLIT_DECIMAL_RE: re.Pattern = re.compile("([.,])")


@cython.cclass
class TimeLex:
    _string: str
    _strlen: cython.int
    _idx: cython.int
    _charstack: list[str]
    _tokenstack: list[str]
    _ended: cython.bint

    def __init__(self, string: str) -> None:
        """Parse string into list of tokens

        :param string: `<str>` Unicode string.
        :raise `ValueError`: if string is not unicode formated.

        ### Example:
        >>> from cytimes import TimeLex
            timestr = "2023-08-08T23:59:59.000001"
            tokens = Timelex(timestr).split()
        """

        if not is_unicode(string):
            raise ValueError(
                "<TimeLex> Invalid 'string': %s %s. Only accepts unicode <str>."
                % (repr(string), type(string))
            )
        self._string = string
        self._strlen = str_len(self._string)
        self._idx = 0
        self._charstack = []
        self._tokenstack = []
        self._ended = False

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _get_nextchar(self) -> str:
        """Get the next char from string (Internal use only)."""

        if self._idx < self._strlen:
            nextchar = self._string[self._idx]
            self._idx += 1
            return nextchar
        else:
            return None

    @cython.cfunc
    @cython.inline(True)
    @cython.wraparound(True)
    def _get_token(self) -> str:
        """Parse the next token from string (Internal use only)."""

        # Token stack exist
        if self._tokenstack:
            return self._tokenstack.pop(0)

        # Parse token
        state: cython.int = 0
        seenchar: cython.bint = False
        token: str = None
        nextchar: str
        nextcuni: cython.int

        while not self._ended:
            # We only realize that we've reached the end of a token when we
            # find a character that's not part of the current token - since
            # that character may be part of the next token, it's stored in the
            # charstack.
            if self._charstack:
                nextchar = self._charstack.pop(0)
            # Get the nextchar from string
            else:
                nextchar = self._get_nextchar()
                while nextchar == "\x00":
                    nextchar = self._get_nextchar()
            # End of token
            if not nextchar:
                self._ended = True
                break

            # First character of the token - determines if we're starting
            # to parse a word, a number or something else.
            nextcuni = uni_loc(self._string, self._idx - 1)
            if state == 0:  # None
                token = nextchar
                if uni_isdigit(nextcuni):
                    state = 1  # "digit"
                elif uni_isalpha(nextcuni):
                    state = 2  # "alpha"
                elif uni_isspace(nextcuni):
                    token = " "
                    break  # emit token
                else:
                    break  # emit token

            # If we've already started reading a number, we keep reading
            # numbers until we find something that doesn't fit.
            elif state == 1:  # "digit"
                if uni_isdigit(nextcuni):
                    token += nextchar
                elif uni_isdot(nextcuni):
                    token += nextchar
                    state = 3  # "digit" w/t "."
                elif uni_iscomma(nextcuni) and str_len(token) >= 2:
                    token += nextchar
                    state = 3  # "digit" w/t "."
                else:
                    list_append(self._charstack, nextchar)
                    break  # emit token

            # If we've seen at least one dot separator, keep going, we'll
            # break up the tokens later.
            elif state == 3:  # "digit" w/t "."
                if uni_isdot(nextcuni) or uni_isdigit(nextcuni):
                    token += nextchar
                elif uni_isalpha(nextcuni) and token[-1] == ".":
                    token += nextchar
                    state = 4  # "aplha w/t "."
                else:
                    list_append(self._charstack, nextchar)
                    break  # emit token

            # If we've already started reading a word, we keep reading
            # letters until we find something that's not part of a word.
            elif state == 2:  # "alpha"
                seenchar = True
                if uni_isalpha(nextcuni):
                    token += nextchar
                elif uni_isdot(nextcuni):
                    token += nextchar
                    state = 4  # "aplha w/t "."
                else:
                    list_append(self._charstack, nextchar)
                    break  # emit token

            # If we've seen some letters and a dot separator, continue
            # parsing, and the tokens will be broken up later.
            elif state == 4:  # "aplha w/t "."
                seenchar = True
                if uni_isdot(nextcuni) or uni_isalpha(nextcuni):
                    token += nextchar
                elif uni_isdigit(nextcuni) and token[-1] == ".":
                    token += nextchar
                    state = 3  # "digit" w/t "."
                else:
                    list_append(self._charstack, nextchar)
                    break  # emit token

        # Case: token with "."
        if state == 3 or state == 4:
            # Stack the token
            if seenchar or token.count(".") > 1 or str_contains(".,", token[-1]):
                tokenlst: list = SPLIT_DECIMAL_RE.split(token)
                tokencnt: cython.int = list_len_noexc(tokenlst)
                token: str = tokenlst[0]
                tok: str
                for tok in tokenlst[1:tokencnt]:
                    if tok:
                        list_append(self._tokenstack, tok)

            # Alpha token with only ","
            if state == 3 and not str_contains(token, "."):
                token = token.replace(",", ".")

        return token

    @cython.ccall
    def split(self) -> list[str]:
        """Split the string into list of tokens."""
        return list(self)

    def __repr__(self) -> str:
        return "<TimeLex (string='%s')>" % self._string

    def __iter__(self) -> str:
        return self

    def __next__(self) -> str:
        token = self._get_token()
        if token is None:
            raise StopIteration
        return token


# YMD -----------------------------------------------------------------------------------------
@cython.cclass
class YMD:
    _century_specified: cython.bint
    _year: cython.int
    _month: cython.int
    _day: cython.int
    _validx: cython.int
    _val0: cython.int
    _val1: cython.int
    _val2: cython.int
    _yidx: cython.int
    _midx: cython.int
    _didx: cython.int

    def __init__(self) -> None:
        """Store time integers, and try be best to resolve which is year, month & day."""

        # Values
        self._century_specified = False
        self._year = -1
        self._month = -1
        self._day = -1
        # Index
        self._validx = -1
        self._val0 = -1
        self._val1 = -1
        self._val2 = -1
        self._yidx = -1
        self._midx = -1
        self._didx = -1

    # Values
    @property
    def year(self) -> int:
        """Year of the date. Always return -1 before calling `resolve()`."""
        return self._year

    @property
    def month(self) -> int:
        """Month of the date. Always return -1 before calling `resolve()`."""
        return self._month

    @property
    def day(self) -> int:
        """Day of the date. Always return -1 before calling `resolve()`."""
        return self._day

    # Resolve
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _get(self, idx: cython.int) -> cython.int:
        """Get store value (Internal use only).

        :param idx: `<int>` The index for the value, for index not between 0 and 2, -1 will be returned.
        :return: `<int>` The value corresponds to the index.
        """

        if idx == 0:
            return self._val0
        elif idx == 1:
            return self._val1
        elif idx == 2:
            return self._val2
        else:
            return -1

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _set(self, val: cython.int):
        """Set (store) a time interger value (Internal use only).

        :param val: `<int>` The value to be stored.
        """

        self._validx += 1
        if self._validx == 0:
            self._val0 = val
        elif self._validx == 1:
            self._val1 = val
        elif self._validx == 2:
            self._val2 = val
        else:
            self._validx -= 1

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _solved_values(self) -> cython.int:
        """Get the how many values have been solved (Internal use only).

        For exmaple: solved_values starts from 0. if one of the values has been
        labeled as `year`, solved_values += 1. If all values are labeled,
        solved_values will be 3.

        :return: `<int>` The number of values that have been solved (labeled).
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
    def could_be_day(self, value: cython.int) -> cython.bint:
        """Determine if a time intger could be the day of a date.

        :param value: `<int>` A time integer.
        :return: `<bool>` Whether the intger has the possibility to be the day of date.
        """

        if self._didx != -1:
            return False
        elif self._midx == -1:
            return 1 <= value <= 31
        elif self._yidx == -1:
            month = self._get(self._midx)
            return 1 <= value <= cydt.days_in_month(2000, month)
        else:
            month = self._get(self._midx)
            year = self._get(self._yidx)
            return 1 <= value <= cydt.days_in_month(year, month)

    @cython.cfunc
    def append(self, value: object, label: cython.int = 0):
        """Append (store) a time integer

        :param value: `<int> or <str>` A time info object.
        :param label: `<int>` Set the label of the time integer, where 1 as `Year`, 2 as `Month` and 3 as `Day`. 0 means undetermined.
        """

        # Adjustment for large value
        val: cython.int = int(value)
        if isinstance(value, str) and len(value) > 2 and value.isdigit():
            self._century_specified = True
            label = 1  # label as year
        elif val >= 100:
            self._century_specified = True
            label = 1  # label as year

        # Set (Store) value
        self._set(val)

        if label == 2 and self._midx == -1:  # label as month
            self._midx = self._validx
        elif label == 3 and self._didx == -1:  # label as day
            self._didx = self._validx
        elif label == 1 and self._yidx == -1:  # label as year
            self._yidx = self._validx

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def resolve(self, dayfirst: cython.bint, yearfirst: cython.bint):
        """Try the best to sort out which value is year,
        month & day base on the giving integer.

        :param dayfirst: `<bool>` Whether the date is dayfirst.
        :param yearfirst: `<bool>` Whether the date is yearfirst.

        After resolve, the corresponding property `year`, `month` & `day`
        of YMD will be updated.
        """

        len_ymd: cython.int = self._validx + 1
        solved: cython.int = self._solved_values()
        year: cython.int = -1
        month: cython.int = -1
        day: cython.int = -1

        # All resolved
        if len_ymd == solved > 0:
            year = self._get(self._yidx)
            month = self._get(self._midx)
            day = self._get(self._didx)

        # One member
        elif len_ymd == 1:
            if self._midx != -1:
                month = self._val0
            elif self._val0 > 31:
                year = self._val0
            else:
                day = self._val0

        # Two members
        elif len_ymd == 2:
            # With month label
            if self._midx != -1:
                if self._midx == 0:
                    month = self._val0
                    if self._val1 > 31:
                        year = self._val1
                    else:
                        day = self._val1
                else:
                    month = self._val1
                    if self._val0 > 31:
                        year = self._val0
                    else:
                        day = self._val0
            # Without month label
            elif self._val0 > 31:
                # 99-Jan
                year, month = self._val0, self._val1
            elif self._val1 > 31:
                # Jan-99
                month, year = self._val0, self._val1
            elif dayfirst and 0 < self._val1 <= 12:
                # 01-Jan
                day, month = self._val0, self._val1
            else:
                # Jan-01
                month, day = self._val0, self._val1

        # Three members
        elif len_ymd == 3:
            # Missing only one label
            if solved == 2:
                # Start with year
                if self._yidx != -1:
                    year = self._get(self._yidx)
                    if self._midx != -1:
                        month = self._get(self._midx)
                        day = self._get(3 - self._yidx - self._midx)
                    else:
                        month = self._get(3 - self._yidx - self._didx)
                        day = self._get(self._didx)
                # Start with month
                elif self._midx != -1:
                    month = self._get(self._midx)
                    if self._yidx != -1:
                        year = self._get(self._yidx)
                        day = self._get(3 - self._yidx - self._midx)
                    else:
                        year = self._get(3 - self._midx - self._didx)
                        day = self._get(self._didx)
                # Start with day
                else:
                    day = self._get(self._didx)
                    if self._yidx != -1:
                        year = self._get(self._yidx)
                        month = self._get(3 - self._yidx - self._didx)
                    else:
                        year = self._get(3 - self._midx - self._didx)
                        month = self._get(self._midx)
            # Missing more than one label
            elif self._midx == 0:
                if self._val1 > 31:
                    # Apr-2003-25
                    month, year, day = self._val0, self._val1, self._val2
                else:
                    month, day, year = self._val0, self._val1, self._val2
            elif self._midx == 1:
                if self._val0 > 31 or (yearfirst and 0 < self._val2 <= 31):
                    # 99-Jan-01
                    year, month, day = self._val0, self._val1, self._val2
                else:
                    # 01-Jan-99
                    day, month, year = self._val0, self._val1, self._val2
            elif self._midx == 2:
                # WTF!?
                if self._val1 > 31:
                    # 01-99-Jan
                    day, year, month = self._val0, self._val1, self._val2
                else:
                    # 99-01-Jan
                    year, day, month = self._val0, self._val1, self._val2
            else:
                if (
                    self._val0 > 31
                    or self._yidx == 0
                    or (yearfirst and 0 < self._val1 <= 12 and 0 < self._val2 <= 31)
                ):
                    if dayfirst and 0 < self._val2 <= 12:
                        # 99-01-Jan
                        year, day, month = self._val0, self._val1, self._val2
                    else:
                        # 99-Jan-01
                        year, month, day = self._val0, self._val1, self._val2
                elif self._val0 > 12 or (dayfirst and 0 < self._val1 <= 12):
                    # 01-Jan-99
                    day, month, year = self._val0, self._val1, self._val2
                else:
                    # Jan-01-99
                    month, day, year = self._val0, self._val1, self._val2

        # Final adjustment
        if month > 12 and 0 < day <= 12:
            month, day = day, month

        # Assign values
        self._year = year
        self._month = month
        self._day = day

    def __repr__(self) -> str:
        return "<YMD (year=%d, month=%d, day=%d, values=[%d, %d, %d])>" % (
            self._year,
            self._month,
            self._day,
            self._val0,
            self._val1,
            self._val2,
        )

    def __len__(self) -> cython.int:
        return self._validx + 1

    def __bool__(self) -> cython.bint:
        return self._validx > -1


# Result --------------------------------------------------------------------------------------
@cython.cclass
class Result:
    _year: cython.int
    _month: cython.int
    _day: cython.int
    _weekday: cython.int
    _hour: cython.int
    _minute: cython.int
    _second: cython.int
    _microsecond: cython.int
    _ampm: cython.int
    _tz_name: str
    _tzoffset: cython.int
    _century_specified: cython.bint

    def __init__(self) -> None:
        self._year = -1
        self._month = -1
        self._day = -1
        self._weekday = -1
        self._hour = -1
        self._minute = -1
        self._second = -1
        self._microsecond = -1
        self._ampm = -1
        self._tz_name = None
        self._tzoffset = -1000000
        self._century_specified = False

    # Info
    @property
    def year(self) -> int:
        """Parsed result for year, -1 means `None`."""
        return self._year

    @property
    def month(self) -> int:
        """Parsed result for month, -1 means `None`."""
        return self._month

    @property
    def day(self) -> int:
        """Parsed result for day, -1 means `None`."""
        return self._day

    # Weekday
    @property
    def weekday(self) -> int:
        """Parsed result for weekday, -1 means `None`."""
        return self._weekday

    # Hour
    @property
    def hour(self) -> int:
        """Parsed result for hour, -1 means `None`."""
        return self._hour

    # Minute
    @property
    def minute(self) -> int:
        """Parsed result for minute, -1 means `None`."""
        return self._minute

    # Second
    @property
    def second(self) -> int:
        """Parsed result for second, -1 means `None`."""
        return self._second

    # Microsecond
    @property
    def microsecond(self) -> int:
        """Parsed result for microsecond, -1 means `None`."""
        return self._microsecond

    # AM/PM
    @property
    def ampm(self) -> int:
        """Parsed result for AM/PM. 0 mean AM, 1 mean PM, -1 means `None`."""
        return self._ampm

    # Timezone name
    @property
    def tzname(self) -> str:
        """Parsed result for timezone name."""
        return self._tz_name

    # Timezone offset
    @property
    def tzoffset(self) -> int:
        """Parsed result for timezone offset in seconds. -1000000 means `None`."""
        return self._tzoffset

    # Country specified
    @property
    def century_specified(self) -> bool:
        """Whether parsed result is century specified."""
        return self._century_specified

    # Special methods
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _represent(self) -> str:
        reprs: list[str] = []
        if self._year != -1:
            reprs.append("year=%d" % self._year)
        if self._month != -1:
            reprs.append("month=%d" % self._month)
        if self._day != -1:
            reprs.append("day=%d" % self._day)
        if self._weekday != -1:
            reprs.append("weekday=%d" % self._weekday)
        if self._hour != -1:
            reprs.append("hour=%d" % self._hour)
        if self._minute != -1:
            reprs.append("minute=%d" % self._minute)
        if self._second != -1:
            reprs.append("second=%d" % self._second)
        if self._microsecond != -1:
            reprs.append("microsecond=%d" % self._microsecond)
        if self._ampm != -1:
            reprs.append("ampm=%d" % self._ampm)
        if self._tz_name:
            reprs.append("tzname='%s'" % self._tz_name)
        if self._tzoffset != -1000000:
            reprs.append("tzoffset=%d" % self._tzoffset)
        return ", ".join(reprs)

    def __repr__(self) -> str:
        return "<Result (%s)>" % self._represent()

    def __bool__(self) -> cython.bint:
        return (
            self._year != -1
            or self._month != -1
            or self._day != -1
            or self._weekday != -1
            or self._hour != -1
            or self._minute != -1
            or self._second != -1
            or self._microsecond != -1
        )


# ParserInfo (config) -------------------------------------------------------------------------
# fmt: off
DEFAULT_INFO_JUMP: set[str] = {
    " ", ".", ",", ";", "-", "/", "'",
    "at", "on", "and", "ad", "m", "t", "of",
    "st", "nd", "rd", "th", "年" ,"月", "日"}
DEFAULT_INFO_WEEKDAY: dict[str, int] = {
    "mon": 0, "monday": 0,    "星期一": 0, "周一": 0,
    "tue": 1, "tuesday": 1,   "星期二": 1, "周二": 1,
    "wed": 2, "wednesday": 2, "星期三": 2, "周三": 2,
    "thu": 3, "thursday": 3,  "星期四": 3, "周四": 3,
    "fri": 4, "friday": 4,    "星期五": 4, "周五": 4,
    "sat": 5, "saturday": 5,  "星期六": 5, "周六": 5,
    "sun": 6, "sunday": 6,    "星期日": 6, "周日": 6}
DEFAULT_INFO_MONTH: dict[str, int] = {
    "jan": 1, "january": 1, "januar": 1, "janvier": 1, "gennaio": 1, "enero": 1, "一月": 1,
    "feb": 2, "february": 2, "febr": 2, "februar": 2, "février": 2, "febbraio": 2, "febrero": 2, "二月": 2,
    "mar": 3, "march": 3, "märz": 3, "mars": 3, "marzo": 3, "三月": 3,
    "apr": 4, "april": 4, "avril": 4, "aprile": 4, "abril": 4, "四月": 4,
    "may": 5, "mai": 5, "maggio": 5, "mayo": 5, "五月": 5,
    "jun": 6, "june": 6, "juni": 6, "juin": 6, "giugno": 6, "junio": 6, "六月": 6,
    "jul": 7, "july": 7, "juli": 7, "juillet": 7, "luglio": 7, "julio": 7, "七月": 7,
    "aug": 8, "august": 8, "août": 8, "agosto": 8, "八月": 8,
    "sep": 9, "september": 9, "sept": 9, "septembre": 9, "settembre": 9, "septiembre": 9, "九月": 9,
    "oct": 10, "october": 10, "oktober": 10, "octobre": 10, "ottobre": 10, "octubre": 10, "十月": 10,
    "nov": 11, "november": 11, "novembre": 11, "novembre": 11, "noviembre": 11, "十一月": 11,
    "dec": 12, "december": 12, "dezember": 12, "décembre": 12, "dicembre": 12, "diciembre": 12, "十二月": 12}
DEFAULT_INFO_HMS: dict[str, int] = {
    "h": 0, "hour": 0, "hours": 0, "小时": 0, "时": 0,
    "m": 1, "minute": 1, "minutes": 1, "分钟": 1, "分": 1,
    "s": 2, "second": 2, "seconds": 2, "秒": 2}
DEFAULT_INFO_AMPM: dict[str, int] = {
    "am": 0, "a": 0, "上午": 0,
    "pm": 1, "p": 1, "下午": 1}
DEFAULT_INFO_UTCTIMEZONE: set[str] = {"utc", "gmt", "z"}
DEFAULT_INFO_TZOFFSET: dict[str, int] = {}
DEFAULT_INFO_PERTAIN: set[str] = {"of"}
# fmt: on


@cython.cclass
class ParserInfo:
    _jump: set[str]
    _weekday: dict[str, int]
    _month: dict[str, int]
    _hms: dict[str, int]
    _ampm: dict[str, int]
    _utczone: set[str]
    _tzoffset: dict[str, int]
    _pertain: set[str]
    _dayfirst: cython.bint
    _yearfirst: cython.bint

    def __init__(
        self,
        dayfirst: cython.bint = False,
        yearfirst: cython.bint = False,
    ):
        """The infomation that affect how `<Parser>` parse time.

        :param dayfirst: `<bool>` Whether to interpret the first value in an ambiguous date (e.g. 01/05/09) as the day (`True`) or month (`False`).
            When `yearfirst` is `True`, this distinguishes between Y/D/M and Y/M/D. This settings
            only takes affect when the `parse()` method set it's `dayfirst` to `None`.
        :param yearfirst: `<bool>` Whether to interpret the first value in an ambiguous date (e.g. 01/05/09) as the year.
            If `True`, the first number is taken as year, otherwise the last number. This settings
            only takes affect when the `parse()` method set it's `yearfirst` to `None`.

        ### Configuration
        The following settings can be modified through `set_xxx()` or `add_xxx()` method.
        If there is an existing `dateutil.parserinfo` you want to use, call instance method
        `from_parserinfo()` to import settings from a `<dateutil.parserinfo>` instance.

        - jump: Words should be jump (skipped) when parsing, e.g: `'and'`, `'at'`, `'on'`.
        - weekday: Words should recognized as weekday when parsing, e.g: `'monday'`, `'tuesday'`.
        - month: Words should recognized as month when parsing, e.g: `'january'`, `'february'`.
        - hms: Words should recognized as HH/MM/SS when parsing, e.g: `'hour'`, `'minute'`.
        - ampm: Words should recognized as AM/PM when parsing, e.g: `'am'`, `'pm'`.
        - utczone: Words should be recognized as UTC timezone when parsing, e.g: `'utc'`, `'gmt'`.
        - tzoffset: Words should be recognized as timezone offset when parsing, e.g: `'PST'`.
        - pertain: Words should be recognized as pertain when parsing, e.g: `'of'`.
        """

        self._jump = gen_set(DEFAULT_INFO_JUMP)
        self._weekday = copy_dict(DEFAULT_INFO_WEEKDAY)
        self._month = copy_dict(DEFAULT_INFO_MONTH)
        self._hms = copy_dict(DEFAULT_INFO_HMS)
        self._ampm = copy_dict(DEFAULT_INFO_AMPM)
        self._utczone = gen_set(DEFAULT_INFO_UTCTIMEZONE)
        self._tzoffset = copy_dict(DEFAULT_INFO_TZOFFSET)
        self._pertain = gen_set(DEFAULT_INFO_PERTAIN)
        self._dayfirst = dayfirst
        self._yearfirst = yearfirst

    # Day & Year first ------------------------------------------------------------
    @property
    def dayfirst(self) -> bool:
        """The default dayfirst behavior for a `<Parser>` when loading this info."""
        return self._dayfirst

    @property
    def yearfirst(self) -> bool:
        """The default yearfirst behavior for a `<Parser>` when loading this info."""
        return self._yearfirst

    # Jump ------------------------------------------------------------------------
    def add_jump(self, *values: str):
        """Add words that should be `jump (skipped)` when parsing.
        Calling this method will add to the existing jump words.
        """
        self._add_to_set(self._jump, values)

    def set_jump(self, *values: str):
        """Set words that should be `jump (skipped)` when parsing.
        Calling this method will clear the existing jump words.
        """
        self._jump = gen_set(self._validate_str(values))

    @cython.ccall
    def jump(self, word: str) -> cython.bint:
        """Determine whether the giving `word` should be jumped
        when parsing. return boolean."""
        return word.lower() in self._jump

    @property
    def jump_words(self) -> list[str]:
        """Get the current Jump words in a list."""
        return sorted(self._jump)

    # Weekday ---------------------------------------------------------------------
    def add_monday(self, *values: str):
        """Add words that should be recognized as `Monday` when parsing.
        Calling this method will add to the existing `Monday` words.
        """
        self._add_to_dict(self._weekday, 0, values)

    def set_monday(self, *values: str):
        """Set words that should be recognized as `Monday` when parsing.
        Calling this method will clear the existing `Monday` words.
        """
        self._set_to_dict(self._weekday, 0, values)

    def add_tuesday(self, *values: str):
        """Add words that should be recognized as `Tuesday` when parsing.
        Calling this method will add to the existing `Tuesday` words.
        """
        self._add_to_dict(self._weekday, 1, values)

    def set_tuesday(self, *values: str):
        """Set words that should be recognized as `Tuesday` when parsing.
        Calling this method will clear the existing `Tuesday` words.
        """
        self._set_to_dict(self._weekday, 1, values)

    def add_wednesday(self, *values: str):
        """Add words that should be recognized as `Wednesday` when parsing.
        Calling this method will add to the existing `Wednesday` words.
        """
        self._add_to_dict(self._weekday, 2, values)

    def set_wednesday(self, *values: str):
        """Set words that should be recognized as `Wednesday` when parsing.
        Calling this method will clear the existing `Wednesday` words.
        """
        self._set_to_dict(self._weekday, 2, values)

    def add_thursday(self, *values: str):
        """Add words that should be recognized as `Thursday` when parsing.
        Calling this method will add to the existing `Thursday` words.
        """
        self._add_to_dict(self._weekday, 3, values)

    def set_thursday(self, *values: str):
        """Set words that should be recognized as `Thursday` when parsing.
        Calling this method will clear the existing `Thursday` words.
        """
        self._set_to_dict(self._weekday, 3, values)

    def add_friday(self, *values: str):
        """Add words that should be recognized as `Friday` when parsing.
        Calling this method will add to the existing `Friday` words.
        """
        self._add_to_dict(self._weekday, 4, values)

    def set_friday(self, *values: str):
        """Set words that should be recognized as `Friday` when parsing.
        Calling this method will clear the existing `Friday` words.
        """
        self._set_to_dict(self._weekday, 4, values)

    def add_saturday(self, *values: str):
        """Add words that should be recognized as `Saturday` when parsing.
        Calling this method will add to the existing `Saturday` words.
        """
        self._add_to_dict(self._weekday, 5, values)

    def set_saturday(self, *values: str):
        """Set words that should be recognized as `Saturday` when parsing.
        Calling this method will clear the existing `Saturday` words.
        """
        self._set_to_dict(self._weekday, 5, values)

    def add_sunday(self, *values: str):
        """Add words that should be recognized as `Sunday` when parsing.
        Calling this method will add to the existing `Sunday` words.
        """
        self._add_to_dict(self._weekday, 6, values)

    def set_sunday(self, *values: str):
        """Set words that should be recognized as `Sunday` when parsing.
        Calling this method will clear the existing `Sunday` words.
        """
        self._set_to_dict(self._weekday, 6, values)

    @cython.ccall
    def weekday(self, word: str) -> cython.int:
        """If the given 'word' can be recognized as a weekday, the weekday interger
        will be returned (0 Monday - 6 Sunday). Otherwise, -1 will be returned.
        """
        return self._weekday.get(word.lower(), -1)

    @property
    def weekday_words(self) -> dict[int, list[str]]:
        """Get the current weekday words in a dict, where key is the weekday interger
        (0 Monday - 6 Sunday), and value is a list of corresponding weekday words.
        """

        res: dict[int, list[str]] = {}
        for k, v in self._weekday.items():
            res.setdefault(v, []).append(k)
        return res

    # Month -----------------------------------------------------------------------
    def add_jan(self, *values: str):
        """Add words that should be recognized as `January` when parsing.
        Calling this method will add to the existing `January` words.
        """
        self._add_to_dict(self._month, 1, values)

    def set_jan(self, *values: str):
        """Set words that should be recognized as `January` when parsing.
        Calling this method will clear the existing `January` words.
        """
        self._set_to_dict(self._month, 1, values)

    def add_feb(self, *values: str):
        """Add words that should be recognized as `February` when parsing.
        Calling this method will add to the existing `February` words.
        """
        self._add_to_dict(self._month, 2, values)

    def set_feb(self, *values: str):
        """Set words that should be recognized as `February` when parsing.
        Calling this method will clear the existing `February` words.
        """
        self._set_to_dict(self._month, 2, values)

    def add_mar(self, *values: str):
        """Add words that should be recognized as `March` when parsing.
        Calling this method will add to the existing `March` words.
        """
        self._add_to_dict(self._month, 3, values)

    def set_mar(self, *values: str):
        """Set words that should be recognized as `March` when parsing.
        Calling this method will clear the existing `March` words.
        """
        self._set_to_dict(self._month, 3, values)

    def add_apr(self, *values: str):
        """Add words that should be recognized as `April` when parsing.
        Calling this method will add to the existing `April` words.
        """
        self._add_to_dict(self._month, 4, values)

    def set_apr(self, *values: str):
        """Set words that should be recognized as `April` when parsing.
        Calling this method will clear the existing `April` words.
        """
        self._set_to_dict(self._month, 4, values)

    def add_may(self, *values: str):
        """Add words that should be recognized as `May` when parsing.
        Calling this method will add to the existing `May` words.
        """
        self._add_to_dict(self._month, 5, values)

    def set_may(self, *values: str):
        """Set words that should be recognized as `May` when parsing.
        Calling this method will clear the existing `May` words.
        """
        self._set_to_dict(self._month, 5, values)

    def add_jun(self, *values: str):
        """Add words that should be recognized as `June` when parsing.
        Calling this method will add to the existing `June` words.
        """
        self._add_to_dict(self._month, 6, values)

    def set_jun(self, *values: str):
        """Set words that should be recognized as `June` when parsing.
        Calling this method will clear the existing `June` words.
        """
        self._set_to_dict(self._month, 6, values)

    def add_jul(self, *values: str):
        """Add words that should be recognized as `July` when parsing.
        Calling this method will add to the existing `July` words.
        """
        self._add_to_dict(self._month, 7, values)

    def set_jul(self, *values: str):
        """Set words that should be recognized as `July` when parsing.
        Calling this method will clear the existing `July` words.
        """
        self._set_to_dict(self._month, 7, values)

    def add_aug(self, *values: str):
        """Add words that should be recognized as `August` when parsing.
        Calling this method will add to the existing `August` words.
        """
        self._add_to_dict(self._month, 8, values)

    def set_aug(self, *values: str):
        """Set words that should be recognized as `August` when parsing.
        Calling this method will clear the existing `August` words.
        """
        self._set_to_dict(self._month, 8, values)

    def add_sep(self, *values: str):
        """Add words that should be recognized as `September` when parsing.
        Calling this method will add to the existing `September` words.
        """
        self._add_to_dict(self._month, 9, values)

    def set_sep(self, *values: str):
        """Set words that should be recognized as `September` when parsing.
        Calling this method will clear the existing `September` words.
        """
        self._set_to_dict(self._month, 9, values)

    def add_oct(self, *values: str):
        """Add words that should be recognized as `October` when parsing.
        Calling this method will add to the existing `October` words.
        """
        self._add_to_dict(self._month, 10, values)

    def set_oct(self, *values: str):
        """Set words that should be recognized as `October` when parsing.
        Calling this method will clear the existing `October` words.
        """
        self._set_to_dict(self._month, 10, values)

    def add_nov(self, *values: str):
        """Add words that should be recognized as `November` when parsing.
        Calling this method will add to the existing `November` words.
        """
        self._add_to_dict(self._month, 11, values)

    def set_nov(self, *values: str):
        """Set words that should be recognized as `November` when parsing.
        Calling this method will clear the existing `November` words.
        """
        self._set_to_dict(self._month, 11, values)

    def add_dec(self, *values: str):
        """Add words that should be recognized as `December` when parsing.
        Calling this method will add to the existing `December` words.
        """
        self._add_to_dict(self._month, 12, values)

    def set_dec(self, *values: str):
        """Set words that should be recognized as `December` when parsing.
        Calling this method will clear the existing `December` words.
        """
        self._set_to_dict(self._month, 12, values)

    @cython.ccall
    def month(self, word: str) -> cython.int:
        """If the given 'word' can be recognized as a month, the month interger
        will be returned (1 Jan - 12 Dec). Otherwise, -1 will be returned.
        """
        return self._month.get(word.lower(), -1)

    @property
    def month_words(self) -> dict[int, list[str]]:
        """Get the current month words in a dict, where key is the month interger
        (1 Jan - 12 Dec), and value is a list of corresponding month words.
        """

        res: dict[int, list[str]] = {}
        for k, v in self._month.items():
            res.setdefault(v, []).append(k)
        return res

    # HMS -------------------------------------------------------------------------
    def add_hour(self, *values: str):
        """Add words that should be recognized as `Hour` when parsing.
        Calling this method will add to the existing `Hour` words.
        """
        self._add_to_dict(self._hms, 0, values)

    def set_hour(self, *values: str):
        """Set words that should be recognized as `Hour` when parsing.
        Calling this method will clear the existing `Hour` words.
        """
        self._set_to_dict(self._hms, 0, values)

    def add_minute(self, *values: str):
        """Add words that should be recognized as `Minute` when parsing.
        Calling this method will add to the existing `Minute` words.
        """
        self._add_to_dict(self._hms, 1, values)

    def set_minute(self, *values: str):
        """Set words that should be recognized as `Minute` when parsing.
        Calling this method will clear the existing `Minute` words.
        """
        self._set_to_dict(self._hms, 1, values)

    def add_second(self, *values: str):
        """Add words that should be recognized as `Second` when parsing.
        Calling this method will add to the existing `Second` words.
        """
        self._add_to_dict(self._hms, 2, values)

    def set_second(self, *values: str):
        """Set words that should be recognized as `Second` when parsing.
        Calling this method will clear the existing `Second` words.
        """
        self._set_to_dict(self._hms, 2, values)

    @cython.ccall
    def hms(self, word: str) -> cython.int:
        """If the given 'word' can be recognized as a HH/MM/DD, the index interger
        will be returned (0 Hour - 2 Second). Otherwise, -1 will be returned.
        """
        return self._hms.get(word.lower(), -1)

    @property
    def hms_words(self) -> dict[int, list[str]]:
        """Get the current HH/MM/SS words in a dict, where key is the index interger
        (0 Hour - 2 Second), and value is a list of corresponding time words.
        """

        res: dict[int, list[str]] = {}
        for k, v in self._hms.items():
            res.setdefault(v, []).append(k)
        return res

    # AM/PM -----------------------------------------------------------------------
    def add_am(self, *values: str):
        """Add words that should be recognized as `AM` when parsing.
        Calling this method will add to the existing `AM` words.
        """
        self._add_to_dict(self._ampm, 0, values)

    def set_am(self, *values: str):
        """Set words that should be recognized as `AM` when parsing.
        Calling this method will clear the existing `AM` words.
        """
        self._set_to_dict(self._ampm, 0, values)

    def add_pm(self, *values: str):
        """Add words that should be recognized as `PM` when parsing.
        Calling this method will add to the existing `PM` words.
        """
        self._add_to_dict(self._ampm, 1, values)

    def set_pm(self, *values: str):
        """Set words that should be recognized as `PM` when parsing.
        Calling this method will clear the existing `PM` words.
        """
        self._set_to_dict(self._ampm, 1, values)

    @cython.ccall
    def ampm(self, word: str) -> cython.int:
        """If the given 'word' can be recognized as a AM/PM, the index interger
        will be returned (0 AM & 1 AM). Otherwise, -1 will be returned.
        """
        return self._ampm.get(word.lower(), -1)

    @property
    def ampm_words(self) -> dict[int, list[str]]:
        """Get the current AM/PM words in a dict, where key is the index interger
        (0 AM & 1 AM), and value is a list of corresponding time words.
        """

        res: dict[int, list[str]] = {}
        for k, v in self._ampm.items():
            res.setdefault(v, []).append(k)
        return res

    # UTC Timezone ----------------------------------------------------------------
    def add_utc(self, *values: str):
        """Add words that should be recognized as `UTC timezone` when parsing.
        Calling this method will add to the existing `UTC timezone` words.
        """
        self._add_to_set(self._utczone, values)

    def set_utc(self, *values: str):
        """Set words that should be recognized as `UTC timezone` when parsing.
        Calling this method will clear the existing `UTC timezone` words.
        """
        self._utczone = gen_set(self._validate_str(values))

    @cython.ccall
    def utczone(self, word: str) -> cython.bint:
        """Determine whether the given 'word' means UTC
        timezone. return boolean."""
        return word.lower() in self._utczone

    @property
    def utczone_words(self) -> list[str]:
        """Get the current UTC timezone words in a list."""

        return sorted(self._utczone)

    # Timezone offset -------------------------------------------------------------
    def add_tzoffset(self, tz: str, offset: cython.int):
        """Add a timezone name and it's corresponding offset in seconds,
        so the `<Parser>` can handle when encountered.

        :param tz: `<str>` The name of the timezone
        :param offset: `<int>` The corresponding offset in seconds.
        """

        tz = tz.lower()
        if tz not in self._utczone:
            self._tzoffset[tz] = offset

    @cython.ccall
    def tzoffset(self, tz: str) -> cython.int:
        """Check if the giving 'tz' matched with a timezone offset.
        If matched the corresponding offset in seconds will be returned.
        Otherwise, -1000000 will be returned (means None).
        """

        tz = tz.lower()
        if tz in self._utczone:
            return 0
        else:
            return self._tzoffset.get(tz, -1000000)

    @property
    def tzoffset_words(self) -> dict[str, int]:
        """Get the current timezone offset words in a dict, where key is
        the timezone name and value is the corresponding offset in seconds.
        """

        return copy_dict(self._tzoffset)

    # Pertain ---------------------------------------------------------------------
    def add_pertain(self, *values: str):
        """Add words that should be recognized as `Pertain` when parsing.
        Calling this method will add to the existing `Pertain` words.
        """
        self._add_to_set(self._pertain, values)

    def set_pertain(self, *values: str):
        """Set words that should be recognized as `Pertain` when parsing.
        Calling this method will clear the existing `Pertain` words.
        """
        self._pertain = gen_set(self._validate_str(values))

    @cython.ccall
    def pertain(self, word: str) -> cython.bint:
        """Determine whether the giving `word` should recognized
        as Pertain when parsing. return boolean."""
        return word.lower() in self._pertain

    @property
    def pertain_words(self) -> list[str]:
        """Get the current Pertain words in a list."""
        return sorted(self._pertain)

    # Utils -----------------------------------------------------------------------
    @cython.cfunc
    def _validate_str(self, values: tuple) -> list[str]:
        return [str(v).lower() for v in values if isinstance(v, str) and v]

    @cython.cfunc
    def _add_to_set(self, set_: set, values: tuple):
        for val in self._validate_str(values):
            set_add(set_, val)

    @cython.cfunc
    def _add_to_dict(self, dict_: dict, val: cython.int, keys: tuple):
        for key in self._validate_str(keys):
            dict_set(dict_, key, val)

    @cython.cfunc
    def _set_to_dict(self, dict_: dict, val: cython.int, keys: tuple):
        for key in [key for key, v in dict_.items() if v == val]:
            dict_del(dict_, key)
        self._add_to_dict(dict_, val, keys)

    # Conversion ------------------------------------------------------------------
    @cython.ccall
    def from_parserinfo(self, info: parserinfo):
        """Import settings from a giving `dateutil.parserinfo` instance."""

        try:
            # Jump
            self.set_jump(*info._jump.keys())
            # Weekday
            mon, tue, wed, thu, fri, sat, sun = [], [], [], [], [], [], []
            for key, val in info._weekdays.items():
                if val == 0:
                    mon.append(key)
                elif val == 1:
                    tue.append(key)
                elif val == 2:
                    wed.append(key)
                elif val == 3:
                    thu.append(key)
                elif val == 4:
                    fri.append(key)
                elif val == 5:
                    sat.append(key)
                elif val == 6:
                    sun.append(key)
            self.set_monday(*mon)
            self.set_tuesday(*tue)
            self.set_wednesday(*wed)
            self.set_thursday(*thu)
            self.set_friday(*fri)
            self.set_saturday(*sat)
            self.set_sunday(*sun)
            # Month
            jan, feb, mar, apr, may, jun = [], [], [], [], [], []
            jul, aug, sep, oct, nov, dec = [], [], [], [], [], []
            for key, val in info._months.items():
                if val == 0:
                    jan.append(key)
                elif val == 1:
                    feb.append(key)
                elif val == 2:
                    mar.append(key)
                elif val == 3:
                    apr.append(key)
                elif val == 4:
                    may.append(key)
                elif val == 5:
                    jun.append(key)
                elif val == 6:
                    jul.append(key)
                elif val == 7:
                    aug.append(key)
                elif val == 8:
                    sep.append(key)
                elif val == 9:
                    oct.append(key)
                elif val == 10:
                    nov.append(key)
                elif val == 11:
                    dec.append(key)
            self.set_jan(*jan)
            self.set_feb(*feb)
            self.set_mar(*mar)
            self.set_apr(*apr)
            self.set_may(*may)
            self.set_jun(*jun)
            self.set_jul(*jul)
            self.set_aug(*aug)
            self.set_sep(*sep)
            self.set_oct(*oct)
            self.set_nov(*nov)
            self.set_dec(*dec)
            # HMS
            hh, mm, ss = [], [], []
            for key, val in info._hms.items():
                if val == 0:
                    hh.append(key)
                elif val == 1:
                    mm.append(key)
                elif val == 2:
                    ss.append(key)
            self.set_hour(*hh)
            self.set_minute(*mm)
            self.set_second(*ss)
            # AM/PM
            am, pm = [], []
            for key, val in info._ampm.items():
                if val == 0:
                    am.append(key)
                elif val == 1:
                    pm.append(key)
            self.set_am(*am)
            self.set_pm(*pm)
            # UTC Timezone
            self.set_utc(*info._utczone.keys())
            # Timezone offset
            self._tzoffset.clear()
            for key, val in info.TZOFFSET.items():
                self.add_tzoffset(key, val)
            # Pertain
            self.set_pertain(*info._pertain.keys())
            # Day & Year first
            self._dayfirst = info.dayfirst
            self._yearfirst = info.yearfirst

        except Exception as err:
            raise ValueError(
                "<ParserInfo> Failed to import settings from `dateutil.parserinfo`: %s %s"
                % (type(info), repr(info))
            ) from err

    # Validate Result -------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _convert_year(
        self,
        year: cython.int,
        century_specified: cython.bint,
    ) -> cython.int:
        """Converts two-digit year to year within [-50, 49]
        range of the current local time.

        :param year: `<int>` The year of a date in two-digit.
        :param century_specified: `<bool>` Whether the year is century_specified.
        :return: `<int>` The year after conversion
        """

        if 0 <= year < 100 and not century_specified:
            # Determine current year and century
            curyear: cython.int = cytime.localtime().tm_year
            century: cython.int = curyear // 100 * 100

            # Adjust for current century
            year += century
            if year >= curyear + 50:  # if too far in future
                year -= 100
            elif year < curyear - 50:  # if too far in past
                year += 100

        return year

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _adjust_result(self, res: Result) -> Result:
        """Adjust the parse result from `<Parser>`."""

        if res._year != -1:
            res._year = self._convert_year(res._year, res._century_specified)

        if res._tzoffset == 0 and not res._tz_name:
            res._tz_name = "UTC"
        elif res._tz_name:
            if res._tz_name == "Z" or res._tz_name == "z":
                res._tz_name = "UTC"
                res._tzoffset = 0
            elif res._tzoffset != 0 and self.utczone(res._tz_name):
                res._tzoffset = 0

        return res

    # Special methods -------------------------------------------------------------
    def __repr__(self) -> str:
        return "<cytimes.ParserInfo>"


DEFAULT_PARSERINFO: ParserInfo = ParserInfo(False, False)


# Parser --------------------------------------------------------------------------------------
@cython.cclass
class Parser:
    __info: ParserInfo

    def __init__(self, info: Union[ParserInfo, parserinfo, None] = None) -> None:
        """The Parser for parsing date/time string.

        :param info: `<ParserInfo>` The parserinfo to use, accepts:
            - `<ParserInfo>`: A ParserInfo instance from `cytimes.ParserInfo`.
            - `<parserinfo>`: A parserinfo instance from `dateutil.parserinfo`.
            - `None`: The default ParserInfo will be used.

        ### Exmaple (Custom ParserInfo):
        >>> from cytimes import ParserInfo, Parser
            # Initiate ParserInfo & set as default behavior
            info = ParserInfo(dayfirst=True, yearfirst=True)
            # Add month words that should be recognized as `January`
            info.add_jan("janvier", "gennaio", "一月")
            # Initiate Parser with the ParserInfo
            parser = Parser(info)
            # Parse date/time string
            dt = parser.parse("2021-01 janvier 12:00:00")
            # result: <datetime.datetime> 2021-01-01 12:00:00

        ### Exmaple (dateutil.parserinfo):
        >>> from dateutil.parser import parserinfo, cytimes import Parser
            # Create custom dateutil.parserinfo.
            class MyParserInfo(parserinfo):
                MONTHS = [
                    ("Jan", "January", "janvier", "gennaio", "一月"), # Add month words
                    ...
                ]
            # Initiate parserinfo & set as default behavior
            info = MyParserInfo(dayfirst=True, yearfirst=True)
            # Initiate Parser with `dateutil.parserinfo`
            parser = Parser(info)
            # Parse date/time string
            dt = parser.parse("2021-01 janvier 12:00:00")
            # result: <datetime.datetime> 2021-01-01 12:00:00
        """

        if info is None:
            self.__info = DEFAULT_PARSERINFO
        elif isinstance(info, ParserInfo):
            self.__info = info
        else:
            self.__info = ParserInfo().from_parserinfo(info)

    # Parse --------------------------------------------------------------------------------
    def parse(
        self,
        timestr: str,
        default: Union[datetime.datetime, datetime.date, None] = None,
        dayfirst: cython.bint = None,
        yearfirst: cython.bint = None,
        ignoretz: cython.bint = False,
        tzinfos: Union[Type[dateutil_tz.tzfile], dict[str, int], None] = None,
        fuzzy: cython.bint = False,
    ) -> datetime.datetime:
        """Parse a string contains date/time stamp into `datetime.datetime`.

        :param timestr: `<str>` A string containing date/time stamp.
        :param default: `<datetime>` The default object, which will be used as the base to fillin missing time elements from parsed results.
            If set to `None` (default), the current local year/month/day will be used as the default base.
        :param dayfirst: `<bool>` Whether to interpret the first value in an ambiguous date (e.g. 01/05/09) as the day (`True`) or month (`False`).
            When `yearfirst` is `True`, this distinguishes between Y/D/M and Y/M/D. If set to `None`,
            the `dayfirst` settings in `ParserInfo` will be used (defaults to False).
        :param yearfirst: `<bool>` Whether to interpret the first value in an ambiguous date (e.g. 01/05/09) as the year.
            If `True`, the first number is taken as year, otherwise the last number. If set to `None`,
            the `yearfirst` settings in `ParserInfo` will be used (defaults to False).
        :param ignoretz: `<bool>` Whether to ignore timezone info and only return naive datetime.
        :param tzinfos: Additional timezone parsing argument. Applicable when `ignoretz` is False. Accepts:
            - `<dict[str, int | tzinfo]>`: A dictionary where timezone name as the key and timezone
              offset in seconds or `tzinfo` object as the value.
            - `<TzinfoFactory>`: A callable which takes tzname and offset in seconds as arguments and
              returns a `tzinfo` object.

        :param fuzzy: `<bool>` Whether to allow fuzzy parsing.
            If `True`, string like "Today is January 1, 2047 at 8:21:00AM" can be parsed into `2047-01-01 08:21:00`.
        :raises `ValueError`: If failed to parse the given `timestr`.
        :return: `<datetime.datetime>` The parsed datetime object.
        """

        try:
            return self._parse(
                timestr, default, dayfirst, yearfirst, ignoretz, tzinfos, fuzzy
            )
        except Exception as err:
            raise ValueError("<Parser> %s" % err) from err

    @cython.cfunc
    @cython.inline(True)
    def _parse(
        self,
        timestr: str,
        default: object,
        dayfirst: cython.bint,
        yearfirst: cython.bint,
        ignoretz: cython.bint,
        tzinfos: object,
        fuzzy: cython.bint,
    ) -> datetime.datetime:
        """(Internal parse method.)

        :param timestr: `<str>` A string containing date/time stamp.
        :param default: `<datetime>` The default object, which will be used as the base to fillin missing time elements from parsed results.
            If set to `None` (default), the current local year/month/day will be used as the default base.
        :param dayfirst: `<bool>` Whether to interpret the first value in an ambiguous date (e.g. 01/05/09) as the day (`True`) or month (`False`).
            When `yearfirst` is `True`, this distinguishes between Y/D/M and Y/M/D. If set to `None`,
            the `dayfirst` settings in `ParserInfo` will be used (defaults to False).
        :param yearfirst: `<bool>` Whether to interpret the first value in an ambiguous date (e.g. 01/05/09) as the year.
            If `True`, the first number is taken as year, otherwise the last number. If set to `None`,
            the `yearfirst` settings in `ParserInfo` will be used (defaults to False).
        :param ignoretz: `<bool>` Whether to ignore timezone info and only return naive datetime.
        :param tzinfos: Additional timezone parsing argument. Applicable when `ignoretz` is False. Accepts:
            - `<dict[str, int | tzinfo]>`: A dictionary where timezone name as the key and timezone
              offset in seconds or `tzinfo` object as the value.
            - `<TzinfoFactory>`: A callable which takes tzname and offset in seconds as arguments and
              returns a `tzinfo` object.

        :param fuzzy: `<bool>` Whether to allow fuzzy parsing.
            If `True`, string like "Today is January 1, 2047 at 8:21:00AM" can be parsed into `2047-01-01 08:21:00`.
        :raises `ValueError`: If failed to parse the given `timestr`.
        :return: `<datetime.datetime>` The parsed datetime object.
        """
        # Split timestr into tokens
        tokens: list[str] = TimeLex(timestr).split()  # l
        token_count: cython.int = len(tokens)  # len_l
        idx: cython.int = 0  # i

        # Initialize YMD & Result
        ymd, res = YMD(), Result()

        try:
            while idx < token_count:
                # Get token
                token: str = tokens[idx]  # value_repr

                # Check numeric
                if self._is_numeric_token(token):
                    idx = self._parse_numeric_token(
                        idx, token, tokens, token_count, fuzzy, ymd, res
                    )
                    idx += 1
                    continue

                # Check weekday
                weekday: cython.int = self.__info.weekday(token)
                if weekday != -1:
                    res._weekday = weekday
                    idx += 1
                    continue

                # Check month
                month: cython.int = self.__info.month(token)
                if month != -1:
                    idx = self._parse_month_token(idx, tokens, token_count, month, ymd)
                    idx += 1
                    continue

                # Check am/pm
                ampm: cython.int = self.__info.ampm(token)
                if ampm != -1 and self._valid_ampm_flag(res, fuzzy):
                    res._hour = self._adjust_ampm(res._hour, ampm)
                    res._ampm = ampm
                    idx += 1
                    continue

                # Check timezone name
                if self._could_be_tzname(token, True, res):
                    idx = self._parse_tzname(idx, token, tokens, token_count, res)
                    idx += 1
                    continue

                # Check timezone offset
                if res._hour != -1 and token == "+" or token == "-":
                    idx = self._prase_tzoffset(idx, token, tokens, token_count, res)
                    idx += 1
                    continue

                # Failed to handle token
                if not self.__info.jump(token) and not fuzzy:
                    raise ValueError("Failed to handle token: %s" % repr(token))

                # Next token
                idx += 1

            # End of loop & process year/month/day
            ymd.resolve(
                self.__info._dayfirst if dayfirst is None else dayfirst,
                self.__info._yearfirst if yearfirst is None else yearfirst,
            )
            res._century_specified = ymd._century_specified
            res._year = ymd._year
            res._month = ymd._month
            res._day = ymd._day

        except Exception as err:
            raise ValueError(
                "Unknown string format %s. Error: %s" % (repr(timestr), err)
            ) from err

        # Validate & Adjust
        if not res:
            raise ValueError("String does not contain any date: %s" % (repr(timestr)))
        res = self.__info._adjust_result(res)

        # Build
        try:
            return self._build(res, default, tzinfos, ignoretz)
        except Exception as err:
            raise ValueError(
                "Failed to build time from: %s. Error: %s" % (repr(timestr), err)
            ) from err

    # Numeric token ------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.boundscheck(True)
    @cython.exceptval(-1, check=False)
    def _parse_numeric_token(
        self,
        idx: cython.int,
        token: str,
        tokens: list,
        token_count: cython.int,
        fuzzy: cython.bint,
        ymd: YMD,
        res: Result,
    ) -> cython.int:
        # Preload token value
        value: cython.double = self._convert_numeric_token(token)
        temp_value: cython.int

        # Preload token info
        token_len: cython.int = len(token)  # len_li
        next_token: str = tokens[idx + 1] if idx + 1 < token_count else None

        # 19990101T23[59]
        if (
            len(ymd) == 3
            and res._hour == -1
            and (token_len == 2 or token_len == 4)
            and (
                idx + 1 >= token_count
                or (next_token != ":" and self.__info.hms(next_token) == -1)
            )
        ):
            res._hour = int(token[0:2])
            if token_len == 4:
                res._minute = int(token[2:4])
            return idx  # exit

        # YYMMDD or HHMMSS[.ss]
        if token_len == 6 or (token_len > 6 and token.find(".") == 6):
            if not ymd and not str_contains(token, "."):
                ymd.append(token[0:2])
                ymd.append(token[2:4])
                ymd.append(token[4:])
            else:
                # 19990101T235959[.59]
                res._hour = int(token[0:2])
                res._minute = int(token[2:4])
                self._set_sec_us(token[4:], res)
            return idx  # exit

        # YYYYMMDD
        if token_len == 8 or token_len == 12 or token_len == 14:
            ymd.append(token[0:4], 1)
            ymd.append(token[4:6])
            ymd.append(token[6:8])
            if token_len > 8:
                res._hour = int(token[8:10])
                res._minute = int(token[10:12])
                if token_len > 12:
                    res._second = int(token[12:14])
            return idx  # exit

        # HH[ ]h or MM[ ]m or SS[.ss][ ]s
        hms_idx = self._find_hms_idx(idx, next_token, tokens, token_count, True)
        if hms_idx != -1:
            hms: cython.int = self.__info.hms(tokens[hms_idx])
            if hms_idx > idx:
                idx = hms_idx
            else:
                hms += 1
            # Assign value
            if hms == 0:
                self._set_hour_min(value, res)
            elif hms == 1:
                self._set_min_sec(value, res)
            elif hms == 2:
                self._set_sec_us(token, res)
            return idx  # exit

        # HH:MM[:SS[.ss]]
        if idx + 2 < token_count and next_token == ":":
            res._hour = int(value)
            mins = self._convert_numeric_token(tokens[idx + 2])
            self._set_min_sec(mins, res)
            if idx + 4 < token_count and tokens[idx + 3] == ":":
                self._set_sec_us(tokens[idx + 4], res)
                idx += 2
            return idx + 2  # exit

        # YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
        if idx + 1 < token_count and (
            next_token == "-" or next_token == "/" or next_token == "."
        ):
            ymd.append(token)
            nxt2_token: str = tokens[idx + 2] if idx + 2 < token_count else None
            if nxt2_token is not None and not self.__info.jump(nxt2_token):
                # 01-01[-01]
                try:
                    temp_value = int(nxt2_token)
                    ymd.append(temp_value)
                # 01-Jan[-01]
                except Exception:
                    temp_value = self.__info.month(nxt2_token)
                    if temp_value != -1:
                        ymd.append(temp_value, 2)
                    else:
                        raise ValueError(
                            "Cannot parse month from %s." % repr(nxt2_token)
                        )

                # We have all three members
                if idx + 3 < token_count and tokens[idx + 3] == next_token:  # as sep
                    nxt4_token: str = tokens[idx + 4]
                    temp_value = self.__info.month(nxt4_token)
                    if temp_value != -1:
                        ymd.append(temp_value, 2)
                    else:
                        ymd.append(nxt4_token)
                    idx += 2
                idx += 1
            return idx + 1  # exit

        # "hour AM" or year|month|day
        if idx + 1 >= token_count or self.__info.jump(next_token):
            if idx + 2 < token_count:
                ampm: cython.int = self.__info.ampm(tokens[idx + 2])
                # 12 AM
                if ampm != -1:
                    res._hour = self._adjust_ampm(int(value), ampm)
                    idx += 1
                # Year, month or day
                else:
                    ymd.append(token)
            # Year, month or day
            else:
                ymd.append(token)
            return idx + 1  # exit

        # "hourAM"
        if 0 <= value < 24:
            ampm: cython.int = self.__info.ampm(next_token)
            # 12am
            if ampm != -1:
                res._hour = self._adjust_ampm(int(value), ampm)
                return idx + 1  # exit

        # Possible a day
        if ymd.could_be_day(int(value)):
            ymd.append(token)
            return idx

        # Invalid
        if not fuzzy:
            raise ValueError("Failed to handle token: %s" % repr(token))

        return idx

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_numeric_token(self, token: str) -> cython.bint:
        try:
            float(token)
            return True
        except Exception:
            return False

    @cython.cfunc
    @cython.inline(True)
    def _convert_numeric_token(self, token: str) -> cython.double:
        try:
            value = float(token)
            if not cymath.is_finite(value):
                raise ValueError("Token is infinite or NaN.")
            else:
                return value
        except Exception as err:
            raise ValueError(
                "Could not convert token %s to float - %s" % (repr(token), err)
            ) from err

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _set_hour_min(self, value: cython.double, res: Result):
        res._hour = int(value)
        remainder: cython.double = value % 1
        if remainder:
            res._minute = int(remainder * 60)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _set_min_sec(self, value: cython.double, res: Result):
        res._minute = int(value)
        remainder: cython.double = value % 1
        if remainder:
            res._second = int(remainder * 60)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _set_sec_us(self, token: str, res: Result):
        if str_contains(token, "."):
            sec, us = token.split(".")
            res._second = int(sec)
            res._microsecond = int(us.ljust(6, "0")[:6])
        else:
            res._second = int(token)
            res._microsecond = 0

    @cython.cfunc
    @cython.inline(True)
    def _find_hms_idx(
        self,
        idx: cython.int,
        next_token: str,
        tokens: list,
        token_count: cython.int,
        allow_jump: cython.bint,
    ) -> cython.int:
        # There is an "h", "m", or "s" label following this token.
        # We take assign the upcoming label to the current token.
        # e.g. the "12" in 12h"
        if idx + 1 < token_count and self.__info.hms(next_token) != -1:
            return idx + 1

        # There is a space and then an "h", "m", or "s" label.
        # e.g. the "12" in "12 h"
        if (
            allow_jump
            and idx + 2 < token_count
            and next_token == " "
            and self.__info.hms(tokens[idx + 2]) != -1
        ):
            return idx + 2

        # There is a "h", "m", or "s" preceding this token. Since neither
        # of the previous cases was hit, there is no label following this
        # token, so we use the previous label.
        # e.g. the "04" in "12h04"
        if idx > 0 and self.__info.hms(tokens[idx - 1]) != -1:
            return idx - 1

        # If we are looking at the final token, we allow for a
        # backward-looking check to skip over a space.
        # TODO: Are we sure this is the right condition here?
        if (
            1 < idx == token_count - 1
            and tokens[idx - 1] == " "
            and self.__info.hms(tokens[idx - 2]) != -1
        ):
            return idx - 2

        # No hms found
        return -1

    # Month token --------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_month_token(
        self,
        idx: cython.int,
        tokens: list,
        token_count: cython.int,
        month: cython.int,
        ymd: YMD,
    ) -> cython.int:
        # Append month
        ymd.append(month, 2)

        # Try to get year & day
        if idx + 1 < token_count:
            next_token: str = tokens[idx + 1]

            # Jan-01[-99?] uncertain
            if next_token == "-" or next_token == "/":
                ymd.append(tokens[idx + 2])
                if idx + 3 < token_count and tokens[idx + 3] == next_token:
                    # Jan-01-99 confirmed
                    ymd.append(tokens[idx + 4])
                    idx += 2
                idx += 2

            # Jan of 01. In this case, 01 is clearly year
            elif (
                idx + 4 < token_count
                and tokens[idx + 3] == next_token == " "
                and self.__info.pertain(tokens[idx + 2])
            ):
                try:
                    temp_value = int(tokens[idx + 4])
                    ymd.append(self.__info._convert_year(temp_value, False), 1)
                except Exception:
                    # Wrong guess
                    pass
                idx += 4

        return idx  # exit

    # AM/PM token --------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _valid_ampm_flag(self, res: Result, fuzzy: cython.bint) -> cython.bint:
        # If there's already an AM/PM flag, this one isn't one.
        if fuzzy and res._ampm != -1:
            return False

        # If AM/PM is found and hour is not, raise a ValueError
        if res._hour == -1:
            if fuzzy:
                return False
            else:
                raise ValueError("Missing hour for AM/PM flag.")

        # If AM/PM is found, it's a 12 hour clock, so raise
        # an error for invalid range
        if not 0 <= res._hour <= 12:
            if fuzzy:
                return False
            else:
                raise ValueError("Invalid hour (%d) for AM/PM flag." % res._hour)

        # Valid AM/PM flag
        return True

    @cython.cfunc
    @cython.inline(True)
    def _adjust_ampm(self, hour: cython.int, ampm: cython.int) -> cython.int:
        if hour < 12 and ampm == 1:
            hour += 12
        elif hour == 12 and ampm == 0:
            hour = 0
        return hour

    # Timezone token -----------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _could_be_tzname(
        self,
        token: str,
        chech_tzoffset: cython.bint,
        res: Result,
    ) -> cython.bint:
        if (
            res._hour != -1
            and (res._tzoffset == -1000000 if chech_tzoffset else True)
            and not res._tz_name
            and len(token) <= 5
        ):
            # Could be a UTC timezone
            if self.__info.utczone(token):
                return True

            # Literal could be a timezone
            ASCII_UPPER: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for i in token:
                if not str_contains(ASCII_UPPER, i):
                    return False
            return True
        else:
            return False

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_tzname(
        self,
        idx: cython.int,
        token: str,
        tokens: list,
        token_count: cython.int,
        res: Result,
    ) -> cython.int:
        # Set timezone name & offset
        res._tz_name = token
        res._tzoffset = self.__info.tzoffset(token)

        # Check for something like GMT+3, or BRST+3. Notice
        # that it doesn't mean "I am 3 hours after GMT", but
        # "my time +3 is GMT". If found, we reverse the
        # logic so that timezone parsing code will get it
        # right.
        if idx + 1 < token_count:
            next_token: str = tokens[idx + 1]
            if next_token == "+":
                tokens[idx + 1] = "-"
            elif next_token == "-":
                tokens[idx + 1] = "+"
            else:
                return idx  # exit
            res._tzoffset = -1000000  # set to `None`

            # With something like GMT+3, the timezone is *not* GMT.
            if self.__info.utczone(res._tz_name):
                res._tz_name = None  # set to `None`

        return idx  # exit

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prase_tzoffset(
        self,
        idx: cython.int,
        token: str,
        tokens: list,
        token_count: cython.int,
        res: Result,
    ) -> cython.int:
        # The next token could be a timezone offset
        next_token: str = tokens[idx + 1]
        next_value: cython.int
        try:
            next_value = int(next_token)
        except Exception:
            return idx  # wrong guess & exit

        # Pre binding
        token_len: cython.int = len(next_token)
        hour_offset: cython.int
        min_offset: cython.int

        try:
            # -0300
            if token_len == 4:
                hour_offset = int(next_token[0:2])
                min_offset = int(next_token[2:4])
            # -03:00
            elif idx + 2 < token_count and tokens[idx + 2] == ":":
                hour_offset = next_value
                min_offset = int(tokens[idx + 3])
                idx += 2
            # -[0]3
            elif token_len <= 2:
                hour_offset = next_value
                min_offset = 0
            # Invalid
            else:
                raise ValueError("Invalid token")
        except Exception as err:
            raise ValueError("Can't parse tzoffset from %s" % repr(next_token)) from err

        # Set timezone offset
        sign: cython.int = 1 if token == "+" else -1
        res._tzoffset = sign * (hour_offset * 3600 + min_offset * 60)

        # -0300(BRST) # with space
        nxt4_token: str = tokens[idx + 4] if idx + 4 < token_count else None
        if nxt4_token is not None and nxt4_token == ")":
            n3rd_token: str = tokens[idx + 3]
            if self._could_be_tzname(n3rd_token, False, res) and tokens[idx + 2] == "(":
                res._tz_name = n3rd_token
                return idx + 4  # exit

        # -0300 (BRST) # w/o space
        nxt5_token: str = tokens[idx + 5] if idx + 5 < token_count else None
        if nxt5_token is not None and nxt5_token == ")":
            if (
                self._could_be_tzname(nxt4_token, False, res)
                and tokens[idx + 3] == "("
                and self.__info.jump(tokens[idx + 2])
            ):
                res._tz_name = nxt4_token
                return idx + 5  # exit

        return idx + 1  # exit

    # Build --------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _build(
        self,
        res: Result,
        default: object,
        tzinfos: object,
        ignoretz: cython.bint,
    ) -> datetime.datetime:
        # Ignore tzinfo (Build naive)
        if ignoretz:
            return self._build_datetime(res, default, None)

        tzinfo: object
        tzdata: object
        dt: datetime.datetime
        # No extra tzinfos provided (default).
        if tzinfos is None:
            # Local timezone (Special case)
            if res._tz_name and res._tz_name in time.tzname:
                # Build with local tzinfo
                dt = self._build_datetime(res, default, None)
                dt = cydt.dt_replace_tzinfo(dt, cydt.gen_timezone_local(dt))
                # Handle ambiguous local datetime
                dt = self._handle_anbiguous_time(dt, res._tz_name)
                # Adjust for winter GMT zones parsed in the UK
                if dt.tzname() != res._tz_name and self.__info.utczone(res._tz_name):
                    dt = cydt.dt_replace_tzinfo(dt, cydt.UTC)
                return dt  # Exit

            # Parsed tzoffset is 0 (UTC)
            if res._tzoffset == 0:
                tzinfo = cydt.UTC
            # Parsed tzname & tzoffset (Custom tzinfo)
            elif res._tzoffset != -1000000:
                tzinfo = cydt.gen_timezone(res._tzoffset, res._tz_name)
            # No tzinfo
            else:
                tzinfo = None
            # Build datetime
            dt = self._build_datetime(res, default, tzinfo)

        # Extra tzinfos provided
        else:
            # Extra tzinfos (dict[str, int | tzinfo])
            if isinstance(tzinfos, dict):
                tzdata = tzinfos.get(res._tz_name)
            # Extra tzinfos (TzinfoFactory)
            elif callable(tzinfos) and res._tzoffset != -1000000:
                try:
                    tzdata = tzinfos(res._tz_name or None, res._tzoffset)
                except Exception:
                    tzdata = tzinfos(res._tzoffset, res._tz_name or None)
            # Invalid tzinfos
            else:
                raise ValueError(
                    "<Parser> Only accepts `dict[str, int | tzinfo]` or `TzinfoFactory` as tzinfos, instead got: %s %s"
                    % (repr(tzinfos), tzinfos)
                )
            # Convert tzdata to tzinfo
            if tzdata is None or cydt.is_tzinfo(tzdata):
                tzinfo = tzdata
            elif isinstance(tzdata, int):
                tzinfo = cydt.gen_timezone(tzdata, res._tz_name)
            elif isinstance(tzdata, str):
                tzinfo = dateutil_tz.tzstr(tzdata)
            else:
                tzinfo = None
            # Build datetime
            dt = self._build_datetime(res, default, tzinfo)
            # Handle ambiguous local datetime
            dt = self._handle_anbiguous_time(dt, res._tz_name)

        # Return datetime
        return dt  # Exit

    @cython.cfunc
    @cython.inline(True)
    def _build_datetime(
        self,
        res: Result,
        default: object,
        tzinfo: object,
    ) -> datetime.datetime:
        # Determine Whether provided valid default
        dflt_is_date: cython.bint = cydt.is_date(default)
        dflt_is_dt: cython.bint = cydt.is_dt(default)
        tm = cytime.localtime()

        # Build datetime year
        year: cython.int
        if res._year > 0:  # satisfied
            year = res._year
        elif dflt_is_date:  # from default
            year = cydt.get_year(default)
        else:  # from localtime
            year = tm.tm_year

        # Build datetime month
        month: cython.int
        if res._month > 0:  # satisfied
            month = res._month
        elif dflt_is_date:  # from default
            month = cydt.get_month(default)
        else:  # from localtime
            month = tm.tm_mon

        # Build datetime day
        day: cython.int
        if res._day > 0:  # satisfied
            day = res._day
        elif dflt_is_dt:  # from default
            day = cydt.get_day(default)
        else:  # from localtime
            day = tm.tm_mday
        if day > 28:  # adjust days in month
            day = min(day, cydt.days_in_month(year, month))

        # Build datetime hour
        hour: cython.int
        if res._hour >= 0:  # satisfied
            hour = res._hour
        elif dflt_is_dt:  # from default
            hour = cydt.get_dt_hour(default)
        else:
            hour = 0

        # Build datetime minute
        minute: cython.int
        if res._minute >= 0:  # satisfied
            minute = res._minute
        elif dflt_is_dt:  # from default
            minute = cydt.get_dt_minute(default)
        else:
            minute = 0

        # Build datetime second
        second: cython.int
        if res._second >= 0:  # satisfied
            second = res._second
        elif dflt_is_dt:  # from default
            second = cydt.get_dt_second(default)
        else:
            second = 0

        # Build datetime microsecond
        microsecond: cython.int
        if res._microsecond >= 0:  # satisfied
            microsecond = res._microsecond
        elif dflt_is_dt:  # from default
            microsecond = cydt.get_dt_microsecond(default)
        else:
            microsecond = 0

        # Generate datetime
        dt: datetime.datetime = cydt.gen_dt(
            year, month, day, hour, minute, second, microsecond, tzinfo, 0
        )

        # Adjust weekday
        if 0 <= res._weekday <= 6:
            dt = cytimedelta(
                days=res._weekday - cydt.get_weekday(dt),
            )._add_date_time(dt)

        # Return datetime
        return dt

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _handle_anbiguous_time(
        self,
        dt: datetime.datetime,
        tzname: str,
    ) -> datetime.datetime:
        if dt.tzname() != tzname:
            new_dt: datetime.datetime = cydt.dt_replace_fold(dt, 1)
            if new_dt.tzname() == tzname:
                return new_dt
        return dt


# Parse ---------------------------------------------------------------------------------------
@cython.ccall
def parse(
    timestr: str,
    default: Union[datetime.datetime, datetime.date, None] = None,
    dayfirst: cython.bint = False,
    yearfirst: cython.bint = False,
    ignoretz: cython.bint = False,
    tzinfos: Union[Type[dateutil_tz.tzfile], dict[str, int], None] = None,
    fuzzy: cython.bint = False,
    parserinfo: Union[ParserInfo, parserinfo, None] = None,
) -> datetime.datetime:
    """Parse a string contains date/time stamp into `datetime.datetime`.

    :param timestr: `<str>` A string containing date/time stamp.
    :param default: `<datetime>` The default object, which will be used as the base to fillin missing time elements from parsed results.
        If set to `None` (default), the current local year/month/day will be used as the default base.
    :param dayfirst: `<bool>` Whether to interpret the first value in an ambiguous date (e.g. 01/05/09) as the day (`True`) or month (`False`).
        When `yearfirst` is `True`, this distinguishes between Y/D/M and Y/M/D. If set to `None`,
        the `dayfirst` settings in `ParserInfo` will be used (defaults to False).
    :param yearfirst: `<bool>` Whether to interpret the first value in an ambiguous date (e.g. 01/05/09) as the year.
        If `True`, the first number is taken as year, otherwise the last number. If set to `None`,
        the `yearfirst` settings in `ParserInfo` will be used (defaults to False).
    :param ignoretz: `<bool>` Whether to ignore timezone info and only return naive datetime.
    :param tzinfos: Additional timezone parsing argument. Applicable when `ignoretz` is False. Accepts:
        - `<dict[str, int | tzinfo]>`: A dictionary where timezone name as the key and timezone
            offset in seconds or `tzinfo` object as the value.
        - `<TzinfoFactory>`: A callable which takes tzname and offset in seconds as arguments and
            returns a `tzinfo` object.

    :param fuzzy: `<bool>` Whether to allow fuzzy parsing.
        If `True`, string like "Today is January 1, 2047 at 8:21:00AM" can be parsed into `2047-01-01 08:21:00`.

    :param info: `<ParserInfo>` The parserinfo to use, accepts:
        - `<ParserInfo>`: A ParserInfo instance from `cytimes.ParserInfo`.
        - `<parserinfo>`: A parserinfo instance from `dateutil.parserinfo`.
        - `None`: The default ParserInfo will be used.

    :raises `ValueError`: If failed to parse the given `timestr`.
    :return: `<datetime.datetime>` The parsed datetime object.
    """
    try:
        return Parser(parserinfo)._parse(
            timestr, default, dayfirst, yearfirst, ignoretz, tzinfos, fuzzy
        )
    except Exception as err:
        raise ValueError("<Parser> %s" % err) from err
