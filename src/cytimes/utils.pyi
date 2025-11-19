import cython
import zoneinfo
import datetime
import numpy as np

# Constants -----------------------------------------------------------------------------------------
# . argument
SENTINEL: int
# . pandas
NAT: object
# . date
ORDINAL_MAX: int
# . datetime
#: EPOCH (1970-01-01)
EPOCH_DT: datetime.datetime
EPOCH_YEAR: int
EPOCH_MONTH: int
EPOCH_DAY: int
EPOCH_HOUR: int
EPOCH_MINUTE: int
EPOCH_SECOND: int
EPOCH_MILLISECOND: int
EPOCH_MICROSECOND: int
# . timezone
UTC: datetime.tzinfo
T_TIMEZONE: type[datetime.timezone]
T_ZONEINFO: type[zoneinfo.ZoneInfo]
NULL_TZOFFSET: int
# . conversion for seconds
SS_MINUTE: int
SS_HOUR: int
SS_DAY: int
# . conversion for milliseconds
MS_SECOND: int
MS_MINUTE: int
MS_HOUR: int
MS_DAY: int
# . conversion for microseconds
US_MILLISECOND: int
US_SECOND: int
US_MINUTE: int
US_HOUR: int
US_DAY: int
# . conversion for nanoseconds
NS_MICROSECOND: int
NS_MILLISECOND: int
NS_SECOND: int
NS_MINUTE: int
NS_HOUR: int
NS_DAY: int
# . conversion for timedelta64
TD64_YY_DAY: int
TD64_YY_SECOND: int
TD64_YY_MILLISECOND: int
TD64_YY_MICROSECOND: int
TD64_YY_NANOSECOND: int
TD64_MM_DAY: int
TD64_MM_SECOND: int
TD64_MM_MILLISECOND: int
TD64_MM_MICROSECOND: int
TD64_MM_NANOSECOND: int
# . datetime64 range
#: Minimum datetime64 in nanoseconds (1677-09-21 00:12:43.145224193)
DT64_NS_YY_MIN: int
DT64_NS_MM_MIN: int
DT64_NS_WW_MIN: int
DT64_NS_DD_MIN: int
DT64_NS_HH_MIN: int
DT64_NS_MI_MIN: int
DT64_NS_SS_MIN: int
DT64_NS_MS_MIN: int
DT64_NS_US_MIN: int
DT64_NS_NS_MIN: int
#: Maximum datetime64 in nanoseconds (2262-04-11 23:47:16.854775807)
DT64_NS_YY_MAX: int
DT64_NS_MM_MAX: int
DT64_NS_WW_MAX: int
DT64_NS_DD_MAX: int
DT64_NS_HH_MAX: int
DT64_NS_MI_MAX: int
DT64_NS_SS_MAX: int
DT64_NS_MS_MAX: int
DT64_NS_US_MAX: int
DT64_NS_NS_MAX: int
# . datetime64 dtype
DT64_DTYPE_YY: np.dtype
DT64_DTYPE_MM: np.dtype
DT64_DTYPE_WW: np.dtype
DT64_DTYPE_DD: np.dtype
DT64_DTYPE_HH: np.dtype
DT64_DTYPE_MI: np.dtype
DT64_DTYPE_SS: np.dtype
DT64_DTYPE_MS: np.dtype
DT64_DTYPE_US: np.dtype
DT64_DTYPE_NS: np.dtype
DT64_DTYPE_PS: np.dtype
DT64_DTYPE_FS: np.dtype
DT64_DTYPE_AS: np.dtype
# . numpy datetime units
DT_NPY_UNIT_YY: int
DT_NPY_UNIT_MM: int
DT_NPY_UNIT_WW: int
DT_NPY_UNIT_DD: int
DT_NPY_UNIT_HH: int
DT_NPY_UNIT_MI: int
DT_NPY_UNIT_SS: int
DT_NPY_UNIT_MS: int
DT_NPY_UNIT_US: int
DT_NPY_UNIT_NS: int
DT_NPY_UNIT_PS: int
DT_NPY_UNIT_FS: int
DT_NPY_UNIT_AS: int

# Math ----------------------------------------------------------------------------------------------
def math_mod(num: int, div: int, offset: int = 0) -> int:
    """(cfunc) Compute module with Python semantics `<'int'>`.

    :param num `<'int'>`: Dividend.
    :param div `<'int'>`: Divisor (non-zero).
    :param offset `<'int'>`: Optional value to add the result. Defaults to `0`
    :returns `<'int'>`: The modulo result.
    :raises `<'ZeroDivisionError'>`: When `div` is zero.

    ## Equivalent
    >>> (num % div) + offset
    """

def math_div_even(num: int, div: int, offset: int = 0) -> int:
    """(cfucn) Divide then round to nearest, ties-to-even (bankers' rounding) `<'int'>`.

    :param num `<'int'>`: Dividend.
    :param div `<'int'>`: Divisor (non-zero).
    :param offset `<'int'>`: Optional value to add the result. Defaults to `0`
    :returns `<'int'>`: The division result.
    :raises `<'ZeroDivisionError'>`: When `div` is zero.
    :raises `<'OverflowError'>`: When the result does not fit in signed 64-bit.

    ## Equivalent
    >>> round(num / div, 0) + offset
    """

def math_div_up(num: int, div: int, offset: int = 0) -> int:
    """(cfunc) Divide then round half away from zero (round-half-up) `<'int'>`.

    :param num `<'int'>`: Dividend.
    :param div `<'int'>`: Divisor (non-zero).
    :param offset `<'int'>`: Optional value to add the result. Defaults to `0`
    :returns `<'int'>`: The division result.
    :raises `<'ZeroDivisionError'>`: When `div` is zero.
    :raises `<'OverflowError'>`: When the result does not fit in signed 64-bit.

    ## Equivalent
    >>> (Decimal(num) / Decimal(div)).to_integral_value(rounding=ROUND_HALF_UP) + offset
    """

def math_div_down(num: int, div: int, offset: int = 0) -> int:
    """(cfunc) Divide then round half toward zero (round-half-down) `<'int'>`.

    :param num `<'int'>`: Dividend.
    :param div `<'int'>`: Divisor (non-zero).
    :param offset `<'int'>`: Optional value to add the result. Defaults to `0`
    :returns `<'int'>`: The division result.
    :raises `<'ZeroDivisionError'>`: When `div` is zero.
    :raises `<'OverflowError'>`: When the result does not fit in signed 64-bit.

    ## Equivalent
    >>> (Decimal(num) / Decimal(div)).to_integral_value(rounding=ROUND_HALF_DOWN) + offset
    """

def math_div_ceil(num: int, div: int, offset: int = 0) -> int:
    """(cfunc) Divide then take the mathematical ceiling `<'int'>`.

    :param num `<'int'>`: Dividend.
    :param div `<'int'>`: Divisor (non-zero).
    :param offset `<'int'>`: Optional value to add the result. Defaults to `0`
    :returns `<'int'>`: The division result.
    :raises `<'ZeroDivisionError'>`: When `div` is zero.
    :raises `<'OverflowError'>`: When the result does not fit in signed 64-bit.

    ## Equivalent
    >>> math.ceil(num / div) + offset
    """

def math_div_floor(num: int, div: int, offset: int = 0) -> int:
    """(cfunc) Divide then take the mathematical floor `<'int'>`.

    :param num `<'int'>`: Dividend.
    :param div `<'int'>`: Divisor (non-zero).
    :param offset `<'int'>`: Optional value to add the result. Defaults to `0`
    :returns `<'int'>`: The division result.
    :raises `<'ZeroDivisionError'>`: When `div` is zero.
    :raises `<'OverflowError'>`: When the result does not fit in signed 64-bit.

    ## Equivalent
    >>> math.floor(num / div) + offset
    """

def math_div_trunc(num: int, div: int, offset: int = 0) -> int:
    """(cfunc) Divide and truncate toward zero (C/CPython style) `<'int'>`.

    :param num `<'int'>`: Dividend.
    :param div `<'int'>`: Divisor (non-zero).
    :param offset `<'int'>`: Optional value to add the result. Defaults to `0`
    :returns `<'int'>`: The division result.
    :raises `<'ZeroDivisionError'>`: When `div` is zero.
    :raises `<'OverflowError'>`: When the result does not fit in signed 64-bit.
    """

def abs_diff_ull(a: int, b: int) -> int:
    """(cfunc) Return |a - b| as uint64 using unsigned arithmetic `<'int'>`.

    Safe for all long long pairs (avoids signed overflow/UB near LLONG_MIN/LLONG_MAX).
    """

# Parser --------------------------------------------------------------------------------------------
# . check: character
def is_dot(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check whether `ch` is a dot `'.'` `<'bool'>`.

    - Dot: `'.'` (46)
    """

def is_comma(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check whether `ch` is a comma `','` `<'bool'>`.

    - Comma: `','` (44)
    """

def is_plus_sign(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check whether `ch` is a plus sign `'+'` `<'bool'>`.

    - Plus sign: `'+'` (43)
    """

def is_minus_sign(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check whether `ch` is a minus sign `'-'` `<'bool'>`.

    - Plus sign: `'-'` (45)
    """

def is_plus_or_minus_sign(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check whether `ch` is a plus or minus sign `'-'` `<'bool'>`.

    - Plus sign : `'+'` (43)
    - Minus sign: `'-'` (45)
    """

def is_iso_sep(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check whether `ch` is an ISO 8601 date/time separator `<'bool'>`.

    - Date / Time separators: `' '` (32) or `'T'` (case-insensitive: 84 & 116).
    """

def is_date_sep(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check whether `ch` is a date-field seperator `<'bool'>`.

    - Date-field separators: `'-'` (45), `'.'` (46) or `'/'` (47)
    """

def is_time_sep(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check whether `ch` is the time-field separator `<'bool'>`.

    - Time-field seperator: `':'` (58)
    """

def is_isoweek_sep(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check whether `ch` is the ISO week designator `<'bool'>`.

    - ISO week designator: `'W'` (case-insensitive: 87 & 119).
    """

def is_ascii_ctl(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check whether `ch` is a control charactor `<'bool'>`.

    - ASCII control characters (0-31) and (127)
    """

def is_ascii_ctl_or_space(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check whether `ch` is a control or space charactor `<'bool'>`.

    - ASCII control characters (0-31) and (127)
    - ASCII space character: (32)
    """

def is_ascii_digit(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check whether `ch` is an ASCII digit `<'bool'>`.

    - ASSCI digits: `'0'` (48) ... `'9'` (57)
    """

def is_ascii_letter_upper(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check whether `ch` is an uppercase ASCII letter `<'bool'>`.

    - Uppercase ASCII letters: `'A'` (65) ... `'Z'` (90)
    """

def is_ascii_letter_lower(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check whether `ch` is a lowercase ASCII letter `<'bool'>`.

    - Lowercase ASCII letters: `'a'` (97) ... `'z'` (122)
    """

def is_ascii_letter(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check whether `ch` is an ASCII letter (case-insensitive) `<'bool'>`.

    - Uppercase ASCII letters: `'A'` (65) ... `'Z'` (90)
    - Lowercase ASCII letters: `'a'` (97) ... `'z'` (122)
    """

def is_alpha(ch: cython.Py_UCS4) -> bool:
    """(cfunc) Check whether `ch` is an alphabetic character `<'bool'>`.

    - Uses Unicode definition of alphabetic characters.
    """

# . check: string
def is_str_dot(token: str, token_len: int = -1) -> bool:
    """(cfunc) Check whether `token` is a single-character dot `'.'` `<'bool'>`.

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` is a single-character dot `'.'`; False otherwise.
    """

def is_str_comma(token: str, token_len: int = -1) -> bool:
    """(cfunc) Check whether `token` is a single-character comma `','` `<'bool'>`.

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` is a single-character comma `'.,`; False otherwise.
    """

def is_str_iso_sep(token: str, token_len: int = -1) -> bool:
    """(cfunc) Check whether `token` is an ISO 8601 date-time separator `<'bool'>`.

    - Date / Time separators: `' '` or `'T'` (case-insensitive).

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` is a single-character ISO
        date-time separator; False otherwise.
    """

def is_str_date_sep(token: str, token_len: int = -1) -> bool:
    """(cfunc) Check whether `token` is a single-character date-field seperator `<'bool'>`.

    - Date-field separators: `'-'`, `'.'` or `'/'`

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` is a single-character date-field separator; False otherwise.
    """

def is_str_time_sep(token: str, token_len: int = -1) -> bool:
    """(cfunc) Check whether `token` is a single-character time-field seperator `<'bool'>`.

    - Time-field seperator: `':'`

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` is a single-character time-field separator; False otherwise.
    """

def is_str_isoweek_sep(token: str, token_len: int = -1) -> bool:
    """(cfunc) Check whether `token` is a single-character ISO week designator `<'bool'>`.

    - ISO week designator: `'W'` (case-insensitive).

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` is a single-character ISO week designator; False otherwise.
    """

def is_str_ascii_ctl(token: str, token_len: int = -1) -> bool:
    """(cfunc) Check whether `token` is a single control charactor `<'bool'>`.

    - ASCII control characters (0-31) and (127)

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` is a single control character; False otherwise.
    """

def is_str_ascii_ctl_or_space(token: str, token_len: int = -1) -> bool:
    """(cfunc) Check whether `token` is a single control or space charactor `<'bool'>`.

    - ASCII control characters (0-31) and (127)
    - ASCII space character: (32)

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` is a single control or space character; False otherwise.
    """

def is_str_ascii_digits(token: str, token_len: int = -1) -> bool:
    """(cfunc) Check whether `token` only contains ASCII digits `<'bool'>`.

    - ASSCI digits: `'0'` ... `'9'`

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` only contains ASCII digits; False otherwise.
    """

def is_str_ascii_letters_upper(token: str, token_len: int = -1) -> bool:
    """(cfunc) Check whether `token` only contains uppercase ASCII letters `<'bool'>`.

    - Uppercase ASCII letters: `'A'` ... `'Z'`

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` only contains uppercase ASCII letters; False otherwise.
    """

def is_str_ascii_letters_lower(token: str, token_len: int = -1) -> bool:
    """(cfunc) Check whether `token` only contains lowercase ASCII letters `<'bool'>`.

    - Lowercase ASCII letters: `'a'` ... `'z'`

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` only contains lowercase ASCII letters; False otherwise.
    """

def is_str_ascii_letters(token: str, token_len: int = -1) -> bool:
    """(cfunc) Check whether `token` only contains letters (case-insensitive) `<'bool'>`.

    - ASCII letters: `'a'` ... `'z'` and `'A'` ... `'Z'`

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` only contains letters (case-insensitive); False otherwise.
    """

def is_str_alphas(token: str, token_len: int = -1) -> bool:
    """(cfunc) Check whether `token` only contains alphabetic characters `<'bool'>`.

    - Uses Unicode definition of alphabetic characters.

    :param token `<'str'>`: The input token string to check.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'bool'>`: True if `token` only contains alphabetic characters; False otherwise.
    """

# . parse
def parse_numeric_kind(token: str, token_len: int = -1) -> int:
    """(cfunc) Classify `token` string as certain numeric kind `<'int'>`.

    :param token `<'str'>`: The input token string to classify.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: Classification result:

        - `1` = integer (ASCII digits only)
        - `2` = decimal (ASCII digits with a single `'.'`; more than one `'.'` is invalid)
        - `0` = not numeric (any other case, including empty string, `'.'` alone and prefixing `'+/-'` signs)
    """

def parse_isoyear(token: str, pos: int, token_len: int = -1) -> int:
    """(cfunc) Parse ISO format year component (YYYY) from a string,
    returns `-1` for invalid ISO years `<'int'>`.

    This function extracts and parses the year component from an ISO date string.
    It reads four characters starting at the specified position and converts them
    into an integer representing the year. The function ensures that the parsed
    year is valid (i.e., between '0000' and '9999').

    :param token `<'str'>`: The input token string containing the ISO year to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO year.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO year [1..9999], or `-1` if invalid.
    """

def parse_isomonth(token: str, pos: int, token_len: int = -1) -> int:
    """(cfunc) Parse ISO format month component (MM) from a string,
    returns `-1` for invalid ISO months `<'int'>`.

    This function extracts and parses the month component from an ISO date string.
    It reads two characters starting at the specified position and converts them
    into an integer representing the month. The function ensures that the parsed
    month is valid (i.e., between '01' and '12').

    :param token `<'str'>`: The input token string containing the ISO month to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO month.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO month `[1..12]`, or `-1` if invalid.
    """

def parse_isoday(token: str, pos: int, token_len: int = -1) -> int:
    """(cfunc) Parse ISO format day component (DD) from a string,
    returns `-1` for invalid ISO days `<'int'>`.

    This function extracts and parses the day component from an ISO date string.
    It reads two characters starting at the specified position and converts them
    into an integer representing the day. The function ensures that the parsed day
    is valid (i.e., between '01' and '31').

    :param token `<'str'>`: The input token string containing the ISO day to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO day.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO day `[1..31]`, or `-1` if invalid.
    """

def parse_isoweek(token: str, pos: int, token_len: int = -1) -> int:
    """(cfunc) Parse an ISO format week number component (WW) from a string,
    returns `-1` for invalid ISO week number `<'int'>`.

    This function extracts and parses the week number from an ISO date string.
    It reads two characters starting at the specified position and converts them
    into an integer representing the week number. The function ensures that the
    parsed week number is valid (i.e., between '01' and '53').

    :param token `<'str'>`: The input token string containing the ISO week number to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO week number.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO week number `[1..53]`, or `-1` if invalid.
    """

def parse_isoweekday(token: str, pos: int, token_len: int = -1) -> int:
    """(cfunc) Parse an ISO format weekday component (D) from a string,
    returns `-1` for invalid ISO weekdays `<'int'>`.

    This function extracts and parses the weekday component from an ISO date string.
    It reads a single character at the specified position and converts it into an
    integer representing the ISO weekday, where Monday is 1 and Sunday is 7.

    :param token `<'str'>`: The input token string containing the ISO weekday to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO weekday.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO weekday `[1..7]`, or `-1` if invalid.
    """

def parse_isodoy(token: str, pos: int, token_len: int = -1) -> int:
    """(cfunc) Parse an ISO format day-of-year component (DDD) from a string,
    returns `-1` for invalid ISO day of the year `<'int'>`.

    This function extracts and parses the day of the year from an ISO date string.
    It reads three characters starting at the specified position and converts them
    into an integer representing the day of the year. The function ensures that the
    parsed days are valid (i.e., between '001' and '366').

    :param token `<'str'>`: The input token string containing the ISO day of the year to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO day of the year.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO day of the year `[1..366]`, or `-1` if invalid.
    """

def parse_isohour(token: str, pos: int, token_len: int = -1) -> int:
    """(cfunc) Parse an ISO format hour (HH) component from a string,
    returns `-1` for invalid ISO hours `<'int'>`.

    This function extracts and parses the hour component from a time string in ISO format.
    It reads two characters starting at the specified position and converts them into an
    integer representing the hours. The function ensures that the parsed hours are valid
    (i.e., between '00' and '23').

    :param token `<'str'>`: The input token string containing the ISO hour to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO hour.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO hour `[0..23]`, or `-1` if invalid.
    """

def parse_isominute(token: str, pos: int, token_len: int = -1) -> int:
    """(cfunc) Parse an ISO format minute (MM) component from a string,
    returns `-1` for invalid ISO minutes `<'int'>`.

    This function extracts and parses the minute component from a time string in ISO format.
    It reads two characters starting at the specified position and converts them into an
    integer representing the minutes. The function ensures that the parsed minutes are valid
    (i.e., between '00' and '59').

    :param token `<'str'>`: The input token string containing the ISO minute to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO minute.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO minute `[0..59]`, or `-1` if invalid.
    """

def parse_isosecond(token: str, pos: int, token_len: int = -1) -> int:
    """(cfunc) Parse an ISO format second (SS) component from a string,
    returns `-1` for invalid ISO seconds `<'int'>`.

    This function extracts and parses the second component from a time string in ISO format.
    It reads two characters starting at the specified position and converts them into an
    integer representing the seconds. The function ensures that the parsed seconds are valid
    (i.e., between '00' and '59').

    :param token `<'str'>`: The input token string containing the ISO second to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO second.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO second `[0..59]`, or `-1` if invalid.
    """

def parse_isofraction(token: str, pos: int, token_len: int = -1) -> int:
    """(cfunc) Parse an ISO fractional time component (fractions of a second) from a string,
    returns `-1` for invalid ISO fraction `<'int'>`.

    This function extracts and parses a fractional time component in ISO format (e.g.,
    the fractional seconds in "2023-11-25T14:30:15.123456Z"). It reads up to six digits
    after the starting position, padding with zeros if necessary to ensure a six-digit
    integer representation.

    :param token `<'str'>`: The input token string containing the ISO fraction to parse.
    :param pos `<'int'>`: The starting position in the `token` of the ISO fraction.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The parsed ISO fraction `[0..999,999]`, or `-1` if invalid.
    """

def parse_second_and_fraction(token: str, pos: int, token_len: int = -1) -> dict:
    """(cfunc) Parse a `seconds` token with an optional fractional part into (second, microsecond) `<'struct:sf'>`.

    This reads a decimal token starting at `pos` for whole `seconds`, optionally
    followed by a dot `.` and up to 6 fractional digits which are scaled to
    microseconds (truncated if more digits appear beyond 6). The function is
    intentionally strict and returns an invalid result for out-of-domain inputs.

    :param token `<'str'>`: The input token string containing `seconds` information.
    :param pos `<'int'>`: The starting position in the `token` of the `seconds`.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'struct:sf'>`: A small struct with fields:

        - `second`: parsed whole seconds (0..59) or -1 if invalid
        - `microsecond`: parsed fractional part in microseconds (0..999_999),
            or -1 if no fraction / value is invalid,

    ## Rules
    - Input format: `'SS'` or `'SS.f'` where `S` and `f` are ASCII digits.
    - Decimal separator: dot only `'.'`; comma is NOT accepted.
    - Fraction length: up to 6 digits kept; extra digits (if any) stop parsing
      early and are `ignored` (no rounding).
    - Seconds domain: 0..59 only. Leap-second `60` is not accepted.
    - Whitespace and other characters terminate parsing.
    - On any invalid input (e.g., no digits, seconds > 59, out-of-bounds indexes),
      returns `sf(second=-1, microsecond=-1)`.
    """

def scale_fraction_to_us(fraction: int, fraction_size: int) -> int:
    """(cfunc) Scale a fractional time component to microseconds based on its size `<'int'>`.

    This function takes a fractional time component (e.g., milliseconds, microseconds)
    and scales it to microseconds based on the number of digits provided. It supports
    fractions with sizes ranging from 1 to 6 digits.

    :param fraction `<'int'>`: The fractional time component to scale.
    :param fraction_size `<'int'>`: The number of digits in the fractional component.
        Must be in the range [1..6].
    :returns `<'int'>`: The scaled fractional time in microseconds.
        Returns `-1` if `fraction_size` is out of range.

    ## Example
    >>> scale_fraction_to_us(123, 3)
        # 123 milliseconds → 123000 microseconds
    """

# . slice and convert
def slice_to_uint(token: str, start: int, size: int, token_len=-1) -> int:
    """(cfunc) Slice a substring from a string and convert to an unsigned integer `<'int'>`.

    This function slices a portion of the input `token` string starting
    at 'start' and spanning 'size' of characters. The sliced substring is
    validated to ensure it contains only ASCII digits, before converting
    to unsigned integer.

    :param token `<'str'>`: The input token string to slice and convert.
    :param start `<'int'>`: The starting index for slicing the `token`.
    :param size `<'int'>`: The size of the slice.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'int'>`: The converted unsigned integer.
    :raises `<'ValueError'>`: When the slice is out of range, or contains non-digit characters.

    ## Notice
    - Only ASCII digits ('0' to '9') are accepted.

    ## Equivalent
    >>> int(data[start:start+size])  # without '+/-' signs
    """

def slice_to_ufloat(token: str, start: int, size: int, token_len=-1) -> int:
    """(cfunc) Slice a substring from a string and convert to a non-negative double-precision float `<'float'>`.

    This function slices a portion of the input `token` string starting
    at 'start' and spanning 'size' of characters. The sliced substring is
    then converted to a non-negative double-precision floating-point number.

    :param token `<'str'>`: The input token string to slice and convert.
    :param start `<'int'>`: The starting index for slicing the `token`.
    :param size `<'int'>`: The size of the slice.
    :param token_len `<'int'>`: Optional precomputed length of `token`. Defaults to `-1`.
        If `token_len <= 0`, the function computes the length.
        Otherwise, `token_len` is treated as the token length.
    :returns `<'float'>`: The converted non-negative double-precision float.
    :raises `<'ValueError'>`: When the slice is out of range, or cannot be converted to float.

    ## Notice
    - Only ASCII digits ('0' to '9') are accepted.
    - At most one decimal point ('.') is allowed.

    ## Equivalent
    >>> float(data[start:start+size])  # without '+/-' signs
    """

# Time ----------------------------------------------------------------------------------------------
# . gmtime
def tm_gmtime(ts: float) -> dict:
    """(cfunc) Convert a Unix timestamp to UTC calendar time `<'struct:tm'>`.

    :param ts `<'float'>`: Unix timestamp (seconds since the Unix Epoch).
        Fractional seconds are floored.
    :returns `<'struct:tm'>`: UTC calendar time.

        - tm.tm_sec   [0..59]
        - tm.tm_min   [0..59]
        - tm.tm_hour  [0..23]
        - tm.tm_mday  [1..31]
        - tm.tm_mon   [1..12]
        - tm.tm_year  [Gregorian year number]
        - tm.tm_wday  [0..6, 0=Monday]
        - tm.tm_yday  [1..366]
        - tm.tm_isdst [-1..1]

    ## Equivalent
    >>> time.gmtime(ts)
    """

def ts_gmtime(ts: float) -> int:
    """(cfunc) Convert a timestamp to UTC seconds ticks since the Unix Epoch `<'int'>`.

    :param ts `<'float'>`: Unix timestamp (seconds since the Unix Epoch).
        Fractional seconds are floored.
    :returns `<'int'>`: Integer in seconds since the Unix Epoch, representing the UTC time.
    """

# . localtime
def tm_localtime(ts: float) -> dict:
    """(cfunc) Convert a Unix timestamp to local calendar time `<'struct:tm'>`.

    :param ts `<'float'>`: Unix timestamp (seconds since the Unix Epoch).
        Fractional seconds are floored.
    :returns `<'struct:tm'>`: Local calendar time.

        - tm.tm_sec   [0..59]
        - tm.tm_min   [0..59]
        - tm.tm_hour  [0..23]
        - tm.tm_mday  [1..31]
        - tm.tm_mon   [1..12]
        - tm.tm_year  [Gregorian year number]
        - tm.tm_wday  [0..6, 0=Monday]
        - tm.tm_yday  [1..366]
        - tm.tm_isdst [-1..1]

    ## Equivalent
    >>> time.localtime(ts)
    """

def ts_localtime(ts: float) -> int:
    """(cfunc) Convert a timestamp to local seconds ticks since the Unix Epoch `<'int'>`.

    :param ts `<'float'>`: Unix timestamp (seconds since the Unix Epoch).
        Fractional seconds are floored.
    :returns `<'int'>`: Integer in seconds since the Unix Epoch, representing the local time.
    """

# . conversion
def sec_to_us(value: float) -> int:
    """(cfunc) Convert seconds (float) to microseconds (int) `<'int'>`.

    Decimal **truncate-towards-zero** at microsecond precision:
    the result is built from the fixed-point string with exactly
    6 fractional digits, ignoring any further digits.

    :param value `<'float'>`: Seconds.
    :returns `<'int'>`: Microseconds.

    ## Examples
    ```
    1.000001    -> 1,000,001  (6 fractional digits)
    1.23456789  -> 1,234,567  (digits beyond 6 are discarded)
    -0.0000005  -> 0          (truncate toward zero)
    ```
    """

def tm_strformat(t: object, fmt: str) -> str:
    """(cfunc) Format a calendar time `tm` using C `strftime`
    and return a Unicode string `<'str'>`.

    :param t `<'struct:tm'>`: Calendar time.

        - Expects a `tm` using the following conventions:

            - tm_year = absolute Gregorian year (e.g., 2025)
            - tm_mon  = 1..12
            - tm_wday = 0..6 (Mon..Sun)
            - tm_yday = 1..365/366

        - The fields are normalized internally back to POSIX conventions:

            - tm_year → years since 1900
            - tm_mon  → 0..11
            - tm_wday → 0..6 (Sun..Sat)
            - tm_yday → 0..365

    :param fmt `<'str'>`: strftime-compatible format string.
    :returns `<'str'>`: Formatted text.

    ## Notice
    - Output decoding assumes a UTF-8 C locale. If the process locale is not UTF-8,
      expansions of `%a`, `%A`, `%b`, `%B`, etc. may fail to decode.
    """

def tm_fr_us(value: int) -> dict:
    """(cfunc) Convert microsecond ticks since epoch to 'struct:tm' `<'struct:tm'>`.

    :param value `<'int'>`: The microsecond ticks since epoch.
    :returns `<'struct:tm'>`: The corresponding 'struct:tm' representation.

        - tm_sec   [0..59]
        - tm_min   [0..59]
        - tm_hour  [0..23]
        - tm_mday  [1..31]
        - tm_mon   [1..12]
        - tm_year  [Gregorian year number]
        - tm_wday  [0..6, 0=Monday]
        - tm_yday  [1..366]
        - tm_isdst [-1]
    """

def tm_fr_sec(value: float) -> dict:
    """(cfunc) Convert from seconds (float) epoch to 'struct:tm' `<'struct:tm'>`.

    :param value `<'float'>`: Seconds since Unix epoch.
    :returns `<'struct:tm'>`: The corresponding 'struct:tm' representation.

        - tm_sec   [0..59]
        - tm_min   [0..59]
        - tm_hour  [0..23]
        - tm_mday  [1..31]
        - tm_mon   [1..12]
        - tm_year  [Gregorian year number]
        - tm_wday  [0..6, 0=Monday]
        - tm_yday  [1..366]
        - tm_isdst [-1]
    """

def hmsf_fr_us(value: int) -> dict:
    """(cfunc) Extract time-of-day from microsecond ticks since epoch `<'struct:hmsf'>`.

    :param value `<'int'>`: Microseconds since Unix epoch.
        `LLONG_MIN` is treated as NaT (all fields -1).
    :returns `<'struct:hmsf'>`: The extracted time-of-day components.

        - hmsf.hour        [0..23]
        - hmsf.minute      [0..59]
        - hmsf.second      [0..59]
        - hmsf.microsecond [0..999_999].
    """

def hmsf_fr_sec(value: float) -> dict:
    """(cfunc) Extract time-of-day from seconds (float) since epoch `<'struct:hmsf'>`.

    :param value `<'float'>`: Seconds since Unix epoch.
    :returns `<'struct:hmsf'>`: The extracted time-of-day components.

        - hmsf.hour [0..23]
        - hmsf.minute [0..59]
        - hmsf.second [0..59]
        - hmsf.microsecond [0..999_999].
    """

# Calender ------------------------------------------------------------------------------------------
# . year
def is_leap_year(year: int) -> bool:
    """(cfunc) Determine whether `year` is a leap year under
    the proleptic Gregorian rules `<'bool'>`.

    :param year `<'int'>`: Gregorian year number.
    :returns `<'bool'>`: True if `year` is a leap year, else False.
    """

def is_long_year(year: int) -> bool:
    """(cfunc) Determine whether `year` is a “long year” (has ISO week 53)
    under the proleptic Gregorian rules `<'bool'>`.

    A year has ISO week 53 **if** January 1 is a Thursday, or January 1 is a
    Wednesday **and** the year is a leap year (ISO-8601 week rules).

    :param year `<'int'>`: Gregorian year number.
    :returns `<'bool'>`: True if `year` is a long year, else False.
    """

def leap_years(year: int, inclusive: bool) -> int:
    """(cfunc) Count leap years between `year` and `0001-01-01` under
    the proleptic Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param inclusive `<'bool'>`: If True, include `year` itself when leap.

        - `False` strictly before Jan 1 of `year` (… < y < year)
        - `True`  up to and including `year`      (… < y ≤ year)

    :returns `<'int'>`: Count of leap years up to the boundary
        (negative when the boundary is earlier than 0001-01-01).
    """

def leaps_bt_years(year1: int, year2: int) -> int:
    """(cfunc) Compute the total number of Gregorian leap years between `year1` and `year2`
    under the proleptic Gregorian rules `<'int'>`.

    :param year1 `<'int'>`: First gregorian year.
    :param year2 `<'int'>`: Second gregorian year.
    :returns `<'int'>`: Number of leap years between the two years.
    """

def days_in_year(year: int) -> int:
    """(cfunc) Determine the number of days in `year` under the
    proleptic Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :returns `<'int'>`: 366 for leap years, else 365.
    """

def days_bf_year(year: int) -> int:
    """(cfunc) Compute the number of days strictly before `January 1` of `year`
    under the proleptic Gregorian rules `<'int'>`.

    This is the signed offset (in days) from `0001-01-01` to `year-01-01`:

        - days_bf_year(1) == 0
        - days_bf_year(0) == -366  (year 0 ≡ 1 BCE, leap)
        - Negative values indicate dates before '0001-01-01'.

    :param year `<'int'>`: Gregorian year number.
    :returns `<'int'>`: Signed count of days before Jan-1 of `year`.
    """

def day_of_year(year: int, month: int, day: int) -> int:
    """(cfunc) Compute the `1-based` ordinal day-of-year for the given Y/M/D
    under the proleptic Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :param day `<'int'>`: Day of month.
        Automatically clamps to the valid range for the (clamped) month and year.
    :returns `<'int'>`: `1..365/366` — ordinal day of year (Jan-01 → 1).
    """

# . quarter
def quarter_of_month(month: int) -> int:
    """(cfunc) Return the calendar quarter index (1..4) for `month`
    under the Gregorian calendar `<'int'>`.

    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :returns `<'int'>`: Quarter number in 1..4
        (Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec).
    """

def days_in_quarter(year: int, month: int) -> int:
    """(cfunc) Return the number of days in calendar quarter containing
    `month` in `year` under the proleptic Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :returns `<'int'>`: Number of days in the quarter.

        - Non-leap [Q1=90, Q2=91, Q3=92, Q4=92]
        - Leap     [Q1=91, Q2=91, Q3=92, Q4=92]
    """

def days_bf_quarter(year: int, month: int) -> int:
    """(cfunc) Return the number of days strictly before the first day of the
    calendar quarter containing `month` in `year` under the proleptic
    Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :returns `<'int'>`: Days before the quarter start.

        - Non-leap [Q1=0, Q2=90, Q3=181, Q4=273]
        - Leap     [Q1=0, Q2=91, Q3=182, Q4=274]
    """

def day_of_quarter(year: int, month: int, day: int) -> int:
    """(cfunc) Compute the number of days between the 1st day of the quarter and the
    given Y/M/D under the proleptic Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :param day `<'int'>`: Day of month.
        Automatically clamps to the valid range for the (clamped) month/year.
    :returns `<'int'>`: 0-based days since the quarter start.
    """

# . month
def days_in_month(year: int, month: int) -> int:
    """(cfunc) Return the number of days in `month` of `year` under
    the proleptic Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :returns `<'int'>`: Number of days in the month (28-31).
    """

def days_bf_month(year: int, month: int) -> int:
    """(cfunc) Return the number of days strictly before the first day of `month`
    in `year` under the proleptic Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :returns `<'int'>`: Days before `year-month-01` (Jan→0, Feb→31, Mar→59/60, ...).
    """

# . week
def ymd_weekday(year: int, month: int, day: int) -> int:
    """(cfunc) Compute the weekday for Y/M/D (`0=Mon … 6=Sun`) under
    the proleptic Gregorian rules `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :param day `<'int'>`: Day of month.
        Automatically clamps to the valid range for the (clamped) month/year.
    :returns `<'int'>`: Weekday number in 0..6 (Mon=0).
    """

# . iso
def ymd_isocalendar(year: int, month: int, day: int) -> dict:
    """(cfunc) Compute the ISO calendar from the Gregorian date Y/M/D `<'struct:iso'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :param day `<'int'>`: Day of month.
        Automatically clamps to the valid range for the (clamped) month/year.
    :returns `<'struct:iso'>`: The ISO calendar components.

        - iso.year    [ISO calendar year]
        - iso.week    [1..52/53]
        - iso.weekday [1..7 (Mon..Sun)]
    """

def ymd_isoyear(year: int, month: int, day: int) -> int:
    """(cfunc) Compute the ISO calendar `year` from the Gregorian date Y/M/D `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :param day `<'int'>`: Day of month.
        Automatically clamps to the valid range for the (clamped) month/year.
    :returns `<'int'>`: The ISO calendar year.
    """

def ymd_isoweek(year: int, month: int, day: int) -> int:
    """(cfunc) Compute the ISO calendar `week` from the Gregorian date Y/M/D `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :param day `<'int'>`: Day of month.
        Automatically clamps to the valid range for the (clamped) month/year.
    :returns `<'int'>`: The ISO calendar week (1..52/53).
    """

def ymd_isoweekday(year: int, month: int, day: int) -> int:
    """(cfunc) Compute the ISO calendar `weekday` from the Gregorian date Y/M/D `<'int'>`.

    :param year `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :param day `<'int'>`: Day of month.
        Automatically clamps to the valid range for the (clamped) month/year.
    :returns `<'int'>`: The ISO calendar weekday (1=Mon...7=Sun).
    """

# . Y/M/D
def ymd_to_ord(year: int, month: int, day: int) -> int:
    """(cfunc) Convert a Gregorian Y/M/D to a **1-based** ordinal day under
    the proleptic Gregorian rules `<'int'>`.

    - Astronomical year numbering (year 0 == 1 BCE).
    - Result is **1-based** with `0001-01-01 -> 1`.
      Dates before `0001-01-01` yield `<= 0`.

    :param year  `<'int'>`: Gregorian year number.
    :param month `<'int'>`: Month number in the range 1..12.
        Automatically clamps out-of-range values to [1..12].
    :param day `<'int'>`: Day of month.
        Automatically clamps to the valid range for the (clamped) month/year.
    :returns `<'int'>`: Ordinal day count (1-based).
    """

def ymd_fr_ord(value: int) -> dict:
    """(cfunc) Convert Gregorian ordinal day to Y/M/D `<'struct:ymd'>`.

    :param value `<'int'>`: Gregorian ordinal day.
    :returns `<'struct:ymd'>`: Converted Y/M/D components.

        - ymd.year  [Gregorian year number]
        - ymd.month [1..12]
        - ymd.day   [1..31]
    """

def ymd_fr_us(value: int) -> dict:
    """(cfunc) Convert microsecond ticks since the Unix epoch to Y/M/D `<'struct:ymd'>`.

    :param value `<'int'>`: The microsecond ticks since epoch.
    :returns `<'struct:ymd'>`: Converted Y/M/D components.

        - ymd.year  [Gregorian year number]
        - ymd.month [1..12]
        - ymd.day   [1..31]
    """

def ymd_fr_sec(value: float) -> dict:
    """(cfunc) Convert seconds (float) since the Unix epoch o Y/M/D `<'struct:ymd'>`.

    :param value `<'float'>`: Seconds since Unix epoch.
    :returns `<'struct:ymd'>`: Converted Y/M/D components.

        - ymd.year  [Gregorian year number]
        - ymd.month [1..12]
        - ymd.day   [1..31]
    """

def ymd_fr_iso(year: int, week: int, weekday: int) -> dict:
    """(cfunc) Create `struct:ymd` from ISO calendar values
    (ISO year, ISO week, ISO weekday) `<'struct:ymd'>`.

    :param year `<'int'>`: ISO year number.
    :param week `<'int'>`: ISO week number.
        Automatically clamped to [1..52/53] (depends on whether is a long year).
    :param weekday `<'int'>`: ISO weekday.
        Automatically clamped to [1=Mon..7=Sun].
    :returns `<'struct:ymd'>`: Gregorian Y/M/D in the proleptic Gregorian calendar.

        - ymd.year  [Gregorian year number]
        - ymd.month [1..12]
        - ymd.day   [1..31]
    """

def ymd_fr_doy(year: int, doy: int) -> dict:
    """(cfunc) Create `struct:ymd` from Gregorian year and day-of-year `<'struct:ymd'>`.

    :param year `<'int'>`: Gregorian year number.
    :param doy `<'int'>`: The day-of-year.
        Automatically clamped to [1..365/366] (depends on whether is a long year).
    :returns `<'struct:ymd'>`: Gregorian Y/M/D in the proleptic Gregorian calendar.

        - ymd.year  [Gregorian year number]
        - ymd.month [1..12]
        - ymd.day   [1..31]
    """

def iso_week1_mon_ord(year: int) -> int:
    """(cfunc) Return the ordinal (1-based) of the Monday starting ISO week 1
    of `year` under the proleptic Gregorian rules `<'int'>`.

    ISO week 1 is the week that contains January 4 (equivalently, the
    week whose Thursday lies in `year`). This returns the Gregorian
    ordinal day number (0001-01-01 = 1) of that week's Monday.

    :param year `<'int'>`: Gregorian year number.
    :returns `<'int'>`: Ordinal of the Monday starting ISO week 1.
    """

# . date & time
def dtm_fr_us(value: int) -> dict:
    """(cfunc) Convert microsecond ticks since the Unix epoch to `struct:dtm` (date + time).

    :param value `<'int'>`: The microsecond ticks since epoch.
    :returns `<'struct:dtm'>`: Converted date/time components.

        - dtm.year          [Gregorian year number]
        - dtm.month         [1..12]
        - dtm.day           [1..31]
        - dtm.hour          [0..23]
        - dtm.minute        [0..59]
        - dtm.second        [0..59]
        - dtm.microsecond   [0..999999]
    """

def dtm_fr_sec(value: float) -> dict:
    """(cfunc) Convert seconds (float) since the Unix epoch to `struct:dtm` (date + time).

    :param value `<'float'>`: Seconds since Unix epoch.
    :returns `<'struct:dtm'>`: Converted date/time components.

        - dtm.year          [Gregorian year number]
        - dtm.month         [1..12]
        - dtm.day           [1..31]
        - dtm.hour          [0..23]
        - dtm.minute        [0..59]
        - dtm.second        [0..59]
        - dtm.microsecond   [0..999999]
    """

# . fractions
def combine_absolute_ms_us(ms: int, us: int) -> int:
    """(cfunc) Combine milliseconds and microseconds into total microseconds (replacement semantics) `<'int'>`.

    :param ms `<'int'>`: Absolute milliseconds (field replacement). Negative means `no change`.
    :param us `<'int'>`: Absolute microseconds (field replacement). Negative means `no change`.
    :returns `<'int'>`: Absolute microseconds in [0, 999_999], or `-1` if both inputs are negative.

    ## Rules
    - Negative input means `no change` for that field.
    - If BOTH ms and us are negative: return `-1` (caller should keep existing value).
    - If ms >= 0:
        * ms sets the millisecond field (clamped to 0..999).
        * us, if >= 0, sets the sub-millisecond remainder (ignored above 999 via `% 1000`).
        * Thousands in `us` are ignored (ms has higher priority).
          => return ms * 1000 + (us % 1000) in [0, 999_999]
    - Else (ms < 0 and us >= 0):
        * us directly sets absolute microseconds (clamped to 0..999_999).
    """

# datetime.date -------------------------------------------------------------------------------------
# . generate
def date_new(
    year: int = 1,
    month: int = 1,
    day: int = 1,
    dclass: type[datetime.date] | None = None,
) -> datetime.date:
    """(cfunc) Create a new date `<'datetime.date'>`.

    :param year  `<'int'>`: Gregorian year number. Defaults to `1`.
    :param month `<'int'>`: Month [1..12]. Defaults to `1`.
    :param day   `<'int'>`: Day [1..31]. Defaults to `1`.
    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.
    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified).
    """

def date_now(
    tzinfo: datetime.tzinfo | None = None,
    dclass: type[datetime.date] | None = None,
) -> datetime.date:
    """(cfunc) Get today's date `<'datetime.date'>`.

    :param tzinfo `<'tzinfo/None'>`: Optional timezone. Defaults to `None`.

        - If specified, return the current date in that timezone.
        - Otherwise, return the current local date (equivalent to `datetime.date.today()`).

    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.

    :returns `<'datetime.date'>`: Today's date (or subclass if `dclass` is specified).
    """

# . type check
def is_date(obj: object) -> bool:
    """(cfunc) Check if an object is an instance or subclass of `datetime.date` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> isinstance(obj, datetime.date)
    """

def is_date_exact(obj: object) -> bool:
    """(cfunc) Check if an object is an exact `datetime.date` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> type(obj) is datetime.date
    """

# . conversion: to
def date_strformat(date: datetime.date, fmt: str) -> str:
    """(cfunc) Format a `datetime.date` with a strftime-style format `<'str'>`.

    :param date `<'datetime.date'>`: Date to format.
    :param fmt  `<'str'>`: Strftime-compatible format string.
    :returns `<'str'>`: Formatted text.

    ## Notice
    - Output decoding assumes a UTF-8 C locale.

    ## Equivalent
    >>> date.strftime(fmt)
    """

def date_isoformat(date: datetime.date) -> str:
    """Format `datetime.date` to the ISO-8601 calendar string `YYYY-MM-DD` `<'str'>`.

    :param date `<'datetime.date'>`: Date to format.
    :returns `<'str'>`: ISO string in the form `YYYY-MM-DD`.
    """

def date_to_tm(date: datetime.date) -> dict:
    """(cfunc) Convert `datetime.date` to `<'struct:tm'>`.

    :param date `<'datetime.date'>`: The date object to convert.
    :returns `<'struct:tm'>`: The corresponding 'struct:tm' representation.

        - tm.tm_sec   [0]
        - tm.tm_min   [0]
        - tm.tm_hour  [0]
        - tm.tm_mday  [1..31]
        - tm.tm_mon   [1..12]
        - tm.tm_year  [Gregorian year number]
        - tm.tm_wday  [0..6, 0=Monday]
        - tm.tm_yday  [1..366]
        - tm.tm_isdst [-1]
    """

def date_to_us(date: datetime.date) -> int:
    """(cfunc) Convert a `datetime.date` to microseconds since the
    Unix epoch (UTC midnight) `<'int64'>`.

    :param date `<'datetime.date'>`: Date to convert.
    :returns `<'int64'>`: Microseconds from epoch to the start of `date`.
        Negative for dates before 1970-01-01.
    """

def date_to_sec(date: datetime.date) -> float:
    """(cfunc) Convert a `datetime.date` to seconds since the
    Unix epoch (UTC midnight) `<'float'>`.

    :param date `<'datetime.date'>`: Date to convert.
    :returns `<'float'>`: Seconds from epoch to the start of `date`.
        Negative for dates before 1970-01-01.
    """

def date_to_ord(date: datetime.date) -> int:
    """(cfunc) Convert a `datetime.date` to the Gregorian ordinal (0001-01-01 = 1) `<'int'>`.

    :param date `<'datetime.date'>`: Date to convert.
    :returns `<'int'>`: Ordinal day number in the proleptic Gregorian calendar.
    """

# . conversion: from
def date_fr_us(
    value: int,
    dclass: type[datetime.date] | None = None,
) -> datetime.date:
    """(cfunc) Create date from microseconds since the Unix epoch `<'datetime.date'>`.

    :param value `<'int'>`: Microseconds since epoch.
    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.
    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified).
    """

def date_fr_sec(
    value: float,
    dclass: type[datetime.date] | None = None,
) -> datetime.date:
    """(cfunc) Create date from seconds since the Unix epoch `<'datetime.date'>`.

    :param value `<'float'>`: Seconds since epoch.
    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.
    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified).
    """

def date_fr_ord(
    value: int,
    dclass: type[datetime.date] | None = None,
) -> datetime.date:
    """(cfunc) Create date from a Gregorian ordinal (0001-01-01 = 1) `<'datetime.date'>`.

    :param value `<'int'>`: Gregorian ordinal day.
    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.
    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified).
    """

def date_fr_iso(
    year: int,
    week: int,
    weekday: int,
    dclass: type[datetime.date] | None = None,
) -> datetime.date:
    """(cfunc) Create date from ISO calendar values (year, week, weekday) `<'datetime.date'>`.

    :param year `<'int'>`: ISO year number.
    :param week `<'int'>`: ISO week number.
        Automatically clamped to [1..52/53] (depends on whether is a long year).
    :param weekday `<'int'>`: ISO weekday.
        Automatically clamped to [1=Mon..7=Sun].
    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.
    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified).
    """

def date_fr_doy(
    year: int,
    doy: int,
    dclass: type[datetime.date] | None = None,
) -> datetime.date:
    """(cfunc) Create date from Gregorian year and day-of-year `<'datetime.date'>`.

    :param year `<'int'>`: Gregorian year number.
    :param doy `<'int'>`: The day-of-year.
        Automatically clamped to [1..365/366] (depends on whether is a long year).
    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.
    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified).
    """

def date_fr_ts(
    value: float,
    dclass: type[datetime.date] | None = None,
) -> datetime.date:
    """(cfunc) Create date from a POSIX timestamp in **local** time `<'datetime.date'>`.

    :param value `<'float'>`: POSIX timestamp in **local** time.
    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.
    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified)
        in the system **local** timezone.
    """

def date_fr_date(
    date: datetime.date,
    dclass: type[datetime.date] | None = None,
) -> datetime.date:
    """(cfunc) Create date from another date (or subclass) `<'datetime.date'>`.

    :param date `<'datetime.date'>`: The source date (including subclasses).
    :param dclass `<'type[datetime.date]/None'>`: Target date class. Defaults to `None`.
        If `None` set to python's built-in `datetime.date`.
        If `date` is already of type `dclass`, returns `date` directly.
    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified)
        with the same date fields.
    """

def date_fr_dt(
    dt: datetime.datetime,
    dclass: type[datetime.date] | None = None,
) -> datetime.date:
    """(cfunc) Create date from a datetime (include subclass) `<'datetime.date'>`.

    :param dt `<'datetime.datetime'>`: Datetime to extract the date from (including subclasses).
    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.
    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified)
        with the same date fields.
    """

# . manipulation
def date_replace(
    date: datetime.date,
    year: int = -1,
    month: int = -1,
    day: int = -1,
    dclass: type[datetime.date] | None = None,
) -> datetime.date:
    """(cfunc) Replace specified fields in date `<'datetime.date'>`.

    :param date `<'datetime.date'>`: The source date.
    :param year `<'int'>`: Absolute year. Defaults to `-1` (no change).
        If specified (greater than `0`), clamps to [1..9999].
    :param month `<'int'>`: Absolute month. Defaults to `-1` (no change).
        If specified (greater than `0`), clamps to [1..12].
    :param day `<'int'>`: Absolute day. Defaults to `-1` (no change).
        If specified (greater than `0`), clamps to [1..maximum days the resulting month].
    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.
    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified)
        after applying the field replacements.
    """

def date_add_delta(
    date: datetime.date,
    years: int = 0,
    quarters: int = 0,
    months: int = 0,
    weeks: int = 0,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
    milliseconds: int = 0,
    microseconds: int = 0,
    year: int = -1,
    month: int = -1,
    day: int = -1,
    weekday: int = -1,
    hour: int = -1,
    minute: int = -1,
    second: int = -1,
    millisecond: int = -1,
    microsecond: int = -1,
    dclass: type[datetime.date] | None = None,
) -> datetime.date:
    """(cfunc) Add relative and absolute deltas to date `<'datetime.date'>`.

    ## Absolute Deltas (Replace specified fields)

    :param year `<'int'>`: Absolute year. Defaults to `-1` (no change).
    :param month `<'int'>`: Absolute month. Defaults to `-1` (no change).
    :param day `<'int'>`: Absolute day. Defaults to `-1` (no change).
    :param weekday `<'int'>`: Absolute weekday (0=Mon...6=Sun). Defaults to `-1` (no change).
    :param hour `<'int'>`: Absolute hour. Defaults to `-1` (no change).
    :param minute `<'int'>`: Absolute minute. Defaults to `-1` (no change).
    :param second `<'int'>`: Absolute second. Defaults to `-1` (no change).
    :param millisecond `<'int'>`: Absolute millisecond. Defaults to `-1` (no change).
        Overrides `microsecond` milliseoncd part if both are provided.
    :param microsecond `<'int'>`: Absolute microsecond. Defaults to `-1` (no change).

    ## Relative Deltas (Add to specified fields)

    :param years `<'int'>`: Relative years. Defaults to `0`.
    :param quarters `<'int'>`: Relative quarters (3 months). Defaults to `0`.
    :param months `<'int'>`: Relative months. Defaults to `0`.
    :param weeks `<'int'>`: Relative weeks (7 days). Defaults to `0`.
    :param days `<'int'>`: Relative days. Defaults to `0`.
    :param hours `<'int'>`: Relative hours. Defaults to `0`.
    :param minutes `<'int'>`: Relative minutes. Defaults to `0`.
    :param seconds `<'int'>`: Relative seconds. Defaults to `0`.
    :param milliseconds `<'int'>`: Relative milliseconds (1,000 us). Defaults to `0`.
    :param microseconds `<'int'>`: Relative microseconds. Defaults to `0`.

    ## Date Class

    :param dclass `<'type[datetime.date]/None'>`: Optional custom date class. Defaults to `None`.
        if `None` uses python's built-in `datetime.date` as the constructor.

    :returns `<'datetime.date'>`: The resulting date (or subclass if `dclass` is specified)
        after applying the specified deltas.
    """

# datetime.datetime ---------------------------------------------------------------------------------
# . generate
def dt_new(
    year: int = 1,
    month: int = 1,
    day: int = 1,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
    tzinfo: datetime.tzinfo | None = None,
    fold: int = 0,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Create a new datetime `<'datetime.datetime'>`.

    :param year  `<'int'>`: Gregorian year number. Defaults to `1`.
    :param month `<'int'>`: Month [1..12]. Defaults to `1`.
    :param day   `<'int'>`: Day [1..31]. Defaults to `1`.
    :param hour `<'int'>`: Hour [0..23]. Defaults to `0`.
    :param minute `<'int'>`: Minute [0..59]. Defaults to `0`.
    :param second `<'int'>`: Second [0..59]. Defaults to `0`.
    :param microsecond `<'int'>`: Microsecond [0..999999]. Defaults to `0`.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone. Defaults to `None`.
    :param fold `<'int'>`: Optional fold flag for ambiguous times (0 or 1). Defaults to `0`.
        Values other than `1` are treated as `0`.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).
    """

def dt_now(
    tzinfo: datetime.tzinfo | None = None,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Get the current datetime `<'datetime.datetime'>`.

    :param tzinfo `<'tzinfo/None'>`: Optional timezone. Defaults to `None`.

        - If specified, return an aware datetime in that timezone.
        - Otherwise, return a naive local-time datetime.

    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.

    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).

    ## Equivalent
    >>> datetime.datetime.now(tzinfo)
    """

# . type check
def is_dt(obj: object) -> bool:
    """(cfunc) Check if an object is an instance or subclass of `datetime.datetime` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> isinstance(obj, datetime.datetime)
    """

def is_dt_exact(obj: object) -> bool:
    """(cfunc) Check if an object is an exact `datetime.datetime` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> type(obj) is datetime.datetime
    """

# . conversion: to
def dt_local_mktime(dt: datetime.date) -> int:
    """(cfunc) Interpret `dt` as a naive local POSIX timestamp `<'int'>`.

    Mirrors CPython's naive branch (uses the **system** local timezone database),
    handling ambiguous/nonexistent local times via `dt.fold`. Any `tzinfo` attached
    to `dt` is ignored, because the function is designed for naive datetime only.

    :param dt `<'datetime.datetime'>`: Timezone-naive datetime only.
    :returns `<'int'>`: Local POSIX timestamp in seconds.
    """

def dt_strformat(dt: datetime.datetime, fmt: str) -> str:
    """(cfunc) Format a `datetime.datetime` with a strftime-style format `<'str'>`.

    :param dt `<'datetime.datetime'>`: Datetime to format.
    :param fmt `<'str'>`: Strftime-compatible format string.
    :returns `<'str'>`: Formatted text.

    ## Equivalent
    >>> dt.strftime(fmt)
    """

def dt_isoformat(dt: datetime.datetime, sep: str = "T", utc: bool = False) -> str:
    """(cfunc) Convert a `datetime.datetime` to ISO string `<'str'>`.

    :param dt `<'datetime.datetime'>`: The datetime to convert.
    :param sep `<'str'>`: Date/time separator. Defaults to `"T"`.
    :param utc <'bool'>: Whether to append a UTC offset. Defaults to False.

        - If False or `dt` is timezone-naive, UTC offset is ignored.
        - If True and `dt` is timezone-aware, append UTC offset (e.g., +0530).

    :returns `<'str'>`: Formatted text.
    """

def dt_to_tm(dt: datetime.datetime, utc: bool = False) -> dict:
    """(cfunc) Convert a `datetime.datetime` to 'struct:tm' `<'struct:tm'>`.

    :param dt `<'datetime.datetime'>`: Datetime to convert.
    :param utc `<'bool'>`: Whether to subtract UTC offset before conversion. Defaults to `False`.

        - If True and `dt` is timezone-aware, subtract `utcoffset(dt)`
          (convert to UTC) and set `tm_isdst = 0`.
        - If True and `dt` is naive, treat it as UTC (no offset), with
          `tm_isdst = 0`.
        - If False, interpret fields as local wall time; for aware
          datetimes set `tm_isdst > 0` iff `dst(dt)` is non-zero,
          else 0; for naive set -1.

    :returns `<'struct:tm'>`: Fields in project conventions:

        - tm.tm_sec   [0..59]
        - tm.tm_min   [0..59]
        - tm.tm_hour  [0..23]
        - tm.tm_mday  [1..31]
        - tm.tm_mon   [1..12]
        - tm.tm_year  [Gregorian year number]
        - tm.tm_wday  [0..6, 0=Monday]
        - tm.tm_yday  [1..366]
        - tm.tm_isdst [-1..1]
    """

def dt_to_ctime(dt: datetime.datetime) -> str:
    """(cfunc) Convert a datetime to a C `ctime-style` string `<'str'>`.

    :param dt `<'datetime.datetime'>`: Datetime to convert.
    :returns `<'str'>`: C time string.

    ## Example
    >>> dt_to_ctime(datetime.datetime(2024, 10, 1, 8, 19, 5))
    >>> 'Tue Oct  1 08:19:05 2024'
    """

def dt_to_us(dt: datetime.datetime, utc: bool = False) -> int:
    """(cfunc) Convert a `datetime.datetime` to microseconds since the Unix epoch `<'int'>`.

    :param dt `<'datetime.datetime'>`: Datetime to convert (naive or aware).
    :param utc `<'bool'>`: Whether to subtract UTC offset before conversion. Defaults to `False`.

        - If False or `dt` is naive, return the total microseconds
          without adjustment (UTC offset ignored).
        - If True and `dt` is timezone-aware, subtract its UTC offset
          (i.e., convert to UTC) from the total microseconds.

    :returns `<'int'>`: Microseconds since the Unix epoch.
    """

def dt_to_sec(dt: datetime.datetime, utc: bool = False) -> float:
    """(cfunc) Convert a `datetime.datetime` to seconds since the Unix epoch `<'float'>`.

    :param dt `<'datetime.datetime'>`: Datetime to convert (naive or aware).
    :param utc `<'bool'>`: Whether to subtract UTC offset before conversion. Defaults to `False`.

        - If False or `dt` is naive, return the total seconds
          without adjustment (UTC offset ignored).
        - If True and `dt` is timezone-aware, subtract its UTC offset
          (i.e., convert to UTC) from the total seconds.

    :returns `<'float'>`: Seconds since the Unix epoch.
    """

def dt_to_ord(dt: datetime.datetime, utc: bool = False) -> int:
    """(cfunc) Convert a `datetime.datetime` to its Gregorian ordinal day `<'int'>`.

    :param dt `<'datetime.datetime'>`: Datetime to convert (naive or aware).
    :param utc `<'bool'>`: Whether to subtract UTC offset before conversion. Defaults to `False`.

        - If False or `dt` is naive, return the ordinal without adjustment
          (UTC offset ignored).
        - If True and `dt` is timezone-aware, adjust by the UTC offset and
          decrement/increment the ordinal by 1 when the UTC wall clock falls
          before 00:00 or on/after 24:00.

    :returns `<'int'>`: Gregorian ordinal day (0001-01-01 == 1).
    """

def dt_to_ts(dt: datetime.datetime) -> float:
    """(cfunc) Convert a `datetime.datetime` to POSIX timestamp (seconds since the Unix epoch) `<'float'>`.

    :param dt `<'datetime.datetime'>`: Datetime to convert (naive or aware).

        - Naive: interpret as local time (like CPython: mktime-style + microseconds)
        - Aware: subtract utcoffset at that wall time.

    :returns `<'float'>`: POSIX timestamp in seconds.

    ## Equivalent
    >>> dt.timestamp()
    """

def dt_as_epoch(dt: datetime.datetime, unit: str, utc: bool = False) -> int:
    """(cfunc) Convert a `datetime.datetime` to an integer count since the Unix epoch `<'int'>`,

    :param dt `<'datetime.datetime'>`: Datetime to convert (naive or aware).
    :param as_unit `<'str'>`: Output unit, supports:

        - 'Y'  (years since 1970)         — calendar-based index
        - 'Q'  (quarters since 1970Q1)    — calendar-based index
        - 'M'  (months since 1970-01)     — calendar-based index
        - 'W'  (weeks since 1970-01-01)   — floor division by 7 days
        - 'D'  (days since 1970-01-01)
        - 'h'  (hours since epoch)
        - 'm'  (minutes since epoch)
        - 's'  (seconds since epoch)
        - 'ms' (milliseconds since epoch)
        - 'us' (microseconds since epoch)

    :param utc `<'bool'>`: Whether to subtract UTC offset before conversion. Defaults to `False`.

        - If False or `dt` is naive, interpret the fields without
          adjustment (UTC offset ignored).
        - If True and `dt` is timezone-aware, subtract its UTC offset
          (i.e., convert to UTC) before converting.

    :returns `<'int'>`: Epoch count in the requested unit.

    ## Semantics
    - Calendar units ('Y','Q','M') are computed as simple calendar indexes relative
      to 1970-01-01 (no leap/length normalization).
    - 'W' uses mathematical floor division of days by 7 (weeks start aligned to the epoch).
    - Subsecond units:

        * 'ms' truncates microseconds toward zero (i.e., floor for non-negative datetimes).
        * 'us' reproduces the exact microsecond field.
    """

def dt_as_epoch_W_iso(dt: datetime.datetime, weekday: int, utc: bool = False) -> int:
    """Convert a `datetime.datetime` to an integer count of whole
    weeks since the Unix epoch, where weeks are aligned to a chosen
    ISO weekday (1=Mon..7=Sun) `<'int'>`.

    :param dt `<'datetime.datetime'>`: Datetime to convert (naive or aware).
    :param weekday `<'int'>`: ISO weekday that defines the *start* of each week bucket
        (1=Mon .. 7=Sun). Values outside 1..7 are clamped.
    :param utc `<'bool'>`: Whether to subtract UTC offset before conversion. Defaults to `False`.

        - If False or `dt` is naive, interpret the fields without
          adjustment (UTC offset ignored).
        - If True and `dt` is timezone-aware, subtract its UTC offset
          (i.e., convert to UTC) before converting.

    :returns `<'int'>`: Number of completed weeks since 1970-01-01 with the given alignment.

    ## Notice
    - 1970-01-01 is Thursday (ISO=4). When `weekday=4`, this matches the default
      Thursday alignment. When `weekday=1`, weeks are Monday-aligned, etc.
    - Uses mathematical floor division so dates before the epoch are handled correctly.
    """

# . conversion: from
def dt_fr_us(
    value: int,
    tzinfo: datetime.tzinfo | None = None,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Create datetime from microseconds since the Unix epoch `<'datetime.datetime'>`.

    :param value `<'int'>`: Microseconds since epoch.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone to **attach**. Defaults to `None`.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).
    """

def dt_fr_sec(
    value: float,
    tzinfo: datetime.tzinfo | None = None,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Create datetime from seconds since the Unix epoch `<'datetime.datetime'>`.

    :param value `<'float'>`: Seconds since epoch.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone to **attach**. Defaults to `None`.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).
    """

def dt_fr_ord(
    value: int,
    tzinfo: datetime.tzinfo | None = None,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Create datetime from a Gregorian ordinal (0001-01-01 = 1) `<'datetime.datetime'>`.

    :param value `<'int'>`: Gregorian ordinal day.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone to **attach**. Defaults to `None`.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).
    """

def dt_fr_iso(
    year: int,
    week: int,
    weekday: int,
    tzinfo: datetime.tzinfo | None = None,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Create datetime from ISO calendar values (year, week, weekday) `<'datetime.datetime'>`.

    :param year `<'int'>`: ISO year number.
    :param week `<'int'>`: ISO week number.
        Automatically clamped to [1..52/53] (depends on whether is a long year).
    :param weekday `<'int'>`: ISO weekday.
        Automatically clamped to [1=Mon..7=Sun].
    :param tzinfo `<'tzinfo/None'>`: Optional timezone to **attach**. Defaults to `None`.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).
    """

def dt_fr_doy(
    year: int,
    doy: int,
    tzinfo: datetime.tzinfo | None = None,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Create datetime from Gregorian year and day-of-year `<'datetime.datetime'>`.

    :param year `<'int'>`: Gregorian year number.
    :param doy `<'int'>`: The day-of-year.
        Automatically clamped to [1..365/366] (depends on whether is a long year).
    :param tzinfo `<'tzinfo/None'>`: Optional timezone to **attach**. Defaults to `None`.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).
    """

def dt_fr_ts(
    value: float,
    tzinfo: datetime.tzinfo | None = None,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Create datetime from a POSIX timestamp with optional timezone `<'datetime.datetime'>`.

    :param value `<'float'>`: POSIX timestamp.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone. Defaults to `None`.

        - If specified, return an aware datetime in that timezone.
        - Otherwise, return a naive local-time datetime.

    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).

    ## Equivalent
    >>> datetime.datetime.fromtimestamp(value, tzinfo)
    """

def dt_fr_date(
    date: datetime.date,
    tzinfo: datetime.tzinfo | None = None,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Create datetime from date (include subclass) `<'datetime.datetime'>`.

    :param date `<'datetime.date'>`: The source date (including subclasses).
    :param tzinfo `<'tzinfo/None'>`: Optional timezone to **attach**. Defaults to `None`.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified)
        with the same date fields (time fields are set to 0).
    """

def dt_fr_dt(
    dt: datetime.datetime,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Create datetime from another datetime (include subclass) `<'datetime.datetime'>`.

    :param dt `<'datetime.datetime'>`: The source datetime (including subclasses).
    :param dtclass `<'type[datetime.datetime]/None'>`: Target datetime class. Defaults to `None`.
        If `None` set to python's built-in `datetime.datetime`.
        If `dt` is already of type `dtclass`, returns `dt` directly.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified)
        with the same fields and tzinfo.
    """

def dt_fr_time(
    time: datetime.time,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Create datetime from time (include subclass) `<'datetime.datetime'>`.

    :param time `<'datetime.time'>`: The source time (including subclasses).
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified)
        with the same time fields and tzinfo (date fields are set to 1970-01-01).
    """

def dt_combine(
    date: datetime.date | None = None,
    time: datetime.time | None = None,
    tzinfo: datetime.tzinfo | None = None,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Create a `datetime.datetime` by combining a date and a time `<'datetime.datetime'>`.

    :param date `<'datetime.date/None'>`: The source date (including subclasses). Defaults to `None`.
        If None, uses today's `local` date.
    :param time `<'datetime.time/None'>`: The source time (including subclasses). Defaults to `None`.
        If None, uses `00:00:00.000000`.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone to **attach**. Defaults to `None`.
        If specifed, overrides `time.tzinfo` (if any).
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).
    """

# . manipulation
def dt_add(
    dt: datetime.datetime,
    days: int = 0,
    seconds: int = 0,
    microseconds: int = 0,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Add a day/second/microsecond delta to a `datetime.datetime` `<'datetime.datetime'>`.

    This performs **wall-clock arithmetic** like `dt + datetime.timedelta(...)`:
    - No timezone normalization is applied; `tzinfo` and `fold` are preserved.
    - For aware datetimes, the local clock fields are shifted by the delta (even
    across DST boundaries) without converting to UTC.

    :param dt `<'datetime.datetime'>`: Base datetime (naive or aware).
    :param days `<'int'>`: Days to add (can be negative). Defaults to 0.
    :param seconds `<'int'>`: Seconds to add (can be negative). Defaults to 0.
    :param microseconds `<'int'>`: Microseconds to add (can be negative). Defaults to 0.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).

    ## Equivalent
    >>> dt + datetime.timedelta(days, seconds, microseconds)
    """

def dt_add_delta(
    dt: datetime.datetime,
    years: int = 0,
    quarters: int = 0,
    months: int = 0,
    weeks: int = 0,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
    milliseconds: int = 0,
    microseconds: int = 0,
    year: int = -1,
    month: int = -1,
    day: int = -1,
    weekday: int = -1,
    hour: int = -1,
    minute: int = -1,
    second: int = -1,
    millisecond: int = -1,
    microsecond: int = -1,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Add relative and absolute deltas to datetime `<'datetime.datetime'>`.

    ## Absolute Deltas (Replace specified fields)

    :param year `<'int'>`: Absolute year. Defaults to `-1` (no change).
    :param month `<'int'>`: Absolute month. Defaults to `-1` (no change).
    :param day `<'int'>`: Absolute day. Defaults to `-1` (no change).
    :param weekday `<'int'>`: Absolute weekday (0=Mon...6=Sun). Defaults to `-1` (no change).
    :param hour `<'int'>`: Absolute hour. Defaults to `-1` (no change).
    :param minute `<'int'>`: Absolute minute. Defaults to `-1` (no change).
    :param second `<'int'>`: Absolute second. Defaults to `-1` (no change).
    :param millisecond `<'int'>`: Absolute millisecond. Defaults to `-1` (no change).
        Overrides `microsecond` milliseoncd part if both are provided.
    :param microsecond `<'int'>`: Absolute microsecond. Defaults to `-1` (no change).

    ## Relative Deltas (Add to specified fields)

    :param years `<'int'>`: Relative years. Defaults to `0`.
    :param quarters `<'int'>`: Relative quarters (3 months). Defaults to `0`.
    :param months `<'int'>`: Relative months. Defaults to `0`.
    :param weeks `<'int'>`: Relative weeks (7 days). Defaults to `0`.
    :param days `<'int'>`: Relative days. Defaults to `0`.
    :param hours `<'int'>`: Relative hours. Defaults to `0`.
    :param minutes `<'int'>`: Relative minutes. Defaults to `0`.
    :param seconds `<'int'>`: Relative seconds. Defaults to `0`.
    :param milliseconds `<'int'>`: Relative milliseconds (1,000 us). Defaults to `0`.
    :param microseconds `<'int'>`: Relative microseconds. Defaults to `0`.

    ## Datetime Class

    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.

    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).
        after applying the specified deltas.
    """

def dt_replace(
    dt: datetime.datetime,
    year: int = -1,
    month: int = -1,
    day: int = -1,
    hour: int = -1,
    minute: int = -1,
    second: int = -1,
    microsecond: int = -1,
    tzinfo: datetime.tzinfo | None = -1,
    fold: int = -1,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Replace specified fields of a datetime `<'datetime.datetime'>`.

    :param dt `<'datetime.datetime'>`: The source datetime (naive or aware).
    :param year `<'int'>`: Absolute year. Defaults to `-1` (no change).
        If specified (greater than `0`), clamps to [1..9999].
    :param month `<'int'>`: Absolute month. Defaults to `-1` (no change).
        If specified (greater than `0`), clamps to [1..12].
    :param day `<'int'>`: Absolute day. Defaults to `-1` (no change).
        If specified (greater than `0`), clamps to [1..maximum days the resulting month].
    :param hour `<'int'>`: Absolute hour. Defaults to `-1` (no change).
        If specified (greater than or equal to `0`), clamps to [0..23].
    :param minute `<'int'>`: Absolute minute. Defaults to `-1` (no change).
        If specified (greater than or equal to `0`), clamps to [0..59].
    :param second `<'int'>`: Absolute second. Defaults to `-1` (no change).
        If specified (greater than or equal to `0`), clamps to [0..59].
    :param microsecond `<'int'>`: Absolute microsecond. Defaults to `-1` (no change).
        If specified (greater than or equal to `0`), clamps to [0..999999].
    :param tzinfo `<'tzinfo/None'>`: The timeone. Defaults to `-1` (no change).
        If specified as `None`, removes tzinfo (makes datetime naive).
        If specified as a `tzinfo` subclass, attaches to `dt`.
    :param fold `<'int'>`: Fold value (0 or 1) for ambiguous times. Defaults to `-1` (no change).
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified),
        after applying the specified field replacements.
    """

def dt_replace_tz_fold(
    dt: datetime.datetime,
    tzinfo: datetime.tzinfo | None,
    fold: int,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Create a copy of `dt` with `tzinfo` and `fold` replaced `<'datetime.datetime'>`.

    :param dt `<'datetime.datetime'>`: Source datetime (naive or aware).
    :param tzinfo `<'tzinfo/None'>`: The target tzinfo to attach.
    :param fold `<'int'>`: Must be 0 or 1; otherwise `ValueError` is raised.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified)
        with the same fields except `tzinfo` and `fold`.

    ## Equivalent
    >>> dt.replace(tzinfo=tzinfo, fold=fold)
    """

def dt_replace_tz(
    dt: datetime.datetime,
    tzinfo: datetime.tzinfo | None,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Create a copy of `dt` with `tzinfo` replaced `<'datetime.datetime'>`.

    :param dt `<'datetime.datetime'>`: Source datetime (naive or aware).
    :param tzinfo `<'tzinfo/None'>`: The target tzinfo to attach.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified)
        with the same fields except `tzinfo`.

    ## Equivalent
    >>> dt.replace(tzinfo=tzinfo)
    """

def dt_replace_fold(
    dt: datetime.datetime,
    fold: int,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Create a copy of `dt` with `fold` set to 0 or 1 `<'datetime.datetime'>`.

    :param dt `<'datetime.datetime'>`: Source datetime (naive or aware).
    :param fold `<'int'>`: Must be 0 or 1; otherwise `ValueError` is raised.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified)
        with the same fields except `fold`.

    ## Equivalent
    >>> dt.replace(fold=fold)
    """

# . check
def dt_compare(
    dt1: datetime.datetime,
    dt2: datetime.datetime,
    allow_mixed: bool = False,
) -> int:
    """(cfunc) Three-way comparison between two datetimes on a common microsecond timeline `<'int'>`.

    :param dt1 `<'datetime.datetime'>`: The 1st datetime instance.
    :param dt2 `<'datetime.datetime'>`: The 2nd datetime instance.
    :param allow_mixed `<'bool'>`: Whether to allow comparisons between naive and aware datetimes. Defaults to `False`.
    :returns `<'int'>`: The comparison result:

        - `1`:  if dt1 > dt2
        - `0`:  if dt1 == dt2
        - `-1`: if dt1 < dt2
        - `2`:  `only when` one operand is timezone-naive and the other is
                timezone-aware and `allow_mixed=True` (no comparison performed).

    :raises `<'TypeError'>`: When comparing naive vs aware datetimes and `allow_mixed=False`.

    ## Rules
    - If exactly one operand is aware and the other is naive,
      return `2` when `allow_mixed=True`, otherwise raise `TypeError`.
    - If both are aware, both are converted to microseconds on a common
      timeline and compared.
    - If both are naive, both are converted to microseconds and compared
      on the naive timeline (no timezone semantics).
    """

def dt_is_first_doy(dt: datetime.datetime) -> bool:
    """(cfunc) Check whether the datetime is on the first day of its year `<'bool'>`.

    - First day of the year: `YYYY-01-01`

    :param dt `<'datetime.datetime'>`: Datetime to check.
    :returns `<'bool'>`: True if the datetime is January 1 of its year; Otherwiase False.
    """

def dt_is_last_doy(dt: datetime.datetime) -> bool:
    """(cfunc) Check whether the datetime is on the last day of its year `<'bool'>`.

    - Last day of the year: `YYYY-12-31`

    :param dt `<'datetime.datetime'>`: Datetime to check.
    :returns `<'bool'>`: True if the datetime is December 31 of its year; Otherwiase False.
    """

def dt_is_first_doq(dt: datetime.datetime) -> bool:
    """(cfunc) Check whether the datetime is on the first day of its quarter `<'bool'>`.

    - First day of the quarter: `YYYY-MM-01` where MM in `{01, 04, 07, 10}`

    :param dt `<'datetime.datetime'>`: Datetime to check.
    :returns `<'bool'>`: True if the datetime is the first day of its quarter; Otherwiase False.
    """

def dt_is_last_doq(dt: datetime.datetime) -> bool:
    """(cfunc) Check whether the datetime is on the last day of its quarter `<'bool'>`.

    - Last day of the quarter: `YYYY-MM-DD` where MM in `{03, 06, 09, 12}`

    :param dt `<'datetime.datetime'>`: Datetime to check.
    :returns `<'bool'>`: True if the datetime is the last day of its quarter; Otherwiase False.
    """

def dt_is_first_dom(dt: datetime.datetime) -> bool:
    """(cfunc) Check whether the datetime is on the first day of its month `<'bool'>`.

    - First day of the month: `YYYY-MM-01`

    :param dt `<'datetime.datetime'>`: Datetime to check.
    :returns `<'bool'>`: True if the datetime is the first day of its month; Otherwiase False.
    """

def dt_is_last_dom(dt: datetime.datetime) -> bool:
    """(cfunc) Check whether the datetime is on the last day of its month `<'bool'>`.

    - Last day of the month: `YYYY-MM-DD` where DD is the maximum days in that month.

    :param dt `<'datetime.datetime'>`: Datetime to check.
    :returns `<'bool'>`: True if the datetime is the last day of its month; Otherwiase False.
    """

def dt_is_start_of_time(dt: datetime.datetime) -> bool:
    """(cfunc) Check whether the datetime is at the start of time `<'bool'>`:

    - Start of time: `YYYY-MM-DD 00:00:00`

    :param dt `<'datetime.datetime'>`: Datetime to check.
    :returns `<'bool'>`: True if datetime is at the start of time; Otherwise False.
    """

def dt_is_end_of_time(dt: datetime.datetime) -> bool:
    """(cfunc) Check whether the datetime is at the end of time `<'bool'>`.

    - End of time: `YYYY-MM-DD 23:59:59.999999`

    :param dt `<'datetime.datetime'>`: Datetime to check.
    :returns `<'bool'>`: True if datetime is at the end of time; Otherwise False.
    """

# . tzinfo
def dt_tzname(dt: datetime.datetime) -> str | None:
    """(cfunc) Get the tzinfo 'tzname' of the datetime `<'str/None'>`.

    ## Equivalent
    >>> dt.tzname()
    """

def dt_dst(dt: datetime.datetime) -> datetime.timedelta | None:
    """(cfunc) Get the tzinfo 'dst' of the datetime `<'datetime.timedelta/None'>`.

    ## Equivalent
    >>> dt.dst()
    """

def dt_utcoffset(dt: datetime.datetime) -> datetime.timedelta | None:
    """(cfunc) Get the tzinfo 'utcoffset' of the datetime `<'datetime.timedelta/None'>`.

    ## Equivalent
    >>> dt.utcoffset()
    """

def dt_astimezone(
    dt: datetime.datetime,
    tzinfo: datetime.tzinfo | None = None,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Convert a `datetime.datetime` to another timezone `<'datetime.datetime'>`.

    :param dt `<'datetime.datetime'>`: Datetime to convert (naive or aware).
    :param tzinfo `<'tzinfo/None'>`: Target timezone. Defaults to `None`.

        - If `None`, the system **local** timezone is used.
        - Must be a `tzinfo-compatible` object when provided.

    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified)
        representing the **same instant** expressed in the target timezone. For naive
        inputs + `tzinfo is None`, this *localizes* the datetime to the system local zone.

    ## Semantics
    - **Aware → Aware:** Convert by subtracting the source `utcoffset(dt)`,
      attach the target tzinfo, then normalize via `tzinfo.fromutc(...)`.
    - **Naive + tzinfo is None:** Treat `dt` as local wall time and simply attach
      the local tzinfo (no clock change).
    - **Naive + tzinfo provided:** Treat `dt` as local wall time, convert the
      instant to UTC, attach `tzinfo`, then normalize via `tzinfo.fromutc(...)`.
    - **Identity fast path:** If `dt.tzinfo is tzinfo`, return `dt` unchanged.

    ## DST / fold
    - Ambiguous/nonexistent local times are resolved by delegating
      to `tzinfo.fromutc(...)`, which honors `dt.fold` per PEP 495.

    ## Equivalent
    >>> dt.astimezone(tzinfo)
    """

def dt_normalize_tz(
    dt: datetime.datetime,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Normalize an aware datetime against its own tzinfo `<'datetime.datetime'>`.

    :param dt `<'datetime.datetime'>`: Datetime to normalize.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The normalized datetime (or subclass if `dtclass` is specified).

    ## Behavior:
      - If `dt` is timezone-niave → returns `dt` as-is.
      - Compute the timezone offsets at the same wall-clock time
        with `fold=0` and `fold=1`.
      - If offset with `fold=1` is greater than with `fold=0` (spring-forward gap),
        shift wall time by the offset delta `Δ`:

          - `fold == 0` → move backward by Δ (to the last valid wall time)
          - `fold == 1` → move forward by Δ (to the next valid wall time)

      - Otherwise (ambiguous or no change), returns `dt` as-is.
      - If either offset is unavailable, returns `dt` as-is.
    """

# datetime.time -------------------------------------------------------------------------------------
# . generate
def time_new(
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
    tzinfo: datetime.tzinfo | None = None,
    fold: int = 0,
    tclass: type[datetime.time] | None = None,
) -> datetime.time:
    """(cfunc) Create a new time `<'datetime.time'>`.

    :param hour `<'int'>`: Hour [0..23]. Defaults to `0`.
    :param minute `<'int'>`: Minute [0..59]. Defaults to `0`.
    :param second `<'int'>`: Second [0..59]. Defaults to `0`.
    :param microsecond `<'int'>`: Microsecond [0..999999]. Defaults to `0`.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone. Defaults to `None`.
    :param fold `<'int'>`: Optional fold flag for ambiguous times (0 or 1). Defaults to `0`.
        Values other than `1` are treated as `0`.
    :param tclass `<'type[datetime.time]/None'>`: Optional custom time class. Defaults to `None`.
        if `None` uses python's built-in `datetime.time` as the constructor.
    :returns `<'datetime.time'>`: The resulting time (or subclass if `tclass` is specified).
    """

def time_now(
    tzinfo: datetime.tzinfo | None = None,
    tclass: type[datetime.time] | None = None,
) -> datetime.time:
    """(cfunc) Get the current time `<'datetime.time'>`.

    :param tzinfo `<'tzinfo/None'>`: Optional timezone. Defaults to `None`.

        - If specified, return an aware time in that timezone.
        - Otherwise, return a naive local time.

    :param tclass `<'type[datetime.time]/None'>`: Optional custom time class. Defaults to `None`.
        if `None` uses python's built-in `datetime.time` as the constructor.

    :returns `<'datetime.time'>`: The current time (or subclass if `tclass` is specified).

    ## Equivalent
    >>> datetime.datetime.now(tzinfo).time()
    """

# . type check
def is_time(obj: object) -> bool:
    """(cfunc) Check if an object is an instance or sublcass of `datetime.time` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> isinstance(obj, datetime.time)
    """

def is_time_exact(obj: object) -> bool:
    """(cfunc) Check if an object is an exact `datetime.time` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> type(obj) is datetime.time
    """

# . conversion: to
def time_isoformat(time: datetime.time, utc: bool = False) -> str:
    """(cfunc) Convert a `datetime.time` to ISO string `<'str'>`.

    :param time `<'datetime.time'>`: The time to convert.
    :param utc <'bool'>: Whether to append a UTC offset. Defaults to False.

        - If False or `time` is timezone-naive, UTC offset is ignored.
        - If True and `time` is timezone-aware, append UTC offset (e.g., +0530).

    :returns `<'str'>`: Formatted text.
    """

def time_to_us(time: datetime.time) -> int:
    """(cfunc) Convert a `datetime.time` to total microseconds `<'int'>`.

    :param time `<'datetime.time'>`: Time to convert.
    :returns `<'int'>`: Microseconds since midnight in [0, 86_400_000_000).
    """

def time_to_sec(time: datetime.time) -> float:
    """(cfunc) Convert a `datetime.time` to total seconds `<'float'>`.

    :param time `<'datetime.time'>`: Time to convert.
    :returns `<'float'>`: Seconds since midnight in [0.0, 86400.0).
    """

# . conversion: from
def time_fr_us(
    value: int,
    tzinfo: datetime.tzinfo | None = None,
    tclass: type[datetime.time] | None = None,
) -> datetime.time:
    """(cfunc) Create time from microseconds since the Unix epoch `<'datetime.time'>`.

    :param value `<'int'>`: Microseconds since epoch.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone to **attach**. Defaults to `None`.
    :param tclass `<'type[datetime.time]/None'>`: Optional custom time class. Defaults to `None`.
        if `None` uses python's built-in `datetime.time` as the constructor.
    :returns `<'datetime.time'>`: The resulting time (or subclass if `tclass` is specified).
    """

def time_fr_sec(
    value: float,
    tzinfo: datetime.tzinfo | None = None,
    tclass: type[datetime.time] | None = None,
) -> datetime.time:
    """(cfunc) Create time from seconds since the Unix epoch `<'datetime.time'>`.

    :param value `<'int'>`: Seconds since epoch.
    :param tzinfo `<'tzinfo/None'>`: Optional timezone to **attach**. Defaults to `None`.
    :param tclass `<'type[datetime.time]/None'>`: Optional custom time class. Defaults to `None`.
        if `None` uses python's built-in `datetime.time` as the constructor.
    :returns `<'datetime.time'>`: The resulting time (or subclass if `tclass` is specified).
    """

def time_fr_time(
    time: datetime.time,
    tclass: type[datetime.time] | None = None,
) -> datetime.time:
    """(cfunc) Create time from another time (include subclass) `<'datetime.time'>`.

    :param time `<'datetime.time'>`: The source time (including subclasses).
    :param tclass `<'type[datetime.time]/None'>`: Target time class. Defaults to `None`.
        If `None` set to python's built-in `datetime.time`.
        If `time` is already of type `tclass`, returns `time` directly.
    :returns `<'datetime.time'>`: The resulting time (or subclass if `tclass` is specified)
        with the same time fields and tzinfo.
    """

def time_fr_dt(
    dt: datetime.datetime,
    tclass: type[datetime.time] | None = None,
) -> datetime.time:
    """(cfunc) Create time from datetime (include subclass) `<'datetime.time'>`.

    :param dt `<'datetime.datetime'>`: The source datetime (including subclasses).
    :param tclass `<'type[datetime.time]/None'>`: Optional custom time class. Defaults to `None`.
        if `None` uses python's built-in `datetime.time` as the constructor.
    :returns `<'datetime.time'>`: The resulting time (or subclass if `tclass` is specified)
        with the same time fields and tzinfo.
    """

# datetime.timedelta --------------------------------------------------------------------------------
# . generate
def td_new(
    days: int = 0,
    seconds: int = 0,
    microseconds: int = 0,
    tdclass: type[datetime.timedelta] | None = None,
) -> datetime.timedelta:
    """(cfunc) Create a new timedelta `<'datetime.timedelta'>`.

    :param days `<'int'>`: Days (can be negative). Defaults to 0.
    :param seconds `<'int'>`: Seconds (can be negative). Defaults to 0.
    :param microseconds `<'int'>`: Microseconds (can be negative). Defaults to 0.
    :param tdclass `<'type[datetime.timedelta]/None'>`: Optional custom timedelta class. Defaults to `None`.
        if `None` uses python's built-in `datetime.timedelta` as the constructor.
    :returns `<'datetime.timedelta'>`: The resulting timedelta (or subclass if `tdclass` is specified).

    ## Equivalent
    >>> datetime.timedelta(days, seconds, microseconds)
    """

# . type check
def is_td(obj: object) -> bool:
    """(cfunc) Check if an object is an instance or subclass of `datetime.timedelta` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> isinstance(obj, datetime.timedelta)
    """

def is_td_exact(obj: object) -> bool:
    """(cfunc) Check if an object is an exact `datetime.timedelta` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> type(obj) is datetime.timedelta
    """

# . conversion: to
def td_isoformat(td: datetime.timedelta) -> str:
    """(cfunc) Convert timedelta to string in ISO-like string (±HH:MM:SS[.f]) `<'str'>`.

    :param td `<'datetime.timedelta'>`: The timedelta to convert.
    :returns `<'str'>`: The ISO-like string in the format of ±HH:MM:SS[.f]
    """

def td_utcformat(td: datetime.timedelta) -> str:
    """(cfunc) Convert a `datetime.timedelta` to a UTC offset string (±HHMM) `<'str'>`.

    :param td `<'datetime.timedelta'>`: The timedelta to convert.
    :returns `<'str'>`: The UTC offset string in the format of ±HHMM
    """

def td_to_us(td: datetime.timedelta) -> int:
    """(cfunc) Convert a `datetime.timedelta` to total microseconds `<'int'>`.

    :param td `<'datetime.timedelta'>`: The timedelta to convert.
    :returns `<'int'>`: Total microseconds of the timedelta
    """

def td_to_sec(td: datetime.timedelta) -> float:
    """(cfunc) Convert a `datetime.timedelta` to total seconds `<'float'>`.

    :param td `<'datetime.timedelta'>`: The timedelta to convert.
    :returns `<'seconds'>`: Total seconds of the timedelta
    """

# . conversion: from
def td_fr_us(
    value: int,
    tdclass: type[datetime.timedelta] | None = None,
) -> datetime.timedelta:
    """(cfunc) Create timedelta from microseconds `<'datetime.timedelta'>`.

    :param value `<'int'>`: Delta in microseconds.
    :param tdclass `<'type[datetime.timedelta]/None'>`: Optional custom timedelta class. Defaults to `None`.
        if `None` uses python's built-in `datetime.timedelta` as the constructor.
    :returns `<'datetime.timedelta'>`: The resulting timedelta (or subclass if `tdclass` is specified).
    """

def td_fr_sec(
    value: float,
    tdclass: type[datetime.timedelta] | None = None,
) -> datetime.timedelta:
    """(cfunc) Create timedelta from seconds `<'datetime.timedelta'>`.

    :param value `<'float'>`: Delta in seconds.
    :param tdclass `<'type[datetime.timedelta]/None'>`: Optional custom timedelta class. Defaults to `None`.
        if `None` uses python's built-in `datetime.timedelta` as the constructor.
    :returns `<'datetime.timedelta'>`: The resulting timedelta (or subclass if `tdclass` is specified).
    """

def td_fr_td(
    td: datetime.timedelta,
    tdclass: type[datetime.timedelta] | None = None,
) -> datetime.timedelta:
    """(cfunc) Create timedelta from another timedelta (or subclass) `<'datetime.timedelta'>`.

    :param td `<'datetime.timedelta'>`: The source timedelta (including subclasses).
    :param tdclass `<'type[datetime.timedelta]/None'>`: Target timedelta class. Defaults to `None`.
        If `None` set to python's built-in `datetime.timedelta`.
        If `td` is already of type `tdclass`, returns `td` directly.
    :returns `<'datetime.timedelta'>`: The resulting timedelta (or subclass if `tdclass` is specified)
        with the same days, seconds and microseconds.
    """

# datetime.tzinfo -----------------------------------------------------------------------------------
# . generate
def tz_new(hours: int = 0, minutes: int = 0, seconds: int = 0) -> datetime.timezone:
    """(cfunc) Create a new fixed-offset timezone `<'datetime.timezone'>`.

    :param hours `<'int'>`: Hour component of the fixed offset. Defaults to 0.
    :param minutes `<'int'>`: Minute component of the fixed offset. Defaults to 0.
    :param seconds `<'int'>`: Second component of the fixed offset. Defaults to 0.
    :returns `<'datetime.timezone'>`: The corresponding fixed-offset timezone.

    Equivalent:
    >>> datetime.timezone(datetime.timedelta(hours=hours, minutes=minutes))
    """

def tz_local(
    dt: datetime.datetime | None = None,
) -> zoneinfo.ZoneInfo | datetime.timezone:
    """(cfunc) Get the process-local timezone `<'zoneinfo.ZoneInfo/datetime.timezone'>`.

    :param dt `<'datetime.datetime/None'>`: Ignored, reserved for future use. Defaults to `None`.
    :returns `<'zoneinfo.ZoneInfo/datetime.timezone'>`: The local timezone.

        - Returns a concrete IANA zone (`zoneinfo.ZoneInfo`) when possible
          using [babel](https://github.com/python-babel/babel) `LOCALTZ` name.
        - If that fails, falls back to a fixed-offset `datetime.timezone` using
          the **current** local UTC offset (but no DST rules).

    ## Notice
    - The local timezone is resolved and cached `once at module import`
      and reused on subsequent calls. It does not track historical or
      future transitions.
    """

def tz_local_sec(dt: datetime.datetime | None = None) -> int:
    """(cfunc) Return the local UTC offset (in whole seconds) at the given instant `<'int'>`.

    :param dt `<'datetime.datetime/None'>`: The datetime to evaluate. Defaults to `None`.

        - If `dt` is aware, its UTC offset is computed for that wall time.
        - If `dt` is naive, it is interpreted as *local time* (mktime-style) and
          DST ambiguity is resolved using `fold`.
        - If `dt` is None, use current time.

    :returns `<'int'>`: The local UTC offset in seconds (positive east of UTC, negative west).
    """

def tz_parse(
    tz: zoneinfo.ZoneInfo | datetime.timezone | str | None,
) -> zoneinfo.ZoneInfo | datetime.timezone | None:
    """(cfunc) Parse timezone input to `<'zoneinfo.ZoneInfo/datetime.timezone/None'>`.

    :param tz `<'datetime.timezone/zoneinfo.ZoneInfo/pytz/str/None'>`: The timezone object.

        - If 'tz' is `None` → return `None`.
        - If 'tz' is `datetime.timezone/zoneinfo.ZoneInfo` → return as-is.
        - If 'tz' is `str` →:

            - `"local"` (case-insensitive) → cached local timezone
            - common UTC aliases (case-insensitive) → UTC
            - otherwise interpreted as a canonical IANA key via ZoneInfo

        - If 'tz' is `pytz` timezone → mapped by its `zone` name to a `ZoneInfo`.

    :returns `<'zoneinfo.ZoneInfo/datetime.timezone/None'>`: The normalize timezone object.
    :raises `InvalidTimezoneError`: If 'tz' is invalid or unrecognized.
    """

# . type check
def is_tz(obj: object) -> bool:
    """(cfunc) Check if an object is an instance or sublcass of `datetime.tzinfo` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> isinstance(obj, datetime.tzinfo)
    """

def is_tz_exact(obj: object) -> bool:
    """(cfunc) Check if an object is an exact `datetime.tzinfo` `<'bool'>`.

    :param obj `<'Any'>`: Object to check.
    :returns `<'bool'>`: Check result.

    ## Equivalent
    >>> type(obj) is datetime.date
    """

# . access
def tz_name(
    tz: datetime.tzinfo | None,
    dt: datetime.datetime | None = None,
) -> str | None:
    """(cfunc) Return the display name from a tzinfo `<'str/None'>`.

    :param tz `<'datetime.tzinfo/None'>`: Time zone object. If `None`, return as-is.
    :param dt `<'datetime.datetime/None'>`: Optional datetime to evaluate
        (some tzinfos vary by date). Defaults to `None`.
    :returns `<'str/None'>`: A zone name like 'UTC', 'EST', 'CET', or `None` if unavailable.

    Equivalent to:
    >>> tz.tzname(dt)
    """

def tz_dst(
    tz: datetime.tzinfo | None,
    dt: datetime.datetime | None = None,
) -> datetime.timedelta | None:
    """(cfunc) Return the DST offset of a tzinfo `<'timedelta/None'>`.

    :param tz `<'datetime.tzinfo/None'>`: Time zone object. If `None`, return as-is.
    :param dt `<'datetime.datetime/None'>`: Optional datetime to evaluate
        (some tzinfos vary by date). Defaults to `None`.
    :returns `<'datetime.timedelta/None'>`: A DST offset, or `None` if not applicable.

    ## Equivalent
    >>> tz.dst(dt)
    """

def tz_utcoffset(
    tz: datetime.tzinfo | None,
    dt: datetime.datetime | None = None,
) -> datetime.timedelta | None:
    """(cfunc) Return the UTC offset from a tzinfo `<'timedelta/None'>`.

    :param tz `<'datetime.tzinfo/None'>`: Time zone object. If `None`, return as-is.
    :param dt `<'datetime.datetime/None'>`: Optional datetime to evaluate
        (some tzinfos vary by date). Defaults to `None`.
    :returns `<'datetime.timedelta/None'>`: A UTC offset, `None` if not applicable.

    ## Equivalent
    >>> tz.utcoffset(dt)
    """

def tz_utcoffset_sec(
    tz: datetime.tzinfo | None,
    dt: datetime.datetime | None = None,
) -> int:
    """(cfunc) Return the UTC offset in total seconds from a tzinfo `<'int'>`.

    :param tz `<'datetime.tzinfo/None'>`: Time zone object.
    :param dt `<'datetime.datetime/None'>`: Optional datetime to evaluate
        (some tzinfos vary by date). Defaults to `None`.
    :returns `<'int'>`: Total whole seconds of the UTC offset,
        or sentinel (-100,000) when the offset is unavailable.
    """

def tz_utcformat(
    tz: datetime.tzinfo | None,
    dt: datetime.datetime | None = None,
) -> str | None:
    """(cfunc) Return the UTC offset in ISO format string (±HHMM) from a tzinfo `<'str/None'>`.

    :param tz `<'datetime.tzinfo/None'>`: Time zone object.
    :param dt `<'datetime.datetime/None'>`: Optional datetime to evaluate
        (some tzinfos vary by date). Defaults to `None`.
    :returns `<'str/None'>`: A string like '+0530' or '-0700',
        or `None` when the offset is unavailable.
    """

# NumPy: time unit ----------------------------------------------------------------------------------
def nptime_unit_int2str(unit: int) -> str:
    """(cfunc) Map numpy time unit to its corresponding string form `<'str'>`.

    :param unit `<'int'>`: An NPY_DATETIMEUNIT enum value.
    :returns `<'str'>`: The corresponding string form:
        'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'Y', etc.
    """

def nptime_unit_int2dt64(unit: int) -> np.dtype:
    """(cfunc) Maps numpy time unit to its corresponding datetime dtype `<'np.dtype'>`.

    :param unit `<'int'>`: An NPY_DATETIMEUNIT enum value.
    :returns `<'np.dtype'>`: The corresponding datetime dtype:
        np.dtype('datetime64[ns]'), np.dtype('datetime64[Y]'), etc.
    """

def nptime_unit_str2int(unit: str) -> int:
    """(cfunc) Maps numpy time unit string form to its corresponding
    NPY_DATETIMEUNIT enum value `<'int'>`.

    :param unit `<'str'>`: Time unit in its string form:
        'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'Y', etc.
    :returns `<'int'>`: The corresponding NPY_DATETIMEUNIT enum value.
    """

def nptime_unit_str2dt64(unit: str) -> np.dtype:
    """(cfunc) Maps numpy time unit string form to its corresponding
    datetime dtype `<'np.dtype'>`.

    :param unit `<'str'>`: Time unit in its string form:
        'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'Y', etc.
    :returns `<'np.dtype'>`: The corresponding datetime dtype:
        np.dtype('datetime64[ns]'), np.dtype('datetime64[Y]'), etc.
    """

def get_arr_nptime_unit(arr: np.ndarray) -> int:
    """(cfunc) Get the time unit (NPY_DATETIMEUNIT enum) of a datetime-like array `<'int'>`.

    :param arr `<'np.ndarray'>`: An array with dtype `datetime64[*]` or `timedelta64[*]`.
    :returns `<'int'>`: The time unit as an integer form
        its corresponding NPY_DATETIMEUNIT enum value
    """

# NumPy: datetime64 ---------------------------------------------------------------------------------
# . type check
def is_dt64(obj: object) -> bool:
    """(cfunc) Check if the object is an instance of np.datetime64 `<'bool'>`.

    ## Equivalent
    >>> isinstance(obj, np.datetime64)
    """

def assure_dt64(obj: object) -> None:
    """(cfunc) Assure the object is an instance of np.datetime64."""

# . conversion
def dt64_as_int64_us(dt64: np.datetime64, offset: int = 0) -> int:
    """(cfunc) Convert a np.datetime64 to int64 microsecond ('us') ticks `<'int'>`.

    :param dt64 `<'np.datetime64'>`: The datetime64 to convert.
    :param offset `<'int'>`: An optional offset added after conversion. Defaults to `0`.
    :returns `<'int'>`: The int64 value representing the datetime64 in microseconds.

    ## Equivalent
    >>> dt64.astype("datetime64[us]").astype("int64") + offset
    """

def dt64_to_dt(
    dt64: np.datetime64,
    tzinfo: datetime.tzinfo | None = None,
    dtclass: type[datetime.datetime] | None = None,
) -> datetime.datetime:
    """(cfunc) Convert np.datetime64 to datetime `<'datetime.datetime'>`.

    :param dt64 `<'np.datetime64'>`: The datetime64 to convert.
    :param tzinfo `<'datetime.tzinfo/None'>`: An optional timezone to attach
        to the resulting datetime. Defaults to `None`.
    :param dtclass `<'type[datetime.datetime]/None'>`: Optional custom datetime class. Defaults to `None`.
        if `None` uses python's built-in `datetime.datetime` as the constructor.
    :returns `<'datetime.datetime'>`: The resulting datetime (or subclass if `dtclass` is specified).
    """

# NumPy: timedelta64 --------------------------------------------------------------------------------
# . type check
def is_td64(obj: object) -> bool:
    """(cfunc) Check if the object is an instance of np.timedelta64 `<'bool'>`.

    ## Equivalent
    >>> isinstance(obj, np.timedelta64)
    """

def assure_td64(obj: object) -> None:
    """(cfunc) Assure the object is an instance of np.timedelta64."""

# . conversion
def td64_as_int64_us(td64: np.timedelta64, offset: int = 0) -> int:
    """(cfunc) Convert a np.timedelta64 to int64 microsecond ('us') ticks `<'int'>`.

    :param dt64 `<'np.timedelta64'>`: The timedelta64 to convert.
    :param offset `<'int'>`: An optional offset added after conversion. Defaults to `0`.
    :returns `<'int'>`: The int64 value representing the timedelta64 in microseconds.

    ## Equivalent
    >>> td64.astype("timedelta64[us]").astype("int64") + offset
    """

def td64_to_td(
    td64: np.timedelta64,
    tdclass: type[datetime.timedelta] | None = None,
) -> datetime.timedelta:
    """(cfunc) Convert np.timedelta64 to timedelta `<'datetime.timedelta'>`.

    :param td64 `<'np.timedelta64'>`: The timedelta64 to convert.
    :param tdclass `<'type[datetime.timedelta]/None'>`: Optional custom timedelta class. Defaults to `None`.
        if `None` uses python's built-in `datetime.timedelta` as the constructor.
    :returns `<'datetime.timedelta'>`: The resulting timedelta (or subclass if `tdclass` is specified).
    """

# NumPy: ndarray ---------------------------------------------------------------------------------------
# . type check
def is_arr(obj: object) -> bool:
    """(cfunc) Check if the object is an instance of np.ndarray `<'bool'>`.

    ## Equivalent
    >>> isinstance(obj, np.ndarray)
    """

def assure_1dim_arr(arr: np.ndarray) -> bool:
    """(cfunc) Assure the array is 1-dimensional."""

def assure_arr_contiguous(arr: np.ndarray) -> np.ndarray:
    """(cfunc) Ensure that an ndarray is C-contiguous in memory `<'np.ndarray'>`.

    :returns `<'np.ndarray'>`: The original array if already contiguous;
        otherwise, returns a contiguous copy.
    """

def is_arr_int(arr: np.ndarray) -> bool:
    """(cfunc) Check if a numpy array is of integer dtype `<'bool'>`.

    - Integer dtype: `int8`, `int16`, `int32` and `int64`.
    """

def is_arr_uint(arr: np.ndarray) -> bool:
    """(cfunc) Check if a numpy array is of unsigned integer dtype `<'bool'>`.

    - Unsigned Integer dtype: `uint8`, `uint16`, `uint32` and `uint64`.
    """

def is_arr_float(arr: np.ndarray) -> bool:
    """(cfunc) Check if a numpy array is of float dtype `<'bool'>`.

    - Float dtype: `float16`, `float32` and `float64`.
    """

# . dtype
def arr_assure_int64(arr: np.ndarray, copy: bool = True) -> np.ndarray[np.int64]:
    """(cfunc) Ensure that a 1-D array is contiguous and dtype int64 `<'ndarray[int64]'>`.

    :param arr `<'ndarray'>`: The 1-D array to ensure.
    :param copy `<'bool'>`: Whether to always create a copy even if the array is
        already dtype int64. Defaults to `True`.
    :returns `<'ndarray[int64]'>`:
        If the array is already the right dtype, return it directly or a copy
        depending on the `copy` flag. Otherwise, cast the array to int64
        (new buffer) and return the result.
    """

def arr_assure_int64_like(arr: np.ndarray, copy: bool = True) -> np.ndarray:
    """(cfunc) Ensure that a 1-D array is contiguous and dtype
    int64 / datetime64[*] / timedelta64[*] `<'ndarray[int64*]'>`.

    :param arr `<'ndarray'>`: The 1-D array to ensure.
    :param copy `<'bool'>`: Whether to always create a copy even if the array is
        already dtype int64 / datetime64[*] / timedelta64[*]. Defaults to `True`.
    :returns `<'ndarray[int64]'>`:
        If the array is already the right dtype, return it directly or a copy
        depending on the `copy` flag. Otherwise, cast the array to int64
        (new buffer) and return the result.
    """

def arr_assure_float64(arr: np.ndarray, copy: bool = True) -> np.ndarray[np.float64]:
    """(cfunc) Ensure that a 1-D array is contiguous and dtype float64 `<'ndarray[float64]'>`.

    :param arr `<'ndarray'>`: The 1-D array to ensure.
    :param copy `<'bool'>`: Whether to always create a copy even if the array is
        already dtype float64. Defaults to `True`.
    :returns `<'ndarray[float64]'>`:
        If the array is already the right dtype, return it directly or a copy
        depending on the `copy` flag. Otherwise, cast the array to float64
        (new buffer) and return the result.
    """

# . create
def arr_zero_int64(size: int) -> np.ndarray[np.int64]:
    """(cfunc) Create a 1-D ndarray[int64] filled with zero `<'ndarray[int64]'>`.

    :param size `<'int'>`: The size of the new array.
    :returns `<'ndarray[int64]'>`: A new 1-D ndarray of the specified size, filled with zeros.

    ## Equivalent
    >>> np.zeros(size, dtype="int64")
    """

def arr_fill_int64(value: int, size: int) -> np.ndarray[np.int64]:
    """(cfunc) Create a 1-D ndarray[int64] filled with a specific integer `<'ndarray[int64]'>`.

    :param value `<'int'>`: The integer value to fill the array with.
    :param size `<'int'>`: The size of the new array.
    :returns `<'ndarray[int64]'>`: A new 1-D ndarray of the specified size, filled with the given integer.

    ## Equivalent
    >>> np.full(size, value, dtype="int64")
    """

# . range
def arr_clamp(
    arr: np.ndarray,
    minimum: int,
    maximum: int,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Clamp the values of a 1-D ndarray between 'minimum' and 'maximum' value `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param minimum `<'int'>`: The minimum value to clamp to.
    :param maximum `<'int'>`: The maximum value to clamp to.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice:
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.clip(arr, minimum, maximum) + offset
    """

def arr_min(
    arr: np.ndarray,
    value: int,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Take the elementwise minimum of a 1-D array with the scalar 'value' `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param value <'int'>: The scalar to compare against.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.minimum(arr, value) + offset
    """

def arr_max(
    arr: np.ndarray,
    value: int,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Take the elementwise maximum of a 1-D array with the scalar 'value' `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param value <'int'>: The scalar to compare against.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.maximum(arr, value) + offset
    """

def arr_min_arr(
    arr1: np.ndarray,
    arr2: np.ndarray,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Take the elementwise minimum of two 1-D arrays <'ndarray[int64]'>.

    :param arr1 `<'np.ndarray'>`: The first 1-D array. If dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param arr2 `<'np.ndarray'>`: The second 1-D array. Same casting rules as 'arr1'.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify `arr1` in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) propagate: if either operand is NaT, the result is NaT.
      LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.minimum(arr1, arr2) + offset
    """

def arr_max_arr(
    arr1: np.ndarray,
    arr2: np.ndarray,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Take the elementwise maximum of two 1-D arrays <'ndarray[int64]'>.

    :param arr1 `<'np.ndarray'>`: The first 1-D array.
    :param arr2 `<'np.ndarray'>`: The second 1-D array.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify `arr1` in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) propagate: if either operand is NaT, the result is NaT.
      LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.maximum(arr1, arr2) + offset
    """

# . arithmetic
def arr_abs(
    arr: np.ndarray, offset: int = 0, copy: bool = True
) -> np.ndarray[np.int64]:
    """(cfunc) Take the elementwise absolute of a 1-D array `<'ndarray[int64]'`>

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.abs(arr) + offset
    """

def arr_neg(
    arr: np.ndarray,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Negate the values of a 1-D array `<'ndarray[int64]'>`

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> -arr + offset
    """

def arr_add(arr: np.ndarray, value: int, copy: bool = True) -> np.ndarray[np.int64]:
    """(cfunc) Add a 'value' to the 1-D array `<'np.ndarray'>`

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param value `<'int'>`: The value to add.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr + value
    """

def arr_mul(
    arr: np.ndarray,
    factor: int,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Multiply the values of a 1-D array by the 'factor' `<'np.ndarray'>`

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: The scalar multiplier.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr * factor + offset
    """

def arr_mod(
    arr: np.ndarray,
    factor: int,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Elementwise modulo of a 1-D array by `factor` with Python semantics
    (remainder has the same sign as the divisor) `<'np.ndarray'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr % factor + offset
    """

def arr_div_even(
    arr: np.ndarray,
    factor: int,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Divide a 1-D array by `factor` and rounds to the nearest integers (half-to-even) <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.round(arr / factor, 0) + offset
    """

def arr_div_even_mul(
    arr: np.ndarray,
    factor: int,
    multiple: int,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Divide a 1-D array by `factor` and rounds to the nearest integers (half-to-even),
    then scale by `multiple` <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param multiple `<'int'>`: The scalar multiplier.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.round(arr / factor, 0) * multiple + offset
    """

def arr_div_up(
    arr: np.ndarray,
    factor: int,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Divide a 1-D array by `factor` and rounds to the nearest
    integers (half-up / away-from-zero) <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    """

def arr_div_up_mul(
    arr: np.ndarray,
    factor: int,
    multiple: int,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Divide a 1-D array by `factor` and rounds to nearest away from
    zero (half-up), then scale by 'multiple' <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param multiple `<'int'>`: The scalar multiplier.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    """

def arr_div_down(
    arr: np.ndarray,
    factor: int,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Divide a 1-D array by `factor` and rounds to the nearest
    integers (half-down / toward-zero) <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    """

def arr_div_down_mul(
    arr: np.ndarray,
    factor: int,
    multiple: int,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Divide a 1-D array by `factor` and rounds to the nearest integers (half-down),
    then scale by `multiple` <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param multiple `<'int'>`: The scalar multiplier.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    """

def arr_div_ceil(
    arr: np.ndarray,
    factor: int,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Divide a 1-D array by `factor` and ceils up to the nearest integers `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.ceil(arr / factor) + offset
    """

def arr_div_ceil_mul(
    arr: np.ndarray,
    factor: int,
    multiple: int,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Divide a 1-D array by `factor` and ceils up to the nearest integers,
    then scale by `multiple` <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param multiple `<'int'>`: The scalar multiplier.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.ceil(arr / factor) * multiple + offset
    """

def arr_div_floor(
    arr: np.ndarray,
    factor: int,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Divide a 1-D array by `factor` and floors down to the nearest integers `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.floor(arr / factor) + offset
    """

def arr_div_floor_mul(
    arr: np.ndarray,
    factor: int,
    multiple: int,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Divide a 1-D array by `factor` and floors down to the nearest integers,
    then scale by `multiple` <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param multiple `<'int'>`: The scalar multiplier.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> np.floor(arr / factor) * multiple + offset
    """

def arr_div_trunc(
    arr: np.ndarray,
    factor: int,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Divide a 1-D array by `factor` and and truncate toward zero `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    """

def arr_div_trunc_mul(
    arr: np.ndarray,
    factor: int,
    multiple: int,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Divide a 1-D array by `factor` and truncate toward zero,
    then scale by `multiple` <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param factor `<'int'>`: Divisor. Must be non-zero.
    :param multiple `<'int'>`: The scalar multiplier.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    """

def arr_add_arr(
    arr1: np.ndarray,
    arr2: np.ndarray,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Elementwise addition of two 1-D arrays <'ndarray[int64]'>.

    :param arr1 `<'np.ndarray'>`: The first 1-D array. If dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param arr2 `<'np.ndarray'>`: The second 1-D array. Same casting rules as arr1.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify `arr1` in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) propagate: if either operand is NaT, the result is NaT.
      LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr1 + arr2 + offset
    """

def arr_sub_arr(
    arr1: np.ndarray,
    arr2: np.ndarray,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Elementwise subtraction of two 1-D arrays <'ndarray[int64]'>.

    :param arr1 `<'np.ndarray'>`: The first 1-D array. If dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param arr2 `<'np.ndarray'>`: The second 1-D array. Same casting rules as 'arr1'.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify `arr1` in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) propagate: if either operand is NaT, the result is NaT.
      LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr1 - arr2 + offset
    """

def arr_mul_arr(
    arr1: np.ndarray,
    arr2: np.ndarray,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Elementwise multiplication of two 1-D arrays <'ndarray[int64]'>.

    :param arr1 `<'np.ndarray'>`: The first 1-D array. If dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param arr2 `<'np.ndarray'>`: The second 1-D array. Same casting rules as 'arr1'.
    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify `arr1` in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` and `offset` are interpreted
      in the array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) propagate: if either operand is NaT, the result is NaT.
      LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr1 * arr2 + offset
    """

# . comparison
def arr_eq(arr: np.ndarray, value: int) -> np.ndarray[bool]:
    """(cfunc) Elementwise equal comparison of a 1-D array to a scalar 'value' `<'ndarray[bool]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param value `<'int'>`: The scalar value to compare against.
    :returns `<'ndarray[bool]'>`: The boolean result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` is interpreted in the
      array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) always yeild `False`, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr == value
    """

def arr_gt(arr: np.ndarray, value: int) -> np.ndarray[bool]:
    """(cfunc) Elementwise greater-than comparison of a 1-D array to a scalar 'value' `<'ndarray[bool]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param value `<'int'>`: The scalar value to compare against.
    :returns `<'ndarray[bool]'>`: The boolean result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` is interpreted in the
      array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) always yeild `False`, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr > value
    """

def arr_ge(arr: np.ndarray, value: int) -> np.ndarray[bool]:
    """(cfunc) Elementwise greater-than-or-equal comparison of a 1-D array to a scalar 'value' `<'ndarray[bool]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param value `<'int'>`: The scalar value to compare against.
    :returns `<'ndarray[bool]'>`: The boolean result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` is interpreted in the
      array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) always yeild `False`, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr >= value
    """

def arr_lt(arr: np.ndarray, value: int) -> np.ndarray[bool]:
    """(cfunc) Elementwise less-than comparison of a 1-D array to a scalar 'value' `<'ndarray[bool]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param value `<'int'>`: The scalar value to compare against.
    :returns `<'ndarray[bool]'>`: The boolean result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` is interpreted in the
      array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) always yeild `False`, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr < value
    """

def arr_le(arr: np.ndarray, value: int) -> np.ndarray[bool]:
    """(cfunc) Elementwise less-than-or-equal comparison of a 1-D array to a scalar 'value' `<'ndarray[bool]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param value `<'int'>`: The scalar value to compare against.
    :returns `<'ndarray[bool]'>`: The boolean result array.

    ## Notice
    - For datetime64/timedelta64 inputs, `value` is interpreted in the
      array's underlying integer unit (e.g., ns for datetime64[ns]).
    - NaT values (LLONG_MIN) always yeild `False`, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr <= value
    """

def arr_eq_arr(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray[bool]:
    """(cfunc) Elementwise equal comparison of two 1-D arrays `<'ndarray[bool]'>`.

    :param arr1 `<'np.ndarray'>`: The first 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param arr2 `<'np.ndarray'>`: The second 1-D array. Same casting rules as 'arr1'.
    :returns `<'ndarray[bool]'>`: The boolean result array.

    ## Notice
    - For datetime64/timedelta64 inputs, comparison is performed on underlying int64 ticks.
    - NaT values (LLONG_MIN) always yeild `False`, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr1 == arr2
    """

def arr_gt_arr(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray[bool]:
    """(cfunc) Elementwise greater-than comparison of two 1-D arrays `<'ndarray[bool]'>`.

    :param arr1 `<'np.ndarray'>`: The first 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param arr2 `<'np.ndarray'>`: The second 1-D array. Same casting rules as 'arr1'.
    :returns `<'ndarray[bool]'>`: The boolean result array.

    ## Notice
    - For datetime64/timedelta64 inputs, comparison is performed on underlying int64 ticks.
    - NaT values (LLONG_MIN) always yeild `False`, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr1 > arr2
    """

def arr_ge_arr(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray[bool]:
    """(cfunc) Elementwise greater-than-or-equal comparison of two 1-D arrays `<'ndarray[bool]'>`.

    :param arr1 `<'np.ndarray'>`: The first 1-D array. If the dtype is not
        int64/datetime64/timedelta64, it will be cast to int64.
    :param arr2 `<'np.ndarray'>`: The second 1-D array. Same casting rules as 'arr1'.
    :returns `<'ndarray[bool]'>`: The boolean result array.

    ## Notice
    - For datetime64/timedelta64 inputs, comparison is performed on underlying int64 ticks.
    - NaT values (LLONG_MIN) always yeild `False`, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr1 >= arr2
    """

# NumPy: ndarray[datetime64] ---------------------------------------------------------------------------
# . type check
def is_dt64arr(arr: np.ndarray) -> bool:
    """(cfunc) Check if the array is dtype of 'datetime64[*]' `<'bool'>`.

    ## Equivalent
    >>> isinstance(arr.dtype, np.dtypes.DateTime64DType)
    """

def assure_dt64arr(arr: np.ndarray) -> bool:
    """(cfunc) Assure the array is dtype of 'datetime64[*]'."""

# . range check
def is_dt64arr_ns_safe(
    arr: np.ndarray,
    arr_reso: int = -1,
) -> bool:
    """(cfunc) Check if a 1-D array can be safely represented as datetime64[ns] `<'bool'>`.

    A value is **ns-safe** if, when interpreted using `arr_reso`,
    it lies strictly inside the datetime64[ns] valid range
    (1677-09-22 .. 2262-04-10, endpoints excluded).
    NaT (LLONG_MIN) is ignored.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :returns `<'bool'>`: True if all non-NaT values are ns-safe; otherwise False.
    """

# . access
def dt64arr_year(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract civil year numbers from a 1-D ndarray[datetime64[*]] `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: The civil year for each element.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype('datetime64[Y]').astype('int64') + 1970 + offset
    """

def dt64arr_quarter(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract quarter numbers (1..4) from a 1-D ndarray[datetime64[*]] `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: The quarter number for each element.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr = arr.astype('datetime64[M]').astype('int64')
        (arr % 12) // 3 + 1 + offset
    """

def dt64arr_month(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract month numbers (1..12) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: The month number for each element.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype('datetime64[M]').astype('int64') % 12 + 1 + offset
    """

def dt64arr_weekday(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract weekday numbers (0=Mon .. 6=Sun) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: The weekday number for each element.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> v = arr.astype('datetime64[D]').astype('int64')
        ((v % 7 + 7) % 7 + 3) % 7 + offset
    """

def dt64arr_day(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract day-of-month (1..31) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: The day-of-month for each element.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    """

def dt64arr_hour(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract hour-of-day (0..23) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: hour-of-day for each element.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype('datetime64[h]').astype('int64') % 24 + offset
    """

def dt64arr_minute(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract minute-of-hour (0..59) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: minute-of-hour for each element.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype('datetime64[m]').astype('int64') % 60 + offset
    """

def dt64arr_second(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract second-of-minute (0..59) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: second-of-minute for each element.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype('datetime64[s]').astype('int64') % 60 + offset
    """

def dt64arr_millisecond(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract millisecond-of-second (0..999) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: millisecond-of-second for each element.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype('datetime64[ms]').astype('int64') % 1000 + offset
    """

def dt64arr_microsecond(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """Extract microsecond-of-second (0..999,999) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: microsecond-of-second for each element.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype('datetime64[us]').astype('int64') % 1_000_000 + offset
    """

def dt64arr_nanosecond(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract nanosecond-of-microsecond (0..999) from a 1-D ndarray[datetime64[*]] <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: nanosecond-of-microsecond for each element.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype('datetime64[ns]').astype('int64') % 1000 + offset
    """

def dt64arr_times(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract time-of-day (elapsed since midnight) in the same unit as `arr` <'ndarray[int64]'>.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns <'ndarray[int64]'>: Time-of-day ticks for each element in the same unit.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    """

# . calendar
def dt64arr_isocalendar(
    arr: np.ndarray,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Return ISO-8601 calendar components for each element of a 1-D datetime64 array `<'ndarray[int64]'>`.

    Output is a (N, 3) int64 array with columns: [iso_year, iso_week, iso_weekday].
    ISO weekday is 1..7 for Mon..Sun. Week numbering follows ISO-8601:
    week 1 is the week containing January 4 (i.e., the first Thursday).
    The earilies date is clamp to '0001-01-01'

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: Shape (N, 3) array: [iso_year, iso_week, iso_weekday] per element.
        NaT values (LLONG_MIN) are propragated as the ISO calendar values.

    Output Example:
    >>> [[1936   11    7]
         [1936   12    1]
         [1936   12    2]
         ...
         [2003   42    6]
         [2003   42    7]
         [2003   43    1]]
    """

def dt64arr_isoyear(
    arr: np.ndarray,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract ISO-8601 year numbers from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The ISO year for each element.
        NaT values (LLONG_MIN) are preserved as the year values.
    """

def dt64arr_isoweek(
    arr: np.ndarray,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract ISO-8601 week numbers from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The ISO week for each element.
        NaT values (LLONG_MIN) are preserved as the week values.
    """

def dt64arr_isoweekday(
    arr: np.ndarray,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract ISO-8601 weekday numbers (1=Mon .. 7=Sun) from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The ISO weekday for each element.
        NaT values (LLONG_MIN) are preserved as the weekday values.
    """

def dt64arr_is_leap_year(
    arr: np.ndarray,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.bool_]:
    """(cfunc) Elementwise check for leap years of a 1-D datetime64 array `<'ndarray[bool]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[bool]'>`: Boolean indicator of leap year for each element.
        NaT values (LLONG_MIN) are emitted as `False`.
    """

def dt64arr_is_long_year(
    arr: np.ndarray,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.bool_]:
    """(cfunc) Extract ISO-8601 week numbers from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The ISO week for each element.
        NaT values (LLONG_MIN) are preserved as the week values.
    """

def dt64arr_leap_bt_years(
    arr: np.ndarray,
    year: int,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Compute the number of leap years between the target `year` and elements
    of a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param year `<'int'>`: The target year to compute against.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The total number of leap years for each element.
        NaT values (LLONG_MIN) are preserved as the leap years value.
    """

def dt64arr_days_in_year(
    arr: np.ndarray,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract the number of days in the year (365/366) from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The number of days in the year for each element.
        NaT values (LLONG_MIN) are preserved as the days-in-year value.
    """

def dt64arr_days_bf_year(
    arr: np.ndarray,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract the number of days strictly before January 1 of year under the
    proleptic Gregorian rules from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The number of days before year for each element.
        NaT values (LLONG_MIN) are preserved as the days-before-year value.
    """

def dt64arr_day_of_year(
    arr: np.ndarray,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract the 1-based ordinal day-of-year from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The day-of-year for each element.
        NaT values (LLONG_MIN) are preserved as the year values.
    """

def dt64arr_days_in_quarter(
    arr: np.ndarray,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract number of days in calendar quarter under the proleptic Gregorian rules
    from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The number of days in calendar quarter for each element.
        NaT values (LLONG_MIN) are preserved as the days values.
    """

def dt64arr_days_bf_quarter(
    arr: np.ndarray,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfucn) Extract number of days strictly before the first day of the
    calendar quarter under the proleptic Gregorian rules from a 1-D
    datetime64 array `<'ndarray[int64]'>`

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The number of days before calendar quarter for each element.
        NaT values (LLONG_MIN) are preserved as the days values.
    """

def dt64arr_day_of_quarter(
    arr: np.ndarray,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract the 1-based ordinal day-of-quarter from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The the number of days between the
        1st day of the quarter and the date for each element.
        NaT values (LLONG_MIN) are preserved as the day values.
    """

def dt64arr_days_in_month(
    arr: np.ndarray,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract number of days in calendar month under the proleptic Gregorian rules
    from a 1-D datetime64 array `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The number of days in calendar month for each element.
        NaT values (LLONG_MIN) are preserved as the days values.
    """

def dt64arr_days_bf_month(
    arr: np.ndarray,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Extract number of days strictly before the first day of the
    calendar month under the proleptic Gregorian rules from a 1-D
    datetime64 array `<'ndarray[int64]'>`

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'ndarray[int64]'>`: The number of days before calendar month for each element.
        NaT values (LLONG_MIN) are preserved as the days values.
    """

# . conversion: int64
def dt64arr_fr_int64(value: int, size: int, unit: str) -> np.ndarray[np.datetime64]:
    """(cfunc) Create a 1-D datetime64 array filled with the specifed integer `value` `<'ndarray[datetime64]'>`.

    :param value `<'int64'>`: The integer value of the datetime64 array.
    :param size `<'int'>`: The length of the datetime64 array.
    :param unit <'str'>: Time unit in its string form:
        'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'Y', etc.

    ## Equivalent
    >>> np.full(size, value, dtype=f"datetime64[{unit}]")
    """

def dt64arr_as_int64(
    arr: np.ndarray,
    as_unit: str,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Convert a 1-D ndarray[datetime64[*]] to int64 ticks in the requested unit `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param as_unit <'str'>: Target unit, supports:
        'ns','us','ms','s','m','h','D', 'W', 'M', 'Q', 'Y' and 'min' (alias for minutes).

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    - For 'W' (weeks) uses Thursday anchoring (same as Numpy).

    ## Equivalent
    >>> arr.astype(f"datetime64[{as_unit}]").astype("int64") + offset
    """

def dt64arr_as_int64_Y(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Convert a 1-D ndarray[datetime64[*]] to int64 year ticks (Y) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[Y]").astype("int64") + offset
    """

def dt64arr_as_int64_Q(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Convert a 1-D ndarray[datetime64[*]] to int64 quarter ticks (Q) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> (arr.astype("datetime64[M]").astype("int64") // 3) + offset
    """

def dt64arr_as_int64_M(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Convert a 1-D ndarray[datetime64[*]] to int64 month ticks (M) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[M]").astype("int64") + offset
    """

def dt64arr_as_int64_W(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Convert a 1-D ndarray[datetime64[*]] to int64 week ticks (W) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[W]").astype("int64") + offset
    """

def dt64arr_as_int64_D(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Convert a 1-D ndarray[datetime64[*]] to int64 day ticks (D) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[D]").astype("int64") + offset
    """

def dt64arr_as_int64_h(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Convert a 1-D ndarray[datetime64[*]] to int64 hour ticks (h) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[h]").astype("int64") + offset
    """

def dt64arr_as_int64_m(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Convert a 1-D ndarray[datetime64[*]] to int64 minute ticks (m) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[m]").astype("int64") + offset
    """

def dt64arr_as_int64_s(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Convert a 1-D ndarray[datetime64[*]] to int64 second ticks (s) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[s]").astype("int64") + offset
    """

def dt64arr_as_int64_ms(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Convert a 1-D ndarray[datetime64[*]] to int64 millisecond ticks (ms) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[ms]").astype("int64") + offset
    """

def dt64arr_as_int64_us(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Convert a 1-D ndarray[datetime64[*]] to int64 microsecond ticks (us) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[us]").astype("int64") + offset
    """

def dt64arr_as_int64_ns(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Convert a 1-D ndarray[datetime64[*]] to int64 nanosecond ticks (ns) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("datetime64[ns]").astype("int64") + offset
    """

def dt64arr_as_W_iso(
    arr: np.ndarray,
    weekday: int,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Convert a 1-D ndarray[datetime64[*]] to int64 week ticks (W) aligned to an ISO weekday `<'ndarray[int64]'>`.

    NumPy's datetime64[W] is **Thursday-anchored** (1970-01-01 is a Thursday).
    This function aligns weeks to any ISO weekday: 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat, 7=Sun.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param weekday <'int'>: ISO weekday to align to, value is clamp to [1, 7].
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    Equivalent to (conceptually):
    >>> arr = arr.astype("datetime64[D]").astype("int64")
        (arr + (4 - weekday)) // 7 + offset
    """

def dt64arr_to_ord(
    arr: np.ndarray,
    arr_reso: int = -1,
    offset: int = 0,
    copy: bool = True,
) -> np.ndarray[np.int64]:
    """(cfunc) Convert a 1-D ndarray[datetime64[*]] to int64 proleptic
    Gregorian ordinals `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.
    - Day 1 is '0001-01-01' (ordinal=1).

    ## Equivalent
    >>> arr.astype("datetime64[D]").astype("int64") + 719163 + offset
    """

# . conversion: float64
def dt64arr_to_ts(
    arr: np.ndarray,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.float64]:
    """(cfunc) Convert a 1-D ndarray[datetime64[*]] to float64 Unix timestamps `<'ndarray[float64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of datetime64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[float64]'>`: The result timestamps array.

    ## Notice
    - NaT values (LLONG_MIN) are emitted as NaN.
    - When down-converting from finer units (e.g., ns → us), sub-microsecond
      parts are **truncated** toward floor.

    ## Equivalent
    >>> arr.astype("datetime64[us]").astype("int64") / 1_000_000
    """

# . conversion: unit
def dt64arr_as_unit(
    arr: np.ndarray,
    as_unit: str,
    arr_reso: int = -1,
    limit: bool = False,
    copy: bool = True,
) -> np.ndarray[np.datetime64]:
    """(cfunc) Convert a 1-D datetime64 array to a specifc NumPy datetime64 unit `<'ndarray[datetime64]'>`

    :param arr `<'np.ndarray'>`: The datetime64 array to convert.
        Also support int64 array ticks along with `arr_reso` argument (see below).

    :param as_unit `<'str'>`: Traget datetime64 unit (e.g. `'ns'`, `'us'`, `'ms'`, `'s'`).

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param limit `<'bool'>`: Whether to limit conversions to sub-second units. Defaults to `False`.

        - If `False` (default), all native datetime64 conversions are supported.
        - If `True`, conversions are **restricted** to sub-second units only:
          `{"s", "ms", "us", "ns"}` for both source and target. A `ValueError`
          is raised if either side is outside that set.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'np.ndarray[datetime64]'>`: An array with dtype `datetime64[as_unit]`
        `NaT` values are preserved.

    ## Equivalent
    >>> arr.astype(f"datetime64[{as_unit}]")
    """

# . manipulation
def dt64arr_add_delta(
    arr: np.ndarray,
    years: int = 0,
    quarters: int = 0,
    months: int = 0,
    weeks: int = 0,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
    milliseconds: int = 0,
    microseconds: int = 0,
    nanoseconds: int = 0,
    arr_reso: int = -1,
) -> np.ndarray[np.datetime64]:
    """(cfunc) Add a mixed calendar/time delta to an ndarray[datetime64] `<'ndarray[datetime64'>`.

    Calendar components (years/quarters/months/weeks/days) are applied first on the
    date part; time components (hours…nanoseconds) are then added and normalized.

    :param arr `<'ndarray[datetime64'>`: Source datetime64 array.
        Supported resolutions: `'ns'`, `'us'`, `'ms'`, `'s'`, `'m'`, `'h'`, `'D'`.
    :param years `<'int'>`: Relative years.
    :param quarters `<'int'>`: Relative quarters (3 months).
    :param months `<'int'>`: Relative months.
    :param weeks `<'int'>`: Relative weeks (7 days).
    :param days `<'int'>`: Relative days.
    :param hours `<'int'>`: Relative hours.
    :param minutes `<'int'>`: Relative minutes.
    :param seconds `<'int'>`: Relative seconds.
    :param milliseconds `<'int'>`: Relative milliseconds (`1000 us`).
    :param microseconds `<'int'>`: Relative microseconds.
    :param nanoseconds `<'int'>`: Relative nanoseconds.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :returns `<'ndarray[datetime64[*]'>`: New datetime64 array at the same resolution as `arr`,
        `except` that for `'ns'` inputs the result may be down-casted to `'us'` if ns range
        would overflow.
    """

def dt64arr_replace_dates(
    arr: np.ndarray,
    year: int,
    month: int,
    day: int,
    arr_reso: int = -1,
) -> np.ndarray[np.int64]:
    """(cfunc) Replace the Y/M/D components of an ndarray[datetime64] and return `int64` day ticks `<'ndarray[int64]'>`.

    This function constructs new calendar dates by optionally replacing the
    `year`, `month`, and/or `day` fields of each element in the array.
    The result is returned as an `int64` array expressed in `days since epoch`
    (not `datetime64`).

    :param arr `<'ndarray'>`: Source `datetime64` or `int64` array (1-D).
    :param year `<'int64'>`: Set the year to this value when `> 0`; otherwise, retain the original year.
    :param month `<'int64'>`: Set the month to this value when `> 0`; otherwise, retain the original month.
    :param day `<'int64'>`: Set the day to this value when `> 0`; otherwise, retain the original day.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :returns `<'ndarray[int64]'>`: Integer `day` ticks (days since epoch)
        after applying the replacements (not in `datetime64` dtype).

    ## Behavior
    - Replacements are `component-wise`: any of the date fields may be left
      as `<= 0` to keep original values.
    - Day values are `clamped` to the maximum valid day in the resulting month.
    - The output is `int64` in day resolution, not `datetime64` dtype.
    """

def dt64arr_replace_times(
    arr: np.ndarray,
    hour: int,
    minute: int,
    second: int,
    microsecond: int,
    nanosecond: int,
    arr_reso: int = -1,
) -> np.ndarray[np.int64]:
    """(cfunc) Replace the h/m/s/us/ns components of an ndarray[datetime64] and return the
    time components `int64` ticks in the original resolution `<'ndarray[int64]'>`.

    This function constructs new time values by optionally replacing the
    `hour`, `minute`, `second`, `microsecond`, and/or `nanosecond` fields
    of each element in the array. The result is returned as an `int64` array
    representing the `time of day` ticks expressed in the original resolution
    (not `datetime64`).

    :param arr `<'ndarray'>`: Source `datetime64` or `int64` array (1-D).
    :param hour `<'int64'>`: Set the hour to this value when `>= 0`; otherwise, retain the original hour.
    :param minute `<'int64'>`: Set the minute to this value when `>= 0`; otherwise, retain the original minute.
    :param second `<'int64'>`: Set the second to this value when `>= 0`; otherwise, retain the original second.
    :param microsecond `<'int64'>`: Set the microsecond to this value when `>= 0`; otherwise, retain the original microsecond.
    :param nanosecond `<'int64'>`: Set the nanosecond to this value when `>= 0`; otherwise, retain the original nanosecond.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :returns `<'ndarray[int64]'>`: Integer `time-of-day` ticks (in original resolution)
        after applying the replacements (not in `datetime64` dtype).

    ## Behavior
    - Replacements are `component-wise`: any of the time fields may be left
      as `< 0` to keep original values.
    - Only time components within the array resolution are modified;
      e.g., when `arr` is in `'s'` resolution, `microsecond` and `nanosecond`
      replacements are ignored.
    - The output is time components `int64` ticks in the original resolution,
      not `datetime64` dtype.
    """

# . arithmetic
def dt64arr_round(
    arr: np.ndarray,
    to_unit: str,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.datetime64]:
    """(cfunc) Round a 1-D datetime64 array to the nearest multiple of `to_unit` (ties-to-even) `<'ndarray[datetime64]'>`

    This function will try to preserve the nanosecond datetime64 resolution
    when `safe`, and `downgrade` to microsecond unit only to avoid overflow.

    :param arr `<'np.ndarray'>`: The datetime64 array to round.
        Also support int64 array ticks along with `arr_reso` argument (see below).

    :param to_unit `<'str'>`: Target rounding granularity.
        Supports: `'ns', 'us', 'ms', 's', 'm', 'h', 'D'`.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'np.ndarray[datetime64]'>`: The rounded array.
        `NaT` values are preserved.
    """

def dt64arr_ceil(
    arr: np.ndarray,
    to_unit: str,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.datetime64]:
    """(cfunc) Ceil a 1-D datetime64 array to the nearest multiple of `to_unit` `<'ndarray[datetime64]'>`.

    This function will try to preserve the nanosecond datetime64 resolution
    when `safe`, and `downgrade` to microsecond unit only to avoid overflow.

    :param arr `<'np.ndarray'>`: The datetime64 array to ceil.
        Also support int64 array ticks along with `arr_reso` argument (see below).

    :param to_unit `<'str'>`: Target ceiling granularity.
        Supports: `'ns', 'us', 'ms', 's', 'm', 'h', 'D'`.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'np.ndarray[datetime64]'>`: The ceiled array.
        `NaT` values are preserved.
    """

def dt64arr_floor(
    arr: np.ndarray,
    to_unit: str,
    arr_reso: int = -1,
    copy: bool = True,
) -> np.ndarray[np.datetime64]:
    """(cfunc) Floor a 1-D datetime64 array to the nearest multiple of `to_unit` `<'ndarray[datetime64]'>`.

    This function will try to preserve the nanosecond datetime64 resolution
    when `safe`, and `downgrade` to microsecond unit only to avoid overflow.

    :param arr `<'np.ndarray'>`: The datetime64 array to floor.
        Also support int64 array ticks along with `arr_reso` argument (see below).

    :param to_unit `<'str'>`: Target flooring granularity.
        Supports: `'ns', 'us', 'ms', 's', 'm', 'h', 'D'`.

    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `datetime64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.

    :returns `<'np.ndarray[datetime64]'>`: The floored array.
        `NaT` values are preserved.
    """

# NumPy: ndarray[timedelta64] --------------------------------------------------------------------------
# . type check
def is_td64arr(arr: np.ndarray) -> bool:
    """(cfunc) Check if the array is dtype of 'timedelta64[*]' `<'bool'>`.

    ## Equivalent
    >>> isinstance(arr.dtype, np.dtypes.TimeDelta64DType)
    """

def assure_td64arr(arr: np.ndarray) -> bool:
    """(cfunc) Assure the array is dtype of 'timedelta64[*]'."""

# . conversion
def td64arr_as_int64_us(
    arr: np.ndarray,
    unit: int = -1,
    offset: int = 0,
) -> np.ndarray[np.int64]:
    """(cfunc) Convert a 1-D ndarray[timedelta64[*]] to int64 microsecond ticks (us) `<'ndarray[int64]'>`.

    :param arr `<'np.ndarray'>`: The 1-D array dtype of timedelta64[*] or int64.
    :param arr_reso `<'int'>`: The unit of `arr` as an `NPY_DATETIMEUNIT` enum value. Defaults to `-1`.

        - If not specified and `arr` is `timedelta64[*]`, the array's
          intrinsic resolution is used.
        - If `arr` dtype is int64, you **MUST** specify the `arr_reso`,
          and values are interpreted as ticks in that unit.

    :param offset `<'int'>`: Optional offset added after the operation. Defaults to `0`.
    :param copy `<'bool'>`: If True, operate on a copy.
        If False modify in place when possible. Defaults to `True`.
    :returns `<'ndarray[int64]'>`: The result array.

    ## Notice
    - NaT values (LLONG_MIN) are preserved and never modified, and LLONG_MIN is treated as NaT.

    ## Equivalent
    >>> arr.astype("timedelta64[us]").astype("int64") + offset
    """
