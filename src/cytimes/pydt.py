# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from __future__ import annotations

# Cython imports
import cython
from cython.cimports.libc import math  # type: ignore
from cython.cimports.libc.limits import LLONG_MAX  # type: ignore
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_READ_CHAR as str_read  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_GET_LENGTH as str_len  # type: ignore
from cython.cimports.cpython.dict import PyDict_GetItem as dict_getitem  # type: ignore
from cython.cimports.cytimes.parser import parse_dtobj as _parse, Configs, CONFIG_MONTH, CONFIG_WEEKDAY  # type: ignore
from cython.cimports.cytimes import typeref, utils  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
import datetime, numpy as np
from zoneinfo import available_timezones as _available_timezones
from cytimes.parser import Configs, parse_dtobj as _parse
from cytimes import typeref, utils, errors

__all__ = ["is_pydt", "pydt_new", "pydt_fr_dt", "pydt_fr_dtobj", "Pydt"]


# Utils ---------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_pydt(o: object) -> cython.bint:
    """(cfunc) Check if the object is an instance of 'Pydt' `<'bool'>`."""
    return isinstance(o, _Pydt)


@cython.cfunc
@cython.inline(True)
def pydt_new(
    year: cython.int = 1,
    month: cython.int = 1,
    day: cython.int = 1,
    hour: cython.int = 0,
    minute: cython.int = 0,
    second: cython.int = 0,
    microsecond: cython.int = 0,
    tz: datetime.tzinfo | str | None = None,
    fold: cython.int = 0,
) -> _Pydt:
    """(cfunc) Create a new Pydt datetime `<'Pydt'>`.

    :param year `<'int'>`: Year value (1-9999), defaults to `1`.
    :param month `<'int'>`: Month value (1-12), defaults to `1`.
    :param day `<'int'>`: Day value (1-31), defaults to `1`.
    :param hour `<'int'>`: Hour value (0-23), defaults to `0`.
    :param minute `<'int'>`: Minute value (0-59), defaults to `0`.
    :param second `<'int'>`: Second value (0-59), defaults to `0`.
    :param microsecond `<'int'>`: Microsecond value (0-999999), defaults to `0`.
    :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
        1. `<'datetime.tzinfo'>` subclass of datetime.tzinfo.
        2. `<'str'>` timezone name supported by 'Zoneinfo' module, or 'local' for local timezone.
        3. `<'NoneType'>` timezone-naive.

    :param fold `<'int'>`: Fold value (0/1), defaults to `0`.
    """
    # Normalize non-fixed timezone
    tz = utils.tz_parse(tz)
    if tz is not None and type(tz) is not typeref.TIMEZONE:
        dt: datetime.datetime = datetime.datetime_new(
            year, month, day, hour, minute, second, microsecond, tz, fold
        )
        try:
            dt_norm = utils.dt_normalize_tz(dt)
        except ValueError as err:
            raise errors.AmbiguousTimezoneError("%s" % err) from err
        if dt is not dt_norm:
            year, month, day = dt_norm.year, dt_norm.month, dt_norm.day
            hour, minute, second = dt_norm.hour, dt_norm.minute, dt_norm.second
            microsecond, fold = dt_norm.microsecond, 0

    # Create Pydt
    if fold == 1:
        return _Pydt.__new__(
            Pydt, year, month, day, hour, minute, second, microsecond, tz, fold=1
        )
    else:
        return _Pydt.__new__(
            Pydt, year, month, day, hour, minute, second, microsecond, tz
        )


@cython.cfunc
@cython.inline(True)
def pydt_fr_dt(dt: datetime.datetime) -> _Pydt:
    """(cfunc) Create a new Pydt from an existing datetime `<'Pydt'>`.

    :param dt `<'datetime'>`: Instance or subclass of datetime.datetime.
    """
    # Normalize non-fixed timezone
    tz = dt.tzinfo
    if tz is not None and type(tz) is not typeref.TIMEZONE:
        # Normalize non-fixed timezone
        try:
            dt_norm = utils.dt_normalize_tz(dt)
        except ValueError as err:
            raise errors.AmbiguousTimezoneError("%s" % err) from err
        if dt is not dt_norm:
            dt = dt_norm

    # Create Pydt
    yy, mm, dd = dt.year, dt.month, dt.day
    hh, mi, ss = dt.hour, dt.minute, dt.second
    us, fold = dt.microsecond, dt.fold
    if fold == 1:
        return _Pydt.__new__(Pydt, yy, mm, dd, hh, mi, ss, us, tz, fold=1)
    else:
        return _Pydt.__new__(Pydt, yy, mm, dd, hh, mi, ss, us, tz)


@cython.cfunc
@cython.inline(True)
def pydt_fr_dtobj(
    dtobj: object,
    default: object | None = None,
    year1st: bool | None = None,
    day1st: bool | None = None,
    ignoretz: cython.bint = False,
    isoformat: cython.bint = True,
    cfg: Configs = None,
) -> _Pydt:
    """(cfunc) Parse datetime-related object into `<'Pydt'>.

    For more information, please refer to 'cytimes.Pydt.parse()' classmethod.
    """
    # Parse default
    if default is not None:
        default = _parse_dtobj(default, None, year1st, day1st, ignoretz, isoformat, cfg)

    # Parse datetime-related object
    dt: datetime.datetime = _parse_dtobj(
        dtobj, default, year1st, day1st, ignoretz, isoformat, cfg
    )
    return pydt_fr_dt(dt)


@cython.cfunc
@cython.inline(True)
def _parse_dtobj(
    dtobj: object,
    default: object | None = None,
    year1st: bool | None = None,
    day1st: bool | None = None,
    ignoretz: cython.bint = False,
    isoformat: cython.bint = True,
    cfg: Configs = None,
) -> datetime.datetime:
    """(cfunc) Parse datetime-related object into `<'datetime.datetime'>.

    For more information, please refer to 'cytimes.parser.parse_dtobj()' function.
    """
    try:
        return _parse(dtobj, default, year1st, day1st, ignoretz, isoformat, cfg)
    except Exception as err:
        raise errors.InvalidDatetimeError("%s" % err) from err


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-2, check=False)
def _parse_month(month: object, raise_error: cython.bint) -> cython.int:
    """(cfunc) Parse the 'month' value to an integer (1=Jan...12=Dec) `<'int'>`.

    :param month `<'int/str'>`: Supports both integer and month string as input.
    :param raise_error `<'bool'>`: Whether to raise error if the 'month' value is invalid.

    ### Note:
    - This function always returns `-1` if 'month' is None or -1.
    - If 'raise_error=True', raises `InvalidMonthError` for invalid
      'month' value. Else, returns `-1` instead.
    """
    # <'NoneType'>
    if month is None:
        return -1  # exit

    # <'str'>
    if isinstance(month, str):
        mth: str = month
        val = dict_getitem(CONFIG_MONTH, mth.lower())
        if val == cython.NULL:
            if raise_error:
                raise errors.InvalidMonthError("invalid month string '%s'." % mth)
            return -1  # eixt
        return cython.cast(object, val)  # exit

    # <'int'>
    if isinstance(month, int):
        num: cython.longlong = month
        if num == -1:
            return -1  # exit
        if not 1 <= num <= 12:
            if raise_error:
                raise errors.InvalidMonthError(
                    "invalid month value '%d', must betweem 1(Jan)...12(Dec)." % num
                )
            return -1  # exit
        return num  # exit

    # Invalid
    if raise_error:
        raise errors.InvalidMonthError(
            "unsupported month type %s, expects <'str/int'>." % type(month)
        )
    return -1


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-2, check=False)
def _parse_weekday(weekday: object, raise_error: cython.bint) -> cython.int:
    """(cfunc) Parse the 'weekday' value to an integer (0=Mon...6=Sun) `<'int'>`.

    :param weekday `<'int/str'>`: Supports both integer and weekday string as input.
    :param raise_error `<'bool'>`: Whether to raise error if the 'weekday' value is invalid.

    ### Note:
    - This function always returns `-1` if 'weekday' is None or -1.
    - If 'raise_error=True', raises `InvalidWeekdayError` for invalid
      'weekday' value. Else, returns `-1` instead.
    """
    # <'NoneType'>
    if weekday is None:
        return -1

    # <'str'>
    if isinstance(weekday, str):
        wkd: str = weekday
        val = dict_getitem(CONFIG_WEEKDAY, wkd.lower())
        if val == cython.NULL:
            if raise_error:
                raise errors.InvalidWeekdayError("invalid weekday string '%s'." % wkd)
            return -1
        return cython.cast(object, val)  # exit

    # <'int'>
    if isinstance(weekday, int):
        num: cython.longlong = weekday
        if num == -1:
            return -1  # exit
        if not 0 <= num <= 6:
            if raise_error:
                raise errors.InvalidWeekdayError(
                    "invalid weekday value '%d', must betweem 0(Mon)...6(Sun)." % num
                )
            return -1
        return num  # exit

    # Invalid
    if raise_error:
        raise errors.InvalidWeekdayError(
            "unsupported weekday type %s, expects <'str/int'>." % type(weekday)
        )
    return -1


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-2, check=False)
def _parse_frequency(freq: str, raise_error: cython.bint) -> cython.longlong:
    """(cfunc) Parse the 'frequency' value to the corresponding frequency divisor `<'int'>`.

    :param freq `<'str'>`: Supported frequency ['D', 'h', 'm', 's', 'ms', 'us'].
    :param raise_error `<'bool'>`: Whether to raise error if the 'freq' value is invalid.

    ### Note:
    - If 'raise_error=True', raises `InvalidFrequencyError` for invalid
      'freq' value. Else, returns `-1` instead.
    """
    freq_len: cython.Py_ssize_t = str_len(freq)
    # 'us', 'ms'
    if freq_len == 2:
        freq_ch: cython.Py_UCS4 = str_read(freq, 0)
        if freq_ch == "u":
            return 1
        if freq_ch == "m":
            return 1_000

    # 's', 'm', 'h'
    elif freq_len == 1:
        freq_ch: cython.Py_UCS4 = str_read(freq, 0)
        if freq_ch == "s":
            return 1_000_000
        if freq_ch == "m":
            return 60_000_000
        if freq_ch == "h":
            return utils.US_HOUR
        if freq_ch in ("D", "d"):
            return utils.US_DAY

    # Invalid
    if raise_error:
        raise errors.InvalidFrequencyError(
            "invalid frequency '%s'.\nSupported frequency: "
            "['D', 'h', 'm', 's', 'ms', 'us']." % freq
        )
    return -1


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-2, check=False)
def _compare_dts(
    dt1: datetime.datetime,
    dt2: datetime.datetime,
    allow_mixed: cython.bint = False,
) -> cython.int:
    """(cfunc) Comparison between two datetimes `<'int'>`.

    :param dt1 `<'datetime.datetime'>`: Instance or subclass of datetime.datetime.
    :param dt2 `<'datetime.datetime'>`: Instance or subclass of datetime.datetime.
    :param allow_mixed `<'bool'>`:
    """
    # Timezone naive & aware mixed
    d1_tz, d2_tz = dt1.tzinfo, dt2.tzinfo
    if d1_tz is not d2_tz and (d1_tz is None or d2_tz is None):
        if not allow_mixed:
            _raise_incomparable_error(dt1, dt2, "compare")
        return 2

    # Comparison
    utc: cython.bint = d1_tz is not None
    d1_us: cython.longlong = utils.dt_to_us(dt1, utc)
    d2_us: cython.longlong = utils.dt_to_us(dt2, utc)
    if d1_us > d2_us:
        return 1
    elif d1_us < d2_us:
        return -1
    else:
        return 0


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-2, check=False)
def _raise_incomparable_error(
    dt1: datetime.datetime,
    dt2: datetime.datetime,
    msg: str = "compare",
) -> cython.bint:
    """(cfunc) Raise `IncomparableError` for comparison
    between timezone-naive & timezone-aware datetimes.

    :param dt1 `<'datetime.datetime'>`: Instance or subclass of datetime.datetime
    :param dt2 `<'datetime.datetime'>`: Instance or subclass of datetime.datetime been compared with
    :param msg `<'str'>`: The custom message for the exception: 'cannot [msg] between'. Defaults to `'compare'`.
    """
    d1_tz, d2_tz = dt1.tzinfo, dt2.tzinfo
    assert d1_tz is not d2_tz and (d1_tz is None or d2_tz is None)
    if d1_tz is None:
        raise errors.IncomparableError(
            "cannot %s between naive & aware datetimes:\n"
            "Timezone-naive '%s' %s\n"
            "Timezone-aware '%s' %s" % (msg, dt1, type(dt1), dt2, type(dt2))
        )
    else:
        raise errors.IncomparableError(
            "cannot %s between naive & aware datetimes:\n"
            "Timezone-aware '%s' %s\n"
            "Timezone-naive '%s' %s" % (msg, dt1, type(dt1), dt2, type(dt2))
        )


# Pydt (Python Datetime) ----------------------------------------------------------------------
@cython.cclass
class _Pydt(datetime.datetime):
    """The base class for `<'Pydt'>`, a subclass of the cpython `<'datetime.datetime'>`.

    ### Do `NOT` instantiate this class directly.
    """

    # Constructor --------------------------------------------------------------------------
    @classmethod
    def parse(
        cls,
        dtobj: object,
        default: object | None = None,
        year1st: bool | None = None,
        day1st: bool | None = None,
        ignoretz: cython.bint = False,
        isoformat: cython.bint = True,
        cfg: Configs = None,
    ) -> _Pydt:
        """Parse from datetime-related object `<'Pydt'>`.

        :param dtobj `<'object'>`: Datetime related object.
            1. `<'str'>` datetime string that contains datetime information.
            2. `<'datetime.datetime'>` instance or subclass of datetime.datetime.
            3. `<'datetime.date'>` instance or subclass of datetime.date. All time values set to 0.
            4. `<'int/float'>` numeric values, treated as total seconds since Unix Epoch.
            5. `<'np.datetime64'>` resolution above 'us' will be discarded.
            6. `<'NoneType'>` passing 'None' returns current local datetime.

        ## Arguments below only take effects when 'dtobj' is type of `<'str'>`.

        :param default `<'datetime/date'>`: The default to fill-in missing datetime values, defaults to `None`.
            1. `<'date/datetime'>`: If parser failed to extract Y/M/D values from the string,
              the give 'default' will be used to fill-in the missing Y/M/D values.
            2. `None`: raise `PaserBuildError` if any Y/M/D values are missing.

        :param year1st `<'bool/None'>`: Interpret the first ambiguous Y/M/D value as year, defaults to `None`.
            When 'year1st=None', use `çfg.year1st` if 'cfg' is specified, else `False` as default.

        :param day1st `<'bool/None'>`: Interpret the first ambiguous Y/M/D values as day, defaults to `None`.
            When 'day1st=None', use `çfg.day1st` if 'cfg' is specified, else `False` as default.

        :param ignoretz `<'bool'>`: Whether to ignore timezone information, defaults to `False`.
            1. `True`: Parser ignores any timezone information and only returns
              timezone-naive datetime. Setting to `True` can increase parser
              performance.
            2. `False`: Parser will try to process the timzone information in
              the string, and generate a timezone-aware datetime if timezone
              has been matched by 'cfg.utc' & 'cfg.tz'.

        :param isoformat `<'bool'>`: Whether to parse 'dtstr' as ISO format, defaults to `True`.
            1. `True`: Parser will first try to process the 'dtstr' as ISO format.
              If failed, fallback to process the 'dtstr' through timelex tokens.
              For most datetime strings, this approach yields the best performance.
            2. `False`: Parser will only process the 'dtstr' through timelex tokens.
              If the 'dtstr' is confirmed not an ISO format, setting to `False`
              can increase parser performance.

        :param cfg `<'Configs/None'>`: The custom Parser configurations, defaults to `None`.

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
        """
        return pydt_fr_dtobj(dtobj, default, year1st, day1st, ignoretz, isoformat, cfg)

    @classmethod
    def now(cls, tz: datetime.tzinfo | str | None = None) -> _Pydt:
        """Get current datetime with optional timezone `<'Pydt'>`.

        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            1. `<'datetime.tzinfo'>` subclass of datetime.tzinfo.
            2. `<'str'>` timezone name supported by 'Zoneinfo' module,
               or `'local'` for local timezone.
            3. `<'NoneType'>` timezone-naive.
        """
        return pydt_fr_dt(utils.dt_now(utils.tz_parse(tz)))

    @classmethod
    def utcnow(cls) -> _Pydt:
        """Get current UTC datetime (timezone-aware) `<'Pydt'>`."""
        return pydt_fr_dt(utils.dt_now(utils.UTC))

    @classmethod
    def today(cls) -> _Pydt:
        """Get current local datetime (timezone-naive) `<'Pydt'>`."""
        return pydt_fr_dt(utils.dt_now(None))

    @classmethod
    def combine(
        cls,
        date: datetime.date | str | None = None,
        time: datetime.time | str | None = None,
        tz: datetime.tzinfo | str | None = None,
    ) -> _Pydt:
        """Combine date and time into a new datetime `<'Pydt'>`.

        :param date `<'date/str/None'>`: Instance or subclass of datetime.date or a date string, defaults to `None`.
            If 'date=None', use current local date.

        :param time `<'time/str/None'>`: Instance or subclass of datetime.time or a time string, defaults to `None`.
            If 'time=None', all time fields set to 0.

        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            1. `<'datetime.tzinfo'>` subclass of datetime.tzinfo.
            2. `<'str'>` timezone name supported by 'Zoneinfo' module,
               or `'local'` for local timezone.
            3. `<'NoneType'>` timezone-naive.
        """
        # Parse date
        if date is not None and not utils.is_date(date):
            date = pydt_fr_dtobj(date)

        # Prase time
        if time is not None and not utils.is_time(time):
            time = pydt_fr_dtobj(time, datetime.date_new(1970, 1, 1)).timetz()

        # Create Pydt
        return pydt_fr_dt(utils.dt_combine(date, time, utils.tz_parse(tz)))

    @classmethod
    def fromordinal(
        cls,
        ordinal: cython.int,
        tz: datetime.tzinfo | str | None = None,
    ) -> _Pydt:
        """Construct from a proleptic Gregorian ordinal with optional timzone `<'Pydt'>`.

        :param ordinal `<'int'>`: The proleptic Gregorian ordinal, '0001-01-01' is day 1 (ordinal=1).
        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            1. `<'datetime.tzinfo'>` subclass of datetime.tzinfo.
            2. `<'str'>` timezone name supported by 'Zoneinfo' module,
               or `'local'` for local timezone.
            3. `<'NoneType'>` timezone-naive.
        """
        _ymd = utils.ymd_fr_ordinal(ordinal)
        return pydt_new(_ymd.year, _ymd.month, _ymd.day, 0, 0, 0, 0, tz, 0)

    @classmethod
    def fromseconds(
        cls,
        seconds: int | float,
        tz: datetime.tzinfo | str | None = None,
    ) -> _Pydt:
        """Construct from total seconds since Unix Epoch with optional timezone `<'Pydt'>`.

        This method does `NOT` take local timezone into consideration
        when constructing the datetime, which is the main difference
        from the 'fromtimestamp() method.

        :param seconds `<'int/float'>`: Seconds since Unix Epoch.
        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            1. `<'datetime.tzinfo'>` subclass of datetime.tzinfo.
            2. `<'str'>` timezone name supported by 'Zoneinfo' module,
               or `'local'` for local timezone.
            3. `<'NoneType'>` timezone-naive.
        """
        return pydt_fr_dt(utils.dt_fr_seconds(float(seconds), utils.tz_parse(tz)))

    @classmethod
    def fromicroseconds(
        cls,
        us: cython.longlong,
        tz: datetime.tzinfo | str | None = None,
    ) -> _Pydt:
        """Construct from total microseconds since Unix Epoch with optional timezone `<'Pydt'>`.

        This method does `NOT` take local timezone into consideration
        when constructing the datetime.

        :param us `<'int'>`: Microseconds since Unix Epoch.
        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            1. `<'datetime.tzinfo'>` subclass of datetime.tzinfo.
            2. `<'str'>` timezone name supported by 'Zoneinfo' module,
               or `'local'` for local timezone.
            3. `<'NoneType'>` timezone-naive.
        """
        return pydt_fr_dt(utils.dt_fr_us(us, utils.tz_parse(tz)))

    @classmethod
    def fromtimestamp(
        cls,
        ts: int | float,
        tz: datetime.tzinfo | str | None = None,
    ) -> _Pydt:
        """Construct from a POSIX timestamp with optional timezone `<'Pydt'>`.

        :param ts `<'int/float'>`: POSIX timestamp.
        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            1. `<'datetime.tzinfo'>` subclass of datetime.tzinfo.
            2. `<'str'>` timezone name supported by 'Zoneinfo' module,
               or `'local'` for local timezone.
            3. `<'NoneType'>` timezone-naive.
        """
        return pydt_fr_dt(utils.dt_fr_ts(float(ts), utils.tz_parse(tz)))

    @classmethod
    def utcfromtimestamp(cls, ts: int | float) -> _Pydt:
        """Construct an UTC datetime (timezone-aware) from a POSIX timestamp `<'Pydt'>`.

        :param ts `<'int/float'>`: POSIX timestamp.
        """
        return pydt_fr_dt(utils.dt_fr_ts(float(ts), utils.UTC))

    @classmethod
    def fromisoformat(cls, dtstr: str) -> _Pydt:
        """Construct from ISO format string `<'Pydt'>`.

        :param dtstr `<'str'>`: The ISO format datetime string.
        """
        return pydt_fr_dt(datetime.datetime.fromisoformat(dtstr))

    @classmethod
    def fromisocalendar(
        cls,
        year: cython.int,
        week: cython.int,
        day: cython.int,
        tz: datetime.tzinfo | str | None = None,
    ) -> _Pydt:
        """Construct from the ISO year, week number and weekday, with optional timezone `<'Pydt'>`.

        :param year `<'int'>`: The ISO year.
        :param week `<'int'>`: The ISO week number (1-53).
        :param day `<'int'>`: The ISO weekday (1=Mon...7=Sun).
        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            1. `<'datetime.tzinfo'>` subclass of datetime.tzinfo.
            2. `<'str'>` timezone name supported by 'Zoneinfo' module,
               or `'local'` for local timezone.
            3. `<'NoneType'>` timezone-naive.
        """
        _ymd = utils.ymd_fr_isocalendar(year, week, day)
        return pydt_new(_ymd.year, _ymd.month, _ymd.day, 0, 0, 0, 0, tz, 0)

    @classmethod
    def fromdate(
        cls,
        date: datetime.date,
        tz: datetime.tzinfo | str | None = None,
    ) -> _Pydt:
        """Construct from a date instance, all time fields set to 0 `<'Pydt'>`.

        :param date `<'datetime.date'>`: Instance or subclass of datetime.date.
        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            1. `<'datetime.tzinfo'>` subclass of datetime.tzinfo.
            2. `<'str'>` timezone name supported by 'Zoneinfo' module,
               or `'local'` for local timezone.
            3. `<'NoneType'>` timezone-naive.
        """
        # fmt: off
        return pydt_new(
            datetime.date_year(date),
            datetime.date_month(date),
            datetime.date_day(date),
            0, 0, 0, 0, tz, 0
        )
        # fmt: on

    @classmethod
    def fromdatetime(cls, dt: datetime.datetime) -> _Pydt:
        """Construct from a datetime instance `<'Pydt'>`.

        :param dt `<'datetime.datetime'>`: Instance or subclass of datetime.datetime.
        """
        return pydt_fr_dt(dt)

    @classmethod
    def fromdatetime64(
        cls,
        dt64: object,
        tz: datetime.tzinfo | str | None = None,
    ) -> _Pydt:
        """Construct from a numpy.datetime64 instance `<'Pydt'>`.

        :param dt64 `<'datetime64'>`: A numpy.datetime64 object.
        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            1. `<'datetime.tzinfo'>` subclass of datetime.tzinfo.
            2. `<'str'>` timezone name supported by 'Zoneinfo' module,
               or `'local'` for local timezone.
            3. `<'NoneType'>` timezone-naive.
        """
        return pydt_fr_dt(utils.dt64_to_dt(dt64, utils.tz_parse(tz)))

    @classmethod
    def strptime(cls, dtstr: str, fmt: str) -> _Pydt:
        """Construct from a datetime string with the given format `<'Pydt'>`.

        :param dtstr `<'str'>`: The datetime string.
        :param format `<'str'>`: The format of the datetime string.
        """
        return pydt_fr_dt(datetime.datetime.strptime(dtstr, fmt))

    @cython.cfunc
    @cython.inline(True)
    def _from_dt(self, dt: datetime.datetime) -> _Pydt:
        """(cfunc) Construct from the passed in datetime `<'Pydt'>`.

        This internal method checks if the passed in datetime is the same
        object as the current instance (self). If so, returns the instance
        directly; otherwise, creates a new Pydt instance from the datetime.

        :param dt `<'datetime.datetime'>`: Instance or subclass of datetime.datetime.
        """
        return self if dt is self else pydt_fr_dt(dt)

    # Convertor ----------------------------------------------------------------------------
    @cython.ccall
    def ctime(self) -> str:
        """Convert to ctime style stirng `<'str'>`.

        ### Example:
        >>> dt.ctime()
        >>> "Tue Oct  1 08:19:05 2024"
        """
        yy: cython.int = datetime.datetime_year(self)
        mm: cython.int = datetime.datetime_month(self)
        dd: cython.int = datetime.datetime_day(self)
        hh: cython.int = datetime.datetime_hour(self)
        mi: cython.int = datetime.datetime_minute(self)
        ss: cython.int = datetime.datetime_second(self)
        # Weekday
        wkd = utils.ymd_weekday(yy, mm, dd)
        if wkd == 0:
            weekday = "Mon"
        elif wkd == 1:
            weekday = "Tue"
        elif wkd == 2:
            weekday = "Wed"
        elif wkd == 3:
            weekday = "Thu"
        elif wkd == 4:
            weekday = "Fri"
        elif wkd == 5:
            weekday = "Sat"
        else:
            weekday = "Sun"
        # Month
        if mm == 1:
            month = "Jan"
        elif mm == 2:
            month = "Feb"
        elif mm == 3:
            month = "Mar"
        elif mm == 4:
            month = "Apr"
        elif mm == 5:
            month = "May"
        elif mm == 6:
            month = "Jun"
        elif mm == 7:
            month = "Jul"
        elif mm == 8:
            month = "Aug"
        elif mm == 9:
            month = "Sep"
        elif mm == 10:
            month = "Oct"
        elif mm == 11:
            month = "Nov"
        else:
            month = "Dec"
        # Format
        return "%s %s %2d %02d:%02d:%02d %04d" % (weekday, month, dd, hh, mi, ss, yy)

    @cython.ccall
    def strftime(self, fmt: str) -> str:
        """Convert to string based on the given format `<'str'>`.

        :param format `<'str'>`: The format of the datetime string.
        """
        return utils.dt_to_strformat(self, fmt)

    @cython.ccall
    def isoformat(self, sep: str = "T") -> str:
        """Convert to an ISO 8601 formatted string `<'str'>`.

        The default format is 'YYYY-MM-DDTHH:MM:SS[.f]' with an optional fractional
        part when 'microsecond' is non-zero. If 'tzinfo' is present, the UTC offset
        is included, resulting in 'YYYY-MM-DDTHH:MM:SS[.f]+HH:MM'.

        :param sep `<'str'>`: The separator between date and time components, defaults to `'T'`.
        """
        return utils.dt_to_isoformat(self, sep, True)

    @cython.ccall
    def timedict(self) -> utils.tm:
        """Convert to local time as a dictionary `<'dict'>`.

        ### Example:
        >>> dt.timedict()
        >>> {
                'tm_sec': 11,
                'tm_min': 14,
                'tm_hour': 8,
                'tm_mday': 11,
                'tm_mon': 10,
                'tm_year': 2024,
                'tm_wday': 4,
                'tm_yday': 285,
                'tm_isdst': 1
            }
        """
        return utils.dt_to_tm(self, False)

    @cython.ccall
    def utctimedict(self) -> utils.tm:
        """Convert to UTC time as a dictionary `<'dict'>`.

        ### Example:
        >>> dt.utctimedict()
        >>> {
                'tm_sec': 6,
                'tm_min': 15,
                'tm_hour': 6,
                'tm_mday': 11,
                'tm_mon': 10,
                'tm_year': 2024,
                'tm_wday': 4,
                'tm_yday': 285,
                'tm_isdst': 0
            }
        """
        return utils.dt_to_tm(self, True)

    @cython.ccall
    def timetuple(self) -> tuple[int, ...]:
        """Convert to local time as a tuple `<'tuple'>`.

        #### Note: this method returns 'tuple' instead of 'time.struct_time'.

        ### Example:
        >>> dt.timetuple()
        >>> (2024, 10, 11, 8, 18, 10, 4, 285, 1)
        """
        _tm = utils.dt_to_tm(self, False)
        return (
            _tm.tm_year,
            _tm.tm_mon,
            _tm.tm_mday,
            _tm.tm_hour,
            _tm.tm_min,
            _tm.tm_sec,
            _tm.tm_wday,
            _tm.tm_yday,
            _tm.tm_isdst,
        )

    @cython.ccall
    def utctimetuple(self) -> tuple[int, ...]:
        """Convert to UTC time as a tuple `<'tuple'>`.

        #### Note: this method returns 'tuple' instead of 'time.struct_time'.

        ### Example:
        >>> dt.utctimetuple()
        >>> (2024, 10, 11, 6, 20, 12, 4, 285, 0)
        """
        _tm = utils.dt_to_tm(self, True)
        return (
            _tm.tm_year,
            _tm.tm_mon,
            _tm.tm_mday,
            _tm.tm_hour,
            _tm.tm_min,
            _tm.tm_sec,
            _tm.tm_wday,
            _tm.tm_yday,
            _tm.tm_isdst,
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def toordinal(self) -> cython.int:
        """Convert to proleptic Gregorian ordinal `<'int'>`.

        '0001-01-01' is day 1 (ordinal=1).
        """
        return utils.dt_to_ordinal(self)

    @cython.ccall
    def seconds(self, utc: cython.bint = False) -> cython.double:
        """Convert to total seconds since Unix Epoch `<'float'>`.

        This method does `NOT` take local timezone into consideration
        when converting timezone-naive datetime to seconds, which is
        the main difference from the 'timestamp()' method.

        :param utc `<'bool'>`: Whether to adjust (subtract) utcoffset, defaults to `False`.
            Only applicable when datetime is timezone-aware.
        """
        return utils.dt_to_seconds(self, utc)

    @cython.ccall
    def microseconds(self, utc: cython.bint = False) -> cython.longlong:
        """Convert to total microseconds since Unix Epoch `<'int'>`.

        This method does `NOT` take local timezone into consideration
        when converting timezone-naive datetime to microseconds.

        :param utc `<'bool'>`: Whether to adjust (subtract) utcoffset, defaults to `False`.
            Only applicable when datetime is timezone-aware.
        """
        return utils.dt_to_us(self, utc)

    @cython.ccall
    def timestamp(self) -> cython.double:
        """Convert to POSIX timestamp `<'float'>`."""
        return utils.dt_to_ts(self)

    @cython.ccall
    def date(self) -> datetime.date:
        """Convert date fields to a date `<'datetime.date'>`."""
        return datetime.date_new(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
        )

    @cython.ccall
    def time(self) -> datetime.time:
        """Convert time fields to a time (`WITHOUT` tzinfo) `<'datetime.time'>`."""
        return datetime.time_new(
            datetime.datetime_hour(self),
            datetime.datetime_minute(self),
            datetime.datetime_second(self),
            datetime.datetime_microsecond(self),
            None,
            datetime.datetime_fold(self),
        )

    @cython.ccall
    def timetz(self) -> datetime.time:
        """Convert time fields to a time (`WITH` tzinfo) `<'datetime.time'>`."""
        return datetime.time_new(
            datetime.datetime_hour(self),
            datetime.datetime_minute(self),
            datetime.datetime_second(self),
            datetime.datetime_microsecond(self),
            datetime.datetime_tzinfo(self),
            datetime.datetime_fold(self),
        )

    # Manipulator --------------------------------------------------------------------------
    @cython.ccall
    def replace(
        self,
        year: cython.int = -1,
        month: cython.int = -1,
        day: cython.int = -1,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        microsecond: cython.int = -1,
        tz: datetime.tzinfo | str | None = -1,
        fold: cython.int = -1,
    ) -> _Pydt:
        """Replace the specified fields with new values `<'Pydt'>`.

        #### Value of '-1' means keep the original field value.

        :param year `<'int'>`: The year value (1-9999), defaults to `-1`.
        :param month `<'int'>`: The month value (1-12), defaults to `-1`.
        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            Automacially clamp by the maximum days in the month.

        :param hour `<'int'>`: The hour value (0-23), defaults to `-1`.
        :param minute `<'int'>`: The minute value (0-59), defaults to `-1`.
        :param second `<'int'>`: The second value (0-59), defaults to `-1`.
        :param microsecond `<'int'>`: The microsecond value (0-999999), defaults to `-1`.
        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `-1`.
            1. `<'int'>` keep the original timezone.
            2. `<'datetime.tzinfo'>` subclass of datetime.tzinfo.
            2. `<'str'>` timezone name supported by 'Zoneinfo' module,
               or `'local'` for local timezone.
            4. `<'NoneType'>` timezone-naive.

        :param fold `<'int'>`: The fold value (0/1), defaults to `-1`.
        """
        # Prase timezone
        if not isinstance(tz, int):
            tz = utils.tz_parse(tz)

        # Replace
        # fmt: off
        return self._from_dt(
            utils.dt_replace(
                self, year, month, day, 
                hour, minute, second, -1, 
                microsecond, tz, fold,
            )
        )
        # fmt: on

    # . year
    @cython.ccall
    def to_curr_year(
        self,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> _Pydt:
        """Adjust the date fields to the specified month
        and day of the current year `<'Pydt'>`.

        :param month `<'int/str/None'>`: The month value, defaults to `None`.
            1. `<'int'>` (1=Jan...12=Dec).
            2. `<'str'>` month name (case-insensitive) such as 'Jan', 'februar', '三月', etc.
            3. `<'NoneType'>` keep the original month value.

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            Value of '-1' means keep the original day value. The final
            day value will be clamped by the maximum days in the month.

        ### Example:
        >>> dt.to_curr_year("Feb", 31)  # The last day of February of the current year
        >>> dt.to_curr_year(11)  # The same day of November of the current year
        >>> dt.to_curr_year(day=1)  # The first day of the current month and year
        """
        # Parse month
        mm: cython.int = _parse_month(month, True)
        if mm == -1 or mm == datetime.datetime_month(self):
            return self.to_curr_month(day)  # exit: same month
        yy: cython.int = datetime.datetime_year(self)

        # Clamp by max days
        dd: cython.int = datetime.datetime_day(self) if day < 1 else day
        if dd > 28:
            dd = min(dd, utils.days_in_month(yy, mm))

        # Generate new datetime
        return pydt_new(
            yy,
            mm,
            dd,
            datetime.datetime_hour(self),
            datetime.datetime_minute(self),
            datetime.datetime_second(self),
            datetime.datetime_microsecond(self),
            datetime.datetime_tzinfo(self),
            datetime.datetime_fold(self),
        )

    @cython.ccall
    def to_prev_year(
        self,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> _Pydt:
        """Adjust the date fields to the specified month
        and day of the previous year `<'Pydt'>`.

        :param month `<'int/str/None'>`: The month value, defaults to `None`.
            1. `<'int'>` (1=Jan...12=Dec).
            2. `<'str'>` month name (case-insensitive) such as 'Jan', 'februar', '三月', etc.
            3. `<'NoneType'>` keep the original month value.

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            Value of '-1' means keep the original day value. The final
            day value will be clamped by the maximum days in the month.

        ### Example:
        >>> dt.to_prev_year("Feb", 31)  # The last day of February of the previous year
        >>> dt.to_prev_year(11)  # The same day of November of the previous year
        >>> dt.to_prev_year(day=1)  # The first day of the current month of the previous year
        """
        return self.to_year(-1, month, day)

    @cython.ccall
    def to_next_year(
        self,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> _Pydt:
        """Adjust the date fields to the specified month
        and day of the next year `<'Pydt'>`.

        :param month `<'int/str/None'>`: The month value, defaults to `None`.
            1. `<'int'>` (1=Jan...12=Dec).
            2. `<'str'>` month name (case-insensitive) such as 'Jan', 'februar', '三月', etc.
            3. `<'NoneType'>` keep the original month value.

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            Value of '-1' means keep the original day value. The final
            day value will be clamped by the maximum days in the month.

        ### Example:
        >>> dt.to_next_year("Feb", 31)  # The last day of February of the next year
        >>> dt.to_next_year(11)  # The same day of November of the next year
        >>> dt.to_next_year(day=1)  # The first day of the current month of the next year
        """
        return self.to_year(1, month, day)

    @cython.ccall
    def to_year(
        self,
        offset: cython.int,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> _Pydt:
        """Adjust the date fields to the specified month
        and day of year (+/-) 'offset' `<'Pydt'>`.

        :param offset `<'int'>`: The year offset (+/-).

        :param month `<'int/str/None'>`: The month value, defaults to `None`.
            1. `<'int'>` (1=Jan...12=Dec).
            2. `<'str'>` month name (case-insensitive) such as 'Jan', 'februar', '三月', etc.
            3. `<'NoneType'>` keep the original month value.

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            Value of '-1' means keep the original day value. The final
            day value will be clamped by the maximum days in the month.

        ### Example:
        >>> dt.to_year(-2, "Feb", 31)  # The last day of February 2 years ago
        >>> dt.to_year(2, 11)  # The same day of November 2 years later
        >>> dt.to_year(2, day=1)  # The first day of the current month 2 years later
        """
        # No offset
        if offset == 0:
            return self.to_curr_year(month, day)  # exit

        # Calculate new year
        yy: cython.int = datetime.datetime_year(self) + offset
        yy = min(max(yy, 1), 9999)

        # Parse month
        mm: cython.int = _parse_month(month, True)
        if mm == -1:
            mm = datetime.datetime_month(self)

        # Clamp by max days
        dd: cython.int = datetime.datetime_day(self) if day < 1 else day
        if dd > 28:
            dd = min(dd, utils.days_in_month(yy, mm))

        # Generate new datetime
        return pydt_new(
            yy,
            mm,
            dd,
            datetime.datetime_hour(self),
            datetime.datetime_minute(self),
            datetime.datetime_second(self),
            datetime.datetime_microsecond(self),
            datetime.datetime_tzinfo(self),
            datetime.datetime_fold(self),
        )

    # . quarter
    @cython.ccall
    def to_curr_quarter(self, month: cython.int = -1, day: cython.int = -1) -> _Pydt:
        """Adjust the date fields to the specified month
        and day of the current quarter. `<'Pydt'>`.

        :param month `<'int'>`: The month (1-3) of the quarter, defaults to `-1`.
            Value of '-1' means keep the original month of the quarter.

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            Value of '-1' means keep the original day value. The final
            day value will be clamped by the maximum days in the month.

        ### Example:
        >>> dt.to_curr_quarter(1, 31)  # The last day of the first month of the current quarter
        >>> dt.to_curr_quarter(2)  # The same day of the second month of the current quarter
        >>> dt.to_curr_quarter(day=1)  # The first day of the current month of the current quarter
        """
        # No adjustment
        if month < 1:
            return self.to_curr_month(day)  # exit

        # Calculate new month
        curr_mm: cython.int = datetime.datetime_month(self)
        mm = utils.quarter_of_month(curr_mm) * 3 - 3 + (month % 3 or 3)
        if mm == curr_mm:
            return self.to_curr_month(day)  # exit: same month
        yy: cython.int = datetime.datetime_year(self)

        # Clamp by max days
        dd: cython.int = datetime.datetime_day(self) if day < 1 else day
        if dd > 28:
            dd = min(dd, utils.days_in_month(yy, mm))

        # Generate new datetime
        return pydt_new(
            yy,
            mm,
            dd,
            datetime.datetime_hour(self),
            datetime.datetime_minute(self),
            datetime.datetime_second(self),
            datetime.datetime_microsecond(self),
            datetime.datetime_tzinfo(self),
            datetime.datetime_fold(self),
        )

    @cython.ccall
    def to_prev_quarter(self, month: cython.int = -1, day: cython.int = -1) -> _Pydt:
        """Adjust the date fields to the specified month
        and day of the previous quarter. `<'Pydt'>`.

        :param month `<'int'>`: The month (1-3) of the previous quarter, defaults to `-1`.
            Value of '-1' means keep the original month of the quarter.

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            Value of '-1' means keep the original day value. The final
            day value will be clamped by the maximum days in the month.

        ### Example:
        >>> dt.to_prev_quarter(1, 31)  # The last day of the first month of the previous quarter
        >>> dt.to_prev_quarter(2)  # The same day of the second month of the previous quarter
        >>> dt.to_prev_quarter(day=1)  # The first day of the current month of the previous quarter
        """
        return self.to_quarter(-1, month, day)

    @cython.ccall
    def to_next_quarter(self, month: cython.int = -1, day: cython.int = -1) -> _Pydt:
        """Adjust the date fields to the specified month
        and day of the next quarter. `<'Pydt'>`.

        :param month `<'int'>`: The month (1-3) of the next quarter, defaults to `-1`.
            Value of '-1' means keep the original month of the quarter.

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            Value of '-1' means keep the original day value. The final
            day value will be clamped by the maximum days in the month.

        ### Example:
        >>> dt.to_next_quarter(1, 31)  # The last day of the first month of the next quarter
        >>> dt.to_next_quarter(2)  # The same day of the second month of the next quarter
        >>> dt.to_next_quarter(day=1)  # The first day of the current month of the next quarter
        """
        return self.to_quarter(1, month, day)

    @cython.ccall
    def to_quarter(
        self,
        offset: cython.int,
        month: cython.int = -1,
        day: cython.int = -1,
    ) -> _Pydt:
        """Adjust the date fields to the specified month
        and day of the quarter (+/-) 'offset'. `<'Pydt'>`.

        :param offset `<'int'>`: The quarter offset (+/-).

        :param month `<'int'>`: The month (1-3) of the quarter, defaults to `-1`.
            Value of '-1' means keep the original month of the quarter.

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            Value of '-1' means keep the original day value. The final
            day value will be clamped by the maximum days in the month.

        ### Example:
        >>> dt.to_quarter(-2, 1, 31)  # The last day of the first month 2 quarters ago
        >>> dt.to_quarter(2, 2)  # The same day of the second month 2 quarters later
        >>> dt.to_quarter(2, day=1)  # The first day of the current month 2 quarters later
        """
        # No offset
        if offset == 0:
            return self.to_curr_quarter(month, day)  # exit

        # Calculate new year & month
        yy: cython.int = datetime.datetime_year(self)
        mm: cython.int = datetime.datetime_month(self)
        if month >= 1:
            mm = utils.quarter_of_month(mm) * 3 - 3 + (month % 3 or 3)
        mm += offset * 3
        if mm > 12:
            yy += mm // 12
            mm %= 12
        elif mm < 1:
            mm = 12 - mm
            yy -= mm // 12
            mm = 12 - mm % 12
        yy = min(max(yy, 1), 9999)
        # Clamp by max days
        dd: cython.int = datetime.datetime_day(self) if day < 1 else day
        if dd > 28:
            dd = min(dd, utils.days_in_month(yy, mm))

        # Generate new datetime
        return pydt_new(
            yy,
            mm,
            dd,
            datetime.datetime_hour(self),
            datetime.datetime_minute(self),
            datetime.datetime_second(self),
            datetime.datetime_microsecond(self),
            datetime.datetime_tzinfo(self),
            datetime.datetime_fold(self),
        )

    # . month
    @cython.ccall
    def to_curr_month(self, day: cython.int = -1) -> _Pydt:
        """Adjust the date fields to the specified day of the current month `<'Pydt'>`.

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            Value of '-1' means keep the original day value. The final
            day value will be clamped by the maximum days in the month.

        ### Example:
        >>> dt.to_curr_month(31)  # The last day of the current month
        >>> dt.to_curr_month(1)  # The first day of the current month
        """
        # No adjustment
        if day < 1:
            return self  # exit

        # Clamp by max days
        yy: cython.int = datetime.datetime_year(self)
        mm: cython.int = datetime.datetime_month(self)
        if day > 28:
            day = min(day, utils.days_in_month(yy, mm))

        # Compare with current day
        if day == datetime.datetime_day(self):
            return self  # exit: same day

        # Generate new datetime
        return pydt_new(
            yy,
            mm,
            day,
            datetime.datetime_hour(self),
            datetime.datetime_minute(self),
            datetime.datetime_second(self),
            datetime.datetime_microsecond(self),
            datetime.datetime_tzinfo(self),
            datetime.datetime_fold(self),
        )

    @cython.ccall
    def to_prev_month(self, day: cython.int = -1) -> _Pydt:
        """Adjust the date fields to the specified day of the previous month `<'Pydt'>`.

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            Value of '-1' means keep the original day value. The final
            day value will be clamped by the maximum days in the month.

        ### Example:
        >>> dt.to_prev_month(31)  # The last day of the previous month
        >>> dt.to_prev_month(1)  # The first day of the previous month
        """
        return self.to_month(-1, day)

    @cython.ccall
    def to_next_month(self, day: cython.int = -1) -> _Pydt:
        """Adjust the date fields to the specified day of the next month `<'Pydt'>`.

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            Value of '-1' means keep the original day value. The final
            day value will be clamped by the maximum days in the month.

        ### Example:
        >>> dt.to_prev_month(31)  # The last day of the next month
        >>> dt.to_prev_month(1)  # The first day of the next month
        """
        return self.to_month(1, day)

    @cython.ccall
    def to_month(self, offset: cython.int, day: cython.int = -1) -> _Pydt:
        """Adjust the date fields to the specified day of month (+/-) 'offset' `<'Pydt'>`.

        :param offset `<'int'>`: The month offset (+/-).

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            Value of '-1' means keep the original day value. The final
            day value will be clamped by the maximum days in the month.

        ### Example:
        >>> dt.to_month(-2, 31)  # The last day of the month 2 months ago
        >>> dt.to_month(2, 1)  # The first day of the month 2 months later
        """
        # No offset
        if offset == 0:
            return self.to_curr_month(day)  # exit

        # Calculate new year & month
        yy: cython.int = datetime.datetime_year(self)
        mm: cython.int = datetime.datetime_month(self) + offset
        if mm > 12:
            yy += mm // 12
            mm %= 12
        elif mm < 1:
            mm = 12 - mm
            yy -= mm // 12
            mm = 12 - mm % 12
        yy = min(max(yy, 1), 9999)

        # Calculate new day
        dd: cython.int = datetime.datetime_day(self) if day < 1 else day
        if dd > 28:
            dd = min(dd, utils.days_in_month(yy, mm))

        # Generate new datetime
        return pydt_new(
            yy,
            mm,
            dd,
            datetime.datetime_hour(self),
            datetime.datetime_minute(self),
            datetime.datetime_second(self),
            datetime.datetime_microsecond(self),
            datetime.datetime_tzinfo(self),
            datetime.datetime_fold(self),
        )

    # . weekday
    @cython.ccall
    def to_monday(self) -> _Pydt:
        """Adjust the date fields to the Monday of the current week `<'Pydt'>`."""
        return self._to_curr_weekday(0)

    @cython.ccall
    def to_tuesday(self) -> _Pydt:
        """Adjust the date fields to the Tuesday of the current week `<'Pydt'>`."""
        return self._to_curr_weekday(1)

    @cython.ccall
    def to_wednesday(self) -> _Pydt:
        """Adjust the date fields to the Wednesday of the current week `<'Pydt'>`."""
        return self._to_curr_weekday(2)

    @cython.ccall
    def to_thursday(self) -> _Pydt:
        """Adjust the date fields to the Thursday of the current week `<'Pydt'>`."""
        return self._to_curr_weekday(3)

    @cython.ccall
    def to_friday(self) -> _Pydt:
        """Adjust the date fields to the Friday of the current week `<'Pydt'>`."""
        return self._to_curr_weekday(4)

    @cython.ccall
    def to_saturday(self) -> _Pydt:
        """Adjust the date fields to the Saturday of the current week `<'Pydt'>`."""
        return self._to_curr_weekday(5)

    @cython.ccall
    def to_sunday(self) -> _Pydt:
        """Adjust the date fields to the Sunday of the current week `<'Pydt'>`."""
        return self._to_curr_weekday(6)

    @cython.ccall
    def to_curr_weekday(self, weekday: int | str = None) -> _Pydt:
        """Adjust the date fields to the specific weekday
        of the current week `<'Pydt'>`.

        :param weekday `<'int/str/None'>`: The weekday value, defaults to `None`.
            1. `<'int'>` (0=Mon...6=Sun).
            2. `<'str'>` weekday name (case-insensitive) such as 'Mon', 'dienstag', '星期三', etc.
            3. `<'NoneType'>` keep the original weekday value.

        ### Example:
        >>> dt.to_curr_weekday(0)  # The Monday of the current week
        >>> dt.to_curr_weekday("Tue")  # The Tuesday of the current week
        """
        return self._to_curr_weekday(_parse_weekday(weekday, True))

    @cython.ccall
    def to_prev_weekday(self, weekday: int | str = None) -> _Pydt:
        """Adjust the date fields to the specific weekday
        of the previous week `<'Pydt'>`.

        :param weekday `<'int/str/None'>`: The weekday value, defaults to `None`.
            1. `<'int'>` (0=Mon...6=Sun).
            2. `<'str'>` weekday name (case-insensitive) such as 'Mon', 'dienstag', '星期三', etc.
            3. `<'NoneType'>` keep the original weekday value.

        ### Example:
        >>> dt.to_prev_weekday(0)  # The Monday of the previous week
        >>> dt.to_prev_weekday("Tue")  # The Tuesday of the previous week
        """
        return self.to_weekday(-1, weekday)

    @cython.ccall
    def to_next_weekday(self, weekday: int | str = None) -> _Pydt:
        """Adjust the date fields to the specific weekday
        of the next week `<'Pydt'>`.

        :param weekday `<'int/str/None'>`: The weekday value, defaults to `None`.
            1. `<'int'>` (0=Mon...6=Sun).
            2. `<'str'>` weekday name (case-insensitive) such as 'Mon', 'dienstag', '星期三', etc.
            3. `<'NoneType'>` keep the original weekday value.

        ### Example:
        >>> dt.to_next_weekday(0)  # The Monday of the next week
        >>> dt.to_next_weekday("Tue")  # The Tuesday of the next week
        """
        return self.to_weekday(1, weekday)

    @cython.ccall
    def to_weekday(self, offset: cython.int, weekday: int | str | None = None) -> _Pydt:
        """Adjust the date fields to the specific weekday
        of week (+/-) 'offset' `<'Pydt'>`.

        :param offset `<'int'>`: The week offset (+/-).

        :param weekday `<'int/str/None'>`: The weekday value, defaults to `None`.
            1. `<'int'>` (0=Mon...6=Sun).
            2. `<'str'>` weekday name (case-insensitive) such as 'Mon', 'dienstag', '星期三', etc.
            3. `<'NoneType'>` keep the original weekday value.

        ### Example:
        >>> dt.to_weekday(-2, 0)  # The Monday of the week 2 weeks ago
        >>> dt.to_weekday(2, "Tue")  # The Tuesday of the week 2 weeks later
        >>> dt.to_weekday(2)  # The same weekday of the week 2 weeks later
        """
        # Parse weekday
        wkd: cython.int = _parse_weekday(weekday, True)

        # No offset
        if offset == 0:
            return self._to_curr_weekday(wkd)  # exit

        # Calculate new weekday
        curr_wkd: cython.int = self.weekday()
        delta: cython.int = offset * 7
        if wkd != -1:
            delta += wkd - curr_wkd

        # Generate new datetime
        return pydt_fr_dt(utils.dt_add(self, days=delta))

    @cython.cfunc
    @cython.inline(True)
    def _to_curr_weekday(self, weekday: cython.int) -> _Pydt:
        """(cfunc) Adjust the date fields to the specific weekday of the current week `<'Pydt'>`.

        :param weekday `<'int'>`: The weekday value (0=Mon...6=Sun).
        """
        # Check weekday
        curr_wkd: cython.int = self.weekday()
        if weekday == curr_wkd:
            return self  # exit: same weekday

        # Generate new datetime
        return pydt_fr_dt(utils.dt_add(self, days=weekday - curr_wkd))

    # . day
    @cython.ccall
    def to_yesterday(self) -> _Pydt:
        """Adjust the date fields to yesterday `<'Pydt'>`."""
        return pydt_fr_dt(utils.dt_add(self, days=-1))

    @cython.ccall
    def to_tomorrow(self) -> _Pydt:
        """Adjust the date fields to tomorrow `<'Pydt'>`."""
        return pydt_fr_dt(utils.dt_add(self, days=1))

    @cython.ccall
    def to_day(self, offset: cython.int) -> _Pydt:
        """Adjust the date fields to day (+/-) 'offset' `<'Pydt'>`.

        :param offset `<'int'>`: The day offset (+/-).
        """
        # No offset
        if offset == 0:
            return self

        # Generate new datetime
        return pydt_fr_dt(utils.dt_add(self, days=offset))

    # . date&time
    @cython.ccall
    def to_datetime(
        self,
        year: cython.int = -1,
        month: cython.int = -1,
        day: cython.int = -1,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        millisecond: cython.int = -1,
        microsecond: cython.int = -1,
    ) -> _Pydt:
        """Adjust (replace) the date and time fields with new values `<'Pydt'>`.

        #### Value of '-1' means keep the original field value.

        :param year `<'int'>`: The year value (1-9999), defaults to `-1`.
        :param month `<'int'>`: The month value (1-12), defaults to `-1`.
        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            Automacially clamp by the maximum days in the month.

        :param hour `<'int'>`: The hour value (0-23), defaults to `-1`.
        :param minute `<'int'>`: The minute value (0-59), defaults to `-1`.
        :param second `<'int'>`: The second value (0-59), defaults to `-1`.
        :param millisecond `<'int'>`: The millisecond value (0-999), defaults to `-1`.
        :param microsecond `<'int'>`: The microsecond value (0-999999), defaults to `-1`.
        """
        # fmt: off
        return self._from_dt(
            utils.dt_replace(
                self, year, month, day,
                hour, minute, second,
                millisecond, microsecond, -1, -1,
            )
        )
        # fmt: on

    @cython.ccall
    def to_date(
        self,
        year: cython.int = -1,
        month: cython.int = -1,
        day: cython.int = -1,
    ) -> _Pydt:
        """Adjust (replace) the specified date fields with new values `<'Pydt'>`.

        #### Value of '-1' means keep the original field value.

        :param year `<'int'>`: The year value (1-9999), defaults to `-1`.
        :param month `<'int'>`: The month value (1-12), defaults to `-1`.
        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            Automacially clamp by the maximum days in the month.
        """
        return self._from_dt(utils.dt_replace_date(self, year, month, day))

    @cython.ccall
    def to_time(
        self,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        millisecond: cython.int = -1,
        microsecond: cython.int = -1,
    ) -> _Pydt:
        """Adjust (replace) the specified time fields with new values `<'Pydt'>`.

        #### Value of '-1' means keep the original field value.

        :param hour `<'int'>`: The hour value (0-23), defaults to `-1`.
        :param minute `<'int'>`: The minute value (0-59), defaults to `-1`.
        :param second `<'int'>`: The second value (0-59), defaults to `-1`.
        :param millisecond `<'int'>`: The millisecond value (0-999), defaults to `-1`.
        :param microsecond `<'int'>`: The microsecond value (0-999999), defaults to `-1`.
        """
        # fmt: off
        return self._from_dt(
            utils.dt_replace_time(
                self, hour, minute, second, 
                millisecond, microsecond,
            )
        )
        # fmt: on

    @cython.ccall
    def to_first_of(self, unit: str) -> _Pydt:
        """Adjust the date fields to the first day of the specified 'unit' `<'Pydt'>`.

        :param unit `<'str'>`: The time unit.

        - `'Y'`: date set to the 1st day of the current year.
        - `'Q'`: date set to the 1st day of the current quarter.
        - `'M'`: date set to the 1st day of the current month.
        - `'W'`: date set to Monday of the current week.
        - `'Month name'` such as 'Jan', 'februar', '三月': date set to the 1st day of that month.
        """
        unit_len: cython.Py_ssize_t = str_len(unit)
        if unit_len == 1:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . weekday
            if unit_ch == "W":
                return self._to_curr_weekday(0)
            # . month
            if unit_ch == "M":
                return self.to_curr_month(1)
            # . quarter
            if unit_ch == "Q":
                return self.to_curr_quarter(1, 1)
            # . year
            if unit_ch == "Y":
                return self.to_date(-1, 1, 1)

        # . month name
        val: cython.int = _parse_month(unit, False)
        if val != -1:
            return self.to_date(-1, val, 1)

        # Invalid
        raise errors.InvalidTimeUnitError(
            "invalid 'first of' time unit '%s'.\nSupported time unit: "
            "['Y', 'Q', 'M', 'W'] or Month name." % unit
        )

    @cython.ccall
    def to_last_of(self, unit: str) -> _Pydt:
        """Adjust the date fields to the last day of the specified 'unit' `<'Pydt'>`.

        :param unit `<'str'>`: The time unit.

        - `'Y'`: date set to the last day of the current year.
        - `'Q'`: date set to the last day of the current quarter.
        - `'M'`: date set to the last day of the current month.
        - `'W'`: date set to Sunday of the current week.
        - `'Month name'` such as 'Jan', 'februar', '三月': date set to the last day of that month.
        """
        unit_len: cython.Py_ssize_t = str_len(unit)
        if unit_len == 1:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . weekday
            if unit_ch == "W":
                return self._to_curr_weekday(6)
            # . month
            if unit_ch == "M":
                return self.to_curr_month(31)
            # . quarter
            if unit_ch == "Q":
                return self.to_curr_quarter(3, 31)
            # . year
            if unit_ch == "Y":
                return self.to_date(-1, 12, 31)

        # . month name
        val: cython.int = _parse_month(unit, False)
        if val != -1:
            return self.to_date(-1, val, 31)

        # Invalid
        raise errors.InvalidTimeUnitError(
            "invalid 'last of' time unit '%s'.\nSupported time unit: "
            "['Y', 'Q', 'M', 'W'] or Month name." % unit
        )

    @cython.ccall
    def to_start_of(self, unit: str) -> _Pydt:
        """Adjust the date and time fields to the start of the specified 'unit' `<'Pydt'>`.

        :param unit `<'str'>`: The time unit.

        - `'Y'`: date set to the 1st day of the current year & time set to '00:00:00.000000'.
        - `'Q'`: date set to the 1st day of the current quarter & time set to '00:00:00.000000'.
        - `'M'`: date set to the 1st day of the current month & time set to '00:00:00.000000'.
        - `'W'`: date set to Monday of the current week & time set to '00:00:00.000000'.
        - `'D'`: original date & time set to '00:00:00.000000'.
        - `'h'`: original date & time set to 'XX:00:00.000000'.
        - `'m'`: original date & time set to 'XX:XX:00.000000'.
        - `'s'`: original date & time set to 'XX:XX:XX.000000'.
        - `'ms'`: original date & time set to 'XX:XX:XX.XXX000'.
        - `'Month name'` such as 'Jan', 'februar', '三月': date set to 1st day of that month & time set to '00:00:00.000000'.
        - `'Weekday name'` such as 'Mon', 'dienstag', '星期三': date set to that weekday & time set to '00:00:00.000000'.
        """
        unit_len: cython.Py_ssize_t = str_len(unit)
        if unit_len == 1:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . second
            if unit_ch == "s":
                return self.to_time(-1, -1, -1, -1, 0)
            # . minute
            if unit_ch == "m":
                return self.to_time(-1, -1, 0, -1, 0)
            # . hour
            if unit_ch == "h":
                return self.to_time(-1, 0, 0, -1, 0)
            # . day
            if unit_ch == "D":
                return self.to_time(0, 0, 0, -1, 0)
            # . week
            if unit_ch == "W":
                # fmt: off
                return self.add(
                    0, 0, 0, 0,
                    -self.weekday(),
                    -datetime.datetime_hour(self),
                    -datetime.datetime_minute(self),
                    -datetime.datetime_second(self), 0,
                    -datetime.datetime_microsecond(self),
                )
                # fmt: on
            # . month
            if unit_ch == "M":
                return self.to_datetime(-1, -1, 1, 0, 0, 0, -1, 0)
            # . quarter
            if unit_ch == "Q":
                mm: cython.int = utils.quarter_1st_month(datetime.datetime_month(self))
                return self.to_datetime(-1, mm, 1, 0, 0, 0, -1, 0)
            # . year
            if unit_ch == "Y":
                return self.to_datetime(-1, 1, 1, 0, 0, 0, -1, 0)
        elif unit_len == 2:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . millisecond
            if unit_ch == "m":
                return self.to_time(
                    -1, -1, -1, -1, datetime.datetime_microsecond(self) // 1000 * 1000
                )

        # . month name
        val: cython.int = _parse_month(unit, False)
        if val != -1:
            return self.to_datetime(-1, val, 1, 0, 0, 0, -1, 0)

        # . weekday name
        val: cython.int = _parse_weekday(unit, False)
        if val != -1:
            # fmt: off
            return self.add(
                0, 0, 0, 0,
                val - self.weekday(),
                -datetime.datetime_hour(self),
                -datetime.datetime_minute(self),
                -datetime.datetime_second(self), 0,
                -datetime.datetime_microsecond(self),
            )
            # fmt: on

        # Invalid
        raise errors.InvalidTimeUnitError(
            "invalid 'start of' time unit '%s'.\nSupported time unit: "
            "['Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms'] "
            "or Month/Weekday name." % unit
        )

    @cython.ccall
    def to_end_of(self, unit: str) -> _Pydt:
        """Adjust the date and time fields to the end of the specified 'unit' `<'Pydt'>`.

        :param unit `<'str'>`: The time unit.

        - `'Y'`: date set to the last day of the current year & time set to '23:59:59.999999'.
        - `'Q'`: date set to the last day of the current quarter & time set to '23:59:59.999999'.
        - `'M'`: date set to the last day of the current month & time set to '23:59:59.999999'.
        - `'W'`: date set to Sunday of the current week & time set to '23:59:59.999999'.
        - `'D'`: original date & time set to '23:59:59.999999'.
        - `'h'`: original date & time set to 'XX:59:59.999999'.
        - `'m'`: original date & time set to 'XX:XX:59.999999'.
        - `'s'`: original date & time set to 'XX:XX:XX.999999'.
        - `'ms'`: original date & time set to 'XX:XX:XX.XXX999'.
        - `'Month name'` such as 'Jan', 'februar', '三月': date set to last day of that month & time set to '23:59:59.999999'.
        - `'Weekday name'` such as 'Mon', 'dienstag', '星期三': date set to that weekday & time set to '23:59:59.999999'.
        """
        unit_len: cython.Py_ssize_t = str_len(unit)
        if unit_len == 1:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . second
            if unit_ch == "s":
                return self.to_time(-1, -1, -1, -1, 999999)
            # . minute
            if unit_ch == "m":
                return self.to_time(-1, -1, 59, -1, 999999)
            # . hour
            if unit_ch == "h":
                return self.to_time(-1, 59, 59, -1, 999999)
            # . day
            if unit_ch == "D":
                return self.to_time(23, 59, 59, -1, 999999)
            # . week
            if unit_ch == "W":
                # fmt: off
                return self.add(
                    0, 0, 0, 0,
                    6 - self.weekday(),
                    23 - datetime.datetime_hour(self),
                    59 - datetime.datetime_minute(self),
                    59 - datetime.datetime_second(self), 0,
                    999999 - datetime.datetime_microsecond(self),
                )
                # fmt: on
            # . month
            if unit_ch == "M":
                return self.to_datetime(-1, -1, 31, 23, 59, 59, -1, 999999)
            # . quarter
            if unit_ch == "Q":
                mm: cython.int = utils.quarter_lst_month(datetime.datetime_month(self))
                return self.to_datetime(-1, mm, 31, 23, 59, 59, -1, 999999)
            # . year
            if unit_ch == "Y":
                return self.to_datetime(-1, 12, 31, 23, 59, 59, -1, 999999)
        elif unit_len == 2:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . millisecond
            if unit_ch == "m":
                # fmt: off
                return self.to_time(
                    -1, -1, -1, -1,
                    datetime.datetime_microsecond(self) // 1000 * 1000 + 999
                )
                # fmt: on

        # . month name
        val: cython.int = _parse_month(unit, False)
        if val != -1:
            return self.to_datetime(-1, val, 31, 23, 59, 59, -1, 999999)

        # . weekday name
        val: cython.int = _parse_weekday(unit, False)
        if val != -1:
            # fmt: off
            return self.add(
                0, 0, 0, 0,
                val - self.weekday(),
                23 - datetime.datetime_hour(self),
                59 - datetime.datetime_minute(self),
                59 - datetime.datetime_second(self), 0,
                999999 - datetime.datetime_microsecond(self),
            )
            # fmt: on

        # Invalid
        raise errors.InvalidTimeUnitError(
            "invalid 'end of' time unit '%s'.\nSupported time unit: "
            "['Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms'] "
            "or Month/Weekday name." % unit
        )

    # . frequency
    @cython.ccall
    def freq_round(self, freq: str) -> _Pydt:
        """Perform round operation to the specified freqency `<'Pydt'>`.

        #### Similar to `pandas.Timestamp.round()`.

        :param freq `<'str'>`: Supported frequency ['D', 'h', 'm', 's', 'ms', 'us'].
        """
        # Parse frequency
        f: cython.longlong = _parse_frequency(freq, True)
        if f == 1:
            return self  # exit: no change

        # Adjust frequency
        us: cython.longlong = utils.dt_to_us(self, False)
        us_r: cython.longlong = math.llround(us / f)
        us_r *= f
        if us == us_r:
            return self  # exit: same value

        # Generate new datetime
        return pydt_fr_dt(utils.dt_fr_us(us_r, datetime.datetime_tzinfo(self)))

    @cython.ccall
    def freq_ceil(self, freq: str) -> _Pydt:
        """Perform ceil operation to the specified freqency `<'Pydt'>`.

        #### Similar to `pandas.Timestamp.ceil()`.

        :param freq `<'str'>`: Supported frequency ['D', 'h', 'm', 's', 'ms', 'us'].
        """
        # Parse frequency
        f: cython.longlong = _parse_frequency(freq, True)
        if f == 1:
            return self  # exit: no change

        # Adjust frequency
        us: cython.longlong = utils.dt_to_us(self, False)
        us_c: cython.longlong = int(math.ceill(us / f))
        us_c *= f
        if us == us_c:
            return self  # exit: same value

        # Generate new datetime
        return pydt_fr_dt(utils.dt_fr_us(us_c, datetime.datetime_tzinfo(self)))

    @cython.ccall
    def freq_floor(self, freq: str) -> _Pydt:
        """Perform floor operation to the specified freqency `<'Pydt'>`.

        #### Similar to `pandas.Timestamp.floor()`.

        :param freq `<'str'>`: Supported frequency ['D', 'h', 'm', 's', 'ms', 'us'].
        """
        # Parse frequency
        f: cython.longlong = _parse_frequency(freq, True)
        if f == 1:
            return self  # exit: no change

        # Adjust frequency
        us: cython.longlong = utils.dt_to_us(self, False)
        us_f: cython.longlong = int(math.floorl(us / f))
        us_f *= f
        if us == us_f:
            return self  # exit: same value

        # Generate new datetime
        return pydt_fr_dt(utils.dt_fr_us(us_f, datetime.datetime_tzinfo(self)))

    # Calendar -----------------------------------------------------------------------------
    # . iso
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def isoweekday(self) -> cython.int:
        """Return the instance ISO calendar weekday (1=Mon...7=Sun) `<'int'>`."""
        return utils.ymd_isoweekday(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def isoweek(self) -> cython.int:
        """Return the instance ISO calendar week number (1-53) `<'int'>`."""
        return utils.ymd_isoweek(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
        )

    @cython.ccall
    def isocalendar(self) -> utils.iso:
        """Return the instance ISO calendar as a dictionary `<'dict'>`.

        ### Example:
        >>> dt.isocalendar()
        >>> {'year': 2024, 'week': 40, 'weekday': 2}
        """
        return utils.ymd_isocalendar(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
        )

    # . year
    @property
    def year(self) -> int:
        """Return the instance year (1-9999) `<'int'>`."""
        return self._prop_year()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_year(self) -> cython.int:
        """(func) Return the instance year (1-9999) `<'int'>`."""
        return datetime.datetime_year(self)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_leap_year(self) -> cython.bint:
        """Check if the instance is in a leap year `<'bool'>`."""
        return utils.is_leap_year(datetime.datetime_year(self))

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_long_year(self) -> cython.bint:
        """Check if the instance is in a long year
        (maximum ISO week number is 53) `<'bool'>`.
        """
        return utils.is_long_year(datetime.datetime_year(self))

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def leap_bt_years(self, year: cython.int) -> cython.int:
        """Calculate the number of leap years between
        the passed in 'year' and the instance `<'int'>`.
        """
        return utils.leap_bt_years(datetime.datetime_year(self), year)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_in_year(self) -> cython.int:
        """Return the maximum days in the instance year `<'int'>`."""
        return utils.days_in_year(datetime.datetime_year(self))

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_bf_year(self) -> cython.int:
        """Return the number of days between the 1st day
        of 1AD and the instance `<'int'>`.
        """
        return utils.days_bf_year(datetime.datetime_year(self))

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_of_year(self) -> cython.int:
        """Return the number of days between the 1st day of
        the instance year and the instance date `<'int'>`.
        """
        return utils.days_of_year(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_year(self, year: cython.int) -> cython.bint:
        """Check if the passed in 'year' is the same
        as the instance year `<'bool'>`."""
        return datetime.datetime_year(self) == year

    # . quarter
    @property
    def quarter(self) -> int:
        """Return the instance quarter (1-4) `<'int'>`."""
        return self._prop_quarter()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_quarter(self) -> cython.int:
        """(func) Return instance the quarter (1-4) `<'int'>`."""
        return utils.quarter_of_month(datetime.datetime_month(self))

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_in_quarter(self) -> cython.int:
        """Return the maximum days in the instance quarter `<'int'>`."""
        return utils.days_in_quarter(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_bf_quarter(self) -> cython.int:
        """Return the number of days between the 1st day of the
        instance year and the 1st day of the instance quarter `<'int'>`.
        """
        return utils.days_bf_quarter(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_of_quarter(self) -> cython.int:
        """Return the number of days between the 1st day of the
        instance quarter and the instance date `<'int'>`."""
        return utils.days_of_quarter(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def quarter_first_month(self) -> cython.int:
        """Return the first month number of the
        instance quarter (1, 4, 7, 10) `<'int'>`.
        """
        return utils.quarter_1st_month(datetime.datetime_month(self))

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def quarter_last_month(self) -> cython.int:
        """Return the last month number of the
        instance quarter (3, 6, 9, 12) `<'int'>`.
        """
        return utils.quarter_lst_month(datetime.datetime_month(self))

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_quarter(self, quarter: cython.int) -> cython.bint:
        """Check if the passed in 'quarter' is the same
        as the instance quarter `<'bool'>`."""
        return utils.quarter_of_month(datetime.datetime_month(self)) == quarter

    # . month
    @property
    def month(self) -> int:
        """Return the instance month (1-12) `<'int'>`."""
        return self._prop_month()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_month(self) -> cython.int:
        """(cfunc) Return the instance month (1-12) `<'int'>`."""
        return datetime.datetime_month(self)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_in_month(self) -> cython.int:
        """Return the maximum days in the instance month `<'int'>`."""
        return utils.days_in_month(
            datetime.datetime_year(self), datetime.datetime_month(self)
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_bf_month(self) -> cython.int:
        """Return the number of days between the 1st day of the
        instance year and the 1st day of the instance month `<'int'>`.
        """
        return utils.days_bf_month(
            datetime.datetime_year(self), datetime.datetime_month(self)
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_of_month(self) -> cython.int:
        """Return the number of days between the 1st day of the
        instance month and the instance date `<'int'>`.

        ### Equivalent to:
        >>> dt.day
        """
        return datetime.datetime_day(self)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_month(self, month: int | str) -> cython.bint:
        """Check if the passed in 'month' is the same as
        the instance month `<'bool'>`.

        :param month `<'int/str'>`: The month value.
            1. `<'int'>` (1=Jan...12=Dec).
            2. `<'str'>` month name (case-insensitive) such as 'Jan', 'februar', '三月', etc.
        """
        return _parse_month(month, True) == datetime.datetime_month(self)

    # . weekday
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def weekday(self) -> cython.int:
        """Return the instance weekday (0=Mon...6=Sun) `<'int'>`."""
        return utils.ymd_weekday(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_weekday(self, weekday: int | str) -> cython.bint:
        """Check if the passed in 'weekday' is the same as
        the instance weekday `<'bool'>`.

        :param weekday `<'int/str'>`: The weekday value.
            1. `<'int'>` (0=Mon...6=Sun).
            2. `<'str'>` weekday name (case-insensitive) such as 'Mon', 'dienstag', '星期三', etc.
        """
        return _parse_weekday(weekday, True) == self.weekday()

    # . day
    @property
    def day(self) -> int:
        """Return the instance day (1-31) `<'int'>`."""
        return self._prop_day()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_day(self) -> cython.int:
        """(cfunc) Return the instance day (1-31) `<'int'>`."""
        return datetime.datetime_day(self)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_day(self, day: cython.int) -> cython.bint:
        """Check if the passed in 'day' is the same as
        the instance day `<'bool'>`."""
        return datetime.datetime_day(self) == day

    # . time
    @property
    def hour(self) -> int:
        """Return the instance hour (0-23) `<'int'>`."""
        return self._prop_hour()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_hour(self) -> cython.int:
        """(cfunc) Return the instance hour (0-23) `<'int'>`."""
        return datetime.datetime_hour(self)

    @property
    def minute(self) -> int:
        """Return the instance minute (0-59) `<'int'>`."""
        return self._prop_minute()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_minute(self) -> cython.int:
        """(cfunc) Return the instance minute (0-59) `<'int'>`."""
        return datetime.datetime_minute(self)

    @property
    def second(self) -> int:
        """Return the instance second (0-59) `<'int'>`."""
        return self._prop_second()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_second(self) -> cython.int:
        """(cfunc) Return the instance second (0-59) `<'int'>`."""
        return datetime.datetime_second(self)

    @property
    def millisecond(self) -> int:
        """Return the instance millisecond (0-999) `<'int'>`."""
        return self._prop_millisecond()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_millisecond(self) -> cython.int:
        """(cfunc) Return the instance millisecond (0-999) `<'int'>`."""
        return datetime.datetime_microsecond(self) // 1000

    @property
    def microsecond(self) -> int:
        """Return the instance microsecond (0-999999) `<'int'>`."""
        return self._prop_microsecond()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_microsecond(self) -> cython.int:
        """(cfunc) Return the instance microsecond (0-999999) `<'int'>`."""
        return datetime.datetime_microsecond(self)

    # . date&time
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_first_of(self, unit: str) -> cython.bint:
        """Check if the instance date is on the first day
        of the specified 'unit' `<'bool'>`.

        :param unit `<'str'>`: The time unit.

        - `'Y'`: date is the 1st day of the year.
        - `'Q'`: date is the 1st day of the quarter.
        - `'M'`: date is the 1st day of the month.
        - `'W'`: date is Monday.
        - `'Month name'` such as 'Jan', 'februar', '三月': date is the 1st day of that month.
        """
        unit_len: cython.Py_ssize_t = str_len(unit)
        if unit_len == 1:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . weekday
            if unit_ch == "W":
                return self.weekday() == 0
            # . month
            if unit_ch == "M":
                return utils.dt_is_1st_of_month(self)
            # . quarter
            if unit_ch == "Q":
                return utils.dt_is_1st_of_quarter(self)
            # . year
            if unit_ch == "Y":
                return utils.dt_is_1st_of_year(self)

        # . month name
        val: cython.int = _parse_month(unit, False)
        if val != -1:
            return (
                datetime.datetime_month(self) == val
                and datetime.datetime_day(self) == 1
            )

        # Invalid
        raise errors.InvalidTimeUnitError(
            "invalid 'first of' time unit '%s'.\nSupported time unit: "
            "['Y', 'Q', 'M', 'W'] or Month name." % unit
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_last_of(self, unit: str) -> cython.bint:
        """Check if the instance date is on the last day
        of the specified 'unit' `<'bool'>`.

        :param unit `<'str'>`: The time unit.

        - `'Y'`: date is the last day of the year.
        - `'Q'`: date is the last day of the quarter.
        - `'M'`: date is the last day of the month.
        - `'W'`: date is Sunday.
        - `'Month name'` such as 'Jan', 'februar', '三月': date is the last day of that month.
        """
        unit_len: cython.Py_ssize_t = str_len(unit)
        if unit_len == 1:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . weekday
            if unit_ch == "W":
                return self.weekday() == 6
            # . month
            if unit_ch == "M":
                return utils.dt_is_lst_of_month(self)
            # . quarter
            if unit_ch == "Q":
                return utils.dt_is_lst_of_quarter(self)
            # . year
            if unit_ch == "Y":
                return utils.dt_is_lst_of_year(self)

        # . month name
        val: cython.int = _parse_month(unit, False)
        if val != -1:
            if datetime.datetime_month(self) != val:
                return False
            dd: cython.int = datetime.datetime_day(self)
            return dd == utils.days_in_month(datetime.datetime_year(self), val)

        # Invalid
        raise errors.InvalidTimeUnitError(
            "invalid 'last of' time unit '%s'.\nSupported time unit: "
            "['Y', 'Q', 'M', 'W'] or Month name." % unit
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_start_of(self, unit: str) -> cython.bint:
        """Check if the instance date and time is at the start
        of the specified 'unit' `<'bool'>`.

        :param unit `<'str'>`: The time unit.

        - `'Y'`: date is the 1st day of the year & time is '00:00:00.000000'.
        - `'Q'`: date is the 1st day of the quarter & time is '00:00:00.000000'.
        - `'M'`: date is the 1st day of the month & time is '00:00:00.000000'.
        - `'W'`: date is on Monday & time is '00:00:00.000000'.
        - `'D'`: time is '00:00:00.000000'.
        - `'h'`: time is 'XX:00:00.000000'.
        - `'m'`: time is 'XX:XX:00.000000'.
        - `'s'`: time is 'XX:XX:XX.000000'.
        - `'ms'`: time is 'XX:XX:XX.XXX000'.
        - `'Month name'` such as 'Jan', 'februar', '三月': date is the 1st day of that month & time is '00:00:00.000000'.
        - `'Weekday name'` such as 'Mon', 'dienstag', '星期三': date is on that weekday & time is '00:00:00.000000'.
        """
        unit_len: cython.Py_ssize_t = str_len(unit)
        start_of_time: cython.int = -1
        if unit_len == 1:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . second
            if unit_ch == "s":
                return datetime.datetime_microsecond(self) == 0
            # . minute
            if unit_ch == "m":
                return (
                    datetime.datetime_second(self) == 0
                    and datetime.datetime_microsecond(self) == 0
                )
            # . hour
            if unit_ch == "h":
                return (
                    datetime.datetime_minute(self) == 0
                    and datetime.datetime_second(self) == 0
                    and datetime.datetime_microsecond(self) == 0
                )
            # - is time start
            start_of_time = utils.dt_is_start_of_time(self)
            if start_of_time == 1:
                # . day
                if unit_ch == "D":
                    return True
                # . week
                if unit_ch == "W":
                    return self.weekday() == 0
                # . month
                if unit_ch == "M":
                    return utils.dt_is_1st_of_month(self)
                # . quarter
                if unit_ch == "Q":
                    return utils.dt_is_1st_of_quarter(self)
                # . year
                if unit_ch == "Y":
                    return utils.dt_is_1st_of_year(self)
            else:
                return False

        elif unit_len == 2:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . millisecond
            if unit_ch == "m":
                return datetime.datetime_microsecond(self) % 1000 == 0

        # - is time start
        if start_of_time == -1:
            start_of_time = utils.dt_is_start_of_time(self)
        if start_of_time == 1:
            # . month name
            val: cython.int = _parse_month(unit, False)
            if val != -1:
                return (
                    datetime.datetime_month(self) == val
                    and datetime.datetime_day(self) == 1
                )
            # . weekday name
            val: cython.int = _parse_weekday(unit, False)
            if val != -1:
                return self.weekday() == val
        else:
            return False

        # Invalid
        raise errors.InvalidTimeUnitError(
            "invalid 'start of' time unit '%s'.\nSupported time unit: "
            "['Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms'] "
            "or Month/Weekday name." % unit
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_end_of(self, unit: str) -> cython.bint:
        """Check if the instance date and time is at the end
        of the specified 'unit' `<'bool'>`.

        :param unit `<'str'>`: The time unit.

        - `'Y'`: date is the last day of the year & time is '23:59:59.999999'.
        - `'Q'`: date is the last day of the quarter & time is '23:59:59.999999'.
        - `'M'`: date is the last day of the month & time is '23:59:59.999999'.
        - `'W'`: date is on Sunday & time is '23:59:59.999999'.
        - `'D'`: time is '23:59:59.999999'.
        - `'h'`: time is 'XX:59:59.999999'.
        - `'m'`: time is 'XX:XX:59.999999'.
        - `'s'`: time is 'XX:XX:XX.999999'.
        - `'ms'`: time is 'XX:XX:XX.XXX999'.
        - Month name such as 'Jan', 'februar', '三月': date is the last day of that month & time is '23:59:59.999999'.
        - Weekday name such as 'Mon', 'dienstag', '星期三': date is on that weekday & time is '23:59:59.999999'.
        """
        unit_len: cython.Py_ssize_t = str_len(unit)
        end_of_time: cython.int = -1
        if unit_len == 1:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . second
            if unit_ch == "s":
                return datetime.datetime_microsecond(self) == 999_999
            # . minute
            if unit_ch == "m":
                return (
                    datetime.datetime_second(self) == 59
                    and datetime.datetime_microsecond(self) == 999_999
                )
            # . hour
            if unit_ch == "h":
                return (
                    datetime.datetime_minute(self) == 59
                    and datetime.datetime_second(self) == 59
                    and datetime.datetime_microsecond(self) == 999_999
                )
            # - is time end
            end_of_time = utils.dt_is_end_of_time(self)
            if end_of_time == 1:
                # . day
                if unit_ch == "D":
                    return True
                # . week
                if unit_ch == "W":
                    return self.weekday() == 6
                # . month
                if unit_ch == "M":
                    return utils.dt_is_lst_of_month(self)
                # . quarter
                if unit_ch == "Q":
                    return utils.dt_is_lst_of_quarter(self)
                # . year
                if unit_ch == "Y":
                    return utils.dt_is_lst_of_year(self)
            else:
                return False

        elif unit_len == 2:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . millisecond
            if unit_ch == "m":
                return datetime.datetime_microsecond(self) % 1000 == 999

        # - is time end
        if end_of_time == -1:
            end_of_time = utils.dt_is_end_of_time(self)
        if end_of_time == 1:
            # . month name
            val: cython.int = _parse_month(unit, False)
            if val != -1:
                return (
                    datetime.datetime_month(self) == val
                    and datetime.datetime_day(self) == self.days_in_month()
                )
            # . weekday name
            val: cython.int = _parse_weekday(unit, False)
            if val != -1:
                return self.weekday() == val
        else:
            return False

        # Invalid
        raise errors.InvalidTimeUnitError(
            "invalid 'end of' time unit '%s'.\nSupported time unit: "
            "['Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms'] "
            "or Month/Weekday name." % unit
        )

    # Timezone -----------------------------------------------------------------------------
    @property
    def tz_available(self) -> set[str]:
        """Return the available timezone names `<'set[str]'>`.

        Equivalent to: 'zoneinfo.available_timezones()'.
        """
        return _available_timezones()

    @property
    def tzinfo(self) -> object:
        """Return the instance timezone info `<'tzinfo/None'>`."""
        return self._prop_tzinfo()

    @cython.cfunc
    @cython.inline(True)
    def _prop_tzinfo(self) -> object:
        """(cfunc) Return the instance timezone info `<'tzinfo/None'>`."""
        return datetime.datetime_tzinfo(self)

    @property
    def fold(self) -> int:
        """Return the instance fold (0/1) `<'int'>`.

        Use to disambiguates local times during daylight
        saving time (DST) transitions.
        """
        return self._prop_fold()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_fold(self) -> cython.int:
        """(cfunc) Return the instance fold (0/1) `<'int'>`.

        Use to disambiguates local times during daylight
        saving time (DST) transitions.
        """
        return datetime.datetime_fold(self)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_local(self) -> cython.bint:
        """Check if the instance is at local timezone `<'bool'>`.

        #### Timezone-naive instance always returns `False`.
        """
        return utils.dt_utcoffset_seconds(self) == utils.tz_local_seconds(self)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_utc(self) -> cython.bint:
        """Check if the instance is at UTC timezone `<'bool'>`.

        #### Timezone-naive instance always returns `False`.
        """
        return utils.dt_utcoffset_seconds(self) == 0

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_dst(self) -> cython.bint:
        """Check if the instance is in Dayligh Saving Time `<'bool'>.

        #### Timezone-naive datetime always returns `False`.
        """
        dst = utils.dt_dst(self)
        return False if dst is None else bool(dst)

    @cython.ccall
    def tzname(self) -> str:
        """Return the instance timezone name `<'str/None'>`.

        #### Timezone-naive instance returns `None`.

        Note that the name is 100% informational -- there's no requirement that
        it mean anything in particular. For example, "GMT", "UTC", "-500",
        "-5:00", "EDT", "US/Eastern", "America/New York" are all valid replies.
        """
        return utils.dt_tzname(self)

    @cython.ccall
    def utcoffset(self) -> datetime.timedelta:
        """Return the instance timezone offset
        (positive east UTC / negative west UTC) `<'datetime.timedelta/None'>`.

        #### Timezone-naive instance returns `None`.
        """
        return utils.dt_utcoffset(self)

    @cython.ccall
    def utcoffset_seconds(self) -> object:
        """Return the instance timezone offset
        (positive east UTC / negative west UTC) in total seconds `<'int/None'>`.

        #### Timezone-naive instance returns `None`.
        """
        ss: cython.int = utils.dt_utcoffset_seconds(self)
        return None if ss == -100_000 else ss

    @cython.ccall
    def dst(self) -> datetime.timedelta:
        """Return the DST offset (positive east / negative west)
        `<'datetime.timedelta/None'>`.

        #### Timezone-naive instance returns `None`.

        This is purely informational. The DST offset has already been
        added to the UTC offset returned by 'utcoffset()' if applicable.
        """
        return utils.dt_dst(self)

    @cython.ccall
    def astimezone(self, tz: datetime.tzinfo | str | None = None) -> _Pydt:
        """Convert the instance to the specified timezone `<'Pydt'>`.

        This method adjust the instance to represent the same point in time
        of the new timezone, and returns a new timezone-aware instance with
        its 'tzinfo' set to the passed in 'tz'.

        - If the instance is timezone-aware, it will be converted to the
          new timezone directly.
        - If the instance is timezone-naive, it will be localized to the
          local timezone first, and then converted to the new timezone.

        :param tz `<'tzinfo/str/None'>`: The timezone to convert to, defaults to `None`.
            1. `<'datetime.tzinfo'>` subclass of datetime.tzinfo.
            2. `<'str'>` timezone name supported by 'Zoneinfo' module,
               or `'local'` for local timezone.
            3. `<'NoneType'>` local timezone.
        """
        return self._from_dt(utils.dt_astimezone(self, utils.tz_parse(tz)))

    @cython.ccall
    def tz_localize(self, tz: datetime.tzinfo | str | None = None) -> _Pydt:
        """Localize (replace) the instance to a specific timezone `<'Pydt'>`.

        :param tz `<'tzinfo/str/None'>`: The timezone to localize to, defaults to `None`.
            1. `<'datetime.tzinfo'>` subclass of datetime.tzinfo.
            2. `<'str'>` timezone name supported by 'Zoneinfo' module,
               or `'local'` for local timezone.
            3. `<'NoneType'>` timezone-naive.

        ### Equivalent to:
        >>> dt.replace(tzinfo=tz)
        """
        return self._from_dt(utils.dt_replace_tz(self, utils.tz_parse(tz)))

    @cython.ccall
    def tz_convert(self, tz: datetime.tzinfo | str | None = None) -> _Pydt:
        """Convert the instance to the specified timezone `<'Pydt'>`.

        #### Alias of 'astimezone()'.

        For more information, please refer to the 'astimezone()' method.
        """
        return self._from_dt(utils.dt_astimezone(self, utils.tz_parse(tz)))

    @cython.ccall
    def tz_switch(
        self,
        targ_tz: datetime.tzinfo | str | None,
        base_tz: datetime.tzinfo | str | None = None,
        naive: cython.bint = False,
    ) -> _Pydt:
        """Switch (convert) the instance from base timezone
        to the target timezone `<'Pydt'>`.

        This method extends the functionality of the 'astimezone()' method
        by allowing user to specify a base timezone for timezone-naive
        instances before converting to the target timezone.

        - If the instance is timezone-aware, argument 'base_tz' is `IGNORED`,
          and this method behaves identically to the 'astimezone()', converting
          the instance to the traget timezone.
        - If the instance is timezone-naive, it first localize the instance
          to the 'base_tz', and then converts to the target timezone.

        :param targ_tz `<'tzinfo/str/None'>`: The target timezone to convert to.
        :param base_tz `<'tzinfo/str/None'>`: The base timezone to localize to for timezone-naive instance, defaults to `None`.
        :param naive `<'bool'>`: If 'True', returns a timezone-naive instance after conversion, defaults to `False`.

        Both 'targ_tz' and 'base_tz' accepts the following as input:
            1. `<'datetime.tzinfo'>` subclass of datetime.tzinfo.
            2. `<'str'>` timezone name supported by 'Zoneinfo' module,
               or `'local'` for local timezone.
            3. `<'NoneType'>` local timezone.
        """
        # Timezone-aware
        targ_tz = utils.tz_parse(targ_tz)
        mytz = datetime.datetime_tzinfo(self)
        if mytz is not None:
            # . mytz is target timezone
            if mytz is targ_tz:
                if naive:
                    dt = utils.dt_replace_tz(self, None)
                else:
                    return self  # exit
            # . mytz => target timezone
            else:
                dt = utils.dt_astimezone(self, targ_tz)
                if naive:
                    dt = utils.dt_replace_tz(dt, None)
            # Generate new datetime
            return self._from_dt(dt)

        # Timezone-naive
        else:
            # . parse base timezone
            if base_tz is None:
                base_tz = utils.tz_local(self)
            else:
                base_tz = utils.tz_parse(base_tz)
            # . base is target timezone
            if base_tz is targ_tz:
                if naive:
                    return self  # exit
                else:
                    dt = utils.dt_replace_tz(self, targ_tz)
            # . localize to base, then convert to target timzone
            else:
                dt = utils.dt_replace_tz(self, base_tz)
                dt = utils.dt_astimezone(dt, targ_tz)
                if naive:
                    dt = utils.dt_replace_tz(dt, None)
            # Generate new datetime
            return self._from_dt(dt)

    # Arithmetic ---------------------------------------------------------------------------
    @cython.ccall
    def add(
        self,
        years: cython.int = 0,
        quarters: cython.int = 0,
        months: cython.int = 0,
        weeks: cython.int = 0,
        days: cython.int = 0,
        hours: cython.int = 0,
        minutes: cython.int = 0,
        seconds: cython.int = 0,
        milliseconds: cython.int = 0,
        microseconds: cython.int = 0,
    ) -> _Pydt:
        """Add relative delta to the instance `<'Pydt'>`.

        :param years `<'int'>`: Relative delta of years, defaults to `0`.
        :param quarters `<'int'>`: Relative delta of quarters, defaults to `0`.
        :param months `<'int'>`: Relative delta of months, defaults to `0`.
        :param weeks `<'int'>`: Relative delta of weeks, defaults to `0`.
        :param days `<'int'>`: Relative delta of days, defaults to `0`.
        :param hours `<'int'>`: Relative delta of hours, defaults to `0`.
        :param minutes `<'int'>`: Relative delta of minutes, defaults to `0`.
        :param seconds `<'int'>`: Relative delta of seconds, defaults to `0`.
        :param milliseconds `<'int'>`: Relative delta of milliseconds, defaults to `0`.
        :param microseconds `<'int'>`: Relative delta of microseconds, defaults to `0`.
        """
        # Calculate delta
        # . year
        my_yy: cython.int = datetime.datetime_year(self)
        yy: cython.int = my_yy + years
        ymd_eq: cython.bint = yy == my_yy
        # . month
        my_mm: cython.int = datetime.datetime_month(self)
        mm: cython.int = my_mm + months + quarters * 3
        if mm != my_mm:
            if mm > 12:
                yy += mm // 12
                mm %= 12
            elif mm < 1:
                mm = 12 - mm
                yy -= mm // 12
                mm = 12 - mm % 12
            if ymd_eq:
                ymd_eq = mm == my_mm
        # . day
        dd: cython.int = datetime.datetime_day(self)
        # . microseconds
        my_us: cython.int = datetime.datetime_microsecond(self)
        us: cython.longlong = my_us + microseconds + milliseconds * 1000
        if us != my_us:
            if us > 999_999:
                seconds += us // 1_000_000
                us %= 1_000_000
            elif us < 0:
                us = 999_999 - us
                seconds -= us // 1_000_000
                us = 999_999 - us % 1_000_000
            hms_eq: cython.bint = us == my_us
        else:
            hms_eq: cython.bint = True
        # . seconds
        my_ss: cython.int = datetime.datetime_second(self)
        ss: cython.longlong = my_ss + seconds
        if ss != my_ss:
            if ss > 59:
                minutes += ss // 60
                ss %= 60
            elif ss < 0:
                ss = 59 - ss
                minutes -= ss // 60
                ss = 59 - ss % 60
            if hms_eq:
                hms_eq = ss == my_ss
        # . minutes
        my_mi: cython.int = datetime.datetime_minute(self)
        mi: cython.longlong = my_mi + minutes
        if mi != my_mi:
            if mi > 59:
                hours += mi // 60
                mi %= 60
            elif mi < 0:
                mi = 59 - mi
                hours -= mi // 60
                mi = 59 - mi % 60
            if hms_eq:
                hms_eq = mi == my_mi
        # . hours
        my_hh: cython.int = datetime.datetime_hour(self)
        hh: cython.longlong = my_hh + hours
        if hh != my_hh:
            if hh > 23:
                days += hh // 24
                hh %= 24
            elif hh < 0:
                hh = 23 - hh
                days -= hh // 24
                hh = 23 - hh % 24
            if hms_eq:
                hms_eq = hh == my_hh
        # . days
        days += weeks * 7

        # Add delta
        if days != 0:
            _ymd = utils.ymd_fr_ordinal(utils.ymd_to_ordinal(yy, mm, dd) + days)
            yy, mm, dd = _ymd.year, _ymd.month, _ymd.day
        elif ymd_eq:
            if hms_eq:
                return self  # exit: no change
        elif dd > 28:
            dd = min(dd, utils.days_in_month(yy, mm))

        # Create Pydt
        # fmt: off
        return pydt_new(
            min(max(yy, 1), 9999), mm, dd, hh, mi, ss, us, 
            datetime.datetime_tzinfo(self), 
            datetime.datetime_fold(self),
        )
        # fmt: on

    @cython.ccall
    def sub(
        self,
        years: cython.int = 0,
        quarters: cython.int = 0,
        months: cython.int = 0,
        weeks: cython.int = 0,
        days: cython.int = 0,
        hours: cython.int = 0,
        minutes: cython.int = 0,
        seconds: cython.int = 0,
        milliseconds: cython.int = 0,
        microseconds: cython.int = 0,
    ) -> _Pydt:
        """Substract relative delta from the instance `<'Pydt'>`.

        :param years `<'int'>`: Relative delta of years, defaults to `0`.
        :param quarters `<'int'>`: Relative delta of quarters, defaults to `0`.
        :param months `<'int'>`: Relative delta of months, defaults to `0`.
        :param weeks `<'int'>`: Relative delta of weeks, defaults to `0`.
        :param days `<'int'>`: Relative delta of days, defaults to `0`.
        :param hours `<'int'>`: Relative delta of hours, defaults to `0`.
        :param minutes `<'int'>`: Relative delta of minutes, defaults to `0`.
        :param seconds `<'int'>`: Relative delta of seconds, defaults to `0`.
        :param milliseconds `<'int'>`: Relative delta of milliseconds, defaults to `0`.
        :param microseconds `<'int'>`: Relative delta of microseconds, defaults to `0`.
        """
        # fmt: off
        return self.add(
            -years, -quarters, -months, -weeks, -days,
            -hours, -minutes, -seconds, -milliseconds,
            -microseconds,
        )
        # fmt: on

    @cython.ccall
    def avg(self, dtobj: object = None) -> _Pydt:
        """Construct the average datetime between the instance
        and another datetime-related object `<'Pydt'>`.

        :param dtobj `<'object'>`: Datetime related object, defaults to `None`.
            1. `<'str'>` datetime string that contains datetime information.
            2. `<'datetime.datetime'>` instance or subclass of datetime.datetime.
            3. `<'datetime.date'>` instance or subclass of datetime.date. All time values set to 0.
            4. `<'int/float'>` numeric values, treated as total seconds since Unix Epoch.
            5. `<'np.datetime64'>` resolution above 'us' will be discarded.
            6. `<'NoneType'>` current datetime (now).
        """
        # Parse & adjust timezone
        my_tz: datetime.tzinfo = datetime.datetime_tzinfo(self)
        utc: cython.bint = my_tz is not None
        if dtobj is None:
            dt: datetime.datetime = utils.dt_now(my_tz)
        else:
            dt: datetime.datetime = pydt_fr_dtobj(dtobj)
            dt_tz = dt.tzinfo
            if my_tz is not dt_tz and (my_tz is None or dt_tz is None):
                _raise_incomparable_error(self, dt, "calculate average")

        # Calculate average
        my_us: cython.longlong = utils.dt_to_us(self, utc)
        dt_us: cython.longlong = utils.dt_to_us(dt, utc)
        return pydt_fr_dt(utils.dt_fr_us(math.llround((my_us + dt_us) / 2), my_tz))

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def diff(
        self,
        dtobj: object,
        unit: str,
        bounds: str = "both",
    ) -> cython.longlong:
        """Calcuate the `ABSOLUTE` difference between the instance
        and another datetime-related object `<'int'>`.

        The difference is computed in the specified time 'unit' and adjusted
        based on the 'bounds' parameter to determine the inclusivity of the
        start and end times.

        :param dtobj `<'object'>`: Datetime related object.
            1. `<'str'>` datetime string that contains datetime information.
            2. `<'datetime.datetime'>` instance or subclass of datetime.datetime.
            3. `<'datetime.date'>` instance or subclass of datetime.date. All time values set to 0.
            4. `<'int/float'>` numeric values, treated as total seconds since Unix Epoch.
            5. `<'np.datetime64'>` resolution above 'us' will be discarded.

        :param unit `<'str'>`: The time unit for calculating the difference.
            Supports: ['Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us'].

        :param bounds `<'str'>`: Specifies the inclusivity of the start and end times, defaults to `'both'`.
            1. `'both'`: Include both the start and end times.
            2. `'one'`: Include either the start or end time.
            3. `'none'`: Exclude both the start and end times.
        """
        # Parse & adjust timezone
        dt: datetime.datetime = pydt_fr_dtobj(dtobj)
        my_tz, dt_tz = datetime.datetime_tzinfo(self), dt.tzinfo
        if my_tz is not dt_tz and (my_tz is None or dt_tz is None):
            _raise_incomparable_error(self, dt, "calculate difference")

        # Handle bounds
        bounds = bounds.lower()
        if bounds == "both":
            offset: cython.int = 1
        elif bounds == "one":
            offset: cython.int = 0
        elif bounds == "none":
            offset: cython.int = -1
        else:
            raise errors.InvalidArgumentError(
                "invalid bounds '%s', accpets: ['both', 'one', 'none']." % bounds
            )

        # Calculate difference
        unit_len: cython.Py_ssize_t = str_len(unit)
        if unit_len == 1:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # Unit: year
            if unit_ch == "Y":
                my_yy: cython.int = datetime.datetime_year(self)
                return abs(my_yy - dt.year) + offset  # exit
            # Unit: quarter
            if unit_ch == "Q":
                my_yy: cython.int = datetime.datetime_year(self)
                my_qq: cython.int = utils.quarter_of_month(
                    datetime.datetime_month(self)
                )
                dt_qq: cython.int = utils.quarter_of_month(dt.month)
                return abs((my_yy - dt.year) * 4 + (my_qq - dt_qq)) + offset  # exit
            # Unit: month
            if unit_ch == "M":
                my_yy: cython.int = datetime.datetime_year(self)
                my_mm: cython.int = datetime.datetime_month(self)
                return abs((my_yy - dt.year) * 12 + (my_mm - dt.month)) + offset  # exit
            # Unit: week
            utc: cython.bint = my_tz is None
            my_us: cython.longlong = utils.dt_to_us(self, utc)
            dt_us: cython.longlong = utils.dt_to_us(dt, utc)
            if unit_ch == "W":
                my_us = my_us // utils.US_DAY
                dt_us = dt_us // utils.US_DAY
                if my_us > dt_us:
                    wkd_off: cython.int = utils.ymd_weekday(dt.year, dt.month, dt.day)
                else:
                    wkd_off: cython.int = utils.ymd_weekday(
                        datetime.datetime_year(self),
                        datetime.datetime_month(self),
                        datetime.datetime_day(self),
                    )
                return (abs(my_us - dt_us) + wkd_off) // 7 + offset  # exit
            # Unit: day
            if unit_ch == "D":
                my_us = my_us // utils.US_DAY
                dt_us = dt_us // utils.US_DAY
                return abs(my_us - dt_us) + offset
            # Unit: hour
            if unit_ch == "h":
                my_us = my_us // utils.US_HOUR
                dt_us = dt_us // utils.US_HOUR
                return abs(my_us - dt_us) + offset  # exit
            # Unit: minute
            if unit_ch == "m":
                my_us //= 60_000_000
                dt_us //= 60_000_000
                return abs(my_us - dt_us) + offset  # exit
            # Unit: second
            if unit_ch == "s":
                my_us //= 1_000_000
                dt_us //= 1_000_000
                return abs(my_us - dt_us) + offset  # exit

        elif unit_len == 2:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            utc: cython.bint = my_tz is None
            my_us: cython.longlong = utils.dt_to_us(self, utc)
            dt_us: cython.longlong = utils.dt_to_us(dt, utc)
            # Unit: millisecond
            if unit_ch == "m":
                my_us //= 1_000
                dt_us //= 1_000
                return abs(my_us - dt_us) + offset  # exit
            # Unit: microsecond
            if unit_ch == "u":
                return abs(my_us - dt_us) + offset  # exit

        # Invalid time unit
        raise errors.InvalidTimeUnitError(
            "invalid time unit '%s'.\nSupported time unit: "
            "['Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us']." % unit
        )

    @cython.cfunc
    @cython.inline(True)
    def _add_timedelta(
        self,
        days: cython.int,
        seconds: cython.int,
        microseconds: cython.int,
    ) -> _Pydt:
        """(cfunc) Add timedelta to the instance `<'Pydt'>`."""
        dd_: cython.longlong = days
        ss_: cython.longlong = seconds
        us: cython.longlong = (dd_ * 86_400 + ss_) * 1_000_000 + microseconds
        if us == 0:
            return self  # no change

        # Add delta
        us += utils.dt_to_us(self, False)
        _us: cython.ulonglong = min(
            max(us + utils.EPOCH_US, utils.DT_US_MIN), utils.DT_US_MAX
        )
        _ymd = utils.ymd_fr_ordinal(_us // utils.US_DAY)
        _hms = utils.hms_fr_us(_us)

        # Create Pydt
        # fmt: off
        return pydt_new(
            _ymd.year, _ymd.month, _ymd.day, 
            _hms.hour, _hms.minute, _hms.second, _hms.microsecond,
            datetime.datetime_tzinfo(self),
            datetime.datetime_fold(self),
        )
        # fmt: on

    def __add__(self, o: object) -> _Pydt:
        # timedelta
        if utils.is_td(o):
            return self._add_timedelta(
                datetime.timedelta_days(o),
                datetime.timedelta_seconds(o),
                datetime.timedelta_microseconds(o),
            )
        if utils.is_td64(o):
            o = utils.td64_to_td(o)
            return self._add_timedelta(
                datetime.timedelta_days(o),
                datetime.timedelta_seconds(o),
                datetime.timedelta_microseconds(o),
            )
        return NotImplemented

    def __radd__(self, o: object) -> _Pydt:
        # timedelta
        if utils.is_td(o):
            return self._add_timedelta(
                datetime.timedelta_days(o),
                datetime.timedelta_seconds(o),
                datetime.timedelta_microseconds(o),
            )
        if utils.is_td64(o):
            o = utils.td64_to_td(o)
            return self._add_timedelta(
                datetime.timedelta_days(o),
                datetime.timedelta_seconds(o),
                datetime.timedelta_microseconds(o),
            )
        return NotImplemented

    def __sub__(self, o: object) -> _Pydt | datetime.timedelta:
        # timedelta
        if utils.is_td(o):
            return self._add_timedelta(
                -datetime.timedelta_days(o),
                -datetime.timedelta_seconds(o),
                -datetime.timedelta_microseconds(o),
            )
        if utils.is_td64(o):
            o = utils.td64_to_td(o)
            return self._add_timedelta(
                -datetime.timedelta_days(o),
                -datetime.timedelta_seconds(o),
                -datetime.timedelta_microseconds(o),
            )
        # datetime
        if utils.is_dt(o):
            pass
        elif utils.is_date(o):
            o = utils.dt_fr_date(o)
        elif isinstance(o, str):
            o = pydt_fr_dtobj(o)
        elif utils.is_dt64(o):
            o = utils.dt64_to_dt(o)
        else:
            return NotImplemented

        # Check timezone parity
        m_tz = datetime.datetime_tzinfo(self)
        o_tz = datetime.datetime_tzinfo(o)
        if m_tz is not o_tz and (m_tz is None or o_tz is None):
            _raise_incomparable_error(self, o, "perform subtraction")

        # Calculate delta
        utc: cython.bint = m_tz is not None
        m_us: cython.longlong = utils.dt_to_us(self, utc)
        o_us: cython.longlong = utils.dt_to_us(o, utc)

        # Create delta
        return utils.td_fr_us(m_us - o_us)

    # Comparison ---------------------------------------------------------------------------
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_past(self) -> cython.bint:
        """Check if the instance is in the past (less than now) `<'bool'>`."""
        return self < utils.dt_now(datetime.datetime_tzinfo(self))

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_future(self) -> cython.bint:
        """Check if the instance is in the future (greater than now) `<'bool'>`."""
        return self > utils.dt_now(datetime.datetime_tzinfo(self))

    def closest(self, *dtobjs: object) -> _Pydt:
        """Parse & find the given datetime-related objects that
        is closest in time to the instance `<'Pydt/None'>`.

        :param dtobjs `<'*objects'>`: One or more datetime-related objects to compare with the instance.
            1. `<'str'>` datetime string that contains datetime information.
            2. `<'datetime.datetime'>` instance or subclass of datetime.datetime.
            3. `<'datetime.date'>` instance or subclass of datetime.date. All time values set to 0.
            4. `<'int/float'>` numeric values, treated as total seconds since Unix Epoch.
            5. `<'np.datetime64'>` resolution above 'us' will be discarded.

        Notes:
        - If multiple datetime objects are equally close to the instance,
          return the first encounter.
        - Returns `None` if no datetime object is provided.
        """
        return self._closest(dtobjs)

    @cython.cfunc
    @cython.inline(True)
    def _closest(self, dtobjs: tuple[object]) -> _Pydt:
        """(cfunc) Parse & find the given datetime-related objects that
        is closest in time to the instance `<'Pydt/None'>`.

        :param dtobjs `<'tuple[object]'>`: One or more datetime-related objects to compare with the instance.
            1. `<'str'>` datetime string that contains datetime information.
            2. `<'datetime.datetime'>` instance or subclass of datetime.datetime.
            3. `<'datetime.date'>` instance or subclass of datetime.date. All time values set to 0.
            4. `<'int/float'>` numeric values, treated as total seconds since Unix Epoch.
            5. `<'np.datetime64'>` resolution above 'us' will be discarded.

        Notes:
        - If multiple datetime objects are equally close to the instance,
          return the first encounter.
        - Returns `None` if no datetime object is provided.
        """
        res: _Pydt = None
        delta: cython.longlong = LLONG_MAX
        my_tz = datetime.datetime_tzinfo(self)
        utc: cython.bint = my_tz is not None
        my_us: cython.longlong = utils.dt_to_us(self, utc)
        for dtobj in dtobjs:
            # Parse & adjust timezone
            dt = pydt_fr_dtobj(dtobj)
            dt_tz = datetime.datetime_tzinfo(dt)
            if my_tz is not dt_tz and (my_tz is None or dt_tz is None):
                _raise_incomparable_error(self, dt, "compare distance")

            # Compare delta
            delta_us = abs(my_us - utils.dt_to_us(dt, utc))
            if delta_us < delta:
                res, delta = dt, delta_us

        # Return result
        return res

    def farthest(self, *dtobjs: object) -> _Pydt:
        """Parse & find the given datetime-related objects that
        is farthest in time to the instance `<'Pydt/None'>`.

        :param dtobjs `<'*objects'>`: One or more datetime-related objects to compare with the instance.
            1. `<'str'>` datetime string that contains datetime information.
            2. `<'datetime.datetime'>` instance or subclass of datetime.datetime.
            3. `<'datetime.date'>` instance or subclass of datetime.date. All time values set to 0.
            4. `<'int/float'>` numeric values, treated as total seconds since Unix Epoch.
            5. `<'np.datetime64'>` resolution above 'us' will be discarded.

        Notes:
        - If multiple datetime objects are equally distance from the instance,
          return the first encounter.
        - Returns `None` if no datetime object is provided.
        """
        return self._farthest(dtobjs)

    @cython.cfunc
    @cython.inline(True)
    def _farthest(self, dtobjs: tuple[object]) -> _Pydt:
        """(cfunc) Parse & find the given datetime-related objects that
        is farthest in time to the instance `<'Pydt/None'>`.

        :param dtobjs `<'tuple[object]'>`: One or more datetime-related objects to compare with the instance.
            1. `<'str'>` datetime string that contains datetime information.
            2. `<'datetime.datetime'>` instance or subclass of datetime.datetime.
            3. `<'datetime.date'>` instance or subclass of datetime.date. All time values set to 0.
            4. `<'int/float'>` numeric values, treated as total seconds since Unix Epoch.
            5. `<'np.datetime64'>` resolution above 'us' will be discarded.

        Notes:
        - If multiple datetime objects are equally distance from the instance,
          return the first encounter.
        - Returns `None` if no datetime object is provided.
        """
        res: _Pydt = None
        delta: cython.longlong = -1
        my_tz = datetime.datetime_tzinfo(self)
        utc: cython.bint = my_tz is not None
        my_us: cython.longlong = utils.dt_to_us(self, utc)
        for dtobj in dtobjs:
            # Parse & adjust timezone
            dt = pydt_fr_dtobj(dtobj)
            dt_tz = datetime.datetime_tzinfo(dt)
            if my_tz is not dt_tz and (my_tz is None or dt_tz is None):
                _raise_incomparable_error(self, dt, "compare distance")

            # Compare delta
            delta_us = abs(my_us - utils.dt_to_us(dt, utc))
            if delta_us > delta:
                res, delta = dt, delta_us

        # Return result
        return res

    def __eq__(self, o: object) -> bool:
        if utils.is_dt(o):
            return _compare_dts(self, o, True) == 0
        if utils.is_date(o):
            return _compare_dts(self, utils.dt_fr_date(o), True) == 0
        if isinstance(o, str):
            return _compare_dts(self, pydt_fr_dtobj(o), True) == 0
        return NotImplemented

    def __le__(self, o: object) -> bool:
        if utils.is_dt(o):
            return _compare_dts(self, o) <= 0
        if utils.is_date(o):
            return _compare_dts(self, utils.dt_fr_date(o)) <= 0
        if isinstance(o, str):
            return _compare_dts(self, pydt_fr_dtobj(o)) <= 0
        return NotImplemented

    def __lt__(self, o: object) -> bool:
        if utils.is_dt(o):
            return _compare_dts(self, o) < 0
        if utils.is_date(o):
            return _compare_dts(self, utils.dt_fr_date(o)) < 0
        if isinstance(o, str):
            return _compare_dts(self, pydt_fr_dtobj(o)) < 0
        return NotImplemented

    def __ge__(self, o: object) -> bool:
        if utils.is_dt(o):
            return _compare_dts(self, o) >= 0
        if utils.is_date(o):
            return _compare_dts(self, utils.dt_fr_date(o)) >= 0
        if isinstance(o, str):
            return _compare_dts(self, pydt_fr_dtobj(o)) >= 0
        return NotImplemented

    def __gt__(self, o: object) -> bool:
        if utils.is_dt(o):
            return _compare_dts(self, o) > 0
        if utils.is_date(o):
            return _compare_dts(self, utils.dt_fr_date(o)) > 0
        if isinstance(o, str):
            return _compare_dts(self, pydt_fr_dtobj(o)) > 0
        return NotImplemented

    # Representation -----------------------------------------------------------------------
    def __repr__(self) -> str:
        yy: cython.int = datetime.datetime_year(self)
        mm: cython.int = datetime.datetime_month(self)
        dd: cython.int = datetime.datetime_day(self)
        hh: cython.int = datetime.datetime_hour(self)
        mi: cython.int = datetime.datetime_minute(self)
        ss: cython.int = datetime.datetime_second(self)
        us: cython.int = datetime.datetime_microsecond(self)
        tz = datetime.datetime_tzinfo(self)
        fd: cython.int = datetime.datetime_fold(self)

        r: str
        if us == 0:
            r = "%d, %d, %d, %d, %d, %d" % (yy, mm, dd, hh, mi, ss)
        else:
            r = "%d, %d, %d, %d, %d, %d, %d" % (yy, mm, dd, hh, mi, ss, us)

        if tz is not None:
            r += ", tzinfo=%r" % tz
        if fd == 1:
            r += ", fold=1"
        return "%s(%s)" % (self.__class__.__name__, r)

    def __str__(self) -> str:
        return self.isoformat(" ")

    def __format__(self, fmt: str) -> str:
        return str(self) if str_len(fmt) == 0 else self.strftime(fmt)

    def __copy__(self) -> _Pydt:
        return pydt_new(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
            datetime.datetime_hour(self),
            datetime.datetime_minute(self),
            datetime.datetime_second(self),
            datetime.datetime_microsecond(self),
            datetime.datetime_tzinfo(self),
            datetime.datetime_fold(self),
        )

    def __deepcopy__(self, _: dict) -> _Pydt:
        return pydt_new(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
            datetime.datetime_hour(self),
            datetime.datetime_minute(self),
            datetime.datetime_second(self),
            datetime.datetime_microsecond(self),
            datetime.datetime_tzinfo(self),
            datetime.datetime_fold(self),
        )


class Pydt(_Pydt):
    """The drop in replacement for the standard Python `<'datetime.datetime'>`,
    with additional functionalities. Makes working with datetime more convenient.
    """

    def __new__(
        cls,
        year: cython.int = 1,
        month: cython.int = 1,
        day: cython.int = 1,
        hour: cython.int = 0,
        minute: cython.int = 0,
        second: cython.int = 0,
        microsecond: cython.int = 0,
        tzinfo: datetime.tzinfo | str | None = None,
        *,
        fold: cython.int = 0,
    ) -> Pydt:
        """The drop in replacement for the standard Python `<'datetime.datetime'>`,
        with additional functionalities. Makes working with datetime more convenient.

        :param year `<'int'>`: Year value (1-9999), defaults to `1`.
        :param month `<'int'>`: Month value (1-12), defaults to `1`.
        :param day `<'int'>`: Day value (1-31), defaults to `1`.
        :param hour `<'int'>`: Hour value (0-23), defaults to `0`.
        :param minute `<'int'>`: Minute value (0-59), defaults to `0`.
        :param second `<'int'>`: Second value (0-59), defaults to `0`.
        :param microsecond `<'int'>`: Microsecond value (0-999999), defaults to `0`.
        :param tzinfo `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            1. `<'datetime.tzinfo'>` subclass of datetime.tzinfo.
            2. `<'str'>` timezone name supported by 'Zoneinfo' module, or 'local' for local timezone.
            3. `<'NoneType'>` timezone-naive.

        :param fold `<'int'>`: Fold value (0/1), defaults to `0`.
        """
        return pydt_new(
            year, month, day, hour, minute, second, microsecond, tzinfo, fold
        )
