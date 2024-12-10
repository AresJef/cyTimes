# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from __future__ import annotations

# Cython imports
import cython
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
from babel.dates import format_date as _format_date
from zoneinfo import available_timezones as _available_timezones
from cytimes.parser import Configs, parse_dtobj as _parse
from cytimes import typeref, utils, errors

__all__ = ["Pydt"]


# Utils ---------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(-1, check=False)
def is_pydt(obj: object) -> cython.bint:
    """(cfunc) Check if the object is an instance of 'Pydt' `<'bool'>`.

    ### Equivalent to:
    >>> isinstance(obj, Pydt)
    """
    return isinstance(obj, _Pydt)


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
    """(cfunc) Construct a new Pydt instance `<'Pydt'>`.

    :param year `<'int'>`: Year value (1-9999), defaults to `1`.
    :param month `<'int'>`: Month value (1-12), defaults to `1`.
    :param day `<'int'>`: Day value (1-31), defaults to `1`.
    :param hour `<'int'>`: Hour value (0-23), defaults to `0`.
    :param minute `<'int'>`: Minute value (0-59), defaults to `0`.
    :param second `<'int'>`: Second value (0-59), defaults to `0`.
    :param microsecond `<'int'>`: Microsecond value (0-999999), defaults to `0`.
    :param tzinfo `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
        - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
        - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
        - `<'None'>` Timezone-naive.

    :param fold `<'int'>`: Fold value (0 or 1) for ambiguous times, defaults to `0`.
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
            raise errors.AmbiguousTimeError(err) from err
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
def _pydt_fr_dt(dt: datetime.datetime) -> _Pydt:
    """(internal) Construct a new Pydt instance from an existing datetime `<'Pydt'>`.

    :param dt `<'datetime'>`: An instance of 'datetime.datetime'.
    """
    # Normalize non-fixed timezone
    tz = dt.tzinfo
    if tz is not None and type(tz) is not typeref.TIMEZONE:
        # Normalize non-fixed timezone
        try:
            dt_norm = utils.dt_normalize_tz(dt)
        except ValueError as err:
            raise errors.AmbiguousTimeError(err) from err
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
def _pydt_fr_dtobj(
    dtobj: object,
    default: object | None = None,
    year1st: bool | None = None,
    day1st: bool | None = None,
    ignoretz: cython.bint = False,
    isoformat: cython.bint = True,
    cfg: Configs = None,
) -> _Pydt:
    """(internal) Parse datetime-like object into Pydt instance `<'Pydt'>.

    For information about the parameters, please refer to
    the 'cytimes.Pydt.parse()' classmethod.
    """
    # Parse default
    if default is not None:
        default = _parse_dtobj(default, None, year1st, day1st, ignoretz, isoformat, cfg)

    # Parse datetime-like object
    dt: datetime.datetime = _parse_dtobj(
        dtobj, default, year1st, day1st, ignoretz, isoformat, cfg
    )
    return _pydt_fr_dt(dt)


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
    """(internal) Parse datetime-like object into
    datetime instance `<'datetime.datetime'>.

    For information about the parameters, please refer to
    the 'cytimes.parser.parse_dtobj()' function.
    """
    try:
        return _parse(dtobj, default, year1st, day1st, ignoretz, isoformat, cfg)
    except Exception as err:
        raise errors.InvalidArgumentError(err) from err


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-2, check=False)
def _parse_month(month: object, raise_error: cython.bint) -> cython.int:
    """(internal) Parse the 'month' value to month number (1=Jan...12=Dec) `<'int'>`.

    :param month `<'int/str'>`: Month value.
        - `<'int'>` Month number (1=Jan...12=Dec).
        - `<'str'>` Month name (case-insensitive), e.g., 'Jan', 'februar', '三月'.

    :param raise_error `<'bool'>`: Whether to raise error if the 'month' value is invalid.
        - If `True`, raises `InvalidMonthError` for invalid input.
        - If `False`, returns `-1` instead.

    ## Notice:
    - Return `-1` directly if 'month' is `None` or `-1`.
    """
    # <'None'>
    if month is None:
        return -1  # exit

    # <'int'>
    if isinstance(month, int):
        num: cython.longlong = month
        if num == -1:
            return -1  # exit
        if not 1 <= num <= 12:
            if raise_error:
                raise errors.InvalidMonthError(
                    "invalid month number '%d', must betweem 1(Jan)...12(Dec)." % num
                )
            return -1  # exit
        return num  # exit

    # <'str'>
    if isinstance(month, str):
        mth: str = month
        val = dict_getitem(CONFIG_MONTH, mth.lower())
        if val == cython.NULL:
            if raise_error:
                raise errors.InvalidMonthError("invalid month name '%s'." % mth)
            return -1  # eixt
        return cython.cast(object, val)  # exit

    # Invalid
    if raise_error:
        raise errors.InvalidMonthError(
            "unsupported month value type %s, expects <'str/int'>." % type(month)
        )
    return -1


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-2, check=False)
def _parse_weekday(weekday: object, raise_error: cython.bint) -> cython.int:
    """(internal) Parse the 'weekday' value to weekday number (0=Mon...6=Sun) `<'int'>`.

    :param weekday `<'int/str/None'>`: Weekday value.
        - `<'int'>` Weekday number (0=Mon...6=Sun).
        - `<'str'>` Weekday name (case-insensitive), e.g., 'Mon', 'dienstag', '星期三'.

    :param raise_error `<'bool'>`: Whether to raise error if the 'weekday' value is invalid.
        - If `True`, raises `InvalidWeekdayError` for invalid input.
        - If `False`, returns `-1` instead.

    ## Notice:
    - Return `-1` directly if 'weekday' is `None` or `-1`.
    """
    # <'None'>
    if weekday is None:
        return -1

    # <'int'>
    if isinstance(weekday, int):
        num: cython.longlong = weekday
        if num == -1:
            return -1  # exit
        if not 0 <= num <= 6:
            if raise_error:
                raise errors.InvalidWeekdayError(
                    "invalid weekday number '%d', must betweem 0(Mon)...6(Sun)." % num
                )
            return -1
        return num  # exit

    # <'str'>
    if isinstance(weekday, str):
        wkd: str = weekday
        val = dict_getitem(CONFIG_WEEKDAY, wkd.lower())
        if val == cython.NULL:
            if raise_error:
                raise errors.InvalidWeekdayError("invalid weekday name '%s'." % wkd)
            return -1
        return cython.cast(object, val)  # exit

    # Invalid
    if raise_error:
        raise errors.InvalidWeekdayError(
            "unsupported weekday value type %s, expects <'str/int'>." % type(weekday)
        )
    return -1


@cython.cfunc
@cython.inline(True)
@cython.exceptval(-2, check=False)
def _parse_unit(unit: str, raise_error: cython.bint) -> cython.longlong:
    """(internal) Parse the datetime 'unit' to the corresponding
    conversion factor for microsecond `<'int'>`.

    :param unit `<'str'>`: The datetime unit: 'D', 'h', 'm', 's', 'ms', 'us'.
    :param raise_error `<'bool'>`: Whether to raise error if the 'unit' value is invalid.
        - If `True`, raises `InvalidTimeUnitError` for invalid input.
        - If `False`, returns `-1` instead.
    """
    unit_len: cython.Py_ssize_t = str_len(unit)

    # Unit: 's', 'm', 'h', 'D'
    if unit_len == 1:
        unit_ch: cython.Py_UCS4 = str_read(unit, 0)
        if unit_ch == "s":
            return utils.US_SECOND
        if unit_ch == "m":
            return utils.US_MINUTE
        if unit_ch == "h":
            return utils.US_HOUR
        if unit_ch == "D":
            return utils.US_DAY

    # Unit: 'ms', 'us', 'ns'
    elif unit_len == 2 and str_read(unit, 1) == "s":
        unit_ch: cython.Py_UCS4 = str_read(unit, 0)
        if unit_ch == "m":
            return utils.US_MILLISECOND
        if unit_ch in ("u", "n"):
            return 1

    # Unit: 'min' for pandas compatibility
    elif unit_len == 3 and unit == "min":
        return utils.US_MINUTE

    # Unsupported unit
    if raise_error:
        raise errors.InvalidTimeUnitError(
            "invalid datetime unit '%s'.\n"
            "Supports: ['D', 'h', 'm', 's', 'ms', 'us']." % unit
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
    """(internal) Compare two datetime objects `<'int'>`.

    :param dt1 `<'datetime.datetime'>`: The first instance of 'datetime.datetime'.
    :param dt2 `<'datetime.datetime'>`: The second instance of 'datetime.datetime'.
    :param allow_mixed `<'bool'>`: Whether to allow comparisons between naive and aware datetimes, defaults to `False`.
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
    desc: str = "compare",
) -> cython.bint:
    """(internal) Raise an `IncomparableError` for comparison
    between timezone-naive & timezone-aware datetimes.

    :param dt1 `<'datetime.datetime'>`: The first instance.
    :param dt2 `<'datetime.datetime'>`: The second instance.
    :param desc `<'str'>`: The description for the comparision, defaults to `'compare'`.
        Displayed as: 'cannot [desc] between naive & aware datetimes...'
    """
    d1_tz, d2_tz = dt1.tzinfo, dt2.tzinfo
    assert d1_tz is not d2_tz and (d1_tz is None or d2_tz is None)
    if d1_tz is None:
        raise errors.IncomparableError(
            "cannot %s between naive & aware datetimes:\n"
            "Timezone-naive '%s' %s\n"
            "Timezone-aware '%s' %s" % (desc, dt1, type(dt1), dt2, type(dt2))
        )
    else:
        raise errors.IncomparableError(
            "cannot %s between naive & aware datetimes:\n"
            "Timezone-aware '%s' %s\n"
            "Timezone-naive '%s' %s" % (desc, dt1, type(dt1), dt2, type(dt2))
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
        """Parse from a datetime-like object `<'Pydt'>`.

        :param dtobj `<'object'>`: Datetime-like object:
            - `<'str'>` A datetime string containing datetime information.
            - `<'datetime.datetime'>` An instance of `datetime.datetime`.
            - `<'datetime.date'>` An instance of `datetime.date` (time fields set to 0).
            - `<'int/float'>` Numeric value treated as total seconds since Unix Epoch.
            - `<'np.datetime64'>` Resolution above microseconds ('us') will be discarded.
            - `<'None'>` Return the current local datetime.

        ## Praser Parameters
        #### Parameters below only take effect when 'dtobj' is of type `<'str'>`.

        :param default `<'object'>`: Datetime-like object to fill in missing date fields, defaults to `None`.
            - If `None`, raises `InvalidArgumentError` if any Y/M/D fields is missing.

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
        return _pydt_fr_dtobj(dtobj, default, year1st, day1st, ignoretz, isoformat, cfg)

    @classmethod
    def now(cls, tz: datetime.tzinfo | str | None = None) -> _Pydt:
        """Contruct the current datetime with optional timezone `<'Pydt'>`.

        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.
        """
        return _pydt_fr_dt(utils.dt_now(utils.tz_parse(tz)))

    @classmethod
    def utcnow(cls) -> _Pydt:
        """Construct the current UTC datetime (timezone-aware) `<'Pydt'>`."""
        return _pydt_fr_dt(utils.dt_now(utils.UTC))

    @classmethod
    def today(cls) -> _Pydt:
        """Construct the current local datetime (timezone-naive) `<'Pydt'>`."""
        return _pydt_fr_dt(utils.dt_now(None))

    @classmethod
    def combine(
        cls,
        date: datetime.date | str | None = None,
        time: datetime.time | str | None = None,
        tz: datetime.tzinfo | str | None = None,
    ) -> _Pydt:
        """Combine date and time into a new datetime `<'Pydt'>`.

        :param date `<'date/str/None'>`: A date-like object, defaults to `None`.
            - `<'datetime.date'>` An instance of `datetime.date`.
            - `<'str'>` A date string.
            - `<'None'>` Use the current local date.

        :param time `<'time/str/None'>`: A time-like object, defaults to `None`.
            - `<'datetime.time'>` An instance of `datetime.time`.
            - `<'str'>` A time string.
            - `<'None'>` Use the current local time.

        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.
        """
        # Parse date
        if date is not None and not utils.is_date(date):
            date = _parse_dtobj(date)

        # Prase time
        if time is not None and not utils.is_time(time):
            time = _parse_dtobj(time, utils.EPOCH_DT).timetz()

        # New instance
        return _pydt_fr_dt(utils.dt_combine(date, time, utils.tz_parse(tz)))

    @classmethod
    def fromordinal(
        cls,
        ordinal: cython.int,
        tz: datetime.tzinfo | str | None = None,
    ) -> _Pydt:
        """Construct from Gregorian ordinal days with optional timzone `<'Pydt'>`.

        :param ordinal `<'int'>`: The proleptic Gregorian ordinal (day 1 is `0001-01-01`).
        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.
        """
        _ymd = utils.ymd_fr_ordinal(ordinal)
        return pydt_new(_ymd.year, _ymd.month, _ymd.day, 0, 0, 0, 0, tz, 0)

    @classmethod
    def fromseconds(
        cls,
        seconds: int | float,
        tz: datetime.tzinfo | str | None = None,
    ) -> _Pydt:
        """Construct from seconds since Unix Epoch with optional timezone `<'Pydt'>`.

        Unlike 'fromtimestamp()', this method does `NOT` take local
        timezone into consideration when constructing the datetime.

        :param seconds `<'int/float'>`: Total seconds since Unix Epoch.
        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.
        """
        return _pydt_fr_dt(utils.dt_fr_seconds(float(seconds), utils.tz_parse(tz)))

    @classmethod
    def fromicroseconds(
        cls,
        us: cython.longlong,
        tz: datetime.tzinfo | str | None = None,
    ) -> _Pydt:
        """Construct from microseconds since Unix Epoch with optional timezone `<'Pydt'>`.

        Unlike 'fromtimestamp()', this method does `NOT` take local
        timezone into consideration when constructing the datetime.

        :param us `<'int'>`: Total microseconds since Unix Epoch.
        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.
        """
        return _pydt_fr_dt(utils.dt_fr_us(us, utils.tz_parse(tz)))

    @classmethod
    def fromtimestamp(
        cls,
        ts: int | float,
        tz: datetime.tzinfo | str | None = None,
    ) -> _Pydt:
        """Construct from a POSIX timestamp with optional timezone `<'Pydt'>`.

        :param ts `<'int/float'>`: POSIX timestamp.
        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.
        """
        return _pydt_fr_dt(utils.dt_fr_ts(float(ts), utils.tz_parse(tz)))

    @classmethod
    def utcfromtimestamp(cls, ts: int | float) -> _Pydt:
        """Construct an UTC datetime (timezone-aware) from a POSIX timestamp `<'Pydt'>`.

        :param ts `<'int/float'>`: POSIX timestamp.
        """
        return _pydt_fr_dt(utils.dt_fr_ts(float(ts), utils.UTC))

    @classmethod
    def fromisoformat(cls, dtstr: str) -> _Pydt:
        """Construct from an ISO format string `<'Pydt'>`.

        :param dtstr `<'str'>`: The ISO format datetime string.
        """
        try:
            dt = datetime.datetime.fromisoformat(dtstr)
        except Exception as err:
            raise errors.InvalidArgumentError(err) from err
        return _pydt_fr_dt(dt)

    @classmethod
    def fromisocalendar(
        cls,
        year: cython.int,
        week: cython.int,
        weekday: cython.int,
        tz: datetime.tzinfo | str | None = None,
    ) -> _Pydt:
        """Construct from the ISO year, week number and weekday, with optional timezone `<'Pydt'>`.

        :param year `<'int'>`: The ISO year (1-9999).
        :param week `<'int'>`: The ISO week number (1-53).
        :param weekday `<'int'>`: The ISO weekday (1=Mon...7=Sun).
        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.
        """
        _ymd = utils.ymd_fr_isocalendar(year, week, weekday)
        return pydt_new(_ymd.year, _ymd.month, _ymd.day, 0, 0, 0, 0, tz, 0)

    @classmethod
    def fromdate(
        cls,
        date: datetime.date,
        tz: datetime.tzinfo | str | None = None,
    ) -> _Pydt:
        """Construct from an instance of date (all time fields set to 0) `<'Pydt'>`.

        :param date `<'datetime.date'>`: An instance of `datetime.date`.
        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.
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
        """Construct from an instance of datetime `<'Pydt'>`.

        :param dt `<'datetime.datetime'>`: An instance of `datetime.datetime`.
        """
        return _pydt_fr_dt(dt)

    @classmethod
    def fromdatetime64(
        cls,
        dt64: object,
        tz: datetime.tzinfo | str | None = None,
    ) -> _Pydt:
        """Construct from a numpy.datetime64 instance `<'Pydt'>`.

        :param dt64 `<'datetime64'>`: A `numpy.datetime64` instance.
        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.
        """
        return _pydt_fr_dt(utils.dt64_to_dt(dt64, utils.tz_parse(tz)))

    @classmethod
    def strptime(cls, dtstr: str, fmt: str) -> _Pydt:
        """Construct from a datetime string according to the given format `<'Pydt'>`.

        :param dtstr `<'str'>`: The datetime string.
        :param format `<'str'>`: The format used to parse the datetime strings.
        """
        try:
            dt = datetime.datetime.strptime(dtstr, fmt)
        except Exception as err:
            raise errors.InvalidArgumentError(err) from err
        return _pydt_fr_dt(dt)

    # . utils
    @cython.cfunc
    @cython.inline(True)
    def _from_dt(self, dt: datetime.datetime) -> _Pydt:
        """(internal) Construct 'Pydt' from the passed-in datetime `<'Pydt'>`.

        This method checks if the passed-in datetime is the same object
        as the instance itself. If so, returns the instance directly;
        otherwise, creates a new Pydt instance from the datetime.

        :param dt `<'datetime.datetime'>`: An instance of `datetime.datetime`.
        """
        return self if dt is self else _pydt_fr_dt(dt)

    # Convertor ----------------------------------------------------------------------------
    @cython.ccall
    def ctime(self) -> str:
        """Convert to string in C time format `<'str'>`.

        - ctime format: 'Tue Oct  1 08:19:05 2024'
        """
        return utils.dt_to_ctime(self)

    @cython.ccall
    def strftime(self, fmt: str) -> str:
        """Convert to string according to the given format `<'str'>`.

        :param format `<'str'>`: The format of the datetime string.
        """
        return utils.dt_to_strformat(self, fmt)

    @cython.ccall
    def isoformat(self, sep: str = "T") -> str:
        """Convert to string representing the date and time in ISO format `<'str'>`.

        The default format is 'YYYY-MM-DDTHH:MM:SS[.f]' with an optional fractional
        part when microseconds are non-zero. If 'tzinfo' is present, the UTC offset
        is included, resulting in 'YYYY-MM-DDTHH:MM:SS[.f]+HH:MM'.

        :param sep `<'str'>`: The separator between date and time components, defaults to `'T'`.
        """
        return utils.dt_to_isoformat(self, sep, True)

    @cython.ccall
    def timedict(self) -> dict[str, int]:
        """Convert a dictionary of time components `<'dict'>`.

        ### Example:
        >>> dt.timedict()
        >>> {
                'tm_year': 2024,
                'tm_mon': 10,
                'tm_mday': 11,
                'tm_hour': 8,
                'tm_min': 14,
                'tm_sec': 11,
                'tm_wday': 4,
                'tm_yday': 285,
                'tm_isdst': 1
            }
        """
        _tm = utils.dt_to_tm(self, False)
        return {
            "tm_year": _tm.tm_year,
            "tm_mon": _tm.tm_mon,
            "tm_mday": _tm.tm_mday,
            "tm_hour": _tm.tm_hour,
            "tm_min": _tm.tm_min,
            "tm_sec": _tm.tm_sec,
            "tm_wday": _tm.tm_wday,
            "tm_yday": _tm.tm_yday,
            "tm_isdst": _tm.tm_isdst,
        }

    @cython.ccall
    def utctimedict(self) -> dict[str, int]:
        """Convert a dictionary of time components representing the UTC time `<'dict'>`.

        ### Example:
        >>> dt.utctimedict()
        >>> {
                'tm_year': 2024,
                'tm_mon': 10,
                'tm_mday': 11,
                'tm_hour': 6,
                'tm_min': 15,
                'tm_sec': 6,
                'tm_wday': 4,
                'tm_yday': 285,
                'tm_isdst': 0
            }
        """
        _tm = utils.dt_to_tm(self, True)
        return {
            "tm_year": _tm.tm_year,
            "tm_mon": _tm.tm_mon,
            "tm_mday": _tm.tm_mday,
            "tm_hour": _tm.tm_hour,
            "tm_min": _tm.tm_min,
            "tm_sec": _tm.tm_sec,
            "tm_wday": _tm.tm_wday,
            "tm_yday": _tm.tm_yday,
            "tm_isdst": _tm.tm_isdst,
        }

    @cython.ccall
    def timetuple(self) -> tuple[int, ...]:
        """Convert a tuple of time components `<'tuple'>`.

        #### Note: this method returns `<'tuple'>` instead of `<'time.struct_time'>`.

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
        """Convert a tuple of time components representing the UTC time `<'tuple'>`.

        #### Note: this method returns `<'tuple'>` instead of `<'time.struct_time'>`.

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
        """Convert to proleptic Gregorian ordinal days `<'int'>`.

        - Day 1 (ordinal=1) is `0001-01-01`.
        """
        return utils.dt_to_ordinal(self)

    @cython.ccall
    def seconds(self, utc: cython.bint = False) -> cython.double:
        """Convert to total seconds since Unix Epoch `<'float'>`.

        Unlike 'timesamp()', this method does `NOT` take local
        timezone into consideration at conversion.

        :param utc `<'bool'>`: Whether to subtract the UTC offset from the result, defaults to `False`.
            Only applicable when instance is timezone-aware; otherwise ignored.
        """
        return utils.dt_to_seconds(self, utc)

    @cython.ccall
    def microseconds(self, utc: cython.bint = False) -> cython.longlong:
        """Convert to total microseconds since Unix Epoch `<'int'>`.

        Unlike 'timesamp()', this method does `NOT` take local
        timezone into consideration at conversion.

        :param utc `<'bool'>`: Whether to subtract the UTC offset from the result, defaults to `False`.
            Only applicable when instance is timezone-aware; otherwise ignored.
        """
        return utils.dt_to_us(self, utc)

    @cython.ccall
    def timestamp(self) -> cython.double:
        """Convert to POSIX timestamp `<'float'>`."""
        return utils.dt_to_ts(self)

    @cython.ccall
    def date(self) -> datetime.date:
        """Convert to date (all time fields set to 0) `<'datetime.date'>`."""
        return datetime.date_new(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
        )

    @cython.ccall
    def time(self) -> datetime.time:
        """Convert to time (`WITHOUT` timezone) `<'datetime.time'>`."""
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
        """Convert to time (`WITH` timezone) `<'datetime.time'>`."""
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
        tzinfo: datetime.tzinfo | str | None = -1,
        fold: cython.int = -1,
    ) -> _Pydt:
        """Replace the specified datetime fields with new values `<'Pydt'>`.

        #### Fields set to `-1` means retaining the original values.

        :param year `<'int'>`: Year value (1-9999), defaults to `-1`.
        :param month `<'int'>`: Yonth value (1-12), defaults to `-1`.
        :param day `<'int'>`: Day value (1-31), automacially clamped to the maximum days in the month, defaults to `-1`.
        :param hour `<'int'>`: Hour value (0-23), defaults to `-1`.
        :param minute `<'int'>`: Minute value (0-59), defaults to `-1`.
        :param second `<'int'>`: Second value (0-59), defaults to `-1`.
        :param microsecond `<'int'>`: Microsecond value (0-999999), defaults to `-1`.
        :param tz `<'tzinfo/str/None'>`: The timezone, defaults to `-1`.
            - `<'int'>` Retains the original timezone.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.

        :param fold `<'int'>`: Fold value (0 or 1) for ambiguous times, defaults to `-1`.
        """
        # Prase timezone
        if not isinstance(tzinfo, int):
            tzinfo = utils.tz_parse(tzinfo)

        # Access date & time fields
        yy: cython.int = datetime.datetime_year(self)
        mm: cython.int = datetime.datetime_month(self)
        dd: cython.int = datetime.datetime_day(self)
        hh: cython.int = datetime.datetime_hour(self)
        mi: cython.int = datetime.datetime_minute(self)
        ss: cython.int = datetime.datetime_second(self)
        us: cython.int = datetime.datetime_microsecond(self)
        tz = datetime.datetime_tzinfo(self)
        fd: cython.int = datetime.datetime_fold(self)

        # New values
        yy_new: cython.int = yy if year < 1 else min(year, 9_999)
        mm_new: cython.int = mm if month < 1 else min(month, 12)
        if day < 1:
            dd_new: cython.int = dd
        elif day < 29:
            dd_new: cython.int = day
        else:
            dd_new: cython.int = min(day, utils.days_in_month(yy_new, mm_new))
        hh_new: cython.int = hh if hour < 0 else min(hour, 23)
        mi_new: cython.int = mi if minute < 0 else min(minute, 59)
        ss_new: cython.int = ss if second < 0 else min(second, 59)
        us_new: cython.int = us if microsecond < 0 else min(microsecond, 999999)
        tz_new = tz if tzinfo is not None and not utils.is_tz(tzinfo) else tzinfo
        fd_new: cython.int = fd if fold not in (0, 1) else fold

        # Same values
        # fmt: off
        if (
            yy == yy_new and mm == mm_new and dd == dd_new 
            and hh == hh_new and mi == mi_new and ss == ss_new 
            and us == us_new and tz is tz_new and fd == fd_new
        ):
            return self  # exit

        # New instance
        return pydt_new(
            yy_new, mm_new, dd_new, 
            hh_new, mi_new, ss_new, 
            us_new, tz_new, fd_new
        )
        # fmt: on

    # . year
    @cython.ccall
    def to_curr_year(
        self,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> _Pydt:
        """Adjust the date to the specified month and day in the current year `<'Pydt'>`.

        :param month `<'int/str/None'>`: Month value, defaults to `None`.
            - `<'int'>` Month number (1=Jan...12=Dec).
            - `<'str'>` Month name (case-insensitive), e.g., 'Jan', 'februar', '三月'.
            - `<'None'>` Retains the original month.

        :param day `<'int'>`: Day value (1-31), defaults to `-1`.
            If `-1`, retains the original day. The final day
            value is clamped to the maximum days in the month.

        ### Example:
        >>> dt.to_curr_year("Feb", 31)  # The last day of February in the current year
        >>> dt.to_curr_year(11)         # The same day of November in the current year
        >>> dt.to_curr_year(day=1)      # The first day of the current month
        """
        # Parse month
        mm: cython.int = _parse_month(month, True)
        if mm == -1 or mm == datetime.datetime_month(self):
            return self.to_curr_month(day)  # exit: same month
        yy: cython.int = datetime.datetime_year(self)

        # Clamp to max days
        dd: cython.int = datetime.datetime_day(self) if day < 1 else day
        if dd > 28:
            dd = min(dd, utils.days_in_month(yy, mm))

        # New instance
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
        """Adjust the date to the specified month and day in the previous year `<'Pydt'>`.

        :param month `<'int/str/None'>`: Month value, defaults to `None`.
            - `<'int'>` Month number (1=Jan...12=Dec).
            - `<'str'>` Month name (case-insensitive), e.g., 'Jan', 'februar', '三月'.
            - `<'None'>` Retains the original month.

        :param day `<'int'>`: Day value (1-31), defaults to `-1`.
            If `-1`, retains the original day. The final day
            value is clamped to the maximum days in the month.

        ### Example:
        >>> dt.to_prev_year("Feb", 31)  # The last day of February in the previous year
        >>> dt.to_prev_year(11)         # The same day of November in the previous year
        >>> dt.to_prev_year(day=1)      # The first day of the current month in the previous year
        """
        return self.to_year(-1, month, day)

    @cython.ccall
    def to_next_year(
        self,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> _Pydt:
        """Adjust the date to the specified month and day in the next year `<'Pydt'>`.

        :param month `<'int/str/None'>`: Month value, defaults to `None`.
            - `<'int'>` Month number (1=Jan...12=Dec).
            - `<'str'>` Month name (case-insensitive), e.g., 'Jan', 'februar', '三月'.
            - `<'None'>` Retains the original month.

        :param day `<'int'>`: Day value (1-31), defaults to `-1`.
            If `-1`, retains the original day. The final day
            value is clamped to the maximum days in the month.

        ### Example:
        >>> dt.to_next_year("Feb", 31)  # The last day of February in the next year
        >>> dt.to_next_year(11)         # The same day of November in the next year
        >>> dt.to_next_year(day=1)      # The first day of the current month in the next year
        """
        return self.to_year(1, month, day)

    @cython.ccall
    def to_year(
        self,
        offset: cython.int,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> _Pydt:
        """Adjust the date to the specified month and day in the year (+/-) offset `<'Pydt'>`.

        :param offset `<'int'>`: The year offset (+/-).

        :param month `<'int/str/None'>`: Month value, defaults to `None`.
            - `<'int'>` Month number (1=Jan...12=Dec).
            - `<'str'>` Month name (case-insensitive), e.g., 'Jan', 'februar', '三月'.
            - `<'None'>` Retains the original month.

        :param day `<'int'>`: Day value (1-31), defaults to `-1`.
            If `-1`, retains the original day. The final day
            value is clamped to the maximum days in the month.

        ### Example:
        >>> dt.to_year(-2, "Feb", 31)  # The last day of February, two years ago
        >>> dt.to_year(2, 11)          # The same day of November, two years later
        >>> dt.to_year(2, day=1)       # The first day of the current month, two years later
        """
        # No offset
        if offset == 0:
            return self.to_curr_year(month, day)  # exit

        # Compute new year
        yy: cython.int = datetime.datetime_year(self) + offset
        yy = min(max(yy, 1), 9999)

        # Parse month
        mm: cython.int = _parse_month(month, True)
        if mm == -1:
            mm = datetime.datetime_month(self)

        # Clamp to max days
        dd: cython.int = datetime.datetime_day(self) if day < 1 else day
        if dd > 28:
            dd = min(dd, utils.days_in_month(yy, mm))

        # New instance
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
        """Adjust the date to the specified month and day in the current quarter. `<'Pydt'>`.

        :param month `<'int'>`: Month (1-3) of the quarter, defaults to `-1`.
            If `-1`, retains the original month of the quarter.

        :param day `<'int'>`: Day value (1-31), defaults to `-1`.
            If `-1`, retains the original day. The final day
            value is clamped to the maximum days in the month.

        ### Example:
        >>> dt.to_curr_quarter(1, 31)  # The last day of the first quarter month in the current quarter
        >>> dt.to_curr_quarter(2)      # The same day of the second quarter month in the current quarter
        >>> dt.to_curr_quarter(day=1)  # The first day of the current quarter month in the current quarter
        """
        # Fast-path: no adjustment
        if month < 1:
            return self.to_curr_month(day)  # exit

        # Compute new month
        curr_mm: cython.int = datetime.datetime_month(self)
        mm = utils.quarter_of_month(curr_mm) * 3 + (min(month, 3) - 3)
        if mm == curr_mm:
            return self.to_curr_month(day)  # exit: same month
        yy: cython.int = datetime.datetime_year(self)

        # Clamp to max days
        dd: cython.int = datetime.datetime_day(self) if day < 1 else day
        if dd > 28:
            dd = min(dd, utils.days_in_month(yy, mm))

        # New instance
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
        """Adjust the date to the specified month and day in the previous quarter. `<'Pydt'>`.

        :param month `<'int'>`: Month (1-3) of the quarter, defaults to `-1`.
            If `-1`, retains the original month of the quarter.

        :param day `<'int'>`: Day value (1-31), defaults to `-1`.
            If `-1`, retains the original day. The final day
            value is clamped to the maximum days in the month.

        ### Example:
        >>> dt.to_prev_quarter(1, 31)  # The last day of the first quarter month in the previous quarter
        >>> dt.to_prev_quarter(2)      # The same day of the second quarter month in the previous quarter
        >>> dt.to_prev_quarter(day=1)  # The first day of the current quarter month in the previous quarter
        """
        return self.to_quarter(-1, month, day)

    @cython.ccall
    def to_next_quarter(self, month: cython.int = -1, day: cython.int = -1) -> _Pydt:
        """Adjust the date to the specified month and day in the next quarter. `<'Pydt'>`.

        :param month `<'int'>`: Month (1-3) of the quarter, defaults to `-1`.
            If `-1`, retains the original month of the quarter.

        :param day `<'int'>`: Day value (1-31), defaults to `-1`.
            If `-1`, retains the original day. The final day
            value is clamped to the maximum days in the month.

        ### Example:
        >>> dt.to_next_quarter(1, 31)  # The last day of the first quarter month in the next quarter
        >>> dt.to_next_quarter(2)      # The same day of the second quarter month in the next quarter
        >>> dt.to_next_quarter(day=1)  # The first day of the current quarter month in the next quarter
        """
        return self.to_quarter(1, month, day)

    @cython.ccall
    def to_quarter(
        self,
        offset: cython.int,
        month: cython.int = -1,
        day: cython.int = -1,
    ) -> _Pydt:
        """Adjust the date to the specified month and day in the quarter (+/-) 'offset'. `<'Pydt'>`.

        :param offset `<'int'>`: The quarter offset (+/-).

        :param month `<'int'>`: Month (1-3) of the quarter, defaults to `-1`.
            If `-1`, retains the original month of the quarter.

        :param day `<'int'>`: Day value (1-31), defaults to `-1`.
            If `-1`, retains the original day. The final day
            value is clamped to the maximum days in the month.

        ### Example:
        >>> dt.to_quarter(-2, 1, 31)  # The last day of the first quarter month, two quarters ago
        >>> dt.to_quarter(2, 2)       # The same day of the second quarter month, two quarters later
        >>> dt.to_quarter(2, day=1)   # The first day of the current quarter month, two quarters later
        """
        # Fast-path: no offset
        if offset == 0:
            return self.to_curr_quarter(month, day)  # exit

        # Compute new year & month
        yy: cython.int = datetime.datetime_year(self)
        mm: cython.int = datetime.datetime_month(self)
        if month >= 1:
            mm = utils.quarter_of_month(mm) * 3 + (min(month, 3) - 3)
        mm += offset * 3
        if mm > 12:
            yy += mm // 12
            mm %= 12
        elif mm < 1:
            mm = 12 - mm
            yy -= mm // 12
            mm = 12 - mm % 12
        yy = min(max(yy, 1), 9999)

        # Clamp to max days
        dd: cython.int = datetime.datetime_day(self) if day < 1 else day
        if dd > 28:
            dd = min(dd, utils.days_in_month(yy, mm))

        # New instance
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
        """Adjust the date to the specified day of the current month `<'Pydt'>`.

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            If `-1`, retains the original day. The final day
            value is clamped to the maximum days in the month.

        ### Example:
        >>> dt.to_curr_month(31)  # The last day of the current month
        >>> dt.to_curr_month(1)   # The first day of the current month
        """
        # Fast-path: no adjustment
        if day < 1:
            return self  # exit

        # Clamp to max days
        yy: cython.int = datetime.datetime_year(self)
        mm: cython.int = datetime.datetime_month(self)
        if day > 28:
            day = min(day, utils.days_in_month(yy, mm))
        if day == datetime.datetime_day(self):
            return self  # exit: same day

        # New instance
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
        """Adjust the date to the specified day of the previous month `<'Pydt'>`.

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            If `-1`, retains the original day. The final day
            value is clamped to the maximum days in the month.

        ### Example:
        >>> dt.to_prev_month(31)  # The last day of the previous month
        >>> dt.to_prev_month(1)   # The first day of the previous month
        """
        return self.to_month(-1, day)

    @cython.ccall
    def to_next_month(self, day: cython.int = -1) -> _Pydt:
        """Adjust the date to the specified day of the next month `<'Pydt'>`.

        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            If `-1`, retains the original day. The final day
            value is clamped to the maximum days in the month.

        ### Example:
        >>> dt.to_next_month(31)  # The last day of the next month
        >>> dt.to_next_month(1)   # The first day of the next month
        """
        return self.to_month(1, day)

    @cython.ccall
    def to_month(self, offset: cython.int, day: cython.int = -1) -> _Pydt:
        """Adjust the date to the specified day of the month (+/-) offest `<'Pydt'>`.

        :param offset `<'int'>`: The month offset (+/-).
        :param day `<'int'>`: The day value (1-31), defaults to `-1`.
            If `-1`, retains the original day. The final day
            value is clamped to the maximum days in the month.

        ### Example:
        >>> dt.to_month(-2, 31)  # The last day of the month, two months ago
        >>> dt.to_month(2, 1)    # The first day of the month, two months later
        """
        # Fast-path: no offset
        if offset == 0:
            return self.to_curr_month(day)  # exit

        # Compute new year & month
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

        # Clamp to max days
        dd: cython.int = datetime.datetime_day(self) if day < 1 else day
        if dd > 28:
            dd = min(dd, utils.days_in_month(yy, mm))

        # New instance
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
        """Adjust the date to the Monday of the current week `<'Pydt'>`."""
        return self._to_curr_weekday(0)

    @cython.ccall
    def to_tuesday(self) -> _Pydt:
        """Adjust the date to the Tuesday of the current week `<'Pydt'>`."""
        return self._to_curr_weekday(1)

    @cython.ccall
    def to_wednesday(self) -> _Pydt:
        """Adjust the date to the Wednesday of the current week `<'Pydt'>`."""
        return self._to_curr_weekday(2)

    @cython.ccall
    def to_thursday(self) -> _Pydt:
        """Adjust the date to the Thursday of the current week `<'Pydt'>`."""
        return self._to_curr_weekday(3)

    @cython.ccall
    def to_friday(self) -> _Pydt:
        """Adjust the date to the Friday of the current week `<'Pydt'>`."""
        return self._to_curr_weekday(4)

    @cython.ccall
    def to_saturday(self) -> _Pydt:
        """Adjust the date to the Saturday of the current week `<'Pydt'>`."""
        return self._to_curr_weekday(5)

    @cython.ccall
    def to_sunday(self) -> _Pydt:
        """Adjust the date to the Sunday of the current week `<'Pydt'>`."""
        return self._to_curr_weekday(6)

    @cython.ccall
    def to_curr_weekday(self, weekday: int | str | None = None) -> _Pydt:
        """Adjust the date to the specific weekday of the current week `<'Pydt'>`.

        :param weekday `<'int/str/None'>`: Weekday value, defaults to `None`.
            - `<'int'>` Weekday number (0=Mon...6=Sun).
            - `<'str'>` Weekday name (case-insensitive), e.g., 'Mon', 'dienstag', '星期三'.
            - `<'None'>` Retains the original weekday.

        ### Example:
        >>> dt.to_curr_weekday(0)      # The Monday of the current week
        >>> dt.to_curr_weekday("Tue")  # The Tuesday of the current week
        """
        return self._to_curr_weekday(_parse_weekday(weekday, True))

    @cython.ccall
    def to_prev_weekday(self, weekday: int | str | None = None) -> _Pydt:
        """Adjust the date to the specific weekday of the previous week `<'Pydt'>`.

        :param weekday `<'int/str/None'>`: Weekday value, defaults to `None`.
            - `<'int'>` Weekday number (0=Mon...6=Sun).
            - `<'str'>` Weekday name (case-insensitive), e.g., 'Mon', 'dienstag', '星期三'.
            - `<'None'>` Retains the original weekday.

        ### Example:
        >>> dt.to_prev_weekday(0)      # The Monday of the previous week
        >>> dt.to_prev_weekday("Tue")  # The Tuesday of the previous week
        """
        return self.to_weekday(-1, weekday)

    @cython.ccall
    def to_next_weekday(self, weekday: int | str | None = None) -> _Pydt:
        """Adjust the date to the specific weekday of the next week `<'Pydt'>`.

        :param weekday `<'int/str/None'>`: Weekday value, defaults to `None`.
            - `<'int'>` Weekday number (0=Mon...6=Sun).
            - `<'str'>` Weekday name (case-insensitive), e.g., 'Mon', 'dienstag', '星期三'.
            - `<'None'>` Retains the original weekday.

        ### Example:
        >>> dt.to_next_weekday(0)      # The Monday of the next week
        >>> dt.to_next_weekday("Tue")  # The Tuesday of the next week
        """
        return self.to_weekday(1, weekday)

    @cython.ccall
    def to_weekday(self, offset: cython.int, weekday: int | str | None = None) -> _Pydt:
        """Adjust the date to the specific weekday of the week (+/-) offset `<'Pydt'>`.

        :param offset `<'int'>`: The week offset (+/-).
        :param weekday `<'int/str/None'>`: Weekday value, defaults to `None`.
            - `<'int'>` Weekday number (0=Mon...6=Sun).
            - `<'str'>` Weekday name (case-insensitive), e.g., 'Mon', 'dienstag', '星期三'.
            - `<'None'>` Retains the original weekday.

        ### Example:
        >>> dt.to_weekday(-2, 0)     # The Monday of the week, two weeks ago
        >>> dt.to_weekday(2, "Tue")  # The Tuesday of the week, two weeks later
        >>> dt.to_weekday(2)         # The same weekday of the week, two weeks later
        """
        # Fast-path: no offset
        wkd: cython.int = _parse_weekday(weekday, True)
        if offset == 0:
            return self._to_curr_weekday(wkd)  # exit

        # Compute new weekday
        days: cython.int = offset * 7
        if wkd != -1:
            days += wkd - self._prop_weekday()

        # New instance
        return self.to_day(days)

    @cython.cfunc
    @cython.inline(True)
    def _to_curr_weekday(self, weekday: cython.int) -> _Pydt:
        """(internal) Adjust the date to the specific weekday of the current week `<'Pydt'>`.

        :param weekday `<'int'>`: Weekday number (0=Mon...6=Sun).
        """
        # Fast-path: no adjustment
        if weekday < 0:
            return self  # exit

        # New instance
        return self.to_day(weekday - self._prop_weekday())

    # . day
    @cython.ccall
    def to_yesterday(self) -> _Pydt:
        """Adjust the date to Yesterday `<'Pydt'>`."""
        return self.to_day(-1)

    @cython.ccall
    def to_tomorrow(self) -> _Pydt:
        """Adjust the date to Tomorrow `<'Pydt'>`."""
        return self.to_day(1)

    @cython.ccall
    def to_day(self, offset: cython.int) -> _Pydt:
        """Adjust the date to day (+/-) offset `<'Pydt'>`.

        :param offset `<'int'>`: The day offset (+/-).
        """
        # Fast-path: no adjustment
        if offset == 0:
            return self  # exit

        # Add day offsets
        ordinal: cython.int = utils.ymd_to_ordinal(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
        )
        _ymd = utils.ymd_fr_ordinal(ordinal + offset)

        # New instance
        return pydt_new(
            _ymd.year,
            _ymd.month,
            _ymd.day,
            datetime.datetime_hour(self),
            datetime.datetime_minute(self),
            datetime.datetime_second(self),
            datetime.datetime_microsecond(self),
            datetime.datetime_tzinfo(self),
            datetime.datetime_fold(self),
        )

    # . date&time
    @cython.ccall
    def normalize(self) -> _Pydt:
        """Set the time fields to midnight (i.e. 00:00:00) `<'Pydt'>`.

        This method is useful in cases, when the time
        does not matter. The timezone is unaffected.
        """
        return self.to_time(0, 0, 0, 0)

    @cython.ccall
    def to_datetime(
        self,
        year: cython.int = -1,
        month: cython.int = -1,
        day: cython.int = -1,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        microsecond: cython.int = -1,
    ) -> _Pydt:
        """Adjust the date and time fields with new values `<'Pydt'>`.

        #### Fields set to `-1` means retaining the original values.

        :param year `<'int'>`: Year value (1-9999), defaults to `-1`.
        :param month `<'int'>`: Yonth value (1-12), defaults to `-1`.
        :param day `<'int'>`: Day value (1-31), automacially clamped to the maximum days in the month, defaults to `-1`.
        :param hour `<'int'>`: Hour value (0-23), defaults to `-1`.
        :param minute `<'int'>`: Minute value (0-59), defaults to `-1`.
        :param second `<'int'>`: Second value (0-59), defaults to `-1`.
        :param microsecond `<'int'>`: Microsecond value (0-999999), defaults to `-1`.

        ### Equivalent to:
        >>> dt.replace(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=second,
                microsecond=microsecond,
            )
        """
        # Access date & time fields
        yy: cython.int = datetime.datetime_year(self)
        mm: cython.int = datetime.datetime_month(self)
        dd: cython.int = datetime.datetime_day(self)
        hh: cython.int = datetime.datetime_hour(self)
        mi: cython.int = datetime.datetime_minute(self)
        ss: cython.int = datetime.datetime_second(self)
        us: cython.int = datetime.datetime_microsecond(self)

        # New values
        yy_new: cython.int = yy if year < 1 else min(year, 9_999)
        mm_new: cython.int = mm if month < 1 else min(month, 12)
        if day < 1:
            dd_new: cython.int = dd
        elif day < 29:
            dd_new: cython.int = day
        else:
            dd_new: cython.int = min(day, utils.days_in_month(yy_new, mm_new))
        hh_new: cython.int = hh if hour < 0 else min(hour, 23)
        mi_new: cython.int = mi if minute < 0 else min(minute, 59)
        ss_new: cython.int = ss if second < 0 else min(second, 59)
        us_new: cython.int = us if microsecond < 0 else min(microsecond, 999999)

        # Same values
        # fmt: off
        if (
            yy == yy_new and mm == mm_new and dd == dd_new 
            and hh == hh_new and mi == mi_new and ss == ss_new and us == us_new
        ):
            return self  # exit

        # New instance
        return pydt_new(
            yy_new, mm_new, dd_new, 
            hh_new, mi_new, ss_new, us_new,
            datetime.datetime_tzinfo(self),
            datetime.datetime_fold(self),
        )
        # fmt on

    @cython.ccall
    def to_date(
        self,
        year: cython.int = -1,
        month: cython.int = -1,
        day: cython.int = -1,
    ) -> _Pydt:
        """Adjust the date with new values `<'Pydt'>`.

        #### Fields set to `-1` means retaining the original values.

        :param year `<'int'>`: Year value (1-9999), defaults to `-1`.
        :param month `<'int'>`: Yonth value (1-12), defaults to `-1`.
        :param day `<'int'>`: Day value (1-31), automacially clamped to the maximum days in the month, defaults to `-1`.

        ### Equivalent to:
        >>> dt.replace(year=year, month=month, day=day)
        """
        # Access date fields
        yy: cython.int = datetime.datetime_year(self)
        mm: cython.int = datetime.datetime_month(self)
        dd: cython.int = datetime.datetime_day(self)

        # New values
        yy_new: cython.int = yy if year < 1 else min(year, 9_999)
        mm_new: cython.int = mm if month < 1 else min(month, 12)
        if day < 1:
            dd_new: cython.int = dd
        elif day < 29:
            dd_new: cython.int = day
        else:
            dd_new: cython.int = min(day, utils.days_in_month(yy_new, mm_new))

        # Same values
        if yy == yy_new and mm == mm_new and dd == dd_new:
            return self  # exit

        # New instance
        # fmt: off
        return pydt_new(
            yy_new, mm_new, dd_new,
            datetime.datetime_hour(self),
            datetime.datetime_minute(self),
            datetime.datetime_second(self),
            datetime.datetime_microsecond(self),
            datetime.datetime_tzinfo(self),
            datetime.datetime_fold(self),
        )
        # fmt: on

    @cython.ccall
    def to_time(
        self,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        microsecond: cython.int = -1,
    ) -> _Pydt:
        """Adjust the time fields with new values `<'Pydt'>`.

        #### Fields set to `-1` means retaining the original values.

        :param hour `<'int'>`: Hour value (0-23), defaults to `-1`.
        :param minute `<'int'>`: Minute value (0-59), defaults to `-1`.
        :param second `<'int'>`: Second value (0-59), defaults to `-1`.
        :param microsecond `<'int'>`: Microsecond value (0-999999), defaults to `-1`.

        ### Equivalent to:
        >>> dt.replace(
                hour=hour,
                minute=minute,
                second=second,
                microsecond=microsecond,
            )
        """
        # Access time fields
        hh: cython.int = datetime.datetime_hour(self)
        mi: cython.int = datetime.datetime_minute(self)
        ss: cython.int = datetime.datetime_second(self)
        us: cython.int = datetime.datetime_microsecond(self)

        # New values
        hh_new: cython.int = hh if hour < 0 else min(hour, 23)
        mi_new: cython.int = mi if minute < 0 else min(minute, 59)
        ss_new: cython.int = ss if second < 0 else min(second, 59)
        us_new: cython.int = us if microsecond < 0 else min(microsecond, 999999)

        # Same values
        if hh == hh_new and mi == mi_new and ss == ss_new and us == us_new:
            return self  # exit

        # New instance
        # fmt: off
        return pydt_new(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
            hh_new, mi_new, ss_new, us_new,
            datetime.datetime_tzinfo(self),
            datetime.datetime_fold(self),
        )
        # fmt: on

    @cython.ccall
    def to_first_of(self, unit: str) -> _Pydt:
        """Adjust the date to the first day of the specified datetime unit `<'Pydt'>`.

        :param unit `<'str'>`: The datetime unit.
            - `'Y'`: Sets to the first day of the current year.
            - `'Q'`: Sets to the first day of the current quarter.
            - `'M'`: Sets to the first day of the current month.
            - `'W'`: Sets to the first day (Monday) of the current week.
            - Month name (e.g., `'Jan'`, `'February'`, `'三月'`): Sets to the first day of that month.
        """
        unit_len: cython.Py_ssize_t = str_len(unit)

        # Unit: 'W', 'M', 'Q', 'Y'
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

        # Month name
        val: cython.int = _parse_month(unit, False)
        if val != -1:
            return self.to_date(-1, val, 1)

        # Unsupported unit
        raise errors.InvalidTimeUnitError(
            "invalid 'first of' datetime unit '%s'.\n"
            "Supports: ['Y', 'Q', 'M', 'W'] or Month name." % unit
        )

    @cython.ccall
    def to_last_of(self, unit: str) -> _Pydt:
        """Adjust the date to the last day of the specified datetime unit `<'Pydt'>`.

        :param unit `<'str'>`: The datetime unit.
            - `'Y'`: Sets to the last day of the current year.
            - `'Q'`: Sets to the last day of the current quarter.
            - `'M'`: Sets to the last day of the current month.
            - `'W'`: Sets to the last day (Sunday) of the current week.
            - Month name (e.g., `'Jan'`, `'February'`, `'三月'`): Sets to the last day of that month.
        """
        unit_len: cython.Py_ssize_t = str_len(unit)

        # Unit: 'W', 'M', 'Q', 'Y'
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

        # Month name
        val: cython.int = _parse_month(unit, False)
        if val != -1:
            return self.to_date(-1, val, 31)

        # Unsupported unit
        raise errors.InvalidTimeUnitError(
            "invalid 'last of' datetime unit '%s'.\n"
            "Supports: ['Y', 'Q', 'M', 'W'] or Month name." % unit
        )

    @cython.ccall
    def to_start_of(self, unit: str) -> _Pydt:
        """Adjust the datetime to the start of the specified datetime unit `<'Pydt'>`.

        :param unit `<'str'>`: The datetime unit.
            - `'Y'`: Sets date to the first day of the current year & time to '00:00:00.000000'.
            - `'Q'`: Sets date to the first day of the current quarter & time to '00:00:00.000000'.
            - `'M'`: Sets date to the first day of the current month & time to '00:00:00.000000'.
            - `'W'`: Sets date to the first day (Monday) of the current week & time to '00:00:00.000000'.
            - `'D'`: Retains the original date & sets time to '00:00:00.000000'.
            - `'h'`: Retains the original date & sets time to 'XX:00:00.000000'.
            - `'m'`: Retains the original date & sets time to 'XX:XX:00.000000'.
            - `'s'`: Retains the original date & sets time to 'XX:XX:XX.000000'.
            - `'ms'`: Retains the original date & sets time to 'XX:XX:XX.XXX000'.
            - `'us'`: Return the instance itself.
            - Month name (e.g., `'Jan'`, `'February'`, `'三月'`): Sets date to first day of that month & time to '00:00:00.000000'.
            - Weekday name (e.g., `'Mon'`, `'Tuesday'`, `'星期三'`): Sets date to that weekday & time to '00:00:00.000000'.
        """
        unit_len: cython.Py_ssize_t = str_len(unit)

        # Unit: 's', 'm', 'h', 'D', 'W', 'M', 'Q', 'Y'
        if unit_len == 1:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . second
            if unit_ch == "s":
                return self.to_time(-1, -1, -1, 0)
            # . minute
            if unit_ch == "m":
                return self.to_time(-1, -1, 0, 0)
            # . hour
            if unit_ch == "h":
                return self.to_time(-1, 0, 0, 0)
            # . day
            if unit_ch == "D":
                return self.to_time(0, 0, 0, 0)
            # . week
            if unit_ch == "W":
                # fmt: off
                return self.add(
                    0, 0, 0, 0,
                    -self._prop_weekday(),
                    -datetime.datetime_hour(self),
                    -datetime.datetime_minute(self),
                    -datetime.datetime_second(self), 0,
                    -datetime.datetime_microsecond(self),
                )
                # fmt: on
            # . month
            if unit_ch == "M":
                return self.to_datetime(-1, -1, 1, 0, 0, 0, 0)
            # . quarter
            if unit_ch == "Q":
                mm: cython.int = datetime.datetime_month(self)
                mm = utils.quarter_of_month(mm) * 3 - 2
                return self.to_datetime(-1, mm, 1, 0, 0, 0, 0)
            # . year
            if unit_ch == "Y":
                return self.to_datetime(-1, 1, 1, 0, 0, 0, 0)

        # Unit: 'ms', 'us', 'ns'
        elif unit_len == 2 and str_read(unit, 1) == "s":
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . millisecond
            if unit_ch == "m":
                # fmt: off
                return self.to_time(
                    -1, -1, -1,
                    datetime.datetime_microsecond(self) // 1000 * 1000,
                )
                # fmt: on
            # . microsecond / nanosecond
            if unit_ch in ("u", "n"):
                return self

        # Unit: 'min' for pandas compatibility
        elif unit_len == 3 and unit == "min":
            return self.to_time(-1, -1, 0, 0)

        # Month name
        val: cython.int = _parse_month(unit, False)
        if val != -1:
            return self.to_datetime(-1, val, 1, 0, 0, 0, 0)

        # Weekday name
        val: cython.int = _parse_weekday(unit, False)
        if val != -1:
            # fmt: off
            return self.add(
                0, 0, 0, 0,
                val - self._prop_weekday(),
                -datetime.datetime_hour(self),
                -datetime.datetime_minute(self),
                -datetime.datetime_second(self), 0,
                -datetime.datetime_microsecond(self),
            )
            # fmt: on

        # Invalid
        raise errors.InvalidTimeUnitError(
            "invalid 'start of' datetime unit '%s'.\n"
            "Supports: ['Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us'] "
            "or Month/Weekday name." % unit
        )

    @cython.ccall
    def to_end_of(self, unit: str) -> _Pydt:
        """Adjust the datetime to the end of the specified datetime unit `<'Pydt'>`.

        :param unit `<'str'>`: The datetime unit.
            - `'Y'`: Sets date to the last day of the current year & time to '23:59:59.999999'.
            - `'Q'`: Sets date to the last day of the current quarter & time to '23:59:59.999999'.
            - `'M'`: Sets date to the last day of the current month & time to '23:59:59.999999'.
            - `'W'`: Sets date to the last day (Sunday) of the current week & time to '23:59:59.999999'.
            - `'D'`: Retains the original date & sets time to '23:59:59.999999'.
            - `'h'`: Retains the original date & sets time to 'XX:59:59.999999'.
            - `'m'`: Retains the original date & sets time to 'XX:XX:59.999999'.
            - `'s'`: Retains the original date & sets time to 'XX:XX:XX.999999'.
            - `'ms'`: Retains the original date & sets time to 'XX:XX:XX.XXX999'.
            - `'us'`: Return the instance itself.
            - `'Month name (e.g., `'Jan'`, `'February'`, `'三月'`): Sets date to last day of that month & time to '23:59:59.999999'.
            - `'Weekday name (e.g., `'Mon'`, `'Tuesday'`, `'星期三'`): Sets date to that weekday & time to '23:59:59.999999'.
        """
        unit_len: cython.Py_ssize_t = str_len(unit)

        # Unit: 's', 'm', 'h', 'D', 'W', 'M', 'Q', 'Y'
        if unit_len == 1:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . second
            if unit_ch == "s":
                return self.to_time(-1, -1, -1, 999999)
            # . minute
            if unit_ch == "m":
                return self.to_time(-1, -1, 59, 999999)
            # . hour
            if unit_ch == "h":
                return self.to_time(-1, 59, 59, 999999)
            # . day
            if unit_ch == "D":
                return self.to_time(23, 59, 59, 999999)
            # . week
            if unit_ch == "W":
                # fmt: off
                return self.add(
                    0, 0, 0, 0,
                    6 - self._prop_weekday(),
                    23 - datetime.datetime_hour(self),
                    59 - datetime.datetime_minute(self),
                    59 - datetime.datetime_second(self), 0,
                    999999 - datetime.datetime_microsecond(self),
                )
                # fmt: on
            # . month
            if unit_ch == "M":
                return self.to_datetime(-1, -1, 31, 23, 59, 59, 999999)
            # . quarter
            if unit_ch == "Q":
                mm: cython.int = datetime.datetime_month(self)
                mm = utils.quarter_of_month(mm) * 3
                return self.to_datetime(-1, mm, 31, 23, 59, 59, 999999)
            # . year
            if unit_ch == "Y":
                return self.to_datetime(-1, 12, 31, 23, 59, 59, 999999)

        # Unit: 'ms', 'us', 'ns'
        elif unit_len == 2 and str_read(unit, 1) == "s":
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . millisecond
            if unit_ch == "m":
                # fmt: off
                return self.to_time(
                    -1, -1, -1,
                    datetime.datetime_microsecond(self) // 1000 * 1000 + 999,
                )
                # fmt: on
            # . microsecond / nanosecond
            if unit_ch in ("u", "n"):
                return self

        # Unit: 'min' for pandas compatibility
        elif unit_len == 3 and unit == "min":
            return self.to_time(-1, -1, 59, 999999)

        # Month name
        val: cython.int = _parse_month(unit, False)
        if val != -1:
            return self.to_datetime(-1, val, 31, 23, 59, 59, 999999)

        # Weekday name
        val: cython.int = _parse_weekday(unit, False)
        if val != -1:
            # fmt: off
            return self.add(
                0, 0, 0, 0,
                val - self._prop_weekday(),
                23 - datetime.datetime_hour(self),
                59 - datetime.datetime_minute(self),
                59 - datetime.datetime_second(self), 0,
                999999 - datetime.datetime_microsecond(self),
            )
            # fmt: on

        # Invalid
        raise errors.InvalidTimeUnitError(
            "invalid 'end of' datetime unit '%s'.\n"
            "Supports: ['Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us'] "
            "or Month/Weekday name." % unit
        )

    # . round / ceil / floor
    @cython.ccall
    def round(self, unit: str) -> _Pydt:
        """Perform round operation to the specified datetime unit `<'Pydt'>`.

        #### Similar to `pandas.Timestamp.round()`.

        :param unit `<'str'>`: The datetime unit to round to.
            Supported datetime units: 'D', 'h', 'm', 's', 'ms', 'us', 'ns'.
        """
        # Parse unit factor
        f: cython.longlong = _parse_unit(unit, True)
        if f == 1:
            return self  # exit: no change

        # Round to unit
        us: cython.longlong = utils.dt_to_us(self, False)
        us_r: cython.longlong = utils.math_round_div(us, f) * f
        if us_r == us:
            return self  # exit: same value

        # New instance
        return _pydt_fr_dt(utils.dt_fr_us(us_r, datetime.datetime_tzinfo(self)))

    @cython.ccall
    def ceil(self, unit: str) -> _Pydt:
        """Perform ceil operation to the specified datetime unit `<'Pydt'>`.

        #### Similar to `pandas.Timestamp.ceil()`.

        :param unit `<'str'>`: The datetime unit to round to.
            Supported datetime units: 'D', 'h', 'm', 's', 'ms', 'us', 'ns'.
        """
        # Parse unit factor
        f: cython.longlong = _parse_unit(unit, True)
        if f == 1:
            return self  # exit: no change

        # Ceil to unit
        us: cython.longlong = utils.dt_to_us(self, False)
        us_c: cython.longlong = utils.math_ceil_div(us, f) * f
        if us_c == us:
            return self  # exit: same value

        # New instance
        return _pydt_fr_dt(utils.dt_fr_us(us_c, datetime.datetime_tzinfo(self)))

    @cython.ccall
    def floor(self, unit: str) -> _Pydt:
        """Perform floor operation to the specified datetime unit `<'Pydt'>`.

        #### Similar to `pandas.Timestamp.floor()`.

        :param unit `<'str'>`: The datetime unit to round to.
            Supported datetime units: 'D', 'h', 'm', 's', 'ms', 'us', 'ns'.
        """
        # Parse unit factor
        f: cython.longlong = _parse_unit(unit, True)
        if f == 1:
            return self  # exit: no change

        # Floor to unit
        us: cython.longlong = utils.dt_to_us(self, False)
        us_f: cython.longlong = utils.math_floor_div(us, f) * f
        if us_f == us:
            return self  # exit: same value

        # New instance
        return _pydt_fr_dt(utils.dt_fr_us(us_f, datetime.datetime_tzinfo(self)))

    # Calendar -----------------------------------------------------------------------------
    # . iso
    @cython.ccall
    def isocalendar(self) -> dict[str, int]:
        """Return the ISO calendar `<'dict'>`.

        ### Example:
        >>> dt.isocalendar()
        >>> {'year': 2024, 'week': 40, 'weekday': 2}
        """
        _iso = utils.ymd_isocalendar(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
        )
        return {"year": _iso.year, "week": _iso.week, "weekday": _iso.weekday}

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def isoyear(self) -> cython.int:
        """Return the ISO calendar year (1-10000) `<'int'>`."""
        return utils.ymd_isoyear(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def isoweek(self) -> cython.int:
        """Return the ISO calendar week number (1-53) `<'int'>`."""
        return utils.ymd_isoweek(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def isoweekday(self) -> cython.int:
        """Return the ISO calendar weekday (1=Mon...7=Sun) `<'int'>`."""
        return utils.ymd_isoweekday(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
        )

    # . year
    @property
    def year(self) -> int:
        """The year (1-9999) `<'int'>`."""
        return self._prop_year()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_year(self) -> cython.int:
        """(internal) Return the year (1-9999) `<'int'>`."""
        return datetime.datetime_year(self)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_leap_year(self) -> cython.bint:
        """Determine if the instance is in a leap year `<'bool'>`."""
        return utils.is_leap_year(datetime.datetime_year(self))

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_long_year(self) -> cython.bint:
        """Determine if the instance is in a long year `<'bool'>`.

        - Long year: maximum ISO week number is 53.
        """
        return utils.is_long_year(datetime.datetime_year(self))

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def leap_bt_year(self, year: cython.int) -> cython.int:
        """Compute the number of leap years between the
        instance and the passed-in 'year' `<'int'>`.
        """
        return utils.leap_bt_year(datetime.datetime_year(self), year)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_in_year(self) -> cython.int:
        """Return the maximum number of days (365, 366)
        in the instance's year `<'int'>`.
        """
        return utils.days_in_year(datetime.datetime_year(self))

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_bf_year(self) -> cython.int:
        """Return the number of days from January 1, 1st AD,
        to the start of the instance's year `<'int'>`.
        """
        return utils.days_bf_year(datetime.datetime_year(self))

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_of_year(self) -> cython.int:
        """Return the number of days since the start
        of the instance's year `<'int'>`.
        """
        return utils.days_of_year(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_year(self, year: cython.int) -> cython.bint:
        """Determine if the instance's year matches
        the passed-in 'year' `<'bool'>.
        """
        return datetime.datetime_year(self) == year

    # . quarter
    @property
    def quarter(self) -> int:
        """The quarter (1-4) `<'int'>`."""
        return self._prop_quarter()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_quarter(self) -> cython.int:
        """(internal) Return the quarter (1-4) `<'int'>`."""
        return utils.quarter_of_month(datetime.datetime_month(self))

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_in_quarter(self) -> cython.int:
        """Return the maximum number of days (90-92)
        in the instance's quarter `<'int'>`.
        """
        return utils.days_in_quarter(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_bf_quarter(self) -> cython.int:
        """Return the number of days from the start of the instance's
        year to the start of the instance's quarter `<'int'>`.
        """
        return utils.days_bf_quarter(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_of_quarter(self) -> cython.int:
        """Return the number of days since the start
        of the instance's quarter `<'int'>`.
        """
        return utils.days_of_quarter(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_quarter(self, quarter: cython.int) -> cython.bint:
        """Determine if the instance's quarter matches
        the passed-in 'quarter' `<'bool'>`.
        """
        return utils.quarter_of_month(datetime.datetime_month(self)) == quarter

    # . month
    @property
    def month(self) -> int:
        """The month (1-12) `<'int'>`."""
        return self._prop_month()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_month(self) -> cython.int:
        """(internal) Return the month (1-12) `<'int'>`."""
        return datetime.datetime_month(self)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_in_month(self) -> cython.int:
        """Return the maximum number of days (28-31)
        in the instance's month `<'int'>`.
        """
        return utils.days_in_month(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_bf_month(self) -> cython.int:
        """Return the number of days from the start of the instance's
        year to the start of the instance's month `<'int'>`."""
        return utils.days_bf_month(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def days_of_month(self) -> cython.int:
        """Return the number of days (1-31) since the start
        of the instance's month `<'int'>`.

        ### Equivalent to:
        >>> dt.day
        """
        return datetime.datetime_day(self)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_month(self, month: int | str) -> cython.bint:
        """Determine if the instance's month matches
        the passed-in 'month' `<'bool'>`.

        :param month `<'int/str'>`: The month value.
            - `<'int'>` Month number (1=Jan...12=Dec).
            - `<'str'>` Month name (case-insensitive), e.g., 'Jan', 'februar', '三月'.
        """
        return _parse_month(month, True) == datetime.datetime_month(self)

    @cython.ccall
    def month_name(self, locale: str | None = None) -> str:
        """Return the month name with specified locale `<'str'>`.

        :param locale `<'str/None'>`: The locale to use for month name, defaults to `None`.
            - Locale determining the language in which to return the month
              name. Default (`None`) is English locale ('en_US.utf8').
            - Use the command locale -a on your terminal on Unix systems to
              find your locale language code.
        """
        if locale is None:
            locale = "en_US"
        try:
            return _format_date(self, format="MMMM", locale=locale)
        except Exception as err:
            raise errors.InvalidArgumentError(err) from err

    # . weekday
    @property
    def weekday(self) -> int:
        """The weekday (0=Mon...6=Sun) `<'int'>`."""
        return self._prop_weekday()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_weekday(self) -> cython.int:
        """(internal) Return the weekday (0=Mon...6=Sun) `<'int'>`."""
        return utils.ymd_weekday(
            datetime.datetime_year(self),
            datetime.datetime_month(self),
            datetime.datetime_day(self),
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_weekday(self, weekday: int | str) -> cython.bint:
        """Determine if the instance's weekday matches
        the passed-in 'weekday' `<'bool'>`.

        :param weekday `<'int/str'>`: The weekday value.
            - `<'int'>` Weekday number (0=Mon...6=Sun).
            - `<'str'>` Weekday name (case-insensitive), e.g., 'Mon', 'dienstag', '星期三'.
        """
        return _parse_weekday(weekday, True) == self._prop_weekday()

    # . day
    @property
    def day(self) -> int:
        """The day (1-31) `<'int'>`."""
        return self._prop_day()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_day(self) -> cython.int:
        """(internal) Return the day (1-31) `<'int'>`."""
        return datetime.datetime_day(self)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_day(self, day: cython.int) -> cython.bint:
        """Determines if the instance's day matches
        with the passed-in 'day' `<'bool'>`.
        """
        return datetime.datetime_day(self) == day

    @cython.ccall
    def day_name(self, locale: str | None = None) -> str:
        """Return the weekday name with specified locale `<'str'>`.

        :param locale `<'str/None'>`: The locale to use for weekday name, defaults to `None`.
            - Locale determining the language in which to return the weekday
              name. Default (`None`) is English locale ('en_US.utf8').
            - Use the command locale -a on your terminal on Unix systems to
              find your locale language code.
        """
        if locale is None:
            locale = "en_US"
        try:
            return _format_date(self, format="EEEE", locale=locale)
        except Exception as err:
            raise errors.InvalidArgumentError(err) from err

    # . time
    @property
    def hour(self) -> int:
        """The hour (0-23) `<'int'>`."""
        return self._prop_hour()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_hour(self) -> cython.int:
        """(internal) Return the hour (0-23) `<'int'>`."""
        return datetime.datetime_hour(self)

    @property
    def minute(self) -> int:
        """The minute (0-59) `<'int'>`."""
        return self._prop_minute()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_minute(self) -> cython.int:
        """(internal) Return the minute (0-59) `<'int'>`."""
        return datetime.datetime_minute(self)

    @property
    def second(self) -> int:
        """The second (0-59) `<'int'>`."""
        return self._prop_second()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_second(self) -> cython.int:
        """(internal) Return the second (0-59) `<'int'>`."""
        return datetime.datetime_second(self)

    @property
    def millisecond(self) -> int:
        """The millisecond (0-999) `<'int'>`."""
        return self._prop_millisecond()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_millisecond(self) -> cython.int:
        """(internal) Return the millisecond (0-999) `<'int'>`."""
        return datetime.datetime_microsecond(self) // 1000

    @property
    def microsecond(self) -> int:
        """The microsecond (0-999999) `<'int'>`."""
        return self._prop_microsecond()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_microsecond(self) -> cython.int:
        """(internal) Return the microsecond (0-999999) `<'int'>`."""
        return datetime.datetime_microsecond(self)

    # . date&time
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_first_of(self, unit: str) -> cython.bint:
        """Determine if the instance is on the frist day of
        the specified datetime unit `<'bool'>`.

        :param unit `<'str'>`: The datetime unit.
            - `'Y'`: Is on the first day of the current year.
            - `'Q'`: Is on the first day of the current quarter.
            - `'M'`: Is on the first day of the current month.
            - `'W'`: Is on the first day (Monday) of the current week.
            - Month name (e.g., `'Jan'`, `'February'`, `'三月'`): Is the first day of that month.
        """
        unit_len: cython.Py_ssize_t = str_len(unit)

        # Unit: 'W', 'M', 'Q', 'Y'
        if unit_len == 1:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . weekday
            if unit_ch == "W":
                return self._prop_weekday() == 0
            # . month
            if unit_ch == "M":
                return self._is_first_of_month()
            # . quarter
            if unit_ch == "Q":
                return self._is_frist_of_quarter()
            # . year
            if unit_ch == "Y":
                return self._is_first_of_year()

        # Month name
        val: cython.int = _parse_month(unit, False)
        if val != -1:
            return (
                datetime.datetime_month(self) == val
                and datetime.datetime_day(self) == 1
            )

        # Invalid
        raise errors.InvalidTimeUnitError(
            "invalid 'first of' datetime unit '%s'.\n"
            "Supports: ['Y', 'Q', 'M', 'W'] or Month name." % unit
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_last_of(self, unit: str) -> cython.bint:
        """Determine if the instance is on the last day of
        the specified datetime unit `<'bool'>`.

        :param unit `<'str'>`: The datetime unit.
            - `'Y'`: Is on the last day of the current year.
            - `'Q'`: Is on the last day of the current quarter.
            - `'M'`: Is on the last day of the current month.
            - `'W'`: Is on the last day (Sunday) of the current week.
            - Month name (e.g., `'Jan'`, `'February'`, `'三月'`): Is the last day of that month.
        """
        unit_len: cython.Py_ssize_t = str_len(unit)

        # Unit: 'W', 'M', 'Q', 'Y'
        if unit_len == 1:
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . weekday
            if unit_ch == "W":
                return self._prop_weekday() == 6
            # . month
            if unit_ch == "M":
                return self._is_last_of_month()
            # . quarter
            if unit_ch == "Q":
                return self._is_last_of_quarter()
            # . year
            if unit_ch == "Y":
                return self._is_last_of_year()

        # Month name
        val: cython.int = _parse_month(unit, False)
        if val != -1:
            if datetime.datetime_month(self) != val:
                return False
            dd: cython.int = datetime.datetime_day(self)
            return dd == utils.days_in_month(datetime.datetime_year(self), val)

        # Invalid
        raise errors.InvalidTimeUnitError(
            "invalid 'last of' datetime unit '%s'.\n"
            "Supports: ['Y', 'Q', 'M', 'W'] or Month name." % unit
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_start_of(self, unit: str) -> cython.bint:
        """Determine if the instance is at the start of
        the specified datetime unit `<'bool'>`.

        :param unit `<'str'>`: The datetime unit.
            - `'Y'`: Is on the first day of the current year at time '00:00:00.000000'.
            - `'Q'`: Is on the first day of the current quarter at time '00:00:00.000000'.
            - `'M'`: Is on the first day of the current month at time '00:00:00.000000'.
            - `'W'`: Is on the first day (Monday) of the current week at time '00:00:00.000000'.
            - `'D'`: Is at time '00:00:00.000000'.
            - `'h'`: Is at time 'XX:00:00.000000'.
            - `'m'`: Is at time 'XX:XX:00.000000'.
            - `'s'`: Is at time 'XX:XX:XX.000000'.
            - `'ms'`: Is at time 'XX:XX:XX.XXX000'.
            - `'us'`: Always returns `True`.
            - Month name (e.g., `'Jan'`, `'February'`, `'三月'`): Is on the first day of that month at time '00:00:00.000000'.
            - Weekday name (e.g., `'Mon'`, `'Tuesday'`, `'星期三'`): Is on that weekday at time '00:00:00.000000'.
        """
        unit_len: cython.Py_ssize_t = str_len(unit)

        # Unit: 's', 'm', 'h', 'D', 'W', 'M', 'Q', 'Y'
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
            # Start of time
            if not self._is_start_of_time():
                return False
            # . day
            if unit_ch == "D":
                return True
            # . week
            if unit_ch == "W":
                return self._prop_weekday() == 0
            # . month
            if unit_ch == "M":
                return self._is_first_of_month()
            # . quarter
            if unit_ch == "Q":
                return self._is_frist_of_quarter()
            # . year
            if unit_ch == "Y":
                return self._is_first_of_year()

        # Unit: 'ms', 'us', 'ns'
        elif unit_len == 2 and str_read(unit, 1) == "s":
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . millisecond
            if unit_ch == "m":
                return datetime.datetime_microsecond(self) % 1000 == 0
            # . microsecond / nanosecond
            if unit_ch in ("u", "n"):
                return True

        # Unit: 'min' for pandas compatibility
        elif unit_len == 3 and unit == "min":
            return (
                datetime.datetime_second(self) == 0
                and datetime.datetime_microsecond(self) == 0
            )

        # Start of time
        if not self._is_start_of_time():
            return False

        # Month name
        val: cython.int = _parse_month(unit, False)
        if val != -1:
            return (
                datetime.datetime_month(self) == val
                and datetime.datetime_day(self) == 1
            )

        # Weekday name
        val: cython.int = _parse_weekday(unit, False)
        if val != -1:
            return self._prop_weekday() == val

        # Invalid
        raise errors.InvalidTimeUnitError(
            "invalid 'start of' datetime unit '%s'.\n"
            "Supports: ['Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us'] "
            "or Month/Weekday name." % unit
        )

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_end_of(self, unit: str) -> cython.bint:
        """Determine if the instance is at the end of the specified datetime unit `<'bool'>`.

        :param unit `<'str'>`: The datetime unit.
            - `'Y'`: Is on the last day of the current year at time '23:59:59.999999'.
            - `'Q'`: Is on the last day of the current quarter at time '23:59:59.999999'.
            - `'M'`: Is on the last day of the current month at time '23:59:59.999999'.
            - `'W'`: Is on the last day (Sunday) of the current week at time '23:59:59.999999'.
            - `'D'`: Is at time '23:59:59.999999'.
            - `'h'`: Is at time 'XX:59:59.999999'.
            - `'m'`: Is at time 'XX:XX:59.999999'.
            - `'s'`: Is at time 'XX:XX:XX.999999'.
            - `'ms'`: Is at time 'XX:XX:XX.XXX999'.
            - `'us'`: Always returns `True`.
            - Month name (e.g., `'Jan'`, `'February'`, `'三月'`): Is on the last day of that month at time '23:59:59.999999'.
            - Weekday name (e.g., `'Mon'`, `'Tuesday'`, `'星期三'`): Is on that weekday at time '23:59:59.999999'.
        """
        unit_len: cython.Py_ssize_t = str_len(unit)

        # Unit: 's', 'm', 'h', 'D', 'W', 'M', 'Q', 'Y'
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
            # End of time
            if not self._is_end_of_time():
                return False
            # . day
            if unit_ch == "D":
                return True
            # . week
            if unit_ch == "W":
                return self._prop_weekday() == 6
            # . month
            if unit_ch == "M":
                return self._is_last_of_month()
            # . quarter
            if unit_ch == "Q":
                return self._is_last_of_quarter()
            # . year
            if unit_ch == "Y":
                return self._is_last_of_year()

        # Unit: 'ms', 'us', 'ns'
        elif unit_len == 2 and str_read(unit, 1) == "s":
            unit_ch: cython.Py_UCS4 = str_read(unit, 0)
            # . millisecond
            if unit_ch == "m":
                return datetime.datetime_microsecond(self) % 1000 == 999
            # . microsecond / nanosecond
            if unit_ch in ("u", "n"):
                return True

        # Unit: 'min' for pandas compatibility
        elif unit_len == 3 and unit == "min":
            return (
                datetime.datetime_second(self) == 59
                and datetime.datetime_microsecond(self) == 999_999
            )

        # End of time
        if not self._is_end_of_time():
            return False

        # Month name
        val: cython.int = _parse_month(unit, False)
        if val != -1:
            return (
                datetime.datetime_month(self) == val
                and datetime.datetime_day(self) == self.days_in_month()
            )

        # Weekday name
        val: cython.int = _parse_weekday(unit, False)
        if val != -1:
            return self._prop_weekday() == val

        # Invalid
        raise errors.InvalidTimeUnitError(
            "invalid 'end of' datetime unit '%s'.\n"
            "Supports: ['Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us'] "
            "or Month/Weekday name." % unit
        )

    # . utils
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_first_of_year(self) -> cython.bint:
        """(internal) Determine if the instance is on the first day of the year `<'bool'>`.

        - First day of the year: 'XXXX-01-01'
        """
        if datetime.datetime_day(self) != 1:
            return False
        return datetime.datetime_month(self) == 1

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_last_of_year(self) -> cython.bint:
        """(internal) Determine if the instance is on the last day of the year `<'bool'>`.

        - Last day of the year: 'XXXX-12-31'
        """
        if datetime.datetime_day(self) != 31:
            return False
        return datetime.datetime_month(self) == 12

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_frist_of_quarter(self) -> cython.bint:
        """(internal) Determine if the instance is on the first day of the quarter `<'bool'>`.

        - First day of the quarter: 'XXXX-MM-01'
            - MM: [1, 4, 7, 10]
        """
        if datetime.datetime_day(self) != 1:
            return False
        return datetime.datetime_month(self) in (1, 4, 7, 10)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_last_of_quarter(self) -> cython.bint:
        """(internal) Determine if the instance is on the last day of the quarter `<'bool'>`.

        - Last day of the quarter: 'XXXX-MM-DD'
            - MM: [3, 6, 9, 12]
            - DD: [30, 31]
        """
        dd: cython.int = datetime.datetime_day(self)
        if dd < 30:
            return False
        mm: cython.int = datetime.datetime_month(self)
        if mm not in (3, 6, 9, 12):
            return False
        return dd == utils.days_in_month(datetime.datetime_year(self), mm)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_first_of_month(self) -> cython.bint:
        """(internal) Determine if the instance is on the first day of the month `<'bool'>`.

        - First day of the month: 'XXXX-XX-01'
        """
        return datetime.datetime_day(self) == 1

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_last_of_month(self) -> cython.bint:
        """(internal) Check if the instance is on the last day of the month `<'bool'>`.

        - Last day of the month: 'XXXX-XX-DD'
            - DD: [28, 29, 30, 31]
        """
        dd: cython.int = datetime.datetime_day(self)
        if dd < 28:
            return False
        return dd == utils.days_in_month(
            datetime.datetime_year(self), datetime.datetime_month(self)
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_start_of_time(self) -> cython.bint:
        """(internal) Determine if the instance is at the start of the time `<'bool'>`.

        - Start of the time: '00:00:00.000000'
        """
        return (
            datetime.datetime_hour(self) == 0
            and datetime.datetime_minute(self) == 0
            and datetime.datetime_second(self) == 0
            and datetime.datetime_microsecond(self) == 0
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _is_end_of_time(self) -> cython.bint:
        """(internal) Check if the instance is at the end of the time `<'bool'>`.

        - End of the time: '23:59:59.999999'
        """
        return (
            datetime.datetime_hour(self) == 23
            and datetime.datetime_minute(self) == 59
            and datetime.datetime_second(self) == 59
            and datetime.datetime_microsecond(self) == 999_999
        )

    # Timezone -----------------------------------------------------------------------------
    @property
    def tz_available(self) -> set[str]:
        """The available timezone names `<'set[str]'>`.

        ### Equivalent to:
        >>> zoneinfo.available_timezones()
        """
        return _available_timezones()

    @property
    def tz(self) -> object:
        """The timezone information `<'tzinfo/None'>`."""
        return self._prop_tzinfo()

    @property
    def tzinfo(self) -> object:
        """The timezone information `<'tzinfo/None'>`."""
        return self._prop_tzinfo()

    @cython.cfunc
    @cython.inline(True)
    def _prop_tzinfo(self) -> object:
        """(internal) Return the timezone information `<'tzinfo/None'>`."""
        return datetime.datetime_tzinfo(self)

    @property
    def fold(self) -> int:
        """The fold value (0 or 1) for ambiguous times `<'int'>`.

        Use to disambiguates local times during
        daylight saving time (DST) transitions.
        """
        return self._prop_fold()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _prop_fold(self) -> cython.int:
        """(internal) Return the fold value (0 or 1) for ambiguous times `<'int'>`.

        Use to disambiguates local times during
        daylight saving time (DST) transitions.
        """
        return datetime.datetime_fold(self)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_local(self) -> cython.bint:
        """Determine if the instance is in the local timezone `<'bool'>`.

        #### Timezone-naive instance always returns `False`.
        """
        return datetime.datetime_tzinfo(self) is utils.tz_local(None)

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_utc(self) -> cython.bint:
        """Determine if the instance is in the UTC timezone `<'bool'>`.

        #### Timezone-naive instance always returns `False`.
        """
        return datetime.datetime_tzinfo(self) is utils.UTC

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_dst(self) -> cython.bint:
        """Determine if the instance is in Dayligh Saving Time (DST) `<'bool'>.

        #### Timezone-naive datetime always returns `False`.
        """
        dst = utils.dt_dst(self)
        return False if dst is None else bool(dst)

    @cython.ccall
    def tzname(self) -> str:
        """Return the timezone name `<'str/None'>`.

        #### Timezone-naive instance always returns `None`.
        """
        return utils.dt_tzname(self)

    @cython.ccall
    def utcoffset(self) -> datetime.timedelta:
        """Return the UTC offset `<'datetime.timedelta/None'>`.

        #### Timezone-naive instance always returns `None`.

        The offset is positive for timezones east of
        UTC and negative for timezones west of UTC.
        """
        return utils.dt_utcoffset(self)

    @cython.ccall
    def utcoffset_seconds(self) -> object:
        """Return the UTC offset in total seconds `<'int/None'>`.

        #### Timezone-naive instance always returns `None`.

        The offset is positive for timezones east of
        UTC and negative for timezones west of UTC.
        """
        tz = datetime.datetime_tzinfo(self)
        if tz is None:
            return None
        ss: cython.int = utils.tz_utcoffset_seconds(tz, self)
        return None if ss == -100_000 else ss

    @cython.ccall
    def dst(self) -> datetime.timedelta:
        """Return the Daylight Saving Time (DST) offset `<'datetime.timedelta/None'>`.

        #### Timezone-naive instance always returns `None`.

        This is purely informational, the DST offset has already
        been added to the UTC offset returned by 'utcoffset()'.
        """
        return utils.dt_dst(self)

    @cython.ccall
    def astimezone(self, tz: datetime.tzinfo | str | None = None) -> _Pydt:
        """Convert the instance to the target timezone
        (retaining the same point in UTC time). `<'Pydt'>`.

        - If the instance is timezone-aware, converts to the target timezone directly.
        - If the instance is timezone-naive, first localizes to the local timezone,
          then converts to the targer timezone.

        :param tz `<'tzinfo/str/None'>`: The target timezone to convert to, defaults to `None`.
            - `<'datetime.tzinfo'>` Subclass of datetime.tzinfo.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Convert to local timezone.
        """
        return self._from_dt(utils.dt_astimezone(self, utils.tz_parse(tz)))

    @cython.ccall
    def tz_localize(self, tz: datetime.tzinfo | str | None) -> _Pydt:
        """Localize timezone-naive instance to the specific timezone;
        or timezone-aware instance to timezone naive (without moving
        the time fields) `<'Pydt'>`.

        #### Similar to `pandas.Timestamp.tz_localize()`.

        :param tz `<'tzinfo/str/None'>`: The timezone to localize to, defaults to `None`.
            - `<'datetime.tzinfo'>` Subclass of datetime.tzinfo.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Localize to timezone-naive.
        """
        # Timezone-aware
        tz = utils.tz_parse(tz)
        mytz = datetime.datetime_tzinfo(self)
        if mytz is not None:
            if tz is not None:
                raise errors.InvalidTimezoneError(
                    "instance is already timezone-aware.\n"
                    "Use 'tz_convert()' or 'tz_switch()' method "
                    "to move to another timezone."
                )
            # . localize: aware => naive
            return self._from_dt(utils.dt_replace_tz(self, None))

        # Timezone-naive
        if tz is None:
            return self
        # . localize: naive => aware
        return self._from_dt(utils.dt_replace_tz(self, tz))

    @cython.ccall
    def tz_convert(self, tz: datetime.tzinfo | str | None) -> _Pydt:
        """Convert timezone-aware instance to another timezone `<'Pydt'>`.

        #### Similar to `pandas.Timestamp.tz_convert()`.

        :param tz `<'tzinfo/str/None'>`: The timezone to localize to, defaults to `None`.
            - `<'datetime.tzinfo'>` Subclass of datetime.tzinfo.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Convert to UTC timezone and localize to timezone-naive.
        """
        # Validate
        if datetime.datetime_tzinfo(self) is None:
            raise errors.InvalidTimezoneError(
                "instance is timezone-naive.\n"
                "Use 'tz_localize()' method instead to localize to a timezone.\n"
                "Use 'tz_switch()' method to convert to the timezone by "
                "providing a base timezone for the instance."
            )

        # Convert: aware => None
        tz = utils.tz_parse(tz)
        if tz is None:
            return _pydt_fr_dt(
                utils.dt_replace_tz(utils.dt_astimezone(self, utils.UTC), None)
            )

        # Convert: aware => aware
        return self._from_dt(utils.dt_astimezone(self, tz))

    @cython.ccall
    def tz_switch(
        self,
        targ_tz: datetime.tzinfo | str | None,
        base_tz: datetime.tzinfo | str | None = None,
        naive: cython.bint = False,
    ) -> _Pydt:
        """Switch (convert) the instance from base timezone to the target timezone `<'Pydt'>`.

        This method extends the functionality of `astimezone()` by allowing
        user to specify a base timezone for timezone-naive instances before
        converting to the target timezone.

        - If the instance is timezone-aware, the 'base_tz' argument is `ignored`,
          and this method behaves identically to `astimezone()`, converting the
          instance to the target timezone.
        - If the instance is timezone-naive, it first localizes the instance
          to the `base_tz`, and then converts to the target timezone.

        :param targ_tz `<'tzinfo/str/None'>`: The target timezone to convert to.
        :param base_tz `<'tzinfo/str/None'>`: The base timezone for timezone-naive instance, defaults to `None`.
        :param naive `<'bool'>`: If 'True', returns timezone-naive instance after conversion, defaults to `False`.
        """
        # Timezone-aware
        targ_tz = utils.tz_parse(targ_tz)
        mytz = datetime.datetime_tzinfo(self)
        if mytz is not None:
            # . target timezone is None
            if targ_tz is None:
                dt = utils.dt_replace_tz(self, None)
            # . target timezone is mytz
            elif targ_tz is mytz:
                if naive:
                    dt = utils.dt_replace_tz(self, None)
                else:
                    return self  # exit
            # . mytz => target timezone
            else:
                dt = utils.dt_astimezone(self, targ_tz)
                if naive:
                    dt = utils.dt_replace_tz(dt, None)
            # New instance
            return self._from_dt(dt)

        # Timezone-naive
        # . target timezone is None
        if targ_tz is None:
            return self  # exit
        # . base is target timezone
        base_tz = utils.tz_parse(base_tz)
        if base_tz is None:
            raise errors.InvalidTimezoneError(
                "instance is timezone-naive.\n"
                "Cannot switch timezone-naive instance to the "
                "target timezone without providing a 'base_tz'."
            )
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
        # New instance
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
        :param quarters `<'int'>`: Relative delta of quarters (3 months), defaults to `0`.
        :param months `<'int'>`: Relative delta of months, defaults to `0`.
        :param weeks `<'int'>`: Relative delta of weeks (7 days), defaults to `0`.
        :param days `<'int'>`: Relative delta of days, defaults to `0`.
        :param hours `<'int'>`: Relative delta of hours, defaults to `0`.
        :param minutes `<'int'>`: Relative delta of minutes, defaults to `0`.
        :param seconds `<'int'>`: Relative delta of seconds, defaults to `0`.
        :param milliseconds `<'int'>`: Relative delta of milliseconds, defaults to `0`.
        :param microseconds `<'int'>`: Relative delta of microseconds, defaults to `0`.
        """
        # Compute delta
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
            hmsf_eq: cython.bint = us == my_us
        else:
            hmsf_eq: cython.bint = True
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
            if hmsf_eq:
                hmsf_eq = ss == my_ss
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
            if hmsf_eq:
                hmsf_eq = mi == my_mi
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
            if hmsf_eq:
                hmsf_eq = hh == my_hh
        # . days
        days += weeks * 7

        # Add delta
        if days != 0:
            _ymd = utils.ymd_fr_ordinal(utils.ymd_to_ordinal(yy, mm, dd) + days)
            yy, mm, dd = _ymd.year, _ymd.month, _ymd.day
        elif ymd_eq:
            if hmsf_eq:
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
        :param quarters `<'int'>`: Relative delta of quarters (3 months), defaults to `0`.
        :param months `<'int'>`: Relative delta of months, defaults to `0`.
        :param weeks `<'int'>`: Relative delta of weeks (7 days), defaults to `0`.
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
    def diff(
        self,
        dtobj: object,
        unit: str,
        absolute: cython.bint = False,
        inclusive: str = "both",
    ) -> cython.longlong:
        """Calculate the difference between the instance and another datetime-like object `<'int'>`.

        The difference is computed in the specified datetime 'unit'
        and adjusted based on the 'inclusive' argument to determine
        the inclusivity of the start and end times.

        :param dtobj `<'object'>`: Datetime-like object:
            - `<'str'>` A datetime string containing datetime information.
            - `<'datetime.datetime'>` An instance of `datetime.datetime`.
            - `<'datetime.date'>` An instance of `datetime.date` (time fields set to 0).
            - `<'int/float'>` Numeric value treated as total seconds since Unix Epoch.
            - `<'np.datetime64'>` Resolution above microseconds ('us') will be discarded.
            - `<'None'>` Current datetime with the same timezone of the instance.

        :param unit `<'str'>`: The datetime unit for calculating the difference.
            Supports: 'Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us'.

        :param absolute `<'bool'>`: If 'True', compute the absolute difference, defaults to `False`.

        :param inclusive `<'str'>`: Specifies the inclusivity of the start and end times, defaults to `'both'`.
            - `'one'`: Include either the start or end time.
            - `'both'`: Include both the start and end times.
            - `'neither'`: Exclude both the start and end times.
        """
        # Parse & check timezone parity
        my_tz = datetime.datetime_tzinfo(self)
        dt: datetime.datetime
        if dtobj is None:
            dt = utils.dt_now(my_tz)
        else:
            dt = _pydt_fr_dtobj(dtobj)
            dt_tz = dt.tzinfo
            if my_tz is not dt_tz and (my_tz is None or dt_tz is None):
                _raise_incomparable_error(self, dt, "calculate difference")

        # Handle inclusive
        if inclusive == "both":
            incl_off: cython.int = 1
        elif inclusive == "one":
            incl_off: cython.int = 0
        elif inclusive == "neither":
            incl_off: cython.int = -1
        else:
            raise errors.InvalidArgumentError(
                "invalid input '%s' for inclusive.\n"
                "Supports: ['one', 'both', 'neither']." % inclusive
            )

        # Compute difference
        utc: cython.bint = my_tz is not None
        if unit == "W":
            my_val: cython.longlong = utils.dt_as_epoch_iso_W(self, 1, utc)
            dt_val: cython.longlong = utils.dt_as_epoch_iso_W(dt, 1, utc)
        else:
            my_val: cython.longlong = utils.dt_as_epoch(self, unit, utc)
            dt_val: cython.longlong = utils.dt_as_epoch(dt, unit, utc)
        delta: cython.longlong = my_val - dt_val
        # . absolute = True
        if absolute:
            delta = (-delta if delta < 0 else delta) + incl_off
        # . absolute = False | adj offset
        elif incl_off != 0:
            delta = delta - incl_off if delta < 0 else delta + incl_off
        return delta

    @cython.cfunc
    @cython.inline(True)
    def _add_timedelta(
        self,
        days: cython.int,
        seconds: cython.int,
        microseconds: cython.int,
    ) -> _Pydt:
        """(internal) Add timedelta to the instance `<'Pydt'>`.

        :param days `<'int'>`: Relative days of the timedelta.
        :param seconds `<'int'>`: Relative seconds of the timedelta.
        :param microseconds `<'int'>`: Relative microseconds of the timedelta.
        """
        dd_: cython.longlong = days
        ss_: cython.longlong = seconds
        us: cython.longlong = (dd_ * 86_400 + ss_) * 1_000_000 + microseconds
        if us == 0:
            return self  # no change

        # Add delta
        us += utils.dt_to_us(self, False) + utils.EPOCH_MICROSECOND
        _ymd = utils.ymd_fr_ordinal(us // utils.US_DAY)
        _hmsf = utils.hmsf_fr_us(us)

        # Create Pydt
        # fmt: off
        return pydt_new(
            _ymd.year, _ymd.month, _ymd.day, 
            _hmsf.hour, _hmsf.minute, _hmsf.second, _hmsf.microsecond,
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
            o = _pydt_fr_dtobj(o)
        elif utils.is_dt64(o):
            o = utils.dt64_to_dt(o)
        else:
            return NotImplemented

        # Check timezone parity
        m_tz = datetime.datetime_tzinfo(self)
        o_tz = datetime.datetime_tzinfo(o)
        if m_tz is not o_tz and (m_tz is None or o_tz is None):
            _raise_incomparable_error(self, o, "perform subtraction")

        # Compute delta
        utc: cython.bint = m_tz is not None
        m_us: cython.longlong = utils.dt_to_us(self, utc)
        o_us: cython.longlong = utils.dt_to_us(o, utc)

        # Create delta
        return utils.td_fr_us(m_us - o_us)

    # Comparison ---------------------------------------------------------------------------
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_past(self) -> cython.bint:
        """Determine if the instance is in the past `<'bool'>`."""
        return self < utils.dt_now(datetime.datetime_tzinfo(self))

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def is_future(self) -> cython.bint:
        """Determine if the instance is in the future `<'bool'>`."""
        return self > utils.dt_now(datetime.datetime_tzinfo(self))

    def closest(self, *dtobjs: object) -> _Pydt:
        """Find the datetime closest in time to the instance `<'Pydt/None'>`.

        :param dtobjs `<'*object'>`: Multiple datetime-like objects to compare with the instance.
            - `<'str'>` A datetime string containing datetime information.
            - `<'datetime.datetime'>` An instance of `datetime.datetime`.
            - `<'datetime.date'>` An instance of `datetime.date` (time fields set to 0).
            - `<'int/float'>` Numeric value treated as total seconds since Unix Epoch.
            - `<'np.datetime64'>` Resolution above microseconds ('us') will be discarded.

        Notes:
        - If multiple datetime-like objects are equally close
          to the instance, the first one in order is returned.
        """
        return self._closest(dtobjs)

    @cython.cfunc
    @cython.inline(True)
    def _closest(self, dtobjs: tuple[object]) -> _Pydt:
        """(internal) Find the datetime closest in time to the instance `<'Pydt/None'>`."""
        res: _Pydt = None
        delta: cython.longlong = LLONG_MAX
        my_tz = datetime.datetime_tzinfo(self)
        utc: cython.bint = my_tz is not None
        my_us: cython.longlong = utils.dt_to_us(self, utc)
        for dtobj in dtobjs:
            # Parse & check timezone parity
            dt = _pydt_fr_dtobj(dtobj)
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
        """Find the datetime farthest in time from the instance `<'Pydt/None'>`.

        :param dtobjs `<'*object'>`: Multiple datetime-like objects to compare with the instance.
            - `<'str'>` A datetime string containing datetime information.
            - `<'datetime.datetime'>` An instance of `datetime.datetime`.
            - `<'datetime.date'>` An instance of `datetime.date` (time fields set to 0).
            - `<'int/float'>` Numeric value treated as total seconds since Unix Epoch.
            - `<'np.datetime64'>` Resolution above microseconds ('us') will be discarded.

        Notes:
        - If multiple datetime-like objects are equally away
          from the instance, the first one in order is returned.
        """
        return self._farthest(dtobjs)

    @cython.cfunc
    @cython.inline(True)
    def _farthest(self, dtobjs: tuple[object]) -> _Pydt:
        """(internal) Find the datetime farthest in time from the instance `<'Pydt/None'>`."""
        res: _Pydt = None
        delta: cython.longlong = -1
        my_tz = datetime.datetime_tzinfo(self)
        utc: cython.bint = my_tz is not None
        my_us: cython.longlong = utils.dt_to_us(self, utc)
        for dtobj in dtobjs:
            # Parse & check timezone parity
            dt = _pydt_fr_dtobj(dtobj)
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
            return _compare_dts(self, _pydt_fr_dtobj(o), True) == 0
        return NotImplemented

    def __le__(self, o: object) -> bool:
        if utils.is_dt(o):
            return _compare_dts(self, o) <= 0
        if utils.is_date(o):
            return _compare_dts(self, utils.dt_fr_date(o)) <= 0
        if isinstance(o, str):
            return _compare_dts(self, _pydt_fr_dtobj(o)) <= 0
        return NotImplemented

    def __lt__(self, o: object) -> bool:
        if utils.is_dt(o):
            return _compare_dts(self, o) < 0
        if utils.is_date(o):
            return _compare_dts(self, utils.dt_fr_date(o)) < 0
        if isinstance(o, str):
            return _compare_dts(self, _pydt_fr_dtobj(o)) < 0
        return NotImplemented

    def __ge__(self, o: object) -> bool:
        if utils.is_dt(o):
            return _compare_dts(self, o) >= 0
        if utils.is_date(o):
            return _compare_dts(self, utils.dt_fr_date(o)) >= 0
        if isinstance(o, str):
            return _compare_dts(self, _pydt_fr_dtobj(o)) >= 0
        return NotImplemented

    def __gt__(self, o: object) -> bool:
        if utils.is_dt(o):
            return _compare_dts(self, o) > 0
        if utils.is_date(o):
            return _compare_dts(self, utils.dt_fr_date(o)) > 0
        if isinstance(o, str):
            return _compare_dts(self, _pydt_fr_dtobj(o)) > 0
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
    """A drop-in replacement for the standard `<'datetime.datetime'>`
    class, providing additional functionalities for more convenient
    datetime operations.
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
        """A drop-in replacement for the standard `<'datetime.datetime'>`
        class, providing additional functionalities for more convenient
        datetime operations.

        :param year `<'int'>`: Year value (1-9999), defaults to `1`.
        :param month `<'int'>`: Month value (1-12), defaults to `1`.
        :param day `<'int'>`: Day value (1-31), defaults to `1`.
        :param hour `<'int'>`: Hour value (0-23), defaults to `0`.
        :param minute `<'int'>`: Minute value (0-59), defaults to `0`.
        :param second `<'int'>`: Second value (0-59), defaults to `0`.
        :param microsecond `<'int'>`: Microsecond value (0-999999), defaults to `0`.
        :param tzinfo `<'tzinfo/str/None'>`: The timezone, defaults to `None`.
            - `<'datetime.tzinfo'>` A subclass of `datetime.tzinfo`.
            - `<'str'>` Timezone name supported by the 'Zoneinfo' module, or `'local'` for local timezone.
            - `<'None'>` Timezone-naive.

        :param fold `<'int'>`: Fold value (0 or 1) for ambiguous times, defaults to `0`.
        """
        return pydt_new(
            year, month, day, hour, minute, second, microsecond, tzinfo, fold
        )
