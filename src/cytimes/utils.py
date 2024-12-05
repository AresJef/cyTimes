# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cytimes import typeref  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
import datetime, numpy as np
from zoneinfo import ZoneInfo
from cytimes import typeref, errors

# Constants --------------------------------------------------------------------------------------------
# . calendar
# fmt: off
DAYS_BR_MONTH: cython.int[13] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
DAYS_IN_MONTH: cython.int[13] = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
DAYS_BR_QUARTER: cython.int[5] = [0, 90, 181, 273, 365]
DAYS_IN_QUARTER: cython.int[5] = [0, 90, 91, 92, 92]
# fmt: on
# . date
ORDINAL_MAX: cython.int = 3_652_059
# . datetime
#: EPOCH (1970-01-01)
EPOCH_DT: datetime.datetime = datetime.datetime_new(
    1970, 1, 1, 0, 0, 0, 0, datetime.get_utc(), 0
)
EPOCH_YEAR: cython.longlong = 1970
EPOCH_MONTH: cython.longlong = 23_628
EPOCH_DAY: cython.longlong = 719_163
EPOCH_HOUR: cython.longlong = EPOCH_DAY * 24
EPOCH_MINUTE: cython.longlong = EPOCH_HOUR * 60
EPOCH_SECOND: cython.longlong = EPOCH_MINUTE * 60
EPOCH_MILLISECOND: cython.longlong = EPOCH_SECOND * 1_000
EPOCH_MICROSECOND: cython.longlong = EPOCH_MILLISECOND * 1_000
# . timezone
UTC: datetime.tzinfo = datetime.get_utc()
# . conversion for seconds
SS_MINUTE: cython.longlong = 60
SS_HOUR: cython.longlong = SS_MINUTE * 60
SS_DAY: cython.longlong = SS_HOUR * 24
# . conversion for milliseconds
MS_SECOND: cython.longlong = 1_000
MS_MINUTE: cython.longlong = MS_SECOND * 60
MS_HOUR: cython.longlong = MS_MINUTE * 60
MS_DAY: cython.longlong = MS_HOUR * 24
# . conversion for microseconds
US_MILLISECOND: cython.longlong = 1_000
US_SECOND: cython.longlong = US_MILLISECOND * 1_000
US_MINUTE: cython.longlong = US_SECOND * 60
US_HOUR: cython.longlong = US_MINUTE * 60
US_DAY: cython.longlong = US_HOUR * 24
# . conversion for nanoseconds
NS_MICROSECOND: cython.longlong = 1_000
NS_MILLISECOND: cython.longlong = NS_MICROSECOND * 1_000
NS_SECOND: cython.longlong = NS_MILLISECOND * 1_000
NS_MINUTE: cython.longlong = NS_SECOND * 60
NS_HOUR: cython.longlong = NS_MINUTE * 60
NS_DAY: cython.longlong = NS_HOUR * 24
# . conversion for timedelta64
TD64_YY_DAY: cython.double = 365.2425  # Exact days in a year for td64
TD64_YY_SECOND: cython.longlong = int(TD64_YY_DAY * SS_DAY)
TD64_YY_MILLISECOND: cython.longlong = TD64_YY_SECOND * 1_000
TD64_YY_MICROSECOND: cython.longlong = TD64_YY_MILLISECOND * 1_000
TD64_YY_NANOSECOND: cython.longlong = TD64_YY_MICROSECOND * 1_000
TD64_MM_DAY: cython.double = 30.436875  # Exact days in a month for td64
TD64_MM_SECOND: cython.longlong = int(TD64_MM_DAY * SS_DAY)
TD64_MM_MILLISECOND: cython.longlong = TD64_MM_SECOND * 1_000
TD64_MM_MICROSECOND: cython.longlong = TD64_MM_MILLISECOND * 1_000
TD64_MM_NANOSECOND: cython.longlong = TD64_MM_MICROSECOND * 1_000
# . datetime64 range
#: Minimum datetime64 in nanoseconds (1677-09-21 00:12:43.145224193)
DT64_NS_YY_MIN: cython.longlong = -293  # >= 1678
DT64_NS_MM_MIN: cython.longlong = -3_508  # >= 1677-10
DT64_NS_WW_MIN: cython.longlong = -15_251  # >= 1677-09-30
DT64_NS_DD_MIN: cython.longlong = -106_751  # >= 1677-09-22
DT64_NS_HH_MIN: cython.longlong = DT64_NS_DD_MIN * 24
DT64_NS_MI_MIN: cython.longlong = DT64_NS_HH_MIN * 60
DT64_NS_SS_MIN: cython.longlong = DT64_NS_MI_MIN * 60
DT64_NS_MS_MIN: cython.longlong = DT64_NS_SS_MIN * 1_000
DT64_NS_US_MIN: cython.longlong = DT64_NS_MS_MIN * 1_000
DT64_NS_NS_MIN: cython.longlong = DT64_NS_US_MIN * 1_000
#: Maximum datetime64 in nanoseconds (2262-04-11 23:47:16.854775807)
DT64_NS_YY_MAX: cython.longlong = 292  # <= 2262
DT64_NS_MM_MAX: cython.longlong = 3_507  # <= 2262-03
DT64_NS_WW_MAX: cython.longlong = 15_250  # <= 2262-04-03
DT64_NS_DD_MAX: cython.longlong = 106_750  # <= 2262-04-10
DT64_NS_HH_MAX: cython.longlong = DT64_NS_DD_MAX * 24
DT64_NS_MI_MAX: cython.longlong = DT64_NS_HH_MAX * 60
DT64_NS_SS_MAX: cython.longlong = DT64_NS_MI_MAX * 60
DT64_NS_MS_MAX: cython.longlong = DT64_NS_SS_MAX * 1_000
DT64_NS_US_MAX: cython.longlong = DT64_NS_MS_MAX * 1_000
DT64_NS_NS_MAX: cython.longlong = DT64_NS_US_MAX * 1_000


# datetime.tzinfo --------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _get_localtimezone() -> object:
    """(internal) Get the local timezone `<'ZoneInfo/timezone'>`."""
    from babel.dates import LOCALTZ

    if isinstance(LOCALTZ, ZoneInfo):
        return LOCALTZ
    try:
        return ZoneInfo(LOCALTZ.zone)
    except Exception:
        return datetime.timezone_new(
            datetime.timedelta_new(0, tz_local_seconds(None), 0), None  # type: ignore
        )


_LOCAL_TZ: object = _get_localtimezone()


@cython.cfunc
@cython.inline(True)
def _prep_timezone_map() -> dict:
    """(internal) Prepare the timezone map `<'dict'>`."""
    from zoneinfo import available_timezones

    # Zoneinfo timezones
    tz_map = {name: ZoneInfo(name) for name in sorted(available_timezones())}
    # Local timezone
    tz_map["local"] = _LOCAL_TZ
    # UTC timezone aliases
    for tz in (
        "UTC",
        "Universal",
        "GMT",
        "GMT+0",
        "GMT-0",
        "GMT0",
        "Greenwich",
        "Zulu",
    ):
        tz_map[tz] = UTC
    return tz_map


_TIMEZONE_MAP: dict = _prep_timezone_map()


@cython.cfunc
@cython.inline(True)
def tz_parse(tz: datetime.tzinfo | str) -> object:
    """(cfunc) Parse 'tz' object into `<'datetime.tzinfo/None'>`.

    :param tz `<'datetime.timezone/Zoneinfo/pytz/str'>`: The timezone object.
        - If 'tz' is an instance of `<'datetime.timezone'>`, return 'tz' directly.
        - If 'tz' is a string or timezone from Zoneinfo or pytz,
           use Python 'Zoneinfo' to (re)-create the timezone object.
    """
    # NoneType
    if tz is None:
        return tz
    # datetime.timezone
    dtype = type(tz)
    if dtype is typeref.TIMEZONE:
        return tz
    # 'str' timezone name
    elif dtype is str:
        try:
            return _TIMEZONE_MAP[tz]
        except Exception:
            pass
    # 'Zoneinfo' timezone
    elif dtype is typeref.ZONEINFO:
        try:
            return _TIMEZONE_MAP[tz.key]
        except Exception:
            pass
    # Subclass of 'Zoneinfo' timezone
    elif isinstance(tz, typeref.ZONEINFO):
        return tz
    # 'pytz' timezone
    elif hasattr(tz, "localize"):
        try:
            return _TIMEZONE_MAP[tz.zone]
        except Exception:
            pass
    # Invalid timezone
    raise errors.InvalidTimezoneError("invalid timezone '%s' %s" % (tz, type(tz)))


########## The REST utility functions are in the utils.pxd file ##########
########## The following functions are for testing purpose only ##########
def _test_utils() -> None:
    # Parser
    _test_parser()
    # Time
    _test_localtime_n_gmtime()
    # Calendar
    _test_is_leap_year()
    _test_days_bf_year()
    _test_quarter_of_month()
    _test_days_in_quarter()
    _test_days_in_month()
    _test_days_bf_month()
    _test_weekday()
    _test_isocalendar()
    _test_iso_1st_monday()
    _test_ymd_to_ordinal()
    _test_ymd_fr_ordinal()
    _test_ymd_fr_isocalendar()
    # datetime.date
    _test_date_generate()
    _test_date_type_check()
    _test_date_conversion()
    # datetime.datetime
    _test_dt_generate()
    _test_dt_type_check()
    _test_dt_tzinfo()
    _test_dt_conversion()
    _test_dt_mainipulate()
    _test_dt_arithmetic()
    # datetime.time
    _test_time_generate()
    _test_time_type_check()
    _test_time_tzinfo()
    _test_time_conversion()
    # datetime.timedelta
    _test_timedelta_generate()
    _test_timedelta_type_check()
    _test_timedelta_conversion()
    # datetime.tzinfo
    _test_tzinfo_generate()
    _test_tzinfo_type_check()
    _test_tzinfo_access()
    # numpy.share
    _test_numpy_share()
    # numpy.datetime64
    _test_datetime64_type_check()
    _test_datetime64_conversion()
    # numpy.timedelta64
    _test_timedelta64_type_check()
    _test_timedelta64_conversion()
    # numpy.ndarray
    _test_ndarray_type_check()
    _test_ndarray_dt64_type_check()
    _test_ndarray_dt64_conversion()
    _test_ndarray_td64_type_check()
    _test_ndarray_td64_conversion()


# Parser
def _test_parser() -> None:
    # boolean
    assert is_iso_sep("t")  # type: ignore
    assert is_iso_sep("T")  # type: ignore
    assert is_iso_sep(" ")  # type: ignore
    assert not is_iso_sep("a")  # type: ignore

    assert is_isodate_sep("-")  # type: ignore
    assert is_isodate_sep("/")  # type: ignore
    assert not is_isodate_sep("a")  # type: ignore

    assert is_isoweek_sep("w")  # type: ignore
    assert is_isoweek_sep("W")  # type: ignore
    assert not is_isoweek_sep("a")  # type: ignore

    assert is_isotime_sep(":")  # type: ignore
    assert not is_isotime_sep("a")  # type: ignore

    for i in "0123456789":
        assert is_ascii_digit(i)  # type: ignore
    assert not is_ascii_digit("a")  # type: ignore

    for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        assert is_ascii_alpha_upper(i)  # type: ignore
    assert not is_ascii_alpha_upper("1")  # type: ignore

    for i in "abcdefghijklmnopqrstuvwxyz":
        assert is_ascii_alpha_lower(i)  # type: ignore
    assert not is_ascii_alpha_lower("1")  # type: ignore

    # Parse
    t: str = "2021-01-02T03:04:05.006007"
    assert parse_isoyear(t, 0, 0) == 2021  # type: ignore
    assert parse_isoyear(t, 1, 0) == -1  # type: ignore
    assert parse_isomonth(t, 5, 0) == 1  # type: ignore
    assert parse_isomonth(t, 6, 0) == -1  # type: ignore
    assert parse_isoday(t, 8, 0) == 2  # type: ignore
    assert parse_isoday(t, 9, 0) == -1  # type: ignore

    t = "2021-W52-6"
    assert parse_isoweek(t, 6, 0) == 52  # type: ignore
    assert parse_isoweek(t, 7, 0) == -1  # type: ignore
    assert parse_isoweekday(t, 9, 0) == 6  # type: ignore
    assert parse_isoweekday(t, 8, 0) == -1  # type: ignore
    assert parse_isoweekday(t, 10, 0) == -1  # type: ignore
    assert parse_isoweekday(t, 1, 0) == -1  # type: ignore
    assert parse_isoweekday(t, 0, 0) == 2  # type: ignore

    t = "2021-365"
    assert parse_isoyearday(t, 5, 0) == 365  # type: ignore
    assert parse_isoyearday(t, 6, 0) == -1  # type: ignore
    assert parse_isoyearday(t, 4, 0) == -1  # type: ignore
    t = "2021-367"
    assert parse_isoyearday(t, 5, 0) == -1  # type: ignore
    t = "2021-000"
    assert parse_isoyearday(t, 5, 0) == -1  # type: ignore

    print("Passed: parser")


# Time
def _test_localtime_n_gmtime() -> None:
    import time

    t = time.time()
    val = tm_localtime(t)  # type: ignore
    cmp = time.localtime(t)
    assert val.tm_sec == cmp.tm_sec, f"{val.tm_sec} != {cmp.tm_sec}"
    assert val.tm_min == cmp.tm_min, f"{val.tm_sec} != {cmp.tm_sec}"
    assert val.tm_hour == cmp.tm_hour, f"{val.tm_sec} != {cmp.tm_sec}"
    assert val.tm_mday == cmp.tm_mday, f"{val.tm_sec} != {cmp.tm_sec}"
    assert val.tm_mon == cmp.tm_mon, f"{val.tm_sec} != {cmp.tm_sec}"
    assert val.tm_year == cmp.tm_year, f"{val.tm_sec} != {cmp.tm_sec}"
    assert val.tm_wday == cmp.tm_wday, f"{val.tm_sec} != {cmp.tm_sec}"
    assert val.tm_yday == cmp.tm_yday, f"{val.tm_sec} != {cmp.tm_sec}"
    assert val.tm_isdst == cmp.tm_isdst, f"{val.tm_sec} != {cmp.tm_sec}"

    val = tm_gmtime(t)  # type: ignore
    cmp = time.gmtime(t)
    assert val.tm_sec == cmp.tm_sec, f"{val.tm_sec} != {cmp.tm_sec}"
    assert val.tm_min == cmp.tm_min, f"{val.tm_sec} != {cmp.tm_sec}"
    assert val.tm_hour == cmp.tm_hour, f"{val.tm_sec} != {cmp.tm_sec}"
    assert val.tm_mday == cmp.tm_mday, f"{val.tm_sec} != {cmp.tm_sec}"
    assert val.tm_mon == cmp.tm_mon, f"{val.tm_sec} != {cmp.tm_sec}"
    assert val.tm_year == cmp.tm_year, f"{val.tm_sec} != {cmp.tm_sec}"
    assert val.tm_wday == cmp.tm_wday, f"{val.tm_sec} != {cmp.tm_sec}"
    assert val.tm_yday == cmp.tm_yday, f"{val.tm_sec} != {cmp.tm_sec}"
    assert val.tm_isdst == cmp.tm_isdst, f"{val.tm_sec} != {cmp.tm_sec}"

    print("Passed: localtime & gmtime")

    del time


# Calendar
def _test_is_leap_year() -> None:
    from _pydatetime import _is_leap  # type: ignore

    for i in range(1, 10000):
        val = is_leap_year(i)  # type: ignore
        cmp = _is_leap(i)
        assert val == cmp, f"{i}: {val} != {cmp}"
    print("Passed: is_leap_year")

    del _is_leap


def _test_days_bf_year() -> None:
    from _pydatetime import _days_before_year  # type: ignore

    for i in range(1, 10000):
        val = days_bf_year(i)  # type: ignore
        cmp = _days_before_year(i)
        assert val == cmp, f"{i}: {val} != {cmp}"
    print("Passed: days_bf_year")

    del _days_before_year


def _test_quarter_of_month() -> None:
    count: cython.int = 0
    value: cython.int = 1
    for i in range(1, 13):
        val = quarter_of_month(i)  # type: ignore
        cmp = value
        assert val == cmp, f"{i}: {val} != {cmp}"
        count += 1
        if count == 3:
            count = 0
            value += 1
    print("Passed: quarter_of_month")


def _test_days_in_quarter() -> None:
    # non-leap
    year: cython.int = 2021
    for i in range(1, 13):
        qtr: cython.int = quarter_of_month(i)  # type: ignore
        val = days_in_quarter(year, i)  # type: ignore
        cmp = DAYS_IN_QUARTER[qtr]
        assert val == cmp, f"{i}: {val} != {cmp}"
    # leap
    year = 2024
    for i in range(1, 13):
        qtr: cython.int = quarter_of_month(i)  # type: ignore
        val = days_in_quarter(year, i)  # type: ignore
        if qtr == 1:
            cmp = DAYS_IN_QUARTER[qtr] + 1
        else:
            cmp = DAYS_IN_QUARTER[qtr]
        assert val == cmp, f"{i}: {val} != {cmp}"
    print("Passed: days_in_quarter")


def _test_days_in_month() -> None:
    from _pydatetime import _days_in_month  # type: ignore

    # non-leap
    year: cython.int = 2021
    for i in range(1, 13):
        val = days_in_month(year, i)  # type: ignore
        cmp = _days_in_month(year, i)
        assert val == cmp, f"{i}: {val} != {cmp}"
    # leap
    year = 2024
    for i in range(1, 13):
        val = days_in_month(year, i)  # type: ignore
        cmp = _days_in_month(year, i)
        assert val == cmp, f"{i}: {val} != {cmp}"
    print("Passed: days_in_month")

    del _days_in_month


def _test_days_bf_month() -> None:
    from _pydatetime import _days_before_month  # type: ignore

    # non-leap
    year: cython.int = 2021
    for i in range(1, 13):
        val = days_bf_month(year, i)  # type: ignore
        cmp = _days_before_month(year, i)
        assert val == cmp, f"{i}: {val} != {cmp}"
    # leap
    year = 2024
    for i in range(1, 13):
        val = days_bf_month(year, i)  # type: ignore
        cmp = _days_before_month(year, i)
        assert val == cmp, f"{i}: {val} != {cmp}"
    print("Passed: days_bf_month")

    del _days_before_month


def _test_weekday() -> None:
    from datetime import date

    year: cython.int
    month: cython.int
    day: cython.int
    for year in range(1, 10000):
        for month in range(1, 13):
            for day in range(1, 32):
                if day > 28:
                    day = min(day, days_in_month(year, month))  # type: ignore
                val = ymd_weekday(year, month, day)  # type: ignore
                cmp = date(year, month, day).weekday()
                assert val == cmp, f"{year}-{month}-{day}: {val} != {cmp}"
    print("Passed: weekday")

    del date


def _test_isocalendar() -> None:
    from datetime import date

    year: cython.int
    month: cython.int
    day: cython.int
    for year in range(1, 10000):
        for month in range(1, 13):
            for day in range(1, 32):
                if day > 28:
                    day = min(day, days_in_month(year, month))  # type: ignore
                iso_calr = ymd_isocalendar(year, month, day)  # type: ignore
                iso_week = ymd_isoweek(year, month, day)  # type: ignore
                iso_year = ymd_isoyear(year, month, day)  # type: ignore
                cmp = date(year, month, day).isocalendar()
                assert (
                    iso_calr.year == cmp.year == iso_year
                ), f"{year}-{month}-{day}: {iso_calr.year} != {cmp.year} != {iso_year}"
                assert (
                    iso_calr.week == cmp.week == iso_week
                ), f"{year}-{month}-{day}: {iso_calr.week} != {cmp.week} != {iso_week}"
                assert (
                    iso_calr.weekday == cmp.weekday
                ), f"{year}-{month}-{day}: {iso_calr.weekday} != {cmp.weekday}"
    print("Passed: isocalendar")

    del date


def _test_iso_1st_monday() -> None:
    from _pydatetime import _isoweek1monday  # type: ignore

    for year in range(1, 10000):
        val = _iso_1st_monday(year)  # type: ignore
        cmp = _isoweek1monday(year)
        assert val == cmp, f"{year}: {val} != {cmp}"
    print("Passed: iso_1st_monday")

    del _isoweek1monday


def _test_ymd_to_ordinal() -> None:
    from _pydatetime import _ymd2ord  # type: ignore

    year: cython.int
    month: cython.int
    day: cython.int
    for year in range(1, 10000):
        for month in range(1, 13):
            for day in range(1, 32):
                if day > 28:
                    day = min(day, days_in_month(year, month))  # type: ignore
                val = ymd_to_ordinal(year, month, day)  # type: ignore
                cmp = _ymd2ord(year, month, day)
                assert val == cmp, f"{year}-{month}-{day}: {val} != {cmp}"
    print("Passed: ymd_to_ordinal")

    del _ymd2ord


def _test_ymd_fr_ordinal() -> None:
    from _pydatetime import _ord2ymd, _MAXORDINAL  # type: ignore

    for i in range(1, _MAXORDINAL + 1):
        val = ymd_fr_ordinal(i)  # type: ignore
        (y, m, d) = _ord2ymd(i)
        assert (
            val.year == y and val.month == m and val.day == d
        ), f"{i}: {val} != {y}-{m}-{d}"
    print("Passed: ymd_fr_ordinal")

    del _ord2ymd, _MAXORDINAL


def _test_ymd_fr_isocalendar() -> None:
    from _pydatetime import _isoweek_to_gregorian  # type: ignore

    year: cython.int
    week: cython.int
    weekday: cython.int
    for year in range(1, 10000):
        for week in range(1, 54):
            for weekday in range(1, 8):
                try:
                    (y, m, d) = _isoweek_to_gregorian(year, week, weekday)
                except ValueError:
                    continue
                val = ymd_fr_isocalendar(year, week, weekday)  # type: ignore
                if y == 10_000 or val.year == 10_000:
                    continue
                assert (
                    val.year == y and val.month == m and val.day == d
                ), f"{year}-{week}-{weekday}: {val} != {y}-{m}-{d}"

    print("Passed: ymd_fr_isocalendar")

    del _isoweek_to_gregorian


# datetime.date
def _test_date_generate() -> None:
    import datetime

    tz = datetime.timezone(datetime.timedelta(hours=23, minutes=59))

    # New
    assert datetime.date(1, 1, 1) == date_new()  # type: ignore
    assert datetime.date(1, 1, 1) == date_new(1)  # type: ignore
    assert datetime.date(1, 1, 1) == date_new(1, 1)  # type: ignore
    assert datetime.date(1, 1, 1) == date_new(1, 1, 1)  # type: ignore

    # Now
    assert datetime.date.today() == date_now()  # type: ignore
    assert datetime.date.today() == date_now(None)  # type: ignore
    assert datetime.datetime.now(UTC).date() == date_now(UTC)  # type: ignore
    assert datetime.datetime.now(tz).date() == date_now(tz)  # type: ignore

    print("Passed: date_generate")

    del datetime


def _test_date_type_check() -> None:
    import datetime

    class CustomDate(datetime.date):
        pass

    date = datetime.date.today()
    assert is_date(date)  # type: ignore
    assert is_date_exact(date)  # type: ignore

    date = CustomDate(1, 1, 1)
    assert is_date(date)  # type: ignore
    assert not is_date_exact(date)  # type: ignore

    print("Passed: date_type_check")

    del CustomDate, datetime


def _test_date_conversion() -> None:
    import datetime

    date = datetime.date(2021, 1, 2)
    dt = datetime.datetime(2021, 1, 2)

    _tm = date_to_tm(date)  # type: ignore
    assert tuple(date.timetuple()) == (
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
    assert "01/02/2021" == date_to_strformat(date, "%m/%d/%Y")  # type: ignore
    assert "2021-01-02" == date_to_isoformat(date)  # type: ignore
    assert date.toordinal() == date_to_ordinal(date)  # type: ignore
    assert (date.toordinal() - EPOCH_DAY) * 86400 == date_to_seconds(date)  # type: ignore
    assert (date.toordinal() - EPOCH_DAY) * 86400_000000 == date_to_us(date)  # type: ignore
    assert int(dt.timestamp()) == date_to_ts(date)  # type: ignore

    class CustomDate(datetime.date):
        pass

    tmp = date_fr_date(CustomDate(2021, 1, 2))  # type: ignore
    assert date == tmp and type(tmp) is datetime.date  # type: ignore

    tmp = date_fr_dt(dt)  # type: ignore
    assert date == tmp and type(tmp) is datetime.date

    tmp = date_fr_ordinal(date.toordinal())  # type: ignore
    assert date == tmp and type(tmp) is datetime.date

    tmp = date_fr_seconds((date.toordinal() - EPOCH_DAY) * 86400)  # type: ignore
    assert date == tmp and type(tmp) is datetime.date

    tmp = date_fr_us((date.toordinal() - EPOCH_DAY) * 86400_000000)  # type: ignore
    assert date == tmp and type(tmp) is datetime.date

    tmp = date_fr_ts(dt.timestamp())  # type: ignore
    assert date == tmp and type(tmp) is datetime.date

    print("Passed: date_conversion")

    del CustomDate, datetime


# datetime.datetime
def _test_dt_generate() -> None:
    import datetime

    tz = datetime.timezone(datetime.timedelta(hours=23, minutes=59))

    # New
    assert datetime.datetime(1, 1, 1, 0, 0, 0, 0) == dt_new()  # type: ignore
    assert datetime.datetime(1, 1, 1, 0, 0, 0, 0) == dt_new(1)  # type: ignore
    assert datetime.datetime(1, 1, 1, 0, 0, 0, 0) == dt_new(1, 1)  # type: ignore
    assert datetime.datetime(1, 1, 1, 0, 0, 0, 0) == dt_new(1, 1, 1)  # type: ignore
    assert datetime.datetime(1, 1, 1, 1, 0, 0, 0) == dt_new(1, 1, 1, 1)  # type: ignore
    assert datetime.datetime(1, 1, 1, 1, 1, 0, 0) == dt_new(1, 1, 1, 1, 1)  # type: ignore
    assert datetime.datetime(1, 1, 1, 1, 1, 1, 0) == dt_new(1, 1, 1, 1, 1, 1)  # type: ignore
    assert datetime.datetime(1, 1, 1, 1, 1, 1, 1) == dt_new(1, 1, 1, 1, 1, 1, 1)  # type: ignore
    assert datetime.datetime(1, 1, 1, 1, 1, 1, 1, tz) == dt_new(1, 1, 1, 1, 1, 1, 1, tz)  # type: ignore

    # Now
    for dt_n, dt_c in (
        (datetime.datetime.now(), dt_now()),  # type: ignore
        (datetime.datetime.now(), dt_now(None)),  # type: ignore
        (datetime.datetime.now(UTC), dt_now(UTC)),  # type: ignore
        (datetime.datetime.now(tz), dt_now(tz)),  # type: ignore
    ):
        assert (
            (dt_n.year == dt_c.year)
            and (dt_n.month == dt_c.month)
            and (dt_n.day == dt_c.day)
            and (dt_n.hour == dt_c.hour)
            and (dt_n.minute == dt_c.minute)
            and (dt_n.second == dt_c.second)
            and (-1000 < dt_n.microsecond - dt_c.microsecond < 1000)
            and (dt_n.tzinfo == dt_c.tzinfo)
        ), f"{dt_n} != {dt_c}"

    print("Passed: dt_generate")

    del datetime


def _test_dt_type_check() -> None:
    import datetime

    class CustomDateTime(datetime.datetime):
        pass

    dt = datetime.datetime.now()
    assert is_dt(dt)  # type: ignore
    assert is_dt_exact(dt)  # type: ignore

    dt = CustomDateTime(1, 1, 1)
    assert is_dt(dt)  # type: ignore
    assert not is_dt_exact(dt)  # type: ignore

    print("Passed: dt_type_check")

    del CustomDateTime, datetime


def _test_dt_tzinfo() -> None:
    import datetime
    from zoneinfo import ZoneInfo

    dt = datetime.datetime(2021, 1, 2, 3, 4, 5, 6)
    tz = datetime.timezone(datetime.timedelta(hours=1, minutes=1))
    dt_tz1 = datetime.datetime(2021, 1, 2, 3, 4, 5, 6, tz)
    dt_tz2 = datetime.datetime(2021, 1, 2, 3, 4, 5, 6, ZoneInfo("CET"))

    for t in (dt, dt_tz1, dt_tz2):
        assert t.tzname() == dt_tzname(t)  # type: ignore
        assert t.dst() == dt_dst(t)  # type: ignore
        assert t.utcoffset() == dt_utcoffset(t)  # type: ignore

    print("Passed: dt_tzinfo")

    del datetime, ZoneInfo


def _test_dt_conversion() -> None:
    import datetime
    from zoneinfo import ZoneInfo
    from pandas import Timestamp

    dt = datetime.datetime(2021, 1, 2, 3, 4, 5, 6)
    tz1 = datetime.timezone(datetime.timedelta(hours=1, minutes=1))
    dt_tz1 = datetime.datetime(2021, 1, 2, 3, 4, 5, 6, tz1)
    tz2 = datetime.timezone(datetime.timedelta(hours=23, minutes=59))
    dt_tz2 = datetime.datetime(2021, 1, 2, 3, 4, 5, 6, tz2)
    tz3 = datetime.timezone(datetime.timedelta(hours=-23, minutes=-59))
    dt_tz3 = datetime.datetime(2021, 1, 2, 3, 4, 5, 6, tz3)
    dt_tz4 = datetime.datetime(2021, 1, 2, 3, 4, 5, 6, ZoneInfo("CET"))

    for d in (dt, dt_tz1, dt_tz2, dt_tz3, dt_tz4):
        _tm = dt_to_tm(d, False)  # type: ignore
        assert tuple(d.timetuple()) == (
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
        _tm = dt_to_tm(d, True)  # type: ignore
        assert tuple(d.utctimetuple()) == (
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

    assert "01/02/2021 000006.05-04-03" == dt_to_strformat(dt, "%m/%d/%Y %f.%S-%M-%H")  # type: ignore
    assert "01/02/2021 000006.05-04-03+0101" == dt_to_strformat(dt_tz1, "%m/%d/%Y %f.%S-%M-%H%z")  # type: ignore
    assert "01/02/2021 000006.05-04-03UTC+01:01" == dt_to_strformat(dt_tz1, "%m/%d/%Y %f.%S-%M-%H%Z")  # type: ignore
    assert "2021-01-02T03:04:05.000006" == dt_to_isoformat(dt_tz1, "T", False)  # type: ignore
    assert "2021-01-02T03:04:05" == dt_to_isoformat(dt_tz1.replace(microsecond=0), "T", False)  # type: ignore
    assert "2021-01-02 03:04:05.000006+0101" == dt_to_isoformat(dt_tz1, " ", True)  # type: ignore
    assert "2021-01-02 03:04:05+0101" == dt_to_isoformat(dt_tz1.replace(microsecond=0), " ", True)  # type: ignore
    assert dt.toordinal() == dt_to_ordinal(dt)  # type: ignore
    assert dt_tz2.toordinal() == dt_to_ordinal(dt_tz2, False)  # type: ignore
    assert dt_tz2.toordinal() - 1 == dt_to_ordinal(dt_tz2, True)  # type: ignore
    assert dt_tz3.toordinal() == dt_to_ordinal(dt_tz3, False)  # type: ignore
    assert dt_tz3.toordinal() + 1 == dt_to_ordinal(dt_tz3, True)  # type: ignore
    secs = (
        (dt.toordinal() - EPOCH_DAY) * 86400
        + dt.hour * 3600
        + dt.minute * 60
        + dt.second
        + dt.microsecond / 1_000_000
    )
    assert secs == dt_to_seconds(dt)  # type: ignore
    assert secs == dt_to_seconds(dt_tz1, False)  # type: ignore
    offset = datetime.timedelta(hours=1, minutes=1).total_seconds()
    assert secs - offset == dt_to_seconds(dt_tz1, True)  # type: ignore
    us = int(secs * 1_000_000)
    assert us == dt_to_us(dt)  # type: ignore
    assert us == dt_to_us(dt_tz1, False)  # type: ignore
    assert us - (offset * 1_000_000) == dt_to_us(dt_tz1, True)  # type: ignore
    for t in (dt, dt_tz1, dt_tz2, dt_tz3, dt_tz4):
        assert t.timestamp() == dt_to_ts(t)  # type: ignore

    date = datetime.date(2021, 1, 2)
    time1 = datetime.time(3, 4, 5, 6)
    time2 = datetime.time(3, 4, 5, 6, tz1)
    assert dt == dt_combine(date, time1)  # type: ignore
    assert dt_tz1 == dt_combine(date, time2)  # type: ignore
    assert datetime.datetime(2021, 1, 2) == dt_combine(date, None)  # type: ignore
    tmp = datetime.datetime.now()
    tmp1 = tmp.replace(hour=3, minute=4, second=5, microsecond=6)
    assert tmp1 == dt_combine(None, time1)  # type: ignore
    tmp2 = tmp1.replace(tzinfo=tz1)
    assert tmp2 == dt_combine(None, time2)  # type: ignore
    tmp3 = tmp.replace(hour=0, minute=0, second=0, microsecond=0)
    assert tmp3 == dt_combine()  # type: ignore

    assert datetime.datetime(2021, 1, 2) == dt_fr_date(date)  # type: ignore
    assert datetime.datetime(2021, 1, 2, tzinfo=tz1) == dt_fr_date(date, tz1)  # type: ignore
    assert dt == dt_fr_dt(Timestamp(dt))  # type: ignore
    assert dt_tz1 == dt_fr_dt(Timestamp(dt_tz1))  # type: ignore
    assert type(dt_fr_dt(Timestamp(dt_tz1))) is datetime.datetime  # type: ignore
    assert datetime.datetime(2021, 1, 2) == dt_fr_ordinal(dt.toordinal())  # type: ignore
    assert datetime.datetime(2021, 1, 2) == dt_fr_ordinal(dt_to_ordinal(dt_tz2, False))  # type: ignore
    assert datetime.datetime(2021, 1, 1) == dt_fr_ordinal(dt_to_ordinal(dt_tz2, True))  # type: ignore
    assert datetime.datetime(2021, 1, 2) == dt_fr_ordinal(dt_to_ordinal(dt_tz3, False))  # type: ignore
    assert datetime.datetime(2021, 1, 3) == dt_fr_ordinal(dt_to_ordinal(dt_tz3, True))  # type: ignore
    assert dt == dt_fr_seconds(dt_to_seconds(dt))  # type: ignore
    assert dt_tz1 == dt_fr_seconds(dt_to_seconds(dt_tz1, False), tz1)  # type: ignore
    assert dt == dt_fr_us(dt_to_us(dt))  # type: ignore
    assert dt_tz1 == dt_fr_us(dt_to_us(dt_tz1, False), tz1)  # type: ignore

    dt = datetime.datetime(2021, 1, 2, 3, 4, 5, 6)
    tz1 = datetime.timezone(datetime.timedelta(hours=1, minutes=1))
    tz2 = datetime.timezone(datetime.timedelta(hours=23, minutes=59))
    tz3 = datetime.timezone(datetime.timedelta(hours=-23, minutes=-59))
    tz4 = ZoneInfo("CET")
    for tz in (None, tz1, tz2, tz3, tz4):
        dt_ = dt.replace(tzinfo=tz1)
        ts = dt_.timestamp()
        assert datetime.datetime.fromtimestamp(ts, tz) == dt_fr_ts(ts, tz)  # type: ignore

    print("Passed: dt_conversion")

    del datetime, ZoneInfo, Timestamp


def _test_dt_mainipulate() -> None:
    import datetime
    from zoneinfo import ZoneInfo

    dt = datetime.datetime(2021, 1, 2, 3, 4, 5, 6)
    tz1 = datetime.timezone(datetime.timedelta(hours=1, minutes=1))
    tz2 = datetime.timezone(datetime.timedelta(hours=23, minutes=59))
    tz3 = datetime.timezone(datetime.timedelta(hours=-23, minutes=-59))
    tz4 = ZoneInfo("CET")

    for tz in (None, tz1, tz2, tz3, tz4):
        assert dt.replace(tzinfo=tz) == dt_replace_tz(dt, tz)  # type: ignore
    assert 1 == dt_replace_fold(dt.replace(tzinfo=tz1, fold=0), 1).fold  # type: ignore

    print("Passed: dt_manipulate")

    del datetime, ZoneInfo


def _test_dt_arithmetic() -> None:
    import datetime

    dt = datetime.datetime(2021, 1, 2, 3, 4, 5, 6)
    td1 = datetime.timedelta(1, 1, 1)
    assert dt_add(dt, 1, 1, 1) == dt + td1  # type: ignore

    td2 = datetime.timedelta(1, 86400, 1)
    assert dt_add(dt, 1, 86400, 1) == dt + td2  # type: ignore

    td3 = datetime.timedelta(1, 86399, 1)
    assert dt_add(dt, 1, 86399, 1) == dt + td3  # type: ignore

    td4 = datetime.timedelta(-1, -1, -1)
    assert dt_add(dt, -1, -1, -1) == dt + td4  # type: ignore

    td5 = datetime.timedelta(-1, -86400, -1)
    assert dt_add(dt, -1, -86400, -1) == dt + td5  # type: ignore

    td6 = datetime.timedelta(-1, -86399, -1)
    assert dt_add(dt, -1, -86399, -1) == dt + td6  # type: ignore

    td7 = datetime.timedelta(1, 60, 100000)
    assert dt_add(dt, 1, 60, 100000) == dt + td7  # type: ignore

    td8 = datetime.timedelta(-1, -60, -100000)
    assert dt_add(dt, -1, -60, -100000) == dt + td8  # type: ignore

    print("Passed: date_arithmetic")

    del datetime


# datetime.time
def _test_time_generate() -> None:
    import datetime

    tz = datetime.timezone(datetime.timedelta(hours=23, minutes=59))

    # New
    assert datetime.time(0, 0, 0, 0) == time_new()  # type: ignore
    assert datetime.time(0, 0, 0, 0) == time_new(0)  # type: ignore
    assert datetime.time(0, 0, 0, 0) == time_new(0, 0)  # type: ignore
    assert datetime.time(0, 0, 0, 0) == time_new(0, 0, 0)  # type: ignore
    assert datetime.time(0, 0, 0, 0) == time_new(0, 0, 0, 0)  # type: ignore
    assert datetime.time(1, 0, 0, 0) == time_new(1)  # type: ignore
    assert datetime.time(1, 1, 0, 0) == time_new(1, 1)  # type: ignore
    assert datetime.time(1, 1, 1, 0) == time_new(1, 1, 1)  # type: ignore
    assert datetime.time(1, 1, 1, 1) == time_new(1, 1, 1, 1)  # type: ignore
    assert datetime.time(1, 1, 1, 1, tz) == time_new(1, 1, 1, 1, tz)  # type: ignore

    # Now
    for t_n, t_c in (
        (datetime.datetime.now().time(), time_now()),  # type: ignore
        (datetime.datetime.now().time(), time_now(None)),  # type: ignore
        (datetime.datetime.now(UTC).timetz(), time_now(UTC)),  # type: ignore
        (datetime.datetime.now(tz).timetz(), time_now(tz)),  # type: ignore
    ):
        assert (
            (t_n.hour == t_c.hour)
            and (t_n.minute == t_c.minute)
            and (t_n.second == t_c.second)
            and (-1000 < t_n.microsecond - t_c.microsecond < 1000)
            and (t_n.tzinfo == t_c.tzinfo)
        ), f"{t_n} != {t_c}"

    print("Passed: time_generate")

    del datetime


def _test_time_type_check() -> None:
    import datetime

    class CustomTime(datetime.time):
        pass

    time = datetime.time(1, 1, 1)
    assert is_time(time)  # type: ignore
    assert is_time_exact(time)  # type: ignore

    time = CustomTime(1, 1, 1)
    assert is_time(time)  # type: ignore
    assert not is_time_exact(time)  # type: ignore

    print("Passed: time_type_check")

    del CustomTime, datetime


def _test_time_tzinfo() -> None:
    import datetime
    from zoneinfo import ZoneInfo

    time = datetime.time(3, 4, 5, 6)
    tz1 = datetime.timezone(datetime.timedelta(hours=1, minutes=1))
    time_tz1 = datetime.time(3, 4, 5, 6, tz1)
    tz2 = datetime.timezone(datetime.timedelta(hours=23, minutes=59))
    time_tz2 = datetime.time(3, 4, 5, 6, tz2)

    for t in (time, time_tz1, time_tz2):
        assert t.tzname() == time_tzname(t)  # type: ignore
        assert t.dst() == time_dst(t)  # type: ignore
        assert t.utcoffset() == time_utcoffset(t)  # type: ignore

    print("Passed: time_tzinfo")

    del datetime, ZoneInfo


def _test_time_conversion() -> None:
    import datetime

    t1 = datetime.time(3, 4, 5, 6)
    tz1 = datetime.timezone(datetime.timedelta(hours=1, minutes=1))
    t_tz1 = datetime.time(3, 4, 5, 6, tz1)
    dt = datetime.datetime(1970, 1, 1, 3, 4, 5, 6)
    dt_tz1 = datetime.datetime(1970, 1, 1, 3, 4, 5, 6, tz1)

    assert "03:04:05.000006" == time_to_isoformat(t_tz1, False)  # type: ignore
    assert "03:04:05.000006+0101" == time_to_isoformat(t_tz1, True)  # type: ignore
    assert "03:04:05" == time_to_isoformat(t_tz1.replace(microsecond=0), False)  # type: ignore
    assert "03:04:05+0101" == time_to_isoformat(t_tz1.replace(microsecond=0), True)  # type: ignore
    secs = t1.hour * 3600 + t1.minute * 60 + t1.second + t1.microsecond / 1_000_000
    assert secs == time_to_seconds(t1)  # type: ignore
    assert secs == time_to_seconds(t_tz1, False)  # type: ignore
    offset = datetime.timedelta(hours=1, minutes=1).total_seconds()
    assert secs - offset == time_to_seconds(t_tz1, True)  # type: ignore
    us = int(secs * 1_000_000)
    assert us == time_to_us(t1)  # type: ignore
    assert us == time_to_us(t_tz1, False)  # type: ignore
    assert us - (offset * 1_000_000) == time_to_us(t_tz1, True)  # type: ignore

    assert datetime.time(3, 4, 5, 6) == time_fr_dt(dt)  # type: ignore
    assert datetime.time(3, 4, 5, 6, tz1) == time_fr_dt(dt_tz1)  # type: ignore

    class CustomTime(datetime.time):
        pass

    tmp = time_fr_time(CustomTime(3, 4, 5, 6))  # type: ignore
    assert t1 == tmp and type(tmp) is datetime.time  # type: ignore
    tmp = time_fr_time(CustomTime(3, 4, 5, 6, tz1))  # type: ignore
    assert t_tz1 == tmp and type(tmp) is datetime.time  # type: ignore
    assert t1 == time_fr_seconds(time_to_seconds(t1))  # type: ignore
    assert t_tz1 == time_fr_seconds(time_to_seconds(t1, False), tz1)  # type: ignore
    assert t1 == time_fr_us(time_to_us(t1))  # type: ignore
    assert t_tz1 == time_fr_us(time_to_us(t1, False), tz1)  # type: ignore

    print("Passed: time_conversion")

    del CustomTime, datetime


# datetime.timedelta
def _test_timedelta_generate() -> None:
    import datetime

    # New
    assert datetime.timedelta(0, 0, 0) == td_new()  # type: ignore
    assert datetime.timedelta(0, 0, 0) == td_new(0)  # type: ignore
    assert datetime.timedelta(0, 0, 0) == td_new(0, 0)  # type: ignore
    assert datetime.timedelta(0, 0, 0) == td_new(0, 0, 0)  # type: ignore
    assert datetime.timedelta(1, 0, 0) == td_new(1)  # type: ignore
    assert datetime.timedelta(1, 1, 0) == td_new(1, 1)  # type: ignore
    assert datetime.timedelta(1, 1, 1) == td_new(1, 1, 1)  # type: ignore
    assert datetime.timedelta(-1, 0, 0) == td_new(-1)  # type: ignore
    assert datetime.timedelta(-1, -1, 0) == td_new(-1, -1)  # type: ignore
    assert datetime.timedelta(-1, -1, -1) == td_new(-1, -1, -1)  # type: ignore

    print("Passed: timedelta_generate")

    del datetime


def _test_timedelta_type_check() -> None:
    import datetime

    class CustomTD(datetime.timedelta):
        pass

    td = datetime.timedelta(1, 1, 1)
    assert is_td(td)  # type: ignore
    assert is_td_exact(td)  # type: ignore

    td = CustomTD(1, 1, 1)
    assert is_td(td)  # type: ignore
    assert not is_td_exact(td)  # type: ignore

    print("Passed: timedelta_type_chech")

    del CustomTD, datetime


def _test_timedelta_conversion() -> None:
    import datetime

    assert "00:00:01" == td_to_isoformat(datetime.timedelta(0, 1))  # type: ignore
    assert "00:01:01" == td_to_isoformat(datetime.timedelta(0, 1, minutes=1))  # type: ignore
    assert "24:01:01" == td_to_isoformat(datetime.timedelta(1, 1, minutes=1))  # type: ignore
    assert "24:01:01.001000" == td_to_isoformat(datetime.timedelta(1, 1, 0, minutes=1, milliseconds=1))  # type: ignore
    assert "24:01:01.000001" == td_to_isoformat(datetime.timedelta(1, 1, 1, minutes=1))  # type: ignore
    assert "24:01:01.001001" == td_to_isoformat(datetime.timedelta(1, 1, 1, minutes=1, milliseconds=1))  # type: ignore
    assert "-00:00:01" == td_to_isoformat(datetime.timedelta(0, -1))  # type: ignore
    assert "-00:01:01" == td_to_isoformat(datetime.timedelta(0, -1, minutes=-1))  # type: ignore
    assert "-24:01:01" == td_to_isoformat(datetime.timedelta(-1, -1, minutes=-1))  # type: ignore
    assert "-24:01:01.001000" == td_to_isoformat(datetime.timedelta(-1, -1, 0, minutes=-1, milliseconds=-1))  # type: ignore
    assert "-24:01:01.000001" == td_to_isoformat(datetime.timedelta(-1, -1, -1, minutes=-1))  # type: ignore
    assert "-24:01:01.001001" == td_to_isoformat(datetime.timedelta(-1, -1, -1, minutes=-1, milliseconds=-1))  # type: ignore

    for h in range(-23, 24):
        for m in range(-59, 60):
            td = datetime.timedelta(hours=h, minutes=m)
            dt_str = str(datetime.datetime.now(datetime.timezone(td)))
            tz_str = dt_str[len(dt_str) - 6 :]
            with cython.wraparound(True):
                tz_str = tz_str[:-3] + tz_str[-2:]
            assert tz_str == td_to_utcformat(td)  # type: ignore

    td = datetime.timedelta(1, 1, 1)
    secs = td.total_seconds()
    assert secs == td_to_seconds(td)  # type: ignore
    assert int(secs * 1_000_000) == td_to_us(td)  # type: ignore
    td = datetime.timedelta(-1, -1, -1)
    secs = td.total_seconds()
    assert secs == td_to_seconds(td)  # type: ignore
    assert int(secs * 1_000_000) == td_to_us(td)  # type: ignore

    class CustomTD(datetime.timedelta):
        pass

    tmp = td_fr_td(CustomTD(-1, -1, -1))  # type: ignore
    assert td == tmp and type(tmp) is datetime.timedelta  # type: ignore
    assert td == td_fr_seconds(td_to_seconds(td))  # type: ignore
    assert td == td_fr_us(td_to_us(td))  # type: ignore

    print("Passed: timedelta_conversion")

    del CustomTD, datetime


# datetime.tzinfo
def _test_tzinfo_generate() -> None:
    import datetime, time
    from babel.dates import LOCALTZ

    # New
    assert datetime.timezone.utc == tz_new()  # type: ignore
    assert datetime.timezone(datetime.timedelta(hours=1, minutes=1)) == tz_new(1, 1)  # type: ignore
    assert datetime.timezone(datetime.timedelta(hours=-1, minutes=-1)) == tz_new(-1, -1)  # type: ignore
    assert datetime.timezone(datetime.timedelta(hours=23, minutes=59)) == tz_new(23, 59)  # type: ignore
    assert datetime.timezone(datetime.timedelta(hours=-23, minutes=-59)) == tz_new(-23, -59)  # type: ignore

    # Local
    assert tz_parse(LOCALTZ) == tz_local()  # type: ignore

    print("Passed: tzinfo_generate")

    del datetime, time, LOCALTZ


def _test_tzinfo_type_check() -> None:
    import datetime

    tz = UTC
    assert is_tz(tz)  # type: ignore
    assert not is_tz_exact(tz)  # type: ignore

    print("Passed: tzinfo_type_check")

    del datetime


def _test_tzinfo_access() -> None:
    import datetime
    from zoneinfo import ZoneInfo

    dt = datetime.datetime.now()
    tz = dt.tzinfo
    assert None == tz_name(tz, dt)  # type: ignore
    assert None == tz_dst(tz, dt)  # type: ignore
    assert None == tz_utcoffset(tz, dt)  # type: ignore

    dt = datetime.datetime.now(UTC)
    tz = dt.tzinfo
    assert "UTC" == tz_name(tz, dt)  # type: ignore
    assert None == tz_dst(tz, dt)  # type: ignore
    assert datetime.timedelta() == tz_utcoffset(tz, dt)  # type: ignore

    dt = datetime.datetime.now(ZoneInfo("Asia/Shanghai"))
    tz = dt.tzinfo
    assert "CST" == tz_name(tz, dt)  # type: ignore
    assert datetime.timedelta() == tz_dst(tz, dt)  # type: ignore
    assert datetime.timedelta(hours=8) == tz_utcoffset(tz, dt)  # type: ignore

    dt = datetime.datetime.now()
    tz = datetime.timezone(datetime.timedelta(hours=23, minutes=59))
    assert "+2359" == tz_utcformat(tz, dt)  # type: ignore
    tz = datetime.timezone(datetime.timedelta(hours=1, minutes=1))
    assert "+0101" == tz_utcformat(tz, dt)  # type: ignore
    tz = datetime.timezone(datetime.timedelta(hours=-1, minutes=-1))
    assert "-0101" == tz_utcformat(tz, dt)  # type: ignore
    tz = datetime.timezone(datetime.timedelta(hours=-23, minutes=-59))
    assert "-2359" == tz_utcformat(tz, dt)  # type: ignore

    print("Passed: tzinfo_access")

    del datetime, ZoneInfo


# . numpy.share
def _test_numpy_share() -> None:
    import numpy as np

    units = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as")

    for unit in units:
        unit == map_nptime_unit_int2str(map_nptime_unit_str2int(unit))  # type: ignore

    for unit in units:
        arr = np.array([], dtype="datetime64[%s]" % unit)
        assert unit == map_nptime_unit_int2str(parse_arr_nptime_unit(arr))  # type: ignore
        arr = np.array([1, 2, 3], dtype="datetime64[%s]" % unit)
        assert unit == map_nptime_unit_int2str(parse_arr_nptime_unit(arr))  # type: ignore
        arr = np.array([], dtype="timedelta64[%s]" % unit)
        assert unit == map_nptime_unit_int2str(parse_arr_nptime_unit(arr))  # type: ignore
        arr = np.array([1, 2, 3], dtype="timedelta64[%s]" % unit)
        assert unit == map_nptime_unit_int2str(parse_arr_nptime_unit(arr))  # type: ignore

    print("Passed: numpy_share")

    del np


# . numpy.datetime64
def _test_datetime64_type_check() -> None:
    import numpy as np

    dt = np.datetime64("2021-01-02")
    assert is_dt64(dt)  # type: ignore
    validate_dt64(dt)  # type: ignore

    dt2 = 1
    assert not is_dt64(dt2)  # type: ignore
    try:
        validate_dt64(dt2)  # type: ignore
    except TypeError:
        pass
    else:
        raise AssertionError("Failed: datetime64_type_check")

    print("Passed: datetime64_type_check")

    del np


def _test_datetime64_conversion() -> None:
    import datetime, numpy as np

    units = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns")
    for unit in units:
        for i in range(-500, 501):
            dt64 = np.datetime64(i, unit)
            us = dt64.astype("datetime64[us]").astype("int64")
            assert dt64_as_int64_us(dt64) == us  # type: ignore
            assert dt64_to_dt(dt64) == dt_fr_us(us)  # type: ignore

    print("Passed: datetime64_conversion")

    del datetime, np


# . numpy.timedelta64
def _test_timedelta64_type_check() -> None:
    import numpy as np

    td = np.timedelta64(1, "D")
    assert is_td64(td)  # type: ignore
    validate_td64(td)  # type: ignore

    td2 = 1
    assert not is_td64(td2)  # type: ignore
    try:
        validate_td64(td2)  # type: ignore
    except TypeError:
        pass
    else:
        raise AssertionError("Failed: timedelta64_type_check")

    print("Passed: timedelta64_type_check")

    del np


def _test_timedelta64_conversion() -> None:
    import datetime, numpy as np

    units = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns")
    for unit in units:
        for i in range(-500, 501):
            td64 = np.timedelta64(i, unit)
            us = td64.astype("timedelta64[us]").astype("int64")
            assert td64_as_int64_us(td64) == us  # type: ignore
            assert td64_to_td(td64) == datetime.timedelta(microseconds=int(us))  # type: ignore

    print("Passed: timedelta64_conversion")

    del datetime, np


# . numpy.ndarray
def _test_ndarray_type_check() -> None:
    import numpy as np

    assert is_arr(np.array([1, 2, 3]))  # type: ignore
    assert is_arr(np.array([]))  # type: ignore
    assert is_arr("a") == False  # type: ignore

    print("Passed: ndarray_type_check")

    del np


# . numpy.ndarray[datetime64]
def _test_ndarray_dt64_type_check() -> None:
    import numpy as np

    assert is_dt64arr(np.array([1, 2, 3], dtype="datetime64[ns]"))  # type: ignore
    assert is_dt64arr(np.array([], dtype="datetime64[ns]"))  # type: ignore
    assert is_dt64arr(np.array([1, 2, 3], dtype="int64")) == False  # type: ignore
    assert is_dt64arr(np.array([], dtype="int64")) == False  # type: ignore

    print("Passed: ndarray_dt64_type_check")

    del np


def _test_ndarray_dt64_conversion() -> None:
    import numpy as np

    units = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns")

    for my_unit in units:
        arr = np.array([i for i in range(-500, 501)], dtype=f"datetime64[{my_unit}]")
        arr_i = arr.astype("int64")
        for to_unit in units:
            cmp = arr.astype(f"datetime64[{to_unit}]").astype("int64")
            val = dt64arr_as_int64(arr, to_unit)  # type: ignore
            assert np.equal(val, cmp).all()
            val = dt64arr_as_int64(arr, to_unit, my_unit)  # type: ignore
            assert np.equal(val, cmp).all()
            val = dt64arr_as_int64(arr_i, to_unit, my_unit)  # type: ignore
            assert np.equal(val, cmp).all()

    for my_unit in units:
        arr = np.array([i for i in range(-500, 501)], dtype=f"datetime64[{my_unit}]")
        for to_unit in units:
            cmp = arr.astype(f"datetime64[{to_unit}]")
            val = dt64arr_as_unit(arr, to_unit)  # type: ignore
            assert np.equal(val, cmp).all()

    print("Passed: ndarray_dt64_conversion")

    del np


# . numpy.ndarray[timedelta64]
def _test_ndarray_td64_type_check() -> None:
    import numpy as np

    assert is_td64arr(np.array([1, 2, 3], dtype="timedelta64[ns]"))  # type: ignore
    assert is_td64arr(np.array([], dtype="timedelta64[ns]"))  # type: ignore
    assert is_td64arr(np.array([1, 2, 3], dtype="int64")) == False  # type: ignore
    assert is_td64arr(np.array([], dtype="int64")) == False  # type: ignore

    print("Passed: ndarray_td64_type_check")

    del np


def _test_ndarray_td64_conversion() -> None:
    import numpy as np

    units = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns")

    for my_unit in units:
        arr = np.array([i for i in range(-500, 501)], dtype=f"timedelta64[{my_unit}]")
        cmp = arr.astype(f"timedelta64[us]")
        val = td64arr_as_int64_us(arr)  # type: ignore
        assert np.equal(val, cmp).all()

    print("Passed: ndarray_td64_conversion")

    del np
