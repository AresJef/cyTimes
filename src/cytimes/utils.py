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
MONTH_TO_QUARTER: cython.int[13] = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
DAYS_BR_QUARTER_NDARRAY: np.ndarray = np.array([0, 90, 181, 273, 365])
# fmt: on
# . microseconds
US_DAY: cython.longlong = 86_400_000_000
US_HOUR: cython.longlong = 3_600_000_000
# . nanoseconds
NS_DAY: cython.longlong = 864_00_000_000_000
NS_HOUR: cython.longlong = 36_00_000_000_000
NS_MINUTE: cython.longlong = 60_000_000_000
# . date
ORDINAL_MAX: cython.int = 3_652_059
# . datetime
UTC: datetime.tzinfo = datetime.get_utc()
EPOCH_DT: datetime.datetime = datetime.datetime_new(1970, 1, 1, 0, 0, 0, 0, UTC, 0)  # type: ignore
EPOCH_US: cython.longlong = 62_135_683_200_000_000
EPOCH_SEC: cython.longlong = 62_135_683_200
EPOCH_DAY: cython.int = 719_163
DT_US_MAX: cython.longlong = 315_537_983_999_999_999
DT_US_MIN: cython.longlong = 86_400_000_000
DT_SEC_MAX: cython.longlong = 315_537_983_999
DT_SEC_MIN: cython.longlong = 86_400
US_FRAC_CORRECTION: cython.int[5] = [100_000, 10_000, 1_000, 100, 10]
# . time
TIME_MIN: datetime.time = datetime.time(0, 0, 0, 0)
TIME_MAX: datetime.time = datetime.time(23, 59, 59, 999999)


# datetime.tzinfo --------------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def tz_parse(tz: datetime.tzinfo | str) -> object:
    """(cfunc) Parse 'tz' object into `<'datetime.tzinfo/None'>`.

    :param tz `<'datetime.tzinfo/str'>`: The timezone object.
        1. If 'tz' is an instance of `<'datetime.tzinfo'>`, return 'tz' directly.
        2. If 'tz' is a string, use Python 'Zoneinfo' to create the timezone object.
    """
    # . <'NoneType'>
    if tz is None:
        return tz

    # . <'ZoneInfo'> or <'datetime.timezone'>
    dtype = type(tz)
    if dtype is typeref.ZONEINFO or dtype is typeref.TIMEZONE:
        return tz

    # . <'str'> timezone name
    if dtype is str:
        try:
            return ZoneInfo(tz)
        except Exception as err:
            # . local
            if tz.lower() == "local":
                return tz_local(None)  # type: ignore
            # . invalid
            raise errors.InvalidTimezoneError(
                "invalid timezone '%s': %s." % (tz, err)
            ) from err

    # . pytz
    if hasattr(tz, "localize"):
        try:
            return ZoneInfo(tz.zone)  # type: ignore
        except Exception:
            pass

    # . unsupported
    raise errors.InvalidTimezoneError("unsupported timezone: %s %r." % (type(tz), tz))


########## The REST utility functions are in the utils.pxd file ##########
########## The following functions are for testing purpose only ##########
def _test_utils() -> None:
    # Delta
    _test_combine_abs_ms_us()
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
    _test_date_manipulation()
    _test_date_arithmetic()
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
    _test_time_manipulate()
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


# Delta
def _test_combine_abs_ms_us() -> None:
    assert combine_abs_ms_us(-1, -1) == -1  # type: ignore
    assert combine_abs_ms_us(0, 0) == 0  # type: ignore
    assert combine_abs_ms_us(-1, 0) == 0  # type: ignore
    assert combine_abs_ms_us(-1, 1) == 1  # type: ignore
    assert combine_abs_ms_us(0, -1) == 0  # type: ignore
    assert combine_abs_ms_us(1, -1) == 1000  # type: ignore
    assert combine_abs_ms_us(1, 1) == 1001  # type: ignore
    assert combine_abs_ms_us(1, 999) == 1999  # type: ignore
    assert combine_abs_ms_us(1, 999999) == 1999  # type: ignore
    assert combine_abs_ms_us(999, 999) == 999999  # type: ignore
    assert combine_abs_ms_us(999, 999999) == 999999  # type: ignore
    print("Passed: combine_abs_ms_us")


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
        val = iso_1st_monday(year)  # type: ignore
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


def _test_date_manipulation() -> None:
    import datetime

    date = datetime.date(2021, 1, 2)
    assert datetime.date(2022, 1, 2) == date_replace(date, 2022)  # type: ignore
    assert datetime.date(2022, 1, 2) == date_replace(date, 2022, -1, -1)  # type: ignore
    assert datetime.date(2021, 2, 2) == date_replace(date, -1, 2)  # type: ignore
    assert datetime.date(2021, 2, 2) == date_replace(date, -1, 2, -1)  # type: ignore
    assert datetime.date(2021, 1, 1) == date_replace(date, -1, -1, 1)  # type: ignore
    assert datetime.date(2022, 2, 28) == date_replace(date, 2022, 2, 28)  # type: ignore
    assert datetime.date(2022, 2, 28) == date_replace(date, 2022, 2, 31)  # type: ignore

    assert datetime.date(2020, 12, 28) == date_chg_weekday(date, -1)  # type: ignore
    assert datetime.date(2020, 12, 28) == date_chg_weekday(date, 0)  # type: ignore
    assert datetime.date(2020, 12, 29) == date_chg_weekday(date, 1)  # type: ignore
    assert datetime.date(2020, 12, 30) == date_chg_weekday(date, 2)  # type: ignore
    assert datetime.date(2020, 12, 31) == date_chg_weekday(date, 3)  # type: ignore
    assert datetime.date(2021, 1, 1) == date_chg_weekday(date, 4)  # type: ignore
    assert date is date_chg_weekday(date, 5)  # type: ignore
    assert datetime.date(2021, 1, 3) == date_chg_weekday(date, 6)  # type: ignore
    assert datetime.date(2021, 1, 3) == date_chg_weekday(date, 7)  # type: ignore

    print("Passed: date_manipulation")

    del datetime


def _test_date_arithmetic() -> None:
    import datetime

    date = datetime.date(2021, 1, 2)
    td1 = datetime.timedelta(1, 1, 1, 1, 1, 1, 1)
    assert date_add(date, 1, 1, 1, 1, 1, 1, 1) == date + td1  # type: ignore

    td2 = datetime.timedelta(1, 86400, 1)
    assert date_add(date, 1, 86400, 1) == date + td2  # type: ignore

    td3 = datetime.timedelta(1, 86399, 1)
    assert date_add(date, 1, 86399, 1) == date + td3  # type: ignore

    td4 = datetime.timedelta(-1, -1, -1, -1, -1, -1, -1)
    assert date_add(date, -1, -1, -1, -1, -1, -1, -1) == date + td4  # type: ignore

    td5 = datetime.timedelta(-1, -86400, -1)
    assert date_add(date, -1, -86400, -1) == date + td5  # type: ignore

    td6 = datetime.timedelta(-1, -86399, -1)
    assert date_add(date, -1, -86399, -1) == date + td6  # type: ignore

    print("Passed: date_arithmetic")

    del datetime


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
    assert "01/02/2021 000006.05-04-03+01:01" == dt_to_strformat(dt_tz1, "%m/%d/%Y %f.%S-%M-%H%z")  # type: ignore
    assert "01/02/2021 000006.05-04-03UTC+01:01" == dt_to_strformat(dt_tz1, "%m/%d/%Y %f.%S-%M-%H%Z")  # type: ignore
    assert "2021-01-02T03:04:05.000006" == dt_to_isoformat(dt_tz1, "T", False)  # type: ignore
    assert "2021-01-02T03:04:05" == dt_to_isoformat(dt_tz1.replace(microsecond=0), "T", False)  # type: ignore
    assert "2021-01-02 03:04:05.000006+01:01" == dt_to_isoformat(dt_tz1, " ", True)  # type: ignore
    assert "2021-01-02 03:04:05+01:01" == dt_to_isoformat(dt_tz1.replace(microsecond=0), " ", True)  # type: ignore
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

    assert datetime.datetime(2022, 1, 2, 3, 4, 5, 6) == dt_replace(dt, 2022)  # type: ignore
    assert datetime.datetime(2022, 1, 2, 3, 4, 5, 6) == dt_replace(dt, 2022, -1, -1)  # type: ignore
    assert datetime.datetime(2021, 2, 2, 3, 4, 5, 6) == dt_replace(dt, -1, 2)  # type: ignore
    assert datetime.datetime(2021, 2, 2, 3, 4, 5, 6) == dt_replace(dt, -1, 2, -1)  # type: ignore
    assert datetime.datetime(2021, 1, 1, 3, 4, 5, 6) == dt_replace(dt, -1, -1, 1)  # type: ignore
    assert datetime.datetime(2022, 2, 28, 2, 2, 2, 2, tz1) == dt_replace(dt, 2022, 2, 28, 2, 2, 2, -1, 2, tz1)  # type: ignore
    assert datetime.datetime(2022, 2, 28, 2, 2, 2, 2, tz1) == dt_replace(dt, 2022, 2, 31, 2, 2, 2, -1, 2, tz1)  # type: ignore
    assert datetime.datetime(2022, 2, 28, 2, 2, 2, 2) == dt_replace(dt, 2022, 2, 31, 2, 2, 2, -1, 2, None)  # type: ignore
    for tz in (None, tz1, tz2, tz3, tz4):
        assert dt.replace(tzinfo=tz) == dt_replace_tz(dt, tz)  # type: ignore
    assert 1 == dt_replace_fold(dt.replace(tzinfo=tz1, fold=0), 1).fold  # type: ignore

    dt = datetime.datetime(2021, 1, 2)
    assert datetime.datetime(2020, 12, 28) == dt_chg_weekday(dt, -1)  # type: ignore
    assert datetime.datetime(2020, 12, 28) == dt_chg_weekday(dt, 0)  # type: ignore
    assert datetime.datetime(2020, 12, 29) == dt_chg_weekday(dt, 1)  # type: ignore
    assert datetime.datetime(2020, 12, 30) == dt_chg_weekday(dt, 2)  # type: ignore
    assert datetime.datetime(2020, 12, 31) == dt_chg_weekday(dt, 3)  # type: ignore
    assert datetime.datetime(2021, 1, 1) == dt_chg_weekday(dt, 4)  # type: ignore
    assert dt is date_chg_weekday(dt, 5)  # type: ignore
    assert datetime.datetime(2021, 1, 3) == dt_chg_weekday(dt, 6)  # type: ignore
    assert datetime.datetime(2021, 1, 3) == dt_chg_weekday(dt, 7)  # type: ignore

    print("Passed: dt_manipulate")

    del datetime, ZoneInfo


def _test_dt_arithmetic() -> None:
    import datetime

    dt = datetime.datetime(2021, 1, 2, 3, 4, 5, 6)
    td1 = datetime.timedelta(1, 1, 1, 1, 1, 1, 1)
    assert dt_add(dt, 1, 1, 1, 1, 1, 1, 1) == dt + td1  # type: ignore

    td2 = datetime.timedelta(1, 86400, 1)
    assert dt_add(dt, 1, 86400, 1) == dt + td2  # type: ignore

    td3 = datetime.timedelta(1, 86399, 1)
    assert dt_add(dt, 1, 86399, 1) == dt + td3  # type: ignore

    td4 = datetime.timedelta(-1, -1, -1, -1, -1, -1, -1)
    assert dt_add(dt, -1, -1, -1, -1, -1, -1, -1) == dt + td4  # type: ignore

    td5 = datetime.timedelta(-1, -86400, -1)
    assert dt_add(dt, -1, -86400, -1) == dt + td5  # type: ignore

    td6 = datetime.timedelta(-1, -86399, -1)
    assert dt_add(dt, -1, -86399, -1) == dt + td6  # type: ignore

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

    _tm = time_to_tm(t1, False)  # type: ignore
    assert tuple(dt.timetuple()) == (
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
    _tm = time_to_tm(t_tz1, True)  # type: ignore
    assert tuple(dt_tz1.utctimetuple()) == (
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
    assert "000006.05:04:03" == time_to_strformat(t1, "%f.%S:%M:%H")  # type: ignore
    assert "000006.05:04:03+01:01" == time_to_strformat(t_tz1, "%f.%S:%M:%H%z")  # type: ignore
    assert "000006.05:04:03UTC+01:01" == time_to_strformat(t_tz1, "%f.%S:%M:%H%Z")  # type: ignore
    assert "03:04:05.000006" == time_to_isoformat(t_tz1, False)  # type: ignore
    assert "03:04:05.000006+01:01" == time_to_isoformat(t_tz1, True)  # type: ignore
    assert "03:04:05" == time_to_isoformat(t_tz1.replace(microsecond=0), False)  # type: ignore
    assert "03:04:05+01:01" == time_to_isoformat(t_tz1.replace(microsecond=0), True)  # type: ignore
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


def _test_time_manipulate() -> None:
    import datetime
    from zoneinfo import ZoneInfo

    t = datetime.time(3, 4, 5, 6)
    tz1 = datetime.timezone(datetime.timedelta(hours=1, minutes=1))
    tz2 = datetime.timezone(datetime.timedelta(hours=23, minutes=59))
    tz3 = datetime.timezone(datetime.timedelta(hours=-23, minutes=-59))
    tz4 = ZoneInfo("CET")

    assert datetime.time(4, 4, 5, 6) == time_replace(t, 4)  # type: ignore
    assert datetime.time(4, 4, 5, 6) == time_replace(t, 4, -1)  # type: ignore
    assert datetime.time(3, 5, 5, 6) == time_replace(t, -1, 5)  # type: ignore
    assert datetime.time(3, 5, 5, 6) == time_replace(t, -1, 5, -1)  # type: ignore
    assert datetime.time(3, 4, 6, 6) == time_replace(t, -1, -1, 6)  # type: ignore
    assert datetime.time(3, 4, 6, 6) == time_replace(t, -1, -1, 6, -1)  # type: ignore
    assert datetime.time(3, 4, 5, 7) == time_replace(t, -1, -1, -1, 7)  # type: ignore
    for tz in (None, tz1, tz2, tz3, tz4):
        assert t.replace(tzinfo=tz) == time_replace_tz(t, tz)  # type: ignore
    assert 1 == time_replace_fold(t.replace(tzinfo=tz1, fold=0), 1).fold  # type: ignore

    print("Passed: time_manipulate")

    del datetime, ZoneInfo


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

    # New
    assert datetime.timezone.utc == tz_new()  # type: ignore
    assert datetime.timezone(datetime.timedelta(hours=1, minutes=1)) == tz_new(1, 1)  # type: ignore
    assert datetime.timezone(datetime.timedelta(hours=-1, minutes=-1)) == tz_new(-1, -1)  # type: ignore
    assert datetime.timezone(datetime.timedelta(hours=23, minutes=59)) == tz_new(23, 59)  # type: ignore
    assert datetime.timezone(datetime.timedelta(hours=-23, minutes=-59)) == tz_new(-23, -59)  # type: ignore

    # Local
    local_offset = -time.timezone if time.localtime().tm_isdst == 0 else -time.altzone
    assert datetime.timezone(datetime.timedelta(seconds=local_offset)) == tz_local()  # type: ignore

    print("Passed: tzinfo_generate")

    del datetime, time


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
    assert "+23:59" == tz_utcformat(tz, dt)  # type: ignore
    tz = datetime.timezone(datetime.timedelta(hours=1, minutes=1))
    assert "+01:01" == tz_utcformat(tz, dt)  # type: ignore
    tz = datetime.timezone(datetime.timedelta(hours=-1, minutes=-1))
    assert "-01:01" == tz_utcformat(tz, dt)  # type: ignore
    tz = datetime.timezone(datetime.timedelta(hours=-23, minutes=-59))
    assert "-23:59" == tz_utcformat(tz, dt)  # type: ignore

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

    #### Positive
    dt_d = np.datetime64("1970-01-02")  # D
    dt_h = np.datetime64("1970-01-01T01")  # h
    dt_m = np.datetime64("1970-01-01T00:01")  # m
    dt_s = np.datetime64("1970-01-01T00:00:01")  # s
    dt_ms = np.datetime64("1970-01-01T00:00:01.123")  # ms
    dt_us = np.datetime64("1970-01-01T00:00:01.123456")  # us
    dt_ns = np.datetime64("1970-01-01T00:00:01.123456789")  # ns
    dt_ps = np.datetime64("1970-01-01T00:00:01.123456789123")  # ps
    dt_fs = np.datetime64("1970-01-01T00:00:01.123456789123456")  # fs
    dt_as = np.datetime64("1970-01-01T00:00:01.123456789123456789")  # as

    cmp = {
        "tm_sec": 1,
        "tm_min": 0,
        "tm_hour": 0,
        "tm_mday": 1,
        "tm_mon": 1,
        "tm_year": 1970,
        "tm_wday": 3,
        "tm_yday": 1,
        "tm_isdst": -1,
    }
    for dt in (dt_s, dt_ms, dt_us, dt_ns, dt_ps, dt_fs, dt_as):
        assert cmp == dt64_to_tm(dt)  # type: ignore

    # to strformat
    fmt = "%Y/%m/%d %H-%M-%S.%f"
    assert "1970/01/01 00-00-01." == dt64_to_strformat(dt_s, fmt, False)  # type: ignore
    assert "1970/01/01 00-00-01.123000" == dt64_to_strformat(dt_ms, fmt, False)  # type: ignore
    assert "1970/01/01 00-00-01.123456" == dt64_to_strformat(dt_us, fmt, False)  # type: ignore
    assert "1970/01/01 00-00-01.123456789" == dt64_to_strformat(dt_ns, fmt, False)  # type: ignore
    assert "1970/01/01 00-00-01.123456789123" == dt64_to_strformat(dt_ps, fmt, False)  # type: ignore
    assert "1970/01/01 00-00-01.123456789123456" == dt64_to_strformat(dt_fs, fmt, False)  # type: ignore
    assert "1970/01/01 00-00-01.123456789123456789" == dt64_to_strformat(dt_as, fmt, False)  # type: ignore

    assert "1970/01/01 00-00-01." == dt64_to_strformat(dt_s, fmt, True)  # type: ignore
    assert "1970/01/01 00-00-01.123000" == dt64_to_strformat(dt_ms, fmt, True)  # type: ignore
    assert "1970/01/01 00-00-01.123456" == dt64_to_strformat(dt_us, fmt, True)  # type: ignore
    assert "1970/01/01 00-00-01.123456" == dt64_to_strformat(dt_ns, fmt, True)  # type: ignore
    assert "1970/01/01 00-00-01.123456" == dt64_to_strformat(dt_ps, fmt, True)  # type: ignore
    assert "1970/01/01 00-00-01.123456" == dt64_to_strformat(dt_fs, fmt, True)  # type: ignore
    assert "1970/01/01 00-00-01.123456" == dt64_to_strformat(dt_as, fmt, True)  # type: ignore

    # To isoformat
    assert "1970-01-01 00:00:01" == dt64_to_isoformat(dt_s, " ", False)  # type: ignore
    assert "1970-01-01T00:00:01.123000" == dt64_to_isoformat(dt_ms, "T", False)  # type: ignore
    assert "1970-01-01 00:00:01.123456" == dt64_to_isoformat(dt_us, " ", False)  # type: ignore
    assert "1970-01-01T00:00:01.123456789" == dt64_to_isoformat(dt_ns, "T", False)  # type: ignore
    assert "1970-01-01 00:00:01.123456789123" == dt64_to_isoformat(dt_ps, " ", False)  # type: ignore
    assert "1970-01-01T00:00:01.123456789123456" == dt64_to_isoformat(dt_fs, "T", False)  # type: ignore
    assert "1970-01-01 00:00:01.123456789123456789" == dt64_to_isoformat(dt_as, " ", False)  # type: ignore

    assert "1970-01-01 00:00:01" == dt64_to_isoformat(dt_s, " ", True)  # type: ignore
    assert "1970-01-01T00:00:01.123000" == dt64_to_isoformat(dt_ms, "T", True)  # type: ignore
    assert "1970-01-01 00:00:01.123456" == dt64_to_isoformat(dt_us, " ", True)  # type: ignore
    assert "1970-01-01T00:00:01.123456" == dt64_to_isoformat(dt_ns, "T", True)  # type: ignore
    assert "1970-01-01 00:00:01.123456" == dt64_to_isoformat(dt_ps, " ", True)  # type: ignore
    assert "1970-01-01T00:00:01.123456" == dt64_to_isoformat(dt_fs, "T", True)  # type: ignore
    assert "1970-01-01 00:00:01.123456" == dt64_to_isoformat(dt_as, " ", True)  # type: ignore

    # To date
    assert datetime.date(1970, 1, 2) == dt64_to_date(dt_d)  # type: ignore
    assert datetime.date(1970, 1, 1) == dt64_to_date(dt_h)  # type: ignore
    assert datetime.date(1970, 1, 1) == dt64_to_date(dt_m)  # type: ignore
    assert datetime.date(1970, 1, 1) == dt64_to_date(dt_s)  # type: ignore
    assert datetime.date(1970, 1, 1) == dt64_to_date(dt_ms)  # type: ignore
    assert datetime.date(1970, 1, 1) == dt64_to_date(dt_us)  # type: ignore
    assert datetime.date(1970, 1, 1) == dt64_to_date(dt_ns)  # type: ignore
    assert datetime.date(1970, 1, 1) == dt64_to_date(dt_ps)  # type: ignore
    assert datetime.date(1970, 1, 1) == dt64_to_date(dt_fs)  # type: ignore
    assert datetime.date(1970, 1, 1) == dt64_to_date(dt_as)  # type: ignore

    # To datetime
    assert datetime.datetime(1970, 1, 2) == dt64_to_dt(dt_d)  # type: ignore
    assert datetime.datetime(1970, 1, 1, 1) == dt64_to_dt(dt_h)  # type: ignore
    assert datetime.datetime(1970, 1, 1, 0, 1) == dt64_to_dt(dt_m)  # type: ignore
    assert datetime.datetime(1970, 1, 1, 0, 0, 1) == dt64_to_dt(dt_s)  # type: ignore
    assert datetime.datetime(1970, 1, 1, 0, 0, 1, 123000) == dt64_to_dt(dt_ms)  # type: ignore
    assert datetime.datetime(1970, 1, 1, 0, 0, 1, 123456) == dt64_to_dt(dt_us)  # type: ignore
    assert datetime.datetime(1970, 1, 1, 0, 0, 1, 123456) == dt64_to_dt(dt_ns)  # type: ignore
    assert datetime.datetime(1970, 1, 1, 0, 0, 1, 123456) == dt64_to_dt(dt_ps)  # type: ignore
    assert datetime.datetime(1970, 1, 1, 0, 0, 1, 123456) == dt64_to_dt(dt_fs)  # type: ignore
    assert datetime.datetime(1970, 1, 1, 0, 0, 1, 123456) == dt64_to_dt(dt_as)  # type: ignore

    # To time
    assert datetime.time(0, 0, 0, 0) == dt64_to_time(dt_d)  # type: ignore
    assert datetime.time(1, 0, 0, 0) == dt64_to_time(dt_h)  # type: ignore
    assert datetime.time(0, 1, 0, 0) == dt64_to_time(dt_m)  # type: ignore
    assert datetime.time(0, 0, 1, 0) == dt64_to_time(dt_s)  # type: ignore
    assert datetime.time(0, 0, 1, 123000) == dt64_to_time(dt_ms)  # type: ignore
    assert datetime.time(0, 0, 1, 123456) == dt64_to_time(dt_us)  # type: ignore
    assert datetime.time(0, 0, 1, 123456) == dt64_to_time(dt_ns)  # type: ignore
    assert datetime.time(0, 0, 1, 123456) == dt64_to_time(dt_ps)  # type: ignore
    assert datetime.time(0, 0, 1, 123456) == dt64_to_time(dt_fs)  # type: ignore
    assert datetime.time(0, 0, 1, 123456) == dt64_to_time(dt_as)  # type: ignore

    # To intger
    dt_d = np.datetime64(1, "D")  # D
    dt_h = np.datetime64(1, "h")  # h
    dt_m = np.datetime64(1, "m")  # m
    dt_s = np.datetime64(1, "s")  # s
    dt_ms = np.datetime64(123, "ms")  # ms
    dt_us = np.datetime64(123456, "us")  # us
    dt_ns = np.datetime64(123456789, "ns")  # ns
    dt_ps = np.datetime64(123456789123, "ps")  # ps
    dt_fs = np.datetime64(123456789123456, "fs")  # fs
    dt_as = np.datetime64(123456789123456789, "as")  # as

    # days
    assert 1 == dt64_to_days(dt_d)  # type: ignore
    assert 24 == dt64_to_hours(dt_d)  # type: ignore
    assert 24 * 60 == dt64_to_minutes(dt_d)  # type: ignore
    assert 24 * 60 * 60 == dt64_to_seconds(dt_d)  # type: ignore
    assert 24 * 60 * 60 * 1_000 == dt64_to_ms(dt_d)  # type: ignore
    assert 24 * 60 * 60 * 1_000_000 == dt64_to_us(dt_d)  # type: ignore
    assert 24 * 60 * 60 * 1_000_000_000 == dt64_to_ns(dt_d)  # type: ignore

    # hours
    assert 0 == dt64_to_days(dt_h)  # type: ignore
    assert 1 == dt64_to_hours(dt_h)  # type: ignore
    assert 60 == dt64_to_minutes(dt_h)  # type: ignore
    assert 60 * 60 == dt64_to_seconds(dt_h)  # type: ignore
    assert 60 * 60 * 1_000 == dt64_to_ms(dt_h)  # type: ignore
    assert 60 * 60 * 1_000_000 == dt64_to_us(dt_h)  # type: ignore
    assert 60 * 60 * 1_000_000_000 == dt64_to_ns(dt_h)  # type: ignore

    # minutes
    assert 0 == dt64_to_days(dt_m)  # type: ignore
    assert 0 == dt64_to_hours(dt_m)  # type: ignore
    assert 1 == dt64_to_minutes(dt_m)  # type: ignore
    assert 60 == dt64_to_seconds(dt_m)  # type: ignore
    assert 60 * 1_000 == dt64_to_ms(dt_m)  # type: ignore
    assert 60 * 1_000_000 == dt64_to_us(dt_m)  # type: ignore
    assert 60 * 1_000_000_000 == dt64_to_ns(dt_m)  # type: ignore

    # seconds
    assert 0 == dt64_to_days(dt_s)  # type: ignore
    assert 0 == dt64_to_hours(dt_s)  # type: ignore
    assert 0 == dt64_to_minutes(dt_s)  # type: ignore
    assert 1 == dt64_to_seconds(dt_s)  # type: ignore
    assert 1_000 == dt64_to_ms(dt_s)  # type: ignore
    assert 1_000_000 == dt64_to_us(dt_s)  # type: ignore
    assert 1_000_000_000 == dt64_to_ns(dt_s)  # type: ignore

    # milliseconds
    assert 0 == dt64_to_days(dt_ms)  # type: ignore
    assert 0 == dt64_to_hours(dt_ms)  # type: ignore
    assert 0 == dt64_to_minutes(dt_ms)  # type: ignore
    assert 0 == dt64_to_seconds(dt_ms)  # type: ignore
    assert 123 == dt64_to_ms(dt_ms)  # type: ignore
    assert 123000 == dt64_to_us(dt_ms)  # type: ignore
    assert 123000000 == dt64_to_ns(dt_ms)  # type: ignore

    # microseconds
    assert 0 == dt64_to_days(dt_us)  # type: ignore
    assert 0 == dt64_to_hours(dt_us)  # type: ignore
    assert 0 == dt64_to_minutes(dt_us)  # type: ignore
    assert 0 == dt64_to_seconds(dt_us)  # type: ignore
    assert 123 == dt64_to_ms(dt_us)  # type: ignore
    assert 123456 == dt64_to_us(dt_us)  # type: ignore
    assert 123456000 == dt64_to_ns(dt_us)  # type: ignore

    # nanoseconds
    assert 0 == dt64_to_days(dt_ns)  # type: ignore
    assert 0 == dt64_to_hours(dt_ns)  # type: ignore
    assert 0 == dt64_to_minutes(dt_ns)  # type: ignore
    assert 0 == dt64_to_seconds(dt_ns)  # type: ignore
    assert 123 == dt64_to_ms(dt_ns)  # type: ignore
    assert 123456 == dt64_to_us(dt_ns)  # type: ignore
    assert 123456789 == dt64_to_ns(dt_ns)  # type: ignore

    # picoseconds
    assert 0 == dt64_to_days(dt_ps)  # type: ignore
    assert 0 == dt64_to_hours(dt_ps)  # type: ignore
    assert 0 == dt64_to_minutes(dt_ps)  # type: ignore
    assert 0 == dt64_to_seconds(dt_ps)  # type: ignore
    assert 123 == dt64_to_ms(dt_ps)  # type: ignore
    assert 123456 == dt64_to_us(dt_ps)  # type: ignore
    assert 123456789 == dt64_to_ns(dt_ps)  # type: ignore

    # femtoseconds
    assert 0 == dt64_to_days(dt_fs)  # type: ignore
    assert 0 == dt64_to_hours(dt_fs)  # type: ignore
    assert 0 == dt64_to_minutes(dt_fs)  # type: ignore
    assert 0 == dt64_to_seconds(dt_fs)  # type: ignore
    assert 123 == dt64_to_ms(dt_fs)  # type: ignore
    assert 123456 == dt64_to_us(dt_fs)  # type: ignore
    assert 123456789 == dt64_to_ns(dt_fs)  # type: ignore

    # attoseconds
    assert 0 == dt64_to_days(dt_as)  # type: ignore
    assert 0 == dt64_to_hours(dt_as)  # type: ignore
    assert 0 == dt64_to_minutes(dt_as)  # type: ignore
    assert 0 == dt64_to_seconds(dt_as)  # type: ignore
    assert 123 == dt64_to_ms(dt_as)  # type: ignore
    assert 123456 == dt64_to_us(dt_as)  # type: ignore
    assert 123456789 == dt64_to_ns(dt_as)  # type: ignore

    #### Negative
    dt_d = np.datetime64("1969-12-31")  # D
    dt_h = np.datetime64("1969-12-31T23")  # h
    dt_m = np.datetime64("1969-12-31T23:59")  # m
    dt_s = np.datetime64("1969-12-31T23:59:59")  # s
    dt_ms = np.datetime64("1969-12-31T23:59:59.123")  # ms
    dt_us = np.datetime64("1969-12-31T23:59:59.123456")  # us
    dt_ns = np.datetime64("1969-12-31T23:59:59.123456789")  # ns
    dt_ps = np.datetime64("1969-12-31T23:59:59.123456789123")  # ps
    dt_fs = np.datetime64("1969-12-31T23:59:59.123456789123456")  # fs
    dt_as = np.datetime64("1969-12-31T23:59:59.123456789123456789")  # as

    cmp = {
        "tm_sec": 59,
        "tm_min": 59,
        "tm_hour": 23,
        "tm_mday": 31,
        "tm_mon": 12,
        "tm_year": 1969,
        "tm_wday": 2,
        "tm_yday": 365,
        "tm_isdst": -1,
    }
    for dt in (dt_s, dt_ms, dt_us, dt_ns, dt_ps, dt_fs, dt_as):
        assert cmp == dt64_to_tm(dt)  # type: ignore

    # to strformat
    fmt = "%Y/%m/%d %H-%M-%S.%f"
    assert "1969/12/31 23-59-59." == dt64_to_strformat(dt_s, fmt, False)  # type: ignore
    assert "1969/12/31 23-59-59.123000" == dt64_to_strformat(dt_ms, fmt, False)  # type: ignore
    assert "1969/12/31 23-59-59.123456" == dt64_to_strformat(dt_us, fmt, False)  # type: ignore
    assert "1969/12/31 23-59-59.123456789" == dt64_to_strformat(dt_ns, fmt, False)  # type: ignore
    assert "1969/12/31 23-59-59.123456789123" == dt64_to_strformat(dt_ps, fmt, False)  # type: ignore
    assert "1969/12/31 23-59-59.123456789123456" == dt64_to_strformat(dt_fs, fmt, False)  # type: ignore
    assert "1969/12/31 23-59-59.123456789123456789" == dt64_to_strformat(dt_as, fmt, False)  # type: ignore

    assert "1969/12/31 23-59-59." == dt64_to_strformat(dt_s, fmt, True)  # type: ignore
    assert "1969/12/31 23-59-59.123000" == dt64_to_strformat(dt_ms, fmt, True)  # type: ignore
    assert "1969/12/31 23-59-59.123456" == dt64_to_strformat(dt_us, fmt, True)  # type: ignore
    assert "1969/12/31 23-59-59.123456" == dt64_to_strformat(dt_ns, fmt, True)  # type: ignore
    assert "1969/12/31 23-59-59.123456" == dt64_to_strformat(dt_ps, fmt, True)  # type: ignore
    assert "1969/12/31 23-59-59.123456" == dt64_to_strformat(dt_fs, fmt, True)  # type: ignore
    assert "1969/12/31 23-59-59.123456" == dt64_to_strformat(dt_as, fmt, True)  # type: ignore

    # To isoformat
    assert "1969-12-31 23:59:59" == dt64_to_isoformat(dt_s, " ", False)  # type: ignore
    assert "1969-12-31T23:59:59.123000" == dt64_to_isoformat(dt_ms, "T", False)  # type: ignore
    assert "1969-12-31 23:59:59.123456" == dt64_to_isoformat(dt_us, " ", False)  # type: ignore
    assert "1969-12-31T23:59:59.123456789" == dt64_to_isoformat(dt_ns, "T", False)  # type: ignore
    assert "1969-12-31 23:59:59.123456789123" == dt64_to_isoformat(dt_ps, " ", False)  # type: ignore
    assert "1969-12-31T23:59:59.123456789123456" == dt64_to_isoformat(dt_fs, "T", False)  # type: ignore
    assert "1969-12-31 23:59:59.123456789123456789" == dt64_to_isoformat(dt_as, " ", False)  # type: ignore

    assert "1969-12-31 23:59:59" == dt64_to_isoformat(dt_s, " ", True)  # type: ignore
    assert "1969-12-31T23:59:59.123000" == dt64_to_isoformat(dt_ms, "T", True)  # type: ignore
    assert "1969-12-31 23:59:59.123456" == dt64_to_isoformat(dt_us, " ", True)  # type: ignore
    assert "1969-12-31T23:59:59.123456" == dt64_to_isoformat(dt_ns, "T", True)  # type: ignore
    assert "1969-12-31 23:59:59.123456" == dt64_to_isoformat(dt_ps, " ", True)  # type: ignore
    assert "1969-12-31T23:59:59.123456" == dt64_to_isoformat(dt_fs, "T", True)  # type: ignore
    assert "1969-12-31 23:59:59.123456" == dt64_to_isoformat(dt_as, " ", True)  # type: ignore

    # To date
    assert datetime.date(1969, 12, 31) == dt64_to_date(dt_d)  # type: ignore
    assert datetime.date(1969, 12, 31) == dt64_to_date(dt_h)  # type: ignore
    assert datetime.date(1969, 12, 31) == dt64_to_date(dt_m)  # type: ignore
    assert datetime.date(1969, 12, 31) == dt64_to_date(dt_s)  # type: ignore
    assert datetime.date(1969, 12, 31) == dt64_to_date(dt_ms)  # type: ignore
    assert datetime.date(1969, 12, 31) == dt64_to_date(dt_us)  # type: ignore
    assert datetime.date(1969, 12, 31) == dt64_to_date(dt_ns)  # type: ignore
    assert datetime.date(1969, 12, 31) == dt64_to_date(dt_ps)  # type: ignore
    assert datetime.date(1969, 12, 31) == dt64_to_date(dt_fs)  # type: ignore
    assert datetime.date(1969, 12, 31) == dt64_to_date(dt_as)  # type: ignore

    # To datetime
    assert datetime.datetime(1969, 12, 31) == dt64_to_dt(dt_d)  # type: ignore
    assert datetime.datetime(1969, 12, 31, 23) == dt64_to_dt(dt_h)  # type: ignore
    assert datetime.datetime(1969, 12, 31, 23, 59) == dt64_to_dt(dt_m)  # type: ignore
    assert datetime.datetime(1969, 12, 31, 23, 59, 59) == dt64_to_dt(dt_s)  # type: ignore
    assert datetime.datetime(1969, 12, 31, 23, 59, 59, 123000) == dt64_to_dt(dt_ms)  # type: ignore
    assert datetime.datetime(1969, 12, 31, 23, 59, 59, 123456) == dt64_to_dt(dt_us)  # type: ignore
    assert datetime.datetime(1969, 12, 31, 23, 59, 59, 123456) == dt64_to_dt(dt_ns)  # type: ignore
    assert datetime.datetime(1969, 12, 31, 23, 59, 59, 123456) == dt64_to_dt(dt_ps)  # type: ignore
    assert datetime.datetime(1969, 12, 31, 23, 59, 59, 123456) == dt64_to_dt(dt_fs)  # type: ignore
    assert datetime.datetime(1969, 12, 31, 23, 59, 59, 123456) == dt64_to_dt(dt_as)  # type: ignore

    # To time
    assert datetime.time(0, 0, 0, 0) == dt64_to_time(dt_d)  # type: ignore
    assert datetime.time(23, 0, 0, 0) == dt64_to_time(dt_h)  # type: ignore
    assert datetime.time(23, 59, 0, 0) == dt64_to_time(dt_m)  # type: ignore
    assert datetime.time(23, 59, 59, 0) == dt64_to_time(dt_s)  # type: ignore
    assert datetime.time(23, 59, 59, 123000) == dt64_to_time(dt_ms)  # type: ignore
    assert datetime.time(23, 59, 59, 123456) == dt64_to_time(dt_us)  # type: ignore
    assert datetime.time(23, 59, 59, 123456) == dt64_to_time(dt_ns)  # type: ignore
    assert datetime.time(23, 59, 59, 123456) == dt64_to_time(dt_ps)  # type: ignore
    assert datetime.time(23, 59, 59, 123456) == dt64_to_time(dt_fs)  # type: ignore
    assert datetime.time(23, 59, 59, 123456) == dt64_to_time(dt_as)  # type: ignore

    # To intger
    dt_d = np.datetime64(-1, "D")  # D
    dt_h = np.datetime64(-1, "h")  # h
    dt_m = np.datetime64(-1, "m")  # m
    dt_s = np.datetime64(-1, "s")  # s
    dt_ms = np.datetime64(-123, "ms")  # ms
    dt_us = np.datetime64(-123456, "us")  # us
    dt_ns = np.datetime64(-123456789, "ns")  # ns
    dt_ps = np.datetime64(-123456789123, "ps")  # ps
    dt_fs = np.datetime64(-123456789123456, "fs")  # fs
    dt_as = np.datetime64(-123456789123456789, "as")  # as

    # days
    assert -1 == dt64_to_days(dt_d)  # type: ignore
    assert -24 == dt64_to_hours(dt_d)  # type: ignore
    assert -24 * 60 == dt64_to_minutes(dt_d)  # type: ignore
    assert -24 * 60 * 60 == dt64_to_seconds(dt_d)  # type: ignore
    assert -24 * 60 * 60 * 1_000 == dt64_to_ms(dt_d)  # type: ignore
    assert -24 * 60 * 60 * 1_000_000 == dt64_to_us(dt_d)  # type: ignore
    assert -24 * 60 * 60 * 1_000_000_000 == dt64_to_ns(dt_d)  # type: ignore

    # hours
    assert -1 == dt64_to_days(dt_h)  # type: ignore
    assert -1 == dt64_to_hours(dt_h)  # type: ignore
    assert -60 == dt64_to_minutes(dt_h)  # type: ignore
    assert -60 * 60 == dt64_to_seconds(dt_h)  # type: ignore
    assert -60 * 60 * 1_000 == dt64_to_ms(dt_h)  # type: ignore
    assert -60 * 60 * 1_000_000 == dt64_to_us(dt_h)  # type: ignore
    assert -60 * 60 * 1_000_000_000 == dt64_to_ns(dt_h)  # type: ignore

    # minutes
    assert -1 == dt64_to_days(dt_m)  # type: ignore
    assert -1 == dt64_to_hours(dt_m)  # type: ignore
    assert -1 == dt64_to_minutes(dt_m)  # type: ignore
    assert -60 == dt64_to_seconds(dt_m)  # type: ignore
    assert -60 * 1_000 == dt64_to_ms(dt_m)  # type: ignore
    assert -60 * 1_000_000 == dt64_to_us(dt_m)  # type: ignore
    assert -60 * 1_000_000_000 == dt64_to_ns(dt_m)  # type: ignore

    # seconds
    assert -1 == dt64_to_days(dt_s)  # type: ignore
    assert -1 == dt64_to_hours(dt_s)  # type: ignore
    assert -1 == dt64_to_minutes(dt_s)  # type: ignore
    assert -1 == dt64_to_seconds(dt_s)  # type: ignore
    assert -1_000 == dt64_to_ms(dt_s)  # type: ignore
    assert -1_000_000 == dt64_to_us(dt_s)  # type: ignore
    assert -1_000_000_000 == dt64_to_ns(dt_s)  # type: ignore

    # milliseconds
    assert -1 == dt64_to_days(dt_ms)  # type: ignore
    assert -1 == dt64_to_hours(dt_ms)  # type: ignore
    assert -1 == dt64_to_minutes(dt_ms)  # type: ignore
    assert -1 == dt64_to_seconds(dt_ms)  # type: ignore
    assert -123 == dt64_to_ms(dt_ms)  # type: ignore
    assert -123000 == dt64_to_us(dt_ms)  # type: ignore
    assert -123000000 == dt64_to_ns(dt_ms)  # type: ignore

    # microseconds
    assert -1 == dt64_to_days(dt_us)  # type: ignore
    assert -1 == dt64_to_hours(dt_us)  # type: ignore
    assert -1 == dt64_to_minutes(dt_us)  # type: ignore
    assert -1 == dt64_to_seconds(dt_us)  # type: ignore
    assert -124 == dt64_to_ms(dt_us)  # type: ignore
    assert -123456 == dt64_to_us(dt_us)  # type: ignore
    assert -123456000 == dt64_to_ns(dt_us)  # type: ignore

    # nanoseconds
    assert -1 == dt64_to_days(dt_ns)  # type: ignore
    assert -1 == dt64_to_hours(dt_ns)  # type: ignore
    assert -1 == dt64_to_minutes(dt_ns)  # type: ignore
    assert -1 == dt64_to_seconds(dt_ns)  # type: ignore
    assert -124 == dt64_to_ms(dt_ns)  # type: ignore
    assert -123457 == dt64_to_us(dt_ns)  # type: ignore
    assert -123456789 == dt64_to_ns(dt_ns)  # type: ignore

    # picoseconds
    assert -1 == dt64_to_days(dt_ps)  # type: ignore
    assert -1 == dt64_to_hours(dt_ps)  # type: ignore
    assert -1 == dt64_to_minutes(dt_ps)  # type: ignore
    assert -1 == dt64_to_seconds(dt_ps)  # type: ignore
    assert -124 == dt64_to_ms(dt_ps)  # type: ignore
    assert -123457 == dt64_to_us(dt_ps)  # type: ignore
    assert -123456790 == dt64_to_ns(dt_ps)  # type: ignore

    # femtoseconds
    assert -1 == dt64_to_days(dt_fs)  # type: ignore
    assert -1 == dt64_to_hours(dt_fs)  # type: ignore
    assert -1 == dt64_to_minutes(dt_fs)  # type: ignore
    assert -1 == dt64_to_seconds(dt_fs)  # type: ignore
    assert -124 == dt64_to_ms(dt_fs)  # type: ignore
    assert -123457 == dt64_to_us(dt_fs)  # type: ignore
    assert -123456790 == dt64_to_ns(dt_fs)  # type: ignore

    # attoseconds
    assert -1 == dt64_to_days(dt_as)  # type: ignore
    assert -1 == dt64_to_hours(dt_as)  # type: ignore
    assert -1 == dt64_to_minutes(dt_as)  # type: ignore
    assert -1 == dt64_to_seconds(dt_as)  # type: ignore
    assert -124 == dt64_to_ms(dt_as)  # type: ignore
    assert -123457 == dt64_to_us(dt_as)  # type: ignore
    assert -123456790 == dt64_to_ns(dt_as)  # type: ignore

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

    #### Positive
    td_d = np.timedelta64(1, "D")  # D
    td_h = np.timedelta64(1, "h")  # h
    td_m = np.timedelta64(1, "m")  # m
    td_s = np.timedelta64(1, "s")  # s
    td_ms = np.timedelta64(123, "ms")  # ms
    td_us = np.timedelta64(123456, "us")  # us
    td_ns = np.timedelta64(123456789, "ns")  # ns
    td_ps = np.timedelta64(123456789123, "ps")  # ps
    td_fs = np.timedelta64(123456789123456, "fs")  # fs
    td_as = np.timedelta64(123456789123456789, "as")  # as

    # days
    assert 1 == td64_to_days(td_d)  # type: ignore
    assert 24 == td64_to_hours(td_d)  # type: ignore
    assert 24 * 60 == td64_to_minutes(td_d)  # type: ignore
    assert 24 * 60 * 60 == td64_to_seconds(td_d)  # type: ignore
    assert 24 * 60 * 60 * 1_000 == td64_to_ms(td_d)  # type: ignore
    assert 24 * 60 * 60 * 1_000_000 == td64_to_us(td_d)  # type: ignore
    assert 24 * 60 * 60 * 1_000_000_000 == td64_to_ns(td_d)  # type: ignore

    # hours
    assert 0 == td64_to_days(td_h)  # type: ignore
    assert 1 == td64_to_hours(td_h)  # type: ignore
    assert 60 == td64_to_minutes(td_h)  # type: ignore
    assert 60 * 60 == td64_to_seconds(td_h)  # type: ignore
    assert 60 * 60 * 1_000 == td64_to_ms(td_h)  # type: ignore
    assert 60 * 60 * 1_000_000 == td64_to_us(td_h)  # type: ignore
    assert 60 * 60 * 1_000_000_000 == td64_to_ns(td_h)  # type: ignore

    # minutes
    assert 0 == td64_to_days(td_m)  # type: ignore
    assert 0 == td64_to_hours(td_m)  # type: ignore
    assert 1 == td64_to_minutes(td_m)  # type: ignore
    assert 60 == td64_to_seconds(td_m)  # type: ignore
    assert 60 * 1_000 == td64_to_ms(td_m)  # type: ignore
    assert 60 * 1_000_000 == td64_to_us(td_m)  # type: ignore
    assert 60 * 1_000_000_000 == td64_to_ns(td_m)  # type: ignore

    # seconds
    assert 0 == td64_to_days(td_s)  # type: ignore
    assert 0 == td64_to_hours(td_s)  # type: ignore
    assert 0 == td64_to_minutes(td_s)  # type: ignore
    assert 1 == td64_to_seconds(td_s)  # type: ignore
    assert 1_000 == td64_to_ms(td_s)  # type: ignore
    assert 1_000_000 == td64_to_us(td_s)  # type: ignore
    assert 1_000_000_000 == td64_to_ns(td_s)  # type: ignore

    # milliseconds
    assert 0 == td64_to_days(td_ms)  # type: ignore
    assert 0 == td64_to_hours(td_ms)  # type: ignore
    assert 0 == td64_to_minutes(td_ms)  # type: ignore
    assert 0 == td64_to_seconds(td_ms)  # type: ignore
    assert 123 == td64_to_ms(td_ms)  # type: ignore
    assert 123_000 == td64_to_us(td_ms)  # type: ignore
    assert 123_000_000 == td64_to_ns(td_ms)  # type: ignore

    # microseconds
    assert 0 == td64_to_days(td_us)  # type: ignore
    assert 0 == td64_to_hours(td_us)  # type: ignore
    assert 0 == td64_to_minutes(td_us)  # type: ignore
    assert 0 == td64_to_seconds(td_us)  # type: ignore
    assert 123 == td64_to_ms(td_us)  # type: ignore
    assert 123_456 == td64_to_us(td_us)  # type: ignore
    assert 123_456_000 == td64_to_ns(td_us)  # type: ignore

    # nanoseconds
    assert 0 == td64_to_days(td_ns)  # type: ignore
    assert 0 == td64_to_hours(td_ns)  # type: ignore
    assert 0 == td64_to_minutes(td_ns)  # type: ignore
    assert 0 == td64_to_seconds(td_ns)  # type: ignore
    assert 123 == td64_to_ms(td_ns)  # type: ignore
    assert 123_457 == td64_to_us(td_ns)  # type: ignore
    assert 123_456_789 == td64_to_ns(td_ns)  # type: ignore

    # picoseconds
    assert 0 == td64_to_days(td_ps)  # type: ignore
    assert 0 == td64_to_hours(td_ps)  # type: ignore
    assert 0 == td64_to_minutes(td_ps)  # type: ignore
    assert 0 == td64_to_seconds(td_ps)  # type: ignore
    assert 123 == td64_to_ms(td_ps)  # type: ignore
    assert 123_457 == td64_to_us(td_ps)  # type: ignore
    assert 123_456_789 == td64_to_ns(td_ps)  # type: ignore

    # femtoseconds
    assert 0 == td64_to_days(td_fs)  # type: ignore
    assert 0 == td64_to_hours(td_fs)  # type: ignore
    assert 0 == td64_to_minutes(td_fs)  # type: ignore
    assert 0 == td64_to_seconds(td_fs)  # type: ignore
    assert 123 == td64_to_ms(td_fs)  # type: ignore
    assert 123_457 == td64_to_us(td_fs)  # type: ignore
    assert 123_456_789 == td64_to_ns(td_fs)  # type: ignore

    # attoseconds
    assert 0 == td64_to_days(td_as)  # type: ignore
    assert 0 == td64_to_hours(td_as)  # type: ignore
    assert 0 == td64_to_minutes(td_as)  # type: ignore
    assert 0 == td64_to_seconds(td_as)  # type: ignore
    assert 123 == td64_to_ms(td_as)  # type: ignore
    assert 123_457 == td64_to_us(td_as)  # type: ignore
    assert 123_456_789 == td64_to_ns(td_as)  # type: ignore

    # To timedelta
    assert datetime.timedelta(days=1) == td64_to_td(td_d)  # type: ignore
    assert datetime.timedelta(hours=1) == td64_to_td(td_h)  # type: ignore
    assert datetime.timedelta(minutes=1) == td64_to_td(td_m)  # type: ignore
    assert datetime.timedelta(seconds=1) == td64_to_td(td_s)  # type: ignore
    assert datetime.timedelta(milliseconds=123) == td64_to_td(td_ms)  # type: ignore
    assert datetime.timedelta(microseconds=123456) == td64_to_td(td_us)  # type: ignore
    assert datetime.timedelta(microseconds=123457) == td64_to_td(td_ns)  # type: ignore
    assert datetime.timedelta(microseconds=123457) == td64_to_td(td_ps)  # type: ignore
    assert datetime.timedelta(microseconds=123457) == td64_to_td(td_fs)  # type: ignore
    assert datetime.timedelta(microseconds=123457) == td64_to_td(td_as)  # type: ignore

    #### Negative
    td_d = np.timedelta64(-1, "D")  # D
    td_h = np.timedelta64(-1, "h")  # h
    td_m = np.timedelta64(-1, "m")  # m
    td_s = np.timedelta64(-1, "s")  # s
    td_ms = np.timedelta64(-123, "ms")  # ms
    td_us = np.timedelta64(-123456, "us")  # us
    td_ns = np.timedelta64(-123456789, "ns")  # ns
    td_ps = np.timedelta64(-123456789123, "ps")  # ps
    td_fs = np.timedelta64(-123456789123456, "fs")  # fs
    td_as = np.timedelta64(-123456789123456789, "as")  # as

    # days
    assert -1 == td64_to_days(td_d)  # type: ignore
    assert -24 == td64_to_hours(td_d)  # type: ignore
    assert -24 * 60 == td64_to_minutes(td_d)  # type: ignore
    assert -24 * 60 * 60 == td64_to_seconds(td_d)  # type: ignore
    assert -24 * 60 * 60 * 1_000 == td64_to_ms(td_d)  # type: ignore
    assert -24 * 60 * 60 * 1_000_000 == td64_to_us(td_d)  # type: ignore
    assert -24 * 60 * 60 * 1_000_000_000 == td64_to_ns(td_d)  # type: ignore

    # hours
    assert 0 == td64_to_days(td_h)  # type: ignore
    assert -1 == td64_to_hours(td_h)  # type: ignore
    assert -60 == td64_to_minutes(td_h)  # type: ignore
    assert -60 * 60 == td64_to_seconds(td_h)  # type: ignore
    assert -60 * 60 * 1_000 == td64_to_ms(td_h)  # type: ignore
    assert -60 * 60 * 1_000_000 == td64_to_us(td_h)  # type: ignore
    assert -60 * 60 * 1_000_000_000 == td64_to_ns(td_h)  # type: ignore

    # minutes
    assert 0 == td64_to_days(td_m)  # type: ignore
    assert 0 == td64_to_hours(td_m)  # type: ignore
    assert -1 == td64_to_minutes(td_m)  # type: ignore
    assert -60 == td64_to_seconds(td_m)  # type: ignore
    assert -60 * 1_000 == td64_to_ms(td_m)  # type: ignore
    assert -60 * 1_000_000 == td64_to_us(td_m)  # type: ignore
    assert -60 * 1_000_000_000 == td64_to_ns(td_m)  # type: ignore

    # seconds
    assert 0 == td64_to_days(td_s)  # type: ignore
    assert 0 == td64_to_hours(td_s)  # type: ignore
    assert 0 == td64_to_minutes(td_s)  # type: ignore
    assert -1 == td64_to_seconds(td_s)  # type: ignore
    assert -1_000 == td64_to_ms(td_s)  # type: ignore
    assert -1_000_000 == td64_to_us(td_s)  # type: ignore
    assert -1_000_000_000 == td64_to_ns(td_s)  # type: ignore

    # milliseconds
    assert 0 == td64_to_days(td_ms)  # type: ignore
    assert 0 == td64_to_hours(td_ms)  # type: ignore
    assert 0 == td64_to_minutes(td_ms)  # type: ignore
    assert 0 == td64_to_seconds(td_ms)  # type: ignore
    assert -123 == td64_to_ms(td_ms)  # type: ignore
    assert -123_000 == td64_to_us(td_ms)  # type: ignore
    assert -123_000_000 == td64_to_ns(td_ms)  # type: ignore

    # microseconds
    assert 0 == td64_to_days(td_us)  # type: ignore
    assert 0 == td64_to_hours(td_us)  # type: ignore
    assert 0 == td64_to_minutes(td_us)  # type: ignore
    assert 0 == td64_to_seconds(td_us)  # type: ignore
    assert -123 == td64_to_ms(td_us)  # type: ignore
    assert -123_456 == td64_to_us(td_us)  # type: ignore
    assert -123_456_000 == td64_to_ns(td_us)  # type: ignore

    # nanoseconds
    assert 0 == td64_to_days(td_ns)  # type: ignore
    assert 0 == td64_to_hours(td_ns)  # type: ignore
    assert 0 == td64_to_minutes(td_ns)  # type: ignore
    assert 0 == td64_to_seconds(td_ns)  # type: ignore
    assert -123 == td64_to_ms(td_ns)  # type: ignore
    assert -123_457 == td64_to_us(td_ns)  # type: ignore
    assert -123_456_789 == td64_to_ns(td_ns)  # type: ignore

    # picoseconds
    assert 0 == td64_to_days(td_ps)  # type: ignore
    assert 0 == td64_to_hours(td_ps)  # type: ignore
    assert 0 == td64_to_minutes(td_ps)  # type: ignore
    assert 0 == td64_to_seconds(td_ps)  # type: ignore
    assert -123 == td64_to_ms(td_ps)  # type: ignore
    assert -123_457 == td64_to_us(td_ps)  # type: ignore
    assert -123_456_789 == td64_to_ns(td_ps)  # type: ignore

    # femtoseconds
    assert 0 == td64_to_days(td_fs)  # type: ignore
    assert 0 == td64_to_hours(td_fs)  # type: ignore
    assert 0 == td64_to_minutes(td_fs)  # type: ignore
    assert 0 == td64_to_seconds(td_fs)  # type: ignore
    assert -123 == td64_to_ms(td_fs)  # type: ignore
    assert -123_457 == td64_to_us(td_fs)  # type: ignore
    assert -123_456_789 == td64_to_ns(td_fs)  # type: ignore

    # attoseconds
    assert 0 == td64_to_days(td_as)  # type: ignore
    assert 0 == td64_to_hours(td_as)  # type: ignore
    assert 0 == td64_to_minutes(td_as)  # type: ignore
    assert 0 == td64_to_seconds(td_as)  # type: ignore
    assert -123 == td64_to_ms(td_as)  # type: ignore
    assert -123_457 == td64_to_us(td_as)  # type: ignore
    assert -123_456_789 == td64_to_ns(td_as)  # type: ignore

    # To timedelta
    assert datetime.timedelta(days=-1) == td64_to_td(td_d)  # type: ignore
    assert datetime.timedelta(hours=-1) == td64_to_td(td_h)  # type: ignore
    assert datetime.timedelta(minutes=-1) == td64_to_td(td_m)  # type: ignore
    assert datetime.timedelta(seconds=-1) == td64_to_td(td_s)  # type: ignore
    assert datetime.timedelta(milliseconds=-123) == td64_to_td(td_ms)  # type: ignore
    assert datetime.timedelta(microseconds=-123456) == td64_to_td(td_us)  # type: ignore
    assert datetime.timedelta(microseconds=-123457) == td64_to_td(td_ns)  # type: ignore
    assert datetime.timedelta(microseconds=-123457) == td64_to_td(td_ps)  # type: ignore
    assert datetime.timedelta(microseconds=-123457) == td64_to_td(td_fs)  # type: ignore
    assert datetime.timedelta(microseconds=-123457) == td64_to_td(td_as)  # type: ignore

    print("Passed: timedelta64_conversion")

    del datetime, np
