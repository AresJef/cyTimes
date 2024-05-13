# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from __future__ import annotations

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.libc.math import roundl, ceill, floorl  # type: ignore
from cython.cimports.cpython.dict import PyDict_GetItem as dict_getitem  # type: ignore
from cython.cimports.cytimes import cydatetime as cydt, typeref  # type: ignore
from cython.cimports.cytimes.cytimedelta import cytimedelta, combine_ms_us  # type: ignore
from cython.cimports.cytimes.cyparser import Config, Parser, CONFIG_MONTH, CONFIG_WEEKDAY  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
from typing import Literal
from time import struct_time
import datetime, numpy as np
from zoneinfo import available_timezones
from pandas import Timestamp
from cytimes.cyparser import Config
from cytimes.cytimedelta import cytimedelta
from cytimes import cydatetime as cydt, typeref, errors

__all__ = ["pydt"]

# Constants -----------------------------------------------------------------------------------
# . timezones
TIMEZONE_AVAILABLE: set[str] = available_timezones()


# pydt (Python Datetime) ----------------------------------------------------------------------
@cython.cclass
class pydt:
    """Represents the pydt (Python Datetime) that makes
    working with datetime easier.
    """

    # Value
    _dt: datetime.datetime
    # Config
    _cfg: Config
    _day1st: object
    _year1st: object
    _ignoretz: object
    _isoformat: object
    _default: object
    # Hashcode
    _hashcode: cython.Py_ssize_t

    @classmethod
    def now(cls, tz: str | datetime.tzinfo | None = None) -> pydt:
        """(Class method) Create the current datetime
        (with specific timezone [Optional]) `<'pydt'>`.

        Equivalent to 'datetime.now(tz)'

        ### Notice
        Param 'tz' accept both string (timezone name) and tzinfo
        instance. However, timezone from `pytz` library should not
        be used, and could yield incorrect result.
        """
        # Without timezone
        if tz is None:
            return cls(cydt.gen_dt_now())
        # With timezone
        return cls(cydt.gen_dt_now_tz(parse_tzinfo(tz)))  # type: ignore

    @classmethod
    def from_datetime(
        cls,
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        microsecond: int = 0,
        tz: str | datetime.tzinfo | None = None,
    ) -> pydt:
        """(Class method) Create from datetime values `<'pydt'>`.

        ### Notice
        Param 'tz' accept both string (timezone name) and tzinfo
        instance. However, timezone from `pytz` library should not
        be used, and could yield incorrect result.
        """
        # Without timezone
        # fmt: off
        if tz is None:
            return cls(datetime.datetime_new(
                year, month, day, hour, minute, 
                second, microsecond, None, 0 )
            )
        # With timezone
        return cls(datetime.datetime_new(
            year, month, day, hour, minute, 
            second, microsecond, parse_tzinfo(tz), 0 ) # type: ignore
        )
        # fmt: on

    @classmethod
    def from_ordinal(
        cls,
        ordinal: cython.int,
        tz: str | datetime.tzinfo | None = None,
    ) -> pydt:
        """(Class method) Create from ordinal of a date `<'pydt'>`.

        ### Notice
        Param 'tz' accept both string (timezone name) and tzinfo
        instance. However, timezone from `pytz` library should not
        be used, and could yield incorrect result.
        """
        # Without timezone
        if tz is None:
            return cls(cydt.dt_fr_ordinal(ordinal, None, 0))
        # With timezone
        return cls(cydt.dt_fr_ordinal(ordinal, parse_tzinfo(tz), 0))  # type: ignore

    @classmethod
    def from_isoformat(
        cls,
        dtstr: str,
        default: object = None,
        day1st: bool | None = None,
        year1st: bool | None = None,
        ignoretz: bool = False,
        isoformat: bool = True,
        cfg: Config | None = None,
    ) -> pydt:
        """(Class method) Create from ISO format `<'pydt'>`.

        Equivalent to 'datetime.fromisoformat(dtstr)'
        """
        return cls(dtstr, default, day1st, year1st, ignoretz, isoformat, cfg)

    @classmethod
    def from_isocalendar(cls, year: int, week: int, weekday: int) -> pydt:
        """(Class method) Create from ISO calendar `<'pydt'>`.

        Equivalent to 'datetime.fromisocalendar(year, week, weekday)'
        """
        try:
            y: cython.uint = year
        except Exception as err:
            raise errors.InvalidIsoCalendarError(
                "Invalid ISO calendar 'year': '%d'." % year
            ) from err
        try:
            w: cython.uint = week
        except Exception as err:
            raise errors.InvalidIsoCalendarError(
                "Invalid ISO calendar 'week': '%d'." % week
            ) from err
        try:
            d: cython.uint = weekday
        except Exception as err:
            raise errors.InvalidIsoCalendarError(
                "Invalid ISO calendar 'weekday': '%d'." % weekday
            ) from err
        ymd = cydt.isocalendar_to_ymd(y, w, d)
        return cls(datetime.date_new(ymd.year, ymd.month, ymd.day))

    @classmethod
    def from_timestamp(
        cls,
        timestamp: int | float,
        tz: str | datetime.tzinfo | None = None,
    ) -> pydt:
        """(Class method) Create from a timestamp `<'pydt'>`.

        ### Notice
        Param 'tz' accept both string (timezone name) and tzinfo
        instance. However, timezone from `pytz` library should not
        be used, and could yield incorrect result.
        """
        # Without timezone
        if tz is None:
            return cls(cydt.dt_fr_timestamp(timestamp, None))
        # With timezone
        return cls(cydt.dt_fr_timestamp(timestamp, parse_tzinfo(tz)))  # type: ignore

    @classmethod
    def from_seconds(
        cls,
        seconds: float,
        tz: str | datetime.tzinfo | None = None,
    ) -> pydt:
        """(Class method) Create from total seconds since EPOCH `<'pydt'>`.

        ### Notice
        Param 'tz' accept both string (timezone name) and tzinfo
        instance. However, timezone from `pytz` library should not
        be used, and could yield incorrect result.
        """
        # Without timezone
        if tz is None:
            return cls(cydt.dt_fr_seconds(seconds, None, 0))
        # With timezone
        return cls(cydt.dt_fr_seconds(seconds, parse_tzinfo(tz), 0))  # type: ignore

    @classmethod
    def from_microseconds(
        cls,
        microseconds: int,
        tz: str | datetime.tzinfo | None = None,
    ) -> pydt:
        """(Class method) Create from total microseconds since EPOCH `<'pydt'>`.

        ### Notice
        Param 'tz' accept both string (timezone name) and tzinfo
        instance. However, timezone from `pytz` library should not
        be used, and could yield incorrect result.
        """
        # Without timezone
        if tz is None:
            return cls(cydt.dt_fr_microseconds(microseconds, None, 0))
        # With timezone
        return cls(cydt.dt_fr_microseconds(microseconds, parse_tzinfo(tz), 0))  # type: ignore

    @classmethod
    def from_strformat(cls, dtstr: str, format: str) -> pydt:
        """(Class method) Create from string format `<'pydt'>`.

        Equivalent to 'datetime.strptime(dtstr, format)'
        """
        return cls(datetime.datetime.strptime(dtstr, format))

    @classmethod
    def combine(cls, date: datetime.date, time: datetime.time) -> pydt:
        """(Class method) Combine date and time `<'pydt'>`.

        Equivalent to 'datetime.combine(date, time)'
        """
        return cls(cydt.dt_fr_date_n_time(date, time))

    def __init__(
        self,
        dtobj: object = None,
        default: object | None = None,
        day1st: bool | None = None,
        year1st: bool | None = None,
        ignoretz: bool = False,
        isoformat: bool = True,
        cfg: Config | None = None,
    ) -> None:
        """The pydt (Python Datetime) that makes working with datetime easier.

        ### Datetime Object
        :param dtobj `<'object'>`: The datetime object to convert to pydt. Defaults to `None`.
        - Supported data types:
        1. `<'str'>`: The datetime string, e.g. "2021-01-01 12:00:00" or "Jan 12, 2023".
        2. `<'datetime.datetime'>`: Python native datetime or subclass.
        3. `<'datetime.date'>`: Python native date or subclass.
        4. `<'pydt'>`: Another pydt instance.
        6. `<'numpy.datetime64'>`: Numpy datetime64 instance.
        7. `<'NoneType'> None`: Defaults to the current local datetime.

        ### Parse 'dtobj' (Only applicable when 'dtobj' is type of `<'str'>`).
        :param default `<'object'>`: The default to fill-in missing datetime elements. Defaults to `None`.
        - `<'date'>`: If parser failed to extract Y/M/D values from the string,
           the give 'date' will be used to fill-in the missing Y/M/D values.
        - `<'datetime'>`: If parser failed to extract datetime elements from
           the string, the given 'datetime' will be used to fill-in the
           missing Y/M/D & H/M/S.f values.
        - `None`: raise error if Y/M/D values are missing.

        :param day1st `<'bool'>`: Whether to interpret first ambiguous date values as day. Defaults to `None`.
        :param year1st `<'bool'>`: Whether to interpret first the ambiguous date value as year. Defaults to `None`.
        - Both the 'day1st' & 'year1st' arguments works together to determine how
          to interpret ambiguous Y/M/D values. If not provided (set to `None`),
          defaults to the 'day1st' & 'year1st' settings of the Parser `<'Config'>`.
        - For more information, please refer to the `cytimes.Parser.parse()` method.

        :param ignoretz `<'bool'>`: Whether to ignore timezone information. Defaults to `False`.
        - `True`: Parser ignores any timezone information and only returns
           timezone-naive datetime. Setting to `True` can increase parser
           performance.
        - `False`: Parser will try to process the timzone information in
           the string, and generate a timezone-aware datetime if timezone
           has been matched by the Parser `<'Config'>` settings: 'utc' & 'tz'.

        :param isoformat `<'bool'>`: Whether the 'dtstr' is in ISO format. Defaults to `True`.
        - `True`: Parser will first try to process the 'dtstr' in ISO format.
           If failed, then process the 'dtstr' through timelex tokens.
        - `False`: Parser will only process the 'dtstr' through timelex tokens.
           If the 'dtstr' is confirmed to not be in ISO format, setting to
           `False` can increase parser performance.

        :param cfg `<'Config'>`: The configurations for the Parser. Defaults to `None`.
        """
        # Set config
        self._cfg = cfg
        self._day1st = day1st
        self._year1st = year1st
        self._ignoretz = ignoretz
        self._isoformat = isoformat
        # Parse default
        if default is not None:
            default = self._parse_dtobj(default, None)
        self._default = default
        # Parse pydt value
        self._dt = self._parse_dtobj(dtobj, default)
        # Initial hashcode
        self._hashcode = -1

    # Access ----------------------------------------------------------------------------------
    @property
    def year(self) -> int:
        """Access the year `<'int'>`."""
        return self._dt.year

    @property
    def quarter(self) -> int:
        """Access the quarter `<'int'>`."""
        return self._c_quarter()

    @property
    def month(self) -> int:
        """Access the month `<'int'>`."""
        return self._dt.month

    @property
    def day(self) -> int:
        """Access the day `<'int'>`."""
        return self._dt.day

    @property
    def hour(self) -> int:
        """Access the hour `<'int'>`."""
        return self._dt.hour

    @property
    def minute(self) -> int:
        """Access the minute `<'int'>`."""
        return self._dt.minute

    @property
    def second(self) -> int:
        """Access the second `<'int'>`."""
        return self._dt.second

    @property
    def microsecond(self) -> int:
        """Access the microsecond `<'int'>`."""
        return self._dt.microsecond

    @property
    def tzinfo(self) -> datetime.tzinfo | None:
        """Access the timezone `<'datetime.tzinfo'>`."""
        return self._dt.tzinfo

    @property
    def tzname(self) -> str | None:
        """Access the timezone name `<'str/None'>`."""
        return self._c_tzname()

    @property
    def dst(self) -> datetime.timedelta | None:
        """Access the DST `<'datetime.timedelta/None'>`."""
        return self._c_dst()

    @property
    def utcoffset(self) -> datetime.timedelta | None:
        """Access the UTC offset `<'datetime.timedelta/None'>`."""
        return self._c_utcoffset()

    @property
    def fold(self) -> int:
        """Access the fold `<'int'>`."""
        return self._dt.fold

    @property
    def dt(self) -> datetime.datetime:
        """Access as `<'datetime.datetime'>`."""
        return self._dt

    @property
    def dt_str(self) -> str:
        """Access as datetime in string format `<'str'>`.
        e.g: "2021-01-01 12:00:00.000001"""
        return self._c_dt_str()

    @property
    def dt_strtz(self) -> str:
        """Access as datetime in string format with timezone `<'str'>`.
        e.g: "2021-01-01 12:00:00.000001+02:00"""
        return self._c_dt_strtz()

    @property
    def dt_iso(self) -> str:
        """Access as datetime in ISO format `<'str'>`.
        e.g: "2021-01-01T12:00:00.000001"""
        return self._c_dt_iso()

    @property
    def dt_isotz(self) -> str:
        """Access as datetime in ISO format with timezone `<'str'>`.
        e.g: "2021-01-01T12:00:00+02:00.000001"""
        return self._c_dt_isotz()

    @property
    def date(self) -> datetime.date:
        """Access as `<'datetime.date'>`."""
        return self._c_date()

    @property
    def date_iso(self) -> str:
        """Access as date in ISO format `<'str'>`."""
        return self._c_date_iso()

    @property
    def time(self) -> datetime.time:
        """Access as `<'datetime.time'>`."""
        return self._c_time()

    @property
    def timetz(self) -> datetime.time:
        """Access as `<'datetime.time'>` with timezone."""
        return self._c_timetz()

    @property
    def time_iso(self) -> str:
        """Access as time in ISO format `<'str'>`."""
        return self._c_time_iso()

    @property
    def stime(self) -> struct_time:
        """Access as `<'time.struct_time'>`."""
        tms = self._c_stime()
        # fmt: off
        return typeref.STRUCT_TIME( ( 
            tms.tm_year, tms.tm_mon, tms.tm_mday, 
            tms.tm_hour, tms.tm_min, tms.tm_sec, 
            tms.tm_wday, tms.tm_yday, tms.tm_isdst )
        )
        # fmt: on

    @property
    def stime_utc(self) -> struct_time:
        """Access as `<'time.struct_time'>` in UTC."""
        tms = self._c_stime_utc()
        # fmt: off
        return typeref.STRUCT_TIME( ( 
            tms.tm_year, tms.tm_mon, tms.tm_mday, 
            tms.tm_hour, tms.tm_min, tms.tm_sec, 
            tms.tm_wday, tms.tm_yday, tms.tm_isdst )
        )
        # fmt: on

    @property
    def ts(self) -> Timestamp:
        """Access as `<'pandas.Timestamp'>`."""
        return typeref.TIMESTAMP(self._dt)

    @property
    def dt64(self) -> np.datetime64:
        """Access as `<'numpy.datetime64'>`."""
        return typeref.DATETIME64(self._c_microseconds_utc(), "us")

    @property
    def ordinal(self) -> int:
        """Access as ordinal of the date `<'int'>`."""
        return self._c_ordinal()

    @property
    def seconds(self) -> float:
        """Access in total seconds since EPOCH, ignoring
        the timezone (if exists) `<'float'>`.

        ### Notice
        This should `NOT` be treated as timestamp.
        """
        return self._c_seconds()

    @property
    def seconds_utc(self) -> float:
        """Access in total seconds since EPOCH `<'float'>`.
        - If `timezone-aware`, return total seconds in UTC.
        - If `timezone-naive`, requivalent to `pydt.seconds`.

        ### Notice
        This should `NOT` be treated as timestamp.
        """
        return self._c_seconds_utc()

    @property
    def microseconds(self) -> int:
        """Access in total microseconds since EPOCH, ignoring
        the timezone (if exists) `<'int'>`."""
        return self._c_microseconds()

    @property
    def microseconds_utc(self) -> int:
        """Access in total microseconds since EPOCH `<'int'>`.
        - If `timezone-aware`, return total microseconds in UTC.
        - If `timezone-naive`, requivalent to `pydt.microseconds`.
        """
        return self._c_microseconds_utc()

    @property
    def timestamp(self) -> float:
        """Access in timestamp `<'float'>`."""
        return self._c_timestamp()

    # Year ------------------------------------------------------------------------------------
    def is_leapyear(self) -> bool:
        """Whether the current date is a leap year `<'bool'>`."""
        return self._c_is_leapyear()

    def leap_bt_years(self, year: cython.int) -> int:
        """Calculate the number of leap years between
        the current date and the given 'year' `<'int'>`."""
        return self._c_leap_bt_years(year)

    @property
    def days_in_year(self) -> int:
        """Get the maximum number of days in the year.
        Expect 365 or 366 (leapyear) `<'int'>`."""
        return self._c_days_in_year()

    @property
    def days_bf_year(self) -> int:
        """Get the number of days betweem the 1st day of 1AD and
        the 1st day of the year of the current date `<'int'>`."""
        return self._c_days_bf_year()

    @property
    def days_of_year(self) -> int:
        """Get the number of days between the 1st day
        of the year and the current date `<'int'>`."""
        return self._c_days_of_year()

    def is_year(self, year: cython.int) -> bool:
        """Whether the current year is a specific year `<'bool'>`."""
        return self._c_is_year(year)

    def is_year_1st(self) -> bool:
        """Whether is the 1st day of the year `<'bool'>`."""
        return self._c_is_year_1st()

    def is_year_lst(self) -> bool:
        """Whether is the last day of the current year `<'bool'>`."""
        return self._c_is_year_lst()

    @cython.ccall
    def to_year_1st(self) -> pydt:
        """Go to the 1st day of the current year `<'pydt'>`."""
        dt: datetime.datetime = self._dt
        # fmt: off
        return self._new(datetime.datetime_new(
            dt.year, 1, 1, dt.hour, dt.minute, 
            dt.second, dt.microsecond, dt.tzinfo, dt.fold )
        )
        # fmt: on

    @cython.ccall
    def to_year_lst(self) -> pydt:
        """Go to the last day of the current year `<'pydt'>`."""
        dt: datetime.datetime = self._dt
        # fmt: off
        return self._new(datetime.datetime_new(
                dt.year, 12, 31, dt.hour, dt.minute, 
                dt.second, dt.microsecond, dt.tzinfo, dt.fold )
        )
        # fmt: on

    @cython.ccall
    def to_curr_year(
        self,
        month: cython.int | str | None = None,
        day: cython.int = -1,
    ) -> pydt:
        """Go to specific 'month' and 'day' of the current year `<'pydt'>`."""
        # Parse new month
        new_mth: cython.int = self._parse_month(month)
        dt: datetime.datetime = self._dt
        if new_mth == -1 or new_mth == dt.month:
            return self.to_curr_month(day)  # exit: no adjustment to month
        cur_yer: cython.int = dt.year

        # Calculate new day
        new_day: cython.uint = dt.day if day < 1 else day
        if new_day > 28:
            new_day = min(new_day, cydt.days_in_month(cur_yer, new_mth))

        # Generate
        # fmt: off
        return self._new(datetime.datetime_new(
            cur_yer, new_mth, new_day, dt.hour, dt.minute, 
            dt.second, dt.microsecond, dt.tzinfo, dt.fold )
        )
        # fmt: on

    @cython.ccall
    def to_prev_year(
        self,
        month: cython.int | str | None = None,
        day: cython.int = -1,
    ) -> pydt:
        """Go to specific 'month' and 'day' of the previous year `<'pydt'>`."""
        return self.to_year(-1, month, day)

    @cython.ccall
    def to_next_year(
        self,
        month: cython.int | str | None = None,
        day: cython.int = -1,
    ) -> pydt:
        """Go to specific 'month' and 'day' of the next year `<'pydt'>`."""
        return self.to_year(1, month, day)

    @cython.ccall
    def to_year(
        self,
        offset: cython.int = 0,
        month: cython.int | str | None = None,
        day: cython.int = -1,
    ) -> pydt:
        """Go to specific 'month' and 'day' of the
        current year (+/-) 'offset' `<'pydt'>`."""
        # No offset adjustment
        if offset == 0:
            return self.to_curr_year(month, day)  # exit: current year

        # Calculate new year
        dt: datetime.datetime = self._dt
        year: cython.int = dt.year + offset
        new_yer: cython.uint = min(max(year, 1), 9_999)

        # Parse new month
        new_mth: cython.int = self._parse_month(month)
        if new_mth == -1:
            new_mth = dt.month

        # Calculate new day
        new_day: cython.uint = dt.day if day < 1 else day
        if new_day > 28:
            new_day = min(new_day, cydt.days_in_month(new_yer, new_mth))

        # Generate
        # fmt: off
        return self._new(datetime.datetime_new(
            new_yer, new_mth, new_day, dt.hour, dt.minute, 
            dt.second, dt.microsecond, dt.tzinfo, dt.fold )
        )
        # fmt: on

    # Quarter ---------------------------------------------------------------------------------
    @property
    def days_in_quarter(self) -> int:
        """Get the maximum number of days in the quarter `<'int'>`."""
        return self._c_days_in_quarter()

    @property
    def days_bf_quarter(self) -> int:
        """Get the number of days between the 1st day of
        the year and the 1st day of the quarter `<'int'>`."""
        return self._c_days_bf_quarter()

    @property
    def days_of_quarter(self) -> int:
        """Get the number of days between the 1st day of
        the quarter and the current date `<'int'>`."""
        return self._c_days_of_quarter()

    @property
    def quarter_1st_month(self) -> int:
        """Get the first month of the quarter.
        Expect 1, 4, 7, 10 `<'int'>`."""
        return self._c_quarter_1st_month()

    @property
    def quarter_lst_month(self) -> int:
        """Get the last month of the quarter.
        Expect 3, 6, 9, 12 `<'int'>`."""
        return self._c_quarter_lst_month()

    def is_quarter(self, quarter: cython.int) -> bool:
        """Whether the current quarter is a specific quarter `<'bool'>`."""
        return self._c_is_quarter(quarter)

    def is_quarter_1st(self) -> bool:
        """Whether is the 1st day of the quarter `<'bool'>`."""
        return self._c_is_quarter_1st()

    def is_quarter_lst(self) -> bool:
        """Whether is the last day of the quarter `<'bool'>`."""
        return self._c_is_quarter_lst()

    @cython.ccall
    def to_quarter_1st(self) -> pydt:
        """Go to the 1st day of the current quarter `<'pydt'>`."""
        dt: datetime.datetime = self._dt
        # fmt: off
        return self._new(datetime.datetime_new(
            dt.year, cydt.quarter_1st_month(dt.month), 1, dt.hour, dt.minute, 
            dt.second, dt.microsecond, dt.tzinfo, dt.fold )
        )
        # fmt: on

    @cython.ccall
    def to_quarter_lst(self) -> pydt:
        """Go to the last day of the current quarter `<'pydt'>`."""
        dt: datetime.datetime = self._dt
        cur_yer: cython.uint = dt.year
        new_mth: cython.uint = cydt.quarter_lst_month(dt.month)
        new_day: cython.uint = cydt.days_in_month(cur_yer, new_mth)
        # fmt: off
        return self._new(datetime.datetime_new(
            cur_yer, new_mth, new_day, dt.hour, dt.minute, 
            dt.second, dt.microsecond, dt.tzinfo, dt.fold )
        )
        # fmt: on

    @cython.ccall
    def to_curr_quarter(self, month: cython.int = -1, day: cython.int = -1) -> pydt:
        """Go to specific 'month (of the quarter [1..3])'
        and 'day' of the current quarter `<'pydt'>`."""
        # No adjustment to month
        if month < 1:
            return self.to_curr_month(day)  # exit: no adjustment to month

        # Calculate new month
        dt: datetime.datetime = self._dt
        new_mth: cython.int = month
        new_mth = cydt.quarter_of_month(dt.month) * 3 - 3 + (month % 3 or 3)
        if new_mth == dt.month:
            return self.to_curr_month(day)  # exit: no adjustment to month
        cur_yer: cython.uint = dt.year

        # Calculate new day
        new_day: cython.uint = dt.day if day < 1 else day
        if new_day > 28:
            new_day = min(new_day, cydt.days_in_month(cur_yer, new_mth))

        # Generate
        # fmt: off
        return self._new(datetime.datetime_new(
            cur_yer, new_mth, new_day, dt.hour, dt.minute, 
            dt.second, dt.microsecond, dt.tzinfo, dt.fold )
        )
        # fmt: on

    @cython.ccall
    def to_prev_quarter(self, month: cython.int = -1, day: cython.int = -1) -> pydt:
        """Go to specific 'month (of the quarter [1..3])'
        and 'day' of the previous quarter `<'pydt'>`."""
        return self.to_quarter(-1, month, day)

    @cython.ccall
    def to_next_quarter(self, month: cython.int = -1, day: cython.int = -1) -> pydt:
        """Go to specific 'month (of the quarter [1..3])'
        and 'day' of the next quarter `<'pydt'>`."""
        return self.to_quarter(1, month, day)

    @cython.ccall
    def to_quarter(
        self,
        offset: cython.int = 0,
        month: cython.int = -1,
        day: cython.int = -1,
    ) -> pydt:
        """Go to specific 'month (of the quarter [1..3])'
        and 'day' of the current quarter (+/-) 'offset' `<'pydt'>`."""
        # No offset adjustment
        if offset == 0:
            return self.to_curr_quarter(month, day)  # exit: current quarter

        # Calculate new year & month
        dt: datetime.datetime = self._dt
        new_yer: cython.int = dt.year
        if month < 1:
            new_mth: cython.int = dt.month
        else:
            new_mth: cython.int = (
                cydt.quarter_of_month(dt.month) * 3 - 3 + (month % 3 or 3)
            )
        new_mth += offset * 3
        if new_mth < 1:
            while new_mth < 1:
                new_mth += 12
                new_yer -= 1
        elif new_mth > 12:
            while new_mth > 12:
                new_mth -= 12
                new_yer += 1
        new_yer = min(max(new_yer, 1), 9_999)

        # Calculate new day
        new_day: cython.uint = dt.day if day < 1 else day
        if new_day > 28:
            new_day = min(new_day, cydt.days_in_month(new_yer, new_mth))

        # Generate
        # fmt: off
        return self._new(datetime.datetime_new(
            new_yer, new_mth, new_day, dt.hour, dt.minute, 
            dt.second, dt.microsecond, dt.tzinfo, dt.fold)
        )
        # fmt: on

    # Month -----------------------------------------------------------------------------------
    @property
    def days_in_month(self) -> int:
        """Get the maximum number of days in the month `<'int'>`."""
        return self._c_days_in_month()

    @property
    def days_bf_month(self) -> int:
        """Get the number of days between the 1st day of
        the year and the 1st day of the month `<'int'>`."""
        return self._c_days_bf_month()

    def is_month(self, month: int | str) -> bool:
        """Whether the current month is a specific 'month' `<'bool'>`."""
        return self._c_is_month(month)

    def is_month_1st(self) -> bool:
        """Whether is the 1st day of the month `<'bool'>`."""
        return self._c_is_month_1st()

    def is_month_lst(self) -> bool:
        """Whether is the last day of the current month `<'bool'>`."""
        return self._c_is_month_lst()

    @cython.ccall
    def to_month_1st(self) -> pydt:
        """Go to the 1st day of the current month `<'pydt'>`."""
        dt: datetime.datetime = self._dt
        # fmt: off
        return self._new(datetime.datetime_new(
            dt.year, dt.month, 1, dt.hour, dt.minute, 
            dt.second, dt.microsecond, dt.tzinfo, dt.fold )
        )
        # fmt: on

    @cython.ccall
    def to_month_lst(self) -> pydt:
        """Go to the last day of the current month `<'pydt'>`."""
        dt: datetime.datetime = self._dt
        # fmt: off
        return self._new(datetime.datetime_new(
                dt.year, dt.month, cydt.days_in_month(dt.year, dt.month), 
                dt.hour, dt.minute, dt.second, dt.microsecond, dt.tzinfo, dt.fold )
        )
        # fmt: on

    @cython.ccall
    def to_curr_month(self, day: cython.int = -1) -> pydt:
        """Go to specific 'day' of the current month `<'pydt'>`."""
        # No adjustment to day
        if day < 1:
            return self  # exit

        # Clip by max days in month
        new_day: cython.uint = day
        dt: datetime.datetime = self._dt
        if new_day > 28:
            new_day = min(new_day, cydt.days_in_month(dt.year, dt.month))

        # Compare with current day
        cur_day: cython.uint = dt.day
        if new_day == cur_day:
            return self  # exit: same day

        # Generate
        # fmt: off
        return self._new(datetime.datetime_new(
            dt.year, dt.month, new_day, dt.hour, dt.minute, 
            dt.second, dt.microsecond, dt.tzinfo, dt.fold )
        )
        # fmt: on

    @cython.ccall
    def to_prev_month(self, day: cython.int = -1) -> pydt:
        """Go to specific day of the previous month `<'pydt'>`."""
        return self.to_month(-1, day)

    @cython.ccall
    def to_next_month(self, day: cython.int = -1) -> pydt:
        """Go to specific day of the next month `<'pydt'>`."""
        return self.to_month(1, day)

    @cython.ccall
    def to_month(self, offset: cython.int = 0, day: cython.int = -1) -> pydt:
        """Go to specific 'day' of the current
        month (+/-) 'offset' `<'pydt'>`."""
        # No offset adjustment
        if offset == 0:
            return self.to_curr_month(day)  # exit: current month

        # Calculate new year & month
        dt: datetime.datetime = self._dt
        new_yer: cython.int = dt.year
        new_mth: cython.int = dt.month + offset
        if new_mth < 1:
            while new_mth < 1:
                new_mth += 12
                new_yer -= 1
        elif new_mth > 12:
            while new_mth > 12:
                new_mth -= 12
                new_yer += 1
        new_yer = min(max(new_yer, 1), 9_999)

        # Calculate new day
        new_day: cython.uint = dt.day if day < 1 else day
        if new_day > 28:
            new_day = min(new_day, cydt.days_in_month(new_yer, new_mth))

        # Generate
        # fmt: off
        return self._new(datetime.datetime_new(
            new_yer, new_mth, new_day, dt.hour, dt.minute, 
            dt.second, dt.microsecond, dt.tzinfo, dt.fold )
        )
        # fmt: on

    # Weekday ---------------------------------------------------------------------------------
    @property
    def weekday(self) -> int:
        """The weekday of the datetime `<'int'>`.
        Values: 0=Monday..6=Sunday."""
        return self._c_weekday()

    @property
    def isoweekday(self) -> int:
        """The ISO calendar weekday of the datetime `<'int'>`.
        Values: 1=Monday..7=Sunday."""
        return self._c_isoweekday()

    @property
    def isoweek(self) -> int:
        """Get the ISO calendar week number `<'int'>`."""
        return self._c_isoweek()

    @property
    def isoyear(self) -> int:
        """Get the ISO calendar year `<'int'>`."""
        return self._c_isoyear()

    @property
    def isocalendar(self) -> dict[str, int]:
        """Get the ISO calendar of the current date `<dict>`."""
        return self._c_isocalendar()

    def is_weekday(self, weekday: int | str) -> bool:
        """Whether the current weekday is a specific 'weekday' `<'bool'>`."""
        return self._c_is_weekday(weekday)

    @cython.ccall
    def to_monday(self) -> pydt:
        """Go to Monday of the current week `<'pydt'>`."""
        return self._c_to_curr_weekday(0)

    @cython.ccall
    def to_tuesday(self) -> pydt:
        """Go to Tuesday of the current week `<'pydt'>`."""
        return self._c_to_curr_weekday(1)

    @cython.ccall
    def to_wednesday(self) -> pydt:
        """Go to Wednesday of the current week `<'pydt'>`."""
        return self._c_to_curr_weekday(2)

    @cython.ccall
    def to_thursday(self) -> pydt:
        """Go to Thursday of the current week `<'pydt'>`."""
        return self._c_to_curr_weekday(3)

    @cython.ccall
    def to_friday(self) -> pydt:
        """Go to Friday of the current week `<'pydt'>`."""
        return self._c_to_curr_weekday(4)

    @cython.ccall
    def to_saturday(self) -> pydt:
        """Go to Saturday of the current week `<'pydt'>`."""
        return self._c_to_curr_weekday(5)

    @cython.ccall
    def to_sunday(self) -> pydt:
        """Go to Sunday of the current week `<'pydt'>`."""
        return self._c_to_curr_weekday(6)

    @cython.ccall
    def to_curr_weekday(self, weekday: cython.int | str | None = None) -> pydt:
        """Go to specific 'weekday' of the current week `<'pydt'>`."""
        # Validate weekday
        dt: datetime.datetime = self._dt
        new_wkd: cython.int = self._parse_weekday(weekday)
        cur_wkd: cython.int = cydt.ymd_weekday(dt.year, dt.month, dt.day)
        if new_wkd == -1 or new_wkd == cur_wkd:
            return self  # exit: same weekday

        # Generate
        delta: cython.int = new_wkd - cur_wkd
        return self._new(cydt.dt_add(dt, days=delta))

    @cython.ccall
    def to_prev_weekday(self, weekday: int | str | None = None) -> pydt:
        """Go to specific 'weekday' of the previous week `<'pydt'>`."""
        return self.to_weekday(-1, weekday)

    @cython.ccall
    def to_next_weekday(self, weekday: int | str | None = None) -> pydt:
        """Go to specific 'weekday' of the next week `<'pydt'>`."""
        return self.to_weekday(1, weekday)

    @cython.ccall
    def to_weekday(
        self,
        offset: cython.int = 0,
        weekday: cython.int | str | None = None,
    ) -> pydt:
        """Go to specific 'weekday' of the current
        week (+/-) 'offset' `<'pydt'>`."""
        # No offset adjustment
        if offset == 0:
            return self.to_curr_weekday(weekday)  # exit: current weekday

        # Validate weekday & calculate delta
        new_wkd: cython.int = self._parse_weekday(weekday)
        dt: datetime.datetime = self._dt
        cur_wkd: cython.int = cydt.ymd_weekday(dt.year, dt.month, dt.day)
        if new_wkd == -1:
            delta: cython.int = offset * 7
        else:
            delta: cython.int = new_wkd - cur_wkd
            delta += offset * 7

        # Generate
        return self._new(cydt.dt_add(dt, delta))

    # Day -------------------------------------------------------------------------------------
    def is_day(self, day: cython.int) -> bool:
        """Whether the current day is a specific 'day' `<'bool'>`."""
        return self._c_is_day(day)

    @cython.ccall
    def to_yesterday(self) -> pydt:
        """Go to the previous day `<'pydt'>`."""
        return self.to_day(-1)

    @cython.ccall
    def to_tomorrow(self) -> pydt:
        """Go to the next day `<'pydt'>`."""
        return self.to_day(1)

    @cython.ccall
    def to_day(self, offset: cython.int = 0) -> pydt:
        """Go to the current day (+/-) 'offset' `<'pydt'>`."""
        return self if offset == 0 else self._new(cydt.dt_add(self._dt, offset, 0, 0))

    # Time ------------------------------------------------------------------------------------
    def is_time_start(self) -> bool:
        """Whether the current time is the
        start of time (00:00:00.000000) `<'bool'>`."""
        return self._c_is_time_start()

    def is_time_end(self) -> bool:
        """Whether the current time is the
        end of time (23:59:59.999999) `<'bool'>`."""
        return self._c_is_time_end()

    @cython.ccall
    def to_time_start(self) -> pydt:
        """Go to the start of time (00:00:00.000000)
        of the current datetime `<'pydt'>`."""
        dt: datetime.datetime = self._dt
        # fmt: off
        return self._new(datetime.datetime_new(
            dt.year, dt.month, dt.day, 0, 0, 0, 0, dt.tzinfo, dt.fold )
        )
        # fmt: on

    @cython.ccall
    def to_time_end(self) -> pydt:
        """Go to the end of time (23:59:59.999999)
        of the time `<'pydt'>`."""
        dt: datetime.datetime = self._dt
        # fmt: off
        return self._new(datetime.datetime_new(
            dt.year, dt.month, dt.day, 23, 59, 59, 999_999, dt.tzinfo, dt.fold )
        )
        # fmt: on

    @cython.ccall
    def to_time(
        self,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        millisecond: cython.int = -1,
        microsecond: cython.int = -1,
    ) -> pydt:
        """Go to specific 'hour', 'minute', 'second', 'millisecond'
        and 'microsecond' of the current time `<'pydt'>`."""
        dt: datetime.datetime = self._dt
        microsecond = combine_ms_us(millisecond, microsecond)
        # fmt: off
        return self._new(datetime.datetime_new(
            dt.year, dt.month, dt.day,
            min(hour, 23) if hour >= 0 else dt.hour,
            min(minute, 59) if minute >= 0 else dt.minute,
            min(second, 59) if second >= 0 else dt.second,
            microsecond if microsecond >= 0 else dt.microsecond,
            dt.tzinfo, dt.fold )
        )
        # fmt: on

    # Timezone --------------------------------------------------------------------------------
    @property
    def tz_available(self) -> set[str]:
        """Access all the available timezone names `<set[str]>`."""
        return TIMEZONE_AVAILABLE

    @cython.ccall
    def tz_localize(self, tz: str | datetime.tzinfo | None = None) -> pydt:
        """Localize to a specific 'tz' timezone `<'pydt'>`.

        Equivalent to `datetime.replace(tzinfo=tz<tzinfo>)`.

        ### Notice
        Param 'tz' accept both string (timezone name) and tzinfo
        instance. However, timezone from `pytz` library should not
        be used, and could yield incorrect result.
        """
        # Parse timezone
        tzinfo = parse_tzinfo(tz)  # type: ignore
        dt: datetime.datetime = self._dt
        if tzinfo is dt.tzinfo:
            return self  # exit: same timezone
        # Localize timezone
        return self._new(cydt.dt_replace_tzinfo(dt, tzinfo))

    @cython.ccall
    def tz_convert(self, tz: str | datetime.tzinfo | None = None) -> pydt:
        """Convert to a specific 'tz' timezone `<'pydt'>`.

        Equivalent to `datetime.astimezone(tz<tzinfo>)`.

        ### Notice
        Param 'tz' accept both string (timezone name) and tzinfo
        instance. However, timezone from `pytz` library should not
        be used, and could yield incorrect result.
        """
        # Parse timezone
        tzinfo = parse_tzinfo(tz)  # type: ignore
        dt: datetime.datetime = self._dt
        if tzinfo is not None and tzinfo is dt.tzinfo:
            return self  # exit: same timezone
        # Convert timezone
        return self._new(cydt.dt_astimezone(dt, tzinfo))

    @cython.ccall
    def tz_switch(
        self,
        targ_tz: str | datetime.tzinfo | None,
        base_tz: str | datetime.tzinfo | None = None,
        naive: bool = False,
    ) -> pydt:
        """Switch to 'targ_tz' timezone from 'base_tz' timezone `<'pydt'>`.

        :param targ_tz `<str/tzinfo/None>`: The target timezone to convert.
        :param base_tz `<str/tzinfo/None>`: The base timezone to localize. Defaults to `None`.
        :param naive `<'bool'>`: Whether to return as timezone-naive. Defaults to `False`.
        :return `<'pydt'>`: pydt after switch of timezone.

        ### Explaination
        - If pydt is timezone-aware, 'base_tz' will be ignored, and only performs
          the convertion to the 'targ_tz' (datetime.astimezone(targ_tz <tzinfo>)).

        - If pydt is timezone-naive, first will localize to the given 'base_tz',
          then convert to the 'targ_tz'. In this case, the 'base_tz' must be
          specified, else it is ambiguous on how to convert to the target timezone.

        ### Notice
        Param 'targ_tz' & 'base_tz' accept both string (timezone name)
        and tzinfo instance. However, timezone from `pytz` library should
        not be used, and could yield incorrect result.
        """
        # Pydt is timezone-aware
        targ_tz = parse_tzinfo(targ_tz)  # type: ignore
        dt: datetime.datetime = self._dt
        curr_tz = dt.tzinfo
        nai: cython.bint = bool(naive)
        if curr_tz is not None:
            # . current == target timezone
            if curr_tz is targ_tz:
                if nai:
                    dt = cydt.dt_replace_tzinfo(dt, None)
                else:
                    return self  # exit: not action
            # . current => target timezone
            else:
                dt = cydt.dt_astimezone(dt, targ_tz)
                if nai:
                    dt = cydt.dt_replace_tzinfo(dt, None)
            # . generate
            return self._new(dt)

        # Pydt is timezone-naive
        if isinstance(base_tz, str) or datetime.PyTZInfo_Check(base_tz):
            base_tz = parse_tzinfo(base_tz)  # type: ignore
            # . base == target timezone
            if targ_tz is base_tz:
                if not nai:
                    dt = cydt.dt_replace_tzinfo(dt, targ_tz)
                else:
                    return self  # exit: not action
            # . localize to base & base => target
            else:
                dt = cydt.dt_replace_tzinfo(dt, base_tz)
                dt = cydt.dt_astimezone(dt, targ_tz)
                if nai:
                    dt = cydt.dt_replace_tzinfo(dt, None)
            # . generate
            return self._new(dt)

        # Invalid
        raise errors.InvalidTimezoneError(
            "<'%s'>\nCannot switch timezone-naive <'pydt'> without "
            "a valid 'base_tz'." % (self.__class__.__name__)
        )

    @cython.ccall
    def astimezone(self, tz: str | datetime.tzinfo | None = None) -> pydt:
        """Convert to a specific 'tz' timezone `<'pydt'>`.

        Equivalent to `datetime.astimezone(tz<tzinfo>)`.
        Alias of `pydt.tz_convert()`.

        ### Notice
        Param 'tz' accept both string (timezone name) and tzinfo
        instance. However, timezone from `pytz` library should not
        be used, and could yield incorrect result.
        """
        # Parse timezone
        tzinfo = parse_tzinfo(tz)  # type: ignore
        dt: datetime.datetime = self._dt
        if tzinfo is not None and tzinfo is dt.tzinfo:
            return self  # exit: same timezone
        # Convert timezone
        return self._new(cydt.dt_astimezone(dt, tzinfo))

    # Frequency -------------------------------------------------------------------------------
    @cython.ccall
    def freq_round(self, freq: Literal["D", "h", "m", "s", "ms", "us"]) -> pydt:
        """Perform round operation to specified freqency `<'pydt'>`.
        Similar to `pandas.DatetimeIndex.round()`.

        :param freq: `<'str'>` frequency to round to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond
        """
        # Parse frequency
        f_num: cython.longlong = self._parse_frequency(freq)
        if f_num == 1:
            return self  # exit: no action
        # Round frequency
        dt: datetime.datetime = self._dt
        us: cython.longlong = cydt.dt_to_microseconds(dt)
        us = int(roundl(us / f_num))
        us *= f_num
        # Generate
        return self._new(cydt.dt_fr_microseconds(us, dt.tzinfo, dt.fold))

    @cython.ccall
    def freq_ceil(self, freq: Literal["D", "h", "m", "s", "ms", "us"]) -> pydt:
        """Perform ceil operation to specified freqency `<'pydt'>`.
        Similar to `pandas.DatetimeIndex.ceil()`.

        :param freq: `<'str'>` frequency to ceil to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond
        """
        # Parse frequency
        f_num: cython.longlong = self._parse_frequency(freq)
        if f_num == 1:
            return self  # exit: no action
        # Ceil frequency
        dt: datetime.datetime = self._dt
        us: cython.longlong = cydt.dt_to_microseconds(dt)
        us = int(ceill(us / f_num))
        us *= f_num
        # Generate
        return self._new(cydt.dt_fr_microseconds(us, dt.tzinfo, dt.fold))

    @cython.ccall
    def freq_floor(self, freq: Literal["D", "h", "m", "s", "ms", "us"]) -> pydt:
        """Perform floor operation to specified freqency `<'pydt'>`.
        Similar to `pandas.DatetimeIndex.floor()`.

        :param freq: `<'str'>` frequency to floor to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond
        """
        # Parse frequency
        f_num: cython.longlong = self._parse_frequency(freq)
        if f_num == 1:
            return self  # exit: no action
        # Floor frequency
        dt: datetime.datetime = self._dt
        us: cython.longlong = cydt.dt_to_microseconds(dt)
        us = int(floorl(us / f_num))
        us *= f_num
        # Generate
        return self._new(cydt.dt_fr_microseconds(us, dt.tzinfo, dt.fold))

    # Delta -----------------------------------------------------------------------------------
    @cython.ccall
    def add_delta(
        self,
        years: cython.int = 0,
        months: cython.int = 0,
        days: cython.int = 0,
        weeks: cython.int = 0,
        hours: cython.int = 0,
        minutes: cython.int = 0,
        seconds: cython.int = 0,
        milliseconds: cython.int = 0,
        microseconds: cython.int = 0,
    ) -> pydt:
        """Add 'timedelta' to the current `<'pydt'>`.

        Equivalent to `pydt/datetime + cytimes.cytimedelta`.
        For more information, please refer to `<cytimedelta>`.

        ### Relative delta
        :param years `<'int'>`: The relative delta of years. Defaults to `0`.
        :param months `<'int'>`: The relative delta of months. Defaults to `0`.
        :param days `<'int'>`: The relative delta of days. Defaults to `0`.
        :param weeks `<'int'>`: The relative delta of weeks. Defaults to `0`.
        :param hours `<'int'>`: The relative delta of hours. Defaults to `0`.
        :param minutes `<'int'>`: The relative delta of minutes. Defaults to `0`.
        :param seconds `<'int'>`: The relative delta of seconds. Defaults to `0`.
        :param milliseconds `<'int'>`: The relative delta of milliseconds. Defaults to `0`.
        :param microseconds `<'int'>`: The relative delta of microseconds. Defaults to `0`.
        """
        # fmt: off
        return self._new(self._dt + cytimedelta(
            years, months, days, weeks, hours, minutes, 
            seconds, milliseconds, microseconds )
        )
        # fmt: on

    @cython.ccall
    def sub_delta(
        self,
        years: cython.int = 0,
        months: cython.int = 0,
        days: cython.int = 0,
        weeks: cython.int = 0,
        hours: cython.int = 0,
        minutes: cython.int = 0,
        seconds: cython.int = 0,
        milliseconds: cython.int = 0,
        microseconds: cython.int = 0,
    ) -> pydt:
        """Substract 'timedelta' to the current `<'pydt'>`.

        Equivalent to `pydt/datetime - cytimes.cytimedelta`.
        For more information, please refer to `<cytimedelta>`.

        ### Relative delta
        :param years `<'int'>`: The relative delta of years. Defaults to `0`.
        :param months `<'int'>`: The relative delta of months. Defaults to `0`.
        :param days `<'int'>`: The relative delta of days. Defaults to `0`.
        :param weeks `<'int'>`: The relative delta of weeks. Defaults to `0`.
        :param hours `<'int'>`: The relative delta of hours. Defaults to `0`.
        :param minutes `<'int'>`: The relative delta of minutes. Defaults to `0`.
        :param seconds `<'int'>`: The relative delta of seconds. Defaults to `0`.
        :param milliseconds `<'int'>`: The relative delta of milliseconds. Defaults to `0`.
        :param microseconds `<'int'>`: The relative delta of microseconds. Defaults to `0`.
        """
        # fmt: off
        return self._new(self._dt - cytimedelta(
            years, months, days, weeks, hours, minutes, 
            seconds, milliseconds, microseconds )
        )
        # fmt: on

    def cal_delta(
        self,
        other: str | datetime.date | datetime.datetime | pydt,
        unit: Literal["Y", "M", "W", "D", "h", "m", "s", "ms", "us"],
        inclusive: Literal["one", "both", "neither"] = "both",
    ) -> cython.longlong:
        """Calcuate the `ABSOLUTE` delta between the current pydt
        and the given object based on the specified 'unit' `<'int'>`.

        :param other `<'str/date/datetime/pydt'>`: The target object.
        :param unit `<'str'>`: The specific time unit to calculate the delta.
        :param inclusive `<'str'>`: Include boundaries, whether to set bounds as closed or open. Defaults to `'both'`.
        :return `<'int'>`: The `ABSOLUTE` delta value.
        """
        return self._c_cal_delta(other, unit, inclusive)

    # Replace ---------------------------------------------------------------------------------
    def replace(
        self,
        year: cython.int = -1,
        month: cython.int = -1,
        day: cython.int = -1,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        millisecond: cython.int = -1,
        microsecond: cython.int = -1,
        tzinfo: str | datetime.tzinfo | None = -1,
        fold: cython.int = -1,
    ) -> pydt:
        """Replacement for the current `<'pydt'>`.

        Equivalent to `datetime.replace()`.

        :param year `<'int'>`: The absolute year value. Defaults to `-1 (no change)`.
        :param month `<'int'>`: The absolute month value. Defaults to `-1 (no change)`.
        :param day `<'int'>`: The absolute day value. Defaults to `-1 (no change)`.
        :param hour `<'int'>`: The absolute hour value. Defaults to `-1 (no change)`.
        :param minute `<'int'>`: The absolute minute value. Defaults to `-1 (no change)`.
        :param second `<'int'>`: The absolute second value. Defaults to `-1 (no change)`.
        :param millisecond `<'int'>`: The absolute millisecond value. Defaults to `-1 (no change)`.
        :param microsecond `<'int'>`: The absolute microsecond value. Defaults to `-1 (no change)`.
        :param tzinfo `<str/tzinfo/None>`: The timezone name or instance. Defaults to `-1 (no change)`.
        :param fold `<'int'>`: The ambiguous timezone fold value. Defaults to `-1 (no change)`.
        :return `<'pydt'>`: pydt after replacement.
        """
        # fmt: off
        if tzinfo != -1:
            tzinfo = parse_tzinfo(tzinfo) # type: ignore
        return self._new(cydt.dt_replace(
            self._dt, year, month, day, hour, minute, second, 
            combine_ms_us(millisecond, microsecond), tzinfo, fold ) # type: ignore
        )
        # fmt: on

    # Format ----------------------------------------------------------------------------------
    def strftime(self, format: str) -> str:
        """Convert to string with the given 'format' `<'str'>`.

        Equivalent to `datetime.strftime(format)`."""
        return cydt.dt_to_strformat(self._dt, format)

    # Internal methods ------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _new(self, dtobj: object) -> pydt:
        """(cfunc) Generate new `<'pydt'>`."""
        # fmt: off
        return pydt(
            dtobj, self._default, self._day1st, self._year1st, 
            self._ignoretz, self._isoformat, self._cfg,
        )
        # fmt: on

    @cython.cfunc
    @cython.inline(True)
    def _parse_dtobj(self, dtobj: object, default: object) -> datetime.datetime:
        """(cfunc) Parse 'dtobj' into `<'datetime.datetime'>`."""
        # <'datetime.datetime'>
        if datetime.PyDateTime_CheckExact(dtobj):
            return dtobj
        # <'str'> datetime string to be parsed
        if isinstance(dtobj, str):
            try:
                return Parser(self._cfg).parse(
                    dtobj,
                    default,
                    self._day1st,
                    self._year1st,
                    self._ignoretz,
                    self._isoformat,
                )
            except Exception as err:
                raise errors.InvalidDatetimeObjectError(
                    "<'%s'> %s." % (self.__class__.__name__, err)
                ) from err
        # <'datetime.date'> subclass
        if datetime.PyDate_Check(dtobj):
            if datetime.PyDateTime_Check(dtobj):
                return cydt.dt_fr_dt(dtobj)
            else:
                return cydt.dt_fr_date(dtobj, None, 0)
        # <'pydt'>
        if isinstance(dtobj, pydt):
            return access_pydt(dtobj)  # type: ignore
        # <'None'>
        if dtobj is None:
            return cydt.gen_dt_now()
        # <'numpy.datetime64'>
        if np.is_datetime64_object(dtobj):
            return cydt.dt64_to_dt(dtobj)
        # Invalid
        raise errors.InvalidDatetimeObjectError(
            "<'%s'>\nUnsupported 'dtobj' type: %s %s."
            % (self.__class__.__name__, type(dtobj), repr(dtobj))
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-2, check=False)
    def _parse_month(self, month: object) -> cython.int:
        """(cfunc) Prase 'month' into numeric value `<'int'>`.

        Accepts both integer or string. Returns -1 if the 'month'
        is None or -1, else raise `InvalidMonthError`."""
        # <'NoneType'>
        if month is None:
            return -1  # exit

        # <'str'>
        if isinstance(month, str):
            mth: str = month
            val = dict_getitem(
                CONFIG_MONTH if self._cfg is None else self._cfg._month,
                mth.lower(),
            )
            if val == cython.NULL:
                raise errors.InvalidMonthError(
                    "<'%s'>\nFailed to parse 'month' string: '%s'."
                    % (self.__class__.__name__, month)
                )
            return cython.cast(object, val)  # exit

        # <'int'>
        if isinstance(month, int):
            try:
                num: cython.int = month
                if num == -1:
                    return -1  # exit
                if not 1 <= num <= 12:
                    raise ValueError
                return num  # exit
            except Exception as err:
                raise errors.InvalidMonthError(
                    "<'%s'>\nInvalid 'month' value '%d', must between "
                    "1(Jan)..12(Dec)." % (self.__class__.__name__, month)
                ) from err

        # Incorrect data type
        raise errors.InvalidMonthError(
            "<'%s'>\nInvalid 'month' data type: %s %s."
            % (self.__class__.__name__, type(month), repr(month))
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-2, check=False)
    def _parse_weekday(self, weekday: object) -> cython.int:
        """(cfunc) Prase 'weekday' into numeric value `<'int'>`.

        Accepts both integer or string. Returns -1 if the 'weekday'
        is None or -1, else raise `InvalidWeekdayError`."""
        # <'NoneType'>
        if weekday is None:
            return -1  # exit

        # <'str'>
        if isinstance(weekday, str):
            wkd: str = weekday
            val = dict_getitem(
                CONFIG_WEEKDAY if self._cfg is None else self._cfg._weekday,
                wkd.lower(),
            )
            if val == cython.NULL:
                raise errors.InvalidWeekdayError(
                    "<'%s'>\nFailed to parse 'weekday' string: '%s'."
                    % (self.__class__.__name__, weekday)
                )
            return cython.cast(object, val)  # exit

        # <'int'>
        if isinstance(weekday, int):
            try:
                num: cython.int = weekday
                if num == -1:
                    return num  # exit
                if not 0 <= num <= 6:
                    raise ValueError
                return num  # exit
            except Exception as err:
                raise errors.InvalidWeekdayError(
                    "<'%s'>\nInvalid 'weekday' value '%d', must between "
                    "0(Monday)..6(Sunday)." % (self.__class__.__name__, weekday)
                ) from err

        # Incorrect data type
        raise errors.InvalidWeekdayError(
            "<'%s'>\nInvalid 'weekday' data type: %s %s."
            % (self.__class__.__name__, type(weekday), repr(weekday))
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_frequency(self, freq: object) -> cython.longlong:
        """(cfunc) Prase 'freq' into coresponding frequency `<'int'>`."""
        # Match frequency
        if freq == "D":
            return cydt.US_DAY
        if freq == "h":
            return cydt.US_HOUR
        if freq == "m":
            return 60_000_000
        if freq == "s":
            return 1_000_000
        if freq == "ms":
            return 1_000
        if freq == "us":
            return 1
        raise errors.InvalidFrequencyError(
            "<'%s'>\nInvalid 'freq' input: %s. Must be one of the following <'str'>: "
            "['D', 'h', 'm', 's', 'ms', 'us']." % (self.__class__.__name__, repr(freq))
        )

    # Special methods: addition ---------------------------------------------------------------
    def __add__(self, o: object) -> pydt:
        # . timedelta
        if isinstance(o, (datetime.timedelta, cytimedelta, typeref.RELATIVEDELTA)):
            return self._new(self._dt + o)
        # . uncommon
        if np.is_timedelta64_object(o):
            return self._new(self._dt + cydt.td64_to_td(o))
        # . unsupported
        return NotImplemented

    def __radd__(self, o: object) -> pydt:
        # . timedelta
        if isinstance(o, (datetime.timedelta, cytimedelta, typeref.RELATIVEDELTA)):
            return self._new(self._dt + o)
        # . uncommon
        # TODO: this will not work since numpy does not return NotImplemented
        if np.is_timedelta64_object(o):
            return self._new(self._dt + cydt.td64_to_td(o))
        # . unsupported
        return NotImplemented

    # Special methods: substraction -----------------------------------------------------------
    def __sub__(self, o: object) -> datetime.timedelta | pydt:
        # . timedelta
        if isinstance(o, (datetime.timedelta, cytimedelta, typeref.RELATIVEDELTA)):
            return self._new(self._dt - o)
        # . datetime
        if isinstance(o, pydt):
            return cydt.dt_sub_dt(self._dt, access_pydt(o))  # type: ignore
        if datetime.PyDateTime_Check(o):
            return cydt.dt_sub_dt(self._dt, o)
        if datetime.PyDate_Check(o):
            return cydt.dt_sub_dt(self._dt, cydt.dt_fr_date(o, None, 0))
        if isinstance(o, str):
            return cydt.dt_sub_dt(self._dt, self._parse_dtobj(o, self._default))
        # . uncommmon
        if np.is_datetime64_object(o):
            return cydt.dt_sub_dt(self._dt, cydt.dt64_to_dt(o))
        if np.is_timedelta64_object(o):
            return self._new(self._dt - cydt.td64_to_td(o))
        # . unsupported
        return NotImplemented

    def __rsub__(self, o: object) -> datetime.timedelta:
        # . datetime
        if datetime.PyDateTime_Check(o):
            return cydt.dt_sub_dt(o, self._dt)
        if datetime.PyDate_Check(o):
            return cydt.dt_sub_dt(cydt.dt_fr_date(o, None, 0), self._dt)
        if isinstance(o, str):
            return cydt.dt_sub_dt(self._parse_dtobj(o, self._default), self._dt)
        # . uncommon
        # TODO this will not work since numpy does not return NotImplemented
        if np.is_datetime64_object(o):
            return cydt.dt_sub_dt(cydt.dt64_to_dt(o), self._dt)
        # . unsupported
        return NotImplemented

    # Special methods: comparison -------------------------------------------------------------
    def __eq__(self, o: object) -> bool:
        if isinstance(o, pydt):
            return self._dt == access_pydt(o)  # type: ignore
        if datetime.PyDateTime_Check(o):
            return self._dt == o
        if datetime.PyDate_Check(o):
            return self._dt == cydt.dt_fr_date(o, None, 0)
        if isinstance(o, str):
            return self._dt == self._parse_dtobj(o, self._default)
        return NotImplemented

    def __ne__(self, o: object) -> bool:
        if isinstance(o, pydt):
            return self._dt != access_pydt(o)  # type: ignore
        if datetime.PyDateTime_Check(o):
            return self._dt != o
        if datetime.PyDate_Check(o):
            return self._dt != cydt.dt_fr_date(o, None, 0)
        if isinstance(o, str):
            return self._dt != self._parse_dtobj(o, self._default)
        return NotImplemented

    def __gt__(self, o: object) -> bool:
        if isinstance(o, pydt):
            return self._dt > access_pydt(o)  # type: ignore
        if datetime.PyDateTime_Check(o):
            return self._dt > o
        if datetime.PyDate_Check(o):
            return self._dt > cydt.dt_fr_date(o, None, 0)
        if isinstance(o, str):
            return self._dt > self._parse_dtobj(o, self._default)
        return NotImplemented

    def __ge__(self, o: object) -> bool:
        if isinstance(o, pydt):
            return self._dt >= access_pydt(o)  # type: ignore
        if datetime.PyDateTime_Check(o):
            return self._dt >= o
        if datetime.PyDate_Check(o):
            return self._dt >= cydt.dt_fr_date(o, None, 0)
        if isinstance(o, str):
            return self._dt >= self._parse_dtobj(o, self._default)
        return NotImplemented

    def __lt__(self, o: object) -> bool:
        if isinstance(o, pydt):
            return self._dt < access_pydt(o)  # type: ignore
        if datetime.PyDateTime_Check(o):
            return self._dt < o
        if datetime.PyDate_Check(o):
            return self._dt < cydt.dt_fr_date(o, None, 0)
        if isinstance(o, str):
            return self._dt < self._parse_dtobj(o, self._default)
        return NotImplemented

    def __le__(self, o: object) -> bool:
        if isinstance(o, pydt):
            return self._dt <= access_pydt(o)  # type: ignore
        if datetime.PyDateTime_Check(o):
            return self._dt <= o
        if datetime.PyDate_Check(o):
            return self._dt <= cydt.dt_fr_date(o, None, 0)
        if isinstance(o, str):
            return self._dt <= self._parse_dtobj(o, self._default)
        return NotImplemented

    # Special methods: represent --------------------------------------------------------------
    def __repr__(self) -> str:
        return "<%s('%s')>" % (self.__class__.__name__, self._c_dt_strtz())

    def __str__(self) -> str:
        return self._c_dt_strtz()

    # Special methods: hash -------------------------------------------------------------------
    def __hash__(self) -> int:
        if self._hashcode == -1:
            dt: datetime.datetime = self._dt
            self._hashcode = hash(
                (
                    "pydt",
                    dt.year,
                    dt.month,
                    dt.day,
                    dt.hour,
                    dt.minute,
                    dt.second,
                    dt.microsecond,
                    dt.tzinfo,
                    dt.fold,
                )
            )
        return self._hashcode

    # C-API: Access ---------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _c_tzname(self) -> str:
        """(cfunc) Access the timezone name `<'str'>`."""
        return cydt.dt_tzname(self._dt)

    @cython.cfunc
    @cython.inline(True)
    def _c_dst(self) -> datetime.timedelta:
        """(cfunc) Access the DST offset `<'datetime.timedelta'>`."""
        return cydt.dt_dst(self._dt)

    @cython.cfunc
    @cython.inline(True)
    def _c_utcoffset(self) -> datetime.timedelta:
        """(cfunc) Access the UTC offset `<'datetime.timedelta'>`."""
        return cydt.dt_utcoffset(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_quarter(self) -> cython.int:
        """(cfunc) Access the quarter of the datetime `<'int'>`."""
        return cydt.quarter_of_month(self._dt.month)

    @cython.cfunc
    @cython.inline(True)
    def _c_dt_str(self) -> str:
        """(cfunc) Access as datetime in string format `<'str'>`."""
        return cydt.dt_to_isospaceformat(self._dt)

    @cython.cfunc
    @cython.inline(True)
    def _c_dt_strtz(self) -> str:
        """(cfunc) Access as datetime in string format with timezone `<'str'>`."""
        return cydt.dt_to_isospaceformat_tz(self._dt)

    @cython.cfunc
    @cython.inline(True)
    def _c_dt_iso(self) -> str:
        """(cfunc) Access as datetime in ISO format `<'str'>`."""
        return cydt.dt_to_isoformat(self._dt)

    @cython.cfunc
    @cython.inline(True)
    def _c_dt_isotz(self) -> str:
        """(cfunc) Access as datetime in ISO format with timezone `<'str'>`."""
        return cydt.dt_to_isoformat_tz(self._dt)

    @cython.cfunc
    @cython.inline(True)
    def _c_date(self) -> datetime.date:
        "(cfunc) Access as `<'datetime.date'>`."
        dt: datetime.datetime = self._dt
        return datetime.date_new(dt.year, dt.month, dt.day)

    @cython.cfunc
    @cython.inline(True)
    def _c_date_iso(self) -> str:
        "(cfunc) Access as date in ISO format `<'str'>`."
        return cydt.date_to_isoformat(self._c_date())

    @cython.cfunc
    @cython.inline(True)
    def _c_time(self) -> datetime.time:
        "(cfunc) Access as `<'datetime.time'>`."
        dt: datetime.datetime = self._dt
        return datetime.time_new(dt.hour, dt.minute, dt.second, dt.microsecond, None, 0)

    @cython.cfunc
    @cython.inline(True)
    def _c_timetz(self) -> datetime.time:
        "(cfunc) Access as `<'datetime.time'>` with timezone."
        dt: datetime.datetime = self._dt
        # fmt: off
        return datetime.time_new(
            dt.hour, dt.minute, dt.second, 
            dt.microsecond, dt.tzinfo, dt.fold,
        )
        # fmt: on

    @cython.cfunc
    @cython.inline(True)
    def _c_time_iso(self) -> str:
        "(cfunc) Access as time in ISO format `<'str'>`."
        return cydt.time_to_isoformat(self._c_time())

    @cython.cfunc
    @cython.inline(True)
    def _c_stime(self) -> cydt.tm:
        """(cfunc) Access as `<'struct:tm'>`."""
        return cydt.dt_to_stime(self._dt)

    @cython.cfunc
    @cython.inline(True)
    def _c_stime_utc(self) -> cydt.tm:
        """(cfunc) Access as `<'struct:tm'>` in UTC."""
        return cydt.dt_to_stime_utc(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_ordinal(self) -> cython.int:
        "(cfunc) Access as ordinal of the date `<'int'>`."
        return cydt.dt_to_ordinal(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _c_seconds(self) -> cython.double:
        """(cfunc) Access in total seconds since EPOCH,
        ignoring the timezone (if exists) `<'float'>`.

        ### Notice
        This should `NOT` be treated as timestamp.
        """
        return cydt.dt_to_seconds(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _c_seconds_utc(self) -> cython.double:
        """(cfunc) Access in total seconds since EPOCH `<'float'>`.
        - If `timezone-aware`, return total seconds in UTC.
        - If `timezone-naive`, requivalent to `pydt.seconds`.

        ### Notice
        This should `NOT` be treated as timestamp.
        """
        return cydt.dt_to_seconds_utc(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _c_microseconds(self) -> cython.longlong:
        """(cfunc) Access in total microseconds since EPOCH,
        ignoring the timezone (if exists) `<'int'>`."""
        return cydt.dt_to_microseconds(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _c_microseconds_utc(self) -> cython.longlong:
        """(cfunc) Access in total microseconds since EPOCH `<'int'>`.
        - If `timezone-aware`, return total microseconds in UTC.
        - If `timezone-naive`, requivalent to `pydt.microseconds`.
        """
        return cydt.dt_to_microseconds_utc(self._dt)

    @cython.cfunc
    @cython.inline(True)
    def _c_timestamp(self) -> cython.double:
        """(cfunc) Access in timestamp `<'float'>`."""
        return cydt.dt_to_timestamp(self._dt)

    # C-API: Year -----------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_is_leapyear(self) -> cython.bint:
        """(cfunc) Whether the current date is a leap year `<'bool'>`."""
        return cydt.is_leapyear(self._dt.year)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_leap_bt_years(self, year: cython.int) -> cython.int:
        """(cfunc) Calculate the number of leap years between
        the current date and the given 'year' `<'int'>`."""
        return cydt.leap_bt_years(self._dt.year, year)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_days_in_year(self) -> cython.int:
        """(cfunc) Get the maximum number of days in the year.
        Expect 365 or 366 (leapyear) `<'int'>`."""
        return cydt.days_in_year(self._dt.year)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_days_bf_year(self) -> cython.int:
        """(cfunc) Get the number of days betweem the 1st day
        of 1AD and the 1st day of the year of the current date `<'int'>`."""
        return cydt.days_bf_year(self._dt.year)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_days_of_year(self) -> cython.int:
        """(cfunc) Get the number of days between the 1st day
        of the year and the current date `<'int'>`."""
        return cydt.dt_days_of_year(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_is_year(self, year: cython.int) -> cython.bint:
        """(cfunc) Whether the current year is a specific year `<'bool'>`."""
        return year == self._dt.year

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_is_year_1st(self) -> cython.bint:
        """(cfunc) Whether is the 1st day of the year `<'bool'>`."""
        dt: datetime.datetime = self._dt
        return dt.month == 1 and dt.day == 1

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_is_year_lst(self) -> cython.bint:
        """(cfunc) Whether is the last day of the current year `<'bool'>`."""
        dt: datetime.datetime = self._dt
        return dt.month == 12 and dt.day == 31

    # C-API: Quarter --------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_days_in_quarter(self) -> cython.int:
        """(cfunc) Get the maximum number of days in the quarter `<'int'>`."""
        return cydt.dt_days_in_quarter(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_days_bf_quarter(self) -> cython.int:
        """(cfunc) Get the number of days between the 1st day of
        the year and the 1st day of the quarter `<'int'>."""
        return cydt.dt_days_bf_quarter(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_days_of_quarter(self) -> cython.int:
        """(cfunc) Get the number of days between the 1st day of
        the quarter and the current date `<'int'>`."""
        return cydt.dt_days_of_quarter(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_quarter_1st_month(self) -> cython.int:
        """(cfunc) Get the first month of the quarter.
        Expect 1, 4, 7, 10 `<'int'>`."""
        return cydt.quarter_1st_month(self._dt.month)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_quarter_lst_month(self) -> cython.int:
        """(cfunc) Get the last month of the quarter.
        Expect 3, 6, 9, 12 `<'int'>`."""
        return cydt.quarter_lst_month(self._dt.month)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_is_quarter(self, quarter: cython.int) -> cython.bint:
        """(cfunc) Whether the current quarter
        is a specific quarter `<'bool'>`."""
        return quarter == self._c_quarter()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_is_quarter_1st(self) -> cython.bint:
        """(cfunc) Whether is the 1st day of the quarter `<'bool'>`."""
        dt: datetime.datetime = self._dt
        return dt.day == 1 and dt.month == cydt.quarter_1st_month(dt.month)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_is_quarter_lst(self) -> cython.bint:
        """(cfunc) Whether is the last day of the quarter `<'bool'>`."""
        dt: datetime.datetime = self._dt
        month: cython.uint = dt.month
        # fmt: off
        return (
            month == cydt.quarter_lst_month(month) 
            and dt.day == cydt.days_in_month(dt.year, month)
        )
        # fmt: on

    # C-API: Month ----------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_days_in_month(self) -> cython.int:
        """(cfunc) Get the maximum number of days in the month `<'int'>`."""
        return cydt.dt_days_in_month(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_days_bf_month(self) -> cython.int:
        """(cfunc) Get the number of days between the 1st day of
        the year and the 1st day of the month `<'int'>`."""
        return cydt.dt_days_bf_month(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_is_month(self, month: object) -> cython.bint:
        """(cfunc) Whether the current month is a specific 'month' `<'bool'>`."""
        return self._parse_month(month) == self._dt.month

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_is_month_1st(self) -> cython.bint:
        """(cfunc) Whether is the 1st day of the month `<'bool'>`."""
        return self._dt.day == 1

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_is_month_lst(self) -> cython.bint:
        """(cfunc) Whether is the last day of the current month `<'bool'>`."""
        dt: datetime.datetime = self._dt
        return dt.day == cydt.days_in_month(dt.year, dt.month)

    # C-API: Weekday --------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_weekday(self) -> cython.int:
        """(cfunc) The weekday of the datetime `<'int'>`.
        Values: 0=Monday..6=Sunday."""
        return cydt.dt_weekday(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_isoweekday(self) -> cython.int:
        """(cfunc) The ISO calendar weekday of the datetime `<'int'>`.
        Values: 1=Monday..7=Sunday."""
        return cydt.dt_isoweekday(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_isoweek(self) -> cython.int:
        """(cfunc) Get the ISO calendar week number `<'int'>`."""
        return cydt.dt_isoweek(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_isoyear(self) -> cython.int:
        """(cfunc) Get the ISO calendar year `<'int'>`."""
        return cydt.dt_isoyear(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _c_isocalendar(self) -> cydt.iso:
        """Get the ISO calendar of the current date `<dict>`."""
        return cydt.dt_isocalendar(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_is_weekday(self, weekday: object) -> cython.bint:
        """(cfunc) Whether the current weekday
        is a specific 'weekday' `<'bool'>`."""
        return self._parse_weekday(weekday) == self._c_weekday()

    @cython.cfunc
    @cython.inline(True)
    def _c_to_curr_weekday(self, weekday: cython.int) -> pydt:
        """(cfunc) Go to specific 'weekday (<'int'>)'
        of the current week `<'pydt'>`."""
        # Validate weekday
        dt: datetime.datetime = self._dt
        cur_wkd: cython.int = cydt.ymd_weekday(dt.year, dt.month, dt.day)
        if weekday == cur_wkd:
            return self  # exit: same weekday

        # Generate
        delta: cython.int = weekday - cur_wkd
        return self._new(cydt.dt_add(dt, days=delta))

    # C-API: Day ------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_is_day(self, day: cython.int) -> cython.bint:
        """(cfunc) Whether the current day is a specific 'day' `<'bool'>`."""
        return day == self._dt.day

    # C-API: Time -----------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_is_time_start(self) -> cython.bint:
        """(cfunc) Whether the current time is the
        start of time (00:00:00.000000) `<'bool'>`."""
        dt: datetime.datetime = self._dt
        # fmt: off
        return dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0
        # fmt: on

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _c_is_time_end(self) -> cython.bint:
        """(cfunc) Whether the current time is the
        end of time (23:59:59.999999) `<'bool'>`."""
        dt: datetime.datetime = self._dt
        return (
            dt.hour == 23
            and dt.minute == 59
            and dt.second == 59
            and dt.microsecond == 999_999
        )

    # C-API: Delta ----------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.cdivision(True)
    @cython.exceptval(-2, check=False)
    def _c_cal_delta(
        self,
        other: object,
        unit: object,
        inclusive: object,
    ) -> cython.longlong:
        """(cfunc) Calcuate the `ABSOLUTE` delta between the current pydt
        and the given object based on the specified 'unit' `<'int'>`.

        :param other `<'str/date/datetime/pydt'>`: The target object.
        :param unit `<'str'>`: The specific time unit to calculate the delta.
        :param inclusive `<'str'>`: Include boundaries, whether to set bounds as closed or open.
        :return `<'int'>`: The `ABSOLUTE` delta value.
        """
        # Parse 'other' to datetime
        o_dt: datetime.datetime = self._parse_dtobj(other, self._default)
        m_dt: datetime.datetime = self._dt

        # Inclusive bounds
        incl: cython.int
        if inclusive == "both":
            incl = 1
        elif inclusive == "one":
            incl = 0
        elif inclusive == "neither":
            incl = -1
        else:
            raise errors.InvalidDeltaInclusiveError(
                "<'%s'>\nInvalid 'inclusive' input: %s. "
                "Must be one of the following <'str'>: ['one', 'both', 'neither']."
                % (self.__class__.__name__, repr(inclusive))
            )

        # Unit: year
        if unit == "Y":
            return abs(m_dt.year - o_dt.year) + incl  # exit
        # Unit: month
        if unit == "M":
            return abs((m_dt.year - o_dt.year) * 12 + (m_dt.month - o_dt.month)) + incl
        # Calculate delta in microseconds
        diff: cython.longlong = cydt.dt_sub_dt_us(m_dt, o_dt)
        delta: cython.longlong = abs(diff)
        # Unit: week
        if unit == "W":
            delta //= cydt.US_DAY
            if diff > 0:
                delta += cydt.ymd_weekday(o_dt.year, o_dt.month, o_dt.day)
            else:
                delta += cydt.ymd_weekday(m_dt.year, m_dt.month, m_dt.day)
            delta //= 7
        # Unit: day
        elif unit == "D":
            delta //= cydt.US_DAY
        # Unit: hour
        elif unit == "h":
            delta //= cydt.US_HOUR
        # Unit: minute
        elif unit == "m":
            delta //= 60_000_000
        # Unit: second
        elif unit == "s":
            delta //= 1_000_000
        # Unit: millisecond
        elif unit == "ms":
            delta //= 1_000
        # Invalid unit: != microsecond
        elif unit != "us":
            raise errors.InvalidDeltaUnitError(
                "<'%s'>\nInvalid delta 'unit' input: %s. "
                "Must be one of the following <'str'>: "
                "['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us']."
                % (self.__class__.__name__, repr(unit))
            )
        # Return delta
        return delta + incl  # exit
