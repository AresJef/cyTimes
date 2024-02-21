# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from __future__ import annotations

# Cython imports
import cython
from cython.cimports.libc import math  # type: ignore
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cpython.dict import PyDict_GetItem as dict_getitem  # type: ignore
from cython.cimports.cytimes import cydatetime as cydt  # type: ignore
from cython.cimports.cytimes.cytimedelta import cytimedelta  # type: ignore
from cython.cimports.cytimes.cyparser import Config, Parser, CONFIG_MONTH, CONFIG_WEEKDAY  # type: ignore

np.import_array()
datetime.import_datetime()

# Python imports
from typing import Literal
import datetime, numpy as np
from zoneinfo import ZoneInfo, available_timezones
from pandas import Timestamp
from dateutil.relativedelta import relativedelta
from cytimes import errors
from cytimes import cydatetime as cydt
from cytimes.cytimedelta import cytimedelta
from cytimes.cyparser import Config, Parser

__all__ = ["pydt"]

# Constants -----------------------------------------------------------------------------------
RLDELTA_DTYPE: object = relativedelta
ZONEINFO_DTYPE: object = ZoneInfo
TIMEZONES_AVAILABLE: set[str] = available_timezones()


# pydt (Python Datetime) ----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def access_pydt_datetime(pt: pydt) -> datetime.datetime:
    return pt._dt


@cython.cfunc
@cython.inline(True)
def parse_tzinfo(tz: object) -> object:
    """Parse 'tz' into `<datetime.tzinfo>`.
    Accept both string (timezone name) and tzinfo instance."""
    if tz is None or cydt.is_tzinfo(tz):
        return tz
    try:
        return ZONEINFO_DTYPE(tz)
    except Exception as err:
        raise errors.InvalidTimezoneError(
            "Failed to create timezone [{}]: {}.".format(repr(tz), err)
        ) from err


@cython.cclass
class pydt:
    """Represents the pydt (Python Datetime) object."""

    # Config
    _cfg: Config
    _default: object
    _day1st: cython.bint
    _year1st: cython.bint
    _ignoretz: cython.bint
    _fuzzy: cython.bint
    # Datetime
    _dt: datetime.datetime
    # Hashcode
    _hashcode: cython.int

    # Class methods ---------------------------------------------------------------------------
    @classmethod
    def now(cls, tz: str | datetime.tzinfo = None) -> pydt:
        """(Class method) Create the current datetime
        (with specific timezone [Optional]) `<pydt>`.

        Equivalent to 'datetime.now(tz)'

        ### Notice
        param 'tz' accept both string (timezone name) and tzinfo
        instance. However, timezone from `pytz` library should not
        be used, and will yield incorrect result.
        """
        # Without a timezone
        if tz is None:
            return cls(cydt.gen_dt_now())

        # With specified timezone
        try:
            tzinfo = parse_tzinfo(tz)
        except Exception as err:
            raise errors.InvalidTimezoneError(f"<{cls.__name__}>\n{err}") from err
        return cls(cydt.gen_dt_now_tz(tzinfo))

    @classmethod
    def from_datetime(
        cls,
        year: cython.int,
        month: cython.int,
        day: cython.int,
        hour: cython.int = 0,
        minute: cython.int = 0,
        second: cython.int = 0,
        microsecond: cython.int = 0,
        tz: str | datetime.tzinfo | None = None,
    ) -> pydt:
        """(Class method) Create from datetime values `<pydt>`.

        ### Notice
        param 'tz' accept both string (timezone name) and tzinfo
        instance. However, timezone from `pytz` library should not
        be used, and will yield incorrect result.
        """
        # Without a timezone
        if tz is None:
            return cls(
                # fmt: off
                cydt.gen_dt(
                    year, month, day, hour, minute, 
                    second, microsecond, None, 0
                )
                # fmt: on
            )

        # With specified timezone
        try:
            tzinfo = parse_tzinfo(tz)
        except Exception as err:
            raise errors.InvalidTimezoneError(f"<{cls.__name__}>\n{err}") from err
        return cls(
            # fmt: off
            cydt.gen_dt(
                year, month, day, hour, minute, 
                second, microsecond, tzinfo, 0)
            # fmt: on
        )

    @classmethod
    def from_ordinal(
        cls,
        ordinal: cython.int,
        tz: str | datetime.tzinfo | None = None,
    ) -> pydt:
        """(Class method) Create from ordinal of a date `<pydt>`.

        ### Notice
        param 'tz' accept both string (timezone name) and tzinfo
        instance. However, timezone from `pytz` library should not
        be used, and will yield incorrect result.
        """
        # Without a timezone
        if tz is None:
            return cls(cydt.dt_fr_ordinal(ordinal, None))

        # With specified timezone
        try:
            tzinfo = parse_tzinfo(tz)
        except Exception as err:
            raise errors.InvalidTimezoneError(f"<{cls.__name__}>\n{err}") from err
        return cls(cydt.dt_fr_ordinal(ordinal, tzinfo))

    @classmethod
    def from_timestamp(
        cls,
        timestamp: int | float,
        tz: str | datetime.tzinfo | None = None,
    ) -> pydt:
        """(Class method) Create from a timestamp `<pydt>`.

        ### Notice
        param 'tz' accept both string (timezone name) and tzinfo
        instance. However, timezone from `pytz` library should not
        be used, and will yield incorrect result.
        """
        # Without a timezone
        if tz is None:
            return cls(cydt.dt_fr_timestamp(timestamp, None))

        # With specified timezone
        try:
            tzinfo = parse_tzinfo(tz)
        except Exception as err:
            raise errors.InvalidTimezoneError(f"<{cls.__name__}>\n{err}") from err
        return cls(cydt.dt_fr_timestamp(timestamp, tzinfo))

    @classmethod
    def from_seconds(
        cls,
        seconds: float,
        tz: str | datetime.tzinfo | None = None,
    ) -> pydt:
        """(Class method) Create from total seconds since EPOCH `<pydt>`.

        ### Notice
        param 'tz' accept both string (timezone name) and tzinfo
        instance. However, timezone from `pytz` library should not
        be used, and will yield incorrect result.
        """
        # Without a timezone
        if tz is None:
            return cls(cydt.dt_fr_seconds(seconds, None))

        # With specified timezone
        try:
            tzinfo = parse_tzinfo(tz)
        except Exception as err:
            raise errors.InvalidTimezoneError(f"<{cls.__name__}>\n{err}") from err
        return cls(cydt.dt_fr_seconds(seconds, tzinfo))

    @classmethod
    def from_microseconds(
        cls,
        microseconds: int,
        tz: str | datetime.tzinfo | None = None,
    ) -> pydt:
        """(Class method) Create from total microseconds since EPOCH `<pydt>`.

        ### Notice
        param 'tz' accept both string (timezone name) and tzinfo
        instance. However, timezone from `pytz` library should not
        be used, and will yield incorrect result.
        """
        # Without a timezone
        if tz is None:
            return cls(cydt.dt_fr_microseconds(microseconds, None))

        # With specified timezone
        try:
            tzinfo = parse_tzinfo(tz)
        except Exception as err:
            raise errors.InvalidTimezoneError(f"<{cls.__name__}>\n{err}") from err
        return cls(cydt.dt_fr_microseconds(microseconds, tzinfo))

    # Initializer -----------------------------------------------------------------------------
    def __init__(
        self,
        timeobj: str | datetime.datetime | datetime.date | None = None,
        default: datetime.datetime | datetime.date | None = None,
        day1st: bool | None = None,
        year1st: bool | None = None,
        ignoretz: bool = False,
        fuzzy: bool = False,
        cfg: Config | None = None,
    ) -> None:
        """"""
        # Conifg
        self._cfg = cfg
        self._default = default
        # fmt: off
        self._day1st = (
            (False if self._cfg is None else self._cfg._day1st) 
            if day1st is None else bool(day1st)
        )
        self._year1st = (
            (False if self._cfg is None else self._cfg._year1st)
            if year1st is None else bool(year1st)
        )
        # fmt: on
        self._ignoretz = bool(ignoretz)
        self._fuzzy = bool(fuzzy)
        # Prase
        self._dt = self._parse_timeobj(timeobj)
        # Hash
        self._hashcode = -1

    # Access ----------------------------------------------------------------------------------
    @property
    def year(self) -> int:
        """Access the year of the datetime `<int>`."""
        return self._capi_year()

    @property
    def quarter(self) -> int:
        """Access the quarter of the datetime `<int>`."""
        return self._capi_quarter()

    @property
    def month(self) -> int:
        """Access the month of the datetime `<int>`."""
        return self._capi_month()

    @property
    def day(self) -> int:
        """Access the day of the datetime `<int>`."""
        return self._capi_day()

    @property
    def hour(self) -> int:
        """Access the hour of the datetime `<int>`."""
        return self._capi_hour()

    @property
    def minute(self) -> int:
        """Access the minute of the datetime `<int>`."""
        return self._capi_minute()

    @property
    def second(self) -> int:
        """Access the second of the datetime `<int>`."""
        return self._capi_second()

    @property
    def microsecond(self) -> int:
        """Access the microsecond of the datetime `<int>`."""
        return self._capi_microsecond()

    @property
    def tzinfo(self) -> datetime.tzinfo | None:
        """Access the timezone of the datetime `<datetime.tzinfo/None>`."""
        return self._capi_tzinfo()

    @property
    def fold(self) -> int:
        """Access the fold of the datetime `<int>`."""
        return self._capi_fold()

    @property
    def dt(self) -> datetime.datetime:
        """Access as '<datetime.datetime>'."""
        return self._dt

    @property
    def dtiso(self) -> str:
        """Access as datetime in ISO format `<str>`."""
        return self._capi_dtiso()

    @property
    def dtisotz(self) -> str:
        """Access as datetime in ISO format with timezone `<str>`."""
        return self._capi_dtisotz()

    @property
    def date(self) -> datetime.date:
        """Access as `<datetime.date>`."""
        return self._capi_date()

    @property
    def dateiso(self) -> str:
        """Access as date in ISO format `<str>`."""
        return self._capi_dateiso()

    @property
    def time(self) -> datetime.time:
        """Access as `<datetime.time>`."""
        return self._capi_time()

    @property
    def timetz(self) -> datetime.time:
        """Access as `<datetime.time>` with timezone."""
        return self._capi_timetz()

    @property
    def timeiso(self) -> str:
        """Access as time in ISO format `<str>`."""
        return self._capi_timeiso()

    @property
    def ts(self) -> Timestamp:
        """Access as `<pandas.Timestamp>`."""
        return Timestamp(self._dt)

    @property
    def dt64(self) -> np.datetime64:
        """Access as `<numpy.datetime64>`."""
        return np.datetime64(cydt.dt_to_microseconds_utc(self._dt), "us")

    @property
    def ordinal(self) -> int:
        """Access as ordinal of the date `<int>`."""
        return self._capi_ordinal()

    @property
    def seconds(self) -> float:
        """Access in total seconds since EPOCH, ignoring
        the timezone (if exists) `<float>`.

        ### Notice
        This should `NOT` be treated as timestamp.
        """
        return self._capi_seconds()

    @property
    def seconds_utc(self) -> float:
        """Access in total seconds since EPOCH `<float>`.
        - If `timezone-aware`, return total seconds in UTC.
        - If `timezone-naive`, requivalent to `pydt.seconds`.

        ### Notice
        This should `NOT` be treated as timestamp.
        """
        return self._capi_seconds_utc()

    @property
    def microseconds(self) -> int:
        """Access in total microseconds since EPOCH, ignoring
        the timezone (if exists) `<int>`."""
        return self._capi_microseconds()

    @property
    def microseconds_utc(self) -> int:
        """Access in total microseconds since EPOCH `<int>`.
        - If `timezone-aware`, return total microseconds in UTC.
        - If `timezone-naive`, requivalent to `pydt.microseconds`.
        """
        return self._capi_microseconds_utc()

    @property
    def timestamp(self) -> float:
        """Access in timestamp `<float>`."""
        return self._capi_timestamp()

    # . c-api - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_year(self) -> cython.uint:
        """(cfunc) Access the year of the datetime `<int>`."""
        return cydt.access_dt_year(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_quarter(self) -> cython.uint:
        """(cfunc) Access the quarter of the datetime `<int>`."""
        return cydt.quarter_of_month(self._capi_month())

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_month(self) -> cython.uint:
        """(cfunc) Access the month of the datetime `<int>`."""
        return cydt.access_dt_month(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_day(self) -> cython.uint:
        """(cfunc) Access the day of the datetime `<int>`."""
        return cydt.access_dt_day(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_hour(self) -> cython.uint:
        """(cfunc) Access the hour of the datetime `<int>`."""
        return cydt.access_dt_hour(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_minute(self) -> cython.uint:
        """(cfunc) Access the minute of the datetime `<int>`."""
        return cydt.access_dt_minute(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_second(self) -> cython.uint:
        """(cfunc) Access the second of the datetime `<int>`."""
        return cydt.access_dt_second(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_microsecond(self) -> cython.uint:
        """(cfunc) Access the microsecond of the datetime `<int>`."""
        return cydt.access_dt_microsecond(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_tzinfo(self) -> datetime.tzinfo:
        """(cfunc) Access the timezone of the datetime `<datetime.tzinfo/None>`."""
        return cydt.access_dt_tzinfo(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_fold(self) -> cython.uint:
        """(cfunc) Access the fold of the datetime `<int>`."""
        return cydt.access_dt_fold(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_dt(self) -> datetime.datetime:
        """(cfunc) Access as '<datetime.datetime>'."""
        return self._dt

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_dtiso(self) -> str:
        "(cfunc) Access as datetime in ISO format `<str>`."
        return cydt.dt_to_isoformat(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_dtisotz(self) -> str:
        "(cfunc) Access as datetime in ISO format with timezone `<str>`."
        return cydt.dt_to_isoformat_tz(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_date(self) -> datetime.date:
        "(cfunc) Access as `<datetime.date>`."
        return cydt.gen_date(self._capi_year(), self._capi_month(), self._capi_day())

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_dateiso(self) -> str:
        "(cfunc) Access as date in ISO format `<str>`."
        return cydt.date_to_isoformat(self._capi_date())

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_time(self) -> datetime.time:
        "(cfunc) Access as `<datetime.time>`."
        return cydt.gen_time(
            self._capi_hour(),
            self._capi_minute(),
            self._capi_second(),
            self._capi_microsecond(),
            None,
            0,
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_timetz(self) -> datetime.time:
        "(cfunc) Access as `<datetime.time>` with timezone."
        return cydt.gen_time(
            self._capi_hour(),
            self._capi_minute(),
            self._capi_second(),
            self._capi_microsecond(),
            self._capi_tzinfo(),
            self._capi_fold(),
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_timeiso(self) -> str:
        "(cfunc) Access as time in ISO format `<str>`."
        return cydt.time_to_isoformat(self._capi_time())

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_ordinal(self) -> cython.uint:
        "(cfunc) Access as ordinal of the date `<int>`."
        return cydt.ymd_to_ordinal(
            self._capi_year(), self._capi_month(), self._capi_day()
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_seconds(self) -> cython.double:
        """(cfunc) Access in total seconds since EPOCH,
        ignoring the timezone (if exists) `<float>`.

        ### Notice
        This should `NOT` be treated as timestamp.
        """
        return cydt.dt_to_seconds(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_seconds_utc(self) -> cython.double:
        """(cfunc) Access in total seconds since EPOCH `<float>`.
        - If `timezone-aware`, return total seconds in UTC.
        - If `timezone-naive`, requivalent to `pydt.seconds`.

        ### Notice
        This should `NOT` be treated as timestamp.
        """
        return cydt.dt_to_seconds_utc(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_microseconds(self) -> cython.longlong:
        """(cfunc) Access in total microseconds since EPOCH,
        ignoring the timezone (if exists) `<int>`."""
        return cydt.dt_to_microseconds(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_microseconds_utc(self) -> cython.longlong:
        """(cfunc) Access in total microseconds since EPOCH `<int>`.
        - If `timezone-aware`, return total microseconds in UTC.
        - If `timezone-naive`, requivalent to `pydt.microseconds`.
        """
        return cydt.dt_to_microseconds_utc(self._dt)

    @cython.cfunc
    @cython.inline(True)
    def _capi_timestamp(self) -> cython.double:
        "(cfunc) Access in timestamp `<float>`."
        return cydt.dt_to_timestamp(self._dt)

    # Calendar: Year --------------------------------------------------------------------------
    def is_leapyear(self) -> bool:
        """Whether the current date is a leap year `<bool>`."""
        return self._capi_is_leapyear()

    def leap_bt_years(self, year: cython.int) -> int:
        """Calculate the number of leap years between
        the current date and the given 'year' `<int>`."""
        return self._capi_leap_bt_years(year)

    @property
    def days_in_year(self) -> int:
        """Get the maximum number of days in the year.
        Expect 365 or 366 (leapyear) `<int>`."""
        return self._capi_days_in_year()

    @property
    def days_bf_year(self) -> int:
        """Get the number of days betweem the 1st day of 1AD and
        the 1st day of the year of the current date `<int>`."""
        return self._capi_days_bf_year()

    @property
    def days_of_year(self) -> int:
        """Get the number of days between the 1st day
        of the year and the current date `<int>`."""
        return self._capi_days_of_year()

    # . c-api - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_is_leapyear(self) -> cython.bint:
        """(cfunc) Whether the current date is a leap year `<bool>`."""
        return cydt.is_leapyear(self._capi_year())

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_leap_bt_years(self, year: cython.uint) -> cython.uint:
        """(cfunc) Calculate the number of leap years between
        the current date and the given 'year' `<int>`."""
        return cydt.leap_bt_years(self._capi_year(), year)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_days_in_year(self) -> cython.uint:
        """(cfunc) Get the maximum number of days in the year.
        Expect 365 or 366 (leapyear) `<int>`."""
        return cydt.days_in_year(self._capi_year())

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_days_bf_year(self) -> cython.uint:
        """(cfunc) Get the number of days betweem the 1st day
        of 1AD and the 1st day of the year of the current date `<int>`."""
        return cydt.days_bf_year(self._capi_year())

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_days_of_year(self) -> cython.uint:
        """(cfunc) Get the number of days between the 1st day
        of the year and the current date `<int>`."""
        return cydt.days_of_year(
            self._capi_year(), self._capi_month(), self._capi_day()
        )

    # Manipulate: Year ------------------------------------------------------------------------
    def is_year(self, year: cython.int) -> bool:
        """Whether the current year is a specific year `<bool>`."""
        return self._capi_is_year(year)

    def is_year_1st(self) -> bool:
        """Whether is the 1st day of the year `<bool>`."""
        return self._capi_is_year_1st()

    def is_year_lst(self) -> bool:
        """Whether is the last day of the current year `<bool>`."""
        return self._capi_is_year_lst()

    def to_year_1st(self) -> pydt:
        """Go to the 1st day of the current year `<pydt>`."""
        return self._capi_to_year_1st()

    def to_year_lst(self) -> pydt:
        """Go to the last day of the current year `<pydt>`."""
        return self._capi_to_year_lst()

    def to_curr_year(
        self,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> pydt:
        """Go to specific 'month' and 'day' of the current year `<pydt>`."""
        return self._capi_to_curr_year(month, day)

    def to_next_year(
        self,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> pydt:
        """Go to specific 'month' and 'day' of the next year `<pydt>`."""
        return self._capi_to_year(1, month, day)

    def to_prev_year(
        self,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> pydt:
        """Go to specific 'month' and 'day' of the previous year `<pydt>`."""
        return self._capi_to_year(-1, month, day)

    def to_year(
        self,
        offset: cython.int = 0,
        month: int | str | None = None,
        day: cython.int = -1,
    ) -> pydt:
        """Go to specific 'month' and 'day' of the
        current year (+/-) 'offset' `<pydt>`."""
        return self._capi_to_year(offset, month, day)

    # . c-api - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_is_year(self, year: cython.int) -> cython.bint:
        """(cfunc) Whether the current year is a specific year `<bool>`."""
        cur_year: cython.int = self._capi_year()
        return cur_year == year

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_is_year_1st(self) -> cython.bint:
        """(cfunc) Whether is the 1st day of the year `<bool>`."""
        return self._capi_month() == 1 and self._capi_day() == 1

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_is_year_lst(self) -> cython.bint:
        """(cfunc) Whether is the last day of the current year `<bool>`."""
        return self._capi_month() == 12 and self._capi_day() == 31

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_to_year_1st(self) -> pydt:
        """(cfunc) Go to the 1st day of the current year `<pydt>`."""
        return self._new(
            cydt.gen_dt(
                self._capi_year(),
                1,
                1,
                self._capi_hour(),
                self._capi_minute(),
                self._capi_second(),
                self._capi_microsecond(),
                self._capi_tzinfo(),
                self._capi_fold(),
            )
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_to_year_lst(self) -> pydt:
        """(cfunc) Go to the last day of the current year `<pydt>`."""
        return self._new(
            cydt.gen_dt(
                self._capi_year(),
                12,
                31,
                self._capi_hour(),
                self._capi_minute(),
                self._capi_second(),
                self._capi_microsecond(),
                self._capi_tzinfo(),
                self._capi_fold(),
            )
        )

    @cython.cfunc
    @cython.inline(True)
    def _capi_to_curr_year(self, month: object, day: cython.int) -> pydt:
        """(cfunc) Go to specific 'month' and 'day'
        of the current year `<pydt>`."""
        # Parse new month
        new_mth: cython.uint = self._parse_month(month)
        if new_mth == 100 or new_mth == self._capi_month():
            return self._capi_to_curr_month(day)  # exit: no adjustment to month
        cur_yer: cython.uint = self._capi_year()

        # Calculate new day
        new_day: cython.uint = self._capi_day() if day < 1 else day
        if new_day > 28:
            new_day = min(new_day, cydt.days_in_month(cur_yer, new_mth))

        # Generate
        return self._new(
            cydt.gen_dt(
                cur_yer,
                new_mth,
                new_day,
                self._capi_hour(),
                self._capi_minute(),
                self._capi_second(),
                self._capi_microsecond(),
                self._capi_tzinfo(),
                self._capi_fold(),
            )
        )

    @cython.cfunc
    @cython.inline(True)
    def _capi_to_year(
        self,
        offset: cython.int,
        month: object,
        day: cython.int,
    ) -> pydt:
        """(cfunc) Go to specific 'month' and 'day'
        of the current year (+/-) 'offset' `<pydt>`."""
        # No offset adjustment
        if offset == 0:
            return self._capi_to_curr_year(month, day)  # exit: current year

        # Calculate new year
        year: cython.int = self._capi_year() + offset
        new_yer: cython.uint = min(max(year, 1), 9_999)

        # Parse new month
        new_mth: cython.uint = self._parse_month(month)
        if new_mth == 100:
            new_mth = self._capi_month()

        # Calculate new day
        new_day: cython.uint = self._capi_day() if day < 1 else day
        if new_day > 28:
            new_day = min(new_day, cydt.days_in_month(new_yer, new_mth))

        # Generate
        return self._new(
            cydt.gen_dt(
                new_yer,
                new_mth,
                new_day,
                self._capi_hour(),
                self._capi_minute(),
                self._capi_second(),
                self._capi_microsecond(),
                self._capi_tzinfo(),
                self._capi_fold(),
            )
        )

    # Calendar: Quarter -----------------------------------------------------------------------
    @property
    def days_in_quarter(self) -> int:
        """Get the maximum number of days in the quarter `<int>`."""
        return self._capi_days_in_quarter()

    @property
    def days_bf_quarter(self) -> int:
        """Get the number of days between the 1st day of
        the year and the 1st day of the quarter `<int>`."""
        return self._capi_days_bf_quarter()

    @property
    def days_of_quarter(self) -> int:
        """Get the number of days between the 1st day of
        the quarter and the current date `<int>`."""
        return self._capi_days_of_quarter()

    @property
    def quarter_1st_month(self) -> int:
        """Get the first month of the quarter.
        Expect 1, 4, 7, 10 `<int>`."""
        return self._capi_quarter_1st_month()

    @property
    def quarter_lst_month(self) -> int:
        """Get the last month of the quarter.
        Expect 3, 6, 9, 12 `<int>`."""
        return self._capi_quarter_lst_month()

    # . c-api - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_days_in_quarter(self) -> cython.uint:
        """(cfunc) Get the maximum number of days in the quarter `<int>`."""
        return cydt.days_in_quarter(self._capi_year(), self._capi_month())

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_days_bf_quarter(self) -> cython.uint:
        """(cfunc) Get the number of days between the 1st day of
        the year and the 1st day of the quarter `<int>."""
        return cydt.days_bf_quarter(self._capi_year(), self._capi_month())

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_days_of_quarter(self) -> cython.uint:
        """(cfunc) Get the number of days between the 1st day of
        the quarter and the current date `<int>`."""
        return cydt.days_of_quarter(
            self._capi_year(), self._capi_month(), self._capi_day()
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_quarter_1st_month(self) -> cython.uint:
        """(cfunc) Get the first month of the quarter.
        Expect 1, 4, 7, 10 `<int>`."""
        return cydt.quarter_1st_month(self._capi_month())

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_quarter_lst_month(self) -> cython.uint:
        """(cfunc) Get the last month of the quarter.
        Expect 3, 6, 9, 12 `<int>`."""
        return cydt.quarter_lst_month(self._capi_month())

    # Manipulate: Quarter ---------------------------------------------------------------------
    def is_quarter(self, quarter: cython.int) -> bool:
        """Whether the current quarter is a specific quarter `<bool>`."""
        return self._capi_is_quarter(quarter)

    def is_quarter_1st(self) -> bool:
        """Whether is the 1st day of the quarter `<bool>`."""
        return self._capi_is_quarter_1st()

    def is_quarter_lst(self) -> bool:
        """Whether is the last day of the quarter `<bool>`."""
        return self._capi_is_quarter_lst()

    def to_quarter_1st(self) -> pydt:
        """Go to the 1st day of the current quarter `<pydt>`."""
        return self._capi_to_quarter_1st()

    def to_quarter_lst(self) -> pydt:
        """Go to the last day of the current quarter `<pydt>`."""
        return self._capi_to_quarter_lst()

    def to_curr_quarter(self, month: cython.int = -1, day: cython.int = -1) -> pydt:
        """Go to specific 'month (of the quarter [1..3])'
        and 'day' of the current quarter `<pydt>`."""
        return self._capi_to_curr_quarter(month, day)

    def to_next_quarter(self, month: cython.int = -1, day: cython.int = -1) -> pydt:
        """Go to specific 'month (of the quarter [1..3])'
        and 'day' of the next quarter `<pydt>`."""
        return self._capi_to_quarter(1, month, day)

    def to_prev_quarter(self, month: cython.int = -1, day: cython.int = -1) -> pydt:
        """Go to specific 'month (of the quarter [1..3])'
        and 'day' of the previous quarter `<pydt>`."""
        return self._capi_to_quarter(-1, month, day)

    def to_quarter(
        self,
        offset: cython.int = 0,
        month: cython.int = -1,
        day: cython.int = -1,
    ) -> pydt:
        """Go to specific 'month (of the quarter [1..3])'
        and 'day' of the current quarter (+/-) 'offset' `<pydt>`."""
        return self._capi_to_quarter(offset, month, day)

    # . c-api - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_is_quarter(self, quarter: cython.int) -> cython.bint:
        """(cfunc) Whether the current quarter
        is a specific quarter `<bool>`."""
        cur_quarter: cython.int = self._capi_quarter()
        return cur_quarter == quarter

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_is_quarter_1st(self) -> cython.bint:
        """(cfunc) Whether is the 1st day of the quarter `<bool>`."""
        return (
            self._capi_day() == 1
            and self._capi_month() == self._capi_quarter_1st_month()
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_is_quarter_lst(self) -> cython.bint:
        """(cfunc) Whether is the last day of the quarter `<bool>`."""
        return (
            self._capi_day() == self._capi_days_in_month()
            and self._capi_month() == self._capi_quarter_lst_month()
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_to_quarter_1st(self) -> pydt:
        """(cfunc) Go to the 1st day of the current quarter `<pydt>`."""
        return self._new(
            cydt.gen_dt(
                self._capi_year(),
                self._capi_quarter_1st_month(),
                1,
                self._capi_hour(),
                self._capi_minute(),
                self._capi_second(),
                self._capi_microsecond(),
                self._capi_tzinfo(),
                self._capi_fold(),
            )
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_to_quarter_lst(self) -> pydt:
        """(cfunc) Go to the last day of the current quarter `<pydt>`."""
        cur_yer: cython.uint = self._capi_year()
        new_mth: cython.uint = self._capi_quarter_lst_month()
        new_day: cython.uint = cydt.days_in_month(cur_yer, new_mth)
        return self._new(
            cydt.gen_dt(
                cur_yer,
                new_mth,
                new_day,
                self._capi_hour(),
                self._capi_minute(),
                self._capi_second(),
                self._capi_microsecond(),
                self._capi_tzinfo(),
                self._capi_fold(),
            )
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_to_curr_quarter(self, month: cython.int, day: cython.int) -> pydt:
        """(cfunc) Go to specific 'month (of the quarter [1..3])'
        and 'day' of the current quarter `<pydt>`."""
        # Calculate new month
        if month < 1:
            return self._capi_to_curr_month(day)  # exit: no adjustment to month
        new_mth: cython.uint = month
        new_mth = self._capi_quarter() * 3 - 3 + (month % 3 or 3)
        if new_mth == self._capi_month():
            return self._capi_to_curr_month(day)  # exit: no adjustment to month
        cur_yer: cython.uint = self._capi_year()

        # Calculate new day
        new_day: cython.uint = self._capi_day() if day < 1 else day
        if new_day > 28:
            new_day = min(new_day, cydt.days_in_month(cur_yer, new_mth))

        # Generate
        return self._new(
            cydt.gen_dt(
                cur_yer,
                new_mth,
                new_day,
                self._capi_hour(),
                self._capi_minute(),
                self._capi_second(),
                self._capi_microsecond(),
                self._capi_tzinfo(),
                self._capi_fold(),
            )
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_to_quarter(
        self,
        offset: cython.int,
        month: cython.int,
        day: cython.int,
    ) -> pydt:
        """(cfunc) Go to specific 'month (of the quarter [1..3])'
        and 'day' of the current quarter (+/-) 'offset' `<pydt>`."""
        # No offset adjustment
        if offset == 0:
            return self._capi_to_curr_quarter(month, day)  # exit: current quarter

        # Calculate new year & month
        new_yer: cython.int = self._capi_year()
        if month < 1:
            new_mth: cython.int = self._capi_month()
        else:
            new_mth: cython.int = self._capi_quarter() * 3 - 3 + (month % 3 or 3)
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
        new_day: cython.uint = self._capi_day() if day < 1 else day
        if new_day > 28:
            new_day = min(new_day, cydt.days_in_month(new_yer, new_mth))

        # Generate
        return self._new(
            cydt.gen_dt(
                new_yer,
                new_mth,
                new_day,
                self._capi_hour(),
                self._capi_minute(),
                self._capi_second(),
                self._capi_microsecond(),
                self._capi_tzinfo(),
                self._capi_fold(),
            )
        )

    # Calendar: Month -------------------------------------------------------------------------
    @property
    def days_in_month(self) -> int:
        """Get the maximum number of days in the month `<int>`."""
        return self._capi_days_in_month()

    @property
    def days_bf_month(self) -> int:
        """Get the number of days between the 1st day of
        the year and the 1st day of the month `<int>`."""
        return self._capi_days_bf_month()

    # . c-api - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_days_in_month(self) -> cython.uint:
        """(cfunc) Get the maximum number of days in the month `<int>`."""
        return cydt.days_in_month(self._capi_year(), self._capi_month())

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_days_bf_month(self) -> cython.uint:
        """(cfunc) Get the number of days between the 1st day of
        the year and the 1st day of the month `<int>`."""
        return cydt.days_bf_month(self._capi_year(), self._capi_month())

    # Manipulate: Month -----------------------------------------------------------------------
    def is_month(self, month: int | str) -> bool:
        """Whether the current month is a specific 'month' `<bool>`."""
        return self._capi_is_month(month)

    def is_month_1st(self) -> bool:
        """Whether is the 1st day of the month `<bool>`."""
        return self._capi_is_month_1st()

    def is_month_lst(self) -> bool:
        """Whether is the last day of the current month `<bool>`."""
        return self._capi_is_month_lst()

    def to_month_1st(self) -> pydt:
        """Go to the 1st day of the current month `<pydt>`."""
        return self._capi_to_month_1st()

    def to_month_lst(self) -> pydt:
        """Go to the last day of the current month `<pydt>`."""
        return self._capi_to_month_lst()

    def to_curr_month(self, day: cython.int = -1) -> pydt:
        """Go to specific 'day' of the current month `<pydt>`."""
        return self._capi_to_curr_month(day)

    def to_next_month(self, day: cython.int = -1) -> pydt:
        """Go to specific day of the next month `<pydt>`."""
        return self._capi_to_month(1, day)

    def to_prev_month(self, day: cython.int = -1) -> pydt:
        """Go to specific day of the previous month `<pydt>`."""
        return self._capi_to_month(-1, day)

    def to_month(self, offset: cython.int = 0, day: cython.int = -1) -> pydt:
        """Go to specific 'day' of the current
        month (+/-) 'offset' `<pydt>`."""
        return self._capi_to_month(offset, day)

    # . c-api - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_is_month(self, month: object) -> cython.bint:
        """(cfunc) Whether the current month is a specific 'month' `<bool>`."""
        return self._capi_month() == self._parse_month(month)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_is_month_1st(self) -> cython.bint:
        """(cfunc) Whether is the 1st day of the month `<bool>`."""
        return self._capi_day() == 1

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_is_month_lst(self) -> cython.bint:
        """(cfunc) Whether is the last day of the current month `<bool>`."""
        return self._capi_day() == self._capi_days_in_month()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_to_month_1st(self) -> pydt:
        """(cfunc) Go to the 1st day of the current month `<pydt>`."""
        return self._new(
            cydt.gen_dt(
                self._capi_year(),
                self._capi_month(),
                1,
                self._capi_hour(),
                self._capi_minute(),
                self._capi_second(),
                self._capi_microsecond(),
                self._capi_tzinfo(),
                self._capi_fold(),
            )
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_to_month_lst(self) -> pydt:
        """(cfunc) Go to the last day of the current month `<pydt>`."""
        return self._new(
            cydt.gen_dt(
                self._capi_year(),
                self._capi_month(),
                self._capi_days_in_month(),
                self._capi_hour(),
                self._capi_minute(),
                self._capi_second(),
                self._capi_microsecond(),
                self._capi_tzinfo(),
                self._capi_fold(),
            )
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_to_curr_month(self, day: cython.int) -> pydt:
        """(cfunc) Go to specific 'day' of the
        current month `<pydt>`."""
        # Invalid day value
        if day < 1:
            return self  # exit: invalid day

        # Clip by max days in month
        new_day: cython.uint = day
        if new_day > 28:
            new_day = min(new_day, self._capi_days_in_month())

        # Compare with current day
        cur_day: cython.uint = self._capi_day()
        if new_day == cur_day:
            return self  # exit: same day

        # Generate
        return self._new(
            cydt.gen_dt(
                self._capi_year(),
                self._capi_month(),
                new_day,
                self._capi_hour(),
                self._capi_minute(),
                self._capi_second(),
                self._capi_microsecond(),
                self._capi_tzinfo(),
                self._capi_fold(),
            )
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_to_month(self, offset: cython.int, day: cython.int) -> pydt:
        """(cfunc) Go to specific 'day' of the
        current month (+/-) 'offset' `<pydt>`."""
        # No offset adjustment
        if offset == 0:
            return self._capi_to_curr_month(day)  # exit: current month

        # Calculate new year & month
        new_yer: cython.int = self._capi_year()
        new_mth: cython.int = self._capi_month() + offset
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
        new_day: cython.uint = self._capi_day() if day < 1 else day
        if new_day > 28:
            new_day = min(new_day, cydt.days_in_month(new_yer, new_mth))

        # Generate
        return self._new(
            cydt.gen_dt(
                new_yer,
                new_mth,
                new_day,
                self._capi_hour(),
                self._capi_minute(),
                self._capi_second(),
                self._capi_microsecond(),
                self._capi_tzinfo(),
                self._capi_fold(),
            )
        )

    # Calendar: Weekday -----------------------------------------------------------------------
    @property
    def weekday(self) -> int:
        """The weekday of the datetime `<int>`.
        Values: 0=Monday...6=Sunday."""
        return self._capi_weekday()

    @property
    def isoweekday(self) -> int:
        """The ISO calendar weekday of the datetime `<int>`.
        Values: 1=Monday...7=Sunday."""
        return self._capi_isoweekday()

    @property
    def isoweek(self) -> int:
        """Get the ISO calendar week number `<int>`."""
        return self._capi_isoweek()

    @property
    def isoyear(self) -> int:
        """Get the ISO calendar year `<int>`."""
        return self._capi_isoyear()

    @property
    def isocalendar(self) -> dict[str, int]:
        """Get the ISO calendar of the current date `<dict>`."""
        return self._capi_isocalendar()

    # . c-api - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_weekday(self) -> cython.uint:
        """(cfunc) The weekday of the datetime `<int>`.
        Values: 0=Monday...6=Sunday."""
        return cydt.ymd_weekday(self._capi_year(), self._capi_month(), self._capi_day())

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_isoweekday(self) -> cython.uint:
        """(cfunc) The ISO calendar weekday of the datetime `<int>`.
        Values: 1=Monday...7=Sunday."""
        return cydt.ymd_isoweekday(
            self._capi_year(), self._capi_month(), self._capi_day()
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_isoweek(self) -> cython.uint:
        """(cfunc) Get the ISO calendar week number `<int>`."""
        return cydt.ymd_isoweek(self._capi_year(), self._capi_month(), self._capi_day())

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_isoyear(self) -> cython.uint:
        """(cfunc) Get the ISO calendar year `<int>`."""
        return cydt.ymd_isoyear(self._capi_year(), self._capi_month(), self._capi_day())

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_isocalendar(self) -> cydt.iso:
        """Get the ISO calendar of the current date `<dict>`."""
        return cydt.ymd_isocalendar(
            self._capi_year(), self._capi_month(), self._capi_day()
        )

    # Manipulate: Weekday ---------------------------------------------------------------------
    def is_weekday(self, weekday: int | str) -> bool:
        """Whether the current weekday is a specific 'weekday' `<bool>`."""
        return self._capi_is_weekday(weekday)

    def to_monday(self) -> pydt:
        """Go to Monday of the current week `<pydt>`."""
        return self._capi_to_curr_weekday_int(0)

    def to_tuesday(self) -> pydt:
        """Go to Tuesday of the current week `<pydt>`."""
        return self._capi_to_curr_weekday_int(1)

    def to_wednesday(self) -> pydt:
        """Go to Wednesday of the current week `<pydt>`."""
        return self._capi_to_curr_weekday_int(2)

    def to_thursday(self) -> pydt:
        """Go to Thursday of the current week `<pydt>`."""
        return self._capi_to_curr_weekday_int(3)

    def to_friday(self) -> pydt:
        """Go to Friday of the current week `<pydt>`."""
        return self._capi_to_curr_weekday_int(4)

    def to_saturday(self) -> pydt:
        """Go to Saturday of the current week `<pydt>`."""
        return self._capi_to_curr_weekday_int(5)

    def to_sunday(self) -> pydt:
        """Go to Sunday of the current week `<pydt>`."""
        return self._capi_to_curr_weekday_int(6)

    def to_curr_weekday(self, weekday: int | str | None = None) -> pydt:
        """Go to specific 'weekday' of the current week `<pydt>`."""
        return self._capi_to_curr_weekday(weekday)

    def to_next_weekday(self, weekday: int | str | None = None) -> pydt:
        """Go to specific 'weekday' of the next week `<pydt>`."""
        return self._capi_to_weekday(1, weekday)

    def to_prev_weekday(self, weekday: int | str | None = None) -> pydt:
        """Go to specific 'weekday' of the previous week `<pydt>`."""
        return self._capi_to_weekday(-1, weekday)

    def to_weekday(
        self,
        offset: cython.int = 0,
        weekday: int | str | None = None,
    ) -> pydt:
        """Go to specific 'weekday' of the current
        week (+/-) 'offset' `<pydt>`."""
        return self._capi_to_weekday(offset, weekday)

    # . c-api - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_is_weekday(self, weekday: object) -> cython.bint:
        """(cfunc) Whether the current weekday
        is a specific 'weekday' `<bool>`."""
        return self._capi_weekday() == self._parse_weekday(weekday)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_to_curr_weekday_int(self, weekday: cython.uint) -> pydt:
        """(cfunc) Go to specific 'weekday (<int>)'
        of the current week `<pydt>`."""
        # Validate weekday
        if weekday > 6:
            weekday = 6
        cur_wkd: cython.uint = self._capi_weekday()
        if weekday == cur_wkd:
            return self  # exit: same weekday

        # Generate
        delta: cython.int = weekday - cur_wkd
        return self._new(
            cydt.gen_dt(
                self._capi_year(),
                self._capi_month(),
                self._capi_day() + delta,
                self._capi_hour(),
                self._capi_minute(),
                self._capi_second(),
                self._capi_microsecond(),
                self._capi_tzinfo(),
                self._capi_fold(),
            )
        )

    @cython.cfunc
    @cython.inline(True)
    def _capi_to_curr_weekday(self, weekday: object) -> pydt:
        """(cfunc) Go to specific 'weekday' of the current week `<pydt>`."""
        # Validate weekday
        new_wkd: cython.uint = self._parse_weekday(weekday)
        cur_wkd: cython.uint = self._capi_weekday()
        if new_wkd == 100 or new_wkd == cur_wkd:
            return self  # exit: same weekday

        # Generate
        delta: cython.int = new_wkd - cur_wkd
        return self._new(
            cydt.gen_dt(
                self._capi_year(),
                self._capi_month(),
                self._capi_day() + delta,
                self._capi_hour(),
                self._capi_minute(),
                self._capi_second(),
                self._capi_microsecond(),
                self._capi_tzinfo(),
                self._capi_fold(),
            )
        )

    @cython.cfunc
    @cython.inline(True)
    def _capi_to_weekday(self, offset: cython.int, weekday: object) -> pydt:
        """(cfunc) Go to specific 'weekday' of the
        current week (+/-) 'offset' `<pydt>`."""
        # No offset adjustment
        if offset == 0:
            return self._capi_to_curr_weekday(weekday)  # exit: current weekday

        # Validate weekday & calculate delta
        new_wkd: cython.uint = self._parse_weekday(weekday)
        cur_wkd: cython.uint = self._capi_weekday()
        if new_wkd == 100:
            delta: cython.int = offset * 7
        else:
            delta: cython.int = new_wkd - cur_wkd
            delta += offset * 7

        # Generate
        return self._new(cydt.dt_add(self._dt, delta))

    # Manipulate: Day -------------------------------------------------------------------------
    def is_day(self, day: cython.int) -> bool:
        """Whether the current day is a specific 'day' `<bool>`."""
        return self._capi_is_day(day)

    def to_tomorrow(self) -> pydt:
        """Go to the next day `<pydt>`."""
        return self._capi_to_day(1)

    def to_yesterday(self) -> pydt:
        """Go to the previous day `<pydt>`."""
        return self._capi_to_day(-1)

    def to_day(self, offset: cython.int) -> pydt:
        """Go to the current day (+/-) 'offset' `<pydt>`."""
        return self._capi_to_day(offset)

    # . c-api - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_is_day(self, day: cython.int) -> cython.bint:
        """(cfunc) Whether the current day is a specific 'day' `<bool>`."""
        cur_day: cython.int = self._capi_day()
        return cur_day == day

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_to_day(self, offset: cython.int) -> pydt:
        """(cfunc) Go to the current day (+/-) 'offset' `<pydt>`."""
        # No offset adjustment
        if offset == 0:
            return self

        # Generate
        return self._new(cydt.dt_add(self._dt, offset))

    # Manipulate: Time ------------------------------------------------------------------------
    def is_time_start(self) -> bool:
        """Whether the current time is the
        start of time (00:00:00.000000) `<bool>`."""
        return self._capi_is_time_start()

    def is_time_end(self) -> bool:
        """Whether the current time is the
        end of time (23:59:59.999999) `<bool>`."""
        return self._capi_is_time_end()

    def to_time_start(self) -> pydt:
        """Go to the start of time (00:00:00.000000)
        of the current datetime `<pydt>`."""
        return self._capi_to_time_start()

    def to_time_end(self) -> pydt:
        """Go to the end of time (23:59:59.999999)
        of the time `<pydt>`."""
        return self._capi_to_time_end()

    def to_time(
        self,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        microsecond: cython.int = -1,
    ) -> pydt:
        """Go to specific 'hour', 'minute', 'second'
        and 'microsecond' of the current datetime `<pydt>`."""
        return self._capi_to_time(hour, minute, second, microsecond)

    # . c-api - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_is_time_start(self) -> cython.bint:
        """(cfunc) Whether the current time is the
        start of time (00:00:00.000000) `<bool>`."""
        return (
            self._capi_hour() == 0
            and self._capi_minute() == 0
            and self._capi_second() == 0
            and self._capi_microsecond() == 0
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _capi_is_time_end(self) -> cython.bint:
        """(cfunc) Whether the current time is the
        end of time (23:59:59.999999) `<bool>`."""
        return (
            self._capi_hour() == 23
            and self._capi_minute() == 59
            and self._capi_second() == 59
            and self._capi_microsecond() == 999_999
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_to_time_start(self) -> pydt:
        """(cfunc) Go to the start of time (00:00:00.000000)
        of the current datetime `<pydt>`."""
        return self._new(
            cydt.gen_dt(
                self._capi_year(),
                self._capi_month(),
                self._capi_day(),
                0,
                0,
                0,
                0,
                self._capi_tzinfo(),
                self._capi_fold(),
            )
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_to_time_end(self) -> pydt:
        """(cfunc) Go to the end of time (23:59:59.999999)
        of the current datetime `<pydt>`."""
        return self._new(
            cydt.gen_dt(
                self._capi_year(),
                self._capi_month(),
                self._capi_day(),
                23,
                59,
                59,
                999_999,
                self._capi_tzinfo(),
                self._capi_fold(),
            )
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_to_time(
        self,
        hour: cython.int,
        minute: cython.int,
        second: cython.int,
        microsecond: cython.int,
    ) -> pydt:
        """(cfunc) Go to specific 'hour', 'minute', "second'
        and 'microsecond' of the time `<pydt>`."""
        return self._new(
            # fmt: off
            cydt.gen_dt(
                self._capi_year(),
                self._capi_month(),
                self._capi_day(),
                hour if 0 <= hour <= 23 else self._capi_hour(),
                minute if 0 <= minute <= 59 else self._capi_minute(),
                second if 0 <= second <= 59 else self._capi_second(),
                microsecond if 0 <= microsecond <= 999_999 else self._capi_microsecond(),
                self._capi_tzinfo(),
                self._capi_fold(),
            )
            # fmt: on
        )

    # Manipulate: Timezone --------------------------------------------------------------------
    @property
    def tz_available(self) -> set[str]:
        """Access all the available timezone names `<set[str]>`."""
        return self._capi_tz_available()

    def tz_localize(self, tz: str | datetime.tzinfo | None = None) -> pydt:
        """Localize to a specific 'tz' (<str> or <tzinfo>) timezone `<pydt>`.
        Equivalent to `datetime.replace(tzinfo=tz<tzinfo>)`.

        ### Notice
        Timezone from `pytz` library should not be used,
        and will yield incorrect result.
        """
        return self._capi_tz_localize(tz)

    def tz_convert(self, tz: str | datetime.tzinfo | None = None) -> pydt:
        """Convert to a specific 'tz' (<str> or <tzinfo>) timezone `<pydt>`.
        Equivalent to `datetime.astimezone(tz<tzinfo>)`.

        ### Notice
        Timezone from `pytz` library should not be used,
        and will yield incorrect result.
        """
        return self._capi_tz_convert(tz)

    def tz_switch(
        self,
        targ_tz: str | datetime.tzinfo,
        base_tz: str | datetime.tzinfo | None = None,
        naive: bool = False,
    ) -> pydt:
        """Switch to 'targ_tz' timezone from 'base_tz' timezone `<pydt>`.

        :param targ_tz `<str/tzinfo>`: The target timezone to convert.
        :param base_tz `<str/tzinfo>`: The base timezone to localize. Defaults to `None`.
        :param naive `<bool>`: Whether to return as timezone-naive. Defaults to `False`.
        :return `<pydt>`: pydt after switch of timezone.

        ### Explaination
        - If pydt is timezone-aware, 'base_tz' will be ignored, and only performs
          the convertion to the 'targ_tz' (datetime.astimezone(targ_tz <tzinfo>)).

        - If pydt is timezone-naive, first will localize to the given 'base_tz',
          then convert to the 'targ_tz'. In this case, the 'base_tz' must be
          specified, else it is ambiguous on how to convert to the target timezone.
        """
        return self._capi_tz_switch(targ_tz, base_tz, naive)

    # . c-api - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _capi_tz_available(self) -> set[str]:
        """(cfunc) Access all the available timezone names `<set[str]>`."""
        return TIMEZONES_AVAILABLE

    @cython.cfunc
    @cython.inline(True)
    def _capi_tz_localize(self, tz: object) -> pydt:
        """(cfunc) Localize to a specific 'tz' (<str> or <tzinfo>) timezone `<pydt>`.
        Equivalent to `datetime.replace(tzinfo=tz<tzinfo>)`.

        ### Notice
        Timezone from `pytz` library should not be used,
        and will yield incorrect result.
        """
        # Parse timezone
        tzinfo = self._parse_tzinfo(tz)
        if tzinfo is self._capi_tzinfo():
            return self  # exit: same timezone

        # Localize timezone
        return self._new(cydt.dt_replace_tzinfo(self._dt, tzinfo))

    @cython.cfunc
    @cython.inline(True)
    def _capi_tz_convert(self, tz: object) -> pydt:
        """"""
        # Parse timezone
        tzinfo = self._parse_tzinfo(tz)
        if tzinfo is not None and tzinfo is self._capi_tzinfo():
            return self  # exit: same timezone

        # Convert timezone
        return self._new(cydt.dt_astimezone(self._dt, tzinfo))

    @cython.cfunc
    @cython.inline(True)
    def _capi_tz_switch(
        self,
        targ_tz: object,
        base_tz: object = None,
        naive: cython.bint = False,
    ) -> pydt:
        """(cfunc) Switch to 'targ_tz' timezone from 'base_tz' timezone `<pydt>`.

        :param targ_tz `<str/tzinfo>`: The target timezone to convert.
        :param base_tz `<str/tzinfo>`: The base timezone to localize. Defaults to `None`.
        :param naive `<bool>`: Whether to return as timezone-naive. Defaults to `False`.
        :return `<pydt>`: pydt after switch of timezone.

        ### Explaination
        - If pydt is timezone-aware, 'base_tz' will be ignored, and only performs
          the convertion to the 'targ_tz' (datetime.astimezone(targ_tz <tzinfo>)).

        - If pydt is timezone-naive, first will localize to the given 'base_tz',
          then convert to the 'targ_tz'. In this case, the 'base_tz' must be
          specified, else it is ambiguous on how to convert to the target timezone.
        """
        # Pydt is timezone-aware
        dt: datetime.datetime
        locl_tz = self._capi_tzinfo()
        targ_tz = self._parse_tzinfo(targ_tz)
        if locl_tz is not None:
            # . local == target timezone
            if locl_tz is targ_tz:
                if naive:
                    dt = cydt.dt_replace_tzinfo(self._dt, None)
                else:
                    return self  # exit: not action
            # . local => target timezone
            else:
                dt = cydt.dt_astimezone(self._dt, targ_tz)
                if naive:
                    dt = cydt.dt_replace_tzinfo(dt, None)

        # Pydt is timezone-naive
        elif isinstance(base_tz, str) or cydt.is_tzinfo(base_tz):
            base_tz = self._parse_tzinfo(base_tz)
            # . base == target timezone
            if targ_tz is base_tz:
                if not naive:
                    dt = cydt.dt_replace_tzinfo(self._dt, targ_tz)
                else:
                    return self  # exit: not action
            # . localize to base & base => target
            else:
                dt = cydt.dt_replace_tzinfo(self._dt, base_tz)
                dt = cydt.dt_astimezone(dt, targ_tz)
                if naive:
                    dt = cydt.dt_replace_tzinfo(dt, None)

        # Invalid
        else:
            raise errors.InvalidTimezoneError(
                "<{}>\nCannot switch timezone-naive pydt without "
                "a valid 'base_tz'.".format(self.__class__.__name__)
            )

        # Generate
        return self._new(dt)

    # Manipulate: Frequency -------------------------------------------------------------------
    def freq_round(self, freq: Literal["D", "h", "m", "s", "ms", "us"]) -> pydt:
        """Perform round operation to specified freqency `<pydt>`.
        Similar to `pandas.DatetimeIndex.round()`.

        :param freq: `<str>` frequency to round to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond
        """
        return self._capi_freq_round(freq)

    def freq_ceil(self, freq: Literal["D", "h", "m", "s", "ms", "us"]) -> pydt:
        """Perform ceil operation to specified freqency `<pydt>`.
        Similar to `pandas.DatetimeIndex.ceil()`.

        :param freq: `<str>` frequency to ceil to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond
        """
        return self._capi_freq_ceil(freq)

    def freq_floor(self, freq: Literal["D", "h", "m", "s", "ms", "us"]) -> pydt:
        """Perform floor operation to specified freqency `<pydt>`.
        Similar to `pandas.DatetimeIndex.floor()`.

        :param freq: `<str>` frequency to floor to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond
        """
        return self._capi_freq_floor(freq)

    # . c-api - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    def _capi_freq_round(self, freq: object) -> pydt:
        """(cfunc) Perform round operation to specified freqency `<pydt>`.
        Similar to `pandas.DatetimeIndex.round()`.

        :param freq: `<str>` frequency to round to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond
        """
        # Parse frequency
        freq_val: cython.longlong = self._parse_frequency(freq)
        if freq_val == 1:
            return self  # exit: no action

        # Round frequency
        us: cython.longlong = self._capi_microseconds()
        us = int(math.roundl(us / freq_val))
        us *= freq_val

        # Generate
        return self._new(cydt.dt_fr_microseconds(us, self._capi_tzinfo()))

    @cython.cfunc
    @cython.inline(True)
    def _capi_freq_ceil(self, freq: object) -> pydt:
        """(cfunc) Perform ceil operation to specified freqency `<pydt>`.
        Similar to `pandas.DatetimeIndex.ceil()`.

        :param freq: `<str>` frequency to ceil to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond
        """
        # Parse frequency
        freq_val: cython.longlong = self._parse_frequency(freq)
        if freq_val == 1:
            return self  # exit: no action

        # Ceil frequency
        us: cython.longlong = self._capi_microseconds()
        us = int(math.ceill(us / freq_val))
        us *= freq_val

        # Generate
        return self._new(cydt.dt_fr_microseconds(us, self._capi_tzinfo()))

    @cython.cfunc
    @cython.inline(True)
    def _capi_freq_floor(self, freq: object) -> pydt:
        """(cfunc) Perform floor operation to specified freqency `<pydt>`.
        Similar to `pandas.DatetimeIndex.floor()`.

        :param freq: `<str>` frequency to floor to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond
        """
        # Parse frequency
        freq_val: cython.longlong = self._parse_frequency(freq)
        if freq_val == 1:
            return self  # exit: no action

        # Floor frequency
        us: cython.longlong = self._capi_microseconds()
        us = int(math.floorl(us / freq_val))
        us *= freq_val

        # Generate
        return self._new(cydt.dt_fr_microseconds(us, self._capi_tzinfo()))

    # Manipulate: Delta -----------------------------------------------------------------------
    def add_delta(
        self,
        years: int = 0,
        months: int = 0,
        days: int = 0,
        weeks: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        miliseconds: int = 0,
        microseconds: int = 0,
        year: int = -1,
        month: int = -1,
        day: int = -1,
        weekday: int | str | None = None,
        hour: int = -1,
        minute: int = -1,
        second: int = -1,
        milisecond: int = -1,
        microsecond: int = -1,
    ) -> pydt:
        """Add 'timedelta' to the current `<pydt>`.

        Equivalent to `pydt/datetime + cytimes.cytimedelta`.
        For more information, please refer to `<cytimedelta>`.

        ### Absolute Delta
        :param year `<int>`: The absolute year value. Defaults to `-1 (no change)`.
        :param month `<int>`: The absolute month value. Defaults to `-1 (no change)`.
        :param day `<int>`: The absolute day value. Defaults to `-1 (no change)`.
        :param weekday `<int/str/None>`: The absolute weekday value. Defaults to `None (no change)`.
            Accepts both integer and string. Where 0=Monday...6=Sunday,
            or string of weekday name (case-insensitive).
        :param hour `<int>`: The absolute hour value. Defaults to `-1 (no change)`.
        :param minute `<int>`: The absolute minute value. Defaults to `-1 (no change)`.
        :param second `<int>`: The absolute second value. Defaults to `-1 (no change)`.
        :param milisecond `<int>`: The absolute milisecond value. Defaults to `-1 (no change)`.
        :param microsecond `<int>`: The absolute microsecond value. Defaults to `-1 (no change)`.

        ### Relative delta
        :param years `<int>`: The relative delta of years. Defaults to `0`.
        :param months `<int>`: The relative delta of months. Defaults to `0`.
        :param days `<int>`: The relative delta of days. Defaults to `0`.
        :param weeks `<int>`: The relative delta of weeks. Defaults to `0`.
        :param hours `<int>`: The relative delta of hours. Defaults to `0`.
        :param minutes `<int>`: The relative delta of minutes. Defaults to `0`.
        :param seconds `<int>`: The relative delta of seconds. Defaults to `0`.
        :param miliseconds `<int>`: The relative delta of miliseconds. Defaults to `0`.
        :param microseconds `<int>`: The relative delta of microseconds. Defaults to `0`.
        """
        # Parse weekday
        if weekday is None:
            weekday = -1
        elif weekday != -1:
            weekday = self._parse_weekday(weekday)
            if weekday == 100:
                weekday = -1

        # Generate
        return self._new(
            # fmt: off
            cytimedelta(
                years, months, days, weeks, hours, minutes, seconds, 
                miliseconds, microseconds, year, month, day, weekday, 
                hour, minute, second, milisecond, microsecond,
            )._add_datetime(self._dt)
            # fmt: on
        )

    def sub_delta(
        self,
        years: int = 0,
        months: int = 0,
        days: int = 0,
        weeks: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        miliseconds: int = 0,
        microseconds: int = 0,
        year: int = -1,
        month: int = -1,
        day: int = -1,
        weekday: int | str | None = None,
        hour: int = -1,
        minute: int = -1,
        second: int = -1,
        milisecond: int = -1,
        microsecond: int = -1,
    ) -> pydt:
        """Substract 'timedelta' to the current `<pydt>`.

        Equivalent to `pydt/datetime - cytimes.cytimedelta`.
        For more information, please refer to `<cytimedelta>`.

        ### Absolute Delta
        :param year `<int>`: The absolute year value. Defaults to `-1 (no change)`.
        :param month `<int>`: The absolute month value. Defaults to `-1 (no change)`.
        :param day `<int>`: The absolute day value. Defaults to `-1 (no change)`.
        :param weekday `<int/str/None>`: The absolute weekday value. Defaults to `None (no change)`.
            Accepts both integer and string. Where 0=Monday...6=Sunday,
            or string of weekday name (case-insensitive).
        :param hour `<int>`: The absolute hour value. Defaults to `-1 (no change)`.
        :param minute `<int>`: The absolute minute value. Defaults to `-1 (no change)`.
        :param second `<int>`: The absolute second value. Defaults to `-1 (no change)`.
        :param milisecond `<int>`: The absolute milisecond value. Defaults to `-1 (no change)`.
        :param microsecond `<int>`: The absolute microsecond value. Defaults to `-1 (no change)`.

        ### Relative delta
        :param years `<int>`: The relative delta of years. Defaults to `0`.
        :param months `<int>`: The relative delta of months. Defaults to `0`.
        :param days `<int>`: The relative delta of days. Defaults to `0`.
        :param weeks `<int>`: The relative delta of weeks. Defaults to `0`.
        :param hours `<int>`: The relative delta of hours. Defaults to `0`.
        :param minutes `<int>`: The relative delta of minutes. Defaults to `0`.
        :param seconds `<int>`: The relative delta of seconds. Defaults to `0`.
        :param miliseconds `<int>`: The relative delta of miliseconds. Defaults to `0`.
        :param microseconds `<int>`: The relative delta of microseconds. Defaults to `0`.
        """
        # Parse weekday
        if weekday is None:
            weekday = -1
        elif weekday != -1:
            weekday = self._parse_weekday(weekday)
            if weekday == 100:
                weekday = -1

        # Generate
        return self._new(
            # fmt: off
            cytimedelta(
                years, months, days, weeks, hours, minutes, seconds, 
                miliseconds, microseconds, year, month, day, weekday, 
                hour, minute, second, milisecond, microsecond,
            )._rsub_datetime(self._dt)
            # fmt: on
        )

    def cal_delta(
        self,
        other: str | datetime.date | datetime.datetime | pydt,
        unit: Literal["Y", "M", "W", "D", "h", "m", "s", "ms", "us"] = "D",
        inclusive: bool = False,
    ) -> cython.longlong:
        """Calcuate the `ABSOLUTE` delta between the current pydt
        and the given object based on the specified 'unit' `<int>`.

        :param other `<str/date/datetime/pydt>`: The target object.
        :param unit `<str>`: The specific time unit for the delta calculation.
        :param inclusive `<bool>`: Whether to include the endpoint (result + 1). Defaults to `False`.
        :return `<int>`: The delta value.
        """
        return self._capi_cal_delta(other, unit, inclusive)

    # . c-api - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    @cython.cdivision(True)
    def _capi_cal_delta(
        self,
        other: object,
        unit: object,
        inclusive: cython.bint = False,
    ) -> cython.longlong:
        """(cfunc) Calcuate the `ABSOLUTE` delta between the current pydt
        and the given object based on the specified 'unit' `<int>`.

        :param other `<str/date/datetime/pydt>`: The target object.
        :param unit `<str>`: The specific time unit for the delta calculation.
        :param inclusive `<bool>`: Whether to include the endpoint (result + 1). Defaults to `False`.
        :return `<int>`: The delta value.
        """
        # Parse to datetime
        dt: datetime.datetime = self._parse_timeobj(other)

        # Unit: year
        delta: cython.longlong
        if unit == "Y":
            base_y: cython.int = self._capi_year()
            targ_y: cython.int = cydt.access_dt_year(dt)
            delta = abs(base_y - targ_y)
            return delta + 1 if inclusive else delta  # exit
        # Unit: month
        if unit == "M":
            base_y: cython.int = self._capi_year()
            targ_y: cython.int = cydt.access_dt_year(dt)
            base_m: cython.int = self._capi_month()
            targ_m: cython.int = cydt.access_dt_month(dt)
            delta = abs((base_y - targ_y) * 12 + (base_m - targ_m))
            return delta + 1 if inclusive else delta  # exit

        # Calculate delta in microseconds
        diff: cython.longlong = cydt.dt_sub_dt_us(self._dt, dt)
        delta = abs(diff)

        # Unit: week
        if unit == "W":
            delta = delta // cydt.US_DAY
            if diff > 0:
                delta += cydt.ymd_weekday(
                    cydt.access_dt_year(dt),
                    cydt.access_dt_month(dt),
                    cydt.access_dt_day(dt),
                )
            else:
                delta += self._capi_weekday()
            delta = delta // 7
        # Unit: day
        elif unit == "D":
            delta = delta // cydt.US_DAY
        # Unit: hour
        elif unit == "h":
            delta = delta // cydt.US_HOUR
        # Unit: minute
        elif unit == "m":
            delta = delta // 60_000_000
        # Unit: second
        elif unit == "s":
            delta = delta // 1_000_000
        # Unit: millisecond
        elif unit == "ms":
            delta = delta // 1_000
        # Invalid unit: != microsecond
        elif unit != "us":
            raise errors.InvalidDeltaUnitError(
                "<{}>\nInvalid delta 'unit': {}. "
                "Must be one of the following <str>: "
                "['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us'].".format(
                    self.__class__.__name__, repr(unit)
                )
            )

        # Return delta
        return delta + 1 if inclusive else delta  # exit

    # Manipulate: Replace ---------------------------------------------------------------------
    def replace(
        self,
        year: cython.int = -1,
        month: cython.int = -1,
        day: cython.int = -1,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        microsecond: cython.int = -1,
        tzinfo: str | datetime.tzinfo | None = -1,
        fold: cython.int = -1,
    ) -> pydt:
        """Replacement for the current `<pydt>`.

        Equivalent to `datetime.replace()`.

        :param year `<int>`: The absolute year value. Defaults to `-1 (no change)`.
        :param month `<int>`: The absolute month value. Defaults to `-1 (no change)`.
        :param day `<int>`: The absolute day value. Defaults to `-1 (no change)`.
        :param hour `<int>`: The absolute hour value. Defaults to `-1 (no change)`.
        :param minute `<int>`: The absolute minute value. Defaults to `-1 (no change)`.
        :param second `<int>`: The absolute second value. Defaults to `-1 (no change)`.
        :param microsecond `<int>`: The absolute microsecond value. Defaults to `-1 (no change)`.
        :param tzinfo `<str/tzinfo/None>`: The timezone name or instance. Defaults to `-1 (no change)`.
        :param fold `<int>`: The ambiguous timezone fold value. Defaults to `-1 (no change)`.
        :return `<pydt>`: pydt after replacement.
        """
        # fmt: off
        return self._capi_replace(
            year, month, day, hour, minute, 
            second, microsecond, tzinfo, fold
        )
        # fmt: on

    # . c-api - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @cython.cfunc
    @cython.inline(True)
    def _capi_replace(
        self,
        year: cython.int = -1,
        month: cython.int = -1,
        day: cython.int = -1,
        hour: cython.int = -1,
        minute: cython.int = -1,
        second: cython.int = -1,
        microsecond: cython.int = -1,
        tzinfo: object = -1,
        fold: cython.int = -1,
    ) -> pydt:
        """(cfunc) Replacement for the current `<pydt>`.

        Equivalent to `datetime.replace()`.

        :param year `<int>`: The absolute year value. Defaults to `-1 (no change)`.
        :param month `<int>`: The absolute month value. Defaults to `-1 (no change)`.
        :param day `<int>`: The absolute day value. Defaults to `-1 (no change)`.
        :param hour `<int>`: The absolute hour value. Defaults to `-1 (no change)`.
        :param minute `<int>`: The absolute minute value. Defaults to `-1 (no change)`.
        :param second `<int>`: The absolute second value. Defaults to `-1 (no change)`.
        :param microsecond `<int>`: The absolute microsecond value. Defaults to `-1 (no change)`.
        :param tzinfo `<str/tzinfo/None>`: The timezone name or instance. Defaults to `-1 (no change)`.
        :param fold `<int>`: The ambiguous timezone fold value. Defaults to `-1 (no change)`.
        :return `<pydt>`: pydt after replacement.
        """
        # Parse timezone
        if tzinfo != -1:
            tzinfo = self._parse_tzinfo(tzinfo)

        # Generate
        return self._new(
            # fmt: off
            cydt.dt_replace(
                self._dt, year, month, day, hour, minute, 
                second, microsecond, tzinfo, fold
            )
            # fmt: on
        )

    # Core methods ----------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _new(self, dt: datetime.datetime) -> pydt:
        """(Internal) Create a new `<pydt>`."""
        return pydt(
            dt,
            self._default,
            self._day1st,
            self._year1st,
            self._ignoretz,
            self._fuzzy,
            self._cfg,
        )

    @cython.cfunc
    @cython.inline(True)
    def _parse_timeobj(self, timeobj: object) -> datetime.datetime:
        """(Internal) Parser time object to `<datetime.datetime>`."""
        # Type of '<datetime>'
        if cydt.is_dt_exact(timeobj):
            return timeobj
        # Type of '<str>'
        if isinstance(timeobj, str):
            return self._parse_timestr(timeobj)
        # Type of '<date>'
        if cydt.is_date(timeobj):
            if cydt.is_dt(timeobj):
                return cydt.dt_fr_dt(timeobj)  # subclase of datetime
            else:
                return cydt.dt_fr_date(timeobj, None)  # datetime.date
        # Type of `<pydt>`
        if isinstance(timeobj, pydt):
            return access_pydt_datetime(timeobj)
        # Type of '<None>'
        if timeobj is None:
            return cydt.gen_dt_now()
        # Type of `<numpy.datetime64>`
        if cydt.is_dt64(timeobj):
            return cydt.dt64_to_dt(timeobj)
        # Invalid
        raise errors.InvalidTimeObjectError(
            "<{}>\nFailed to parse 'timeobj': {}.\n"
            "Error: Unsupported data type {}.".format(
                self.__class__.__name__, repr(timeobj), type(timeobj)
            )
        )

    @cython.cfunc
    @cython.inline(True)
    def _parse_timestr(self, timestr: object) -> datetime.datetime:
        """(Internal) Parse time string to `<datetime.datetime>`."""
        try:
            return Parser(self._cfg).parse(
                timestr,
                self._default,
                self._day1st,
                self._year1st,
                self._ignoretz,
                self._fuzzy,
            )
        except Exception as err:
            raise errors.InvalidTimeObjectError(
                "<{}>\nFailed to parse 'timeobj': {}.\n"
                "Error: {}.".format(self.__class__.__name__, repr(timestr), err)
            ) from err

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_month(self, month: object) -> cython.uint:
        """(Internal) Prase month into numeric value `<int>`.

        Only accepts integer or string. Returns the month
        value if input is valid, and `100` if not the correct
        data type."""
        # Type of '<int>'
        if isinstance(month, int):
            mth: cython.int = month
            if not 1 <= mth <= 12:
                raise errors.InvalidMonthError(
                    "<{}>\nInvalid 'month' value [{}], must between "
                    "1(Jan)..12(Dec).".format(self.__class__.__name__, mth)
                )
            return mth

        # Type of '<str>'
        if isinstance(month, str):
            month_l = month.lower()
            if self._cfg is None:
                val = dict_getitem(CONFIG_MONTH, month_l)
            else:
                val = dict_getitem(self._cfg._month, month_l)
            if val == cython.NULL:
                raise errors.InvalidMonthError(
                    "<{}>\nUnable to recognize 'month' string: "
                    "{}.".format(self.__class__.__name__, repr(month))
                )
            return cython.cast(object, val)

        # Incorrect data type
        return 100

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_weekday(self, weekday: object) -> cython.uint:
        """(Internal) Prase weekday into numeric value `<int>`.

        Only accepts integer or string. Returns the weekday
        value if input is valid, and `100` if not the correct
        data type."""
        # Type of '<int>'
        if isinstance(weekday, int):
            wkd: cython.int = weekday
            if not 0 <= wkd <= 6:
                raise errors.InvalidWeekdayError(
                    "<{}>\nInvalid 'weekday' value [{}], must between "
                    "0(Monday)..6(Sunday).".format(self.__class__.__name__, wkd)
                )
            return wkd

        # Type of '<str>'
        if isinstance(weekday, str):
            weekday_l = weekday.lower()
            if self._cfg is None:
                val = dict_getitem(CONFIG_WEEKDAY, weekday_l)
            else:
                val = dict_getitem(self._cfg._weekday, weekday_l)
            if val == cython.NULL:
                raise errors.InvalidWeekdayError(
                    "<{}>\nUnable to recognize 'weekday' string: "
                    "{}.".format(self.__class__.__name__, repr(weekday))
                )
            return cython.cast(object, val)

        # Incorrect data type
        return 100

    @cython.cfunc
    @cython.inline(True)
    def _parse_tzinfo(self, tz: object) -> object:
        """(Internal) Parse 'tz' into `<datetime.tzinfo>`."""
        try:
            return parse_tzinfo(tz)
        except Exception as err:
            raise errors.InvalidTimezoneError(
                "<{}>\n{}".format(self.__class__.__name__, err)
            ) from err

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_frequency(self, freq: object) -> cython.longlong:
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
            "<{}>\nInvalid 'freq': {}. "
            "Must be one of the following `<str>`: "
            "['D', 'h', 'm', 's', 'ms', 'us'].".format(
                self.__class__.__name__, repr(freq)
            )
        )

    # Special methods: addition ---------------------------------------------------------------
    def __add__(self, other: object) -> pydt:
        # . common
        if cydt.is_delta(other):
            return self._add_cytimedelta(other)
        if isinstance(other, cytimedelta):
            return self._add_cytimedelta(other)
        if isinstance(other, RLDELTA_DTYPE):
            return self._add_relativedelta(other)
        # . unlikely numpy object
        if cydt.is_delta64(other):
            return self._add_timedelta(cydt.delta64_to_delta(other))
        # . unsupported
        return NotImplemented

    def __radd__(self, other: object) -> pydt:
        # . common
        if cydt.is_delta(other):
            return self._add_cytimedelta(other)
        if isinstance(other, cytimedelta):
            return self._add_cytimedelta(other)
        if isinstance(other, RLDELTA_DTYPE):
            return self._add_relativedelta(other)
        # . unlikely numpy object
        # TODO: this will not work since numpy does not return NotImplemented
        if cydt.is_delta64(other):
            return self._add_timedelta(cydt.delta64_to_delta(other))
        # . unsupported
        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    def _add_timedelta(self, delta: datetime.timedelta) -> pydt:
        """(Internal) Add 'timedelta' to the current `<pydt>`."""
        return self._new(cydt.dt_add_delta(self._dt, delta))

    @cython.cfunc
    @cython.inline(True)
    def _add_cytimedelta(self, delta: cytimedelta) -> pydt:
        """(Internal) Add 'cytimedelta' to the current `<pydt>`."""
        return self._new(delta._add_datetime(self._dt))

    @cython.cfunc
    @cython.inline(True)
    def _add_relativedelta(self, delta: relativedelta) -> pydt:
        """(Internal) Add 'relativedelta' to the current `<pydt>`."""
        return self._new(self._dt + delta)

    # Special methods: substraction -----------------------------------------------------------
    def __sub__(self, other: object) -> datetime.timedelta | pydt:
        # . common
        if isinstance(other, pydt):
            return self._sub_datetime(access_pydt_datetime(other))
        if cydt.is_dt(other):
            return self._sub_datetime(other)
        if cydt.is_date(other):
            return self._sub_datetime(cydt.dt_fr_date(other, None))
        if isinstance(other, str):
            return self._sub_datetime(self._parse_timestr(other))
        if cydt.is_delta(other):
            return self._sub_timedelta(other)
        if isinstance(other, cytimedelta):
            return self._sub_cytimedelta(other)
        if isinstance(other, RLDELTA_DTYPE):
            return self._sub_relativedelta(other)
        # . unlikely numpy object
        if cydt.is_dt64(other):
            return self._sub_datetime(cydt.dt64_to_dt(other))
        if cydt.is_delta64(other):
            return self._sub_timedelta(cydt.delta64_to_delta(other))
        # . unsupported
        return NotImplemented

    def __rsub__(self, other: object) -> datetime.timedelta:
        # . common
        if cydt.is_dt(other):
            return self._rsub_datetime(other)
        if cydt.is_date(other):
            return self._rsub_datetime(cydt.dt_fr_date(other, None))
        if isinstance(other, str):
            return self._rsub_datetime(self._parse_timestr(other))
        # . unlikely numpy object
        # TODO this will not work since numpy does not return NotImplemented
        if cydt.is_dt64(other):
            return self._rsub_datetime(cydt.dt64_to_dt(other))
        # . unsupported
        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    def _sub_timedelta(self, delta: datetime.timedelta) -> pydt:
        """(Internal) Substract 'timedelta' from the current `<pydt>`."""
        return self._new(cydt.dt_sub_delta(self._dt, delta))

    @cython.cfunc
    @cython.inline(True)
    def _sub_cytimedelta(self, delta: cytimedelta) -> pydt:
        """(Internal) Substract 'cytimedelta' from the current `<pydt>`."""
        return self._new(delta._rsub_datetime(self._dt))

    @cython.cfunc
    @cython.inline(True)
    def _sub_relativedelta(self, delta: relativedelta) -> pydt:
        """(Internal) Substract 'relativedelta' from the current `<pydt>`."""
        return self._new(self._dt - delta)

    @cython.cfunc
    @cython.inline(True)
    def _sub_datetime(self, dt: datetime.datetime) -> datetime.timedelta:
        """(Internal) Substract `<datetime.datetime>` from the current `<pydt>`."""
        return cydt.dt_sub_dt(self._dt, dt)

    @cython.cfunc
    @cython.inline(True)
    def _rsub_datetime(self, dt: datetime.datetime) -> datetime.timedelta:
        """(Internal) Substract the current `<pydt>` from `<datetime.datetime>`."""
        return cydt.dt_sub_dt(dt, self._dt)

    # Special methods: comparison -------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if isinstance(other, pydt):
            return self._dt == access_pydt_datetime(other)
        if cydt.is_dt(other):
            return self._dt == other
        if isinstance(other, str):
            try:
                dt: datetime.datetime = self._parse_timestr(other)
            except Exception:
                return NotImplemented
            return self._dt == dt
        if cydt.is_date(other):
            return NotImplemented
        return False

    def __ne__(self, other: object) -> bool:
        if isinstance(other, pydt):
            return self._dt != access_pydt_datetime(other)
        if cydt.is_dt(other):
            return self._dt != other
        if isinstance(other, str):
            try:
                dt: datetime.datetime = self._parse_timestr(other)
            except Exception:
                return NotImplemented
            return self._dt != dt
        if cydt.is_date(other):
            return NotImplemented
        return False

    def __gt__(self, other: object) -> bool:
        if isinstance(other, pydt):
            return self._dt > access_pydt_datetime(other)
        if cydt.is_dt(other):
            return self._dt > other
        if isinstance(other, str):
            try:
                dt: datetime.datetime = self._parse_timestr(other)
            except Exception:
                return NotImplemented
            return self._dt > dt
        if cydt.is_date(other):
            return NotImplemented
        return False

    def __ge__(self, other: object) -> bool:
        if isinstance(other, pydt):
            return self._dt >= access_pydt_datetime(other)
        if cydt.is_dt(other):
            return self._dt >= other
        if isinstance(other, str):
            try:
                dt: datetime.datetime = self._parse_timestr(other)
            except Exception:
                return NotImplemented
            return self._dt >= dt
        if cydt.is_date(other):
            return NotImplemented
        return False

    def __lt__(self, other: object) -> bool:
        if isinstance(other, pydt):
            return self._dt < access_pydt_datetime(other)
        if cydt.is_dt(other):
            return self._dt < other
        if isinstance(other, str):
            try:
                dt: datetime.datetime = self._parse_timestr(other)
            except Exception:
                return NotImplemented
            return self._dt < dt
        if cydt.is_date(other):
            return NotImplemented
        return False

    def __le__(self, other: object) -> bool:
        if isinstance(other, pydt):
            return self._dt <= access_pydt_datetime(other)
        if cydt.is_dt(other):
            return self._dt <= other
        if isinstance(other, str):
            try:
                dt: datetime.datetime = self._parse_timestr(other)
            except Exception:
                return NotImplemented
            return self._dt <= dt
        if cydt.is_date(other):
            return NotImplemented
        return False

    # Special methods: represent --------------------------------------------------------------
    def __repr__(self) -> str:
        return "<%s (datetime='%s')>" % (
            self.__class__.__name__,
            self._capi_dtisotz(),
        )

    def __str__(self) -> str:
        return self._capi_dtisotz()

    # Special methods: hash -------------------------------------------------------------------
    def __hash__(self) -> int:
        return self._hash()

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _hash(self) -> cython.int:
        """(Internal) Calculate the hashcode of the pydt `<int>`."""
        if self._hashcode == -1:
            self._hashcode = hash(("pydt", hash(self._dt)))
        return self._hashcode
