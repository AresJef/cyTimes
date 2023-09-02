# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from __future__ import annotations

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cytimes import cymath  # type: ignore
from cython.cimports.cytimes import cydatetime as cydt  # type: ignore
from cython.cimports.cytimes.cytimedelta import cytimedelta  # type: ignore
from cython.cimports.cytimes.cyparser import Parser, ParserInfo, DEFAULT_PARSERINFO  # type: ignore

np.import_array()
datetime.import_datetime()

# Python imports
import datetime, zoneinfo
from typing import Union, Type, Literal
import numpy as np, pandas as pd
from dateutil.parser._parser import parserinfo
from dateutil.relativedelta import relativedelta
from cytimes import cydatetime as cydt
from cytimes.cytimedelta import cytimedelta
from cytimes.cyparser import Parser, ParserInfo

__all__ = ["pydt", "PydtValueError"]


# Constants -----------------------------------------------------------------------------------
US_NULL: cython.longlong = -62135597000000000  # NULL dt us < 0001-01-01 00:00:00


# pydt (PythonDatetime) -----------------------------------------------------------------------
@cython.cclass
class pydt:
    _default: object
    _dayfirst: cython.bint
    _yearfirst: cython.bint
    _ignoretz: cython.bint
    _tzinfos: object
    _fuzzy: cython.bint
    _parserinfo: object
    _dt: datetime.datetime
    _hashcode: cython.int
    _year: cython.int
    _month: cython.int
    _day: cython.int
    _hour: cython.int
    _minute: cython.int
    _second: cython.int
    _microsecond: cython.int
    _quarter: cython.int
    _month_days: cython.int
    _weekday: cython.int
    _fold: cython.int
    _tzinfo: datetime.tzinfo
    _microseconds: cython.longlong

    def __init__(
        self,
        timeobj: Union[datetime.datetime, datetime.date, str, None] = None,
        default: Union[datetime.datetime, datetime.date, None] = None,
        dayfirst: cython.bint = False,
        yearfirst: cython.bint = False,
        ignoretz: cython.bint = False,
        tzinfos: Union[Type[datetime.timezone], dict[str, int], None] = None,
        fuzzy: cython.bint = False,
        parserinfo: Union[ParserInfo, parserinfo, None] = None,
    ) -> None:
        """pydt (PythonDatetime).
        A wrapper for python's datetime/date combined with parsing and delta adjustment.

        #### Time object arguments
        :param timeobj: Accepts `<datetime>`/`<date>`/`<str>`/`None`
            - `None` (default) equivalent to the current local datetime `datetime.now()`

        #### Datetime parsing arguments (Takes affect when 'timeobj' is `<str>`)
        :param default: `<datetime>` The default date, which will be used as the base to fillin missing time elements from parsed results.
            - If set to `None` (default), the current local year/month/day will be used as the default base.

        :param dayfirst: `<bool>` Whether to interpret the first value in an ambiguous date (e.g. 01/05/09) as the day (`True`) or month (`False`).
            - When `yearfirst` is `True`, this distinguishes between Y/D/M and Y/M/D.
            - If set to `None`, the `dayfirst` settings in `ParserInfo` will be used (defaults to False).

        :param yearfirst: `<bool>` Whether to interpret the first value in an ambiguous date (e.g. 01/05/09) as the year.
            - If `True`, the first number is taken as year, otherwise the last number.
            - If set to `None`, the `yearfirst` settings in `ParserInfo` will be used (defaults to False).

        :param ignoretz: `<bool>` Whether to ignore timezone info and only return naive datetime.

        :param tzinfos: Additional timezone parsing argument. Applicable when `ignoretz` is False. Accepts:
            - `<dict[str, int | tzinfo]>`: A dictionary where timezone name as the key and timezone
                offset in seconds or `tzinfo` object as the value.
            - `<TzinfoFactory>`: A callable which takes tzname and offset in seconds as arguments and
                returns a `tzinfo` object.

        :param fuzzy: `<bool>` Whether to allow fuzzy parsing.
            - If `True`, string like "Today is January 1, 2047 at 8:21:00AM" can be parsed into `2047-01-01 08:21:00`.

        :param parserinfo: `<ParserInfo>` The parserinfo to use, accepts:
            - `<ParserInfo>`: A ParserInfo instance from `cytimes.ParserInfo`.
            - `<parserinfo>`: A parserinfo instance from `dateutil.parserinfo`.
            - `None`: The default ParserInfo will be used.

        :raises `PydtValueError`: If any error occurs.

        #### Addition
        - Left/Right addition with `timedelta`, `relativedelta`, `cytimedelta`, `np.timedelta64<Left only>`
          returns `pydt`. Equivalent to `datetime + delta`.

        #### Subtraction
        - Left/Right substraction with `pydt`, `datetime`, `timestr`, `np.datetime64<Left only>`
          returns `datetime.timedelta`. Equivalent to `datetime - datetime`.
        - Left substraction with `timedelta`, `relativedelta`, `cytimedelta`, `np.timedelta64`
          returns `pydt`. Equivalent to `datetime - delta`.

        #### Comparison
        - Support direct comparison between `pydt`, `datetime`, `timestr`.
          Equivalent to `datetime <op> datetime`.

        #### Hash
        - Even when the underlying datetime is the same, the hashcode of `pydt` will be
          different from `datetime`.
        """

        # Settings
        self._default = default
        self._dayfirst = dayfirst
        self._yearfirst = yearfirst
        self._ignoretz = ignoretz
        self._tzinfos = tzinfos
        self._fuzzy = fuzzy
        self._parserinfo = parserinfo
        # To datetime
        try:
            if timeobj is None:
                self._dt = cydt.gen_dt_local()
            else:
                self._dt = self._to_datetime(timeobj)
        except PydtValueError:
            raise
        except Exception as err:
            raise PydtValueError("<pydt> %s" % err) from err
        # cache
        self._hashcode = -1
        self._year = -1
        self._month = -1
        self._day = -1
        self._hour = -1
        self._minute = -1
        self._second = -1
        self._microsecond = -1
        self._quarter = -1
        self._month_days = -1
        self._weekday = -1
        self._fold = -1
        self._tzinfo = None
        self._microseconds = US_NULL

    # Static methods
    @staticmethod
    def from_ordinal(ordinal: int) -> pydt:
        """Create `pydt` from ordinal.
        :param ordinal: `<int>` ordinal.
        :return: `<pydt>`.
        """
        return pydt(cydt.dt_fr_ordinal(ordinal))

    @staticmethod
    def from_timestamp(
        timestamp: Union[int, float],
        tzinfo: datetime.tzinfo = None,
    ) -> pydt:
        """Create `pydt` from timestamp.
        :param timestamp: `<int>`/`<float>` timestamp.
        :param tzinfo: `<datetime.tzinfo>` timezone info, default `None`.
        :return: `<pydt>`.
        """
        return pydt(cydt.dt_fr_timestamp(timestamp, tzinfo))

    @staticmethod
    def from_seconds(
        seconds: float,
        tzinfo: datetime.tzinfo = None,
        fold: int = 0,
    ) -> pydt:
        """Create `pydt` from total seconds.
        :param seconds: `<float>` totla seconds after EPOCH.
        :param tzinfo: `<datetime.tzinfo>` timezone info, default `None`.
        :param fold: `<int>` fold, default `0`.
        :return: `<pydt>`.
        """
        return pydt(cydt.dt_fr_seconds(seconds, tzinfo, fold))

    @staticmethod
    def from_microseconds(
        microseconds: int,
        tzinfo: datetime.tzinfo = None,
        fold: int = 0,
    ) -> pydt:
        """Create `pydt` from total microseconds.
        :param microseconds: `<int>` total microseconds after EPOCH.
        :param tzinfo: `<datetime.tzinfo>` timezone info, default `None`.
        :param fold: `<int>` fold, default `0`.
        :return: `<pydt>`.
        """
        return pydt(cydt.dt_fr_microseconds(microseconds, tzinfo, fold))

    # Data type
    @property
    def dt(self) -> datetime.datetime:
        "Access as `datetime.datetime`."
        return self._dt

    @property
    def dtiso(self) -> str:
        "Access `datetime.datetime` in ISO format as `str`."
        return cydt.dt_to_isoformat(self._dt)

    @property
    def dtisotz(self) -> str:
        "Access `datetime.datetime` in ISO format with timezone as `str`."
        return cydt.dt_to_isoformat_tz(self._dt)

    @property
    def date(self) -> datetime.date:
        "Access as `datetime.date`."
        return cydt.gen_date(self._get_year(), self._get_month(), self._get_day())

    @property
    def dateiso(self) -> str:
        "Access `datetime.date` in ISO format as `str`."
        return cydt.date_to_isoformat(self.date)

    @property
    def time(self) -> datetime.time:
        "Access as `datetime.time`."
        return cydt.gen_time(
            self._get_hour(),
            self._get_minute(),
            self._get_second(),
            self._get_microsecond(),
            None,
            0,
        )

    @property
    def timeiso(self) -> str:
        "Access `datetime.time` in ISO format as `str`."
        return cydt.time_to_isoformat(self.time)

    @property
    def timetz(self) -> datetime.time:
        "Access as `datetime.time` with timezone."
        return cydt.gen_time(
            self._get_hour(),
            self._get_minute(),
            self._get_second(),
            self._get_microsecond(),
            self._get_tzinfo(),
            self._get_fold(),
        )

    @property
    def timeisotz(self) -> str:
        "Access `datetime.time` in ISO format with timezone as `str`."
        return cydt.time_to_isoformat_tz(self.timetz)

    @property
    def ts(self) -> pd.Timestamp:
        "Access as `pandas.Timestamp`."
        return pd.Timestamp(self._dt)

    @property
    def dt64(self) -> np.datetime64:
        """Access as `numpy.datetime64`.
        Timezone will be normalized to UTC (naive) with unit of 'us'.
        """
        return np.datetime64(cydt.dt_to_microseconds_utc(self._dt), "us")

    @property
    def ordinal(self) -> int:
        "Access `datetime.date` in ordinal as `<int>`."
        return cydt.to_ordinal(self._dt)

    @property
    def seconds(self) -> float:
        """Access `datetime.datetime` in total seconds (naive) after EPOCH as `<float>`.
        For datetime out of `Timestamp` range, microsecond percision will be lost.
        """
        return cydt.dt_to_seconds(self._dt)

    @property
    def seconds_utc(self) -> float:
        """Access `datetime.datetime` in total seconds after EPOCH as `<float>`.
        For datetime out of `Timestamp` range, microsecond percision will be lost.
        - If timezone-aware, return total seconds in UTC.
        - If timezone-naive, requivalent to `seconds`.

        #### Notice
        This should `NOT` be treated as timestamp, but rather adjustment of the
        total seconds of the datetime from utcoffset.
        """
        return cydt.dt_to_seconds_utc(self._dt)

    @property
    def microseconds(self) -> int:
        "Access `datetime.datetime` in total microseconds (naive) after EPOCH as `<int>`."
        return self._get_microseconds()

    @property
    def microseconds_utc(self) -> int:
        """Access `datetime.datetime` in total microseconds (naive) after EPOCH as `<int>`.
        - If timezone-aware, return total microseconds in UTC.
        - If timezone-naive, requivalent to `microseconds`.

        #### Notice
        This should `NOT` be treated as timestamp, but rather adjustment of the
        total microseconds of the datetime from utcoffset.
        """
        return cydt.dt_to_microseconds_utc(self._dt)

    @property
    def timestamp(self) -> float:
        "Access `datetime.datetime` in timestamp as `<int>`."
        return cydt.dt_to_timestamp(self._dt)

    # Absolute
    @property
    def year(self) -> int:
        "`<int>` The year attribute."
        return self._get_year()

    @property
    def month(self) -> int:
        "`<int>` The month attribute."
        return self._get_month()

    @property
    def day(self) -> int:
        "`<int>` The day attribute."
        return self._get_day()

    @property
    def hour(self) -> int:
        "`<int>` The hour attribute."
        return self._get_hour()

    @property
    def minute(self) -> int:
        "`<int>` The minute attribute."
        return self._get_minute()

    @property
    def second(self) -> int:
        "`<int>` The second attribute."
        return self._get_second()

    @property
    def microsecond(self) -> int:
        "`<int>` The microsecond attribute."
        return self._get_microsecond()

    @property
    def tzinfo(self) -> datetime.tzinfo:
        "`<datetime.tzinfo>` The tzinfo attribute."
        return self._get_tzinfo()

    # Calendar
    @property
    def is_leapyear(self) -> bool:
        "`<bool>` Whether is a leap year."
        return cydt.get_is_leapyear(self._dt)

    @property
    def days_in_year(self) -> int:
        "`<int>` Number of days in the year."
        return cydt.get_days_in_year(self._dt)

    @property
    def days_bf_year(self) -> int:
        "`<int>` Number of days before January 1st of the year."
        return cydt.get_days_bf_year(self._dt)

    @property
    def days_of_year(self) -> int:
        "`<int>` Number of days into the year."
        return cydt.get_days_of_year(self._dt)

    @property
    def quarter(self) -> int:
        "`<int>` The quarter of the date."
        return self._get_quarter()

    @property
    def days_in_quarter(self) -> int:
        "`<int>` Number of days in the quarter."
        return cydt.get_days_in_quarter(self._dt)

    @property
    def days_bf_quarter(self) -> int:
        "`<int>` Number of days in the year preceding first day of the quarter."
        return cydt.get_days_bf_quarter(self._dt)

    @property
    def days_of_quarter(self) -> int:
        "`<int>` Number of days into the quarter."
        return cydt.get_days_of_quarter(self._dt)

    @property
    def days_in_month(self) -> int:
        "`<int>` Number of days in the month."
        return self._get_month_days()

    @property
    def days_bf_month(self) -> int:
        "`<int>` Number of days in the year preceding first day of the month."
        return cydt.get_days_bf_month(self._dt)

    @property
    def weekday(self) -> int:
        "`<int>` The the weekday, where Monday == 0 ... Sunday == 6."
        return self._get_weekday()

    @property
    def isoweekday(self) -> int:
        "`<int>` The the ISO weekday, where Monday == 1 ... Sunday == 7."
        return self._get_weekday() + 1

    @property
    def isoweek(self) -> int:
        "`<int>` The ISO calendar week number."
        return cydt.get_isoweek(self._dt)

    @property
    def isoyear(self) -> int:
        "`<int>` The ISO calendar year."
        return cydt.get_isoyear(self._dt)

    @property
    def isocalendar(self) -> tuple[int, int, int]:
        "`<tuple[int, int, int]>` The ISO calendar year, week number and weekday."
        iso = cydt.get_isocalendar(self._dt)
        return (iso.year, iso.week, iso.weekday)

    # Time manipulation
    @property
    def start_time(self) -> pydt:
        "Start time (00:00:00.000000) of the current datetime."
        return self._new(
            cydt.gen_dt(
                self._get_year(),
                self._get_month(),
                self._get_day(),
                0,
                0,
                0,
                0,
                self._get_tzinfo(),
                self._get_fold(),
            )
        )

    @property
    def end_time(self) -> pydt:
        "End time (23:59:59.999999) of the current datetime."
        return self._new(
            cydt.gen_dt(
                self._get_year(),
                self._get_month(),
                self._get_day(),
                23,
                59,
                59,
                999999,
                self._get_tzinfo(),
                self._get_fold(),
            )
        )

    # Day manipulation
    @property
    def tomorrow(self) -> pydt:
        "Tomorrow."
        return self._add_days(1)

    @property
    def yesterday(self) -> pydt:
        "Yesterday."
        return self._add_days(-1)

    # Weekday manipulation
    @property
    def monday(self) -> pydt:
        "Monday of the current week."
        return self._add_days(-self._get_weekday())

    @property
    def tuesday(self) -> pydt:
        "Tuesday of the current week."
        return self._add_days(-self._get_weekday() + 1)

    @property
    def wednesday(self) -> pydt:
        "Wednesday of the current week."
        return self._add_days(-self._get_weekday() + 2)

    @property
    def thursday(self) -> pydt:
        "Thursday of the current week."
        return self._add_days(-self._get_weekday() + 3)

    @property
    def friday(self) -> pydt:
        "Friday of the current week."
        return self._add_days(-self._get_weekday() + 4)

    @property
    def saturday(self) -> pydt:
        "Saturday of the current week."
        return self._add_days(-self._get_weekday() + 5)

    @property
    def sunday(self) -> pydt:
        "Sunday of the current week."
        return self._add_days(-self._get_weekday() + 6)

    def curr_week(self, weekday: Union[int, str]) -> pydt:
        """Specific weekday of the currect week.
        :param weekday: `<int>` 0 as Monday to 6 as Sunday / `<str>` 'mon', 'tuesday', etc.
        """
        return self._curr_week(weekday)

    def next_week(self, weekday: Union[int, str, None] = None) -> pydt:
        """Specific weekday of the next week.
        :param weekday: `<int>` 0 as Monday to 6 as Sunday / `<str>` 'mon', 'tuesday', etc.
            If set to `None` (default), will move to the same weekday of the next week.
        """
        return self._to_week(1, weekday)

    def last_week(self, weekday: Union[int, str, None] = None) -> pydt:
        """Specific weekday of the last week.
        :param weekday: `<int>` 0 as Monday to 6 as Sunday / `<str>` 'mon', 'tuesday', etc.
            If set to `None` (default), will move to the same weekday of the last week.
        """
        return self._to_week(-1, weekday)

    def to_week(self, offset: int, weekday: Union[int, str, None] = None) -> pydt:
        """Specific weekday of the week (+/-) offset.
        :param offset: `<int>` number of weeks offset from the current week.
        :param weekday: `<int>` 0 as Monday to 6 as Sunday / `<str>` 'mon', 'Tuesday', etc.
            If set to `None` (default), will move to the same weekday of the week (+/-) offset.
        """
        return self._to_week(offset, weekday)

    def is_weekday(self, weekday: Union[int, str]) -> bool:
        """Whether the current datetime is a specific weekday.
        :param weekday: `<int>` 0 as Monday to 6 as Sunday / `<str>` 'mon', 'tuesday', etc.
        """
        if weekday is None:
            return True
        else:
            return self._get_weekday() == self._parse_weekday(weekday)

    # Month manipulation
    @property
    def month_1st(self) -> pydt:
        "First day of the current month."
        return self._new(
            cydt.gen_dt(
                self._get_year(),
                self._get_month(),
                1,
                self._get_hour(),
                self._get_minute(),
                self._get_second(),
                self._get_microsecond(),
                self._get_tzinfo(),
                self._get_fold(),
            )
        )

    def is_month_1st(self) -> bool:
        "`<bool>` Whether the current datetime is the first day of the month."
        return self._get_day() == 1

    @property
    def month_lst(self) -> pydt:
        "Last day of the current month."
        return self._new(
            cydt.gen_dt(
                self._get_year(),
                self._get_month(),
                self._get_month_days(),
                self._get_hour(),
                self._get_minute(),
                self._get_second(),
                self._get_microsecond(),
                self._get_tzinfo(),
                self._get_fold(),
            )
        )

    def is_month_lst(self) -> bool:
        "`<bool>` Whether the current datetime is the last day of the month."
        return self._get_day() == self._get_month_days()

    def curr_month(self, day: int) -> pydt:
        """Specifc day of the current month.
        :param day: `<int>` day of the current month.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the current month.
        """
        return self._curr_month(day)

    def next_month(self, day: int = 0) -> pydt:
        """Specifc day of the next month.
        :param day: `<int>` day of the next month.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the next month.
        """
        return self._to_month(1, day)

    def last_month(self, day: int = 0) -> pydt:
        """Specifc day of the last month.
        :param day: `<int>` day of the last month.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the last month.
        """
        return self._to_month(-1, day)

    def to_month(self, offset: int, day: int = 0) -> pydt:
        """Specifc day of the month (+/-) offset.
        :param offset: `<int>` number of months offset from the current month.
        :param day: `<int>` day of the month (+/-) offset
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the month (+/-) offset.
        """
        return self._to_month(offset, day)

    def is_month(self, month: Union[int, str]) -> bool:
        """Whether the current datetime is a specific month.
        :param month: `<int>` 1 as January to 12 as December / `<str>` 'jan', 'February', etc.
        """
        if month is None:
            return True
        else:
            return self._get_month() == self._parse_month(month)

    # Quarter manipulation
    @property
    def quarter_1st(self) -> pydt:
        "First day of the current quarter."
        return self._new(
            cydt.gen_dt(
                self._get_year(),
                cydt.get_quarter_1st_month(self._dt),
                1,
                self._get_hour(),
                self._get_minute(),
                self._get_second(),
                self._get_microsecond(),
                self._get_tzinfo(),
                self._get_fold(),
            )
        )

    def is_quarter_1st(self) -> bool:
        "`<bool>` Whether the current datetime is the first day of the quarter."
        return self._get_day() == 1 and cydt.get_month(
            self._dt
        ) == cydt.get_quarter_1st_month(self._dt)

    @property
    def quarter_lst(self) -> pydt:
        "Last day of the current quarter."
        year: cython.int = self._get_year()
        month: cython.int = cydt.get_quarter_lst_month(self._dt)
        day: cython.int = cydt.days_in_month(year, month)
        return self._new(
            cydt.gen_dt(
                year,
                month,
                day,
                self._get_hour(),
                self._get_minute(),
                self._get_second(),
                self._get_microsecond(),
                self._get_tzinfo(),
                self._get_fold(),
            )
        )

    def is_quarter_lst(self) -> bool:
        "`<bool>` Whether the current datetime is the last day of the quarter."
        return self._get_day() == cydt.get_days_in_month(
            self._dt
        ) and self._get_month() == cydt.get_quarter_lst_month(self._dt)

    def curr_quarter(self, month: int, day: int = 0) -> pydt:
        """Specifc day of the current quarter.
        :param month: `<int>` 1 as 1st month of to 3 as 3rd month of the quarter.
        :param day: `<int>` day of the quarter month.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the current quarter month.
        """
        return self._curr_quarter(month, day)

    def next_quarter(self, month: int, day: int = 0) -> pydt:
        """Specifc day of the next quarter.
        :param month: `<int>` 1 as 1st month of to 3 as 3rd month of the quarter.
        :param day: `<int>` day of the quarter month.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the next quarter month.
        """
        return self._to_quarter(1, month, day)

    def last_quarter(self, month: int, day: int = 0) -> pydt:
        """Specifc day of the last quarter.
        :param month: `<int>` 1 as 1st month of to 3 as 3rd month of the quarter.
        :param day: `<int>` day of the quarter month.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the last quarter month.
        """
        return self._to_quarter(-1, month, day)

    def to_quarter(self, offset: int, month: int, day: int = 0) -> pydt:
        """Specifc day of the quarter (+/-) offset.
        :param offset: `<int>` number of quarters offset from the current quarter.
        :param month: `<int>` 1 as 1st month of to 3 as 3rd month of the quarter.
        :param day: `<int>` day of the quarter month.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the quarter (+/-) offset.
        """
        return self._to_quarter(offset, month, day)

    def is_quarter(self, quarter: int) -> bool:
        """Whether the current datetime is a specific quarter.
        :param quarter: `<int>` 1 as 1st quarter to 4 as 4th quarter.
        """
        return self._get_quarter() == quarter

    # Year manipulation
    @property
    def year_1st(self) -> pydt:
        "First day of the current year."
        return self._new(
            cydt.gen_dt(
                self._get_year(),
                1,
                1,
                self._get_hour(),
                self._get_minute(),
                self._get_second(),
                self._get_microsecond(),
                self._get_tzinfo(),
                self._get_fold(),
            )
        )

    def is_year_1st(self) -> bool:
        "`<bool>` Whether the current datetime is the first day of the year."
        return self._get_month() == 1 and self._get_day() == 1

    @property
    def year_lst(self) -> pydt:
        "Last day of the current year."
        return self._new(
            cydt.gen_dt(
                self._get_year(),
                12,
                31,
                self._get_hour(),
                self._get_minute(),
                self._get_second(),
                self._get_microsecond(),
                self._get_tzinfo(),
                self._get_fold(),
            )
        )

    def is_year_lst(self) -> bool:
        "`<bool>` Whether the current datetime is the last day of the year."
        return self._get_month() == 12 and self._get_day() == 31

    def curr_year(self, month: Union[int, str, None] = None, day: int = 0) -> pydt:
        """Specifc month and day of the current year.
        :param month: `<int>` 1 as January to 12 as December / `<str>` 'jan', 'February', etc.
            If set to `None` (default), month will not be affected.
        :param day: `<int>` day of the month for the year.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the month.
        """
        return self._curr_year(month, day)

    def next_year(self, month: Union[int, str, None] = None, day: int = 0) -> pydt:
        """Specifc month and day of the next year.
        :param month: `<int>` 1 as January to 12 as December / `<str>` 'jan', 'February', etc.
            If set to `None` (default), month will not be affected.
        :param day: `<int>` day of the month for the next year.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the month.
        """
        return self._to_year(1, month, day)

    def last_year(self, month: Union[int, str, None] = None, day: int = 0) -> pydt:
        """Specifc month and day of the last year.
        :param month: `<int>` 1 as January to 12 as December / `<str>` 'jan', 'February', etc.
            If set to `None` (default), month will not be affected.
        :param day: `<int>` day of the month for the last year.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the month.
        """
        return self._to_year(-1, month, day)

    def to_year(
        self,
        offset: int,
        month: Union[int, str, None] = None,
        day: int = 0,
    ) -> pydt:
        """Specifc month and day of the year (+/-) offset.
        :param offset: `<int>` number of years offset from the current year.
        :param month: `<int>` 1 as January to 12 as December / `<str>` 'jan', 'February', etc.
            If set to `None` (default), month will not be affected.
        :param day: `<int>` day of the month for the year (+/-) offset.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the month.
        """
        return self._to_year(offset, month, day)

    def is_year(self, year: int) -> bool:
        """Whether the current datetime is a specific year.
        :param year: `<int>` year.
        """
        return self._get_year() == year

    # Timezone manipulation
    @property
    def tz_available(self) -> set[str]:
        "`<set[str]>` All available timezone names accept by localize/convert/switch methods."
        return zoneinfo.available_timezones()

    def tz_localize(self, tz: Union[datetime.timezone, str, None]) -> pydt:
        """Localize to a specific timezone. Equivalent to `datetime.replace(tzinfo=tz)`.

        :param tz: `<datetime.timezone>`/`<str (timezone name)>`/`None (remove timezone)`.
            - `<datetime.timezone>`: The timezone to localize to. Only native python
              timezone is supported, which means `tzinfo` from pytz and dateutil will
              `NOT` work properly.
            - `<str>`: The timezone name to localize to. Must be one of the timezone
              names in `pydt.tz_available`.
            - `None`: Remove timezone awareness.
        """
        return self._new(self._tz_localize(self._dt, tz))

    def tz_convert(self, tz: Union[datetime.timezone, str, None]) -> pydt:
        """Convert to a specific timezone. Equivalent to `datetime.astimezone(tz)`.

        :param tz: `<datetime.timezone>`/`<str (timezone name)>`/`None (local timezone)`.
            - `<datetime.timezone>`: The timezone to convert to. Only native python
              timezone is supported, which means `tzinfo` from pytz and dateutil will
              `NOT` work properly.
            - `<str>`: The timezone name to convert to. Must be one of the timezone
              names in `pydt.tz_available`.
            - `None`: Convert to system's local timezone.
        """
        return self._new(self._tz_convert(self._dt, tz))

    def tz_switch(
        self,
        targ_tz: Union[datetime.timezone, str, None],
        base_tz: Union[datetime.timezone, str] = None,
        naive: bool = False,
    ) -> pydt:
        """Switch from base timezone to target timezone.

        - When `datetime` is timezone-aware, this method is equivalent to `tz_convert`,
          and only the `targ_tz` parameter is required.
        - When `datetime` is timezone-naive, this method is equivalent 1st `tz_localize`
          to the `base_tz`, and then `tz_convert` to the `targ_tz`.

        :param targ_tz: `<datetime.tzinfo>`/`<str (timezone name)>`/`None (local timezone)`.
            - `<datetime.timezone>`: The timezone to convert to. Only native python
              timezone is supported, which means `tzinfo` from pytz and dateutil will
              `NOT` work properly.
            - `<str>`: The timezone name to convert to. Must be one of the timezone
              names in `pydt.tz_available`.
            - `None`: Convert to system's local timezone.

        :param base_tz: `<datetime.tzinfo>`/`<str (timezone name)>`.
            - `<datetime.timezone>`: The timezone to localize to. Only native python
              timezone is supported, which means `tzinfo` from pytz and dateutil will
              `NOT` work properly.
            - `<str>`: The timezone name to localize to. Must be one of the timezone
              names in `pydt.tz_available`.
            - * Notice: `None` is invalid when `datetime` is timezone-naive.

        :param naive: `<bool>` whether to convert to timezone-naive after conversion.
        """
        return self._new(self._tz_switch(self._dt, targ_tz, base_tz, naive))

    # Frequency manipulation
    def round(self, freq: Literal["D", "h", "m", "s", "ms", "us"]) -> pydt:
        """Perform round operation to specified freqency.
        Similar to `pandas.DatetimeIndex.round()`.

        :param freq: `<str>` frequency to round to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond
        """
        return self._round_frequency(freq)

    def ceil(self, freq: Literal["D", "h", "m", "s", "ms", "us"]) -> pydt:
        """Perform ceil operation to specified freqency.
        Similar to `pandas.DatetimeIndex.ceil()`.

        :param freq: `<str>` frequency to ceil to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond
        """
        return self._ceil_frequency(freq)

    def floor(self, freq: Literal["D", "h", "m", "s", "ms", "us"]) -> pydt:
        """Perform floor operation to specified freqency.
        Similar to `pandas.DatetimeIndex.floor()`.

        :param freq: `<str>` frequency to floor to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond
        """
        return self._floor_frequency(freq)

    # Delta adjustment
    def delta(
        self,
        years: int = 0,
        months: int = 0,
        days: int = 0,
        weeks: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        microseconds: int = 0,
    ) -> pydt:
        """Adjustment with delta. Equivalent to `pydt + cytimedelta`.

        :param years: `<int>` The relative number of years.
        :param months: `<int>` The relative number of months.
        :param days: `<int>` The relative number of days.
        :param weeks: `<int>` The relative number of weeks.
        :param hours: `<int>` The relative number of hours.
        :param minutes: `<int>` The relative number of minutes.
        :param seconds: `<int>` The relative number of seconds.
        :param microseconds: `<int>` The relative number of microseconds.
        """
        try:
            return self._delta(
                years, months, days, weeks, hours, minutes, seconds, microseconds
            )
        except PydtValueError:
            raise
        except Exception as err:
            raise PydtValueError("<pydt> %s" % err) from err

    # Replace adjustment
    def replace(
        self,
        year: int = -1,
        month: int = -1,
        day: int = -1,
        hour: int = -1,
        minute: int = -1,
        second: int = -1,
        microsecond: int = -1,
        tzinfo: Union[datetime.tzinfo, None] = -1,
        fold: int = -1,
    ) -> pydt:
        """Replace the current datetime. Equivalent to `datetime.replace()`.

        :param year: `<int>` The absolute year, -1 means `IGNORE`.
        :param month: `<int>` The absolute month, -1 means `IGNORE`.
        :param day: `<int>` The absolute day, -1 means `IGNORE`.
        :param hour: `<int>` The absolute hour, -1 means `IGNORE`.
        :param minute: `<int>` The absolute minute, -1 means `IGNORE`.
        :param second: `<int>` The absolute second, -1 means `IGNORE`.
        :param microsecond: `<int>` The absolute microsecond, -1 means `IGNORE`.
        :param tzinfo: `<datetime.tzinfo>`/`<int>`/`<None>` The tzinfo, -1 means `IGNORE`.
        :param fold: `<int>` The fold, -1 means `IGNORE`.
        """
        try:
            return self._replace(
                year, month, day, hour, minute, second, microsecond, tzinfo, fold
            )
        except PydtValueError:
            raise
        except Exception as err:
            raise PydtValueError("<pydt> %s" % err) from err

    # Between calculation
    def between(
        self,
        other: Union[datetime.datetime, datetime.date, pydt, str],
        unit: Literal["Y", "M", "W", "D", "h", "m", "s", "ms", "us"] = "D",
        inclusive: bool = False,
    ) -> int:
        """Calculate the `ABSOLUTE` delta between two time in the given unit.

        :param other: `<datetime.datetime>`/`<datetime.date>`/`<pydt>`/`<str>` The other time to compare with.
        :param unit: `<str>` The unit to calculate the delta, accepts: 'Y', 'M', 'D', 'h', 'm', 's', 'ms', 'us'.
        :param inclusive: `<bool>` Setting to `True` will add 1 to the final result.
            Take 'Y' (year) as an example. If `True`, 'Y' (year) delta between 2023-01-01 and
            2024-01-01 will be 2. If `False`, delta between 2023-01-01 and 2024-01-01 will be 1.
        :return: `<int>` The `ABSOLUTE` delta between two time (positive `<int>`).
        """
        try:
            return self._between(other, unit, inclusive)
        except PydtValueError:
            raise
        except Exception as err:
            raise PydtValueError("<pydt> %s" % err) from err

    # Core methods
    @cython.cfunc
    @cython.inline(True)
    def _new(self, dt: datetime.datetime) -> pydt:
        return pydt(
            dt,
            self._default,
            self._dayfirst,
            self._yearfirst,
            self._ignoretz,
            self._tzinfos,
            self._fuzzy,
            self._parserinfo,
        )

    @cython.cfunc
    @cython.inline(True)
    def _to_datetime(self, timeobj: object) -> datetime.datetime:
        # Is excat datetime
        if cydt.is_dt_exact(timeobj):
            return timeobj

        # Is string
        if isinstance(timeobj, str):
            return self._parse_datetime(timeobj)

        # Is date / subclass of datetime
        if cydt.is_date(timeobj):
            if cydt.is_dt(timeobj):
                return cydt.dt_fr_dt(timeobj)  # subclass of datetime.datetime
            else:
                return cydt.dt_fr_date(timeobj)  # datetime.date

        # Is time
        if cydt.is_time(timeobj):
            if cydt.is_date(self._default):  # combine default date with time
                return cydt.dt_fr_date_time(self._default, timeobj)
            else:  # combine local date with time
                return cydt.dt_fr_time(timeobj)

        # Is datetime64
        if cydt.is_dt64(timeobj):
            return cydt.dt64_to_dt(timeobj)

        # Invalid
        raise PydtValueError(
            "<pydt> Unsupported data type: %s %s" % (type(timeobj), repr(timeobj))
        )

    @cython.cfunc
    @cython.inline(True)
    def _parse_datetime(self, timestr: str) -> datetime.datetime:
        return Parser(self._parserinfo)._parse(
            timestr,
            self._default,
            self._dayfirst,
            self._yearfirst,
            self._ignoretz,
            self._tzinfos,
            self._fuzzy,
        )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_year(self) -> cython.int:
        if self._year == -1:
            self._year = cydt.get_year(self._dt)
        return self._year

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_month(self) -> cython.int:
        if self._month == -1:
            self._month = cydt.get_month(self._dt)
        return self._month

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_day(self) -> cython.int:
        if self._day == -1:
            self._day = cydt.get_day(self._dt)
        return self._day

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_hour(self) -> cython.int:
        if self._hour == -1:
            self._hour = cydt.get_dt_hour(self._dt)
        return self._hour

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_minute(self) -> cython.int:
        if self._minute == -1:
            self._minute = cydt.get_dt_minute(self._dt)
        return self._minute

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_second(self) -> cython.int:
        if self._second == -1:
            self._second = cydt.get_dt_second(self._dt)
        return self._second

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_microsecond(self) -> cython.int:
        if self._microsecond == -1:
            self._microsecond = cydt.get_dt_microsecond(self._dt)
        return self._microsecond

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_quarter(self) -> cython.int:
        if self._quarter == -1:
            self._quarter = cydt.get_quarter(self._dt)
        return self._quarter

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_weekday(self) -> cython.int:
        if self._weekday == -1:
            self._weekday = cydt.get_weekday(self._dt)
        return self._weekday

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_month_days(self) -> cython.int:
        if self._month_days == -1:
            self._month_days = cydt.get_days_in_month(self._dt)
        return self._month_days

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _get_fold(self) -> cython.int:
        if self._fold == -1:
            self._fold = cydt.get_dt_fold(self._dt)
        return self._fold

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _get_tzinfo(self) -> datetime.tzinfo:
        if self._tzinfo is None:
            self._tzinfo = cydt.get_dt_tzinfo(self._dt)
        return self._tzinfo

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _get_microseconds(self) -> cython.longlong:
        if self._microseconds == US_NULL:
            self._microseconds = cydt.dt_to_microseconds(self._dt)
        return self._microseconds

    @cython.cfunc
    @cython.inline(True)
    def _add_days(self, days: cython.int) -> pydt:
        return self._new(cytimedelta(days=days)._add_date_time(self._dt))

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_weekday(self, weekday: object) -> cython.int:
        wday: cython.int
        if isinstance(weekday, int):
            wday = weekday
        elif isinstance(weekday, str):
            wday = DEFAULT_PARSERINFO.weekday(weekday)
        else:
            wday = -1
        if not 0 <= wday <= 6:
            raise PydtValueError("<pydt> Invalid weekday: %s" % repr(weekday))
        return wday

    @cython.cfunc
    @cython.inline(True)
    def _curr_week(self, weekday: object) -> pydt:
        if weekday is None:
            return self
        else:
            return self._add_days(self._parse_weekday(weekday) - self._get_weekday())

    @cython.cfunc
    @cython.inline(True)
    def _to_week(self, offset: cython.int, weekday: object) -> pydt:
        if weekday is None:
            return self._add_days(offset * 7)
        else:
            return self._add_days(
                self._parse_weekday(weekday) + offset * 7 - self._get_weekday()
            )

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_month(self, month: object) -> cython.int:
        mth: cython.int
        if isinstance(month, int):
            mth = month
        elif isinstance(month, str):
            mth = DEFAULT_PARSERINFO.month(month)
        else:
            mth = -1
        if not 1 <= mth <= 12:
            raise PydtValueError("<pydt> Invalid month: %s" % repr(month))
        return mth

    @cython.cfunc
    @cython.inline(True)
    def _curr_month(self, day: cython.int) -> pydt:
        if day < 1:
            return self
        else:
            return self._new(cytimedelta(day=day)._add_date_time(self._dt))

    @cython.cfunc
    @cython.inline(True)
    def _to_month(self, offset: cython.int, day: cython.int) -> pydt:
        dt: datetime.datetime
        if day < 1:
            dt = cytimedelta(months=offset)._add_date_time(self._dt)
        else:
            dt = cytimedelta(months=offset, day=day)._add_date_time(self._dt)
        return self._new(dt)

    @cython.cfunc
    @cython.inline(True)
    def _curr_quarter(self, month: cython.int, day: cython.int) -> pydt:
        # Validate month
        if not 1 <= month <= 3:
            raise PydtValueError(
                "<pydt> Invalid quarter month: %s. Accepts `<int>` 1-3." % repr(month)
            )

        # Convert
        quarter: cython.int = self._get_quarter()
        month = quarter * 3 - 3 + (month % 3 or 3)
        dt: datetime.datetime
        if day < 1:
            dt = cytimedelta(month=month)._add_date_time(self._dt)
        else:
            dt = cytimedelta(month=month, day=day)._add_date_time(self._dt)
        return self._new(dt)

    @cython.cfunc
    @cython.inline(True)
    def _to_quarter(
        self,
        offset: cython.int,
        month: cython.int,
        day: cython.int,
    ) -> pydt:
        # Validate month
        if not 1 <= month <= 3:
            raise PydtValueError(
                "<pydt> Invalid quarter month: %s. Accepts `<int>` 1-3." % repr(month)
            )

        # Convert
        quarter: cython.int = self._get_quarter()
        month = quarter * 3 - 3 + (month % 3 or 3)
        dt: datetime.datetime
        if day < 1:
            dt = cytimedelta(months=offset * 3, month=month)._add_date_time(self._dt)
        else:
            dt = cytimedelta(months=offset * 3, month=month, day=day)._add_date_time(
                self._dt
            )
        return self._new(dt)

    @cython.cfunc
    @cython.inline(True)
    def _curr_year(self, month: object, day: cython.int) -> pydt:
        if month is None:
            return self._curr_month(day)

        dt: datetime.datetime
        if day < 1:
            dt = cytimedelta(month=self._parse_month(month))._add_date_time(self._dt)
        else:
            dt = cytimedelta(month=self._parse_month(month), day=day)._add_date_time(
                self._dt
            )
        return self._new(dt)

    @cython.cfunc
    @cython.inline(True)
    def _to_year(self, offset: cython.int, month: object, day: cython.int) -> pydt:
        dt: datetime.datetime
        if month is None:
            if day < 1:
                dt = cytimedelta(years=offset)._add_date_time(self._dt)
            else:
                dt = cytimedelta(years=offset, day=day)._add_date_time(self._dt)
        else:
            if day < 1:
                dt = cytimedelta(
                    years=offset, month=self._parse_month(month)
                )._add_date_time(self._dt)
            else:
                dt = cytimedelta(
                    years=offset, month=self._parse_month(month), day=day
                )._add_date_time(self._dt)
        return self._new(dt)

    @cython.cfunc
    @cython.inline(True)
    def _parse_tzinfo(self, tz: object) -> datetime.tzinfo:
        if isinstance(tz, str):
            try:
                return zoneinfo.ZoneInfo(tz)
            except Exception as err:
                raise PydtValueError("<pydt> Invalid timezone: %s" % repr(tz)) from err
        if tz is None:
            return tz
        if isinstance(tz, datetime.tzinfo):
            return tz
        raise PydtValueError(
            "<pydt> Invalid tzinfo: %s. Accept: `<tzinfo>`, `<str>` or None." % repr(tz)
        )

    @cython.cfunc
    @cython.inline(True)
    def _tz_localize(self, dt: datetime.datetime, tz: object) -> datetime.datetime:
        try:
            return cydt.dt_replace_tzinfo(dt, self._parse_tzinfo(tz))
        except PydtValueError:
            raise
        except Exception as err:
            raise PydtValueError("<pydt> %s" % err) from err

    @cython.cfunc
    @cython.inline(True)
    def _tz_convert(self, dt: datetime.datetime, tz: object) -> datetime.datetime:
        try:
            return dt.astimezone(self._parse_tzinfo(tz))
        except PydtValueError:
            raise
        except Exception as err:
            raise PydtValueError("<pydt> %s" % err) from err

    @cython.cfunc
    @cython.inline(True)
    def _tz_switch(
        self,
        dt: datetime.datetime,
        targ_tz: object,
        base_tz: object,
        naive: cython.bint,
    ) -> datetime.datetime:
        # Already timezone-aware: convert to targ_tz
        if cydt.get_dt_tzinfo(dt) is not None:
            dt = self._tz_convert(dt, targ_tz)
        # Localize to base_tz & convert to targ_tz
        elif isinstance(base_tz, (str, datetime.tzinfo)):
            dt = self._tz_convert(self._tz_localize(dt, base_tz), targ_tz)
        # Invalid base_tz
        else:
            raise PydtValueError(
                "<pydt> Cannot switch timezone without 'base_tz' for naive datetime."
            )
        # Return
        return cydt.dt_replace_tzinfo(dt, None) if naive else dt

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _parse_frequency(self, freq: str) -> cython.longlong:
        frequency = freq.lower()
        if frequency == "d":
            return cydt.US_DAY
        elif frequency == "h":
            return cydt.US_HOUR
        elif frequency == "m":
            return cydt.US_MINUTE
        elif frequency == "s":
            return cydt.US_SECOND
        elif frequency == "ms":
            return cydt.US_MILLISECOND
        elif frequency == "us":
            return cydt.US_MICROSECOND
        else:
            raise PydtValueError(
                "<pydt> Invalid frequency: %s. Accept: 'D'/'h'/'m'/'s'/'ms'/'us'/'ns'."
                % repr(freq)
            )

    @cython.cfunc
    @cython.inline(True)
    def _round_frequency(self, freq: str) -> pydt:
        frequency: cython.longlong = self._parse_frequency(freq)
        if frequency == cydt.US_MICROSECOND:
            return self

        us: cython.longlong = self._get_microseconds()
        us = int(cymath.round_l(us / frequency)) * frequency
        return self._new(
            cydt.dt_fr_microseconds(us, self._get_tzinfo(), self._get_fold())
        )

    @cython.cfunc
    @cython.inline(True)
    def _ceil_frequency(self, freq: str) -> pydt:
        frequency: cython.longlong = self._parse_frequency(freq)
        if frequency == cydt.US_MICROSECOND:
            return self

        us: cython.longlong = self._get_microseconds()
        us = int(cymath.ceil_l(us / frequency)) * frequency
        return self._new(
            cydt.dt_fr_microseconds(us, self._get_tzinfo(), self._get_fold())
        )

    @cython.cfunc
    @cython.inline(True)
    def _floor_frequency(self, freq: str) -> pydt:
        frequency: cython.longlong = self._parse_frequency(freq)
        if frequency == cydt.US_MICROSECOND:
            return self

        us: cython.longlong = self._get_microseconds()
        us = int(cymath.floor_l(us / frequency)) * frequency
        return self._new(
            cydt.dt_fr_microseconds(us, self._get_tzinfo(), self._get_fold())
        )

    @cython.cfunc
    @cython.inline(True)
    def _delta(
        self,
        years: cython.int,
        months: cython.int,
        days: cython.int,
        weeks: cython.int,
        hours: cython.int,
        minutes: cython.int,
        seconds: cython.int,
        microseconds: cython.int,
    ) -> pydt:
        return self._add_cytimedelta(
            cytimedelta(
                years=years,
                months=months,
                days=days,
                weeks=weeks,
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                microseconds=microseconds,
            )
        )

    @cython.cfunc
    @cython.inline(True)
    def _replace(
        self,
        year: cython.int,
        month: cython.int,
        day: cython.int,
        hour: cython.int,
        minute: cython.int,
        second: cython.int,
        microsecond: cython.int,
        tzinfo: object,
        fold: cython.int,
    ) -> pydt:
        return self._new(
            cydt.dt_replace(
                self._dt,
                year,
                month,
                day,
                hour,
                minute,
                second,
                microsecond,
                tzinfo,
                fold,
            )
        )

    @cython.cfunc
    @cython.inline(True)
    def _between(
        self,
        other: object,
        unit: str,
        inclusive: cython.bint,
    ) -> cython.longlong:
        if cydt.is_dt(other):
            return self._between_datetime(other, unit, inclusive)
        elif cydt.is_date(other):
            return self._between_datetime(cydt.dt_fr_date(other), unit, inclusive)
        elif isinstance(other, pydt):
            return self._between_pydt(other, unit, inclusive)
        elif isinstance(other, (str, np.datetime64)):
            return self._between_datetime(self._parse_datetime(other), unit, inclusive)
        else:
            raise PydtValueError("<pydt> Unsupported data type: %s" % (type(other)))

    @cython.cfunc
    @cython.inline(True)
    def _between_pydt(
        self,
        pt: pydt,
        unit: str,
        inclusive: cython.bint,
    ) -> cython.longlong:
        return self._between_datetime(pt._dt, unit, inclusive)

    @cython.cfunc
    @cython.inline(True)
    @cython.cdivision(True)
    def _between_datetime(
        self,
        dt: datetime.datetime,
        unit: str,
        inclusive: cython.bint,
    ) -> cython.longlong:
        # Pre binding
        delta: cython.longlong

        # Year
        if unit == "Y":
            delta = cymath.abs_ll(self._get_year() - cydt.get_year(dt))
            return delta + 1 if inclusive else delta  # exit
        # Month
        if unit == "M":
            delta = cymath.abs_ll(
                (self._get_year() - cydt.get_year(dt)) * 12
                + (self._get_month() - cydt.get_month(dt))
            )
            return delta + 1 if inclusive else delta  # exit

        # Calculate delta
        delta = cymath.abs_ll(cydt.dt_sub_dt_microseconds(self._dt, dt))

        # Week
        if unit == "W":
            delta = delta // cydt.US_DAY
            if self._compare_datetime(dt, False) > 0:  # find base weekday
                delta += cydt.get_weekday(dt)
            else:
                delta += self._get_weekday()
            delta = delta // 7
        # Day
        elif unit == "D":
            delta = delta // cydt.US_DAY
        # Hour
        elif unit == "h":
            delta = delta // cydt.US_HOUR
        # Minute
        elif unit == "m":
            delta = delta // cydt.US_MINUTE
        # Second
        elif unit == "s":
            delta = delta // cydt.US_SECOND
        # Millisecond
        elif unit == "ms":
            delta = delta // cydt.US_MILLISECOND
        # Microsecond
        elif unit != "us":
            raise PydtValueError("<pydt> Invalid time `unit`: %s" % repr(unit))

        # Return Delta
        return delta + 1 if inclusive else delta  # exit

    # Special methods - addition
    @cython.cfunc
    @cython.inline(True)
    def _add_timedelta(self, other: datetime.timedelta) -> pydt:
        return self._new(
            cydt.dt_fr_microseconds(
                self._get_microseconds() + cydt.delta_to_microseconds(other),
                self._get_tzinfo(),
                self._get_fold(),
            )
        )

    @cython.cfunc
    @cython.inline(True)
    def _add_cytimedelta(self, other: cytimedelta) -> pydt:
        return self._new(other._add_date_time(self._dt))

    @cython.cfunc
    @cython.inline(True)
    def _add_relativedelta(self, other: relativedelta) -> pydt:
        return self._new(other + self._dt)

    def __add__(self, other: object) -> pydt:
        if cydt.is_delta(other):
            return self._add_timedelta(other)
        if isinstance(other, cytimedelta):
            return self._add_cytimedelta(other)
        if isinstance(other, relativedelta):
            return self._add_relativedelta(other)
        if cydt.is_delta64(other):
            return self._add_timedelta(cydt.delta64_to_delta(other))

        return NotImplemented

    def __radd__(self, other: object) -> pydt:
        if cydt.is_delta(other):
            return self._add_timedelta(other)
        if isinstance(other, cytimedelta):
            return self._add_cytimedelta(other)
        if isinstance(other, relativedelta):
            return self._add_relativedelta(other)
        # TODO: this will not work since numpy does not return NotImplemented
        if cydt.is_delta64(other):
            return self._add_timedelta(cydt.delta64_to_delta(other))

        return NotImplemented

    # Special methods - substraction
    @cython.cfunc
    @cython.inline(True)
    def _sub_pydt(self, other: pydt) -> datetime.timedelta:
        return self._sub_datetime(other._dt)

    @cython.cfunc
    @cython.inline(True)
    def _sub_datetime(self, other: datetime.datetime) -> datetime.timedelta:
        return cydt.dt_sub_dt(self._dt, other)

    @cython.cfunc
    @cython.inline(True)
    def _sub_timedelta(self, other: datetime.timedelta) -> pydt:
        return self._new(
            cydt.dt_fr_microseconds(
                self._get_microseconds() - cydt.delta_to_microseconds(other),
                self._get_tzinfo(),
                self._get_fold(),
            )
        )

    @cython.cfunc
    @cython.inline(True)
    def _sub_cytimedelta(self, other: cytimedelta) -> pydt:
        return self._new(other._rsub_date_time(self._dt))

    @cython.cfunc
    @cython.inline(True)
    def _sub_relativedelta(self, other: relativedelta) -> pydt:
        return self._new(self._dt - other)

    def __sub__(self, other: object) -> Union[pydt, datetime.timedelta]:
        if isinstance(other, pydt):
            return self._sub_pydt(other)
        if cydt.is_dt(other):
            return self._sub_datetime(other)
        if cydt.is_delta(other):
            return self._sub_timedelta(other)
        if isinstance(other, cytimedelta):
            return self._sub_cytimedelta(other)
        if isinstance(other, relativedelta):
            return self._sub_relativedelta(other)
        if isinstance(other, str):
            return self._sub_datetime(self._parse_datetime(other))
        if cydt.is_dt64(other):
            return self._sub_datetime(cydt.dt64_to_dt(other))
        if cydt.is_delta64(other):
            return self._sub_timedelta(cydt.delta64_to_delta(other))

        return NotImplemented

    @cython.cfunc
    @cython.inline(True)
    def _rsub_datetime(self, other: datetime.datetime) -> datetime.timedelta:
        return cydt.delta_fr_microseconds(
            cydt.dt_to_microseconds(other) - self._get_microseconds()
        )

    def __rsub__(self, other: object) -> datetime.timedelta:
        if cydt.is_dt(other):
            return self._rsub_datetime(other)
        if isinstance(other, str):
            return self._rsub_datetime(self._parse_datetime(other))
        # TODO this will not work since numpy does not return NotImplemented
        if cydt.is_dt64(other):
            return self._rsub_datetime(cydt.dt64_to_dt(other))

        return NotImplemented

    # Special methods - comparison
    @cython.cfunc
    @cython.inline(True)
    def _compare_pydt(self, other: pydt, allow_mixed: cython.bint) -> cython.int:
        return self._compare_datetime(other._dt, allow_mixed)

    @cython.cfunc
    @cython.inline(True)
    def _compare_datetime(
        self,
        other: datetime.datetime,
        allow_mixed: cython.bint,
    ) -> cython.int:
        my_tzinfo: object = self._get_tzinfo()
        to_tzinfo: object = cydt.get_dt_tzinfo(other)
        if my_tzinfo is to_tzinfo:
            return self._compare_datetime_base(other)

        myoff_us: cython.longlong
        tooff_us: cython.longlong
        temp_dt: datetime.datetime
        if my_tzinfo is not None:
            myoff_us = cydt.delta_to_microseconds(self._dt.utcoffset())
            if allow_mixed:
                temp_dt = cydt.dt_replace_fold(self._dt, self._get_fold() + 1)
                if myoff_us != cydt.delta_to_microseconds(temp_dt.utcoffset()):
                    return 2
        else:
            myoff_us = 0

        if to_tzinfo is not None:
            tooff_us = cydt.delta_to_microseconds(other.utcoffset())
            if allow_mixed:
                temp_dt = cydt.dt_replace_fold(other, cydt.get_dt_fold(other) + 1)
                if tooff_us != cydt.delta_to_microseconds(temp_dt.utcoffset()):
                    return 2
        else:
            tooff_us = 0

        if my_tzinfo is None or to_tzinfo is None:
            if allow_mixed:
                return 2  # arbitrary non-zero value
            else:
                raise PydtValueError("<pydt> Cannot compare naive and aware datetimes.")
        elif myoff_us == tooff_us:
            return self._compare_datetime_base(other)
        else:
            return self._compare_datetime_delta(other)

    @cython.cfunc
    @cython.inline(True)
    def _compare_datetime_base(self, other: datetime.datetime) -> cython.int:
        mydt_us: cython.longlong = self._get_microseconds()
        todt_us: cython.longlong = cydt.dt_to_microseconds(other)
        if mydt_us == todt_us:
            return 0
        elif mydt_us > todt_us:
            return 1
        else:
            return -1

    @cython.cfunc
    @cython.inline(True)
    def _compare_datetime_delta(self, other: datetime.datetime) -> cython.int:
        delta: datetime.timedelta = self._sub_datetime(other)
        return -1 if cydt.get_delta_days(delta) < 0 else delta and 1 or 0

    def __eq__(self, other: object) -> cython.bint:
        if cydt.is_dt(other):
            return self._compare_datetime(other, True) == 0
        if isinstance(other, pydt):
            return self._compare_pydt(other, True) == 0
        if isinstance(other, str):
            return self._compare_datetime(self._parse_datetime(other), True) == 0
        if cydt.is_date(other):
            return NotImplemented
        return False

    def __ne__(self, other: object) -> cython.bint:
        if cydt.is_dt(other):
            return self._compare_datetime(other, True) != 0
        if isinstance(other, pydt):
            return self._compare_pydt(other, True) != 0
        if isinstance(other, str):
            return self._compare_datetime(self._parse_datetime(other), True) != 0
        if cydt.is_date(other):
            return NotImplemented
        return True

    def __gt__(self, other):
        if cydt.is_dt(other):
            return self._compare_datetime(other, False) > 0
        if isinstance(other, pydt):
            return self._compare_pydt(other, False) > 0
        if isinstance(other, str):
            return self._compare_datetime(self._parse_datetime(other), False) > 0
        if cydt.is_date(other):
            return NotImplemented
        raise PydtValueError(
            "<pydt> Can't compare '%s' to '%s'."
            % (type(self).__name__, type(other).__name__)
        )

    def __ge__(self, other):
        if cydt.is_dt(other):
            return self._compare_datetime(other, False) >= 0
        if isinstance(other, pydt):
            return self._compare_pydt(other, False) >= 0
        if isinstance(other, str):
            return self._compare_datetime(self._parse_datetime(other), False) >= 0
        if cydt.is_date(other):
            return NotImplemented
        raise PydtValueError(
            "<pydt> Can't compare '%s' to '%s'."
            % (type(self).__name__, type(other).__name__)
        )

    def __lt__(self, other):
        if cydt.is_dt(other):
            return self._compare_datetime(other, False) < 0
        if isinstance(other, pydt):
            return self._compare_pydt(other, False) < 0
        if isinstance(other, str):
            return self._compare_datetime(self._parse_datetime(other), False) < 0
        if cydt.is_date(other):
            return NotImplemented
        raise PydtValueError(
            "<pydt> Can't compare '%s' to '%s'."
            % (type(self).__name__, type(other).__name__)
        )

    def __le__(self, other):
        if cydt.is_dt(other):
            return self._compare_datetime(other, False) <= 0
        if isinstance(other, pydt):
            return self._compare_pydt(other, False) <= 0
        if isinstance(other, str):
            return self._compare_datetime(self._parse_datetime(other), False) <= 0
        if cydt.is_date(other):
            return NotImplemented
        raise PydtValueError(
            "<pydt> Can't compare '%s' to '%s'."
            % (type(self).__name__, type(other).__name__)
        )

    # Special methods - represent
    def __repr__(self) -> str:
        return "<pydt (datetime='%s')>" % cydt.dt_to_isoformat_tz(self._dt)

    def __str__(self) -> str:
        return cydt.dt_to_isoformat_tz(self._dt)

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _hash(self) -> cython.int:
        if self._hashcode != -1:
            return self._hashcode

        tzinfo: object = self._get_tzinfo()
        if tzinfo is None:
            self._hashcode = hash(("pytd", self._get_microseconds()))
        else:
            dt: datetime.datetime
            if self._get_fold():
                dt = cydt.dt_replace_fold(self._dt, 0)
            else:
                dt = self._dt
            self._hashcode = hash(
                (
                    "pydt",
                    cydt.dt_to_microseconds(dt),
                    cydt.delta_to_microseconds(tzinfo.utcoffset(dt)),
                )
            )
        return self._hashcode

    def __hash__(self) -> int:
        return self._hash()


# Exceptions ----------------------------------------------------------------------------------
class PydtValueError(ValueError):
    """The one and only exception this module will raise."""
