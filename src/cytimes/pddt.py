# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from __future__ import annotations

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cytimes.pydt import pydt  # type: ignore
from cython.cimports.cytimes import cydatetime as cydt  # type: ignore
from cython.cimports.cytimes.cyparser import DEFAULT_PARSERINFO  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
import datetime, zoneinfo
from typing import Union, Literal, Iterator
import numpy as np, pandas as pd
from pandas.tseries import offsets
from cytimes.pydt import pydt
from cytimes import cydatetime as cydt

__all__ = ["pddt", "PddtValueError"]


# Constants -----------------------------------------------------------------------------------
DAYS_BR_QUARTER: np.ndarray = np.array([0, 90, 181, 273, 365])
FIXED_FREQUENCY: np.ndarray = np.array(["d", "h", "m", "s", "ms", "us", "ns"])


# pddt (PandasDatetime) -----------------------------------------------------------------------
@cython.cclass
class pddt:
    _default: datetime.datetime
    _dayfirst: cython.bint
    _yearfirst: cython.bint
    _utc: cython.bint
    _format: str
    _exact: cython.bint
    _series: pd.Series
    _index: pd.Series
    _naive: pd.Series
    # Cache
    _year: pd.Series
    _year_1st: pd.Series
    _year_lst: pd.Series
    _quarter: pd.Series
    _quarter_1st: pd.Series
    _quarter_lst: pd.Series
    _month: pd.Series
    _month_1st: pd.Series
    _month_lst: pd.Series
    _month_days: pd.Series
    _day: pd.Series
    _weekday: pd.Series
    _is_leapyear: pd.Series
    _days_of_year: pd.Series

    def __init__(
        self,
        timeobj: Union[pd.Series, list],
        default: Union[datetime.datetime, datetime.date, None] = None,
        dayfirst: cython.bint = False,
        yearfirst: cython.bint = False,
        utc: cython.bint = False,
        format: str = None,
        exact: cython.bint = True,
    ) -> None:
        """pddt (PandasDatetime)
        A wrapper for pandas' time Series combined with parsing and delta adjustment.

        #### Time object arguments
        :param timeobj: Accepts `<pd.Series>`/`<list>` that can be converted to 'Series[Timestamp]`.

        #### Datetime parsing arguments (Takes affect when 'timeobj' is not `<pd.Series[datetime64[ns]]>`)
        :param default: `<datetime>` The default date, which will be used to fillin elements that can't be parse into datetime.
            - If set to `None` (default), `<PddtValueError>` will be raise when parsing failed.

        :param dayfirst: `<bool>` Whether to interpret the first value in an ambiguous date (e.g. 01/05/09) as the day (`True`) or month (`False`).
            - When `yearfirst` is `True`, this distinguishes between Y/D/M and Y/M/D.
            - If set to `None`, the `dayfirst` settings in `ParserInfo` will be used (defaults to False).

        :param yearfirst: `<bool>` Whether to interpret the first value in an ambiguous date (e.g. 01/05/09) as the yetimear.
            - If `True`, the first number is taken as year, otherwise the last number.
            - If set to `None`, the `yearfirst` settings in `ParserInfo` will be used (defaults to False).

        :param utc: `<bool`> Control timezone-related parsing, localization and conversion.
            - If `True`, `ALWAYS` parse to tinzome-aware UTC-localized `pandas.Timetamp`. Timezone-naive inputs
              are `LOCALIZED` as UTC, while timezone-aware inputs are `CONVERTED` to UTC.
            - If `False`, Timezone-naive inputs remain naive, while timezone-aware ones will keep their timezone.
              However, mixed offset is not allowed and will raise `<PddtValueError>`. This is different from
              `pandas.to_datetime()`, because most <pddt> methods relies on `pandas.Timestamp`.
            - For more information, please refer to `pandas.to_datetime()` documentation.
              <https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html>

        :param format: `<str>` The strftime to parse timeobj with, accepts:
            - strftime format (e.g. "%d/%m/%Y"): Note that "%f" will parse all the way to nanoseconds.
              For more infomation, please refer to <https://docs.python.org/3/library/datetime.html
              #strftime-and-strptime-behavior>.
            - "ISO8601": Parse any `ISO8601` time string (not necessarily in exactly the same format).
              For more infomation, please refer to <https://en.wikipedia.org/wiki/ISO_8601>.
            - "mixed": Infer the format for each element individually. This is risky, and should probably use
              it along with `dayfirst`.

        :param exact: `<bool>` Whether to parse with the exact provided 'format'. Applicable when 'format' is provided.
            - If `True` (default), perform an exact 'format' match.
            - If `False`, allow the `format` to match anywhere in the string.
            - Can `NOT` be used alongside `format='ISO8601'` or `format='mixed'`.

        :raises `PddtValueError`: If any error occurs.

        #### Addition
        - Left/Right addition with `pandas.Series[Timedelta]`, `pd.Timedelta`, `pd.tseries.Offset`, etc.
          returns `pddt`. Equivalent to `pandas.Series[Timestamp] + delta`.

        #### Subtraction
        - Left/Right substraction with `pddt`, `pandas.Series[Timestamp]`, `list`, etc.
          returns `pandas.Series[Timedelta]`. Equivalent to `pandas.Series[Timestamp] - pandas.Series[Timestamp]`.
        - Left substraction with `pandas.Series[Timedelta]`, `pd.Timedelta`, `pd.tseres.Offset`, etc.
          returns `pddt`. Equivalent to `pandas.Series[Timestamp] - delta`.

        #### Comparison
        - Support direct comparison between `pddt`, `pandas.Series[Timestamp]`, `list`, etc.
          Equivalent to `pandas.Series[Timestamp] <op> pandas.Series[Timestamp]`.
        """

        # Settings
        self._default = None if default is None else pydt(default).dt
        self._dayfirst = dayfirst
        self._yearfirst = yearfirst
        self._utc = utc
        self._format = format
        self._exact = exact
        # To datetime
        try:
            self._series = self._to_datetime(timeobj)
        except PddtValueError:
            raise
        except Exception as err:
            raise PddtValueError("<pddt> %s" % err) from err
        self._index = None
        self._naive = None
        # Cache
        self._year = None
        self._year_1st = None
        self._year_lst = None
        self._quarter = None
        self._quarter_1st = None
        self._quarter_lst = None
        self._month = None
        self._month_1st = None
        self._month_lst = None
        self._month_days = None
        self._day = None
        self._weekday = None
        self._is_leapyear = None
        self._days_of_year = None

    # Access
    @property
    def dt(self) -> pd.Series[pd.Timestamp]:
        "Access datetime as `Series[Timestamp]`."
        return self._series.copy(True)

    @property
    def dtiso(self) -> pd.Series[str]:
        "Access datetime ISO format as `Series[str]`."
        return self._series.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

    @property
    def dtisotz(self) -> pd.Series[str]:
        "Access datetime ISO format with timzone as `Series[str]`."
        return self._series.dt.strftime("%Y-%m-%dT%H:%M:%S.%f%Z")

    @property
    def date(self) -> pd.Series[datetime.date]:
        "Access date as `Series[datetime.date]`."
        return self._series.dt.date

    @property
    def dateiso(self) -> pd.Series[str]:
        "Access date ISO format as `Series[str]`."
        return self._series.dt.strftime("%Y-%m-%d")

    @property
    def time(self) -> pd.Series[datetime.time]:
        "Access time as `Series[datetime.time]`."
        return self._series.dt.time

    @property
    def timeiso(self) -> pd.Series[str]:
        "Access time ISO format as `Series[str]`."
        return self._series.dt.strftime("%H:%M:%S.%f")

    @property
    def timetz(self) -> pd.Series[datetime.time]:
        "Access time with timezone as `Series[datetime.time]`."
        return self._series.dt.timetz

    @property
    def timeisotz(self) -> pd.Series[str]:
        "Access time ISO format with timezone as `Series[str]`."
        return self._series.dt.strftime("%H:%M:%S.%f%Z")

    @property
    def dtpy(self) -> list[datetime.datetime]:
        "Access datetime as `list[datetime.datetime]` (Python datetime)."
        return [cydt.dt_fr_dt(dt) for dt in self._series]

    @property
    def dt64(self) -> np.ndarray[np.datetime64]:
        """Access datetime as `np.ndarray[np.datetime64]`.
        Timezone will be normalized to UTC (naive) with unit of 'ns'.
        """
        return self._series.values

    @property
    def ordinal(self) -> pd.Series[int]:
        "Access date in ordinal as `Series[int]`."
        return cydt.seriesdt64_to_ordinal(self._get_naive())

    @property
    def seconds(self) -> pd.Series[float]:
        "Access datetime in total seconds (naive) after EPOCH as `Series[float]`."
        return cydt.seriesdt64_to_seconds(self._get_naive())

    @property
    def seconds_utc(self) -> pd.Series[float]:
        """Access datetime in total seconds (naive) after EPOCH as `Series[float]`.
        - If timezone-aware, return total seconds in UTC.
        - If timezone-naive, requivalent to `seconds`.

        #### Notice
        This should `NOT` be treated as timestamp, but rather adjustment of the
        total seconds of the datetime from utcoffset.
        """
        return cydt.seriesdt64_to_seconds(self._series)

    @property
    def microseconds(self) -> pd.Series[int]:
        "Access datetime in total microseconds (naive) after EPOCH as `Series[int]`."
        return cydt.seriesdt64_to_microseconds(self._get_naive())

    @property
    def microseconds_utc(self) -> pd.Series[int]:
        """Access datetime in total microseconds (naive) after EPOCH as `Series[int]`.
        - If timezone-aware, return total microseconds in UTC.
        - If timezone-naive, requivalent to `microseconds`.

        #### Notice
        This should `NOT` be treated as timestamp, but rather adjustment of the
        total microseconds of the datetime from utcoffset.
        """
        return cydt.seriesdt64_to_microseconds(self._series)

    @property
    def timestamp(self) -> pd.Series[float]:
        "Access datetime in timestamp as `Series[float]`."
        return cydt.seriesdt64_to_seconds(self._series)

    # Absolute
    @property
    def year(self) -> pd.Series[int]:
        "Access year as `Series[int]`."
        return self._get_year()

    @property
    def month(self) -> pd.Series[int]:
        "Access month as `Series[int]`."
        return self._get_month()

    @property
    def day(self) -> pd.Series[int]:
        "Access day as `Series[int]`."
        return self._get_day()

    @property
    def hour(self) -> pd.Series[int]:
        "Access hour as `Series[int]`."
        return self._series.dt.hour

    @property
    def minute(self) -> pd.Series[int]:
        "Access minute as `Series[int]`."
        return self._series.dt.minute

    @property
    def second(self) -> pd.Series[int]:
        "Access second as `Series[int]`."
        return self._series.dt.second

    @property
    def microsecond(self) -> pd.Series[int]:
        "Access microsecond as `Series[int]`."
        return self._series.dt.microsecond

    @property
    def tzinfo(self) -> datetime.tzinfo:
        "The timezone of the Series `<datetime.tzinfo>`."
        return self._series.dt.tz

    # Calendar
    @property
    def is_leapyear(self) -> pd.Series[bool]:
        "Whether is a leap year as `Series[bool]`."
        return self._get_is_leapyear()

    @property
    def days_in_year(self) -> pd.Series[int]:
        "Number of days in the year as `Series[int]`."
        return self._np_to_series(self._get_is_leapyear().values + 365)

    @property
    def days_bf_year(self) -> pd.Series[int]:
        "Number of days before January 1st of the year as `Series[int]`."
        return self._np_to_series(self.ordinal.values - self._get_days_of_year().values)

    @property
    def days_of_year(self) -> pd.Series[int]:
        "Number of days into the year as `Series[int]`."
        return self._get_days_of_year()

    @property
    def quarter(self) -> pd.Series[int]:
        "The quarter of the date as `Series[int]`."
        return self._get_quarter()

    @property
    def days_in_quarter(self) -> pd.Series[int]:
        "Number of days in the quarter as `Series[int]`."
        quarter = self._get_quarter().values
        days = DAYS_BR_QUARTER[quarter] - DAYS_BR_QUARTER[quarter - 1]
        return self._np_to_series(days + self._get_is_leapyear().values)

    @property
    def days_bf_quarter(self) -> pd.Series[int]:
        "Number of days in the year preceding first day of the quarter as `Series[int]`."
        quarter = self._get_quarter().values
        days = DAYS_BR_QUARTER[quarter - 1]
        leap = self._get_is_leapyear().values * (quarter >= 2)
        return self._np_to_series(days + leap)

    @property
    def days_of_quarter(self) -> pd.Series[int]:
        "Number of days into the quarter as `Series[int]`."
        days_y = self._get_days_of_year().values
        quarter = self._get_quarter().values
        days = DAYS_BR_QUARTER[quarter - 1]
        leap = self._get_is_leapyear().values * (quarter >= 2)
        return self._np_to_series(days_y - days - leap)

    @property
    def days_in_month(self) -> pd.Series[int]:
        "Number of days in the month as `Series[int]`."
        return self._get_month_days()

    @property
    def days_bf_month(self) -> pd.Series[int]:
        "Number of days in the year preceding first day of the month as `Series[int]`."
        return self._np_to_series(
            self._get_days_of_year().values - self._get_day().values
        )

    @property
    def weekday(self) -> pd.Series[int]:
        "The weekday, where Monday == 0 ... Sunday == 6, as `Series[int]`."
        return self._get_weekday()

    @property
    def isoweekday(self) -> pd.Series[int]:
        "The ISO weekday, where Monday == 1 ... Sunday == 7, as `Series[int]`."
        return self._np_to_series(self._get_weekday().values + 1)

    @property
    def isoweek(self) -> pd.Series[int]:
        "The ISO calendar week number as `Series[int]`."
        return self._series.dt.isocalendar().week

    @property
    def isoyear(self) -> pd.Series[int]:
        "The ISO calendar year as `Series[int]`."
        return self._series.dt.isocalendar().year

    @property
    def isocalendar(self) -> pd.DataFrame[int]:
        "Access ISO calendar as `DataFrame[int]`."
        return self._series.dt.isocalendar()

    # Time manipulation
    @property
    def start_time(self) -> pddt:
        "Start time (00:00:00.000000) of the current datetime."
        return self._new(self._series.dt.floor("D", "infer", "shift_backward"))

    @property
    def end_time(self) -> pddt:
        "End time (23:59:59.999999) of the current datetime."
        return self._new(
            self._series.dt.ceil("D", "infer", "shift_forward") - offsets.Micro(1)
        )

    # Day manipulation
    @property
    def tomorrow(self) -> pddt:
        "Tomorrow."
        return self._new(self._series + offsets.Day(1))

    @property
    def yesterday(self) -> pddt:
        "Yesterday."
        return self._new(self._series + offsets.Day(-1))

    # Weekday manipulation
    @property
    def monday(self) -> pddt:
        "Monday of the current week."
        return self._new(
            self._series - pd.TimedeltaIndex(self._get_weekday(), unit="D")
        )

    @property
    def tuesday(self) -> pddt:
        "Tuesday of the current week."
        return self._new(
            self._series - pd.TimedeltaIndex(self._get_weekday() - 1, unit="D")
        )

    @property
    def wednesday(self) -> pddt:
        "Wednesday of the current week."
        return self._new(
            self._series - pd.TimedeltaIndex(self._get_weekday() - 2, unit="D")
        )

    @property
    def thursday(self) -> pddt:
        "Thursday of the current week."
        return self._new(
            self._series - pd.TimedeltaIndex(self._get_weekday() - 3, unit="D")
        )

    @property
    def friday(self) -> pddt:
        "Friday of the current week."
        return self._new(
            self._series - pd.TimedeltaIndex(self._get_weekday() - 4, unit="D")
        )

    @property
    def saturday(self) -> pddt:
        "Saturday of the current week."
        return self._new(
            self._series - pd.TimedeltaIndex(self._get_weekday() - 5, unit="D")
        )

    @property
    def sunday(self) -> pddt:
        "Sunday of the current week."
        return self._new(
            self._series - pd.TimedeltaIndex(self._get_weekday() - 6, unit="D")
        )

    def curr_week(self, weekday: Union[int, str, None]) -> pddt:
        """Specific weekday of the currect week.
        :param weekday: `<int>` 0 as Monday to 6 as Sunday / `<str>` 'mon', 'tuesday', etc.
        """
        return self._curr_week(weekday)

    def next_week(self, weekday: Union[int, str, None] = None) -> pddt:
        """Specific weekday of the next week.
        :param weekday: `<int>` 0 as Monday to 6 as Sunday / `<str>` 'mon', 'tuesday', etc.
            If set to `None` (default), will move to the same weekday of the next week.
        """
        return self._to_week(1, weekday)

    def last_week(self, weekday: Union[int, str, None] = None) -> pddt:
        """Specific weekday of the last week.
        :param weekday: `<int>` 0 as Monday to 6 as Sunday / `<str>` 'mon', 'tuesday', etc.
            If set to `None` (default), will move to the same weekday of the last week.
        """
        return self._to_week(-1, weekday)

    def to_week(self, offset: int, weekday: Union[int, str, None] = None) -> pddt:
        """Specific weekday of the week (+/-) offset.
        :param offset: `<int>` number of weeks offset from the current week.
        :param weekday: `<int>` 0 as Monday to 6 as Sunday / `<str>` 'mon', 'Tuesday', etc.
            If set to `None` (default), will move to the same weekday of the week (+/-) offset.
        """
        return self._to_week(offset, weekday)

    def is_weekday(self, weekday: Union[int, str]) -> pd.Series[bool]:
        """Whether the current datetime is a specific weekday as `Seires[bool]`.
        :param weekday: `<int>` 0 as Monday to 6 as Sunday / `<str>` 'mon', 'tuesday', etc.
        """
        if weekday is None:
            return pd.Series(True, index=self._get_index())
        else:
            return self._get_weekday() == self._parse_weekday(weekday)

    # Month manipulation
    @property
    def month_1st(self) -> pddt:
        "First day of the current month."
        return self._new(self._get_month_1st())

    def is_month_1st(self) -> pd.Series[bool]:
        "Whether the current datetime is the first day of the month as `Seires[bool]`."
        return self._series.dt.is_month_start

    @property
    def month_lst(self) -> pddt:
        "Last day of the current month."
        return self._new(self._get_month_lst())

    def is_month_lst(self) -> pd.Series[bool]:
        "Whether the current datetime is the last day of the month as `Seires[bool]`."
        return self._series.dt.is_month_end

    def curr_month(self, day: int) -> pddt:
        """Specifc day of the current month.
        :param day: `<int>` day of the current month.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the current month.
        """
        return self._curr_month(day)

    def next_month(self, day: int = 0) -> pddt:
        """Specifc day of the next month.
        :param day: `<int>` day of the next month.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the next month.
        """
        return self._to_month(1, day)

    def last_month(self, day: int = 0) -> pddt:
        """Specifc day of the last month.
        :param day: `<int>` day of the last month.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the last month.
        """
        return self._to_month(-1, day)

    def to_month(self, offset: int, day: int = 0) -> pddt:
        """Specifc day of the month (+/-) offset.
        :param offset: `<int>` number of months offset from the current month.
        :param day: `<int>` day of the month (+/-) offset
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the month (+/-) offset.
        """
        return self._to_month(offset, day)

    def is_month(self, month: Union[int, str]) -> pd.Series[bool]:
        """Whether the current datetime is a specific month as `Seires[bool]`.
        :param month: `<int>` 1 as January to 12 as December / `<str>` 'jan', 'feb', etc.
        """
        if month is None:
            return pd.Series(True, index=self._get_index())
        else:
            return self._get_month() == self._parse_month(month)

    # Quarter manipulation
    @property
    def quarter_1st(self) -> pddt:
        "First day of the current quarter."
        return self._new(self._get_quarter_1st())

    def is_quarter_1st(self) -> pd.Series[bool]:
        "Whether the current datetime is the first day of the quarter as `Seires[bool]`."
        return self._series.dt.is_quarter_start

    @property
    def quarter_lst(self) -> pddt:
        "Last day of the current quarter."
        return self._new(self._get_quarter_lst())

    def is_quarter_lst(self) -> pd.Series[bool]:
        "Whether the current datetime is the last day of the quarter as `Seires[bool]`."
        return self._series.dt.is_quarter_end

    def curr_quarter(self, month: int, day: int = 0) -> pddt:
        """Specifc day of the current quarter.
        :param month: `<int>` 1 as 1st month of to 3 as 3rd month of the quarter.
        :param day: `<int>` day of the quarter month.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the current quarter month.
        """
        return self._curr_quarter(month, day)

    def next_quarter(self, month: int, day: int = 0) -> pddt:
        """Specifc day of the next quarter.
        :param month: `<int>` 1 as 1st month of to 3 as 3rd month of the quarter.
        :param day: `<int>` day of the quarter month.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the next quarter month.
        """
        return self._to_quarter(1, month, day)

    def last_quarter(self, month: int, day: int = 0) -> pddt:
        """Specifc day of the last quarter.
        :param month: `<int>` 1 as 1st month of to 3 as 3rd month of the quarter.
        :param day: `<int>` day of the quarter month.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the last quarter month.
        """
        return self._to_quarter(-1, month, day)

    def to_quarter(self, offset: int, month: int, day: int = 0) -> pddt:
        """Specifc day of the quarter (+/-) offset.
        :param offset: `<int>` number of quarters offset from the current quarter.
        :param month: `<int>` 1 as 1st month of to 3 as 3rd month of the quarter.
        :param day: `<int>` day of the quarter month.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the quarter (+/-) offset.
        """
        return self._to_quarter(offset, month, day)

    def is_quarter(self, quarter: int) -> pd.Series[bool]:
        """Whether the current datetime is a specific quarter as `Seires[bool]`.
        :param quarter: `<int>` 1 as 1st quarter to 4 as 4th quarter.
        """
        return self._get_quarter() == quarter

    # Year manipulation
    @property
    def year_1st(self) -> pddt:
        "First day of the current year."
        return self._new(self._get_year_1st())

    def is_year_1st(self) -> pd.Series[bool]:
        "Whether the current datetime is the first day of the year as `Seires[bool]`."
        return self._series.dt.is_year_start

    @property
    def year_lst(self) -> pddt:
        "Last day of the current year."
        return self._new(self._get_year_lst())

    def is_year_lst(self) -> pd.Series[bool]:
        "Whether the current datetime is the last day of the year as `Seires[bool]`."
        return self._series.dt.is_year_end

    def curr_year(self, month: Union[int, str, None] = None, day: int = 0) -> pddt:
        """Specifc month and day of the current year.
        :param month: `<int>` 1 as January to 12 as December / `<str>` 'jan', 'February', etc.
            If set to `None` (default), month will not be affected.
        :param day: `<int>` day of the month for the year.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the month.
        """
        return self._curr_year(month, day)

    def next_year(self, month: Union[int, str, None] = None, day: int = 0) -> pddt:
        """Specifc month and day of the next year.
        :param month: `<int>` 1 as January to 12 as December / `<str>` 'jan', 'February', etc.
            If set to `None` (default), month will not be affected.
        :param day: `<int>` day of the month for the year.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the month.
        """
        return self._to_year(1, month, day)

    def last_year(self, month: Union[int, str, None] = None, day: int = 0) -> pddt:
        """Specifc month and day of the last year.
        :param month: `<int>` 1 as January to 12 as December / `<str>` 'jan', 'February', etc.
            If set to `None` (default), month will not be affected.
        :param day: `<int>` day of the month for the year.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the month.
        """
        return self._to_year(-1, month, day)

    def to_year(
        self,
        offset: int,
        month: Union[int, str, None] = None,
        day: int = 0,
    ) -> pddt:
        """Specifc month and day of the year (+/-) offset.
        :param offset: `<int>` number of years offset from the current year.
        :param month: `<int>` 1 as January to 12 as December / `<str>` 'jan', 'February', etc.
            If set to `None` (default), month will not be affected.
        :param day: `<int>` day of the month for the year.
            `Value < 1` will be ignored, otherwise will automatically cap by the max days in the month.
        """
        return self._to_year(offset, month, day)

    def is_year(self, year: int) -> pd.Series[bool]:
        """Whether the current datetime is a specific year as `Seires[bool]`.
        :param year: `<int>` year.
        """
        return self._get_year() == year

    # Timezone manipulation
    @property
    def tz_available(self) -> set[str]:
        "All available timezone names accept by localize/convert/switch methods `<set[str]>`."
        return zoneinfo.available_timezones()

    def tz_localize(
        self,
        tz: Union[datetime.tzinfo, str, None],
        ambiguous: Union[bool, pd.Series[bool], Literal["raise", "infer"]] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> pddt:
        """Localize to a specific timezone. Equivalent to `Series.dt.tz_localize`.

        :param tz: `<datetime.tzinfo>`/`<str (timezone name)>`/`None (remove timezone)`.
            - `<datetime.tzinfo>`: The timezone to localize to. Support python native
              timezone and tzinfo from pytz & dateutil.
            - `<str>`: The timezone name to localize to. Must be one of the timezone
              names in `pddt.tz_available`.
            - `None`: Remove timezone awareness.

        :param ambiguous: `<bool>`/`<Series[bool]>`/`'raise'`/`'infer'`.
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from 03:00
            DST to 02:00 non-DST, 02:30:00 local time occurs both at 00:30:00 UTC
            and at 01:30:00 UTC. In such a situation, the ambiguous parameter
            dictates how ambiguous times should be handled.
            - `<bool>`: Marks all times as DST time (True) and non-DST time (False).
            - `<Series[bool]>`: Marks specific times (matching index) as DST time (True)
              and non-DST time (False).
            - `'raise'`: Raises an `PddtValueError` if there are ambiguous times.
            - `'infer'`: Attempt to infer fall dst-transition hours based on order.
            - * Notice: `'NaT'` is not supported.

        :param nonexistent: `'shift_forward'`/`'shift_backward'`/`'raise'`.
            A nonexistent time does not exist in a particular timezone where clocks moved
            forward due to DST.
            - `'shift_forward'`: Shifts nonexistent times forward to the closest existing time.
            - `'shift_backward'`: Shifts nonexistent times backward to the closest existing time.
            - `'raise'`: Raises an `PddtValueError` if there are nonexistent times.
            - * Notice: `'NaT'` is not supported.
        """
        return self._new(self._tz_localize(self._series, tz, ambiguous, nonexistent))

    def tz_convert(self, tz: Union[datetime.tzinfo, str, None]) -> pddt:
        """Convert to a specific timezone. Equivalent to `Series.dt.tz_convert`.

        :param tz: `<datetime.tzinfo>`/`<str (timezone name)>`/`None (local timezone)`.
        - `<datetime.tzinfo>`: The timezone to convert to. Support python native
              timezone and tzinfo from pytz & dateutil.
            - `<str>`: The timezone name to convert to. Must be one of the timezone
              names in `pddt.tz_available`.
            - `None`: Convert to system's local timezone.
        """
        return self._new(self._tz_convert(self._series, tz))

    def tz_switch(
        self,
        targ_tz: Union[datetime.tzinfo, str, None],
        base_tz: Union[datetime.tzinfo, str] = None,
        ambiguous: Union[bool, pd.Series[bool], Literal["raise", "infer"]] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
        naive: bool = False,
    ) -> pddt:
        """Switch from base timezone to target timezone.

        - When `Series` is timezone-aware, this method is equivalent to `tz_convert`,
          and only the `targ_tz` parameter is required.
        - When `Series` is timezone-naive, this method is equivalent 1st `tz_localize`
          to the `base_tz`, and then `tz_convert` to the `targ_tz`.

        :param targ_tz: `<datetime.tzinfo>`/`<str (timezone name)>`/`None (local timezone)`.
            - `<datetime.tzinfo>`: The timezone to convert to. Support python native
              timezone and tzinfo from pytz & dateutil.
            - `<str>`: The timezone name to convert to. Must be one of the timezone
              names in `pddt.tz_available`.
            - `None`: Convert to system's local timezone.

        :param base_tz: `<datetime.tzinfo>`/`<str (timezone name)>`.
            - `<datetime.tzinfo>`: The timezone to localize to. Support python native
              timezone and tzinfo from pytz & dateutil.
            - `<str>`: The timezone name to localize to. Must be one of the timezone
              names in `pddt.tz_available`.
            - * Notice: `None` is invalid when `Series` is timezone-naive.

        :param ambiguous: `<bool>`/`<Series[bool]>`/`'raise'`/`'infer'`.
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from 03:00
            DST to 02:00 non-DST, 02:30:00 local time occurs both at 00:30:00 UTC
            and at 01:30:00 UTC. In such a situation, the ambiguous parameter
            dictates how ambiguous times should be handled.
            - `<bool>`: Marks all times as DST time (True) and non-DST time (False).
            - `<Series[bool]>`: Marks specific times (matching index) as DST time (True)
              and non-DST time (False).
            - `'raise'`: Raises an `PddtValueError` if there are ambiguous times.
            - `'infer'`: Attempt to infer fall dst-transition hours based on order.
            - * Notice: `'NaT'` is not supported.

        :param nonexistent: `'shift_forward'`/`'shift_backward'`/`'raise'`.
            A nonexistent time does not exist in a particular timezone where clocks moved
            forward due to DST.
            - `'shift_forward'`: Shifts nonexistent times forward to the closest existing time.
            - `'shift_backward'`: Shifts nonexistent times backward to the closest existing time.
            - `'raise'`: Raises an `PddtValueError` if there are nonexistent times.
            - * Notice: `'NaT'` is not supported.

        :param naive: `<bool>` whether to convert to timezone-naive after conversion.
        """
        return self._new(
            self._tz_switch(
                self._series, targ_tz, base_tz, ambiguous, nonexistent, naive
            )
        )

    # Frequency manipulation
    def round(
        self,
        freq: Literal["D", "h", "m", "s", "ms", "us", "ns"],
        ambiguous: Union[bool, pd.Series[bool], Literal["raise", "infer"]] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> pddt:
        """Perform round operation to specified freqency.

        :param freq: `<str>` frequency to round to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond / `'ns'`: Nanosecond

        :param ambiguous: `<bool>`/`<Series[bool]>`/`'raise'`/`'infer'`.
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from 03:00
            DST to 02:00 non-DST, 02:30:00 local time occurs both at 00:30:00 UTC
            and at 01:30:00 UTC. In such a situation, the ambiguous parameter
            dictates how ambiguous times should be handled.
            - `<bool>`: Marks all times as DST time (True) and non-DST time (False).
            - `<Series[bool]>`: Marks specific times (matching index) as DST time (True)
              and non-DST time (False).
            - `'raise'`: Raises an `PddtValueError` if there are ambiguous times.
            - `'infer'`: Attempt to infer fall dst-transition hours based on order.
            - * Notice: `'NaT'` is not supported.

        :param nonexistent: `'shift_forward'`/`'shift_backward'`/`'raise'`.
            A nonexistent time does not exist in a particular timezone where clocks moved
            forward due to DST.
            - `'shift_forward'`: Shifts nonexistent times forward to the closest existing time.
            - `'shift_backward'`: Shifts nonexistent times backward to the closest existing time.
            - `'raise'`: Raises an `PddtValueError` if there are nonexistent times.
            - * Notice: `'NaT'` is not supported.
        """
        return self._round_frequency(freq, ambiguous, nonexistent)

    def ceil(
        self,
        freq: Literal["D", "h", "m", "s", "ms", "us", "ns"],
        ambiguous: Union[bool, pd.Series[bool], Literal["raise", "infer"]] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> pddt:
        """Perform ceil operation to specified freqency.

        :param freq: `<str>` frequency to ceil to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond / `'ns'`: Nanosecond

        :param ambiguous: `<bool>`/`<Series[bool]>`/`'raise'`/`'infer'`.
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from 03:00
            DST to 02:00 non-DST, 02:30:00 local time occurs both at 00:30:00 UTC
            and at 01:30:00 UTC. In such a situation, the ambiguous parameter
            dictates how ambiguous times should be handled.
            - `<bool>`: Marks all times as DST time (True) and non-DST time (False).
            - `<Series[bool]>`: Marks specific times (matching index) as DST time (True)
              and non-DST time (False).
            - `'raise'`: Raises an `PddtValueError` if there are ambiguous times.
            - `'infer'`: Attempt to infer fall dst-transition hours based on order.
            - * Notice: `'NaT'` is not supported.

        :param nonexistent: `'shift_forward'`/`'shift_backward'`/`'raise'`.
            A nonexistent time does not exist in a particular timezone where clocks moved
            forward due to DST.
            - `'shift_forward'`: Shifts nonexistent times forward to the closest existing time.
            - `'shift_backward'`: Shifts nonexistent times backward to the closest existing time.
            - `'raise'`: Raises an `PddtValueError` if there are nonexistent times.
            - * Notice: `'NaT'` is not supported.
        """
        return self._ceil_frequency(freq, ambiguous, nonexistent)

    def floor(
        self,
        freq: Literal["D", "h", "m", "s", "ms", "us", "ns"],
        ambiguous: Union[bool, pd.Series[bool], Literal["raise", "infer"]] = "raise",
        nonexistent: Literal["shift_forward", "shift_backward", "raise"] = "raise",
    ) -> pddt:
        """Perform floor operation to specified freqency.

        :param freq: `<str>` frequency to floor to.
            `'D'`: Day / `'h'`: Hour / `'m'`: Minute / `'s'`: Second /
            `'ms'`: Millisecond / `'us'`: Microsecond / `'ns'`: Nanosecond

        :param ambiguous: `<bool>`/`<Series[bool]>`/`'raise'`/`'infer'`.
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from 03:00
            DST to 02:00 non-DST, 02:30:00 local time occurs both at 00:30:00 UTC
            and at 01:30:00 UTC. In such a situation, the ambiguous parameter
            dictates how ambiguous times should be handled.
            - `<bool>`: Marks all times as DST time (True) and non-DST time (False).
            - `<Series[bool]>`: Marks specific times (matching index) as DST time (True)
              and non-DST time (False).
            - `'raise'`: Raises an `PddtValueError` if there are ambiguous times.
            - `'infer'`: Attempt to infer fall dst-transition hours based on order.
            - * Notice: `'NaT'` is not supported.

        :param nonexistent: `'shift_forward'`/`'shift_backward'`/`'raise'`.
            A nonexistent time does not exist in a particular timezone where clocks moved
            forward due to DST.
            - `'shift_forward'`: Shifts nonexistent times forward to the closest existing time.
            - `'shift_backward'`: Shifts nonexistent times backward to the closest existing time.
            - `'raise'`: Raises an `PddtValueError` if there are nonexistent times.
            - * Notice: `'NaT'` is not supported.
        """
        return self._floor_frequency(freq, ambiguous, nonexistent)

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
    ) -> pddt:
        """Adjustment with delta. Equivalent to `pddt + pandas.tseries.offsets.DateOffset`.

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
        except PddtValueError:
            raise
        except Exception as err:
            raise PddtValueError("<pddt> %s" % err) from err

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
    ) -> pddt:
        """Replace the current timestamp. Equivalent to `Series.apply(lambda dt: dt.replace())`.

        The core `replace` method has been cythonized but this method is not vectorized,
        so when dealing with large amount of data, it can still be slow.

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
        except PddtValueError:
            raise
        except Exception as err:
            raise PddtValueError("<pddt> %s" % err) from err

    # Between calculation
    def between(
        self,
        other: Union[pd.Series, pddt, list],
        unit: Literal["Y", "M", "W", "D", "h", "m", "s", "ms", "us"] = "D",
        inclusive: bool = False,
    ) -> pd.Series[int]:
        """Calculate the `ABSOLUTE` delta between two time in the given unit.

        :param other: `<Series>`/`<pddt>`/`list` The other time to compare with.
        :param unit: `<str>` The unit to calculate the delta, accepts: 'Y', 'M', 'D', 'h', 'm', 's', 'ms', 'us'.
        :param inclusive: `<bool>` Setting to `True` will add 1 to the final result.
            Take 'Y' (year) as an example. If `True`, 'Y' (year) delta between 2023-01-01 and
            2024-01-01 will be 2. If `False`, delta between 2023-01-01 and 2024-01-01 will be 1.
        :return: `<pd.Sereis[int]>` The `ABSOLUTE` delta between two time.
        """
        try:
            return self._between(other, unit, inclusive)
        except PddtValueError:
            raise
        except Exception as err:
            raise PddtValueError("<pddt> %s" % err) from err

    # Core methods
    @cython.cfunc
    @cython.inline(True)
    def _new(self, series: object) -> pddt:
        pt = pddt(
            series,
            default=self._default,
            dayfirst=self._dayfirst,
            yearfirst=self._yearfirst,
            utc=self._utc,
            format=self._format,
            exact=self._exact,
        )
        # Pass index
        pt._index = self._index
        return pt

    @cython.cfunc
    @cython.inline(True)
    def _to_datetime(self, timeobj: object) -> object:
        # If series is already datetime64[ns], deepcopy directly
        if isinstance(timeobj, pd.Series) and timeobj.dtype.str.endswith("M8[ns]"):
            return timeobj.copy(True)
        # Try parsing
        else:
            return self._parse_datetime(timeobj)

    @cython.cfunc
    @cython.inline(True)
    def _parse_datetime(self, timeobj: object) -> object:
        # Try parsing
        res = pd.to_datetime(
            timeobj,
            errors="raise" if self._default is None else "coerce",
            dayfirst=self._dayfirst,
            yearfirst=self._yearfirst,
            utc=self._utc,
            format=self._format,
            exact=self._exact,
            unit="ns",
            origin="unix",
            cache=True,
        )
        if isinstance(res, pd.Series):
            try:
                res = cydt.seriesdt64_adjust_to_ns(res)
            except Exception as err:
                raise PddtValueError(
                    "<pddt> Can't fully convert 'timeobj' to `<pandas.Timestamp>`:\n%s\nError: %s"
                    % (res, err)
                ) from err
            return self._fill_default(res)
        elif isinstance(res, pd.DatetimeIndex):
            return self._fill_default(pd.Series(res))
        else:
            raise PddtValueError("<pddt> Unsupported data type: %s" % type(timeobj))

    @cython.cfunc
    @cython.inline(True)
    def _fill_default(self, series: pd.Series) -> object:
        if self._default is not None and series.hasnans:
            return series.fillna(cydt.dt_replace_tzinfo(self._default, series.dt.tz))
        else:
            return series

    @cython.cfunc
    @cython.inline(True)
    def _np_to_series(self, array: np.ndarray) -> object:
        return pd.Series(array, index=self._get_index())

    @cython.cfunc
    @cython.inline(True)
    def _get_index(self) -> object:
        if self._index is None:
            self._index = self._series.index
        return self._index

    @cython.cfunc
    @cython.inline(True)
    def _get_naive(self) -> object:
        if self._naive is None:
            self._naive = self._series.dt.tz_localize(None)
        return self._naive

    @cython.cfunc
    @cython.inline(True)
    def _get_year(self) -> object:
        if self._year is None:
            self._year = self._series.dt.year
        return self._year

    @cython.cfunc
    @cython.inline(True)
    def _get_year_1st(self) -> object:
        if self._year_1st is None:
            self._year_1st = self._get_year_lst() - offsets.YearBegin(1, month=1)
        return self._year_1st

    @cython.cfunc
    @cython.inline(True)
    def _get_year_lst(self) -> object:
        if self._year_lst is None:
            self._year_lst = self._series + offsets.YearEnd(0)
        return self._year_lst

    @cython.cfunc
    @cython.inline(True)
    def _get_quarter(self) -> object:
        if self._quarter is None:
            self._quarter = self._series.dt.quarter
        return self._quarter

    @cython.cfunc
    @cython.inline(True)
    def _get_quarter_1st(self) -> object:
        if self._quarter_1st is None:
            self._quarter_1st = self._get_quarter_lst() - offsets.QuarterBegin(
                1, startingMonth=1
            )
        return self._quarter_1st

    @cython.cfunc
    @cython.inline(True)
    def _get_quarter_lst(self) -> object:
        if self._quarter_lst is None:
            self._quarter_lst = self._series + offsets.QuarterEnd(0)
        return self._quarter_lst

    @cython.cfunc
    @cython.inline(True)
    def _get_month(self) -> object:
        if self._month is None:
            self._month = self._series.dt.month
        return self._month

    @cython.cfunc
    @cython.inline(True)
    def _get_month_1st(self) -> object:
        if self._month_1st is None:
            self._month_1st = self._get_month_lst() - offsets.MonthBegin(1)
        return self._month_1st

    @cython.cfunc
    @cython.inline(True)
    def _get_month_lst(self) -> object:
        if self._month_lst is None:
            self._month_lst = self._series + offsets.MonthEnd(0)
        return self._month_lst

    @cython.cfunc
    @cython.inline(True)
    def _get_month_days(self) -> object:
        if self._month_days is None:
            self._month_days = self._series.dt.days_in_month
        return self._month_days

    @cython.cfunc
    @cython.inline(True)
    def _get_day(self) -> object:
        if self._day is None:
            self._day = self._series.dt.day
        return self._day

    @cython.cfunc
    @cython.inline(True)
    def _get_weekday(self) -> object:
        if self._weekday is None:
            self._weekday = self._series.dt.weekday
        return self._weekday

    @cython.cfunc
    @cython.inline(True)
    def _get_is_leapyear(self) -> object:
        if self._is_leapyear is None:
            self._is_leapyear = self._series.dt.is_leap_year
        return self._is_leapyear

    @cython.cfunc
    @cython.inline(True)
    def _get_days_of_year(self) -> object:
        if self._days_of_year is None:
            self._days_of_year = self._series.dt.day_of_year
        return self._days_of_year

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
            raise PddtValueError("<pddt> Invalid weekday: %s" % repr(weekday))
        return wday

    @cython.cfunc
    @cython.inline(True)
    def _curr_week(self, weekday: object) -> pddt:
        if weekday is None:
            return self
        else:
            delta = pd.TimedeltaIndex(
                self._parse_weekday(weekday) - self._get_weekday().values,
                unit="D",
            )
            return self._new(self._series + delta)

    @cython.cfunc
    @cython.inline(True)
    def _to_week(self, offset: cython.int, weekday: object) -> pddt:
        if weekday is None:
            return self._new(self._series + pd.Timedelta(days=offset * 7))
        else:
            delta = pd.TimedeltaIndex(
                offset * 7 + self._parse_weekday(weekday) - self._get_weekday().values,
                unit="D",
            )
            return self._new(self._series + delta)

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
            raise PddtValueError("<pddt> Invalid month: %s" % repr(month))
        return mth

    @cython.cfunc
    @cython.inline(True)
    def _curr_month(self, day: cython.int) -> pddt:
        if day < 1:
            return self
        elif day == 1:
            return self._new(self._get_month_1st())
        elif 1 < day <= 28:
            return self._new(self._get_month_1st() + offsets.Day(day - 1))
        elif 29 <= day < 31:
            delta = pd.TimedeltaIndex(
                np.minimum(self._get_month_days().values, day) - 1, unit="D"
            )
            return self._new(self._get_month_1st() + delta)
        else:
            return self._new(self._get_month_lst())

    @cython.cfunc
    @cython.inline(True)
    def _to_month(self, offset: cython.int, day: cython.int) -> pddt:
        if day < 1:
            return self._new(self._series + offsets.DateOffset(months=offset))
        elif day == 1:
            return self._new(self._get_month_1st() + offsets.DateOffset(months=offset))
        elif 1 < day <= 28:
            return self._new(
                self._get_month_1st() + offsets.DateOffset(months=offset, days=day - 1)
            )
        elif 29 <= day < 31:
            base = self._get_month_1st() + offsets.DateOffset(months=offset)
            delta = pd.TimedeltaIndex(
                np.minimum(base.dt.days_in_month.values, day) - 1,
                unit="D",
            )
            return self._new(base + delta)
        else:
            return self._new(
                self._get_month_lst()
                + offsets.DateOffset(months=offset)
                + offsets.MonthEnd(0)
            )

    @cython.cfunc
    @cython.inline(True)
    def _curr_quarter(self, month: cython.int, day: cython.int) -> pddt:
        # Validate
        if not 1 <= month <= 3:
            raise PddtValueError(
                "<pddt> Invalid quarter month: %s. Accepts `<int>` 1-3." % repr(month)
            )

        # Convert
        if day < 1:
            base = self._get_quarter_lst() - offsets.QuarterBegin(
                1, startingMonth=month
            )
            delta = pd.TimedeltaIndex(
                np.minimum(base.dt.days_in_month.values, self._get_day().values) - 1,
                unit="D",
            )
            return self._new(base + delta)
        elif day == 1:
            return self._new(
                self._get_quarter_lst() - offsets.QuarterBegin(1, startingMonth=month)
            )
        elif 1 < day <= 28:
            return self._new(
                self._get_quarter_lst()
                - offsets.QuarterBegin(1, startingMonth=month)
                + offsets.Day(day - 1)
            )
        elif 29 <= day < 31:
            base = self._get_quarter_lst() - offsets.QuarterBegin(
                1, startingMonth=month
            )
            delta = pd.TimedeltaIndex(
                np.minimum(base.dt.days_in_month.values, day) - 1,
                unit="D",
            )
            return self._new(base + delta)
        else:
            return self._new(
                self._get_quarter_lst()
                - offsets.QuarterBegin(1, startingMonth=month)
                + offsets.MonthEnd(0)
            )

    @cython.cfunc
    @cython.inline(True)
    def _to_quarter(
        self,
        offset: cython.int,
        month: cython.int,
        day: cython.int,
    ) -> pddt:
        # Validate
        if not 1 <= month <= 3:
            raise PddtValueError(
                "<pddt> Invalid quarter month: %s. Accepts `<int>` 1-3." % repr(month)
            )

        # Convert
        if day < 1:
            base = self._get_quarter_lst() + offsets.DateOffset(
                months=offset * 3 - 3 + month
            )
            delta = pd.TimedeltaIndex(
                np.minimum(base.dt.days_in_month.values, self._get_day().values)
                - base.dt.day.values,
                unit="D",
            )
            return self._new(base + delta)
        elif day == 1:
            return self._new(
                self._get_quarter_lst()
                + offsets.DateOffset(months=offset * 3)
                - offsets.QuarterBegin(1, startingMonth=month)
            )
        elif 1 < day <= 28:
            return self._new(
                self._get_quarter_lst()
                - offsets.QuarterBegin(1, startingMonth=month)
                + offsets.DateOffset(months=offset * 3, days=day - 1)
            )
        elif 29 <= day < 31:
            base = self._get_quarter_lst() + offsets.DateOffset(
                months=offset * 3 - 3 + month
            )
            delta = pd.TimedeltaIndex(
                np.minimum(base.dt.days_in_month.values, day) - base.dt.day.values,
                unit="D",
            )
            return self._new(base + delta)
        else:
            return self._new(
                self._get_quarter_lst()
                + offsets.DateOffset(months=offset * 3 - 3 + month)
                + offsets.MonthEnd(0)
            )

    @cython.cfunc
    @cython.inline(True)
    def _curr_year(self, month: object, day: cython.int) -> pddt:
        # Validate
        if month is None:
            return self._curr_month(day)
        month: cython.int = self._parse_month(month)

        # Convert
        if day < 1:
            base = self._get_year_lst() - offsets.YearBegin(1, month=month)
            delta = pd.TimedeltaIndex(
                np.minimum(base.dt.days_in_month.values, self._get_day().values) - 1,
                unit="D",
            )
            return self._new(base + delta)
        elif day == 1:
            return self._new(self._get_year_lst() - offsets.YearBegin(1, month=month))
        elif 1 < day <= 28:
            return self._new(
                self._get_year_lst()
                - offsets.YearBegin(1, month=month)
                + offsets.Day(day - 1)
            )
        elif 29 <= day < 31:
            base = self._get_year_lst() - offsets.YearBegin(1, month=month)
            delta = pd.TimedeltaIndex(
                np.minimum(base.dt.days_in_month.values, day) - 1,
                unit="D",
            )
            return self._new(base + delta)
        else:
            return self._new(
                self._get_year_lst()
                - offsets.YearBegin(1, month=month)
                + offsets.MonthEnd(0)
            )

    @cython.cfunc
    @cython.inline(True)
    def _to_year(self, offset: cython.int, month: object, day: cython.int) -> pddt:
        # Validate
        if month is None:
            return self._new(
                self._series + offsets.DateOffset(years=offset)
            )._curr_month(day)
        month: cython.int = self._parse_month(month)

        # Convert
        if day < 1:
            base = self._get_year_lst() + offsets.DateOffset(
                years=offset, months=month - 12
            )
            delta = pd.TimedeltaIndex(
                np.minimum(base.dt.days_in_month.values, self._get_day().values)
                - base.dt.day.values,
                unit="D",
            )
            return self._new(base + delta)
        elif day == 1:
            return self._new(
                self._get_year_lst()
                + offsets.DateOffset(years=offset)
                - offsets.YearBegin(1, month=month)
            )
        elif 1 < day <= 28:
            return self._new(
                self._get_year_lst()
                - offsets.YearBegin(1, month=month)
                + offsets.DateOffset(years=offset, days=day - 1)
            )
        elif 29 <= day < 31:
            base = self._get_year_lst() + offsets.DateOffset(
                years=offset, months=month - 12
            )
            delta = pd.TimedeltaIndex(
                np.minimum(base.dt.days_in_month.values, day) - base.dt.day.values,
                unit="D",
            )
            return self._new(base + delta)
        else:
            return self._new(
                self._get_year_lst()
                + offsets.DateOffset(years=offset, months=month - 12)
                + offsets.MonthEnd(0)
            )

    @cython.cfunc
    @cython.inline(True)
    def _validate_am_non(self, ambiguous: object, nonexistent: str):
        if ambiguous == "NaT":
            raise PddtValueError("<pddt> `ambiguous == 'NaT'` is not supported.")
        if nonexistent == "NaT":
            raise PddtValueError("<pddt> `nonexistent == 'NaT'` is not supported.")

    @cython.cfunc
    @cython.inline(True)
    def _tz_localize(
        self,
        series: pd.Series,
        tz: object,
        ambiguous: object,
        nonexistent: str,
    ) -> object:
        self._validate_am_non(ambiguous, nonexistent)
        try:
            return series.dt.tz_localize(tz, ambiguous, nonexistent)
        except Exception as err:
            if isinstance(tz, str) and tz in str(err):
                raise PddtValueError("<pddt> Invalid timezone: %s" % repr(tz)) from err
            else:
                raise PddtValueError("<pddt> %s" % err) from err

    @cython.cfunc
    @cython.inline(True)
    def _tz_convert(self, series: pd.Series, tz: object) -> object:
        try:
            return series.dt.tz_convert(cydt.gen_timezone_local() if tz is None else tz)
        except Exception as err:
            if isinstance(tz, str) and tz in str(err):
                raise PddtValueError("<pddt> Invalid timezone: %s" % repr(tz)) from err
            else:
                raise PddtValueError("<pddt> %s" % err) from err

    @cython.cfunc
    @cython.inline(True)
    def _tz_switch(
        self,
        series: pd.Series,
        targ_tz: object,
        base_tz: object,
        ambiguous: object,
        nonexistent: str,
        naive: cython.bint,
    ) -> object:
        # Already timezone-aware: convert to targ_tz
        if self.tzinfo is not None:
            series = self._tz_convert(series, targ_tz)
        # Localize to base_tz & convert to targ_tz
        elif isinstance(base_tz, (str, datetime.tzinfo)):
            series = self._tz_convert(
                self._tz_localize(series, base_tz, ambiguous, nonexistent), targ_tz
            )
        # Invalid base_tz
        else:
            raise PddtValueError(
                "<pddt> Cannot switch timezone without 'base_tz' for naive datetime."
            )
        # Return
        return series.dt.tz_localize(None) if naive else series

    @cython.cfunc
    @cython.inline(True)
    def _parse_frequency(self, freq: str) -> str:
        frequency = freq.lower()
        if frequency == "m":
            return "T"
        if frequency not in FIXED_FREQUENCY:
            raise PddtValueError(
                "<pddt> Invalid frequency: %s. Accept: 'D'/'h'/'m'/'s'/'ms'/'us'/'ns'."
                % repr(freq)
            )
        return frequency

    @cython.cfunc
    @cython.inline(True)
    def _round_frequency(self, freq: str, ambiguous: object, nonexistent: str) -> pddt:
        self._validate_am_non(ambiguous, nonexistent)
        return self._new(
            self._series.dt.round(self._parse_frequency(freq), ambiguous, nonexistent)
        )

    @cython.cfunc
    @cython.inline(True)
    def _ceil_frequency(self, freq: str, ambiguous: object, nonexistent: str) -> pddt:
        self._validate_am_non(ambiguous, nonexistent)
        return self._new(
            self._series.dt.ceil(self._parse_frequency(freq), ambiguous, nonexistent)
        )

    @cython.cfunc
    @cython.inline(True)
    def _floor_frequency(self, freq: str, ambiguous: object, nonexistent: str) -> pddt:
        self._validate_am_non(ambiguous, nonexistent)
        return self._new(
            self._series.dt.floor(self._parse_frequency(freq), ambiguous, nonexistent)
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
    ) -> pddt:
        return self._new(
            self._series
            + offsets.DateOffset(
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
    ) -> pddt:
        return self._new(
            self._series.apply(
                lambda dt: cydt.dt_replace(
                    dt,
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
        )

    @cython.cfunc
    @cython.inline(True)
    def _between(
        self,
        other: object,
        unit: str,
        inclusive: cython.bint,
    ) -> object:
        if isinstance(other, pd.Series):
            if other.dtype.str.endswith("M8[ns]"):
                return self._between_series(other, unit, inclusive)
            if "M8" in other.dtype.str or other.dtype == "object":
                return self._between_series(
                    self._parse_datetime(other), unit, inclusive
                )
            raise PddtValueError(
                "<pddt> Can only compare with `pandas.Series` with datetime64 dtype."
            )
        elif isinstance(other, pddt):
            return self._between_pddt(other, unit, inclusive)
        elif isinstance(other, list):
            return self._between_series(self._parse_datetime(other), unit, inclusive)
        else:
            raise PddtValueError("<pddt> Unsupported data type: %s" % (type(other)))

    @cython.cfunc
    @cython.inline(True)
    def _between_pddt(self, pt: pddt, unit: str, inclusive: cython.bint) -> object:
        return self._between_series(pt._series, unit, inclusive)

    @cython.cfunc
    @cython.inline(True)
    def _between_series(
        self,
        series: pd.Series,
        unit: str,
        inclusive: cython.bint,
    ) -> object:
        # Validate
        if self._series.__len__() != series.__len__():
            raise PddtValueError(
                "<pddt> Can't campare between Series with different length: %d vs %d"
                % (self._series.__len__(), series.__len__())
            )

        # Determine cast type
        cast_type = "<M8[D]" if unit == "W" else "<M8[%s]" % unit

        # Calculate delta
        try:
            delta: np.ndarray = np.abs(
                (
                    self._series.values.astype(cast_type).astype(np.int64)
                    - series.values.astype(cast_type).astype(np.int64)
                )
            )
        except Exception as err:
            if unit in str(err):
                raise PddtValueError(
                    "<pddt> Invalid time unit: %s" % repr(unit)
                ) from err
            else:
                raise PddtValueError("<pddt> %s" % err) from err

        # Adjustment for week
        if unit == "W":
            base: np.ndarray = np.minimum(self._series.values, series.values)
            base = pd.DatetimeIndex(base).weekday.values
            delta = (delta + base) // 7

        # Adjustment for inclusive
        if inclusive:
            delta += 1

        # Return Delta
        return self._np_to_series(delta)

    # Sepcial methods - addition
    @cython.cfunc
    @cython.inline(True)
    def _to_pddt(self, other: object) -> object:
        if isinstance(other, pd.Series) and "M8" in other.dtype.str:
            return self._new(other)
        else:
            return other

    def __add__(self, other: object) -> Union[pddt, pd.Series]:
        try:
            return self._to_pddt(self._series + other)
        except PddtValueError:
            raise
        except Exception as err:
            raise PddtValueError("<pddt> %s" % err) from err

    def __radd__(self, other: object) -> Union[pddt, pd.Series]:
        try:
            return self._to_pddt(other + self._series)
        except PddtValueError:
            raise
        except Exception as err:
            raise PddtValueError("<pddt> %s" % err) from err

    # Sepcial methods - substruction
    @cython.cfunc
    @cython.inline(True)
    def _adj_other(self, other: object) -> object:
        if isinstance(other, pd.Series):
            if other.dtype.str.endswith("M8[ns]"):
                return other
            elif "M8" in other.dtype.str or other.dtype == "object":
                return self._parse_datetime(other)
            else:
                return other
        elif isinstance(other, pddt):
            return other.dt
        elif isinstance(other, list):
            return self._parse_datetime(other)
        else:
            return other

    def __sub__(self, other: object) -> Union[pddt, pd.Series]:
        try:
            return self._to_pddt(self._series - self._adj_other(other))
        except PddtValueError:
            raise
        except Exception as err:
            raise PddtValueError("<pddt> %s" % err) from err

    def __rsub__(self, other: object) -> Union[pddt, pd.Series]:
        try:
            return self._to_pddt(self._adj_other(other) - self._series)
        except PddtValueError:
            raise
        except Exception as err:
            raise PddtValueError("<pddt> %s" % err) from err

    # Special methods - comparison
    def __eq__(self, other: object) -> pd.Series[bool]:
        try:
            return self._series == self._adj_other(other)
        except PddtValueError:
            raise
        except Exception as err:
            raise PddtValueError("<pddt> %s" % err) from err

    def __ne__(self, other: object) -> pd.Series[bool]:
        try:
            return self._series != self._adj_other(other)
        except PddtValueError:
            raise
        except Exception as err:
            raise PddtValueError("<pddt> %s" % err) from err

    def __gt__(self, other: object) -> pd.Series[bool]:
        try:
            return self._series > self._adj_other(other)
        except PddtValueError:
            raise
        except Exception as err:
            raise PddtValueError("<pddt> %s" % err) from err

    def __ge__(self, other: object) -> pd.Series[bool]:
        try:
            return self._series >= self._adj_other(other)
        except PddtValueError:
            raise
        except Exception as err:
            raise PddtValueError("<pddt> %s" % err) from err

    def __lt__(self, other: object) -> pd.Series[bool]:
        try:
            return self._series < self._adj_other(other)
        except PddtValueError:
            raise
        except Exception as err:
            raise PddtValueError("<pddt> %s" % err) from err

    def __le__(self, other: object) -> pd.Series[bool]:
        try:
            return self._series <= self._adj_other(other)
        except PddtValueError:
            raise
        except Exception as err:
            raise PddtValueError("<pddt> %s" % err) from err

    def equals(self, other: Union[pd.Series, list]) -> bool:
        """Test whether two objects contain the same elements.
        (Equivalent to `pandas.Series.equal` method).

        Support comparison between `pddt` and `pandas.Series`."""
        return self._series.equals(self._adj_other(other))

    # Special methods - copy
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(check=False)
    def _copy(self) -> pddt:
        return self._new(self._series)

    def copy(self) -> pddt:
        "Make a (deep)copy of this object's data."
        return self._copy()

    def __copy__(self, *args, **kwargs) -> pddt:
        return self._copy()

    def __deepcopy__(self, *args, **kwargs) -> pddt:
        return self._copy()

    # Special methods - represent
    def __repr__(self) -> str:
        return "<pddt> %s" % self._series.__repr__()

    def __str__(self) -> str:
        return self._series.__repr__()

    def __len__(self) -> int:
        return self._series.__len__()

    def __contains__(self, key) -> bool:
        "True if the key is in the info axis"
        return self._series.__contains__(key)

    def __getitem__(self, key) -> pd.Timestamp:
        return self._series.__getitem__(key)

    def __iter__(self) -> Iterator[pd.Timestamp]:
        return self._series.__iter__()

    def __array__(self) -> np.ndarray:
        return self._series.__array__()


# Exceptions ----------------------------------------------------------------------------------
class PddtValueError(ValueError):
    """The one and only exception this module will raise."""
