from typing import Literal
import datetime, numpy as np
from typing_extensions import overload, Self, TypeVar
from cytimes.parser import Configs

# Types
_DateT = TypeVar("_DateT", bound=datetime.date)
_DatetimeT = TypeVar("_DatetimeT", bound=datetime.datetime)
_TimedeltaT = TypeVar("_TimedeltaT", bound=datetime.timedelta)

# Pydt
class Pydt(datetime.datetime):
    def __new__(
        cls,
        year: int = 1,
        month: int = 1,
        day: int = 1,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        microsecond: int = 0,
        tzinfo: datetime.tzinfo | str | None = None,
        *,
        fold: int = 0,
    ) -> Self: ...
    # Constructor ------------------------------------------------------
    @classmethod
    def parse(
        cls,
        dtobj: object,
        default: object | None = None,
        year1st: bool | None = None,
        day1st: bool | None = None,
        ignoretz: bool = False,
        isoformat: bool = True,
        cfg: Configs | None = None,
    ) -> Self: ...
    @classmethod
    def now(cls, tz: datetime.tzinfo | str | None = None) -> Self: ...
    @classmethod
    def utcnow(cls) -> Self: ...
    @classmethod
    def today(cls) -> Self: ...
    @classmethod
    def combine(
        cls,
        date: datetime.date | str | None = None,
        time: datetime.time | str | None = None,
        tz: datetime.tzinfo | str | None = None,
    ) -> Self: ...
    @classmethod
    def fromordinal(
        cls,
        ordinal: int,
        tz: datetime.tzinfo | str | None = None,
    ) -> Self: ...
    @classmethod
    def fromseconds(
        cls,
        seconds: int | float,
        tz: datetime.tzinfo | str | None = None,
    ) -> Self: ...
    @classmethod
    def fromicroseconds(
        cls,
        us: int,
        tz: datetime.tzinfo | str | None = None,
    ) -> Self: ...
    @classmethod
    def fromtimestamp(
        cls,
        ts: int | float,
        tz: datetime.tzinfo | str | None = None,
    ) -> Self: ...
    @classmethod
    def utcfromtimestamp(cls, ts: int | float) -> Self: ...
    @classmethod
    def fromisoformat(cls, dtstr: str) -> Self: ...
    @classmethod
    def fromisocalendar(
        cls,
        year: int,
        week: int,
        day: int,
        tz: datetime.tzinfo | str | None = None,
    ) -> Self: ...
    @classmethod
    def fromdate(
        cls,
        date: datetime.date,
        tz: datetime.tzinfo | str | None = None,
    ) -> Self: ...
    @classmethod
    def fromdatetime(cls, dt: datetime.datetime) -> Self: ...
    @classmethod
    def fromdatetime64(
        cls,
        dt64: np.datetime64,
        tz: datetime.tzinfo | str | None = None,
    ) -> Self: ...
    @classmethod
    def strptime(cls, dtstr: str, fmt: str) -> Self: ...
    # Convertor --------------------------------------------------------
    def ctime(self) -> str: ...
    def strftime(self, fmt: str) -> str: ...
    def isoformat(self, sep: str = "T") -> str: ...
    def timedict(self) -> dict[str, int]: ...
    def utctimedict(self) -> dict[str, int]: ...
    def timetuple(self) -> tuple[int, ...]: ...
    def utctimetuple(self) -> tuple[int, ...]: ...
    def toordinal(self) -> int: ...
    def seconds(self, utc: bool = False) -> float: ...
    def microseconds(self, utc: bool = False) -> int: ...
    def timestamp(self) -> float: ...
    def date(self) -> datetime.date: ...
    def time(self) -> datetime.time: ...
    def timetz(self) -> datetime.time: ...
    # Manipulator ------------------------------------------------------
    def replace(
        self,
        year: int = -1,
        month: int = -1,
        day: int = -1,
        hour: int = -1,
        minute: int = -1,
        second: int = -1,
        microsecond: int = -1,
        tz: datetime.tzinfo | str | None = -1,
        fold: int = -1,
    ) -> Self: ...
    # . year
    def to_curr_year(self, month: int | str | None = None, day: int = -1) -> Self: ...
    def to_prev_year(self, month: int | str | None = None, day: int = -1) -> Self: ...
    def to_next_year(self, month: int | str | None = None, day: int = -1) -> Self: ...
    def to_year(
        self,
        offset: int,
        month: int | str | None = None,
        day: int = -1,
    ) -> Self: ...
    # . quarter
    def to_curr_quarter(self, month: int = -1, day: int = -1) -> Self: ...
    def to_prev_quarter(self, month: int = -1, day: int = -1) -> Self: ...
    def to_next_quarter(self, month: int = -1, day: int = -1) -> Self: ...
    def to_quarter(self, offset: int, month: int = -1, day: int = -1) -> Self: ...
    # . month
    def to_curr_month(self, day: int = -1) -> Self: ...
    def to_prev_month(self, day: int = -1) -> Self: ...
    def to_next_month(self, day: int = -1) -> Self: ...
    def to_month(self, offset: int, day: int = -1) -> Self: ...
    # . weekday
    def to_monday(self) -> Self: ...
    def to_tuesday(self) -> Self: ...
    def to_wednesday(self) -> Self: ...
    def to_thursday(self) -> Self: ...
    def to_friday(self) -> Self: ...
    def to_saturday(self) -> Self: ...
    def to_sunday(self) -> Self: ...
    def to_curr_weekday(self, weekday: int | str = None) -> Self: ...
    def to_prev_weekday(self, weekday: int | str = None) -> Self: ...
    def to_next_weekday(self, weekday: int | str = None) -> Self: ...
    def to_weekday(self, offset: int, weekday: int | str = None) -> Self: ...
    # . day
    def to_yesterday(self) -> Self: ...
    def to_tomorrow(self) -> Self: ...
    def to_day(self, offset: int) -> Self: ...
    # . date&time
    def to_datetime(
        self,
        year: int = -1,
        month: int = -1,
        day: int = -1,
        hour: int = -1,
        minute: int = -1,
        second: int = -1,
        millisecond: int = -1,
        microsecond: int = -1,
    ) -> Self: ...
    def to_date(self, year: int = -1, month: int = -1, day: int = -1) -> Self: ...
    def to_time(
        self,
        hour: int = -1,
        minute: int = -1,
        second: int = -1,
        millisecond: int = -1,
        microsecond: int = -1,
    ) -> Self: ...
    def to_first_of(self, unit: str | Literal["Y", "Q", "M", "W"]) -> Self: ...
    def to_last_of(self, unit: str | Literal["Y", "Q", "M", "W"]) -> Self: ...
    def to_start_of(
        self,
        unit: str | Literal["Y", "Q", "M", "W", "D", "h", "m", "s", "ms"],
    ) -> Self: ...
    def to_end_of(
        self,
        unit: str | Literal["Y", "Q", "M", "W", "D", "h", "m", "s", "ms"],
    ) -> Self: ...
    # . frequency
    def freq_round(self, freq: Literal["D", "h", "m", "s", "ms", "us"]) -> Self: ...
    def freq_ceil(self, freq: Literal["D", "h", "m", "s", "ms", "us"]) -> Self: ...
    def freq_floor(self, freq: Literal["D", "h", "m", "s", "ms", "us"]) -> Self: ...
    # Calendar ---------------------------------------------------------
    # . iso
    def isoweekday(self) -> int: ...
    def isoweek(self) -> int: ...
    def isocalendar(self) -> dict[str, int]: ...
    # . year
    @property
    def year(self) -> int: ...
    def is_leap_year(self) -> bool: ...
    def is_long_year(self) -> bool: ...
    def leap_bt_years(self, year: int) -> int: ...
    def days_in_year(self) -> int: ...
    def days_bf_year(self) -> int: ...
    def days_of_year(self) -> int: ...
    def is_year(self, year: int) -> bool: ...
    # . quarter
    @property
    def quarter(self) -> int: ...
    def days_in_quarter(self) -> int: ...
    def days_bf_quarter(self) -> int: ...
    def days_of_quarter(self) -> int: ...
    def quarter_first_month(self) -> int: ...
    def quarter_last_month(self) -> int: ...
    def is_quarter(self, quarter: int) -> bool: ...
    # . month
    @property
    def month(self) -> int: ...
    def days_in_month(self) -> int: ...
    def days_bf_month(self) -> int: ...
    def days_of_month(self) -> int: ...
    def is_month(self, month: int | str) -> bool: ...
    # . weekday
    def weekday(self) -> int: ...
    def is_weekday(self, weekday: int | str) -> bool: ...
    # . day
    @property
    def day(self) -> int: ...
    def is_day(self, day: int) -> bool: ...
    # . time
    @property
    def hour(self) -> int: ...
    @property
    def minute(self) -> int: ...
    @property
    def second(self) -> int: ...
    @property
    def millisecond(self) -> int: ...
    @property
    def microsecond(self) -> int: ...
    # . date&time
    def is_first_of(self, unit: str | Literal["Y", "Q", "M", "W"]) -> bool: ...
    def is_last_of(self, unit: str | Literal["Y", "Q", "M", "W"]) -> bool: ...
    def is_start_of(
        self,
        unit: str | Literal["Y", "Q", "M", "W", "D", "h", "m", "s", "ms"],
    ) -> bool: ...
    def is_end_of(
        self,
        unit: str | Literal["Y", "Q", "M", "W", "D", "h", "m", "s", "ms"],
    ) -> bool: ...
    # Timezone ---------------------------------------------------------
    @property
    def tz_available(self) -> set[str]: ...
    @property
    def tzinfo(self) -> datetime.tzinfo | None: ...
    @property
    def fold(self) -> int: ...
    def is_local(self) -> bool: ...
    def is_utc(self) -> bool: ...
    def is_dst(self) -> bool: ...
    def tzname(self) -> str | None: ...
    def utcoffset(self) -> datetime.timedelta | None: ...
    def utcoffset_seconds(self) -> int | None: ...
    def dst(self) -> datetime.timedelta | None: ...
    def astimezone(self, tz: datetime.tzinfo | str | None = None) -> Self: ...
    def tz_localize(self, tz: datetime.tzinfo | str | None = None) -> Self: ...
    def tz_convert(self, tz: datetime.tzinfo | str | None = None) -> Self: ...
    def tz_switch(
        self,
        targ_tz: datetime.tzinfo | str | None,
        base_tz: datetime.tzinfo | str | None = None,
        naive: bool = False,
    ) -> Self: ...
    # Arithmetic ---------------------------------------------------------
    def add(
        self,
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
    ) -> Self: ...
    def sub(
        self,
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
    ) -> Self: ...
    def avg(self, dtobj: object = None) -> Self: ...
    def diff(
        self,
        dtobj: object,
        unit: Literal["Y", "Q", "M", "W", "D", "h", "m", "s", "ms", "us"],
        bounds: Literal["both", "one", "none"] = "both",
    ) -> int: ...
    # . addition
    @overload
    def __add__(self, o: _TimedeltaT) -> Self: ...
    @overload
    def __add__(self, o: np.timedelta64) -> Self: ...
    # . right addition
    @overload
    def __radd__(self, o: _TimedeltaT) -> Self: ...
    # . subtraction
    @overload
    def __sub__(self, o: _TimedeltaT) -> Self: ...
    @overload
    def __sub__(self, o: np.timedelta64) -> Self: ...
    @overload
    def __sub__(self, o: _DatetimeT) -> datetime.timedelta: ...
    @overload
    def __sub__(self, o: _DateT) -> datetime.timedelta: ...
    @overload
    def __sub__(self, o: str) -> datetime.timedelta: ...
    @overload
    def __sub__(self, o: np.datetime64) -> datetime.timedelta: ...
    # Comparison ---------------------------------------------------------
    def is_past(self) -> bool: ...
    def is_future(self) -> bool: ...
    def closest(self, *dtobjs: object) -> Self: ...
    def farthest(self, *dtobjs: object) -> Self: ...
    def __eq__(self, o: object) -> bool: ...
    def __le__(self, o: object) -> bool: ...
    def __lt__(self, o: object) -> bool: ...
    def __ge__(self, o: object) -> bool: ...
    def __gt__(self, o: object) -> bool: ...
    # Representation -----------------------------------------------------
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __format__(self, fmt: str) -> str: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, _: dict) -> Self: ...
