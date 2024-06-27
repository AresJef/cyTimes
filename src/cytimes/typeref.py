# cython: language_level=3

# Python imports
from time import struct_time
from zoneinfo import ZoneInfo
import numpy as np
from pandas import Series, offsets
from pandas import TimedeltaIndex, Timedelta
from pandas import DatetimeIndex, Timestamp, DatetimeTZDtype
from pandas._libs.tslibs.offsets import BaseOffset
from dateutil.parser import parserinfo
from dateutil.relativedelta import relativedelta

# Constants -------------------------------------------------------------------------
# . native types
ZONEINFO: type[ZoneInfo] = ZoneInfo
STRUCT_TIME: type[struct_time] = struct_time

# . numpy types
DATETIME64: type[np.datetime64] = np.datetime64
DT64_ARRAY: type = np.dtypes.DateTime64DType
TIMEDELTA64: type[np.timedelta64] = np.timedelta64
TD64_ARRAY: type = np.dtypes.TimeDelta64DType

# . pandas types
SERIES: type[Series] = Series
DATETIMEINDEX: type[DatetimeIndex] = DatetimeIndex
TIMESTAMP: type[Timestamp] = Timestamp
DT64TZ_ARRAY: type[DatetimeTZDtype] = DatetimeTZDtype
TIMEDELTAINDEX: type[TimedeltaIndex] = TimedeltaIndex
TIMEDELTA: type[Timedelta] = Timedelta

# . pandas offsets
BASEOFFSET: type[BaseOffset] = BaseOffset
OFST_DATEOFFSET: type[offsets.DateOffset] = offsets.DateOffset
OFST_MICRO: type[offsets.Micro] = offsets.Micro
OFST_DAY: type[offsets.Day] = offsets.Day
OFST_MONTHBEGIN: type[offsets.MonthBegin] = offsets.MonthBegin
OFST_MONTHEND: type[offsets.MonthEnd] = offsets.MonthEnd
OFST_QUARTERBEGIN: type[offsets.QuarterBegin] = offsets.QuarterBegin
OFST_QUARTEREND: type[offsets.QuarterEnd] = offsets.QuarterEnd
OFST_YEARBEGIN: type[offsets.YearBegin] = offsets.YearBegin
OFST_YEAREND: type[offsets.YearEnd] = offsets.YearEnd

# . dateutil types
PARSERINFO: type[parserinfo] = parserinfo
RELATIVEDELTA: type[relativedelta] = relativedelta
