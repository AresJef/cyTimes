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
ZONEINFO: type = ZoneInfo
STRUCT_TIME: type = struct_time

# . numpy types
DATETIME64: type = np.datetime64
DT64_ARRAY: type = np.dtypes.DateTime64DType
TIMEDELTA64: type = np.timedelta64
TD64_ARRAY: type = np.dtypes.TimeDelta64DType

# . pandas types
SERIES: type = Series
DATETIMEINDEX: type = DatetimeIndex
TIMESTAMP: type = Timestamp
DT64TZ_ARRAY: type = DatetimeTZDtype
TIMEDELTAINDEX: type = TimedeltaIndex
TIMEDELTA: type = Timedelta

# . pandas offsets
BASEOFFSET: type = BaseOffset
OFST_DATEOFFSET: object = offsets.DateOffset
OFST_MICRO: object = offsets.Micro
OFST_DAY: object = offsets.Day
OFST_MONTHBEGIN: object = offsets.MonthBegin
OFST_MONTHEND: object = offsets.MonthEnd
OFST_QUARTERBEGIN: object = offsets.QuarterBegin
OFST_QUARTEREND: object = offsets.QuarterEnd
OFST_YEARBEGIN: object = offsets.YearBegin
OFST_YEAREND: object = offsets.YearEnd

# . dateutil types
PARSERINFO: type = parserinfo
RELATIVEDELTA: type = relativedelta
