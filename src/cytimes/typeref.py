# cython: language_level=3

# Python imports
from datetime import timezone
from zoneinfo import ZoneInfo
from dateutil.parser import parserinfo
from dateutil.relativedelta import relativedelta

# Constants -------------------------------------------------------------------------
# . native types
ZONEINFO: type[ZoneInfo] = ZoneInfo
TIMEZONE: type[timezone] = timezone
# . dateutil types
PARSERINFO: type[parserinfo] = parserinfo
RELATIVEDELTA: type[relativedelta] = relativedelta
