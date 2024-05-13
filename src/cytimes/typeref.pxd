# cython: language_level=3

# Constants
cdef:
    # . native types
    object ZONEINFO, STRUCT_TIME
    # . numpy type
    object DATETIME64, DT64_ARRAY, TIMEDELTA64, TD64_ARRAY
    # . pandas type
    object SERIES, DATETIMEINDEX, TIMESTAMP, DT64TZ_ARRAY, TIMEDELTAINDEX, TIMEDELTA
    # . pandas offset
    object BASEOFFSET, OFST_DATEOFFSET, OFST_MICRO, OFST_DAY
    object OFST_MONTHBEGIN, OFST_MONTHEND
    object OFST_QUARTERBEGIN, OFST_QUARTEREND
    object OFST_YEARBEGIN, OFST_YEAREND
    # . dateutil types
    object PARSERINFO, RELATIVEDELTA
