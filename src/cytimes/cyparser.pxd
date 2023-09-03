# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

from cpython cimport datetime

# Unicode
cdef bint uni_isdot(int obj) except -1
cdef bint uni_iscomma(int obj) except -1

# TimeLex
cdef class TimeLex:
    # Attributes
    cdef:
        str _string
        int _strlen, _idx
        list _charstack, _tokenstack
        bint _ended
    # Methods
    cdef str _get_nextchar(self) noexcept
    cdef str _get_token(self) except *
    cpdef list split(self)

# Result
cdef class Result:
    # Attributes
    cdef: 
        int _year, _month, _day, _weekday, _ampm
        int _hour, _minute, _second, _microsecond
        str _tz_name
        int _tzoffset
        bint _century_specified
    # Special methods
    cdef str _represent(self) noexcept

# YMD
cdef class YMD:
    # Attributes
    cdef:
        bint _century_specified
        int _year, _month, _day
        int _validx, _val0, _val1, _val2
        int _yidx, _midx, _didx
    # Resolve
    cdef int _get(self, int idx) noexcept
    cdef _set(self, int val) noexcept
    cdef int _solved_values(self) noexcept
    cdef bint could_be_day(self, int value) except -1
    cdef append(self, object value, int label=?) except *
    cdef resolve(self, bint dayfirst, bint yearfirst) noexcept

# ParserInfo (config)
cdef class ParserInfo:
    # Attributes
    cdef:
        set _jump, _utczone, _pertain
        dict _weekday, _month, _hms, _ampm, _tzoffset
        bint _dayfirst, _yearfirst
    # Methods
    cpdef bint jump(self, str word)
    cpdef int weekday(self, str word)
    cpdef int month(self, str word)
    cpdef int hms(self, str word)
    cpdef int ampm(self, str word)
    cpdef bint utczone(self, str word)
    cpdef int tzoffset(self, str tz)
    cpdef bint pertain(self, str word)
    # Utils
    cdef list _validate_str(self, tuple values) except *
    cdef _add_to_set(self, set set_, tuple values) except *
    cdef _add_to_dict(self, dict dict_, int val, tuple keys) except *
    cdef _set_to_dict(self, dict dict_, int val, tuple keys) except *
    # Conversion
    cpdef from_parserinfo(self, object info)
    # Validate Result
    cdef int _convert_year(self, int year, bint century_specified) noexcept
    cdef Result _adjust_result(self, Result res) noexcept

cdef ParserInfo DEFAULT_PARSERINFO
# Parser
cdef class Parser:
    # Attributes
    cdef ParserInfo __info
    # Parse
    cdef datetime.datetime _parse(self, str timestr, object default, bint dayfirst, bint yearfirst, bint ignoretz, object tzinfos, bint fuzzy) except *
    # Numeric token
    cdef int _parse_numeric_token(self, int idx, str token, list tokens, int token_count, bint fuzzy, YMD ymd, Result res) except -1
    cdef bint _is_numeric_token(self, str token) except -1
    cdef double _convert_numeric_token(self, str token) except *
    cdef _set_hour_min(self, double value, Result res) noexcept
    cdef _set_min_sec(self, double value, Result res) noexcept
    cdef _set_sec_us(self, str token, Result res) noexcept
    cdef int _find_hms_idx(self, int idx, str next_token, list tokens, int token_count, bint allow_jump) except *
    # Month token
    cdef int _parse_month_token(self, int idx, list tokens, int token_count, int month, YMD ymd) except -1
    # AM/PM token
    cdef bint _valid_ampm_flag(self, Result res, bint fuzzy) except -1
    cdef int _adjust_ampm(self, int hour, int ampm) except *
    # Timezone token
    cdef bint _could_be_tzname(self, str token, bint check_tzoffset, Result res) except -1
    cdef int _parse_tzname(self, int idx, str token, list tokens, int token_count, Result res) except -1
    cdef int _prase_tzoffset(self, int idx, str token, list tokens, int token_count, Result res) except -1
    # Build
    cdef datetime.datetime _build(self, Result res, object default, object tzinfos, bint ignoretz) except *
    cdef datetime.datetime _build_datetime(self, Result res, object default, object tzinfos) except *
    cdef datetime.datetime _handle_anbiguous_time(self, datetime.datetime dt, str tzname) noexcept
