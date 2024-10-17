# cython: language_level=3

from cpython cimport datetime
from cytimes cimport utils
from cytimes.parser cimport Configs

# Utils
cdef bint is_pydt(object o) except -1
cdef _Pydt pydt_new(
    int year=?, int month=?, int day=?,
    int hour=?, int minute=?, int second=?,
    int microsecond=?, object tz=?, int fold=?
)
cdef _Pydt pydt_fr_dt(datetime.datetime dt)
cdef _Pydt pydt_fr_dtobj(
    object dtobj, object default=?, 
    object year1st=?, object day1st=?,
    bint ignoretz=?, bint isoformat=?, Configs cfg=?
)

# Pydt (Python Datetime)
cdef class _Pydt(datetime.datetime):
    # Constructor
    cdef inline _Pydt _from_dt(self, datetime.datetime dt)
    # Convertor
    cpdef str ctime(self)
    cpdef str strftime(self, str format)
    cpdef str isoformat(self, str sep=?)
    cpdef utils.tm timedict(self)
    cpdef utils.tm utctimedict(self)
    cpdef tuple timetuple(self)
    cpdef tuple utctimetuple(self)
    cpdef int toordinal(self) except -1
    cpdef double seconds(self, bint utc=?)
    cpdef long long microseconds(self, bint utc=?)
    cpdef double timestamp(self)
    cpdef datetime.date date(self)
    cpdef datetime.time time(self)
    cpdef datetime.time timetz(self)
    # Manipulator
    cpdef _Pydt replace(
        self, int year=?, int month=?, int day=?,
        int hour=?, int minute=?, int second=?,
        int microsecond=?, object tz=?, int fold=?,
    )
    # . year
    cpdef _Pydt to_curr_year(self, object month=?, int day=?)
    cpdef _Pydt to_prev_year(self, object month=?, int day=?)
    cpdef _Pydt to_next_year(self, object month=?, int day=?)
    cpdef _Pydt to_year(self, int offset, object month=?, int day=?)
    # . quarter
    cpdef _Pydt to_curr_quarter(self, int month=?, int day=?)
    cpdef _Pydt to_prev_quarter(self, int month=?, int day=?)
    cpdef _Pydt to_next_quarter(self, int month=?, int day=?)
    cpdef _Pydt to_quarter(self, int offset, int month=?, int day=?)
    # . month
    cpdef _Pydt to_curr_month(self, int day=?)
    cpdef _Pydt to_prev_month(self, int day=?)
    cpdef _Pydt to_next_month(self, int day=?)
    cpdef _Pydt to_month(self, int offset, int day=?)
    # . weekday
    cpdef _Pydt to_monday(self)
    cpdef _Pydt to_tuesday(self)
    cpdef _Pydt to_wednesday(self)
    cpdef _Pydt to_thursday(self)
    cpdef _Pydt to_friday(self)
    cpdef _Pydt to_saturday(self)
    cpdef _Pydt to_sunday(self)
    cpdef _Pydt to_curr_weekday(self, object weekday=?)
    cpdef _Pydt to_prev_weekday(self, object weekday=?)
    cpdef _Pydt to_next_weekday(self, object weekday=?)
    cpdef _Pydt to_weekday(self, int offset, object weekday=?)
    cdef inline _Pydt _to_curr_weekday(self, int weekday)
    # . day
    cpdef _Pydt to_yesterday(self)
    cpdef _Pydt to_tomorrow(self)
    cpdef _Pydt to_day(self, int offset)
    # . date&time
    cpdef _Pydt to_datetime(
        self, int year=?, int month=?, int day=?,
        int hour=?, int minute=?, int second=?,
        int millisecond=?, int microsecond=?,
    )
    cpdef _Pydt to_date(self, int year=?, int month=?, int day=?)
    cpdef _Pydt to_time(
        self, int hour=?, int minute=?, int second=?,
        int millisecond=?, int microsecond=?,
    )
    cpdef _Pydt to_first_of(self, str unit)
    cpdef _Pydt to_last_of(self, str unit)
    cpdef _Pydt to_start_of(self, str unit)
    cpdef _Pydt to_end_of(self, str unit)
    # . frequency
    cpdef _Pydt freq_round(self, str freq)
    cpdef _Pydt freq_ceil(self, str freq)
    cpdef _Pydt freq_floor(self, str freq)
    # Calendar
    # . iso
    cpdef int isoweekday(self) except -1
    cpdef int isoweek(self) except -1
    cpdef utils.iso isocalendar(self)
    # . year
    cdef inline int _prop_year(self) except -1
    cpdef bint is_leap_year(self) except -1
    cpdef bint is_long_year(self) except -1
    cpdef int leap_bt_years(self, int year) except -1
    cpdef int days_in_year(self) except -1
    cpdef int days_bf_year(self) except -1
    cpdef int days_of_year(self) except -1
    cpdef bint is_year(self, int year) except -1
    # . quarter
    cdef inline int _prop_quarter(self) except -1
    cpdef int days_in_quarter(self) except -1
    cpdef int days_bf_quarter(self) except -1
    cpdef int days_of_quarter(self) except -1
    cpdef int quarter_first_month(self) except -1
    cpdef int quarter_last_month(self) except -1
    cpdef bint is_quarter(self, int quarter) except -1
    # . month
    cdef inline int _prop_month(self) except -1
    cpdef int days_in_month(self) except -1
    cpdef int days_bf_month(self) except -1
    cpdef int days_of_month(self) except -1
    cpdef bint is_month(self, object month) except -1
    # . weekday
    cpdef int weekday(self) except -1
    cpdef bint is_weekday(self, object weekday) except -1
    # . day
    cdef inline int _prop_day(self) except -1
    cpdef bint is_day(self, int day) except -1
    # . time
    cdef inline int _prop_hour(self) except -1
    cdef inline int _prop_minute(self) except -1
    cdef inline int _prop_second(self) except -1
    cdef inline int _prop_millisecond(self) except -1
    cdef inline int _prop_microsecond(self) except -1
    # . date&time
    cpdef bint is_first_of(self, str unit) except -1
    cpdef bint is_last_of(self, str unit) except -1
    cpdef bint is_start_of(self, str unit) except -1
    cpdef bint is_end_of(self, str unit) except -1
    # Timezone
    cdef inline object _prop_tzinfo(self)
    cdef inline int _prop_fold(self) except -1
    cpdef bint is_local(self) except -1
    cpdef bint is_utc(self) except -1
    cpdef bint is_dst(self) except -1
    cpdef str tzname(self)
    cpdef datetime.timedelta utcoffset(self)
    cpdef object utcoffset_seconds(self)
    cpdef datetime.timedelta dst(self)
    cpdef _Pydt astimezone(self, object tz=?)
    cpdef _Pydt tz_localize(self, object tz=?)
    cpdef _Pydt tz_convert(self, object tz=?)
    cpdef _Pydt tz_switch(self, object targ_tz, object base_tz=?, bint naive=?)
    # Arithmetic
    cpdef _Pydt add(
        self, int years=?, int quarters=?, int months=?, 
        int weeks=?, int days=?, int hours=?, int minutes=?, 
        int seconds=?, int milliseconds=?, int microseconds=?
    )
    cpdef _Pydt sub(
        self, int years=?, int quarters=?, int months=?, 
        int weeks=?, int days=?, int hours=?, int minutes=?, 
        int seconds=?, int milliseconds=?, int microseconds=?
    )
    cpdef _Pydt avg(self, object dtobj=?)
    cpdef long long diff(self, object dtobj, str unit, str bounds=?) except -1
    cdef inline _Pydt _add_timedelta(self, int days, int seconds, int microseconds)
    # Comparison
    cpdef bint is_past(self) except -1
    cpdef bint is_future(self) except -1
    cdef inline _Pydt _closest(self, tuple dtobjs)
    cdef inline _Pydt _farthest(self, tuple dtobjs)
