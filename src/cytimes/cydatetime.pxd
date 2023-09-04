# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

cimport numpy as np
from cpython cimport datetime
from cytimes cimport cytime

# Constants
cdef:
    int MAX_ORD, DI400Y, DI100Y, DI4Y, DI1Y
    int[13] DAYS_BR_MONTH
    int[5] DAYS_BR_QUARTER
    object UTC
    datetime.datetime EPOCH_NAI, EPOCH_UTC
    long long EPOCH_US, EPOCH_SEC, EPOCH_DAY
    long long DT_MIN_US, DT_MAX_US
    long long NS_DAY, NS_HOUR, NS_MINUTE, NS_SECOND, NS_MILLISECOND, NS_MICROSECOND, NS_NANOSECOND
    long long US_DAY, US_HOUR, US_MINUTE, US_SECOND, US_MILLISECOND, US_MICROSECOND
    long long MS_DAY, MS_HOUR, MS_MINUTE, MS_SECOND, MS_MILLISECOND
    long long SEC_DAY, SEC_HOUR, SEC_MINUTE, SEC_SECOND
    long long MIN_DAY, MIN_HOUR, MIN_MINUTE
    long long HOUR_DAY, HOUR_HOUR

# Struct
cdef struct ymd:
    int year
    int month
    int day

cdef struct hms:
    int hour
    int minute
    int second
    int microsecond

cdef struct iso:
    int year
    int week
    int weekday

# Calendar Year
cdef bint is_leapyear(int year) except -1
cdef int leap_bt_years(int year1, int year2) except -1
cdef int days_in_year(int year) except -1
cdef int days_bf_year(int year) except -1
cdef int days_of_year(int year, int month, int day) except -1
# Calendar Quarter
cdef int quarter_of_month(int month) except -1
cdef int days_in_quarter(int year, int month) except -1
cdef int days_bf_quarter(int year, int month) except -1
cdef int days_of_quarter(int year, int month, int day) except -1
cdef int quarter_1st_month(int month) except -1
cdef int quarter_lst_month(int month) except -1
# Calendar Month
cdef int days_in_month(int year, int month) except -1
cdef int days_bf_month(int year, int month) except -1
# Calendar Week
cdef int ymd_weekday(int year, int month, int day) except -1
cdef int ymd_isoweekday(int year, int month, int day) except -1
cdef int ymd_isoweek(int year, int month, int day) except -1
cdef int ymd_isoyear(int year, int month, int day) except -1
cdef iso ymd_isocalendar(int year, int month, int day) noexcept
cdef int isoweek_1st_ordinal(int year) except -1
# Calendar Conversion
cdef int ymd_to_ordinal(int year, int month, int day) except -1
cdef ymd ordinal_to_ymd(int ordinal) noexcept
cdef hms microseconds_to_hms(long long microseconds) noexcept
# Time
cdef double time() noexcept
cdef cytime.tm localtime() except *
cdef cytime.tm localize_time(double timestamp) except *
cdef long long localize_timestamp(double timestamp) except *
# Date
cdef datetime.date gen_date(int year=?, int month=?, int day=?) except *
cdef datetime.date gen_date_local() noexcept
cdef long long date_mktime(datetime.date date) except *
cdef bint is_date(object obj) except -1
cdef bint is_date_exact(object obj) except -1
cdef int get_year(object obj) except -1
cdef int get_month(object obj) except -1
cdef int get_day(object obj) except -1
cdef int get_weekday(object obj) except -1
cdef int get_isoweekday(object obj) except -1
cdef int get_isoweek(object obj) except -1
cdef int get_isoyear(object obj) except -1
cdef iso get_isocalendar(object obj) except *
cdef bint get_is_leapyear(object obj) except -1
cdef int get_days_in_year(object obj) except -1
cdef int get_days_bf_year(object obj) except -1
cdef int get_days_of_year(object obj) except -1
cdef int get_quarter(object obj) except -1
cdef int get_days_in_quarter(object obj) except -1
cdef int get_days_bf_quarter(object obj) except -1
cdef int get_days_of_quarter(object obj) except -1
cdef int get_quarter_1st_month(object obj) except -1
cdef int get_quarter_lst_month(object obj) except -1
cdef int get_days_in_month(object obj) except -1
cdef int get_days_bf_month(object obj) except -1
cdef int to_ordinal(object obj) except -1
cdef str date_to_isoformat(datetime.date date) noexcept
cdef long long date_to_seconds(datetime.date date) noexcept
cdef long long date_to_microseconds(datetime.date date) noexcept
cdef long long date_to_timestamp(datetime.date) except *
cdef datetime.date date_fr_dt(datetime.datetime dt) noexcept
cdef datetime.date date_fr_ordinal(int ordinal) noexcept
cdef datetime.date date_fr_seconds(double seconds) noexcept
cdef datetime.date date_fr_microseconds(long long microseconds) noexcept
cdef datetime.date date_fr_timestamp(double timestamp) except *
cdef datetime.date date_add(datetime.date date, int days=?, long long seconds=?, long long microseconds=?) noexcept
cdef datetime.date date_add_delta(datetime.date date, datetime.timedelta delta) noexcept
cdef int date_sub_date_days(datetime.date date_l, datetime.date date_r) noexcept
cdef datetime.timedelta date_sub_date(datetime.date date_l, datetime.date date_r) noexcept
cdef datetime.date date_replace(datetime.date date, int year=?, int month=?, int day=?) noexcept
# Datetime
cdef datetime.datetime gen_dt(int year=?, int month=?, int day=?, int hour=?, int minute=?, int second=?, int microsecond=?, object tzinfo=?, int fold=?) except *
cdef datetime.datetime gen_dt_local() noexcept
cdef long long dt_mktime(datetime.datetime dt) except *
cdef bint is_dt(object obj) except -1
cdef bint is_dt_exact(object obj) except -1
cdef int get_dt_hour(datetime.datetime dt) except -1
cdef int get_dt_minute(datetime.datetime dt) except -1
cdef int get_dt_second(datetime.datetime dt) except -1
cdef int get_dt_microsecond(datetime.datetime dt) except -1
cdef datetime.tzinfo get_dt_tzinfo(datetime.datetime dt) noexcept
cdef int get_dt_fold(datetime.datetime dt) except -1
cdef str dt_to_isoformat(datetime.datetime dt) noexcept
cdef str dt_to_isoformat_tz(datetime.datetime dt) noexcept
cdef double dt_to_seconds(datetime.datetime dt) noexcept
cdef double dt_to_seconds_utc(datetime.datetime dt) noexcept
cdef long long dt_to_microseconds(datetime.datetime dt) noexcept
cdef long long dt_to_microseconds_utc(datetime.datetime dt) noexcept
cdef double dt_to_timestamp(datetime.datetime dt) except *
cdef datetime.datetime dt_fr_dt(datetime.datetime dt) noexcept
cdef datetime.datetime dt_fr_date(datetime.date date) noexcept
cdef datetime.datetime dt_fr_time(datetime.time time) noexcept
cdef datetime.datetime dt_fr_date_time(datetime.date date, datetime.time time) noexcept
cdef datetime.datetime dt_fr_ordinal(int ordinal) noexcept
cdef datetime.datetime dt_fr_seconds(double seconds, object tzinfo=?, int fold=?) noexcept
cdef datetime.datetime dt_fr_microseconds(long long microseconds, object tzinfo=?, int fold=?) noexcept
cdef datetime.datetime dt_fr_timestamp(double timestamp, object tzinfo=?) except *
cdef datetime.datetime dt_add(datetime.datetime dt, int days=?, long long seconds=?, long long microseconds=?) noexcept
cdef datetime.datetime dt_add_delta(datetime.datetime dt, datetime.timedelta delta) noexcept
cdef datetime.timedelta dt_sub_dt(datetime.datetime dt_l, datetime.datetime dt_r) noexcept
cdef long long dt_sub_dt_microseconds(datetime.datetime dt_l, datetime.datetime dt_r) noexcept
cdef datetime.datetime dt_replace(datetime.datetime dt, int year=?, int month=?, int day=?, int hour=?, int minute=?, int second=?, int microsecond=?, object tzinfo=?, int fold=?) noexcept
cdef datetime.datetime dt_replace_tzinfo(datetime.datetime dt, object tzinfo=?) noexcept
cdef datetime.datetime dt_replace_fold(datetime.datetime dt, int fold=?) noexcept
# Datetime.Time
cdef datetime.time gen_time(int hour=?, int minute=?, int second=?, int microsecond=?, object tzinfo=?, int fold=?) except *
cdef datetime.time gen_time_local() noexcept
cdef bint is_time(object obj) except -1
cdef bint is_time_exact(object obj) except -1
cdef int get_time_hour(datetime.time time) except -1
cdef int get_time_minute(datetime.time time) except -1
cdef int get_time_second(datetime.time time) except -1
cdef int get_time_microsecond(datetime.time time) except -1
cdef datetime.tzinfo get_time_tzinfo(datetime.time time) noexcept
cdef int get_time_fold(datetime.time time) except -1
cdef str time_to_isoformat(datetime.time time) noexcept
cdef str time_to_isoformat_tz(datetime.time time) noexcept
cdef double time_to_seconds(datetime.time time) noexcept
cdef long long time_to_microseconds(datetime.time time) noexcept
cdef datetime.time time_fr_dt(datetime.datetime dt) noexcept
cdef datetime.time time_fr_dt_tz(datetime.datetime dt) noexcept
cdef datetime.time time_fr_seconds(double seconds, object tzinfo=?, int fold=?) noexcept
cdef datetime.time time_fr_microseconds(long long microseconds, object tzinfo=?, int fold=?) noexcept
cdef datetime.time time_replace(datetime.time time, int hour=?, int minute=?, int second=?, int microsecond=?, object tzinfo=?, int fold=?) noexcept
cdef datetime.time time_replace_tzinfo(datetime.time time, object tzinfo=?) noexcept
cdef datetime.time time_replace_fold(datetime.time time, int fold=?) noexcept
# Timedelta
cdef datetime.timedelta gen_delta(int days=?, long long seconds=?, long long microseconds=?) except *
cdef bint is_delta(object obj) except -1
cdef bint is_delta_exact(object obj) except -1
cdef int get_delta_days(datetime.timedelta delta) noexcept
cdef int get_delta_seconds(datetime.timedelta delta) noexcept
cdef int get_delta_microseconds(datetime.timedelta delta) noexcept
cdef str delta_to_isoformat(datetime.timedelta delta) noexcept
cdef double delta_to_seconds(datetime.timedelta delta) noexcept
cdef long long delta_to_microseconds(datetime.timedelta delta) noexcept
cdef datetime.timedelta delta_fr_seconds(double seconds) noexcept
cdef datetime.timedelta delta_fr_microseconds(long long microseconds) noexcept
cdef datetime.timedelta delta_add(datetime.timedelta delta, int days=?, long long seconds=?, long long microseconds=?) noexcept
cdef datetime.timedelta delta_add_delta(datetime.timedelta delta1, datetime.timedelta delta2) noexcept
cdef datetime.timedelta delta_sub_delta(datetime.timedelta delta_l, datetime.timedelta delta_r) noexcept
# Timezone
cdef datetime.tzinfo gen_timezone(int offset_seconds, str tzname=?) except *
cdef datetime.tzinfo gen_timezone_local(datetime.datetime dt=?) except *
cdef bint is_tzinfo(object obj) except -1
cdef bint is_tzinfo_exact(object obj) except -1
cdef str format_utcoffset(datetime.timedelta utcoffset) except *
# Datetime64
cdef bint is_dt64(object obj) except -1
cdef str dt64_to_isoformat(object dt64) except *
cdef long long dt64_to_int(object dt64, str unit) except *
cdef long long dt64_to_days(object dt64) except *
cdef long long dt64_to_hours(object dt64) except *
cdef long long dt64_to_minutes(object dt64) except *
cdef long long dt64_to_seconds(object dt64) except *
cdef long long dt64_to_miliseconds(object dt64) except *
cdef long long dt64_to_microseconds(object dt64) except *
cdef long long dt64_to_nanoseconds(object dt64) except *
cdef datetime.date dt64_to_date(object dt64) except *
cdef datetime.datetime dt64_to_dt(object dt64) except *
cdef datetime.time dt64_to_time(object dt64) except *
# Timedelta64
cdef bint is_delta64(object obj) except -1
cdef str delta64_to_isoformat(object delta64) except *
cdef long long delta64_to_int(object delta64, str unit) except *
cdef long long delta64_to_days(object delta64) except *
cdef long long delta64_to_hours(object delta64) except *
cdef long long delta64_to_minutes(object delta64) except *
cdef long long delta64_to_seconds(object delta64) except *
cdef long long delta64_to_miliseconds(object delta64) except *
cdef long long delta64_to_microseconds(object delta64) except *
cdef long long delta64_to_nanoseconds(object delta64) except *
cdef datetime.timedelta delta64_to_delta(object delta64) except *
# ndarray[dateimte64]
cdef np.ndarray arraydt64_to_arrayint(np.ndarray arr, str unit) except *
cdef np.ndarray arraydt64_to_arrayint_day(np.ndarray arr) except *
cdef np.ndarray arraydt64_to_arrayint_hour(np.ndarray arr) except *
cdef np.ndarray arraydt64_to_arrayint_min(np.ndarray arr) except *
cdef np.ndarray arraydt64_to_arrayint_sec(np.ndarray arr) except *
cdef np.ndarray arraydt64_to_arrayint_ms(np.ndarray arr) except *
cdef np.ndarray arraydt64_to_arrayint_us(np.ndarray arr) except *
cdef np.ndarray arraydt64_to_arrayint_ns(np.ndarray arr) except *
# ndarray[timedelta64]
cdef np.ndarray arraydelta64_to_arrayint(np.ndarray arr, str unit) except *
cdef np.ndarray arraydelta64_to_arrayint_day(np.ndarray arr) except *
cdef np.ndarray arraydelta64_to_arrayint_hour(np.ndarray arr) except *
cdef np.ndarray arraydelta64_to_arrayint_min(np.ndarray arr) except *
cdef np.ndarray arraydelta64_to_arrayint_sec(np.ndarray arr) except *
cdef np.ndarray arraydelta64_to_arrayint_ms(np.ndarray arr) except *
cdef np.ndarray arraydelta64_to_arrayint_us(np.ndarray arr) except *
cdef np.ndarray arraydelta64_to_arrayint_ns(np.ndarray arr) except *
# pandas.Series[datetime64]
cdef np.ndarray seriesdt64_to_arrayint(object series, str unit) except *
cdef np.ndarray seriesdt64_to_arrayint_day(object series) except *
cdef np.ndarray seriesdt64_to_arrayint_hour(object series) except *
cdef np.ndarray seriesdt64_to_arrayint_min(object series) except *
cdef np.ndarray seriesdt64_to_arrayint_sec(object series) except *
cdef np.ndarray seriesdt64_to_arrayint_ms(object series) except *
cdef np.ndarray seriesdt64_to_arrayint_us(object series) except *
cdef np.ndarray seriesdt64_to_arrayint_ns(object series) except *
cdef object seriesdt64_adjust_to_ns(object series) except *
cdef object seriesdt64_to_ordinal(object series) except *
cdef object seriesdt64_to_seconds(object series) except *
cdef object seriesdt64_to_microseconds(object series) except *
# pandas.Series[timedelta64]
cdef np.ndarray seriesdelta64_to_arrayint(object series, str unit) except *
cdef np.ndarray seriesdelta64_to_arrayint_day(object series) except *
cdef np.ndarray seriesdelta64_to_arrayint_hour(object series) except *
cdef np.ndarray seriesdelta64_to_arrayint_min(object series) except *
cdef np.ndarray seriesdelta64_to_arrayint_sec(object series) except *
cdef np.ndarray seriesdelta64_to_arrayint_ms(object series) except *
cdef np.ndarray seriesdelta64_to_arrayint_us(object series) except *
cdef np.ndarray seriesdelta64_to_arrayint_ns(object series) except *
cdef object seriesdelta64_adjust_to_ns(object series) except *


