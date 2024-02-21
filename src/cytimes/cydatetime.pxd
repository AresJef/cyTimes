# cython: language_level=3

cimport numpy as np
from cpython cimport datetime
from cytimes cimport cytime

# Constants
cdef:
    # . calendar
    unsigned int[13] DAYS_BR_MONTH
    unsigned int[13] DAYS_IN_MONTH
    unsigned int[5] DAYS_BR_QUARTER
    unsigned int[5] DAYS_IN_QUARTER
    datetime.tzinfo UTC
    # . datetime EPOCH
    datetime.datetime EPOCH_NAI, EPOCH_UTC
    long long EPOCH_US, EPOCH_SEC
    int EPOCH_DAY
    # . max & min datetime seconds
    long long DT_MIN_US, DT_MAX_US
    # . nanoseconds conversion
    long long NS_DAY, NS_HOUR, NS_MINUTE
    # . microseconds conversion
    long long US_DAY, US_HOUR
    # . numpy dtype
    object DT64ARRAY_DTYPE, DELTA64ARRAY_DTYPE, PDSERIES_DTYPE

# Struct
cdef struct ymd:
    unsigned int year
    unsigned int month
    unsigned int day

cdef struct hms:
    unsigned int hour
    unsigned int minute
    unsigned int second
    unsigned int microsecond

cdef struct iso:
    unsigned int year
    unsigned int week
    unsigned int weekday

# Calendar: year
cdef bint is_leapyear(unsigned int year) except -1
cdef unsigned int leap_bt_years(unsigned int year1, unsigned int year2) except -1
cdef unsigned int days_in_year(unsigned int year) except -1
cdef unsigned int days_bf_year(unsigned int year) except -1
cdef unsigned int days_of_year(unsigned int year, unsigned int month, unsigned int day) except -1
# Calendar: quarter
cdef unsigned int quarter_of_month(unsigned int month) except -1
cdef unsigned int days_in_quarter(unsigned int year, unsigned int month) except -1
cdef unsigned int days_bf_quarter(unsigned int year, unsigned int month) except -1
cdef unsigned int days_of_quarter(unsigned int year, unsigned int month, unsigned int day) except -1
cdef unsigned int quarter_1st_month(unsigned int month) except -1
cdef unsigned int quarter_lst_month(unsigned int month) except -1
# Calendar: month
cdef unsigned int days_in_month(unsigned int year, unsigned int month) except -1
cdef unsigned int days_bf_month(unsigned int year, unsigned int month) except -1
# Calendar: week
cdef unsigned int ymd_weekday(unsigned int year, unsigned int month, unsigned int day) except -1
cdef unsigned int ymd_isoweekday(unsigned int year, unsigned int month, unsigned int day) except -1
cdef unsigned int ymd_isoweek(unsigned int year, unsigned int month, unsigned int day) except -1
cdef unsigned int ymd_isoyear(unsigned int year, unsigned int month, unsigned int day) except -1
cdef iso ymd_isocalendar(unsigned int year, unsigned int month, unsigned int day) noexcept
cdef unsigned int iso1st_ordinal(unsigned int year) except -1
# Calendar: conversion
cdef unsigned int ymd_to_ordinal(unsigned int year, unsigned int month, unsigned int day) except -1
cdef ymd ordinal_to_ymd(int ordinal) noexcept
cdef hms microseconds_to_hms(long long microseconds) noexcept

# Time
cdef double time() noexcept
cdef cytime.tm localtime() except *
cdef cytime.tm localize_time(double timestamp) except *
cdef long long localize_ts(double timestamp) except *

# Datetime.date: generate
cdef datetime.date gen_date(unsigned int year=?, unsigned int month=?, unsigned int day=?) except *
cdef datetime.date gen_date_now() noexcept
# Datetime.date: check types
cdef bint is_date(object obj) except -1
cdef bint is_date_exact(object obj) except -1
# Datetime.date: attributes
cdef unsigned int access_year(datetime.date date) except -1
cdef unsigned int access_month(datetime.date date) except -1
cdef unsigned int access_day(datetime.date date) except -1
# Datetime.date: calendar - year
cdef bint date_is_leapyear(datetime.date date) except -1
cdef unsigned int date_leap_bt_years(datetime.date date1, datetime.date date2) except -1
cdef unsigned int date_days_in_year(datetime.date date) except -1
cdef unsigned int date_days_bf_year(datetime.date date) except -1
cdef unsigned int date_days_of_year(datetime.date date) except -1
# Datetime.date: calendar - quarter
cdef unsigned int date_quarter(datetime.date date) except -1
cdef unsigned int date_days_in_quarter(datetime.date date) except -1
cdef unsigned int date_days_bf_quarter(datetime.date date) except -1
cdef unsigned int date_days_of_quarter(datetime.date date) except -1
cdef unsigned int date_quarter_1st_month(datetime.date date) except -1
cdef unsigned int date_quarter_lst_month(datetime.date date) except -1
# Datetime.date: calendar - month
cdef unsigned int date_days_in_month(datetime.date date) except -1
cdef unsigned int date_days_bf_month(datetime.date date) except -1
# Datetime.date: calendar - week
cdef unsigned int date_weekday(datetime.date date) except -1
cdef unsigned int date_isoweekday(datetime.date date) except -1
cdef unsigned int date_isoweek(datetime.date date) except -1
cdef unsigned int date_isoyear(datetime.date date) except -1
cdef unsigned int date_iso1st_ordinal(datetime.date date) except -1
cdef iso date_isocalendar(datetime.date date) noexcept
# Datetime.date: conversion
cdef str date_to_isoformat(datetime.date date) noexcept
cdef unsigned int date_to_ordinal(datetime.date date) except -1
cdef long long date_to_seconds(datetime.date date) noexcept
cdef long long date_to_microseconds(datetime.date date) noexcept
cdef long long date_to_timestamp(datetime.date) except *
cdef datetime.date date_fr_date(datetime.date dt) noexcept
cdef datetime.date date_fr_ordinal(int ordinal) noexcept
cdef datetime.date date_fr_seconds(double seconds) noexcept
cdef datetime.date date_fr_microseconds(long long microseconds) noexcept
cdef datetime.date date_fr_timestamp(double timestamp) except *
# Datetime.date: arithmetic
cdef datetime.date date_add(datetime.date date, int days=?, long long seconds=?, long long microseconds=?) noexcept
cdef datetime.date date_add_delta(datetime.date date, datetime.timedelta delta) noexcept
cdef datetime.date date_sub_delta(datetime.date date, datetime.timedelta delta) noexcept
cdef datetime.timedelta date_sub_date(datetime.date date_l, datetime.date date_r) noexcept
cdef int date_sub_date_days(datetime.date date_l, datetime.date date_r) noexcept
# Datetime.date: manipulation
cdef datetime.date date_replace(datetime.date date, int year=?, int month=?, int day=?) noexcept
cdef datetime.date date_adj_weekday(datetime.date date, unsigned int weekday) noexcept

# Datetime.datetime: generate
cdef datetime.datetime gen_dt(
    unsigned int year=?, unsigned int month=?, unsigned int day=?, 
    unsigned int hour=?, unsigned int minute=?, unsigned int second=?, 
    unsigned int microsecond=?, datetime.tzinfo tzinfo=?, unsigned int fold=?) except *
cdef datetime.datetime gen_dt_now() noexcept
cdef datetime.datetime gen_dt_now_utc() noexcept
cdef datetime.datetime gen_dt_now_tz(datetime.tzinfo tzinfo) noexcept
# Datetime.datetime: check types
cdef bint is_dt(object obj) except -1
cdef bint is_dt_exact(object obj) except -1
# Datetime.datetime: attributes
cdef unsigned int access_dt_year(datetime.datetime dt) except -1
cdef unsigned int access_dt_month(datetime.datetime dt) except -1
cdef unsigned int access_dt_day(datetime.datetime dt) except -1
cdef unsigned int access_dt_hour(datetime.datetime dt) except -1
cdef unsigned int access_dt_minute(datetime.datetime dt) except -1
cdef unsigned int access_dt_second(datetime.datetime dt) except -1
cdef unsigned int access_dt_microsecond(datetime.datetime dt) except -1
cdef datetime.tzinfo access_dt_tzinfo(datetime.datetime dt) noexcept
cdef unsigned int access_dt_fold(datetime.datetime dt) except -1
# Datetime.datetime: conversion
cdef str dt_to_isoformat(datetime.datetime dt) noexcept
cdef str dt_to_isoformat_tz(datetime.datetime dt) noexcept
cdef double dt_to_seconds(datetime.datetime dt) noexcept
cdef double dt_to_seconds_utc(datetime.datetime dt) noexcept
cdef long long dt_to_microseconds(datetime.datetime dt) noexcept
cdef long long dt_to_microseconds_utc(datetime.datetime dt) noexcept
cdef long long dt_to_posixts(datetime.datetime dt) except *
cdef double dt_to_timestamp(datetime.datetime dt) except *
cdef datetime.datetime dt_fr_dt(datetime.datetime dt) noexcept
cdef datetime.datetime dt_fr_date(datetime.date date, datetime.tzinfo tzinfo=?) noexcept
cdef datetime.datetime dt_fr_time(datetime.time time) noexcept
cdef datetime.datetime dt_fr_date_n_time(datetime.date date, datetime.time time) noexcept
cdef datetime.datetime dt_fr_ordinal(int ordinal, datetime.tzinfo tzinfo=?) noexcept
cdef datetime.datetime dt_fr_seconds(double seconds, datetime.tzinfo tzinfo=?) noexcept
cdef datetime.datetime dt_fr_microseconds(long long microseconds, datetime.tzinfo tzinfo=?) noexcept
cdef datetime.datetime dt_fr_timestamp(double timestamp, datetime.tzinfo tzinfo=?) except *
# Datetime.datetime: arithmetic
cdef datetime.datetime dt_add(datetime.datetime dt, int days=?, long long seconds=?, long long microseconds=?) noexcept
cdef datetime.datetime dt_add_delta(datetime.datetime dt, datetime.timedelta delta) noexcept
cdef datetime.datetime dt_sub_delta(datetime.datetime dt, datetime.timedelta delta) noexcept
cdef datetime.timedelta dt_sub_dt(datetime.datetime dt_l, datetime.datetime dt_r) noexcept
cdef long long dt_sub_dt_us(datetime.datetime dt_l, datetime.datetime dt_r) noexcept
# Datetime.datetime: manipulation 
cdef datetime.datetime dt_replace(
    datetime.datetime dt, int year=?, int month=?, int day=?, int hour=?, 
    int minute=?, int second=?, int microsecond=?, object tzinfo=?, int fold=?) noexcept
cdef datetime.datetime dt_replace_tzinfo(datetime.datetime dt, datetime.tzinfo tzinfo) noexcept
cdef datetime.datetime dt_replace_fold(datetime.datetime dt, unsigned int fold) noexcept
cdef datetime.datetime dt_adj_weekday(datetime.datetime dt, unsigned int weekday) noexcept
cdef datetime.datetime dt_astimezone(datetime.datetime dt, datetime.tzinfo tzinfo=?) noexcept

# Datetime.time: generate
cdef datetime.time gen_time(
    unsigned int hour=?, unsigned int minute=?, unsigned int second=?, 
    unsigned int microsecond=?, datetime.tzinfo tzinfo=?, unsigned int fold=?) except *
cdef datetime.time gen_time_now() noexcept
cdef datetime.time gen_time_now_utc() noexcept
cdef datetime.time gen_time_now_tz(datetime.tzinfo tzinfo) noexcept
# Datetime.time: check types
cdef bint is_time(object obj) except -1
cdef bint is_time_exact(object obj) except -1
# Datetime.time: attributes
cdef unsigned int access_time_hour(datetime.time time) except -1
cdef unsigned int access_time_minute(datetime.time time) except -1
cdef unsigned int access_time_second(datetime.time time) except -1
cdef unsigned int access_time_microsecond(datetime.time time) except -1
cdef datetime.tzinfo access_time_tzinfo(datetime.time time) noexcept
cdef unsigned int access_time_fold(datetime.time time) except -1
# Datetime.time: conversion
cdef str time_to_isoformat(datetime.time time) noexcept
cdef double time_to_seconds(datetime.time time) noexcept
cdef long long time_to_microseconds(datetime.time time) noexcept
cdef datetime.time time_fr_dt(datetime.datetime dt) noexcept
cdef datetime.time time_fr_seconds(double seconds, datetime.tzinfo tzinfo=?) noexcept
cdef datetime.time time_fr_microseconds(long long microseconds, datetime.tzinfo tzinfo=?) noexcept
# Datetime.time: manipulation
cdef datetime.time time_replace(
    datetime.time time, int hour=?, int minute=?, int second=?, 
    int microsecond=?, object tzinfo=?, int fold=?) noexcept
cdef datetime.time time_replace_tzinfo(datetime.time time, datetime.tzinfo tzinfo) noexcept
cdef datetime.time time_replace_fold(datetime.time time, unsigned int fold) noexcept

# Datetime.timedelta: generate
cdef datetime.timedelta gen_delta(int days=?, int seconds=?, int microseconds=?) except *
# Datetime.timedelta: check types
cdef bint is_delta(object obj) except -1
cdef bint is_delta_exact(object obj) except -1
# Datetime.timedelta: attributes
cdef int access_delta_days(datetime.timedelta delta) noexcept
cdef int access_delta_seconds(datetime.timedelta delta) noexcept
cdef int access_delta_microseconds(datetime.timedelta delta) noexcept
# Datetime.timedelta: conversion
cdef str delta_to_isoformat(datetime.timedelta delta) noexcept
cdef str delta_to_utcformat(datetime.timedelta delta) noexcept
cdef double delta_to_seconds(datetime.timedelta delta) noexcept
cdef long long delta_to_microseconds(datetime.timedelta delta) noexcept
cdef datetime.timedelta delta_fr_delta(datetime.timedelta delta) noexcept
cdef datetime.timedelta delta_fr_seconds(double seconds) noexcept
cdef datetime.timedelta delta_fr_microseconds(long long microseconds) noexcept
# Datetime.timedelta: arithmetic
cdef datetime.timedelta delta_add(datetime.timedelta delta, int days=?, int seconds=?, int microseconds=?) noexcept
cdef datetime.timedelta delta_add_delta(datetime.timedelta delta_l, datetime.timedelta delta_r) noexcept
cdef datetime.timedelta delta_sub_delta(datetime.timedelta delta_l, datetime.timedelta delta_r) noexcept

# Datetime.tzinfo: generate
cdef datetime.tzinfo gen_tzinfo(int offset, str tzname=?) except *
cdef datetime.tzinfo gen_tzinfo_local(datetime.datetime dt=?) except *
# Datetime.tzinfo: check types
cdef bint is_tzinfo(object obj) except -1
cdef bint is_tzinfo_exact(object obj) except -1

# numpy.datetime64: check types
cdef bint is_dt64(object obj) except -1
cdef validate_dt64(object obj) except *
# numpy.datetime64: conversion
cdef str dt64_to_isoformat(object dt64) except *
cdef long long dt64_to_int(object dt64, object unit) except *
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

# numpy.timedelta64: check types
cdef bint is_delta64(object obj) except -1
cdef validate_delta64(object obj) except *
# numpy.timedelta64: conversion
cdef str delta64_to_isoformat(object delta64) except *
cdef long long delta64_to_int(object delta64, object unit) except *
cdef long long delta64_to_days(object delta64) except *
cdef long long delta64_to_hours(object delta64) except *
cdef long long delta64_to_minutes(object delta64) except * 
cdef long long delta64_to_seconds(object delta64) except *
cdef long long delta64_to_miliseconds(object delta64) except *
cdef long long delta64_to_microseconds(object delta64) except *
cdef long long delta64_to_nanoseconds(object delta64) except *
cdef datetime.timedelta delta64_to_delta(object delta64) except *

# numpy.ndarray[datetime64]: check types
cdef bint is_dt64array(np.ndarray arr) except -1
cdef validate_dt64array(np.ndarray arr) except *
# numpy.ndarray[datetime64]: conversion
cdef np.ndarray dt64array_to_int(np.ndarray arr, object unit) except *
cdef np.ndarray dt64array_to_days(np.ndarray arr) except *
cdef np.ndarray dt64array_to_hours(np.ndarray arr) except *
cdef np.ndarray dt64array_to_minutes(np.ndarray arr) except *
cdef np.ndarray dt64array_to_seconds(np.ndarray arr) except *
cdef np.ndarray dt64array_to_miliseconds(np.ndarray arr) except *
cdef np.ndarray dt64array_to_microseconds(np.ndarray arr) except *
cdef np.ndarray dt64array_to_nanoseconds(np.ndarray arr) except *

# numpy.ndarray[timedelta64]: check types
cdef bint is_delta64array(np.ndarray arr) except -1
cdef validate_delta64array(np.ndarray arr) except *
# numpy.ndarray[timedelta64]: conversion
cdef np.ndarray delta64array_to_int(np.ndarray arr, object unit) except *
cdef np.ndarray delta64array_to_days(np.ndarray arr) except *
cdef np.ndarray delta64array_to_hours(np.ndarray arr) except *
cdef np.ndarray delta64array_to_minutes(np.ndarray arr) except *
cdef np.ndarray delta64array_to_seconds(np.ndarray arr) except *
cdef np.ndarray delta64array_to_miliseconds(np.ndarray arr) except *
cdef np.ndarray delta64array_to_microseconds(np.ndarray arr) except *
cdef np.ndarray delta64array_to_nanoseconds(np.ndarray arr) except *

# pandas.Series: check types
cdef bint is_pdseries(object obj) except -1
cdef validate_pdseries(object obj) except *

# pandas.Series[datetime64]: conversion
cdef object dt64series_to_int(object s, object unit) except *
cdef object dt64series_to_days(object s) except *
cdef object dt64series_to_hours(object s) except *
cdef object dt64series_to_minutes(object s) except *
cdef object dt64series_to_seconds(object s) except *
cdef object dt64series_to_miliseconds(object s) except *
cdef object dt64series_to_microseconds(object s) except *
cdef object dt64series_to_nanoseconds(object s) except *
cdef object dt64series_to_ordinals(object s) except *
cdef object dt64series_to_timestamps(object s) except *
# pandas.Series[datetime64]: adjustment
cdef object dt64series_adj_to_ns(object s) except *

# pandas.Series[timedelta64]: conversion
cdef object delta64series_to_int(object s, object unit) except *
cdef object delta64series_to_days(object s) except *
cdef object delta64series_to_hours(object s) except *
cdef object delta64series_to_minutes(object s) except *
cdef object delta64series_to_seconds(object s) except *
cdef object delta64series_to_miliseconds(object s) except *
cdef object delta64series_to_microseconds(object s) except *
cdef object delta64series_to_nanoseconds(object s) except *
# pandas.Series[timedelta64]: adjustment
cdef object delta64series_adj_to_ns(object series) except *
