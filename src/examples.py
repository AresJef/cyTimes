import datetime, numpy as np
from cytimes import Pydt, Pddt

# Pydt ==============================================================
# Construct ---------------------------------------------------------
dt = Pydt(1970, 1, 1, tzinfo="UTC")  # 1970-01-01 00:00:00+0000
dt = Pydt.parse("1970 Jan 1 00:00:01 PM")  # 1970-01-01 12:00:01
dt = Pydt.now()  # 2024-12-06 10:37:25.619593
dt = Pydt.utcnow()  # 2024-12-06 09:37:36.743159+0000
dt = Pydt.combine("1970-01-01", "00:00:01")  # 1970-01-01 00:00:01
dt = Pydt.fromordinal(1)  # 0001-01-01 00:00:00
dt = Pydt.fromseconds(1)  # 1970-01-01 00:00:01
dt = Pydt.fromicroseconds(1)  # 1970-01-01 00:00:00.000001
dt = Pydt.fromtimestamp(1, datetime.UTC)  # 1970-01-01 00:00:01+0000
dt = Pydt.utcfromtimestamp(1)  # 1970-01-01 00:00:01+0000
dt = Pydt.fromisoformat("1970-01-01T00:00:01")  # 1970-01-01 00:00:01
dt = Pydt.fromisocalendar(1970, 1, 4)  # 1970-01-01 00:00:00
dt = Pydt.fromdate(datetime.date(1970, 1, 1))  # 1970-01-01 00:00:00
dt = Pydt.fromdatetime(datetime.datetime(1970, 1, 1))  # 1970-01-01 00:00:00
dt = Pydt.fromdatetime64(np.datetime64(1, "s"))  # 1970-01-01 00:00:01
dt = Pydt.strptime("00:00:01 1970-01-01", "%H:%M:%S %Y-%m-%d")  # 1970-01-01 00:00:01

# Convert -----------------------------------------------------------
dt = Pydt(1970, 1, 1, tzinfo="CET")  # 1970-01-01 00:00:00+0100
res = dt.ctime()  # "Thu Jan  1 00:00:00 1970"
res = dt.strftime("%Y-%m-%d %H:%M:%S %Z")  # "1970-01-01 00:00:00 CET"
res = dt.isoformat()  # "1970-01-01T00:00:00+01:00"
res = dt.timetuple()  # (1970, 1, 1, 0, 0, 0, 3, 1, 0)
res = dt.toordinal()  # 719163
res = dt.seconds()  # 0.0
res = dt.microseconds()  # 0
res = dt.timestamp()  # -3600.0
res = dt.date()  # 1970-01-01
res = dt.time()  # 00:00:00
res = dt.timetz()  # 00:00:00

# Manipulate --------------------------------------------------------
dt = Pydt(1970, 2, 2, 2, 2, 2, 2, "CET")  # 1970-02-02 02:02:02.000002+0100
# . replace
res = dt.replace(
    year=2007, microsecond=1, tzinfo="UTC"
)  # 2007-02-02 02:02:02.000001+0000
# . year
res = dt.to_curr_year(3, 15)  # 1970-03-15 02:02:02.000002+0100
res = dt.to_prev_year("Feb", 30)  # 1969-02-28 02:02:02.000002+0100
res = dt.to_next_year("十二月", 31)  # 1971-12-31 02:02:02.000002+0100
res = dt.to_year(100, "noviembre", 30)  # 2070-11-30 02:02:02.000002+0100
# . quarter
res = dt.to_curr_quarter(3, 15)  # 1970-03-15 02:02:02.000002+0100
res = dt.to_prev_quarter(3, 15)  # 1969-12-15 02:02:02.000002+0100
res = dt.to_next_quarter(3, 15)  # 1970-06-15 02:02:02.000002+0100
res = dt.to_quarter(100, 3, 15)  # 1995-03-15 02:02:02.000002+0100
# . month
res = dt.to_curr_month(15)  # 1970-02-15 02:02:02.000002+0100
res = dt.to_prev_month(15)  # 1970-01-15 02:02:02.000002+0100
res = dt.to_next_month(15)  # 1970-03-15 02:02:02.000002+0100
res = dt.to_month(100, 15)  # 1978-06-15 02:02:02.000002+0200
# . weekday
res = dt.to_monday()  # 1970-02-02 02:02:02.000002+0100
res = dt.to_sunday()  # 1970-02-08 02:02:02.000002+0100
res = dt.to_curr_weekday(4)  # 1970-02-06 02:02:02.000002+0100
res = dt.to_prev_weekday(4)  # 1970-01-30 02:02:02.000002+0100
res = dt.to_next_weekday(4)  # 1970-02-13 02:02:02.000002+0100
res = dt.to_weekday(100, 4)  # 1972-01-07 02:02:02.000002+0100
# . day
res = dt.to_yesterday()  # 1970-02-01 02:02:02.000002+0100
res = dt.to_tomorrow()  # 1970-02-03 02:02:02.000002+0100
res = dt.to_day(100)  # 1970-05-13 02:02:02.000002+0100
# . date&time
res = dt.to_first_of("Y")  # 1970-01-01 02:02:02.000002+0100
res = dt.to_last_of("Q")  # 1970-03-31 02:02:02.000002+0100
res = dt.to_start_of("M")  # 1970-02-01 00:00:00+0100
res = dt.to_end_of("W")  # 1970-02-08 23:59:59.999999+0100
# . round / ceil / floor
res = dt.round("h")  # 1970-02-02 02:00:00+0100
res = dt.ceil("m")  # 1970-02-02 02:03:00+0100
res = dt.floor("s")  # 1970-02-02 02:02:02+0100

# Calendar ----------------------------------------------------------
dt = Pydt(1970, 2, 2, tzinfo="UTC")  # 1970-01-01 00:00:00+0000

# . iso
res = dt.isocalendar()  # {'year': 1970, 'week': 6, 'weekday': 1}
res = dt.isoyear()  # 1970
res = dt.isoweek()  # 6
res = dt.isoweekday()  # 1

# . year
res = dt.is_leap_year()  # False
res = dt.is_long_year()  # True
res = dt.leap_bt_year(2007)  # 9
res = dt.days_in_year()  # 365
res = dt.days_bf_year()  # 719162
res = dt.days_of_year()  # 33
res = dt.is_year(1970)  # True

# . quarter
res = dt.days_in_quarter()  # 90
res = dt.days_bf_quarter()  # 0
res = dt.days_of_quarter()  # 33
res = dt.is_quarter(1)  # True

# . month
res = dt.days_in_month()  # 28
res = dt.days_bf_month()  # 31
res = dt.days_of_month()  # 2
res = dt.is_month("Feb")  # True
res = dt.month_name("es")  # "febrero"

# . weekday
res = dt.is_weekday("Monday")  # True

# . day
res = dt.is_day(2)  # True
res = dt.day_name("fr")  # "lundi"

# . date&time
res = dt.is_first_of("Y")  # False
res = dt.is_last_of("Q")  # False
res = dt.is_start_of("M")  # False
res = dt.is_end_of("W")  # False

# Timezone ----------------------------------------------------------
dt = Pydt(1970, 1, 1, tzinfo="UTC")  # 1970-01-01 00:00:00+0000

res = dt.is_local()  # False
res = dt.is_utc()  # True
res = dt.is_dst()  # False
res = dt.tzname()  # "UTC"
res = dt.utcoffset()  # 0:00:00
res = dt.utcoffset_seconds()  # 0
res = dt.dst()  # None
res = dt.astimezone("CET")  # 1970-01-01 01:00:00+0100
res = dt.tz_localize(None)  # 1970-01-01 00:00:00
res = dt.tz_convert("CET")  # 1970-01-01 01:00:00+0100
res = dt.tz_switch("CET")  # 1970-01-01 01:00:00+0100

# Arithmetic --------------------------------------------------------
dt = Pydt(1970, 1, 1, tzinfo="UTC")  # 1970-01-01 00:00:00+0000

res = dt.add(years=1, weeks=1, microseconds=1)  # 1971-01-08 00:00:00.000001+0000
res = dt.sub(quarters=1, days=1, seconds=1)  # 1969-09-29 23:59:59+0000
res = dt.diff("2007-01-01 01:01:01+01:00", "s")  # -1167609662

# Comparison --------------------------------------------------------
dt = Pydt(1970, 1, 1)  # 1970-01-01 00:00:00

res = dt.is_past()  # True
res = dt.is_future()  # False
res = dt.closest("1970-01-02", "2007-01-01")  # 1970-01-02 00:00:00
res = dt.farthest("1970-01-02", "2007-01-01")  # 2007-01-01 00:00:00

# Pddt ==============================================================
pt = Pddt(["9999-01-01 00:00:00+00:00", "9999-01-02 00:00:00+00:00"])

pt = Pddt(["1970-01-01 00:00:00+00:00", "1970-01-02 00:00:00+00:00"])
pt = pt.to_year(1000, "Feb", 30)
