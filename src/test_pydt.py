import warnings
from zoneinfo import ZoneInfo
import time, unittest, datetime
import numpy as np, pandas as pd, pendulum as pl
from cytimes import errors
from cytimes.pydt import Pydt
from cytimes.delta import Delta

warnings.filterwarnings("ignore")


class TestCase(unittest.TestCase):
    name: str = "Case"

    def test_all(self) -> None:
        pass

    # Utils
    def assertEqualDtsMs(self, dt1: datetime.datetime, dt2: datetime.datetime) -> None:
        self.assertTrue((dt1 - dt2).total_seconds() < 0.1)

    def log_start(self, msg: str) -> None:
        msg = "START TEST '%s': %s" % (self.name, msg)
        print(msg.ljust(60), end="\r")
        self._start_time = time.perf_counter()

    def log_ended(self, msg: str, skip: bool = False) -> None:
        self._ended_time = time.perf_counter()
        msg = "%s TEST '%s': %s" % ("SKIP" if skip else "PASS", self.name, msg)
        if self._start_time is not None:
            msg += " (%.6fs)" % (self._ended_time - self._start_time)
        print(msg.ljust(60))


class TestPydt(TestCase):
    name = "Pydt"

    def test_all(self) -> None:
        self.test_parse()
        self.test_constructor()
        self.test_converter()
        self.test_manipulator()
        self.test_calendar()
        self.test_timezone()
        self.test_arithmetic()
        self.test_comparison()
        self.test_subclass()

    def test_parse(self) -> None:
        test = "Parse"
        self.log_start(test)

        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = datetime.datetime.now(tz)
            # Parse datetime & subclasses
            self.assertEqual(dt, Pydt.parse(dt))
            self.assertEqual(dt, Pydt.parse(pd.Timestamp(dt)))
            self.assertEqual(dt, Pydt.parse(Pydt.parse(dt)))
            # Parse date & subclasses
            date = dt.date()
            self.assertEqual(date, Pydt.parse(date).date())
            self.assertEqual(
                date, Pydt.parse(pl.date(date.year, date.month, date.day)).date()
            )
            self.assertEqual(date, Pydt.parse(Pydt.parse(date)).date())
            # Parse datetime string
            self.assertEqual(dt, Pydt.parse(str(dt), ignoretz=False))

        # Test parser error
        with self.assertRaises(errors.InvalidArgumentError):
            Pydt.parse("2", default="XXX")
        with self.assertRaises(errors.InvalidArgumentError):
            Pydt.parse("xxx")
        with self.assertRaises(errors.InvalidArgumentError):
            Pydt.parse("2", default=None)

        # Test default
        for default in (
            datetime.datetime(1970, 1, 1),
            datetime.datetime(1970, 1, 1, 3, 4, 5, 6),
            "1970-01-01" "19700101 030405.000006",
        ):
            self.assertEqual(
                datetime.datetime(1970, 1, 2), Pydt.parse("2", default=default)
            )

        # Parse integer & float
        for i in (-100, -10 - 1.1, -1, 0, 0.0, 1, 1.1, 10, 100):
            self.assertEqual(pd.Timestamp(i, unit="s"), Pydt.parse(i))

        # Parse np.datetime64
        for unit, dt_base in (
            ("Y", datetime.datetime(1971, 1, 1)),
            ("M", datetime.datetime(1970, 2, 1)),
            ("D", datetime.datetime(1970, 1, 2)),
            ("h", datetime.datetime(1970, 1, 1, 1)),
            ("m", datetime.datetime(1970, 1, 1, 0, 1)),
            ("s", datetime.datetime(1970, 1, 1, 0, 0, 1)),
            ("ms", datetime.datetime(1970, 1, 1, 0, 0, 0, 1000)),
            ("us", datetime.datetime(1970, 1, 1, 0, 0, 0, 1)),
            ("ns", datetime.datetime(1970, 1, 1)),
        ):
            self.assertEqual(dt_base, Pydt.parse(np.datetime64(1, unit)))

        self.log_ended(test)

    def test_constructor(self) -> None:
        test = "Constructor"
        self.log_start(test)

        # now()
        with self.assertRaises(errors.InvalidTimezoneError):
            Pydt.now("XXX")
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            self.assertEqualDtsMs(datetime.datetime.now(tz), Pydt.now(tz))

        # utcnow()
        try:
            base = datetime.datetime.utcnow().replace(tzinfo=datetime.UTC)
        except Exception:
            pass  # classmethod offically removed
        else:
            self.assertEqualDtsMs(base, Pydt.utcnow())

        # today()
        self.assertEqualDtsMs(datetime.datetime.today(), Pydt.today())

        # combine()
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = datetime.datetime(1970, 1, 2, 3, 4, 5, 6, tz)
            pt = Pydt.combine(datetime.date(1970, 1, 2), datetime.time(3, 4, 5, 6, tz))
            self.assertEqual(dt, pt)
            pt = Pydt.combine(datetime.date(1970, 1, 2), datetime.time(3, 4, 5, 6), tz)
            self.assertEqual(dt, pt)
            pt = Pydt.combine("1970-01-02", datetime.time(3, 4, 5, 6, tz))
            self.assertEqual(dt, pt)
            pt = Pydt.combine("1970-01-02", datetime.time(3, 4, 5, 6), tz)
            self.assertEqual(dt, pt)
            pt = Pydt.combine(datetime.date(1970, 1, 2), "03:04:05.000006", tz)
            self.assertEqual(dt, pt)
            pt = Pydt.combine("1970-01-02", "03:04:05.000006", tz)
            self.assertEqual(dt, pt)

        # fromordinal()
        for ordinal in (2, 100, 400, 800000):
            for tz in (None, datetime.UTC, ZoneInfo("CET")):
                self.assertEqual(
                    datetime.datetime.fromordinal(ordinal).replace(tzinfo=tz),
                    Pydt.fromordinal(ordinal, tz),
                )

        # fromseconds()
        for seconds in (-78914791.123, -1, 0, 1, 78914791.123):
            self.assertEqual(
                datetime.datetime.fromtimestamp(seconds)
                .astimezone(None)
                .astimezone(datetime.UTC)
                .replace(tzinfo=None),
                Pydt.fromseconds(seconds),
            )

        # fromicroseconds()
        for seconds in (-78914791.123, -1, 0, 1, 78914791.123):
            us = int(seconds * 1_000_000)
            self.assertEqual(Pydt.fromseconds(seconds), Pydt.frommicroseconds(us))
            self.assertEqual(
                Pydt.fromseconds(seconds, datetime.UTC),
                Pydt.frommicroseconds(us, "UTC"),
            )

        # fromtimestamp()
        for ts in (-78914791.123, -1, 0, 1, 78914791.123):
            for tz in (None, datetime.UTC, ZoneInfo("CET")):
                self.assertEqual(
                    datetime.datetime.fromtimestamp(ts, tz), Pydt.fromtimestamp(ts, tz)
                )

        # utcfromtimestamp
        for ts in (-78914791.123, -1, 0, 1, 78914791.123):
            try:
                base = datetime.datetime.utcfromtimestamp(ts)
                base = base.replace(tzinfo=datetime.UTC)
            except Exception:
                break  # classmethod offically removed
            else:
                self.assertEqual(base, Pydt.utcfromtimestamp(ts))

        # fromisoformat()
        for iso in ("1970-01-02T03:04:05.000006", "1970-01-02T03:04:05+01:00"):
            self.assertEqual(
                datetime.datetime.fromisoformat(iso), Pydt.fromisoformat(iso)
            )

        # fromisocalendar()
        self.assertEqual(
            datetime.datetime.fromisocalendar(1970, 1, 2),
            Pydt.fromisocalendar(1970, 1, 2),
        )

        # from day-of-year
        for doy in (1, 32, 60, 365):
            self.assertEqual(
                datetime.datetime(1970, 1, 1) + datetime.timedelta(days=doy - 1),
                Pydt.fromdayofyear(1970, doy),
            )
        self.assertEqual(datetime.datetime(1970, 12, 31), Pydt.fromdayofyear(1970, 366))
        self.assertEqual(datetime.datetime(1970, 12, 31), Pydt.fromdayofyear(1970, 367))

        # fromdate()
        for date in (datetime.date(1970, 1, 2), pl.date(1970, 1, 2)):
            self.assertEqual(date, Pydt.fromdate(date).date())

        # fromdatetime()
        for dt in (
            datetime.datetime(1970, 1, 2, 3, 4, 5, 6),
            pd.Timestamp("1970-01-02 03:04:05.000006"),
            pl.datetime(1970, 1, 2, 3, 4, 5, 6),
        ):
            for tz in (None, datetime.UTC, ZoneInfo("CET")):
                dt = dt.replace(tzinfo=tz)
                self.assertEqual(dt, Pydt.fromdatetime(dt))

        # fromdatetime64()
        for unit, dt_base in (
            ("Y", datetime.datetime(1971, 1, 1)),
            ("M", datetime.datetime(1970, 2, 1)),
            ("D", datetime.datetime(1970, 1, 2)),
            ("h", datetime.datetime(1970, 1, 1, 1)),
            ("m", datetime.datetime(1970, 1, 1, 0, 1)),
            ("s", datetime.datetime(1970, 1, 1, 0, 0, 1)),
            ("ms", datetime.datetime(1970, 1, 1, 0, 0, 0, 1000)),
            ("us", datetime.datetime(1970, 1, 1, 0, 0, 0, 1)),
            ("ns", datetime.datetime(1970, 1, 1)),
        ):
            dt64 = np.datetime64(1, unit)
            self.assertEqual(dt_base, Pydt.fromdatetime64(dt64))

        # strptime()
        dt_str = "03:04:05.000006 1970-01-02 +01:00"
        fmt = "%H:%M:%S.%f %Y-%m-%d %z"
        self.assertEqual(
            datetime.datetime.strptime(dt_str, fmt),
            Pydt.strptime(dt_str, fmt),
        )

        self.log_ended(test)

    def test_converter(self) -> None:
        test = "Converter"
        self.log_start(test)

        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = datetime.datetime(2023, 1, 31, 3, 4, 5, 6, tz)
            pt = Pydt.parse(dt)
            # ctime()
            self.assertEqual(dt.ctime(), pt.ctime())
            # strftime()
            for fmt in (
                "%d/%m/%Y, %H:%M:%S",
                "%d/%m/%Y, %H:%M:%S%z",
                "%d/%m/%Y, %H:%M:%S%Z",
            ):
                self.assertEqual(dt.strftime(fmt), pt.strftime(fmt))
            # isoformat()
            for sep in (" ", "T", "x"):
                dt_str = dt.isoformat(sep)
                if tz is not None:
                    dt_str = dt_str[:-3] + dt_str[-2:]
                self.assertEqual(dt_str, pt.isoformat(sep))
            # timetuple()
            self.assertEqual(dt.timetuple(), pt.timetuple())
            # utctimetuple()
            self.assertEqual(dt.utctimetuple(), pt.utctimetuple())
            # toordinal()
            self.assertEqual(dt.toordinal(), pt.toordinal())
            # seconds()
            self.assertEqual(dt, Pydt.fromseconds(pt.toseconds(), tz))
            # microseconds()
            self.assertEqual(dt, Pydt.frommicroseconds(pt.tomicroseconds(), tz))
            # timestamp()
            self.assertEqual(dt.timestamp(), pt.timestamp())
            # date()
            self.assertEqual(dt.date(), pt.date())
            # time()
            self.assertEqual(dt.time(), pt.time())
            # timetz()
            self.assertEqual(dt.timetz(), pt.timetz())

        self.log_ended(test)

    def test_manipulator(self) -> None:
        test = "Manipulator"
        self.log_start(test)

        # replace()
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = datetime.datetime(2023, 5, 15, 3, 4, 5, 6, tz)
            pt = Pydt.fromdatetime(dt)
            for targ_tz in (None, datetime.UTC, ZoneInfo("CET")):
                for args in (
                    # dates only
                    [2000, -1, -1, -1, -1, -1, -1, -1, -1],
                    [2000, 3, -1, -1, -1, -1, -1, -1, -1],
                    [2000, 3, 31, -1, -1, -1, -1, -1, -1],
                    [-1, 3, -1, -1, -1, -1, -1, -1, -1],
                    [-1, 3, 31, -1, -1, -1, -1, -1, -1],
                    [-1, -1, 31, -1, -1, -1, -1, -1, -1],
                    # times only
                    [-1, -1, -1, 23, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, 59, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, 59, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, 999999, -1, -1],
                    [-1, -1, -1, 23, 59, -1, -1, -1, -1],
                    [-1, -1, -1, 23, 59, 59, -1, -1, -1],
                    [-1, -1, -1, 23, 59, 59, 999999, -1, -1],
                    # dates & times
                    [2000, -1, -1, 23, -1, -1, -1, -1, -1],
                    [2000, 3, -1, 23, -1, -1, -1, -1, -1],
                    [2000, 3, 31, 23, -1, -1, -1, -1, -1],
                    [-1, 3, -1, 23, -1, -1, -1, -1, -1],
                    [-1, 3, 31, 23, -1, -1, -1, -1, -1],
                    [-1, -1, 31, 23, -1, -1, -1, -1, -1],
                    [2000, -1, -1, 23, -1, -1, -1, -1, -1],
                    [2000, -1, -1, -1, 59, -1, -1, -1, -1],
                    [2000, -1, -1, -1, -1, 59, -1, -1, -1],
                    [2000, -1, -1, -1, -1, -1, 999999, -1, -1],
                    [2000, -1, -1, 23, 59, -1, -1, -1, -1],
                    [2000, -1, -1, 23, 59, 59, -1, -1, -1],
                    [2000, -1, -1, 23, 59, 59, 999999, -1, -1],
                    [2000, 3, 31, 23, 59, 59, 999999, -1, -1],
                    # Timezone
                    [2000, 3, 31, 23, 59, 59, 999999, targ_tz, -1],
                    # Fold
                    [2000, 3, 31, 23, 59, 59, 999999, targ_tz, 0],
                    [2000, 3, 31, 23, 59, 59, 999999, targ_tz, 1],
                ):
                    kwargs_pt = {
                        "year": args[0],
                        "month": args[1],
                        "day": args[2],
                        "hour": args[3],
                        "minute": args[4],
                        "second": args[5],
                        "microsecond": args[6],
                        "tzinfo": args[7],
                        "fold": args[8],
                    }
                    kwargs_dt = {k: v for k, v in kwargs_pt.items() if v != -1}
                    dt_repl = dt.replace(**kwargs_dt)
                    pt_repl = pt.replace(**kwargs_pt)
                    self.assertEqual(dt_repl, pt_repl)
                    self.assertEqual(dt_repl.fold, pt_repl.fold)

        # to_*()
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = Pydt(2023, 5, 15, 3, 4, 5, 6, tz)
            for offset in (-1, 0, 1):
                # to_curr_year() / to_year()
                for month in range(1, 13):
                    for day in range(1, 32):
                        self.assertEqual(
                            dt.to_curr_year(month, day),
                            dt.replace(month=month, day=day),
                        )
                        self.assertEqual(
                            dt.to_year(offset, month, day),
                            dt.replace(year=dt.year + offset, month=month, day=day),
                        )
                # to_curr_quarter() / to_quarter()
                for month in range(1, 4):
                    for day in range(1, 32):
                        new_month = dt.quarter * 3 + month - 3
                        self.assertEqual(
                            dt.to_curr_quarter(month, day),
                            dt.replace(month=new_month, day=day),
                        )
                        new_month = (offset + dt.quarter) * 3 + month - 3
                        self.assertEqual(
                            dt.to_quarter(offset, month, day),
                            dt.replace(month=new_month, day=day),
                        )
                # to_curr_month() / to_month()
                for day in range(1, 32):
                    self.assertEqual(
                        dt.to_curr_month(day),
                        dt.replace(day=day),
                    )
                    self.assertEqual(
                        dt.to_month(offset, day),
                        dt.replace(month=dt.month + offset, day=day),
                    )
                # to_weekday()
                for weekday in range(6):
                    self.assertEqual(
                        dt.to_weekday(offset, weekday),
                        dt.replace(day=offset * 7 + dt.day + weekday - dt.weekday),
                    )
                # to_day()
                for day in (-1, 0, 1):
                    self.assertEqual(
                        dt.to_day(day),
                        dt.replace(day=dt.day + day),
                    )

            # Test offset special cases
            for offset in range(-1000, 1001):
                # to_year()
                self.assertEqual(dt.to_year(offset), dt + Delta(years=offset))
                # to_quarter()
                self.assertEqual(dt.to_quarter(offset), dt + Delta(quarters=offset))
                # to_month()
                self.assertEqual(dt.to_month(offset), dt + Delta(months=offset))
                # to_weekday()
                self.assertEqual(dt.to_weekday(offset), dt + Delta(weeks=offset))
                # to_day()
            self.assertEqual(dt.to_day(offset), dt + Delta(days=offset))

        # to_datetime()
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = datetime.datetime(2023, 5, 15, 3, 4, 5, 6, tz)
            pt = Pydt.fromdatetime(dt)
            for args in (
                # dates only
                [2000, -1, -1, -1, -1, -1, -1, -1, -1],
                [2000, 3, -1, -1, -1, -1, -1, -1, -1],
                [2000, 3, 31, -1, -1, -1, -1, -1, -1],
                [-1, 3, -1, -1, -1, -1, -1, -1, -1],
                [-1, 3, 31, -1, -1, -1, -1, -1, -1],
                [-1, -1, 31, -1, -1, -1, -1, -1, -1],
                # times only
                [-1, -1, -1, 23, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, 59, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, 59, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, 999999, -1, -1],
                [-1, -1, -1, 23, 59, -1, -1, -1, -1],
                [-1, -1, -1, 23, 59, 59, -1, -1, -1],
                [-1, -1, -1, 23, 59, 59, 999999, -1, -1],
                # dates & times
                [2000, -1, -1, 23, -1, -1, -1, -1, -1],
                [2000, 3, -1, 23, -1, -1, -1, -1, -1],
                [2000, 3, 31, 23, -1, -1, -1, -1, -1],
                [-1, 3, -1, 23, -1, -1, -1, -1, -1],
                [-1, 3, 31, 23, -1, -1, -1, -1, -1],
                [-1, -1, 31, 23, -1, -1, -1, -1, -1],
                [2000, -1, -1, 23, -1, -1, -1, -1, -1],
                [2000, -1, -1, -1, 59, -1, -1, -1, -1],
                [2000, -1, -1, -1, -1, 59, -1, -1, -1],
                [2000, -1, -1, -1, -1, -1, 999999, -1, -1],
                [2000, -1, -1, 23, 59, -1, -1, -1, -1],
                [2000, -1, -1, 23, 59, 59, -1, -1, -1],
                [2000, -1, -1, 23, 59, 59, 999999, -1, -1],
                [2000, 3, 31, 23, 59, 59, 999999, -1, -1],
            ):
                kwargs_pt = {
                    "year": args[0],
                    "month": args[1],
                    "day": args[2],
                    "hour": args[3],
                    "minute": args[4],
                    "second": args[5],
                    "microsecond": args[6],
                }
                kwargs_dt = {k: v for k, v in kwargs_pt.items() if v != -1}
                if "tz" in kwargs_dt:
                    kwargs_dt["tzinfo"] = kwargs_dt.pop("tz")

                dt_repl = dt.replace(**kwargs_dt)
                pt_repl = pt.to_datetime(**kwargs_pt)
                self.assertEqual(dt_repl, pt_repl)

        # to_date()
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = datetime.datetime(2023, 5, 15, 3, 4, 5, 6, tz)
            pt = Pydt.fromdatetime(dt)
            for args in (
                [2000, -1, -1],
                [2000, 3, -1],
                [2000, 3, 31],
                [-1, 3, -1],
                [-1, 3, 31],
                [-1, -1, 31],
            ):
                kwargs_pt = {"year": args[0], "month": args[1], "day": args[2]}
                kwargs_dt = {k: v for k, v in kwargs_pt.items() if v != -1}
                if "tz" in kwargs_dt:
                    kwargs_dt["tzinfo"] = kwargs_dt.pop("tz")

                dt_repl = dt.replace(**kwargs_dt)
                pt_repl = pt.to_date(**kwargs_pt)
                self.assertEqual(dt_repl, pt_repl)

        # to_time()
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = datetime.datetime(2023, 5, 15, 3, 4, 5, 6, tz)
            pt = Pydt.fromdatetime(dt)
            for args in (
                [23, -1, -1, -1],
                [-1, 59, -1, -1],
                [-1, -1, 59, -1],
                [-1, -1, -1, 999999],
                [23, 59, -1, -1],
                [23, 59, 59, -1],
                [23, 59, 59, 999999],
            ):
                kwargs_pt = {
                    "hour": args[0],
                    "minute": args[1],
                    "second": args[2],
                    "microsecond": args[3],
                }
                kwargs_dt = {k: v for k, v in kwargs_pt.items() if v != -1}
                if "tz" in kwargs_dt:
                    kwargs_dt["tzinfo"] = kwargs_dt.pop("tz")

                dt_repl = dt.replace(**kwargs_dt)
                pt_repl = pt.to_time(**kwargs_pt)
                self.assertEqual(dt_repl, pt_repl)

        # to_first_of / to_last_of
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = datetime.datetime(2023, 5, 16, 3, 4, 5, 6, tz)
            time = dt.timetz()
            pt = Pydt.fromdatetime(dt)
            # to first of
            self.assertEqual(
                datetime.datetime.combine(datetime.date(2023, 1, 1), time),
                pt.to_first_of("Y"),
            )
            self.assertEqual(
                datetime.datetime.combine(datetime.date(2023, 4, 1), time),
                pt.to_first_of("Q"),
            )
            self.assertEqual(
                datetime.datetime.combine(datetime.date(2023, 5, 1), time),
                pt.to_first_of("M"),
            )
            self.assertEqual(
                datetime.datetime.combine(datetime.date(2023, 5, 15), time),
                pt.to_first_of("W"),
            )
            self.assertEqual(
                datetime.datetime.combine(datetime.date(2023, 2, 1), time),
                pt.to_first_of("Feb"),
            )
            # to last of
            self.assertEqual(
                datetime.datetime.combine(datetime.date(2023, 12, 31), time),
                pt.to_last_of("Y"),
            )
            self.assertEqual(
                datetime.datetime.combine(datetime.date(2023, 6, 30), time),
                pt.to_last_of("Q"),
            )
            self.assertEqual(
                datetime.datetime.combine(datetime.date(2023, 5, 31), time),
                pt.to_last_of("M"),
            )
            self.assertEqual(
                datetime.datetime.combine(datetime.date(2023, 5, 21), time),
                pt.to_last_of("W"),
            )
            self.assertEqual(
                datetime.datetime.combine(datetime.date(2023, 2, 28), time),
                pt.to_last_of("Feb"),
            )

        # to_start_of / to_end_of
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = datetime.datetime(2023, 5, 16, 3, 4, 5, 666666, tz)
            pt = Pydt.fromdatetime(dt)
            # to start of
            self.assertEqual(
                datetime.datetime(2023, 1, 1, 0, 0, 0, 0, tz),
                pt.to_start_of("Y"),
            )
            self.assertEqual(
                datetime.datetime(2023, 4, 1, 0, 0, 0, 0, tz),
                pt.to_start_of("Q"),
            )
            self.assertEqual(
                datetime.datetime(2023, 5, 1, 0, 0, 0, 0, tz),
                pt.to_start_of("M"),
            )
            self.assertEqual(
                datetime.datetime(2023, 5, 15, 0, 0, 0, 0, tz),
                pt.to_start_of("W"),
            )
            self.assertEqual(
                datetime.datetime(2023, 5, 16, 0, 0, 0, 0, tz),
                pt.to_start_of("D"),
            )
            self.assertEqual(
                datetime.datetime(2023, 5, 16, 3, 0, 0, 0, tz),
                pt.to_start_of("h"),
            )
            self.assertEqual(
                datetime.datetime(2023, 5, 16, 3, 4, 0, 0, tz),
                pt.to_start_of("m"),
            )
            self.assertEqual(
                datetime.datetime(2023, 5, 16, 3, 4, 5, 0, tz),
                pt.to_start_of("s"),
            )
            self.assertEqual(
                datetime.datetime(2023, 5, 16, 3, 4, 5, 666000, tz),
                pt.to_start_of("ms"),
            )
            self.assertEqual(dt, pt.to_start_of("us"))
            self.assertEqual(
                datetime.datetime(2023, 2, 1, 0, 0, 0, 0, tz),
                pt.to_start_of("Feb"),
            )
            self.assertEqual(
                datetime.datetime(2023, 5, 15, 0, 0, 0, 0, tz),
                pt.to_start_of("Mon"),
            )
            # to end of
            self.assertEqual(
                datetime.datetime(2023, 12, 31, 23, 59, 59, 999999, tz),
                pt.to_end_of("Y"),
            )
            self.assertEqual(
                datetime.datetime(2023, 6, 30, 23, 59, 59, 999999, tz),
                pt.to_end_of("Q"),
            )
            self.assertEqual(
                datetime.datetime(2023, 5, 31, 23, 59, 59, 999999, tz),
                pt.to_end_of("M"),
            )
            self.assertEqual(
                datetime.datetime(2023, 5, 21, 23, 59, 59, 999999, tz),
                pt.to_end_of("W"),
            )
            self.assertEqual(
                datetime.datetime(2023, 5, 16, 23, 59, 59, 999999, tz),
                pt.to_end_of("D"),
            )
            self.assertEqual(
                datetime.datetime(2023, 5, 16, 3, 59, 59, 999999, tz),
                pt.to_end_of("h"),
            )
            self.assertEqual(
                datetime.datetime(2023, 5, 16, 3, 4, 59, 999999, tz),
                pt.to_end_of("m"),
            )
            self.assertEqual(
                datetime.datetime(2023, 5, 16, 3, 4, 5, 999999, tz),
                pt.to_end_of("s"),
            )
            self.assertEqual(
                datetime.datetime(2023, 5, 16, 3, 4, 5, 666999, tz),
                pt.to_end_of("ms"),
            )
            self.assertEqual(dt, pt.to_end_of("us"))
            self.assertEqual(
                datetime.datetime(2023, 2, 28, 23, 59, 59, 999999, tz),
                pt.to_end_of("Feb"),
            )
            self.assertEqual(
                datetime.datetime(2023, 5, 21, 23, 59, 59, 999999, tz),
                pt.to_end_of("Sun"),
            )

        # is_first_of / is_last_of
        units = ["W", "M", "Q", "Y", "Jan", "Feb", "Dec"]
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = datetime.datetime(2023, 5, 16, 3, 4, 5, 666666, tz)
            pt = Pydt.fromdatetime(dt)
            for unit in ["W", "M", "Q", "Y", "Jan", "Feb", "Dec"]:
                self.assertTrue(pt.to_first_of(unit).is_first_of(unit))
                self.assertTrue(pt.to_last_of(unit).is_last_of(unit))

        # is_start_of / is_end_of
        units += ["us", "ms", "s", "m", "h", "D", "Mon", "Sun"]
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = datetime.datetime(2023, 5, 16, 3, 4, 5, 666666, tz)
            pt = Pydt.fromdatetime(dt)
            for unit in units:
                self.assertTrue(pt.to_start_of(unit).is_start_of(unit))
                self.assertTrue(pt.to_end_of(unit).is_end_of(unit))

        # round() / ceil() / floor()
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            for dt in (
                datetime.datetime(2023, 5, 16, 1, 4, 5, 666666, tz),
                datetime.datetime(2023, 8, 1, 12, 4, 5, 666666, tz),
                datetime.datetime(2023, 5, 16, 23, 4, 5, 666666, tz),
                datetime.datetime(2023, 8, 1, 3, 59, 5, 666666, tz),
                datetime.datetime(2023, 5, 16, 3, 5, 59, 666666, tz),
                datetime.datetime(2023, 8, 1, 3, 5, 59, 444444, tz),
                datetime.datetime(1800, 5, 16, 1, 4, 5, 666666, tz),
                datetime.datetime(1800, 8, 1, 12, 4, 5, 666666, tz),
                datetime.datetime(1800, 5, 16, 23, 4, 5, 666666, tz),
                datetime.datetime(1800, 8, 1, 3, 59, 5, 666666, tz),
                datetime.datetime(1800, 5, 16, 3, 5, 59, 666666, tz),
                datetime.datetime(1800, 8, 1, 3, 5, 59, 444444, tz),
            ):
                ts = pd.Timestamp(dt)
                pt = Pydt.fromdatetime(dt)
                for freq in ("D", "h", "m", "s", "ms", "us"):
                    ts_freq = "min" if freq == "m" else freq
                    self.assertEqual(ts.round(ts_freq), pt.round(freq))
                    self.assertEqual(ts.ceil(ts_freq), pt.ceil(freq))
                    self.assertEqual(ts.floor(ts_freq), pt.floor(freq))

        # fsp()
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            for dt in (
                datetime.datetime(2000, 1, 1, 0, 0, 0, 666666, tz),
                datetime.datetime(1970, 1, 1, 0, 0, 0, 666666, tz),
                datetime.datetime(1800, 1, 1, 0, 0, 0, 666666, tz),
                datetime.datetime(2000, 1, 1, 0, 0, 0, 444444, tz),
                datetime.datetime(1970, 1, 1, 0, 0, 0, 444444, tz),
                datetime.datetime(1800, 1, 1, 0, 0, 0, 444444, tz),
                datetime.datetime(2000, 1, 1, 0, 0, 0, 500000, tz),
                datetime.datetime(1970, 1, 1, 0, 0, 0, 500000, tz),
                datetime.datetime(1800, 1, 1, 0, 0, 0, 500000, tz),
            ):
                for precision in range(8):
                    pt = Pydt.fromdatetime(dt).fsp(precision)
                    self.assertEqual(dt.date(), pt.date())
                    self.assertEqual(dt.hour, pt.hour)
                    self.assertEqual(dt.minute, pt.minute)
                    self.assertEqual(dt.second, pt.second)
                    f = 10 ** (6 - min(precision, 6))
                    self.assertEqual(dt.microsecond // f * f, pt.microsecond)

        self.log_ended(test)

    def test_calendar(self) -> None:
        test = "Calendar"
        self.log_start(test)

        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = datetime.datetime(2023, 5, 16, 3, 4, 5, 666666, tz)
            pt = Pydt.fromdatetime(dt)
            # ISO ------------------------------------------------------------
            # isocalendar()
            self.assertEqual(dt.isocalendar(), tuple(pt.isocalendar().values()))
            # isoyear()
            self.assertEqual(dt.isocalendar()[0], pt.isoyear())
            # isoweek()
            self.assertEqual(dt.isocalendar()[1], pt.isoweek())
            # isoweekday()
            self.assertEqual(dt.isoweekday(), pt.isoweekday())

            # Year -----------------------------------------------------------
            self.assertFalse(pt.is_leap_year())
            self.assertTrue(Pydt.parse("2024-01-01").is_leap_year())
            self.assertFalse(pt.is_long_year())
            self.assertTrue(Pydt.parse("2026-01-01").is_long_year())
            self.assertEqual(13, pt.leap_bt_year(1970))
            self.assertEqual(365, pt.days_in_year())
            self.assertEqual(738520, pt.days_bf_year())
            self.assertEqual(136, pt.day_of_year())
            self.assertTrue(pt.is_year(2023))
            self.assertFalse(pt.is_year(2024))

            # Quarter --------------------------------------------------------
            self.assertEqual(91, pt.days_in_quarter())
            self.assertEqual(90, pt.days_bf_quarter())
            self.assertEqual(46, pt.day_of_quarter())
            self.assertTrue(pt.is_quarter(2))
            self.assertFalse(pt.is_quarter(1))

            # Month ----------------------------------------------------------
            self.assertEqual(31, pt.days_in_month())
            self.assertEqual(120, pt.days_bf_month())
            self.assertEqual(16, pt.day_of_month())
            self.assertTrue(pt.is_month(5))
            self.assertTrue(pt.is_month("May"))
            self.assertFalse(pt.is_month("Jan"))

            # Weekday --------------------------------------------------------
            self.assertEqual(1, pt.weekday)
            self.assertTrue(pt.is_weekday("Tue"))
            self.assertFalse(pt.is_weekday(3))
            self.assertTrue(pt.to_monday().is_weekday("Mon"))
            self.assertTrue(pt.to_tuesday().is_weekday("Tue"))
            self.assertTrue(pt.to_wednesday().is_weekday("Wed"))
            self.assertTrue(pt.to_thursday().is_weekday("Thu"))
            self.assertTrue(pt.to_friday().is_weekday("Fri"))
            self.assertTrue(pt.to_saturday().is_weekday("Sat"))
            self.assertTrue(pt.to_sunday().is_weekday("Sun"))

            # Day ------------------------------------------------------------
            self.assertTrue(pt.is_day(16))
            self.assertFalse(pt.is_day(1))
            self.assertTrue(pt.to_yesterday().is_day(15))
            self.assertTrue(pt.to_tomorrow().is_day(17))

            # Time -----------------------------------------------------------
            self.assertEqual(
                (dt.hour, dt.minute, dt.second, dt.microsecond // 1000, dt.microsecond),
                (pt.hour, pt.minute, pt.second, pt.millisecond, pt.microsecond),
            )

        self.log_ended(test)

    def test_timezone(self) -> None:
        test = "Timezone"
        self.log_start(test)

        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = datetime.datetime(2023, 5, 16, 3, 4, 5, 6, tz)
            pt = Pydt.fromdatetime(dt)
            # utcoffset
            self.assertEqual(dt.utcoffset(), pt.utcoffset())
            # tzname
            self.assertEqual(dt.tzname(), pt.tzname())
            # dst
            self.assertEqual(dt.dst(), pt.dst())
            for targ_tz in (
                None,
                datetime.UTC,
                ZoneInfo("CET"),
                datetime.timezone(datetime.timedelta(hours=-5)),
            ):
                # astimezone
                self.assertEqual(dt.astimezone(targ_tz), pt.astimezone(targ_tz))

        # . extra
        self.assertFalse(Pydt.now().is_local())
        if Pydt.now("Asia/Shanghai").is_local():
            self.assertFalse(Pydt.now("CET").is_local())
        self.assertFalse(Pydt.now().is_utc())
        self.assertTrue(Pydt.now("UTC").is_utc())
        self.assertFalse(Pydt.now().is_dst())
        self.assertFalse(Pydt.now("UTC").is_dst())
        self.assertFalse(Pydt.now("Asia/Shanghai").is_dst())
        self.assertTrue(Pydt(2013, 3, 31, 3, tzinfo="Europe/Paris").is_dst())
        self.assertEqual(0, Pydt.now("UTC").utcoffset_seconds())
        self.assertEqual(8 * 3600, Pydt.now("Asia/Shanghai").utcoffset_seconds())

        # . normalization
        self.assertEqual(
            datetime.datetime(2013, 3, 31, 3, 30, tzinfo=ZoneInfo("Europe/Paris")),
            Pydt(2013, 3, 31, 2, 30, tzinfo="Europe/Paris"),
        )

        # . dst transitions
        self.assertEqual(
            datetime.datetime(2013, 3, 31, 3, tzinfo=ZoneInfo("Europe/Paris")),
            Pydt(2013, 3, 31, 1, 59, 59, 999999, tzinfo="Europe/Paris")
            + datetime.timedelta(microseconds=1),
        )
        self.assertEqual(
            datetime.datetime(2013, 3, 31, 3, tzinfo=ZoneInfo("Europe/Paris")),
            Pydt(2013, 3, 31, 1, 59, 59, 999999, tzinfo="Europe/Paris").add(
                microseconds=1
            ),
        )

        # . compatability with pytz
        from pytz import timezone

        tz = timezone("Europe/Paris")
        dt = tz.localize(datetime.datetime(2013, 3, 31, 3, 30))
        self.assertEqual(dt, Pydt(2013, 3, 31, 3, 30, tzinfo=tz))
        self.assertEqual(dt, Pydt(2013, 3, 31, 3, 30, tzinfo="Europe/Paris"))

        self.log_ended(test)

    def test_arithmetic(self) -> None:
        test = "Arithmetic"
        self.log_start(test)

        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = Pydt(2023, 5, 16, 3, 4, 5, 6, tz)
            for args in [
                (1, 0, 0, 0, 0, 0, 0),
                (0, 1, 0, 0, 0, 0, 0),
                (0, 0, 1, 0, 0, 0, 0),
                (0, 0, 0, 1, 0, 0, 0),
                (0, 0, 0, 0, 1, 0, 0),
                (0, 0, 0, 0, 0, 1, 0),
                (0, 0, 0, 0, 0, 0, 1),
                (1, 1, 0, 0, 0, 0, 0),
                (1, 1, 1, 0, 0, 0, 0),
                (1, 1, 1, 1, 0, 0, 0),
                (1, 1, 1, 1, 1, 0, 0),
                (1, 1, 1, 1, 1, 1, 0),
                (1, 1, 1, 1, 1, 1, 1),
                (1, 1, 23, 1, 1, 1, 1),
                (1, 1, 23, 59, 1, 1, 1),
                (1, 1, 23, 59, 59, 1, 1),
                (1, 1, 23, 59, 59, 999, 1),
                (1, 1, 23, 59, 59, 999, 999),
                (1, 1, 24, 1, 1, 1, 1),
                (1, 1, 24, 60, 1, 1, 1),
                (1, 1, 24, 60, 60, 1, 1),
                (1, 1, 24, 60, 60, 1000, 1),
                (1, 1, 24, 60, 60, 1000, 1000000),
            ]:
                # add()
                kwargs = {
                    "weeks": args[0],
                    "days": args[1],
                    "hours": args[2],
                    "minutes": args[3],
                    "seconds": args[4],
                    "milliseconds": args[5],
                    "microseconds": args[6],
                }
                self.assertEqual(dt + datetime.timedelta(**kwargs), dt.add(**kwargs))

                # sub()
                self.assertEqual(dt - datetime.timedelta(**kwargs), dt.sub(**kwargs))

        # diff()
        inclusive = ("neither", "one", "both")
        # . microseconds
        unit = "us"
        dt_strs = ("1970-01-01 00:00:00.000001", "1969-12-31 23:59:59.999999")
        dt = Pydt.parse("1970-01-01 00:00:00.000000")
        for dt_str in dt_strs:
            for d, b in enumerate(inclusive):
                self.assertEqual(d, dt.diff(dt_str, unit, True, b))
        # . milliseconds
        unit = "ms"
        dt_strs = (
            "1970-01-01 00:00:00.001000",
            "1970-01-01 00:00:00.001001",
            "1970-01-01 00:00:00.001999",
            "1969-12-31 23:59:59.999000",
            "1969-12-31 23:59:59.999001",
            "1969-12-31 23:59:59.999999",
        )
        for dt_str in dt_strs:
            for d, b in enumerate(inclusive):
                self.assertEqual(d, dt.diff(dt_str, unit, True, b))
        # . seconds
        unit = "s"
        dt_strs = (
            "1970-01-01 00:00:01.000000",
            "1970-01-01 00:00:01.001000",
            "1970-01-01 00:00:01.999999",
            "1969-12-31 23:59:59.000000",
            "1969-12-31 23:59:59.001000",
            "1969-12-31 23:59:59.999999",
        )
        for dt_str in dt_strs:
            for d, b in enumerate(inclusive):
                self.assertEqual(d, dt.diff(dt_str, unit, True, b))
        # . minutes
        unit = "m"
        dt_strs = (
            "1970-01-01 00:01:00",
            "1970-01-01 00:01:01",
            "1970-01-01 00:01:59",
            "1969-12-31 23:59:00",
            "1969-12-31 23:59:01",
            "1969-12-31 23:59:59",
        )
        for dt_str in dt_strs:
            for d, b in enumerate(inclusive):
                self.assertEqual(d, dt.diff(dt_str, unit, True, b))
        # . hours
        unit = "h"
        dt_strs = (
            "1970-01-01 01:00:00",
            "1970-01-01 01:01:00",
            "1970-01-01 01:59:59",
            "1969-12-31 23:00:00",
            "1969-12-31 23:01:00",
            "1969-12-31 23:59:59",
        )
        for dt_str in dt_strs:
            for d, b in enumerate(inclusive):
                self.assertEqual(d, dt.diff(dt_str, unit, True, b))
        # . days
        unit = "D"
        dt_strs = (
            "1970-01-02 00:00:00",
            "1970-01-02 01:00:00",
            "1970-01-02 23:59:59",
            "1969-12-31 00:00:00",
            "1969-12-31 01:00:00",
            "1969-12-31 23:59:59",
        )
        for dt_str in dt_strs:
            for d, b in enumerate(inclusive):
                self.assertEqual(d, dt.diff(dt_str, unit, True, b))
        # . weeks
        unit = "W"
        dt_strs = (
            "1970-01-08 00:00:00",
            "1970-01-08 01:00:00",
            "1970-01-08 23:59:59",
            "1969-12-25 00:00:00",
            "1969-12-25 01:00:00",
            "1969-12-25 23:59:59",
        )
        for dt_str in dt_strs:
            for d, b in enumerate(inclusive):
                self.assertEqual(d, dt.diff(dt_str, unit, True, b))
        # . months
        unit = "M"
        dt_strs = (
            "1970-02-01",
            "1970-02-02",
            "1970-02-28",
            "1969-12-01",
            "1969-12-02",
            "1969-12-31",
        )
        for dt_str in dt_strs:
            for d, b in enumerate(inclusive):
                self.assertEqual(d, dt.diff(dt_str, unit, True, b))
        # . quarters
        unit = "Q"
        dt_strs = (
            "1970-04-01",
            "1970-04-02",
            "1970-06-30",
            "1969-10-01",
            "1969-10-02",
            "1969-12-31",
        )
        for dt_str in dt_strs:
            for d, b in enumerate(inclusive):
                self.assertEqual(d, dt.diff(dt_str, unit, True, b))
        # . years
        unit = "Y"
        dt_strs = (
            "1971-01-01",
            "1971-01-02",
            "1971-12-31",
            "1969-01-01",
            "1969-01-02",
            "1969-12-31",
        )
        for dt_str in dt_strs:
            for d, b in enumerate(inclusive):
                self.assertEqual(d, dt.diff(dt_str, unit, True, b))
        # . incomparable
        with self.assertRaises(errors.MixedTimezoneError):
            dt.diff("1972-02-02 01:01:01+01:00", "us")

        # addition
        dt = datetime.datetime.now()
        td = datetime.timedelta(1, 1, 1)
        pdt = Pydt.parse(dt)
        res = []
        for delta in (td, pd.Timedelta(td), np.timedelta64(td)):
            res.append(pdt + delta)
        self.assertTrue(all(r == res[0] and isinstance(r, Pydt) for r in res))
        self.assertEqual(res[0], dt + td)

        # right addition
        res = []
        for delta in (td, pd.Timedelta(td), np.timedelta64(td)):
            res.append(delta + pdt)
        self.assertTrue(
            all(r == res[0] and type(r) in (Pydt, pd.Timestamp)) for r in res
        )
        self.assertEqual(res[0], td + dt)

        # substraction
        dt_sub = datetime.datetime.now()
        res = []
        for dt_sub in (dt_sub, pd.Timestamp(dt_sub), np.datetime64(dt_sub)):
            res.append(pdt - dt_sub)
        self.assertTrue(
            all(r == res[0] and type(r) in (datetime.timedelta, pd.Timedelta))
            for r in res
        )
        self.assertEqual(res[0], dt - dt_sub)

        res = []
        for delta in (td, pd.Timedelta(td), np.timedelta64(td)):
            res.append(pdt - delta)
        self.assertTrue(all(r == res[0] and isinstance(r, Pydt) for r in res))
        self.assertEqual(res[0], dt - td)

        self.log_ended(test)

    def test_comparison(self) -> None:
        test = "Comparison"
        self.log_start(test)

        # is_past / is_future
        dt = Pydt.now().sub(minutes=1)
        self.assertTrue(dt.is_past())
        self.assertFalse(dt.is_future())
        dt = Pydt.now().add(minutes=1)
        self.assertFalse(dt.is_past())
        self.assertTrue(dt.is_future())

        # Comparison
        dt = Pydt(1970, 1, 1)
        eq = datetime.datetime(1970, 1, 1)
        gt = datetime.datetime(1970, 1, 1, microsecond=1)
        lt = datetime.datetime(1969, 12, 31, microsecond=999999)
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = dt.replace(tzinfo=tz)
            eq = eq.replace(tzinfo=tz)
            gt = gt.replace(tzinfo=tz)
            lt = lt.replace(tzinfo=tz)

            # Equal
            self.assertTrue(dt == eq)
            self.assertTrue(dt != gt and dt != lt)
            self.assertTrue(dt == pd.Timestamp(eq))
            self.assertTrue(dt != pd.Timestamp(gt) and dt != pd.Timestamp(lt))
            self.assertTrue(dt == Pydt.parse(eq))
            self.assertTrue(dt != Pydt.parse(gt) and dt != Pydt.parse(lt))
            if tz is None:
                self.assertTrue(dt == np.datetime64(eq))
                self.assertTrue(dt != np.datetime64(gt) and dt != np.datetime64(lt))

            # Greater
            self.assertTrue(dt > lt)
            self.assertTrue(dt > pd.Timestamp(lt))
            self.assertTrue(dt > Pydt.parse(lt))
            if tz is None:
                self.assertTrue(dt > np.datetime64(lt))

            # Greater / Equal
            self.assertTrue(dt >= eq and dt > lt)
            self.assertTrue(dt >= pd.Timestamp(eq) and pd.Timestamp(dt) > lt)
            self.assertTrue(dt >= Pydt.parse(eq) and Pydt.parse(dt) > lt)
            if tz is None:
                self.assertTrue(dt >= np.datetime64(eq) and np.datetime64(dt) > lt)

            # Less
            self.assertTrue(dt < gt)
            self.assertTrue(dt < pd.Timestamp(gt))
            self.assertTrue(dt < Pydt.parse(gt))
            if tz is None:
                self.assertTrue(dt < np.datetime64(gt))

            # Less / Equal
            self.assertTrue(dt <= eq and dt < gt)
            self.assertTrue(dt <= pd.Timestamp(eq) and pd.Timestamp(dt) < gt)
            self.assertTrue(dt <= Pydt.parse(eq) and Pydt.parse(dt) < gt)
            if tz is None:
                self.assertTrue(dt <= np.datetime64(eq) and np.datetime64(dt) < gt)

        # Incomparable
        with self.assertRaises(TypeError):
            dt > "1972-02-02 01:01:01+01:00"
        with self.assertRaises(TypeError):
            dt < datetime.timedelta(1)

        self.log_ended(test)

    def test_subclass(self) -> None:
        test = "Subclass"
        self.log_start(test)

        class MyPydt(Pydt):
            def greet(self) -> str:
                return "Hello"

        my_dt = MyPydt(2023, 5, 16, 3, 4, 5, 6)
        self.assertEqual("Hello", my_dt.greet())
        self.assertIsInstance(my_dt, Pydt)
        self.assertIsInstance(my_dt, MyPydt)
        self.assertIsInstance(my_dt.to_curr_month(20), MyPydt)

        self.log_ended(test)


if __name__ == "__main__":
    TestPydt().test_all()
