import warnings
from zoneinfo import ZoneInfo
import time, unittest, datetime
import numpy as np, pandas as pd, pendulum as pl
from cytimes.pydt import Pydt
from cytimes.pddt import Pddt
from cytimes import errors

warnings.filterwarnings("ignore")


class TestCase(unittest.TestCase):
    name: str = "Case"

    def test_all(self) -> None:
        pass

    # Utils
    def assertTimeUnit(self, pt: Pddt, unit: str):
        self.assertTrue(pt.unit == unit)

    def assertEqualPtDt(self, pt: Pddt, dt: datetime.datetime) -> None:
        self.assertEqual(dt, pt[0])

    def assertEqualPtDtMS(self, pt: Pddt, dt: datetime.datetime) -> None:
        self.assertTrue((dt - pt[0]).total_seconds() < 0.1)

    def assertEqualPtDt_unit(self, pt: Pddt, dt: Pydt, unit: str) -> None:
        if unit == "ns" or unit == "us":
            self.assertEqual(dt, pt[0])
        elif pt[0].microsecond % 1000 > 0:
            self.assertEqual(dt, pt[0])
        else:
            if not isinstance(dt, Pydt):
                dt = Pydt.fromdatetime(dt)
            self.assertEqual(dt.to_start_of(unit), pt[0])

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


class TestPddt(TestCase):
    name = "Pddt"

    _timezones = (
        None,
        datetime.UTC,
        ZoneInfo("CET"),
        ZoneInfo("America/New_York"),
        datetime.timezone(datetime.timedelta(hours=-5)),
    )
    _int_dtypes = (np.int8, np.int16, np.int32, np.int64)
    _uint_dtypes = (np.uint8, np.uint16, np.uint32, np.uint64)
    _float_dtypes = (np.float16, np.float32, np.float64)

    def test_all(self) -> None:
        self.test_parse()
        self.test_constructor()
        self.test_converter()
        self.test_manipulator()
        self.test_manipulator_edge()
        self.test_calendar()
        self.test_timezone()
        self.test_arithmetic()
        self.test_comparision()
        self.test_nat()

    def test_parse(self) -> None:
        test = "Parse"
        self.log_start(test)

        # nanoseconds parse timezone-naive
        data = [
            "2023-01-01 00:00:00",
            "2023-01-02 00:00:00",
            "2023-01-03 00:00:00",
        ]
        for tz in (None, datetime.UTC, ZoneInfo("CET")):
            dt = pd.DatetimeIndex(data, tz=tz)
            pt = Pddt(data, tz=tz)
            self.assertTrue(dt.equals(pt))
            self.assertTimeUnit(pt, "ns")
            self.assertEqual(pt.tzinfo is None, tz is None)
        data = [pd.Timestamp.min, pd.Timestamp.max]
        dt = pd.DatetimeIndex(data)
        pt = Pddt(data)
        self.assertTrue(dt.equals(pt))
        self.assertTimeUnit(pt, "ns")
        self.assertTrue(pt.tzinfo is None)

        # microseconds parse timezone-naive
        data = [
            "9999-01-01 00:00:00",
            "9999-01-02 00:00:00",
            "9999-01-03 00:00:00",
        ]
        for tz in (None, ZoneInfo("CET"), datetime.UTC):
            pt = Pddt(data, tz=tz)
            dt = pd.DatetimeIndex(data, dtype="datetime64[us]")
            if tz is not None:
                dt = dt.tz_localize(tz)
            self.assertTrue(dt.equals(pt))
            self.assertTimeUnit(pt, "us")
            if tz is None:
                self.assertTrue(pt.tzinfo is None)
            else:
                self.assertTrue(pt.tzinfo is not None)

        # nanoseconds parse timezone-aware
        data = [
            "2023-01-01 00:00:00+00:00",
            "2023-01-02 00:00:00+00:00",
            "2023-01-03 00:00:00+00:00",
        ]
        for tz in (None, ZoneInfo("CET"), datetime.UTC):
            pt = Pddt(data, tz=tz)
            if tz is None:
                dt = pd.DatetimeIndex(data)
            else:
                dt = pd.DatetimeIndex(data, tz=tz)
            self.assertTrue(dt.equals(pt))
            self.assertTimeUnit(pt, "ns")
            self.assertTrue(pt.tzinfo is not None)

        # microseconds parse timezone-aware
        data = [
            "9999-01-01 00:00:00+00:00",
            "9999-01-02 00:00:00+00:00",
            "9999-01-03 00:00:00+00:00",
        ]
        # fmt: off
        comp = (
            (None, ["9999-01-01 00:00:00", "9999-01-02 00:00:00", "9999-01-03 00:00:00"]),
            (ZoneInfo("CET"), ["9999-01-01 01:00:00", "9999-01-02 01:00:00", "9999-01-03 01:00:00"]),
            (datetime.UTC, ["9999-01-01 00:00:00", "9999-01-02 00:00:00", "9999-01-03 00:00:00"]),
        )
        # fmt: on
        for tz, dts in comp:
            pt = Pddt(data, tz=tz)
            dt = pd.DatetimeIndex(dts, dtype="datetime64[us]")
            dt = dt.tz_localize(datetime.UTC if tz is None else tz)
            self.assertTrue(dt.equals(pt))
            self.assertTimeUnit(pt, "us")
            self.assertTrue(pt.tzinfo is not None)

        # nanoseconds parse mixed-timezone
        data = [
            "2023-01-01 00:00:00",
            "2023-01-02 00:00:00+00:00",
            "2023-01-03 00:00:00+01:00",
        ]
        # fmt: off
        comp = (
            (None, ["2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-02 23:00:00"]),
            (ZoneInfo("CET"), ["2023-01-01 00:00:00", "2023-01-02 01:00:00", "2023-01-03 00:00:00"]),
            (datetime.UTC, ["2023-01-01 00:00:00", "2023-01-02 00:00:00", "2023-01-02 23:00:00"]),
        )
        # fmt: on
        for tz, dts in comp:
            pt = Pddt(data, tz=tz)
            dt = pd.DatetimeIndex(dts, tz=datetime.UTC if tz is None else tz)
            self.assertTrue(dt.equals(pt))
            self.assertTimeUnit(pt, "ns")
            self.assertTrue(pt.tzinfo is not None)

        # microseconds parse mixed-timezone
        data = [
            "9999-01-01 00:00:00",
            "9999-01-02 00:00:00+00:00",
            "9999-01-03 00:00:00+01:00",
        ]
        # fmt: off
        comp = (
            (None, ["9999-01-01 00:00:00", "9999-01-02 00:00:00", "9999-01-02 23:00:00"]),
            (ZoneInfo("CET"), ["9999-01-01 00:00:00", "9999-01-02 01:00:00", "9999-01-03 00:00:00"]),
            (datetime.UTC, ["9999-01-01 00:00:00", "9999-01-02 00:00:00", "9999-01-02 23:00:00"]),
        )
        # fmt: on
        for tz, dts in comp:
            pt = Pddt(data, tz=tz)
            dt = pd.DatetimeIndex(dts, dtype="datetime64[us]")
            dt = dt.tz_localize(datetime.UTC if tz is None else tz)
            self.assertTrue(dt.equals(pt))
            self.assertTimeUnit(pt, "us")
            self.assertTrue(pt.tzinfo is not None)

        self.log_ended(test)

    def test_constructor(self) -> None:
        test = "Constructor"
        self.log_start(test)

        # now()
        for tz in self._timezones:
            pt = Pddt.now(1, tz=tz)
            dt = Pydt.now(tz=tz)
            self.assertEqualPtDtMS(pt, dt)

        # utcnow()
        self.assertEqualPtDtMS(Pddt.utcnow(1), Pydt.utcnow())

        # today()
        self.assertEqualPtDtMS(Pddt.today(1), Pydt.today())

        # combine
        for tz in self._timezones:
            time_tz = ZoneInfo(tz) if isinstance(tz, str) else tz
            # fmt: off
            dt = Pydt.combine(datetime.date(1970, 1, 2), datetime.time(3, 4, 5, 6, time_tz))
            pt = Pddt.combine(1, datetime.date(1970, 1, 2), datetime.time(3, 4, 5, 6), tz)
            self.assertEqualPtDt(pt, dt)
            dt = Pydt.combine(datetime.date(1970, 1, 2), datetime.time(3, 4, 5, 6), tz)
            pt = Pddt.combine(1, datetime.date(1970, 1, 2), datetime.time(3, 4, 5, 6), tz)
            self.assertEqualPtDt(pt, dt)
            dt = Pydt.combine("1970-01-02", datetime.time(3, 4, 5, 6, time_tz))
            pt = Pddt.combine(1, "1970-01-02", datetime.time(3, 4, 5, 6), tz)
            self.assertEqualPtDt(pt, dt)
            dt = Pydt.combine("1970-01-02", datetime.time(3, 4, 5, 6), tz)
            pt = Pddt.combine(1, "1970-01-02", datetime.time(3, 4, 5, 6), tz)
            self.assertEqualPtDt(pt, dt)
            dt = Pydt.combine(datetime.date(1970, 1, 2), "03:04:05.000006", tz)
            pt = Pddt.combine(1, datetime.date(1970, 1, 2), "03:04:05.000006", tz)
            self.assertEqualPtDt(pt, dt)
            dt = Pydt.combine("1970-01-02", "03:04:05.000006", tz)
            pt = Pddt.combine(1, "1970-01-02", "03:04:05.000006", tz)
            self.assertEqualPtDt(pt, dt)
            # fmt: on

        # fromordinal
        for tz in self._timezones:
            for ordinal in (2, 100, 400, 800000):
                dt = Pydt.fromordinal(ordinal, tz=tz)
                for pt in (
                    Pddt.fromordinal(ordinal, 1, tz=tz),
                    Pddt.fromordinal([ordinal], tz=tz),
                    Pddt.fromordinal((ordinal,), tz=tz),
                ):
                    self.assertEqualPtDt(pt, dt)
            for ordinal in (2, 10, 50, 100):
                dt = Pydt.fromordinal(ordinal, tz=tz)
                for dtype in self._int_dtypes + self._uint_dtypes:
                    for pt in (
                        Pddt.fromordinal(np.array([ordinal], dtype=dtype), tz=tz),
                        Pddt.fromordinal(pd.Series([ordinal], dtype=dtype), tz=tz),
                    ):
                        self.assertEqualPtDt(pt, dt)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromordinal(1)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromordinal(["x"])
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromordinal(pd.Series(["1"]))
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromordinal("x")

        # fromseconds()
        for tz in self._timezones:
            for seconds in (-78914791.123, -1, 0, 1, 78914791.123):
                dt = Pydt.fromseconds(seconds, tz=tz)
                for pt in (
                    Pddt.fromseconds(seconds, 1, tz=tz),
                    Pddt.fromseconds([seconds], tz=tz),
                    Pddt.fromseconds((seconds,), tz=tz),
                ):
                    self.assertEqualPtDt(pt, dt)
            for seconds in (2, 10, 50, 100):
                dt = Pydt.fromseconds(seconds, tz=tz)
                for dtype in self._int_dtypes + self._uint_dtypes + self._float_dtypes:
                    for pt in (
                        Pddt.fromseconds(np.array([seconds], dtype=dtype), tz=tz),
                        Pddt.fromseconds(pd.Series([seconds], dtype=dtype), tz=tz),
                    ):
                        self.assertEqualPtDt(pt, dt)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromseconds(1)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromseconds(["x"])
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromseconds(pd.Series(["1"]))
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromseconds("x")

        # fromicroseconds()
        for tz in self._timezones:
            for us in (-78914791, -1, 0, 1, 78914791):
                dt = Pydt.frommicroseconds(us, tz=tz)
                for pt in (
                    Pddt.frommicroseconds(us, 1, tz=tz),
                    Pddt.frommicroseconds([us], tz=tz),
                    Pddt.frommicroseconds((us,), tz=tz),
                ):
                    self.assertEqualPtDt(pt, dt)
            for us in (2, 10, 50, 100):
                dt = Pydt.frommicroseconds(us, tz=tz)
                for dtype in self._int_dtypes + self._uint_dtypes:
                    for pt in (
                        Pddt.frommicroseconds(np.array([us], dtype=dtype), tz=tz),
                        Pddt.frommicroseconds(pd.Series([us], dtype=dtype), tz=tz),
                    ):
                        self.assertEqualPtDt(pt, dt)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.frommicroseconds(1)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.frommicroseconds(["x"])
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.frommicroseconds(pd.Series(["1"]))
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.frommicroseconds("x")

        # fromtimestamp()
        for tz in self._timezones:
            for ts in (-78914791.123, -1, 0, 1, 78914791.123):
                dt = Pydt.fromtimestamp(ts, tz=tz)
                for pt in (
                    Pddt.fromtimestamp(ts, 1, tz=tz),
                    Pddt.fromtimestamp([ts], tz=tz),
                    Pddt.fromtimestamp((ts,), tz=tz),
                ):
                    self.assertEqualPtDt(pt, dt)
            for ts in (2, 10, 50, 100):
                dt = Pydt.fromtimestamp(ts, tz=tz)
                for dtype in self._int_dtypes + self._uint_dtypes + self._float_dtypes:
                    for pt in (
                        Pddt.fromtimestamp(np.array([ts], dtype=dtype), tz=tz),
                        Pddt.fromtimestamp(pd.Series([ts], dtype=dtype), tz=tz),
                    ):
                        self.assertEqualPtDt(pt, dt)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromtimestamp(1)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromtimestamp(1231239871298.123, 1)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromtimestamp(["x"])
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromtimestamp([1231239871298.123])
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromtimestamp(pd.Series(["1"]))
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromtimestamp(pd.Series([1231239871298.123]))
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromtimestamp("x")

        # utcfromtimestamp()
        for ts in (-78914791.123, -1, 0, 1, 78914791.123):
            dt = Pydt.utcfromtimestamp(ts)
            for pt in (
                Pddt.utcfromtimestamp(ts, 1),
                Pddt.utcfromtimestamp([ts]),
                Pddt.utcfromtimestamp((ts,)),
            ):
                self.assertEqualPtDt(pt, dt)
        for ts in (2, 10, 50, 100):
            for dtype in self._int_dtypes + self._uint_dtypes + self._float_dtypes:
                dt = Pydt.utcfromtimestamp(ts)
                for pt in (
                    Pddt.utcfromtimestamp(np.array([ts], dtype=dtype)),
                    Pddt.utcfromtimestamp(pd.Series([ts], dtype=dtype)),
                ):
                    self.assertEqualPtDt(pt, dt)

        # fromisoformat()
        for iso in (
            "1970-01-02T03:04:05.000006",
            "1970-01-02T03:04:05+01:00",
            "9999-01-01",
        ):
            dt = Pydt.fromisoformat(iso)
            for pt in (
                Pddt.fromisoformat(iso, 1),
                Pddt.fromisoformat(np.array([iso])),
                Pddt.fromisoformat(pd.Series([iso])),
                Pddt.fromisoformat([iso]),
                Pddt.fromisoformat((iso,)),
            ):
                self.assertEqualPtDt(pt, dt)

        # fromisocalendar()
        base_dt = Pydt.now()
        iso_dict1 = base_dt.isocalendar()
        iso_dict2 = {
            "year": base_dt.year,
            "week": iso_dict1["week"],
            "day": iso_dict1["weekday"],
        }
        for iso in (
            iso_dict1,
            iso_dict2,
            (iso_dict1, iso_dict2),
            [iso_dict1, iso_dict2],
            pd.DataFrame([iso_dict1]),
            pd.DataFrame([iso_dict2]),
        ):
            for tz in self._timezones:
                dt = base_dt.replace(tzinfo=tz).normalize()
                pt = Pddt.fromisocalendar(iso, size=1, tz=tz)
                self.assertEqualPtDt(pt, dt)
        Pddt.fromisocalendar({"year": 20000, "week": 1, "day": 1}, size=1)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromisocalendar(1)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromisocalendar({})
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromisocalendar({"year": 2023}, 1)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromisocalendar({"year": 2023, "week": 1}, 1)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromisocalendar({"week": 1, "day": 1}, 1)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromisocalendar({"year": "1", "week": 1, "day": 1}, 1)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromisocalendar([1], 1)

        # fromdayofyear()
        base_dt = Pydt.now()
        doy_dict1 = {"year": base_dt.year, "doy": base_dt.day_of_year()}
        doy_dict2 = {"year": base_dt.year, "day": base_dt.day_of_year()}
        for doy in (
            doy_dict1,
            doy_dict2,
            (doy_dict1, doy_dict2),
            [doy_dict1, doy_dict2],
            pd.DataFrame([doy_dict1]),
            pd.DataFrame([doy_dict2]),
        ):
            for tz in self._timezones:
                dt = base_dt.replace(tzinfo=tz).normalize()
                pt = Pddt.fromdayofyear(doy, size=1, tz=tz)
                self.assertEqualPtDt(pt, dt)
        Pddt.fromdayofyear({"year": 20000, "doy": 1}, size=1)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromdayofyear(1)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromdayofyear({})
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromdayofyear({"year": 2023}, 1)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromdayofyear({"doy": 1}, 1)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromdayofyear({"year": "1", "doy": 1}, 1)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromdayofyear([1], 1)

        # fromdate()
        for tz in self._timezones:
            for date in (datetime.date(1970, 1, 2), pl.date(1970, 1, 2)):
                dt = Pydt.fromdate(date, tz)
                for pt in (
                    Pddt.fromdate(date, 1, tz=tz),
                    Pddt.fromdate(np.array([date]), tz=tz),
                    Pddt.fromdate(pd.Series([date]), tz=tz),
                    Pddt.fromdate(pd.DatetimeIndex([date]), tz=tz),
                    Pddt.fromdate([date], tz=tz),
                    Pddt.fromdate((date,), tz=tz),
                ):
                    self.assertEqualPtDt(pt, dt)

        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromdate("XXX", 1)

        # fromdatetime()
        for dt in (
            datetime.datetime(1970, 1, 2, 3, 4, 5, 6),
            pl.datetime(1970, 1, 2, 3, 4, 5, 6),
            pd.Timestamp("1970-01-02 03:04:05.000006"),
            Pydt(1970, 1, 2, 3, 4, 5, 6),
        ):
            for tz in self._timezones:
                dt = dt.replace(tzinfo=tz)
                for pt in (
                    Pddt.fromdatetime(dt, 1),
                    Pddt.fromdatetime(np.array([dt])),
                    Pddt.fromdatetime(pd.Series([dt])),
                    Pddt.fromdatetime(pd.DatetimeIndex([dt])),
                    Pddt.fromdatetime([dt]),
                    Pddt.fromdatetime((dt,)),
                ):
                    self.assertEqualPtDt(pt, dt)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.fromdatetime("XXX", 1)

        # fromdatetime64()
        for unit in ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"):
            for tz in self._timezones:
                dt64 = np.datetime64(1, unit)
                dt = Pydt.fromdatetime64(dt64, tz=tz)
                for pt in (
                    Pddt.fromdatetime64(dt64, 1, tz=tz),
                    Pddt.fromdatetime64(np.array([dt64]), tz=tz),
                    Pddt.fromdatetime64(pd.Series([dt64]), tz=tz),
                    Pddt.fromdatetime64(pd.DatetimeIndex([dt64]), tz=tz),
                    Pddt.fromdatetime64([dt64], tz=tz),
                    Pddt.fromdatetime64((dt64,), tz=tz),
                ):
                    self.assertEqualPtDt(pt, dt)

        # strptime()
        dt_str = "03:04:05.000006 1970-01-02 +01:00"
        fmt = "%H:%M:%S.%f %Y-%m-%d %z"
        dt = Pydt.strptime(dt_str, fmt)
        for pt in (
            Pddt.strptime(dt_str, fmt, 1),
            Pddt.strptime(np.array([dt_str]), fmt),
            Pddt.strptime(pd.Series([dt_str]), fmt),
            Pddt.strptime([dt_str], fmt),
            Pddt.strptime((dt_str,), fmt),
        ):
            self.assertEqualPtDt(pt, dt)
        with self.assertRaises(errors.InvalidArgumentError):
            Pddt.strptime(1, fmt, 1),

        # End of test
        self.log_ended(test)

    def test_converter(self) -> None:
        test = "Converter"
        self.log_start(test)

        for tz in self._timezones:
            dt = Pydt(2023, 1, 31, 3, 4, 5, 6, tz)
            pt = Pddt.fromdatetime(dt, 1)
            # ctime()
            self.assertEqualPtDt(pt.ctime(), dt.ctime())
            # strftime()
            for fmt in (
                "%d/%m/%Y, %H:%M:%S",
                "%d/%m/%Y, %H:%M:%S%z",
                "%d/%m/%Y, %H:%M:%S%Z",
            ):
                self.assertEqualPtDt(pt.strftime(fmt), dt.strftime(fmt))
            # isoformat()
            for sep in (" ", "T", "x"):
                self.assertEqualPtDt(pt.isoformat(sep), dt.isoformat(sep))
            # timedf()
            self.assertEqual(
                dt.timetuple()[:8],
                tuple(pt.timedf().iloc[0].tolist()),
            )
            # utctimedf()
            self.assertEqual(
                dt.utctimetuple()[:8],
                tuple(pt.utctimedf().iloc[0].tolist()),
            )
            # toordinal()
            self.assertEqualPtDt(pt.toordinal(), dt.toordinal())
            # seconds()
            self.assertEqual(pt.toseconds(False), dt.toseconds(False))
            self.assertEqual(pt.toseconds(True), dt.toseconds(True))
            # microseconds()
            self.assertEqual(pt.tomicroseconds(False), dt.tomicroseconds(False))
            self.assertEqual(pt.tomicroseconds(True), dt.tomicroseconds(True))
            # timestamp()
            self.assertEqual(pt.timestamp(), dt.timestamp())
            # datetime()
            self.assertEqualPtDt(pt.datetime(), dt)
            # date()
            self.assertEqual(pt.date(), dt.date())
            # time()
            self.assertEqual(pt.time(), dt.time())
            # timetz()
            self.assertEqual(pt.timetz(), dt.timetz())

        self.log_ended(test)

    def test_manipulator(self) -> None:
        test = "Manipulator"
        self.log_start(test)

        base_dts = [
            Pydt(2023, 8, 17, 13, 31, 31, 666666),
            Pydt(2023, 5, 14, 11, 29, 28, 444444),
            Pydt(1800, 8, 17, 13, 31, 31, 666666),
            Pydt(1800, 5, 14, 11, 29, 28, 444444),
            Pydt(2023, 5, 14, 11, 29, 28, 444444, ZoneInfo("CET")),
            Pydt(1800, 6, 15, 12, 30, 30, 500000, datetime.UTC),
            pd.Timestamp.min,
            pd.Timestamp.max,
        ]
        my_units = ("ns", "us", "ms", "s")
        # replace()
        for base_dt in base_dts:
            for unit in my_units:
                for targ_tz in (None, datetime.UTC, ZoneInfo("CET")):
                    dt = Pydt.fromdatetime(base_dt)
                    pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit)
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
                    ):
                        kwargs = {
                            "year": args[0],
                            "month": args[1],
                            "day": args[2],
                            "hour": args[3],
                            "minute": args[4],
                            "second": args[5],
                            "microsecond": args[6],
                            "tzinfo": args[7],
                        }
                        self.assertEqualPtDt_unit(
                            pt.replace(**kwargs), dt.replace(**kwargs), unit
                        )

        # to_*()
        for base_dt in base_dts:
            for unit in my_units:
                dt = Pydt.fromdatetime(base_dt)
                pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit)
                for offset in range(-3, 4):
                    # to_curr_year() / to_year()
                    for month in range(1, 13):
                        for day in (0, 1, 28, 29, 30, 31, 32):
                            self.assertEqualPtDt_unit(
                                pt.to_curr_year(month, day),
                                dt.to_curr_year(month, day),
                                unit,
                            )
                            self.assertEqualPtDt_unit(
                                pt.to_year(offset, month, day),
                                dt.to_year(offset, month, day),
                                unit,
                            )
                    # to_curr_quarter() / to_quarter()
                    for month in range(5):
                        for day in (0, 1, 28, 29, 30, 31, 32):
                            self.assertEqualPtDt_unit(
                                pt.to_curr_quarter(month, day),
                                dt.to_curr_quarter(month, day),
                                unit,
                            )
                            self.assertEqualPtDt_unit(
                                pt.to_quarter(offset, month, day),
                                dt.to_quarter(offset, month, day),
                                unit,
                            )
                    # to_curr_month() / to_month()
                    for day in (0, 1, 28, 29, 30, 31, 32):
                        self.assertEqualPtDt_unit(
                            pt.to_curr_month(day), dt.to_curr_month(day), unit
                        )
                        self.assertEqualPtDt_unit(
                            pt.to_month(offset, day), dt.to_month(offset, day), unit
                        )
                    # to_weekday()
                    for weekday in range(7):
                        self.assertEqualPtDt_unit(
                            pt.to_weekday(offset, weekday),
                            dt.to_weekday(offset, weekday),
                            unit,
                        )
                    # to_day()
                    for day in range(-32, 33):
                        self.assertEqualPtDt_unit(pt.to_day(day), dt.to_day(day), unit)

                # Test large offset
                for offset in (-1000, -500, -100, -1, 0, 100, 500, 1000):
                    dt = Pydt.fromdatetime(base_dt)
                    pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit)
                    # to_year()
                    self.assertEqualPtDt_unit(
                        pt.to_year(offset), dt.to_year(offset), unit
                    )
                    # to_quarter()
                    self.assertEqualPtDt_unit(
                        pt.to_quarter(offset), dt.to_quarter(offset), unit
                    )
                    # to_month()
                    self.assertEqualPtDt_unit(
                        pt.to_month(offset), dt.to_month(offset), unit
                    )
                    # to_week()
                    self.assertEqualPtDt_unit(
                        pt.to_weekday(offset), dt.to_weekday(offset), unit
                    )
                    # to_day()
                    self.assertEqualPtDt_unit(
                        pt.to_day(offset), dt.to_day(offset), unit
                    )

        # to_datetime()
        for base_dt in base_dts:
            for unit in my_units:
                dt = Pydt.fromdatetime(base_dt)
                pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit)
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
                    kwargs = {
                        "year": args[0],
                        "month": args[1],
                        "day": args[2],
                        "hour": args[3],
                        "minute": args[4],
                        "second": args[5],
                        "microsecond": args[6],
                    }
                    self.assertEqualPtDt_unit(
                        pt.to_datetime(**kwargs), dt.to_datetime(**kwargs), unit
                    )

        # to_date()
        for base_dt in base_dts:
            for unit in my_units:
                dt = Pydt.fromdatetime(base_dt)
                pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit)
                for args in (
                    [2000, -1, -1],
                    [2000, 3, -1],
                    [2000, 3, 31],
                    [-1, 3, -1],
                    [-1, 3, 31],
                    [-1, -1, 31],
                ):
                    kwargs = {"year": args[0], "month": args[1], "day": args[2]}
                    self.assertEqualPtDt_unit(
                        pt.to_date(**kwargs), dt.to_date(**kwargs), unit
                    )

        # to_time()
        for base_dt in base_dts:
            for unit in my_units:
                dt = Pydt.fromdatetime(base_dt)
                pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit)
                for args in (
                    [23, -1, -1, -1],
                    [-1, 59, -1, -1],
                    [-1, -1, 59, -1],
                    [-1, -1, -1, 999999],
                    [23, 59, -1, -1],
                    [23, 59, 59, -1],
                    [23, 59, 59, 999999],
                ):
                    kwargs = {
                        "hour": args[0],
                        "minute": args[1],
                        "second": args[2],
                        "microsecond": args[3],
                    }
                    self.assertEqualPtDt_unit(
                        pt.to_time(**kwargs), dt.to_time(**kwargs), unit
                    )

        # to_first_of / to_last_of
        to_units = ("Y", "Q", "M", "W", "Feb", "Sep")
        for base_dt in base_dts:
            for unit in my_units:
                dt = Pydt.fromdatetime(base_dt)
                pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit)
                for to_unit in to_units:
                    self.assertEqualPtDt_unit(
                        pt.to_first_of(to_unit), dt.to_first_of(to_unit), unit
                    )
                    self.assertEqualPtDt_unit(
                        pt.to_last_of(to_unit), dt.to_last_of(to_unit), unit
                    )

        # to_start_of / to_end_of
        to_units += ("D", "h", "m", "s", "ms", "us", "Mon", "Fri")
        for base_dt in base_dts:
            for unit in my_units:
                dt = Pydt.fromdatetime(base_dt)
                pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit)
                for to_unit in to_units:
                    self.assertEqualPtDt_unit(
                        pt.to_start_of(to_unit),
                        dt.to_start_of(to_unit),
                        unit,
                    )
                    self.assertEqualPtDt_unit(
                        pt.to_end_of(to_unit),
                        dt.to_end_of(to_unit),
                        unit,
                    )

        # is_first_of / is_last_of / is_start_of / is_end_of
        base_dts = [
            Pydt(2023, 8, 17, 13, 31, 31, 666666),
            Pydt(2023, 5, 14, 11, 29, 28, 444444),
            Pydt(1800, 8, 17, 13, 31, 31, 666666),
            Pydt(1800, 5, 14, 11, 29, 28, 444444),
            Pydt(2023, 5, 14, 11, 29, 28, 444444, ZoneInfo("CET")),
            Pydt(1800, 6, 15, 12, 30, 30, 500000, datetime.UTC),
            pd.Timestamp.min,
            pd.Timestamp.max,
        ]
        for base_dt in base_dts:
            for unit in my_units:
                dt = Pydt.fromdatetime(base_dt)
                pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit)
                # is_first_of / is_last_of
                units = ["W", "M", "Q", "Y", "Jan", "Feb", "Dec"]
                for unit in units:
                    self.assertEqual(
                        dt.to_first_of(unit).is_first_of(unit),
                        pt.to_first_of(unit).is_first_of(unit)[0],
                    )
                    self.assertEqual(
                        dt.to_last_of(unit).is_last_of(unit),
                        pt.to_last_of(unit).is_last_of(unit)[0],
                    )
                # is_start_of / is_end_of
                units += ["us", "ms", "s", "m", "h", "D", "Mon", "Sun"]
                for unit in units:
                    self.assertEqual(
                        dt.to_start_of(unit).is_start_of(unit),
                        pt.to_start_of(unit).is_start_of(unit)[0],
                    )
                    self.assertEqual(
                        dt.to_end_of(unit).is_end_of(unit),
                        pt.to_end_of(unit).is_end_of(unit)[0],
                    )

        # round()
        base_dts = [
            Pydt(2023, 8, 17, 13, 31, 31, 666666),
            Pydt(2023, 5, 14, 11, 29, 28, 444444),
            Pydt(1800, 8, 17, 13, 31, 31, 666666),
            Pydt(1800, 5, 14, 11, 29, 28, 444444),
            Pydt(1800, 6, 15, 12, 30, 30, 500000, datetime.UTC),
        ]
        to_units = ("ns", "us", "ms", "s", "min", "h", "D")
        for base_dt in base_dts:
            for unit in my_units:
                for to_unit in to_units:
                    dt = pd.DatetimeIndex([base_dt]).as_unit(unit)
                    pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit)
                    dt_res = dt.round(to_unit)
                    pt_res = pt.round(to_unit)
                    self.assertEqual(dt_res[0], pt_res[0])

        # ceil()
        for base_dt in base_dts + [pd.Timestamp.min]:
            for unit in my_units:
                for to_unit in to_units:
                    dt = pd.DatetimeIndex([base_dt]).as_unit(unit)
                    pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit)
                    dt_res = dt.ceil(to_unit)
                    pt_res = pt.ceil(to_unit)
                    self.assertEqual(dt_res[0], pt_res[0])

        # floor()
        for base_dt in base_dts + [pd.Timestamp.max]:
            for unit in my_units:
                for to_unit in to_units:
                    dt = pd.DatetimeIndex([base_dt]).as_unit(unit)
                    pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit)
                    dt_res = dt.floor(to_unit)
                    pt_res = pt.floor(to_unit)
                    self.assertEqual(dt_res[0], pt_res[0])

        # fsp()
        base_dts = (
            # fmt: off
            pd.Timestamp(year=2023, month=8, day=17, hour=13, minute=31, second=31, microsecond=666666, nanosecond=666),
            pd.Timestamp(year=2023, month=5, day=14, hour=11, minute=29, second=28, microsecond=444444, nanosecond=444),
            pd.Timestamp(year=1800, month=8, day=17, hour=13, minute=31, second=31, microsecond=666666, nanosecond=666),
            pd.Timestamp(year=1800, month=5, day=14, hour=11, minute=29, second=28, microsecond=444444, nanosecond=444),
            pd.Timestamp(year=2023, month=5, day=14, hour=11, minute=29, second=28, microsecond=444444, nanosecond=444, tzinfo=ZoneInfo("CET")),
            pd.Timestamp(year=1800, month=6, day=15, hour=12, minute=30, second=30, microsecond=500000, nanosecond=000, tzinfo=datetime.UTC),
            # fmt: on
        )
        for base_dt in base_dts:
            for unit in my_units:
                for precision in range(10):
                    dt: pd.DatetimeIndex = pd.DatetimeIndex([base_dt]).as_unit(unit)
                    pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit).fsp(precision)
                    if precision >= 9:
                        self.assertTrue(np.all(dt == pt))
                    elif precision <= 6:
                        f = 10 ** (6 - precision)
                        self.assertTrue(
                            np.all(dt.microsecond // f * f == pt.microsecond)
                        )
                        self.assertTrue(np.all(pt.nanosecond == 0))
                    else:
                        if unit == "ns":
                            f = 10 ** (9 - min(precision, 9))
                            self.assertTrue(
                                np.all(dt.nanosecond // f * f == pt.nanosecond)
                            )
                        else:
                            self.assertTrue(np.all(pt.nanosecond == 0))

        self.log_ended(test)

    def test_manipulator_edge(self) -> None:
        test = "Manipulator (edge)"
        self.log_start(test)

        # Edge cases -------------------------------------------------------------------------------------------
        #: min - '1677-09-21 00:12:43.145224193'
        #: max - '2262-04-11 23:47:16.854775807'
        pt = Pddt([pd.Timestamp.min, pd.Timestamp.max])

        # replace
        self.assertIsNone(pt.tzinfo)
        pt_tmp = pt.replace(tzinfo="Asia/Tokyo")
        self.assertIsNotNone(pt_tmp.tzinfo)
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.microsecond == pt.microsecond))

        # to_curr_year()
        pt_tmp = pt.to_curr_year(2, 15)
        self.assertTrue(np.all(pt_tmp.year == np.array([1677, 2262])))
        self.assertTrue(np.all(pt_tmp.month == 2))
        self.assertTrue(np.all(pt_tmp.day == 15))

        # to_year()
        pt_tmp = pt.to_year(10, 3, 10)
        self.assertTrue(np.all(pt_tmp.year == np.array([1687, 2272])))
        self.assertTrue(np.all(pt_tmp.month == 3))
        self.assertTrue(np.all(pt_tmp.day == 10))
        pt_tmp = pt.to_year(-10, 3, 10)
        self.assertTrue(np.all(pt_tmp.year == np.array([1667, 2252])))
        self.assertTrue(np.all(pt_tmp.month == 3))
        self.assertTrue(np.all(pt_tmp.day == 10))

        # to_curr_quarter()
        pt_tmp = pt.to_curr_quarter(2, 15)
        self.assertTrue(np.all(pt_tmp.year == np.array([1677, 2262])))
        self.assertTrue(np.all(pt_tmp.month == np.array([8, 5])))
        self.assertTrue(np.all(pt_tmp.day == 15))

        # to_quarter()
        pt_tmp = pt.to_quarter(2, 3, 10)
        self.assertTrue(np.all(pt_tmp.year == np.array([1678, 2262])))
        self.assertTrue(np.all(pt_tmp.month == np.array([3, 12])))
        self.assertTrue(np.all(pt_tmp.day == 10))
        pt_tmp = pt.to_quarter(-2, 3, 10)
        self.assertTrue(np.all(pt_tmp.year == np.array([1677, 2261])))
        self.assertTrue(np.all(pt_tmp.month == np.array([3, 12])))
        self.assertTrue(np.all(pt_tmp.day == 10))

        # to_curr_month()
        pt_tmp = pt.to_curr_month(15)
        self.assertTrue(np.all(pt_tmp.year == np.array([1677, 2262])))
        self.assertTrue(np.all(pt_tmp.month == np.array([9, 4])))
        self.assertTrue(np.all(pt_tmp.day == 15))

        # to_month()
        pt_tmp = pt.to_month(12, 15)
        self.assertTrue(np.all(pt_tmp.year == np.array([1678, 2263])))
        self.assertTrue(np.all(pt_tmp.month == np.array([9, 4])))
        self.assertTrue(np.all(pt_tmp.day == 15))
        pt_tmp = pt.to_month(-12, 15)
        self.assertTrue(np.all(pt_tmp.year == np.array([1676, 2261])))
        self.assertTrue(np.all(pt_tmp.month == np.array([9, 4])))
        self.assertTrue(np.all(pt_tmp.day == 15))

        # to_day()
        pt_tmp = pt.to_day(25)
        self.assertTrue(np.all(pt_tmp.year == np.array([1677, 2262])))
        self.assertTrue(np.all(pt_tmp.month == np.array([10, 5])))
        self.assertTrue(np.all(pt_tmp.day == np.array([16, 6])))
        pt_tmp = pt.to_day(-25)
        self.assertTrue(np.all(pt_tmp.year == np.array([1677, 2262])))
        self.assertTrue(np.all(pt_tmp.month == np.array([8, 3])))
        self.assertTrue(np.all(pt_tmp.day == np.array([27, 17])))

        # to_datetime()
        self.assertTrue(np.all(pt.to_datetime(year=2000).year == 2000))
        self.assertTrue(np.all(pt.to_datetime(month=1).month == 1))
        self.assertTrue(np.all(pt.to_datetime(day=1).day == 1))
        self.assertTrue(np.all(pt.to_datetime(hour=1).hour == 1))
        self.assertTrue(np.all(pt.to_datetime(minute=1).minute == 1))
        self.assertTrue(np.all(pt.to_datetime(second=1).second == 1))
        self.assertTrue(np.all(pt.to_datetime(microsecond=1).microsecond == 1))
        pt_tmp = pt.to_datetime(
            year=2000, month=1, day=1, hour=1, minute=1, second=1, microsecond=1
        )
        self.assertTrue(np.all(pt_tmp.year == 2000))
        self.assertTrue(np.all(pt_tmp.month == 1))
        self.assertTrue(np.all(pt_tmp.day == 1))
        self.assertTrue(np.all(pt_tmp.hour == 1))
        self.assertTrue(np.all(pt_tmp.minute == 1))
        self.assertTrue(np.all(pt_tmp.second == 1))
        self.assertTrue(np.all(pt_tmp.microsecond == 1))

        # to_date()
        self.assertTrue(np.all(pt.to_date(year=2000).year == 2000))
        self.assertTrue(np.all(pt.to_date(month=1).month == 1))
        self.assertTrue(np.all(pt.to_date(day=1).day == 1))
        pt_tmp = pt.to_date(year=2000, month=1, day=1)
        self.assertTrue(np.all(pt_tmp.year == 2000))
        self.assertTrue(np.all(pt_tmp.month == 1))
        self.assertTrue(np.all(pt_tmp.day == 1))

        # to_time()
        self.assertTrue(np.all(pt.to_time(hour=1).hour == 1))
        self.assertTrue(np.all(pt.to_time(minute=1).minute == 1))
        self.assertTrue(np.all(pt.to_time(second=1).second == 1))
        self.assertTrue(np.all(pt.to_time(microsecond=1).microsecond == 1))
        pt_tmp = pt.to_time(hour=1, minute=1, second=1, microsecond=1)
        self.assertTrue(np.all(pt_tmp.hour == 1))
        self.assertTrue(np.all(pt_tmp.minute == 1))
        self.assertTrue(np.all(pt_tmp.second == 1))
        self.assertTrue(np.all(pt_tmp.microsecond == 1))

        # to_first_of()
        pt_tmp = pt.to_first_of("Y")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == 1))
        self.assertTrue(np.all(pt_tmp.day == 1))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.microsecond == pt.microsecond))
        pt_tmp = pt.to_first_of("Q")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == np.array([7, 4])))
        self.assertTrue(np.all(pt_tmp.day == 1))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.microsecond == pt.microsecond))
        pt_tmp = pt.to_first_of("M")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.day == 1))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.microsecond == pt.microsecond))
        pt_tmp = pt.to_first_of("W")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.weekday == 0))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.microsecond == pt.microsecond))

        # to_last_of()
        pt_tmp = pt.to_last_of("Y")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == 12))
        self.assertTrue(np.all(pt_tmp.day == 31))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.microsecond == pt.microsecond))
        pt_tmp = pt.to_last_of("Q")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == np.array([9, 6])))
        self.assertTrue(np.all(pt_tmp.day == np.array([30, 30])))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.microsecond == pt.microsecond))
        pt_tmp = pt.to_last_of("M")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == np.array([30, 30])))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.microsecond == pt.microsecond))
        pt_tmp = pt.to_last_of("W")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.weekday == 6))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.microsecond == pt.microsecond))

        # to_start_of()
        pt_tmp = pt.to_start_of("Y")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == 1))
        self.assertTrue(np.all(pt_tmp.day == 1))
        self.assertTrue(np.all(pt_tmp.hour == 0))
        self.assertTrue(np.all(pt_tmp.minute == 0))
        self.assertTrue(np.all(pt_tmp.second == 0))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.to_start_of("Q")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == np.array([7, 4])))
        self.assertTrue(np.all(pt_tmp.day == 1))
        self.assertTrue(np.all(pt_tmp.hour == 0))
        self.assertTrue(np.all(pt_tmp.minute == 0))
        self.assertTrue(np.all(pt_tmp.second == 0))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.to_start_of("M")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == 1))
        self.assertTrue(np.all(pt_tmp.hour == 0))
        self.assertTrue(np.all(pt_tmp.minute == 0))
        self.assertTrue(np.all(pt_tmp.second == 0))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.to_start_of("W")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.weekday == 0))
        self.assertTrue(np.all(pt_tmp.hour == 0))
        self.assertTrue(np.all(pt_tmp.minute == 0))
        self.assertTrue(np.all(pt_tmp.second == 0))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.to_start_of("D")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == 0))
        self.assertTrue(np.all(pt_tmp.minute == 0))
        self.assertTrue(np.all(pt_tmp.second == 0))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.to_start_of("h")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == 0))
        self.assertTrue(np.all(pt_tmp.second == 0))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.to_start_of("m")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == 0))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.to_start_of("s")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.to_start_of("ms")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.millisecond == pt.millisecond))
        pt_tmp = pt.to_start_of("us")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.microsecond == pt.microsecond))

        # to_end_of()
        pt_tmp = pt.to_end_of("Y")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == 12))
        self.assertTrue(np.all(pt_tmp.day == 31))
        self.assertTrue(np.all(pt_tmp.hour == 23))
        self.assertTrue(np.all(pt_tmp.minute == 59))
        self.assertTrue(np.all(pt_tmp.second == 59))
        self.assertTrue(np.all(pt_tmp.microsecond == 999999))
        pt_tmp = pt.to_end_of("Q")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == np.array([9, 6])))
        self.assertTrue(np.all(pt_tmp.day == np.array([30, 30])))
        self.assertTrue(np.all(pt_tmp.hour == 23))
        self.assertTrue(np.all(pt_tmp.minute == 59))
        self.assertTrue(np.all(pt_tmp.second == 59))
        self.assertTrue(np.all(pt_tmp.microsecond == 999999))
        pt_tmp = pt.to_end_of("M")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == np.array([30, 30])))
        self.assertTrue(np.all(pt_tmp.hour == 23))
        self.assertTrue(np.all(pt_tmp.minute == 59))
        self.assertTrue(np.all(pt_tmp.second == 59))
        self.assertTrue(np.all(pt_tmp.microsecond == 999999))
        pt_tmp = pt.to_end_of("W")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.weekday == 6))
        self.assertTrue(np.all(pt_tmp.hour == 23))
        self.assertTrue(np.all(pt_tmp.minute == 59))
        self.assertTrue(np.all(pt_tmp.second == 59))
        self.assertTrue(np.all(pt_tmp.microsecond == 999999))
        pt_tmp = pt.to_end_of("D")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == 23))
        self.assertTrue(np.all(pt_tmp.minute == 59))
        self.assertTrue(np.all(pt_tmp.second == 59))
        self.assertTrue(np.all(pt_tmp.microsecond == 999999))
        pt_tmp = pt.to_end_of("h")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == 59))
        self.assertTrue(np.all(pt_tmp.second == 59))
        self.assertTrue(np.all(pt_tmp.microsecond == 999999))
        pt_tmp = pt.to_end_of("m")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == 59))
        self.assertTrue(np.all(pt_tmp.microsecond == 999999))
        pt_tmp = pt.to_end_of("s")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.microsecond == 999999))
        pt_tmp = pt.to_end_of("ms")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.millisecond == pt.millisecond))
        pt_tmp = pt.to_end_of("us")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.microsecond == pt.microsecond))

        # round()
        pt_tmp = pt.round("D")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == np.array([21, 12])))
        self.assertTrue(np.all(pt_tmp.hour == 0))
        self.assertTrue(np.all(pt_tmp.minute == 0))
        self.assertTrue(np.all(pt_tmp.second == 0))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.round("h")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == np.array([21, 12])))
        self.assertTrue(np.all(pt_tmp.hour == 0))
        self.assertTrue(np.all(pt_tmp.minute == 0))
        self.assertTrue(np.all(pt_tmp.second == 0))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.round("m")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == np.array([13, 47])))
        self.assertTrue(np.all(pt_tmp.second == 0))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.round("s")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == np.array([43, 17])))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.round("ms")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.millisecond == np.array([145, 855])))
        pt_tmp = pt.round("us")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.microsecond == np.array([145224, 854776])))

        # ceil()
        pt_tmp = pt.ceil("D")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == np.array([22, 12])))
        self.assertTrue(np.all(pt_tmp.hour == 0))
        self.assertTrue(np.all(pt_tmp.minute == 0))
        self.assertTrue(np.all(pt_tmp.second == 0))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.ceil("h")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == np.array([21, 12])))
        self.assertTrue(np.all(pt_tmp.hour == np.array([1, 0])))
        self.assertTrue(np.all(pt_tmp.minute == 0))
        self.assertTrue(np.all(pt_tmp.second == 0))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.ceil("m")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == np.array([13, 48])))
        self.assertTrue(np.all(pt_tmp.second == 0))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.ceil("s")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == np.array([44, 17])))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.ceil("ms")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.millisecond == np.array([146, 855])))
        pt_tmp = pt.ceil("us")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.microsecond == np.array([145225, 854776])))

        # floor()
        pt_tmp = pt.floor("D")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == 0))
        self.assertTrue(np.all(pt_tmp.minute == 0))
        self.assertTrue(np.all(pt_tmp.second == 0))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.floor("h")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == 0))
        self.assertTrue(np.all(pt_tmp.second == 0))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.floor("m")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == 0))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.floor("s")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.microsecond == 0))
        pt_tmp = pt.floor("ms")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.millisecond == pt.millisecond))
        pt_tmp = pt.floor("us")
        self.assertTrue(np.all(pt_tmp.year == pt.year))
        self.assertTrue(np.all(pt_tmp.month == pt.month))
        self.assertTrue(np.all(pt_tmp.day == pt.day))
        self.assertTrue(np.all(pt_tmp.hour == pt.hour))
        self.assertTrue(np.all(pt_tmp.minute == pt.minute))
        self.assertTrue(np.all(pt_tmp.second == pt.second))
        self.assertTrue(np.all(pt_tmp.microsecond == pt.microsecond))

        # fsp()
        for precision in range(10):
            pt_tmp = pt.fsp(precision)
            if precision >= 9:
                self.assertTrue(np.all(pt == pt_tmp))
            else:
                self.assertTrue(np.all(pt_tmp.year == pt.year))
                self.assertTrue(np.all(pt_tmp.month == pt.month))
                self.assertTrue(np.all(pt_tmp.day == pt.day))
                self.assertTrue(np.all(pt_tmp.hour == pt.hour))
                self.assertTrue(np.all(pt_tmp.minute == pt.minute))
                self.assertTrue(np.all(pt_tmp.second == pt.second))
                if precision <= 6:
                    f = 10 ** (6 - precision)
                    self.assertTrue(
                        np.all((pt.microsecond // f * f) == pt_tmp.microsecond)
                    )
                else:
                    self.assertTrue(np.all(pt.microsecond == pt_tmp.microsecond))
                self.assertTrue(np.all(pt_tmp.nanosecond == 0))

        self.log_ended(test)

    def test_calendar(self) -> None:
        test = "Calendar"
        self.log_start(test)

        base_dts = [
            Pydt(2023, 8, 17, 13, 31, 31, 666666),
            Pydt(2023, 5, 14, 11, 29, 28, 444444),
            Pydt(1800, 8, 17, 13, 31, 31, 666666),
            Pydt(1800, 5, 14, 11, 29, 28, 444444),
            Pydt(2023, 5, 14, 11, 29, 28, 444444, ZoneInfo("CET")),
            Pydt(1800, 6, 15, 12, 30, 30, 500000, datetime.UTC),
            pd.Timestamp.min,
            pd.Timestamp.max,
        ]
        my_units = ("ns", "us", "ms", "s")
        for base_dt in base_dts:
            for unit in my_units:
                dt = Pydt.fromdatetime(base_dt)
                pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit)
                # ISO ------------------------------------------------------------
                # isocalendar()
                dt_iso = dt.isocalendar()
                pt_iso = pt.isocalendar().iloc[0]
                self.assertEqual(
                    (dt_iso["year"], dt_iso["week"], dt_iso["weekday"]),
                    (pt_iso["year"], pt_iso["week"], pt_iso["weekday"]),
                )
                # isoyear()
                self.assertEqual(dt.isoyear(), pt.isoyear()[0])
                # isoweek()
                self.assertEqual(dt.isoweek(), pt.isoweek()[0])
                # isoweekday()
                self.assertEqual(dt.isoweekday(), pt.isoweekday()[0])

                # Year -----------------------------------------------------------
                self.assertEqual(dt.year, pt.year[0])
                self.assertEqual(dt.is_year(2023), pt.is_year(2023)[0])
                self.assertEqual(dt.is_leap_year(), pt.is_leap_year()[0])
                self.assertEqual(dt.is_long_year(), pt.is_long_year()[0])
                self.assertEqual(dt.leap_bt_year(1970), pt.leap_bt_year(1970)[0])
                self.assertEqual(dt.days_in_year(), pt.days_in_year()[0])
                self.assertEqual(dt.days_bf_year(), pt.days_bf_year()[0])
                self.assertEqual(dt.day_of_year(), pt.day_of_year()[0])

                # Quarter --------------------------------------------------------
                self.assertEqual(dt.quarter, pt.quarter[0])
                self.assertEqual(dt.is_quarter(3), pt.is_quarter(3)[0])
                self.assertEqual(dt.days_in_quarter(), pt.days_in_quarter()[0])
                self.assertEqual(dt.days_bf_quarter(), pt.days_bf_quarter()[0])
                self.assertEqual(dt.day_of_quarter(), pt.day_of_quarter()[0])

                # Month ----------------------------------------------------------
                self.assertEqual(dt.month, pt.month[0])
                self.assertEqual(dt.days_in_month(), pt.days_in_month()[0])
                self.assertEqual(dt.days_bf_month(), pt.days_bf_month()[0])
                self.assertEqual(dt.day_of_month(), pt.day_of_month()[0])
                self.assertEqual(dt.is_month("Aug"), pt.is_month("Aug")[0])
                self.assertEqual(dt.month_name(), pt.month_name()[0])

                # Weekday --------------------------------------------------------
                self.assertEqual(dt.weekday, pt.weekday[0])
                for wkd in (3, "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"):
                    self.assertEqual(dt.is_weekday(wkd), pt.is_weekday(wkd)[0])
                self.assertEqual(dt.weekday_name(), pt.weekday_name()[0])

                # Day ------------------------------------------------------------
                self.assertEqual(dt.day, pt.day[0])
                self.assertEqual(dt.is_day(14), pt.is_day(14)[0])

                # Time -----------------------------------------------------------
                if unit in ("ns", "us"):
                    self.assertEqual(
                        (dt.hour, dt.minute, dt.second, dt.millisecond, dt.microsecond),
                        (
                            pt.hour[0],
                            pt.minute[0],
                            pt.second[0],
                            pt.millisecond[0],
                            pt.microsecond[0],
                        ),
                    )

        self.log_ended(test)

    def test_timezone(self) -> None:
        test = "Timezone"
        self.log_start(test)

        # Validate with DatetimeIndex
        data = [
            "2023-01-01 01:01:01",
            "2023-01-02 01:01:01",
            "2023-01-03 01:01:01",
        ]
        for tz in (datetime.UTC, ZoneInfo("CET")):
            # tz_localize(): naive -> aware
            dt = pd.DatetimeIndex(data).tz_localize(tz)
            pt = Pddt(data).tz_localize(tz)
            self.assertTrue(np.equal(dt, pt).all())
            # tz_localize(): aware -> naive
            dt = pd.DatetimeIndex(data, tz=tz).tz_localize(None)
            pt = Pddt(data, tz=tz).tz_localize(None)
            self.assertTrue(np.equal(dt, pt).all())
            # tz_convert()
            for targ_tz in (datetime.UTC, ZoneInfo("CET")):
                dt = pd.DatetimeIndex(data, tz=tz).tz_convert(targ_tz)
                pt = Pddt(data, tz=tz).tz_convert(targ_tz)
                self.assertTrue(np.equal(dt, pt).all())

        # tz_localize: out of 'ns' range
        for tz in (ZoneInfo("CET"), ZoneInfo("PST8PDT")):
            pt = Pddt([pd.Timestamp.min, pd.Timestamp.max])
            #: should be created in 'ns' resolution
            self.assertEqual(pt.unit, "ns")
            #: should not raise
            pt_tmp = pt.tz_localize(tz)
            # : should auto convert to us to avoid overflow
            self.assertTrue(pt_tmp.unit == "us")
            # Should be timezone aware now
            self.assertIs(pt_tmp.tzinfo, tz)
            # Validate datetime values
            self.assertTrue(np.equal(pt.year, pt_tmp.year).all())
            self.assertTrue(np.equal(pt.month, pt_tmp.month).all())
            self.assertTrue(np.equal(pt.day, pt_tmp.day).all())
            self.assertTrue(np.equal(pt.hour, pt_tmp.hour).all())
            self.assertTrue(np.equal(pt.minute, pt_tmp.minute).all())
            self.assertTrue(np.equal(pt.second, pt_tmp.second).all())
            self.assertTrue(np.equal(pt.microsecond, pt_tmp.microsecond).all())

        # tz_convert: out of 'ns' range - [min]
        tz_n1 = datetime.timezone(datetime.timedelta(hours=-1))
        tz_n2 = datetime.timezone(datetime.timedelta(hours=-2))
        pt = Pddt([pd.Timestamp.min], tz=tz_n1)
        #: should be created in 'ns' resolution
        self.assertEqual(pt.unit, "ns")
        #: should not raise
        pt_tmp = pt.tz_convert(tz_n2)
        #: should auto convert to us to avoid overflow
        self.assertEqual(pt_tmp.unit, "us")
        #: Validate datetime values
        self.assertTrue(np.equal(pt.year, pt_tmp.year).all())
        self.assertTrue(np.equal(pt.month, pt_tmp.month).all())
        self.assertEqual(pt_tmp.day[0], 20)
        self.assertEqual(pt_tmp.hour[0], 23)
        self.assertTrue(np.equal(pt.minute, pt_tmp.minute).all())
        self.assertTrue(np.equal(pt.second, pt_tmp.second).all())
        self.assertTrue(np.equal(pt.microsecond, pt_tmp.microsecond).all())

        # tz_convert: out of 'ns' range - [max]
        tz_p1 = datetime.timezone(datetime.timedelta(hours=1))
        tz_p2 = datetime.timezone(datetime.timedelta(hours=2))
        pt = Pddt([pd.Timestamp.max], tz=tz_p1)
        #: should be created in 'ns' resolution
        self.assertEqual(pt.unit, "ns")
        #: should not raise
        pt_tmp = pt.tz_convert(tz_p2)
        #: should auto convert to us to avoid overflow
        self.assertEqual(pt_tmp.unit, "us")
        #: Validate datetime values
        self.assertTrue(np.equal(pt.year, pt_tmp.year).all())
        self.assertTrue(np.equal(pt.month, pt_tmp.month).all())
        self.assertEqual(pt_tmp.day[0], 12)
        self.assertEqual(pt_tmp.hour[0], 0)
        self.assertTrue(np.equal(pt.minute, pt_tmp.minute).all())
        self.assertTrue(np.equal(pt.second, pt_tmp.second).all())
        self.assertTrue(np.equal(pt.microsecond, pt_tmp.microsecond).all())

        # Validate with Pydt: part1
        base_dts = [
            Pydt(2023, 8, 17, 13, 31, 31, 666666),
            Pydt(2023, 5, 14, 11, 29, 28, 444444),
            pd.Timestamp.min,
            pd.Timestamp.max,
        ]
        my_units = ("ns", "us", "ms", "s")
        for base_dt in base_dts:
            for unit in my_units:
                for tz in self._timezones:
                    dt = Pydt.fromdatetime(base_dt).replace(tzinfo=tz)
                    pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit).replace(tzinfo=tz)
                    # . is_utc()
                    self.assertEqual(dt.is_utc(), pt.is_utc())
                    for targ_tz in self._timezones:
                        # astimezone()
                        dt = Pydt.fromdatetime(base_dt).replace(tzinfo=tz)
                        dt = dt.astimezone(targ_tz)
                        pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit).replace(
                            tzinfo=tz
                        )
                        pt = pt.astimezone(targ_tz)
                        self.assertEqualPtDt_unit(pt, dt, unit)
                        # tz_switch()
                        for naive in (True, False):
                            dt = Pydt.fromdatetime(base_dt).replace(tzinfo=tz)
                            dt = dt.tz_switch(
                                targ_tz, datetime.UTC if tz is None else tz, naive
                            )
                            pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit).replace(
                                tzinfo=tz
                            )
                            pt = pt.tz_switch(
                                targ_tz, datetime.UTC if tz is None else tz, naive
                            )
                            self.assertEqualPtDt_unit(pt, dt, unit)

        # Validate with Pydt: part2
        base_dts += [
            Pydt(1800, 8, 17, 13, 31, 31, 666666),
            Pydt(1800, 5, 14, 11, 29, 28, 444444),
            pd.Timestamp.min,
            pd.Timestamp.max,
        ]
        timezones = [i for i in self._timezones if i is not None]
        for base_dt in base_dts:
            for unit in my_units:
                for tz in timezones:
                    # tz_localize(): naive -> aware
                    dt = Pydt.fromdatetime(base_dt).tz_localize(tz)
                    pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit).tz_localize(tz)
                    self.assertEqualPtDt_unit(pt, dt, unit)
                    # tz_localize(): aware -> naive
                    dt = dt.tz_localize(None)
                    pt = pt.tz_localize(None)
                    self.assertEqualPtDt_unit(pt, dt, unit)
                    # tz_convert()
                    for targ_tz in timezones:
                        dt = (
                            Pydt.fromdatetime(base_dt)
                            .tz_localize(tz)
                            .tz_convert(targ_tz)
                        )
                        pt = (
                            Pddt.fromdatetime(base_dt, 1, as_unit=unit)
                            .tz_localize(tz)
                            .tz_convert(targ_tz)
                        )
                        self.assertEqualPtDt_unit(pt, dt, unit)

        self.log_ended(test)

    def test_arithmetic(self) -> None:
        test = "Arithmetic"
        self.log_start(test)

        # add() / sub()
        base_dts = [
            Pydt(2023, 8, 17, 13, 31, 31, 666666),
            Pydt(2023, 5, 14, 11, 29, 28, 444444),
            Pydt(1800, 8, 17, 13, 31, 31, 666666),
            Pydt(1800, 5, 14, 11, 29, 28, 444444),
            pd.Timestamp.min,
            pd.Timestamp.max,
        ]
        my_units = ("ns", "us", "ms", "s")
        fsp_map = {"ns": 9, "us": 6, "ms": 3, "s": 0}
        for base_dt in base_dts:
            for unit in my_units:
                for tz in self._timezones:
                    for args in [
                        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
                        (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                        (1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
                        (1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
                        (1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
                        (1, 1, 1, 1, 1, 0, 0, 0, 0, 0),
                        (1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
                        (1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
                        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                        (1, 1, 1, 1, 1, 23, 1, 1, 1, 1),
                        (1, 1, 1, 1, 1, 23, 59, 1, 1, 1),
                        (1, 1, 1, 1, 1, 23, 59, 59, 1, 1),
                        (1, 1, 1, 1, 1, 23, 59, 59, 999, 1),
                        (1, 1, 1, 1, 1, 23, 59, 59, 999, 999),
                        (1, 1, 1, 1, 1, 24, 1, 1, 1, 1),
                        (1, 1, 1, 1, 1, 24, 60, 1, 1, 1),
                        (1, 1, 1, 1, 1, 24, 60, 60, 1, 1),
                        (1, 1, 1, 1, 1, 24, 60, 60, 1000, 1),
                        (1, 1, 1, 1, 1, 24, 60, 60, 1000, 1000000),
                        (1000, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                        (1000, 1, 1, 1, 1, 23, 1, 1, 1, 1),
                        (1000, 1, 1, 1, 1, 23, 59, 1, 1, 1),
                        (1000, 1, 1, 1, 1, 23, 59, 59, 1, 1),
                        (1000, 1, 1, 1, 1, 23, 59, 59, 999, 1),
                        (1000, 1, 1, 1, 1, 23, 59, 59, 999, 999),
                        (1000, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                        (1000, 1, 1, 1, 1, 24, 1, 1, 1, 1),
                        (1000, 1, 1, 1, 1, 24, 60, 1, 1, 1),
                        (1000, 1, 1, 1, 1, 24, 60, 60, 1, 1),
                        (1000, 1, 1, 1, 1, 24, 60, 60, 1000, 1),
                        (1000, 1, 1, 1, 1, 24, 60, 60, 1000, 1000000),
                    ]:
                        # add()
                        kwargs = {
                            "years": args[0],
                            "quarters": args[1],
                            "months": args[2],
                            "weeks": args[3],
                            "days": args[4],
                            "hours": args[5],
                            "minutes": args[6],
                            "seconds": args[7],
                            "milliseconds": args[8],
                            "microseconds": args[9],
                        }
                        dt = (
                            Pydt.fromdatetime(base_dt)
                            .replace(tzinfo=tz)
                            .fsp(fsp_map[unit])
                        )
                        pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit).replace(
                            tzinfo=tz
                        )
                        # add()
                        self.assertEqualPtDt_unit(
                            pt.add(**kwargs), dt.add(**kwargs), unit
                        )
                        # sub()
                        self.assertEqualPtDt_unit(
                            pt.sub(**kwargs), dt.sub(**kwargs), unit
                        )

        # add() edge cases: min '1677-09-21 00:12:43.145224193'
        pt = Pddt([pd.Timestamp.min])
        self.assertEqual(pt.unit, "ns")
        pt_tmp = pt.add(years=1)
        self.assertEqual(pt_tmp.unit, "ns")
        self.assertEqual(pt_tmp.year[0], 1678)
        pt_tmp = pt.add(months=1)
        self.assertEqual(pt_tmp.unit, "ns")
        self.assertEqual(pt_tmp.month[0], 10)
        pt_tmp = pt.add(days=1)
        self.assertEqual(pt_tmp.unit, "us")
        self.assertEqual(pt_tmp.day[0], 22)
        pt_tmp = pt.add(hours=1)
        self.assertEqual(pt_tmp.unit, "us")
        self.assertEqual(pt_tmp.hour[0], 1)
        pt_tmp = pt.add(minutes=1)
        self.assertEqual(pt_tmp.unit, "us")
        self.assertEqual(pt_tmp.minute[0], 13)
        pt_tmp = pt.add(seconds=1)
        self.assertEqual(pt_tmp.unit, "us")
        self.assertEqual(pt_tmp.second[0], 44)
        pt_tmp = pt.add(milliseconds=1)
        self.assertEqual(pt_tmp.unit, "us")
        self.assertEqual(pt_tmp.millisecond[0], 146)
        pt_tmp = pt.add(microseconds=1)
        self.assertEqual(pt_tmp.unit, "us")
        self.assertEqual(pt_tmp.microsecond[0], 145225)
        pt_tmp = pt.add(nanoseconds=1000)
        self.assertEqual(pt_tmp.unit, "us")
        self.assertEqual(pt_tmp.microsecond[0], 145225)
        pt_tmp = pt.add(
            years=1,
            months=1,
            days=1,
            hours=1,
            minutes=1,
            seconds=1,
            milliseconds=1,
            microseconds=1,
            nanoseconds=1,
        )
        self.assertEqual(pt_tmp.unit, "ns")
        self.assertEqual(pt_tmp.year[0], 1678)
        self.assertEqual(pt_tmp.month[0], 10)
        self.assertEqual(pt_tmp.day[0], 22)
        self.assertEqual(pt_tmp.hour[0], 1)
        self.assertEqual(pt_tmp.minute[0], 13)
        self.assertEqual(pt_tmp.second[0], 44)
        self.assertEqual(pt_tmp.microsecond[0], 146225)
        self.assertEqual(pt_tmp.nanosecond[0], 194)

        # sub() edge cases: max '2262-04-11 23:47:16.854775807'
        pt = Pddt([pd.Timestamp.max])
        self.assertEqual(pt.unit, "ns")
        pt_tmp = pt.sub(years=1)
        self.assertEqual(pt_tmp.unit, "ns")
        self.assertEqual(pt_tmp.year[0], 2261)
        pt_tmp = pt.sub(months=1)
        self.assertEqual(pt_tmp.unit, "ns")
        self.assertEqual(pt_tmp.month[0], 3)
        pt_tmp = pt.sub(days=1)
        self.assertEqual(pt_tmp.unit, "us")
        self.assertEqual(pt_tmp.day[0], 10)
        pt_tmp = pt.sub(hours=1)
        self.assertEqual(pt_tmp.unit, "us")
        self.assertEqual(pt_tmp.hour[0], 22)
        pt_tmp = pt.sub(minutes=1)
        self.assertEqual(pt_tmp.unit, "us")
        self.assertEqual(pt_tmp.minute[0], 46)
        pt_tmp = pt.sub(seconds=1)
        self.assertEqual(pt_tmp.unit, "us")
        self.assertEqual(pt_tmp.second[0], 15)
        pt_tmp = pt.sub(milliseconds=1)
        self.assertEqual(pt_tmp.unit, "us")
        self.assertEqual(pt_tmp.millisecond[0], 853)
        pt_tmp = pt.sub(microseconds=1)
        self.assertEqual(pt_tmp.unit, "us")
        self.assertEqual(pt_tmp.microsecond[0], 854774)
        pt_tmp = pt.sub(nanoseconds=1000)
        self.assertEqual(pt_tmp.unit, "us")
        self.assertEqual(pt_tmp.microsecond[0], 854774)
        pt_tmp = pt.sub(
            years=1,
            months=1,
            days=1,
            hours=1,
            minutes=1,
            seconds=1,
            milliseconds=1,
            microseconds=1,
            nanoseconds=1,
        )
        self.assertEqual(pt_tmp.unit, "ns")
        self.assertEqual(pt_tmp.year[0], 2261)
        self.assertEqual(pt_tmp.month[0], 3)
        self.assertEqual(pt_tmp.day[0], 10)
        self.assertEqual(pt_tmp.hour[0], 22)
        self.assertEqual(pt_tmp.minute[0], 46)
        self.assertEqual(pt_tmp.second[0], 15)
        self.assertEqual(pt_tmp.microsecond[0], 853774)
        self.assertEqual(pt_tmp.nanosecond[0], 806)

        # diff(): timezone-naive
        diff_units = ("Y", "Q", "M", "W", "D", "h", "m", "s", "ms", "us")
        for base_dt in base_dts:
            for unit in ("us", "ns"):
                for comp_dt in (
                    Pydt(2023, 8, 17, 13, 31, 31, 666666),
                    Pydt(2023, 5, 14, 11, 29, 28, 444444),
                    Pydt(1800, 8, 17, 13, 31, 31, 666666),
                    Pydt(1800, 5, 14, 11, 29, 28, 444444),
                    pd.Timestamp.min,
                    pd.Timestamp.max,
                ):
                    dt = Pydt.fromdatetime(base_dt)
                    pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit)
                    for diff_unit in diff_units:
                        for incl in ("neither", "one", "both"):
                            for absl in (True, False):
                                self.assertEqual(
                                    dt.diff(comp_dt, diff_unit, absl, incl),
                                    pt.diff(comp_dt, diff_unit, absl, incl)[0],
                                )

        # diff(): timezone-aware
        tzinfos = (datetime.UTC, ZoneInfo("Asia/Shanghai"))
        for base_dt in base_dts:
            for unit in ("us", "ns"):
                for comp_dt in (
                    Pydt(2023, 8, 17, 13, 31, 31, 666666),
                    Pydt(2023, 5, 14, 11, 29, 28, 444444),
                    Pydt(1800, 8, 17, 13, 31, 31, 666666),
                    Pydt(1800, 5, 14, 11, 29, 28, 444444),
                ):
                    for my_tz in tzinfos:
                        for to_tz in tzinfos:
                            dt = Pydt.fromdatetime(base_dt).replace(tzinfo=my_tz)
                            pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit).replace(
                                tzinfo=my_tz
                            )
                            for diff_unit in diff_units:
                                for incl in ("neither", "one", "both"):
                                    comp_dt = comp_dt.replace(tzinfo=to_tz)
                                    for absl in (True, False):
                                        self.assertEqual(
                                            dt.diff(comp_dt, diff_unit, absl, incl),
                                            pt.diff(comp_dt, diff_unit, absl, incl)[0],
                                        )

        self.log_ended(test)

    def test_comparision(self) -> None:
        test = "Comparision"
        self.log_start(test)

        base_dts = [
            Pydt(3023, 8, 17, 13, 31, 31, 666666),
            Pydt(3023, 5, 14, 11, 29, 28, 444444),
            Pydt(2023, 8, 17, 13, 31, 31, 666666),
            Pydt(2023, 5, 14, 11, 29, 28, 444444),
            Pydt(1800, 8, 17, 13, 31, 31, 666666),
            Pydt(1800, 5, 14, 11, 29, 28, 444444),
            pd.Timestamp.min,
            pd.Timestamp.max,
        ]
        my_units = ("ns", "us", "ms", "s")
        for base_dt in base_dts:
            for unit in my_units:
                for tz in self._timezones:
                    dt = Pydt.fromdatetime(base_dt).replace(tzinfo=tz)
                    pt = Pddt.fromdatetime(base_dt, 1, as_unit=unit).replace(tzinfo=tz)
                    # is_past()
                    self.assertEqual(dt.is_past(), pt.is_past()[0])
                    # is_future()
                    self.assertEqual(dt.is_future(), pt.is_future()[0])

        self.log_ended(test)

    def test_nat(self) -> None:
        test = "Preserve 'NaT'"
        self.log_start(test)

        NaT_int: int = -9223372036854775808
        base_pt = Pddt(
            ["2025-11-02 01:30:00", "2025-11-02 03:30:00.123456789"],
            tz="America/New_York",
            ambiguous="NaT",
        )
        for pt in (base_pt, base_pt.tz_localize(None), base_pt.tz_convert("UTC")):
            for as_unit in ("ns", "us", "ms", "s"):
                pt = pt.as_unit(as_unit)

                # Convertor ----------------------------------------------
                self.assertIs(pt[0], pd.NaT)
                self.assertIs(pt.ctime()[0], np.nan)
                self.assertIs(pt.strftime("%Y-%m-%d %H:%M:%S")[0], np.nan)
                self.assertIs(pt.isoformat("T")[0], np.nan)
                self.assertEqual(pt.toordinal()[0], NaT_int)
                self.assertEqual(str(pt.toseconds()[0]), "nan")
                self.assertEqual(pt.tomicroseconds()[0], NaT_int)
                self.assertEqual(str(pt.timestamp()[0]), "nan")
                self.assertIs(pt.datetime()[0], pd.NaT)
                self.assertIs(pt.date()[0], pd.NaT)
                self.assertIs(pt.time()[0], pd.NaT)
                self.assertIs(pt.timetz()[0], pd.NaT)
                self.assertIs(pt.to_period("D")[0], pd.NaT)

                # Manipulator --------------------------------------------
                self.assertIs(pt.replace(year=1000)[0], pd.NaT)
                self.assertIs(pt.replace(year=2026)[0], pd.NaT)
                self.assertIs(pt.replace(month=12)[0], pd.NaT)
                self.assertIs(pt.replace(day=1)[0], pd.NaT)
                self.assertIs(pt.replace(day=25)[0], pd.NaT)
                self.assertIs(pt.replace(day=31)[0], pd.NaT)
                self.assertIs(pt.replace(hour=2)[0], pd.NaT)
                self.assertIs(pt.replace(minute=2)[0], pd.NaT)
                self.assertIs(pt.replace(second=2)[0], pd.NaT)
                self.assertIs(pt.replace(microsecond=2)[0], pd.NaT)
                self.assertIs(pt.replace(nanosecond=2)[0], pd.NaT)
                self.assertIs(pt.replace(tzinfo="UTC")[0], pd.NaT)
                self.assertIs(pt.to_curr_year("Feb", 31)[0], pd.NaT)
                self.assertIs(pt.to_year(1, "Feb", 31)[0], pd.NaT)
                self.assertIs(pt.to_curr_quarter(3, 31)[0], pd.NaT)
                self.assertIs(pt.to_quarter(1, 3, 31)[0], pd.NaT)
                self.assertIs(pt.to_curr_month(31)[0], pd.NaT)
                self.assertIs(pt.to_month(1, 31)[0], pd.NaT)
                self.assertIs(pt.to_curr_weekday("Mon")[0], pd.NaT)
                self.assertIs(pt.to_weekday(1, "Mon")[0], pd.NaT)
                self.assertIs(pt.to_day(1)[0], pd.NaT)
                self.assertIs(pt.to_day(-1)[0], pd.NaT)
                self.assertIs(pt.normalize()[0], pd.NaT)
                self.assertIs(pt.snap("MS")[0], pd.NaT)
                for unit in ("Y", "Q", "M", "W"):
                    self.assertIs(pt.to_first_of(unit)[0], pd.NaT)
                    self.assertFalse(pt.is_first_of(unit)[0])
                    self.assertIs(pt.to_last_of(unit)[0], pd.NaT)
                    self.assertFalse(pt.is_last_of(unit)[0])
                for unit in ("Y", "Q", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"):
                    self.assertIs(pt.to_start_of(unit)[0], pd.NaT)
                    self.assertFalse(pt.is_start_of(unit)[0])
                    self.assertIs(pt.to_end_of(unit)[0], pd.NaT)
                    self.assertFalse(pt.is_end_of(unit)[0])
                for unit in ("D", "h", "m", "s", "ms", "us", "ns"):
                    self.assertIs(pt.round(unit)[0], pd.NaT)
                    self.assertIs(pt.floor(unit)[0], pd.NaT)
                    self.assertIs(pt.ceil(unit)[0], pd.NaT)
                for fsp in range(10):
                    self.assertIs(pt.fsp(fsp)[0], pd.NaT)

                # Calendar -----------------------------------------------
                self.assertEqual(pt.isoyear()[0], NaT_int)
                self.assertEqual(pt.isoweek()[0], NaT_int)
                self.assertEqual(pt.isoweekday()[0], NaT_int)
                self.assertEqual(pt.year[0], NaT_int)
                self.assertFalse(pt.is_leap_year()[0])
                self.assertFalse(pt.is_long_year()[0])
                self.assertEqual(pt.leap_bt_year(3000)[0], NaT_int)
                self.assertEqual(pt.days_in_year()[0], NaT_int)
                self.assertEqual(pt.days_bf_year()[0], NaT_int)
                self.assertEqual(pt.day_of_year()[0], NaT_int)
                self.assertEqual(pt.quarter[0], NaT_int)
                self.assertFalse(pt.is_quarter(3)[0])
                self.assertEqual(pt.days_in_quarter()[0], NaT_int)
                self.assertEqual(pt.days_bf_quarter()[0], NaT_int)
                self.assertEqual(pt.day_of_quarter()[0], NaT_int)
                self.assertEqual(pt.month[0], NaT_int)
                self.assertFalse(pt.is_month("Aug")[0])
                self.assertEqual(pt.days_in_month()[0], NaT_int)
                self.assertEqual(pt.days_bf_month()[0], NaT_int)
                self.assertEqual(pt.day_of_month()[0], NaT_int)
                self.assertIs(pt.month_name()[0], np.nan)
                self.assertEqual(pt.weekday[0], NaT_int)
                self.assertFalse(pt.is_weekday("Mon")[0])
                self.assertIs(pt.weekday_name()[0], np.nan)
                self.assertEqual(pt.day[0], NaT_int)
                self.assertFalse(pt.is_day(14)[0])
                self.assertEqual(pt.hour[0], NaT_int)
                self.assertEqual(pt.minute[0], NaT_int)
                self.assertEqual(pt.second[0], NaT_int)
                self.assertEqual(pt.millisecond[0], NaT_int)
                self.assertEqual(pt.microsecond[0], NaT_int)
                self.assertEqual(pt.nanosecond[0], NaT_int)

                # Timezone -----------------------------------------------
                self.assertIs(pt.astimezone("CET")[0], pd.NaT)
                if pt.tzinfo is None:
                    self.assertIs(pt.tz_localize("CET")[0], pd.NaT)
                    self.assertIs(pt.tz_switch("CET", "UTC")[0], pd.NaT)
                else:
                    self.assertIs(pt.tz_localize(None)[0], pd.NaT)
                    self.assertIs(pt.tz_convert("CET")[0], pd.NaT)
                    self.assertIs(pt.tz_switch("CET")[0], pd.NaT)

                # Values -------------------------------------------------
                for unit in ("ns", "us", "ms", "s"):
                    self.assertIs(pt.as_unit(unit)[0], pd.NaT)

                # Arithmetic ---------------------------------------------
                self.assertIs(pt.add(years=1)[0], pd.NaT)
                self.assertIs(pt.add(months=1)[0], pd.NaT)
                self.assertIs(pt.add(days=1)[0], pd.NaT)
                self.assertIs(pt.add(hours=1)[0], pd.NaT)
                self.assertIs(pt.add(minutes=1)[0], pd.NaT)
                self.assertIs(pt.add(seconds=1)[0], pd.NaT)
                self.assertIs(pt.add(milliseconds=1)[0], pd.NaT)
                self.assertIs(pt.add(microseconds=1)[0], pd.NaT)
                self.assertIs(pt.add(nanoseconds=1)[0], pd.NaT)
                for unit in ("Y", "Q", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"):
                    for incl in ("neither", "one", "both"):
                        for abso in (True, False):
                            self.assertEqual(
                                pt.diff(
                                    datetime.datetime.now(pt.tzinfo),
                                    unit=unit,
                                    absolute=abso,
                                    inclusive=incl,
                                )[0],
                                NaT_int,
                            )

                # Comparison ---------------------------------------------
                self.assertFalse(pt.is_past()[0])
                self.assertFalse(pt.is_future()[0])

        self.log_ended(test)


if __name__ == "__main__":
    TestPddt().test_all()
