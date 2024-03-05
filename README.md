## Easy management of python datetime & pandas time Series.

Created to be used in a project, this package is published to github 
for ease of management and installation across different modules.

### Installation
Install from `PyPi`
``` bash
pip install cytimes
```

Install from `github`
``` bash
pip install git+https://github.com/AresJef/cyTimes.git
```

### Compatibility
Supports Python 3.10 and above.

### Features
Provides two classes to make working with datetime easier in Python.
- `pydt` (Python Datetime)
- `pddt` (Pandas Timestamp Series)

Both provide similar functionalities:
- Parse time string
- Access in different data types
- Conversion to `int/float` (ordinal, total_seconds, timestamp, etc.)
- Calender properties (days_in_month, weekday, etc.)
- Day manipulation (to_next_week, to_week, etc.)
- Month manipulation (to_next_month, to_to_month, etc.)
- Quarter manipulation (to_next_quarter, to_quarter, etc.)
- Year manipulation (to_next_year, to_year, etc.)
- Timezone manipulation (tz_localize, tz_convert, etc.)
- Frequency manipulation (freq_round, freq_ceil, freq_floor, etc.)
- Delta adjustment (Equivalent to adding `relativedelta` and `pandas.DateOffset`)
- Delta difference (Calcualte the absolute delta between two datetimes)
- Replacement (Equivalent to `datetime.replace` and custom `pandas.Series.dt.replace`)
- Supports addition / substruction / comparision.

### Performance
A major focus of this package is to optimize the datetime string parsing speed.
Performance gain primarily comes from re-writing the parsing codes in cython 
over python codes. The following results are tested on an Apple M1 Pro:

##### Strict Isoformat without Timezone
```
-------------------------Strict Isoformat w/o Timezone--------------------------
Text: '2023-08-01 12:00:00.000001'               Rounds: 100000
- pydt():                0.0727469171397388
- direct create:         0.013459332985803485    Perf Diff: -5.404942222357538x
- dt.fromisoformat():    0.009812582982704043    Perf Diff: -7.413635866108314x
- pendulum.parse():      0.4064326249063015      Perf Diff: 5.5869395004820515x
- dateutil.isoparse():   0.29077695799060166     Perf Diff: 3.997103512057442x
- dateutil.parse():      1.9294464997947216      Perf Diff: 26.522725301038776x
```

##### Strict Isoformat with Timezone
```
-------------------------Strict Isoformat w/t Timezone--------------------------
Text: '2023-08-01 12:00:00.000001+02:00'         Rounds: 100000
- pydt():                0.09097179211676121
- direct create:         0.014719957951456308    Perf Diff: -6.180166574984066x
- dt.fromisoformat():    0.012942458968609571    Perf Diff: -7.028941898707402x
- pendulum.parse():      0.40357612492516637     Perf Diff: 4.436277614572892x
- dateutil.isoparse():   0.42020483408123255     Perf Diff: 4.619067342785824x
- dateutil.parse():      2.508688542060554       Perf Diff: 27.576554047002638x
```
##### Loose Isoformat without Timezone
```
--------------------------Loose Isoformat w/o Timezone--------------------------
Text: '2023/08/01 12:00:00.000001'               Rounds: 100000
- pydt():                0.07247600005939603
- pendulum.parse():      0.8165926251094788      Perf Diff: 11.267076334790264x
- dateutil.parse():      1.9235264579765499      Perf Diff: 26.540185115074898x
```

##### Loose Isoformat with Timezone
```
--------------------------Loose Isoformat w/t Timezone--------------------------
Text: '2023/08/01 12:00:00.000001+02:00'         Rounds: 100000
- pydt():                0.09058075002394617
- dateutil.parse():      2.50379070895724        Perf Diff: 27.64153209479201x
```

##### Parsing Datetime Strings
```
----------------------------Datetime Stings Parsing-----------------------------
Total datetime strings: 374                      Rounds: 1000
- pydt():                0.6471776249818504
- dateutil.parse():      7.1415234580636024      Perf Diff: 11.034873862123835x
```

### Usage (pydt)
``` python
from cytimes import pydt, cytimedelta
import datetime, numpy as np, pandas as pd

# Create
pt = pydt('2021-01-01 00:00:00')
pt = pydt("2021 Jan 1 11:11 AM")
pt = pydt(datetime.datetime(2021, 1, 1, 0, 0, 0))
pt = pydt(datetime.date(2021, 1, 1))
pt = pydt(datetime.time(12, 0, 0))
pt = pydt(pd.Timestamp("2021-01-01 00:00:00"))
pt = pydt(np.datetime64("2021-01-01 00:00:00"))
pt = pydt.now()
pt = pydt.from_datetime(2021, 1, 1)
pt = pydt.from_ordinal(1)
pt = pydt.from_timestamp(1)
...

# Access in different data types
pt.dt # -> datetime.datetime
pt.date # -> datetime.date
pt.time # -> datetime.time
pt.timetz # -> datetime.time (with timezone)
pt.ts # -> pandas.Timestamp
pt.dt64 # -> numpy.datetime64
...

# Conversion
pt.dt_iso # -> str
pt.ordinal # -> int
pt.timestamp # -> float
...

# Calender
pt.is_leapyear() # -> bool
pt.days_bf_year # -> int
pt.days_in_month # -> int
pt.weekday # -> int
pt.isocalendar # -> tuple
...

# Year manipulation
pt.to_year_lst() # Go to the last day of the current year.
pt.to_curr_year("Feb", 30) # Go to the last day in February of the current year.
pt.to_year(-3, "Mar", 15) # Go to the 15th day in March of the current year - 3.
...

# Quarter manipulation
pt.to_quarter_1st() # Go to the first day of the current quarter.
pt.to_curr_quarter(2, 0) # Go the the 2nd month of the current quarter with the same day.
pt.to_quarter(3, 2, 31) # Go the the last day of the 2nd month of the current quarter + 3.
...

# Month manipulation
pt.to_month_lst() # Go to the last day of the current month.
pt.to_next_month(31) # Go to the last day of the next month.
pt.to_month(3, 15) # Go the the 15th day of the current month + 3.
...

# Weekday manipulation
pt.to_monday() # Go to Monday of the current week.
pt.to_curr_weekday("Sun") # Go to Sunday of the current week.
pt.to_weekday(-2, "Sat") # Go to Saturday of the current week - 2.
...

# Day manipulation
pt.to_tomorrow() # Go to Tomorrow.
pt.to_yesterday() # Go to Yesterday.
pt.to_day(-2) # Go to today - 2.
...

# Time manipulation
pt.to_time_start() # Go to the start of the time (00:00:00).
pt.to_time_end() # Go to the end of the time (23:59:59.999999).
pt.to_time(1, 1, 1, 1, 1) # Go to time (01:01:01.001001).
...

# Timezone manipulation
pt.tz_localize("UTC") # Equivalent to 'datetime.replace(tzinfo=UTC).
pt.tz_convert("CET") # Convert to "CET" timezone.
pt.tz_switch(targ_tz="CET", base_tz="UTC") # Localize to "UCT" & convert to "CET".

# Frequency manipulation
pt.freq_round("D") # Round datetime to the precisioin of hour.
pt.freq_ceil("s") # Ceil datetime to the precisioin of second.
pt.freq_floor("us") # Floor datetime to the precisioin of microsecond.

# Delta
pt.add_delta(years=1, months=1, days=1, milliseconds=1) # Add Y/M/D & ms to pydt.
pt.cal_delta("2023-01-01 12:00:00", unit="D") # Calcualte the absolute delta in days.
...

# Replace
pt.replace(year=1970, month=2, day=30) # Replace to the last day in "1970/02"

# Addition
pt = pt + datetime.timedelta(1)
pt = pt + cytimedelta(years=1, months=1)
...

# Substraction
delta = pt - datetime.datetime(1970, 1, 1)
delta = pt - "1970-01-01"
pt = pt - datetime.timedelta(1)
...

# Comparison
res = pt == datetime.datetime(1970, 1, 1)
res = pt == "1970-01-01"
res = pt >= datetime.datetime(1970, 1, 1)
res = pt >= "1970-01-01"
...
```

### Usage (pddt)
`pddt` provides similar functionality to `pydt` (Same methods and properties),
but is designed to work with `pandas.Series` instead. For datetime strings that 
are out of bounds for nanoseconds, pddt will try to convert it to `datetime64[us]` 
for greater time range compatibility.

##### Nanoseconds (datetime64[ns])
```python
from cytimes import pddt
dts = []
for year in range(2000, 2025 + 1):
    for month in range(1, 12 + 1):
        dts.append("%d-%02d-02 03:04:05.000006" % (year, month))
print(pddt(dts))
```
```
0     2000-01-02 03:04:05.000006
1     2000-02-02 03:04:05.000006
                 ...            
310   2025-11-02 03:04:05.000006
311   2025-12-02 03:04:05.000006
Length: 312, dtype: datetime64[ns]
```

##### Microseconds (datetime64[us])
```python
dts += ["2300-01-02 03:04:05.000006"]
print(pddt(dts))
```
```
0     2000-01-02 03:04:05.000006
1     2000-02-02 03:04:05.000006
                 ...            
311   2025-12-02 03:04:05.000006
312   2300-01-02 03:04:05.000006
Length: 313, dtype: datetime64[us]
```

### Acknowledgements
cyTimes is based on several open-source repositories.
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)

cyTimes makes modification of the following open-source repositories:
- [dateutil](https://github.com/dateutil/dateutil)

This package created cythonized versions of dateutil.parser (cyparser) and
dateutil.relativedelta (cytimedelta). All credits go to the original authors 
and contributors of `dateutil`.
