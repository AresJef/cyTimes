## Easy management of python datetime & pandas time Series.

Created to be used in a project, this package is published to github  for ease of management and installation across different modules.

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
- `pddt` (Pandas Series / DatetimeIndex)

Both provide similar functionalities:
- Parse datetime strings or datetime objects.
- Access in different data types.
- Conversion to numeric values (ordinal, total_seconds, timestamp, etc.)
- Calender properties (days_in_month, weekday, etc.)
- Year manipulation (to_next_year, to_year, etc.)
- Quarter manipulation (to_next_quarter, to_quarter, etc.)
- Month manipulation (to_next_month, to_to_month, etc.)
- Day manipulation (to_next_week, to_week, etc.)
- Time manipulation (to_time_start, to_time, etc.)
- Timezone manipulation (tz_localize, tz_convert, etc.)
- Frequency manipulation (freq_round, freq_ceil, freq_floor, etc.)
- Delta adjustment (Equivalent to adding `relativedelta` or `pandas.DateOffset`)
- Delta difference (Calcualte the absolute delta between two datetimes)
- Supports addition / substruction / comparision.

### Parser Performance
A major focus of this package is to optimize the datetime string parsing speed (through Cython), meanwhile maintains the maximum support for different datetime string formats. The following results are tested on an Apple M1 Pro:

##### Strict Isoformat without Timezone
```
------------------------ Strict Isoformat w/o Timezone -------------------------
Text:   '2023-08-01 12:00:00.000001'        Rounds: 100,000
- pydt():                   0.056599s
- direct create:            0.013991s       Perf Diff: -3.045365x
- dt.fromisoformat():	    0.010218s       Perf Diff: -4.539231x
- pendulum.parse():         0.406704s       Perf Diff: +6.185740x
- dateutil.isoparse():	    0.301066s       Perf Diff: +4.319307x
- dateutil.parse():         2.122079s       Perf Diff: +36.493413x

##### Strict Isoformat with Timezone
```

##### Strict Isoformat with Timezone
```
------------------------ Strict Isoformat w/t Timezone -------------------------
Text:   '2023-08-01 12:00:00.000001+02:00'  Rounds: 100,000
- pydt():                   0.065986s
- direct create:            0.014609s       Perf Diff: -3.516726x
- dt.fromisoformat():       0.013402s       Perf Diff: -3.923484x
- pendulum.parse():         0.412670s       Perf Diff: +5.253882x
- dateutil.isoparse():      0.457038s       Perf Diff: +5.926272x
- dateutil.parse():         2.611803s       Perf Diff: +38.581074x
```

##### Loose Isoformat without Timezone
```
------------------------- Loose Isoformat w/o Timezone -------------------------
Text:   '2023/08/01 12:00:00.000001'        Rounds: 100,000
- pydt():                   0.057039s
- pendulum.parse():         0.838589s       Perf Diff: +13.701917x
- dateutil.parse():         2.062576s       Perf Diff: +35.160516x
```

##### Loose Isoformat with Timezone
```
------------------------- Loose Isoformat w/t Timezone -------------------------
Text:   '2023/08/01 12:00:00.000001+02:00'  Rounds: 100,000
- pydt():                   0.066949s
- dateutil.parse():         2.612083s       Perf Diff: +38.016035x
```

##### Parse Datetime Strings
```
---------------------------- Parse Datetime Strings ----------------------------
Total datetime strings: #378                Rounds: 1,000
- pydt():                   0.587047s
- dateutil.parse():         7.182461s       Perf Diff: +11.234897x
```

### Usage for <'pydt'>
For more detail information, please refer to class methods' documentation.
``` python
from cytimes import pydt, cytimedelta
import datetime, numpy as np, pandas as pd

# Create
pt = pydt('2021-01-01 00:00:00')  # ISO format string
pt = pydt("2021 Jan 1 11:11 AM")  # datetime string
pt = pydt(datetime.datetime(2021, 1, 1, 0, 0, 0))  # <'datetime.datetime'>
pt = pydt(datetime.date(2021, 1, 1))  # <'datetime.date'>
pt = pydt(pd.Timestamp("2021-01-01 00:00:00"))  # <'pandas.Timestamp'>
pt = pydt(np.datetime64("2021-01-01 00:00:00"))  # <'numpy.datetime64'>
pt = pydt.now()  # current time
pt = pydt.from_ordinal(1)
pt = pydt.from_timestamp(1)
...

# . multi-language support
# . common month / weekday / ampm
# . EN / DE / FR / IT / ES / PT / NL / SE / PL / TR / CN
pt = pydt("februar 23, 2023")  # DE
pt = pydt("martes mayo 23, 2023")  # ES
pt = pydt("2023年3月15日 12时15分50秒")  # CN
...

# Access in different data types
pt.dt  # <'datetime.datetime'>
pt.date  # <'datetime.date'>
pt.time  # <'datetime.time'>
pt.timetz  # <'datetime.time'> (with timezone)
pt.ts  # <'pandas.Timestamp'>
pt.dt64  # <'numpy.datetime64'>
...

# Conversion
pt.dt_iso  # <'str'> ISO format
pt.ordinal  # <'int'> ordinal of the date
pt.timestamp  # <'float'> timestamp
...

# Calender
pt.is_leapyear()  # <'bool'>
pt.days_bf_year  # <'int'>
pt.days_in_month  # <'int'>
pt.weekday  # <'int'>
pt.isocalendar  # <'dict'>
...

# Year manipulation
pt.to_year_lst()  # Go to the last day of the current year.
pt.to_curr_year("Feb", 30)  # Go to the last day in February of the current year.
pt.to_year(-3, "Mar", 15)  # Go to the 15th day in March of the current year(-3).
...

# Quarter manipulation
pt.to_quarter_1st()  # Go to the first day of the current quarter.
pt.to_curr_quarter(2, 0)  # Go the the 2nd month of the current quarter with the same day.
pt.to_quarter(3, 2, 31)  # Go the the last day of the 2nd month of the current quarter(+3).
...

# Month manipulation
pt.to_month_lst()  # Go to the last day of the current month.
pt.to_next_month(31)  # Go to the last day of the next month.
pt.to_month(3, 15)  # Go the the 15th day of the current month(+3).
...

# Weekday manipulation
pt.to_monday()  # Go to Monday of the current week.
pt.to_curr_weekday("Sun")  # Go to Sunday of the current week.
pt.to_weekday(-2, "Sat")  # Go to Saturday of the current week(-2).
...

# Day manipulation
pt.to_tomorrow() # Go to Tomorrow.
pt.to_yesterday() # Go to Yesterday.
pt.to_day(-2) # Go to today(-2).
...

# Time manipulation
pt.to_time_start() # Go to the start of the time (00:00:00).
pt.to_time_end() # Go to the end of the time (23:59:59.999999).
pt.to_time(1, 1, 1, 1, 1) # Go to specific time (01:01:01.001001).
...

# Timezone manipulation
pt.tz_localize("UTC")  # Equivalent to 'datetime.replace(tzinfo=UTC).
pt.tz_convert("CET")  # Convert to "CET" timezone.
pt.tz_switch(targ_tz="CET", base_tz="UTC")  # Localize to "UTC" & convert to "CET".

# Frequency manipulation
pt.freq_round("D")  # Round datetime to the resolution of hour.
pt.freq_ceil("s")  # Ceil datetime to the resolution of second.
pt.freq_floor("us")  # Floor datetime to the resolution of microsecond.

# Delta
pt.add_delta(years=1, months=1, days=1, milliseconds=1)  # Add Y/M/D & ms.
pt.cal_delta("2023-01-01 12:00:00", unit="D", inclusive="both")  # Calcualte the absolute delta in days.
...

# Addition Support
# <'datetime.timedelta>, <'pandas.Timedelta'>, <'numpy.timedelta64'>
# <'dateutil.relativedelta'>, <'cytimes.cytimedelta'>
pt = pt + datetime.timedelta(1)
pt = pt + cytimedelta(years=1, months=1)
...

# Substraction Support
# <'datetime.datetime'>, <'pandas.Timestamp'>, <'numpy.datetime64'>, <'str'>, <'pydt'>
# <'datetime.timedelta>, <'pandas.Timedelta'>, <'numpy.timedelta64'>
# <'dateutil.relativedelta'>, <'cytimes.cytimedelta'>
delta = pt - datetime.datetime(1970, 1, 1)
delta = pt - "1970-01-01"
pt = pt - datetime.timedelta(1)
...

# Comparison Support
# <'datetime.datetime'>, <'pandas.Timestamp'>, <'str'>, <'pydt'>
res = pt == datetime.datetime(1970, 1, 1)
res = pt == "1970-01-01"
...
```

### Usage for <'pddt'>
Class `pddt` provides similar functionality to `pydt` (methods and properties, see examples for `pydt`), but is designed to work with `<'pandas.Series'>` and `<'pandas.DatetimeIndex>`. 

##### Out of bounds for nanoseconds
When encountering datetime values that are out of bounds for nanoseconds `datetime64[ns]`, `pddt` will automatically try to parse the value into microseconds `datetime64[us]` for greater compatibility.
```python
from cytimes import pddt

dts = [
    "2000-01-02 03:04:05.000006",
    "2100-01-02 03:04:05.000006",
    "2200-01-02 03:04:05.000006",
    "2300-01-02 03:04:05.000006",  # out of bounds
]
pt = pddt(dts)
print(pt)
```
```
0   2000-01-02 03:04:05.000006
1   2100-01-02 03:04:05.000006
2   2200-01-02 03:04:05.000006
3   2300-01-02 03:04:05.000006
dtype: datetime64[us]
```

##### Specify desired time unit resolution
Sometimes the initial data is alreay a <'pandas.Series'> but defaults to `datetime64[ns]`, <'pddt'> supports specifing the the desired time 'unit' so adding `delta` or manipulating `year` can be within bounds.
```python
from pandas import Series
from datetime import datetime

dts = [
    datetime(2000, 1, 2),
    datetime(2100, 1, 2),
    datetime(2200, 1, 2),
]
ser = Series(dts)  # Series defaults to 'datetime64[ns]'
pt = pddt(ser, unit="us")
```
```
0   2000-01-02
1   2100-01-02
2   2200-01-02
dtype: datetime64[us]
```

##### Direct assignment to DataFrame
```python
from pandas import DataFrame

df = DataFrame()
df["pddt"] = pt
print(df)
```
```
                        pddt
0 2000-01-02 03:04:05.000006
1 2100-01-02 03:04:05.000006
2 2200-01-02 03:04:05.000006
3 2300-01-02 03:04:05.000006
```

### Acknowledgements
cyTimes is based on several open-source repositories.
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)

cyTimes makes modification of the following open-source repositories:
- [dateutil](https://github.com/dateutil/dateutil)
    The class <'Parser'> and <'cytimedelta'> in this package is basically a cythonized version of <'dateutil.parser'> and <'dateutil.relativedelta'>. All credits go to the original authors and contributors of the `dateutil` library.