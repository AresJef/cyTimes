## Easy management of python datetime & pandas time Series.

Created to be used in a project, this package is published to github 
for ease of management and installation across different modules.

### Features
Provides two classes for datetime and pandas time series management.
- `pydt` (Python Datetime)
- `pddt` (Pandas Time Series)

Both provide similar functionalities:
- Parse time string
- Access in different data types
- Conversion to `int/float` (ordinal, total_seconds, timestamp, etc.)
- Calender properties (days_in_month, weekday, etc.)
- Day manipulation (next_week, to_week, etc.)
- Month manipulation (next_month, to_month, etc.)
- Quarter manipulation (next_quarter, to_quarter, etc.)
- Year manipulation (next_year, to_year, etc.)
- Timezone manipulation (tz_localize, tz_convert, etc.)
- Frequency manipulation (round, ceil, floor, etc.)
- Delta adjustment (Equivalent to adding `relativedelta` and `pandas.DateOffset`)
- Replace adjustment (Equivalent to `datetime.replace` and custom `pandas.Series.replace`)

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
Only support for python 3.10 and above.

### Usage (pydt)
``` python
import datetime, numpy as np, pandas as pd
from cytimes import pydt

# Parse time string
pt = pydt('2021-01-01 00:00:00')
# 2021-01-01T00:00:00
pt = pydt("2021 Jan 1 11:11 AM")
# 2021-01-01T11:11:00

# dateimte/date/time
pt = pydt(datetime.datetime(2021, 1, 1, 0, 0, 0))
# 2021-01-01T00:00:00
pt = pydt(datetime.date(2021, 1, 1))
# 2021-01-01T00:00:00
pt = pydt(datetime.time(12, 0, 0)) # date defaults to today. Can change through default arugment.
# 2023-09-01T12:00:00

# pandas.Timestamp
pt = pydt(pd.Timestamp("2021-01-01 00:00:00"))
# 2021-01-01T00:00:00

# numpy.datetime64
pt = pydt(np.datetime64("2021-01-01 00:00:00"))
# 2021-01-01T00:00:00

# Access in different data types
pt.dt # -> datetime.datetime
pt.date # -> datetime.date
pt.time # -> datetime.time
pt.timetz # -> datetime.time with timezone
pt.ts # -> pandas.Timestamp
pt.dt64 # -> numpy.datetime64
...

# Conversion to int/float
pt.ordinal # -> int
pt.timestamp # -> float
...

# Calender properties
pt.is_leap_year # -> bool
pt.days_bf_year # -> int
pt.days_in_month # -> int
pt.weekday # -> int
pt.isocalendar # -> tuple

# Day manipulation
pt.monday # -> pydt (monday of the week)
pt.tuesday # -> pydt (tuesday of the week)
pt.next_week("monday") # -> pydt (next monday)
pt.to_week(3, "Mon") # -> pydt (three weeks later monday )
...

# Month manipulation
pt.month_lst # -> pydt (last day of the month)
pt.next_month(3) # -> pydt (next month 3rd day)
pt.to_month(3, 31) # -> pydt (three months later last day)
...

# Quarter manipulation
pt.quarter_1st # -> pydt (first day of the quarter)
pt.next_quarter(2, 10) # -> pydt (next quarter 2nd month 10th day)
pt.to_quarter(2, 10, 15) # -> pydt (two quarters later 2nd month 15th day)
...

# Year manipulation
pt.year_lst # -> pydt (last day of the year)
pt.next_year("jan", 1) # -> pydt (next year jan 1st)
pt.to_year(2, "feb", 29) # -> pydt (two years later feb last day)
...

# Timezone manipulation
pt.tz_localize("UTC") # -> pydt (UTC time)
pt.tz_switch(targ_tz="CET", base_tz="UTC") # -> pydt (Setting base timezone to UTC and convert to CET)
...

# Frequency manipulation
pt.round("H") # -> pydt (round to hour)
...

# Delta adjustment
pt.delta(days=1) # -> pydt (add one day)

# Replace adjustment
pt.replace(year=2022) # -> pydt (replace year to 2022)
```

### Usage (pddt)
`pddt` accepts `list` and `pandas.Series` as argument instead of  `str`/`datetime` 
comparing to `pydt`. Properties and methods are similar to `pydt`, except `pddt` 
is designed to work with `pandas.Series[datetime64]`.

### Acknowledgements
cyTimes is based on several open-source repositories.
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)

cyTimes makes modification of the following open-source repositories:
- [dateutil](https://github.com/dateutil/dateutil)

This package created a Cythonized version of dateutil.parser (cyparser) and
dateutil.relativedelta (cytimedelta). As a result, these two modules in
this package have sacrificed flexibility and readability in exchange for
enhancements in performance. All credits go to the original authors and
contributors of dateutil.
