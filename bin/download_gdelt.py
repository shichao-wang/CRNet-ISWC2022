import datetime

try:
    import gdelt
except ImportError:
    raise ImportError(
        "We require `gdelt` package to get access to GDELT dataset"
    )

import datetimerange

# https://arxiv.org/pdf/1705.05742.pdf
DATE_RANGE = ["2015 Apr 1", "2016 Mar 31"]


def date_string(dt: datetime.datetime):
    return "%d %d %d" % (dt.year, dt.month, dt.day)


sdt = datetime.datetime(2015, 4, 1)
edt = datetime.datetime(2016, 5, 31)
dtr = datetimerange.DateTimeRange(sdt, edt)

gd = gdelt.gdelt()
date_time_list = list(dtr.range(days=30))
for s, e in zip(date_time_list, date_time_list[1:]):
    date_range = [date_string(s), date_string(e)]
    results = gd.Search(DATE_RANGE, coverage=True, output="json")
