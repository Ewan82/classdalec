import datetime
import numpy as np
import netCDF4 as nc
import ephem

#event_time is just a date time corresponding to an sql timestamp
def type_of_light(latitude, longitude, event_time, utc_time, horizon):

    o = ephem.Observer()
    o.lat, o.long, o.date, o.horizon = latitude, longitude, event_time, horizon

    print "event date ", o.date

    print "prev rising: ", o.previous_rising(ephem.Sun())
    print "prev setting: ", o.previous_setting(ephem.Sun())
    print "next rise: ", o.next_rising(ephem.Sun())
    print "next set: ", o.next_setting(ephem.Sun())


    if o.previous_rising(ephem.Sun()) > o.previous_setting(ephem.Sun()):
        return "day"
    else:
        return "night"

def ah_day_night(event_time, horizon='0'):
    """calculate day and night for alice holt straits
    """
    time_str = event_time.strftime("%Y/%m/%d %H:%M")
    light = type_of_light('51.153526','0.858348', time_str,'0', horizon)
    if light=="day":
        return 1
    else:
        return 0