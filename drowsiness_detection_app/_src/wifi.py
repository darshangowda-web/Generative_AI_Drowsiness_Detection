import gps
session = gps.gps(mode=gps.WATCH_ENABLE)
report = session.next()
if report['class'] == 'TPV':
    latitude = report.lat
    longitude = report.lon
    print(f"Latitude: {latitude}, Longitude: {longitude}")
