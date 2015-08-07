import sys
from urllib.error import HTTPError 
from geopy.geocoders import Nominatim
from geopy.point import Point
geolocator = Nominatim(timeout=30, country_bias="us")


location = geolocator.reverse("46.30221, -92.5222259")
print(location.address)
print((location.latitude, location.longitude))
print (location.raw)
temp = location.raw

code2locMap = {}
try:
    with open ( 'weatherData/ghcnd-stations.txt', 'r') as f:
        for i, line in enumerate(f):
            if line[0:2] == "US":
                s_line = line.split()
                #print(s_line)
                code = s_line[0]
                location = geolocator.reverse(",".join(s_line[1:3]))
                #print( i, location.address )
                temp = location.raw
                county = temp['address']['county'] if ('county' in temp['address']) else ''
                state = temp ['address']['state']  if ('state' in temp['address']) else ''
                code2locMap [ code ] = "," + county + "," + state + ",\"" + location.address  + "\""
                #print (code2locMap[code])
                #break
except HTTPError :
    print (i)
finally:
    with open( 'weatherData/code2locMap.csv', 'w') as f:
        for code in code2locMap:
            f.write(code + code2locMap[code] + "\n")
        
