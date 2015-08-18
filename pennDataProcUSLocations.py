
import csv, gzip
import sys
import re
import numpy as np
from numpy import random

def genCountyCode(stateMap, cc2PopMap, state):
    total = 0
    for code in stateMap[state]:
        total = total + cc2PopMap[code]
    dist = []
    codes = []
    for code in stateMap[state]:
        dist.append(cc2PopMap[code]/total)
        codes.append(code)
    sample = random.multinomial(1, dist, size=1)[0]
    #print (sample)
    index = np.nonzero(sample)[0][0]
    #print (index)
    #print (codes[index])
    return codes[index]

def genCityCC(cityStateMap, cc2PopMap, city):
    total = 0
    dist = []
    codes = []
    #print ("key, pop, county code")
    for state in cityStateMap:
        key = state.split(',')[0]
        if city == key:
            county_code = cityStateMap[state]
            codes.append(county_code)
            pop =  cc2PopMap[county_code]
            dist.append(pop)
            total = total + pop
            #print (state, pop, county_code)
            
            
    if total == 0:
        return ""
    
    for i, item in enumerate(dist):
        dist[i] = dist[i]/total
    sample = random.multinomial(1, dist, size=1)[0]
    #print (sample)
    index = np.nonzero(sample)[0][0]
    #print (index)
    #print (codes[index])
    return codes[index]

csv.field_size_limit(sys.maxsize)
# processing the data output
i = 0
heading = []
code2cntyInfo = {}
for ll in csv.reader( open("hivRatesData/AIDSVu_County_2012.csv", 'r' )):
    if i == 0:
        heading = ll
    else:
        county_rate = int(ll[3])
        if (  county_rate > 0 ): 
            county_name = ll[2]
            state_name = ll[1]
            key = county_name + " " + state_name
            code = ll[0]
            if len (code) < 5:
                temp = ['0']*(5-len(code))
                code = "".join(temp + [code])
            code2cntyInfo[code] = {'rate':str(county_rate), 'name':key.strip()}
        
    i = i + 1

cityState2cc={}
state2codes={}
code2population={}
nz_count = 0
#with open ('pennData/population_usa_cities/all_2011_data.csv', encoding='latin-1') as f:
with open ('pennData/population_usa_cities/all_2011_data.csv') as f:
    datareader = csv.reader(f)
    for row in datareader:
        cc = ''.join(row[1:3])
        if (row[0] == "050"):
            nz_count = nz_count + 1
            code2population[cc] = int(row[9])
        
            
        if cc in code2cntyInfo :
            city = re.sub(r'\([^)]*\)', '', row[6].lower().strip())
            city = city.strip()
            c_name = city.replace("city", "").strip()
            state = row[7].lower()
            if state not in state2codes:
                state2codes[state] = set([])
            state2codes[state].add(cc)

            key = ','.join([c_name,state])
            #if key  in cityState2cc:
            cityState2cc[ key ] = cc

#example ussage:
#print (genCountyCode(state2codes, code2population, 'california'))
#print (genCityCC (cityState2cc, code2population, 'vandalia'))

count = 0
total = 0
#map counties to tweets
counties2tweets = {}
#with gzip.open('pennData/test_pennUSLocations.csv.gz', 'rt') as csvfile:
print "processing data..."
with gzip.open('pennData/test_pennUSLocations.csv.gz') as csvfile:
    datareader = csv.reader(csvfile)
    for row in datareader:
        total= total+1
        #case 1: it has both city and state
        if  not (row[2] == "") and not (row[3] == ""):
            twt_key = row[2]+','+row[3]
            if (twt_key in cityState2cc):
                county_code = cityState2cc[twt_key]
            else:
                county_code = ""
                
        #case 2: it only has state(no city)
        elif not (row[3] == ""):
            if row[3] in state2codes:
                county_code = genCountyCode(state2codes, code2population, row[3])
            else:
                county_code = ""
            
        #case 3: only has city (no state)
        elif not (row[2] == ""):
            county_code = genCityCC (cityState2cc, code2population, row[2])
            
        #if row[2] == row[3]:
        #    count = count + 1
        #break
        if not (county_code == ""):
            if county_code not in counties2tweets:
                counties2tweets[county_code] = []
            
            counties2tweets[county_code].append(row[1])
            count = count + 1

output_path = "temp_cities/"
print "writing tweets to" + output_path
for county_code in counties2tweets:
    output_temp = code2cntyInfo[county_code]['name'] + " " +code2cntyInfo[county_code]['rate']
    output_name = "_".join(output_temp.split())+".txt"
    #print (output_name)
    with open (output_path + output_name, 'w') as f:
        f.write('\n'.join(counties2tweets[county_code]))
