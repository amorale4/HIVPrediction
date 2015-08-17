import gzip
import csv
import pickle
import re

s = """Alabama - AL
Alaska - AK
Arizona - AZ
Arkansas - AR
California - CA
Colorado - CO
Connecticut - CT
Delaware - DE
Florida - FL
Georgia - GA
Hawaii - HI
Idaho - ID
Illinois - IL
Indiana - IN
Iowa - IA
Kansas - KS
Kentucky - KY
Louisiana - LA
Maine - ME
Maryland - MD
Massachusetts - MA
Michigan - MI
Minnesota - MN
Mississippi - MS
Missouri - MO
Montana - MT
Nebraska - NE
Nevada - NV
New Hampshire - NH
New Jersey - NJ
New Mexico - NM
New York - NY
North Carolina - NC
North Dakota - ND
Ohio - OH
Oklahoma - OK
Oregon - OR
Pennsylvania - PA
Rhode Island - RI
South Carolina - SC
South Dakota - SD
Tennessee - TN
Texas - TX
Utah - UT
Vermont - VT
Virginia - VA
Washington - WA
West Virginia - WV
Wisconsin - WI
Wyoming - WY"""

abriv = {}
for word in s.replace("-", " " ).split('\n'):
    #print word
    l_words = word.lower().split()
    abriv[l_words[-1]]  = " ".join(l_words[:-1])

abriv['dc'] = 'district of columbia'
merica = ['america', 'merica', 'us', 'usa', 'us of a', 'the states', 'united states of america' ]
with open ('state2city_data.pickle', 'rb') as f:
    state2city = pickle.load(f)

states = state2city.keys()
timezones = ['Eastern Time (US & Canada)', 'Central Time (US & Canada)', 'Mountain Time (US & Canada)'
             , 'Pacific Time (US & Canada)','Alaska', 'Hawaii']
pop_locations = {'nyc': '"new york city","new york"', 'la': '"los angeles","california"', }
topCanCities = set([])
topCanMun = set([])
temp_count = 0
with open ('Canada_cities_pop.tsv') as csvfile:
    for row in csv.reader(csvfile, delimiter='\t'):
        #print (row)
        can_loc = row[1].lower()
        topCanCities.add(re.sub(r'\([^)]*\)', '', can_loc).replace("city", "").strip())
        can_loc = row[2].lower()
        topCanMun.add(re.sub(r'\([^)]*\)', '', can_loc).replace("city", "").strip())

# 0  - "id",
# 1  - "message_id",
# 2  - "term",
# 3  - "category",
# 4  - "message",
# 5  - "created_time",
# 6  - "coordinates",
# 7  - "coordinates_state",
# 8  - "coordinates_address",
# 9  - "from_id",
# 10 - "in_reply_to_message_id",
# 11 - "in_reply_to_from_id",
# 12 - "retweet_message_id",
# 13 - "location",
# 14 - "friend_count",
# 15 - "followers_count",
# 16 - "time_zone",
# 17 - "lang",
# 18 - "has_hashtag"
#ids = []
#tweets = 0
#row_count = 0

with gzip.open('all_data.csv.gz', 'rb') as csvfile:
    datareader = csv.reader(csvfile)
    for row in datareader:
        #row_count = row_count + 1
        msg_id = row[0].replace("'", "")
        #print row
        if '.' not in msg_id:
            twt_location = row[13].strip()
            #need to check for null only in the small dataset remove in large dataset
            if twt_location != 'NULL' and twt_location != '""' :
                #tweets = tweets + 1
                twt_msg = row[4].replace('"', '')
                #print '"' + msg_id + '","' + row [4] + '","' + twt_location+'"'
                twt_location = twt_location.replace('.', '')
                original_location = twt_location.replace('"', '')
                twt_location = re.sub(r'\([^)]*\)', '', twt_location)
                s_location = twt_location.lower().split(',')
                loc_len = len(s_location)
                twt_location = twt_location.lower()
                
                # pre-emtive checks 
                # 1 - does it belong in an US timezone?
                if row[16] not in timezones:
                    continue
                
                # 2 - does it belong in on of the top 100 largest cities or province
                if twt_location in topCanMun or twt_location in topCanCities:
                    continue
                
                #three cases 
                #case 1 
                if loc_len == 1:
                    # it could be city, state or country
                    if twt_location in merica:
                        print '"' + msg_id + '","' + twt_msg + '","' +'","'+'","' + original_location + '"'
                        #found = found + 1
                    
                    elif twt_location in pop_locations:
                        print '"' + msg_id + '","' + twt_msg  + '",' + pop_locations[twt_location] +',"' + original_location + '"'
                        #found = found + 1
                        
                    elif twt_location in abriv:
                        print '"' + msg_id + '","' + twt_msg + '","' +'","' + abriv[twt_location]+'","' + original_location + '"'
                        #found = found + 1
                    elif twt_location in states:
                        print '"' + msg_id + '","' + twt_msg + '","' +'","' + twt_location+'","' + original_location + '"'
                        #found = found + 1
                    #do not save the cities alone since there will 
                    else:
                        for state in state2city:
                            if twt_location in state2city[state]:
                                print '"' + msg_id + '","' + twt_msg + '","'+ twt_location +'","'+'","' + original_location + '"'
                                #print row[16]
                                #found = found + 1
                                break
                            
                    #if not found:
                    #    print (s_location)
                elif loc_len == 2:
                    #check for city, state or state, country
                    my_loc1 = s_location[-1].strip()
                    my_loc0 = " ".join(s_location[:-1]).strip().replace("city", "").strip()
                    #print my_city
                    if my_loc1 in abriv:
                        #if my_sate
                        if my_loc0 in  state2city[ abriv[my_loc1]] :
                            print '"' + msg_id + '","' + twt_msg + '","'+ my_loc0 +'","'+ abriv[my_loc1]+'","' + original_location + '"'
                            #found = found + 1
                    elif my_loc1 in states:
                        #print "state: " + my_state
                        if my_loc0 in  state2city[my_loc1]:
                            print '"' + msg_id + '","' + twt_msg + '","'+ my_loc0 +'","'+ my_loc1+'","' + original_location + '"'
                            #found = found + 1
                    elif my_loc1 in merica:
                        if my_loc0 in states:
                            print '"' + msg_id + '","' + twt_msg + '","'+'","'+ my_loc0+'","' + original_location + '"'
                            #found = found + 1
                            
                    #else:
                    #    print (s_location)
                        #print twt_location
                elif loc_len == 3:
                    #print (s_location)
                    if s_location[-1].strip() in merica:
                        my_state = s_location[-2].strip()
                        my_city = " ".join(s_location[:-2]).strip().replace("city", "").strip()
                        if my_state in abriv:
                            if my_city in state2city[ abriv[my_state] ]:
                                print '"' + msg_id + '","' + twt_msg + '","'+my_city+'","'+ abriv[my_state]+'","' + original_location + '"'
                                #found = found + 1
                                #print "found"
                                
                        if my_state in states:
                            if my_city in state2city[my_state]:
                                print '"' + msg_id + '","' + twt_msg + '","'+my_city+'","'+ my_state+'","' + original_location + '"'
                                #found = found + 1
                                #print "found"
                    #else:
                    #print(s_location)
                    #check for city, state, and country
                
#        if tweets > 5000:
#            #print row_count, tweets
#            break
        
