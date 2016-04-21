''' Here is the example text of a single game of a single team from a single year...

info,hometeam,ANA         # if line[0] == 'info' and line[1] == 'hometeam': home_team.appened(line[2])
info,site,ANA01
info,date,2010/04/05      # I need  if line[0] == 'info' and line[1] == 'date': game_date.appened(line[2])
info,number,0
info,starttime,7:08PM     # I need    if line[0] == 'info' and line[1] == 'starttime': start_time.appened(line[2])
info,daynight,night     
info,usedh,true
info,umphome,mcclt901     # I need   if line[0] == 'info' and line[1] == 'umphome': plate_ump.appened(line[2])
info,ump1b,everm901
info,ump2b,fleta901
info,ump3b,johna901
info,howscored,park
info,pitches,pitches
info,temp,58              # I need   if line[0] == 'info' and line[1] == 'temp': temp.appened(line[2])
info,winddir,tocf
info,windspeed,11
info,fieldcond,unknown 
info,precip,unknown       # I need    if line[0] == 'info' and line[1] == 'precip': precip.appened(line[2])
info,sky,cloudy           # I need    if line[0] == 'info' and line[1] == 'sky': sky.appened(line[2])
'''


# later, will need to create categoricals for start_time (night/day) and precip(yes/no)

from collections import defaultdict
import datetime
from pymongo import MongoClient
import glob


def init_mongo_client():
    '''
    Start the Pymongo client and update game info for each file
    '''
    client = MongoClient()
    db = client.gameData_1
    table = db.game_data
    return table

gameData_1 = init_mongo_client()

path =  "/Users/MessyJesse/Desktop/pitch_FX_project/savant_game_logs_2010-2015/*"
for fname in glob.glob(path):

    with open(fname) as f:
        date = ''
        home_team = ''
        start_time = ''
        plate_ump = ''
        temp = ''
        precip = ''
        sky = ''
        for line in f:
            line = line.split(",")
            if line[0] == 'info' and line[1] == 'precip': precip = line[2].rstrip()
            elif line[0] == 'info' and line[1] == 'temp': temp = line[2].rstrip()
            elif line[0] == 'info' and line[1] == 'umphome': plate_ump = line[2].rstrip()
            elif line[0] == 'info' and line[1] == 'hometeam': home_team = line[2].rstrip()
            elif line[0] == 'info' and line[1] == 'sky':
                sky = line[2].rstrip()
                temp = {'date': date, 'home_team':home_team, 'plate_ump':plate_ump, 'sky':sky, 'temp':temp, 'precip':precip}
                gameData_1.insert_one(temp)
            elif line[0] == 'info' and line[1] == 'date':
                date = line[2].rstrip()
                date = datetime.datetime.strptime(date, '%Y/%m/%d').strftime('%y-%m-%d')
            elif line[0] == 'info' and line[1] == 'starttime': start_time = line[2].rstrip()
            else: pass

            #gameData[date] = (home_team, start_time, plate_ump, temp, sky, precip)

