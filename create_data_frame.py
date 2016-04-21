import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def make_df(csv):
    df = pd.read_csv(csv)
    #df['ballpark']=df.park_sv_id.str[-3:]

    # remove bad data: 2 bad games at the beginning of 2010, and a bad date stamp
    try:
        df = df[df.gid != 'gid_2010_04_04_nyamlb_bosmlb_1/']
        df = df[df.gid != 'gid_2010_04_04_seamlb_sfnmlb_1/']
        df = df[df.date_stamp != '0000-00-00']
    except Exception:
        pass
    return df
    
def convert_row_to_df(row): 
    pitch_count = pd.DataFrame(pd.Series(np.arange(1, row['pitch_count'] + 1)))
    pitch_count.columns = ['pitch_count']
    pitch_count['gid'] = row['gid']
    pitch_count['pitcher_id'] = row['pitcher_id']
    pitch_count['inning'] = row['inning']
    return pitch_count

def add_pitch_count_inning(df):
    
    # create the group-by object which is how many pitches a pitcher threw in a game
    group = pd.DataFrame(df.groupby(['gid','pitcher_id', 'inning']).size())
    group.columns = ['pitch_count']
    group.reset_index(inplace=True)

    pitch_counts = pd.concat([convert_row_to_df(group.ix[row])\
                              for row in xrange(group.shape[0])])

    df.sort_values(['gid', 'pitcher_id', 'inning'], inplace=True)
    pitch_counts.sort_values(['gid', 'pitcher_id', 'inning'], inplace=True)
    df.reset_index(inplace=True)
    pitch_counts.reset_index(inplace=True)

    # this is where pitch counts is appended to the side of the \
    # original df to create 1_pitch_fx_with_pitch_counts
    df['pitch_count_inning'] = pitch_counts['pitch_count']
    return df

def cat_pitches(df):
    fastball = ['FF','FT', 'FA', 'FC', 'FS', 'SI']
    slider = ['SL']
    curveball = ['CU']
    changeup = ['CH']
    drop = ['EP', 'KC', 'KN', 'AS', 'AB', 'UN', 'SC', 'PO', 'IN', 'FO']
    
    df['fastball'] = 0
    df['slider'] = 0
    df['curve'] = 0
    df['changeup'] = 0
    df['drop'] = False
    
    df['slider'][df.mlbam_pitch_name.isin(slider)] = 1
    df['fastball'][df.mlbam_pitch_name.isin(fastball)] = 1
    df['curve'][df.mlbam_pitch_name.isin(curveball)] = 1
    df['changeup'][df.mlbam_pitch_name.isin(changeup)] = 1
    df['drop'][df.mlbam_pitch_name.isin(drop)] = True

    return df

def clean_pitches(df):
    # drop pitches
    
    dfc = df.copy()
    dfc = df[df_pitch_cats['drop'].isin([False])]
    return dfc

def cat_result(df):
    # bin the results of each pitch
    
    ball = ['Ball', 'Ball In Dirt', 'Hit By Pitch']
    swg_strike = ['Swinging Strike', 'Foul Tip', 'Swinging Strike (Blocked)', \
                  'Foul Bunt', 'Missed Bunt']
    call_strike = ['Called Strike']
    contact = ['Foul', 'In play, out(s)', 'In play, no out', 'In play, run(s)', \
               'Foul (Runner Going)']
    drop = ['Intent Ball', 'Pitchout', 'Swinging Pitchout', 'Unknown Strike']

    df['swg_strike'] = 0
    df['call_strike'] = 0
    df['contact'] = 0
    df['ball'] = 0
    df['dropPit'] = False

    df['swg_strike'][df.pdes.isin(swg_strike)] = 1
    df['call_strike'][df.pdes.isin(call_strike)] = 1
    df['contact'][df.pdes.isin(contact)] = 1
    df['ball'][df.pdes.isin(ball)] = 1
    df['dropPit'][df.pdes.isin(drop)] = True

    return df

def make_ballpark(df):
    #make a 3-dgit for ballparks and drop ~30K  weird values
    
    df['ballpark'] = df.park_sv_id.str[-3:]

    vc = df.ballpark.value_counts()
    lst = ['tex', 'bos', 'sln', 'tba', 'nyn', 'kca', 'det', 'sfn', 'cin', 'tor',\
            'lan', 'cle', 'chn', 'phi', 'mil', 'ari', 'cha', 'oak', 'min', 'sdn',\
            'bal', 'atl', 'hou', 'nya', 'was', 'col', 'sea', 'ana', 'pit', 'mia', \
            'flo']

    df2 = df[df.ballpark.isin(lst)]

    return df2

def pitcher_home(df):
    # create a new bool col for if pitcher is home or not
    df['pitcher_home'] = False
    df['pitcher_home'][df.pitcher_team == df.ballpark] = True
    return df

def handedness(df):
    # create a new bool col for if pitcher is same handed as batter
    df['same_hand'] = False
    df['same_hand'][df.stand == df.p_throws] = True
    return df

def add_weather(df, dfw):
    # merge weather variables: sky, temp, wet?

    #dfw.drop(['Unnamed: 0', '_id', 'plate_ump'], axis=1, inplace=True)
    dfw.home_team = dfw.home_team.str.lower()
    wet = ['rain', 'drizzle']
    dfw['wet'] = False
    dfw['wet'][dfw.precip.isin(wet)] = True

    # prepare for merge by fixing datetime columns
    date = dfw.date
    dfw['date_stamp'] = pd.to_datetime(date, yearfirst=True)

    dfw.drop(['date','precip'], axis=1, inplace=True)

    # allign ballpark name

    dfw.rename(columns={'home_team':'ballpark'},inplace=True)

    #git rid of double-headers, which confuses the merge (assume weather data same for both)
    dfw.drop_duplicates(subset=['date_stamp', 'ballpark'],inplace=True)

    # allign date format
    date2 = df.date_stamp
    df['date_stamp'] = pd.to_datetime(date2)

    # now, merge on date and ballpark
    df_wet = pd.merge(df, dfw, how='left', on=['date_stamp', 'ballpark'])
  
    return df_wet

def add_sky_dummeis(df):
    sunny = ['sunny', 'unknown']
    cloudy = ['cloudy']
    overcast = ['overcast']
    dome =['dome']

    df['cloudy'] = 0
    df['overcast'] = 0
    df['dome'] = 0
    df['sunny'] = 0

    df['sunny'][ultra.sky.isin(sunny)] = 1
    df['cloudy'][ultra.sky.isin(cloudy)] = 1
    df['overcast'][ultra.sky.isin(overcast)] = 1
    df['dome'][ultra.sky.isin(dome)] = 1

def add_ops(df, dfp):
    # Add the OPS values from batters

    #get dfb ready for merge..
    dfp.drop('PLAYERNAME', axis=1, inplace=True)

    #for the merge, we only need 1 OPS val per day
    dfp.drop_duplicates(subset=['datetime','MLBID'], inplace=True)
    dfp_drop.rename(columns={'datetime':'date_stamp'},inplace=True)

    # prep df for merge also
    id = df.batter_id.astype(float)
    df['MLBID'] = id

    #left outer join (merge)

    df_ops = pd.merge(df, dfp, how='left', on=['MLBID', 'date_stamp'])
    
def shorten(df):
    lst = ['Unnamed: 0','Unnamed: 0.1','ab_count','ab_total',
           'balls','batter_id',
           'des','drop','dropPit','ftime','gid','id','index','level_0',
           'p_throws','park_sv_id','pfx_xdatafile','pfx_zdatafile','pitch_con',
           'pitcher_team','pzold','sb','stand','strikes',
           'tstart','type','uncorrected_pfx_x','uncorrected_pfx_z','vy0','vystart', 
           'x0','y0','z0']
    return df.drop(lst, axis=1)
    

def clean_up(df):
    lst = ['MLBID','Unnamed: 0','Unnamed: 0.1','ab_count','ab_id','ab_total',
           'ballpark','balls','batter_id','date_stamp',
           'des','drop','dropPit','ftime','gid','id','index','level_0','mlbam_pitch_name',
           'p_throws','park_sv_id','pdes','pfx_xdatafile','pfx_zdatafile','pitch_con',
           'pitcher_id','pitcher_team','pzold','sb','stand','strikes',
           'tstart','type','uncorrected_pfx_x','uncorrected_pfx_z','vy0','vystart', 
           'x0','y0','z0']
    return df.drop(lst, axis=1)

    

    
if __name__ == "__main__":
    
    pass
    #designed to be imported as a module
