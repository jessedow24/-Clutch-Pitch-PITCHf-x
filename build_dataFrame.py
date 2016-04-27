

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
    
def see_one_game(df, gid):

    game = df[df.gid.isin([gid])]
    return game

def make_short(df):

    dfShort = df[['date_stamp','inning', 'ab_total', 'pitcher_id', 'pitcher_team','batter_id', 'batter_name',\
                  'ab_count', 'ab_id', 'id', 'des', 'strikes', 'balls', 'ballpark', 'pdes']]
     
    # get the game in sequence
    dfShort.sort_values(['date_stamp', 'ballpark', 'ab_id', 'ab_count'])
    return dfShort

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

def make_3_pitch():
    return make_df('../data/3_pitch_fx_cleaned_ballparks.csv')

def make_3_game():
    return make_df('../data/3_game_data_fixed_ballparks.csv')

def make_4_pitch():
    return make_df('../data/4_pitch_fx_merged_game_cleaned.csv')

def make_5_pitch():
    reutrn make_df('../data/5.5_pitch_fx_most_batter_names.csv')

if __name__ == "__main__":
    
    '''
    df = make_3_pitch()
    dg = make_3_game()

    merge = pd.merge(df,dg, howon=['ballpark', 'date'])

    merge.to_csv('merged.csv')
    '''
