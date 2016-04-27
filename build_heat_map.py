import matplotlib.pyplot as plt
import cPickle as pickle
import seaborn as sns
import pandas as pd
import numpy as np
from sys import argv
from matplotlib import colors


def pitch_heatmap(data):

    #sns.palplot(sns.color_palette("hls", 8))

    #data = data.query('pitch == "fastball"')
    #pivot = data.pivot('loc','event','prob').reshape(2,5,5)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))

    #sns.heatmap(pivot)
    
    sns.heatmap(data.query('pitch == "fastball"')['event'].reshape(5, 5), cbar=False,\
                square=True, linewidths=0.5, xticklabels=False, yticklabels=False, ax=ax1, cmap="YlGnBu")
    ax1.set_title('Fastball')

    sns.heatmap(data.query('pitch == "changeup"')['event'].reshape(5, 5), cbar=False, \
                square=True, linewidths=0.5, xticklabels=False, yticklabels=False, ax=ax2,  cmap="YlGnBu")
    ax2.set_title('Changeup')
    sns.heatmap(data.query('pitch == "curveball"')['event'].reshape(5, 5), cbar=False, \
                square=True, linewidths=0.5, xticklabels=False, yticklabels=False, ax=ax3,  cmap="YlGnBu")
    ax3.set_title('Curveball')
    sns.heatmap(data.query('pitch == "slider"')['event'].reshape(5, 5), cbar=False, \
                square=True, linewidths=0.5, xticklabels=False, yticklabels=False, ax=ax4,  cmap="YlGnBu")
    ax4.set_title('Slider')
    
    plt.tight_layout()
    #plt.legend(frameon=True) 
    plt.show()
    


def makeX(csv):
    
    dfc = pd.read_csv(csv)
    #print dfc
    dfc.dropna(inplace=True)
    print 'dfc cols:', dfc.columns
    dfc.drop(['pdes_rev=ball','pdes_rev=call_strike', \
              'pdes_rev=contact', 'pdes_rev=swg_strike'], axis=1, inplace=True)
    #print 'cols: ',dfc.columns
    X = dfc.values
    zl = dfc.zone_location.values
    #print X
    return X


if __name__ == '__main__':
    #script, csv = argv
    #print 'working'

    #df = pd.read_csv(csv)
    X_cur = makeX('../data/curve_final.csv')
    X_chg = makeX('../data/change_final.csv')
    X_fst = makeX('../data/fastball_final.csv')
    X_sld = makeX('../data/slider_final.csv')
    
    with open('xbg_final.pkl') as f:
        xgb = pickle.load(f)
    slider = xgb.predict_proba(X_sld)
    changeup = xgb.predict_proba(X_chg)
    curveball = xgb.predict_proba(X_cur)
    fastball = xgb.predict_proba(X_fst)
    z_fst = 0
    z_sld = 0
    z_cur = 0
    z_chg = 0
    lst_fst = []
    lst_sld = []
    lst_chg = []
    lst_cur = []
    #print "fastball :",fastball
    
    #print probs
    
    for i in fastball:
        lst_fst.append(('fastball', z_fst, np.argmax(i), np.max(i)))
        z_fst += 1
    #print lst_fst[1]
    df_fst = pd.DataFrame(lst_fst)
    df_fst.columns = ['pitch', 'loc', 'event', 'prob']
    
  
    for i in slider:
        lst_sld.append(('slider', z_sld, np.argmax(i), np.max(i)))
        z_sld += 1
    df_sld = pd.DataFrame(lst_sld)
    df_sld.columns = ['pitch', 'loc', 'event', 'prob']
    #print lst_sld

    
    for i in curveball:
        lst_cur.append(('curveball', z_cur, np.argmax(i), np.max(i)))
        z_cur += 1
    df_cur = pd.DataFrame(lst_cur)
    df_cur.columns = ['pitch', 'loc', 'event', 'prob']
    #print df_cur
    print lst_cur
    
    for i in changeup:
        lst_chg.append(('changeup', z_chg, np.argmax(i), np.max(i)))
        z_chg += 1
    df_chg = pd.DataFrame(lst_chg)
    df_chg.columns = ['pitch', 'loc', 'event', 'prob']
    #print df_chg
    
    df = pd.concat([df_fst, df_chg, df_cur, df_sld], axis=0, ignore_index=True)
    df.to_csv('lookatthis.csv',index=False)
    #print 'df.shape ',df


    pitch_heatmap(df)
        
