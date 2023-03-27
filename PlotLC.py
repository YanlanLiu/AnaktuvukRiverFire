import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
import os
arrayid = int(os.environ['SLURM_ARRAY_TASK_ID'])
#arrayid = 0
#growth_type = ['FOR', 'GRA', 'LIC', 'LIV', 'MOS', 'SHD', 'SSD', 'SSE', 'SUB'] 
#growth_type = ['GRA','SHD','SHE', 'MOS', 'SUB', 'LIV', 'FOR']
growth_type = ['SUB','GRA','FOR','SSE','LIV','SSD','MOS']
def plot_LC(table,tag,year):
    lat_values = np.arange(68.8,69.5,0.2)
    lat_idx = [np.argmin(np.abs(table.index.values[::-1]-itm)) for itm in lat_values]
    long_values = np.arange(-151,-149.9,0.2)
    long_idx = [np.argmin(np.abs(np.array(list(table))-itm)) for itm in long_values]
    long_lable = [int(itm*10)/10 for itm in long_values]

    plt.figure(figsize=(11,6.5))

    if 'LC' in tag:
        cmap = plt.cm.Set1 #plt.cm.terrain
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
        bounds = np.arange(0.5,len(growth_type)+1,1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        plt.imshow(np.flipud(table),cmap=cmap,norm=norm)
        cbar = plt.colorbar(ticks=np.arange(1,len(growth_type)+1))
        cbar.ax.set_yticklabels(growth_type)
    else:
        plt.imshow(np.flipud(table),vmin=0,vmax=1)
        plt.colorbar()
    plt.xticks(long_idx,labels=long_lable)
    plt.yticks(lat_idx,labels=lat_values)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(tag +', '+str(year))
    plt.savefig('Figures7_tr/'+tag+str(year)+'.png',dpi=300,bbox_inches='tight')



unique_year = np.arange(2005,2021)
for year in unique_year[arrayid:arrayid+1]:
    print(year)
    df_lc = pd.read_csv('LC7_tr/LC_'+str(year)+'.csv')
    Prob = df_lc[growth_type].values
    df_lc['LC1'] = np.argmax(Prob,axis=1)+1 # landcover type with the highest likelihood, same as np.argsort(Prob,axis=1)[:,-1]
    df_lc['LC2'] = np.argsort(Prob,axis=1)[:,-2]+1 # landcover type with the second highest likelihood

    df_lc['P_SSD'] = Prob[:,growth_type.index('SSD')]
    df_lc['P_SSE'] = Prob[:,growth_type.index('SSE')]

    table = pd.pivot_table(df_lc, values='P_SSD', index='latitude',columns='longitude',fill_value=0)
    plot_LC(table,'P_SSD',year)
    
    table = pd.pivot_table(df_lc, values='LC1', index='latitude',columns='longitude',fill_value=0)
    plot_LC(table,'LC1',year)




