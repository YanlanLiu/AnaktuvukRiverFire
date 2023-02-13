import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
import os
arrayid = int(os.environ['SLURM_ARRAY_TASK_ID'])

growth_type = ['FOR', 'GRA', 'LIC', 'LIV', 'MOS', 'SHD', 'SSD', 'SSE', 'SUB'] 
def plot_LC(table,year):
    lat_values = np.arange(68.8,69.5,0.2)
    lat_idx = [np.argmin(np.abs(table.index.values[::-1]-itm)) for itm in lat_values]
    long_values = np.arange(-151,-149.9,0.2)
    long_idx = [np.argmin(np.abs(np.array(list(table))-itm)) for itm in long_values]
    long_lable = [int(itm*10)/10 for itm in long_values]
    cmap = plt.cm.Set1 #plt.cm.terrain
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.arange(0.5,len(growth_type)+1,1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    plt.figure(figsize=(11,6.5))
    plt.imshow(np.flipud(table),cmap=cmap,norm=norm)
    plt.xticks(long_idx,labels=long_lable)
    plt.yticks(lat_idx,labels=lat_values)
    cbar = plt.colorbar(ticks=np.arange(1,11))
    cbar.ax.set_yticklabels(growth_type) 
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(year)
    plt.savefig('Figures/LC_'+str(year)+'.png',dpi=300,bbox_inches='tight')



unique_year = np.arange(2005,2022)
for year in unique_year[arrayid:arrayid+1]:
    print(year)
    LC = pd.read_csv('LC7/LC_'+str(year)+'.csv')
    z = np.zeros(len(LC))
    for i,typ in enumerate(growth_type):
        z[LC['LC']==typ] = i+1
    LC['ID'] = z.copy()

    table = pd.pivot_table(LC, values='ID', index='latitude',columns='longitude',fill_value=0)
    plot_LC(table,year)



