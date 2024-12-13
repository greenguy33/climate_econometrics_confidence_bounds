import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.tsatools import add_lag
import math

# load Bayesian model

model = pd.read_pickle("models/kotz_reproduction_full_covar_coefs.pkl")

# load dataset

data = pd.read_stata("data/kotz_et_al_dataset.dta")
model_vars = [
    "dlgdp_pc_usd",
    "T5_varm",
    "T5_seas_diff_mXT5_varm",
    "T5_mean_diff",
    "T5_mean_mXT5_mean_diff",
    "T5_mean_diff_lag",
    "T5_mean_mXT5_mean_diff_lag",
    "P5_totalpr"
]
data["T5_mean_diff"] = data.groupby("ID")["T5_mean"].diff()
data["T5_mean_diff_lag"] = np.insert(add_lag(data["T5_mean_diff"])[:,1], 0, np.NaN)
t5_mean_diff_lag = []
last_region = ""
for row in data.itertuples():
    if row.ID != last_region:
        t5_mean_diff_lag.append(np.NaN)
    else:
        t5_mean_diff_lag.append(row.T5_mean_diff_lag)
    last_region = row.ID
data["T5_mean_diff_lag"] = t5_mean_diff_lag

data["T5_seas_diff_mXT5_varm"] = data["T5_seas_diff_m"] * data["T5_varm"]
data["T5_mean_mXT5_mean_diff"] = data["T5_mean_m"] * data["T5_mean_diff"]
data["T5_mean_mXT5_mean_diff_lag"] = data["T5_mean_m"] * data["T5_mean_diff_lag"]

data = data.dropna(subset=model_vars).reset_index(drop=True)

# unscale coefficients of interest

t5_coef = model["t5_varm"].data.flatten() * np.std(data.dlgdp_pc_usd) / np.std(data.T5_varm) * 100
t5_seas_diff_mXt5_varm_data = model["t5_seas_diff_mXt5_varm"].data.flatten() * np.std(data.dlgdp_pc_usd) / np.std(data.T5_seas_diff_mXT5_varm) * 100

# make figure 4 - this may take 30+ minutes
make_fig4 = False

if make_fig4:

	maskc=gpd.read_file('data/gadm36_levels.gpkg')
	mask=gpd.read_file('data/gadm36_levels.gpkg',layer=1)

	GID0list=pd.unique(mask.GID_0)
	isolist=pd.unique(data.iso)
	country_level_uncertainty = {}

	#empty list to hold values
	proj=[]

	#calculate marginal effects to plot 
	for i in range(len(GID0list)):
	    iso=GID0list[i]
	    no_regions=len(mask.loc[mask.GID_0==iso])
	    if iso in isolist:
	        country_level_uncertainty[iso] = []
	        wlrd1list=pd.unique(data.loc[data.iso==iso,'wrld1id_1'])
	        for j in range(no_regions):
	            reg_no=j+1
	            if reg_no in wlrd1list:    
	                T_diff=data.loc[(data.iso==iso) & (data.wrld1id_1==reg_no),'T5_seas_diff_m'].mean()
	                region_vals = sorted(t5_coef + t5_seas_diff_mXt5_varm_data*T_diff)
	                proj.append(region_vals)
	                country_level_uncertainty[iso].append(np.quantile(region_vals, .95) - np.quantile(region_vals, .05))
	            else:    
	                proj.append([np.nan]*len(t5_coef))
	    else:
	        for j in range(no_regions):
	            proj.append([np.nan]*len(t5_coef))

	mask['quant_5']=list(map(lambda x: np.quantile(x, .05), proj))
	mask['quant_95']=list(map(lambda x: np.quantile(x, .95), proj))
	mask["maximum_likelihood"]=list(map(lambda x: np.mean(x), proj))
	mask["diff"]=list(map(lambda x: (np.quantile(x, .95) - np.quantile(x, .05)), proj))

	mask['geometry_simpl']=mask.geometry.simplify(tolerance=720/43200,preserve_topology=True)
	mask_simp=mask.copy()
	mask_simp['geometry']=mask_simp.geometry.simplify(tolerance=720/43200,preserve_topology=True)

	def build_map(data_column, axis, vmin, vmax, cmap, label):
	    
	    degree_sign= u'\N{DEGREE SIGN}'
	    C=degree_sign + 'C'
	    
	    i=5
	    maskc.plot(ax=axis,edgecolor='black',linewidth=0.1,color='grey')
	    mask_simp[~mask_simp[data_column].isnull()].plot(ax=axis,column=data_column,cmap=cmap,vmin=vmin,vmax=vmax)
	    mask_simp[mask_simp[data_column].isnull()].plot(ax=axis,color='grey')
	    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin,vmax=vmax))
	    sm._A = []
	    cbar = fig.colorbar(sm,ax = axis,orientation='horizontal',fraction=0.046, pad=0.05)
	    cbar.ax.tick_params(labelsize=8)
	    axis.set_ylim([-60,85])
	    axis.set_xlim([-180,180])
	    axis.tick_params(labelsize='large')
	    axis.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
	    axis.set_title(label)
	    axis.title.set_size(15)

	fig, axes = plt.subplots(2,2, figsize=(12,12))
	fig.subplots_adjust(hspace=.3, wspace=0.0)

	label = [
	    'Point Estimate',
	    'Lower Bound Estimate (5th percentile)',
	    'Upper Bound Estimate (95th percentile)',
	    'Width of 90\% CI'
	]

	build_map("maximum_likelihood", axes[0][0], -12, 0, "RdYlBu", label[0])
	build_map("quant_5", axes[0][1], -12, 0, "RdYlBu", label[1])
	build_map("quant_95", axes[1][0], -12, 0, "RdYlBu", label[2])
	build_map("diff", axes[1][1], 0, 5, "RdYlBu_r", label[3])

	plt.savefig('figures/fig4.png',bbox_inches='tight',pad_inches=0)
	plt.close()

# make figure 5

pop_by_region = pd.read_csv("data/pop_by_region.csv")

regional_weights = {}
for country in set(data.iso):
    country_sum = 0
    regions = set(data.loc[data.iso == country]["ID"])
    for region in regions:
        country_sum += pop_by_region.loc[pop_by_region.ID == region]["x"].item()
    for region in regions:
        regional_weights[region] = pop_by_region.loc[pop_by_region.ID == region]["x"].item() / country_sum

country_averages = {}
for country in set(data.iso):
    country_averages[country] = []
    for region in pd.unique(data.loc[data.iso==country,'ID']):
        T_diff=data.loc[(data.iso==country) & (data.ID==region),'T5_seas_diff_m'].mean()
        region_vals = sorted(t5_coef + t5_seas_diff_mXt5_varm_data*T_diff)
        last_year_region_gdp = data.loc[(data.iso==country) & (data.ID==region),'lgdp_pc_usd'].iloc[-1]
        country_averages[country].append((np.array(region_vals) / 100) * math.exp(last_year_region_gdp) * regional_weights[region])
    country_averages[country] = np.sum(country_averages[country], axis=0)

sorted_country_averages = {key:val for key,val in sorted(country_averages.items(), key=lambda x : np.std(x[1]))}

indices = [0,1,2,3,4,5,6,7,8,9,10]
fig,axis = plt.subplots(1,1)
countries_to_include = ["IDN","CHN","GRC","USA"]
bars = []
for country in countries_to_include:
    bars.append(np.quantile(country_averages[country], .95))
    bars.append(np.quantile(country_averages[country], .05))
    bars.append(0)
    print(country)
    print(np.quantile(country_averages[country], .95))
    print(np.quantile(country_averages[country], .05))
bars = bars[:-1]
axis.bar(
    indices,
    [-1*val for val in bars],
    color=["red","blue","white"],
    linewidth=0,
    width=1
)
axis.set_xticks(indices,["IDN\n(low)","IDN\n(high)","","CHN\n(low)","CHN\n(high)", "", "GRC\n(low)","GRC\n(high)","","USA\n(low)","USA\n(high)"])
axis.set_xlabel("Country", size=15)
axis.set_ylabel("Per capita loss (USD)", size=15)

plt.savefig("figures/fig5.png", bbox_inches="tight")
