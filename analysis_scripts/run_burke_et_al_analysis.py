import pandas as pd
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import seaborn as sns
import geopandas
import folium

# load Bayesian model

burke_model = pd.read_pickle('models/burke_et_all_model_full_covar_coefs.pkl')

# load original dataset

data = pd.read_csv("data/burke_et_al_dataset.csv")
missing_indices = []
no_nan_cols = ["UDel_temp_popweight","UDel_precip_popweight","growthWDI"]
for index, row in enumerate(data.itertuples()):
    if any(np.isnan(getattr(row,col)) for col in no_nan_cols):
        missing_indices.append(index)
data = data.drop(missing_indices).reset_index(drop=True)

# unscale coefficients

burke_coef1_full = (burke_model["burke_coef1_full_scaled"] * np.std(data.growthWDI) / np.std(data.UDel_temp_popweight)) - (2 * ((burke_model["burke_coef2_full_scaled"] * np.mean(data.UDel_temp_popweight) * np.std(data.growthWDI)) / np.square(np.std(data.UDel_temp_popweight))))
burke_coef2_full = (burke_model["burke_coef2_full_scaled"] * np.std(data.growthWDI) / np.square(np.std(data.UDel_temp_popweight)))

# generate values for temperature threshold from posterior samples

numerator = [val for val in burke_coef1_full]
denominator = [-2*(val) for val in burke_coef2_full]
threshold = np.array(numerator) / np.array(denominator)
bayes_interval = np.quantile(sorted(threshold),[.05,.95])

print("Threshold distribution mean:", np.mean(threshold))
print("Threshold distribution lower bound (5th percentile):", bayes_interval[0])
print("Threshold distribution upper bound (95th percentile):", bayes_interval[1])

# make figure 1

def make_hist_plot(axis, data, color, labels, show_hist=True):
    x = np.linspace(np.mean(data) - 3*np.std(data), np.mean(data) + 3*np.std(data), 100)
    _, bins, patches = axis.hist(data, bins=200, density=True)
    sns.kdeplot(data, ax=axis, color="black")
    axis.set_xlim(min(x)-1.5, max(x)+1.5)
    axis.xaxis.label.set_size(20)
    axis.yaxis.label.set_size(20)
    axis.xaxis.set_tick_params(labelsize=15)
    axis.yaxis.set_tick_params(labelsize=15)

    q1 = np.quantile(data, .05)
    q2 = np.quantile(data, .95)
    
    axis.axvline(x = np.mean(data), color = 'r', lw = 4, label = labels[0])
    axis.axvline(x = q1, color = 'dodgerblue', lw = 4, label = labels[1])
    axis.axvline(x = q2, color = 'dodgerblue', lw = 4, label = labels[2])
    axis.axvline(x = 13.06, color = 'green', lw = 4, label = labels[2])
    axis.set_xlabel("Vertex (°C)", weight="bold")
    axis.set_ylabel("Probability Density", weight="bold")
    axis.set_title(labels[4], weight="bold")
    axis.title.set_size(20)

    for index, bin in enumerate(bins):
        if index != len(bins)-1:
            if bin < q1 or bin > q2:
                patches[index].set_facecolor("gray")
            else:
                patches[index].set_facecolor("orange")

fig, axis = plt.subplots()
make_hist_plot(axis, threshold, "orange",
    labels=[
    'Bayesian Inference Vertex Mean', 'Bayesian Inference Vertex Lower Bound',
    'Bayesian Inference Vertex Upper Bound', 'Original Burke et al. Estimate',
    'Burke et al. Temp/GDP Threshold'
])

plt.savefig("figures/fig2.png", bbox_inches='tight')
plt.close()

# make figure 2

temp_projections = pd.read_csv("data/cmip6-x0.25_timeseries_tas_timeseries-tas-annual-mean_annual_2015-2100_median_ssp119_ensemble_all_mean - all.csv")
countries_at_risk = {"country":[],"bayes":[],"point_est":[]}

bayes_countries, point_est_countries = set(), set()

for row in temp_projections.iterrows():
    row = row[1]
    country = row.code
    if "." not in country:
        temp_curr = row["2024-07"]
        for year in range(2025,2101):
            temp_fut = row[f"{year}-07"]
            if temp_fut < bayes_interval[1] and temp_fut > bayes_interval[0]:
                bayes_countries.add(country)
            if temp_curr < 13.06 and temp_fut > 13.06:
                point_est_countries.add(country)
            
for country in set(temp_projections.code):
    countries_at_risk["country"].append(country)
    if country in bayes_countries:
        countries_at_risk["bayes"].append(1)
    else:
        countries_at_risk["bayes"].append(0)
    if country in point_est_countries:
        countries_at_risk["point_est"].append(1)
    else:
        countries_at_risk["point_est"].append(0)

countries_at_risk_df = pd.DataFrame.from_dict(countries_at_risk)

country_geopandas = geopandas.read_file(
    geopandas.datasets.get_path('naturalearth_lowres')
)
country_geopandas = country_geopandas.merge(
    countries_at_risk_df,
    how='inner', 
    left_on=['iso_a3'],
    right_on=['country']
)

fig, axes = plt.subplots(2,1, figsize=(8,8))
fig.subplots_adjust(hspace=.3, wspace=0.0)
color_list = ["gray","red"]
custom_cmap = clr.ListedColormap(color_list)
axis_map = {0:axes[0], 1:axes[1]} 
title_map = {"point_est":"Point Estimate","bayes":"90% Confidence Interval","bootstrap":"Block Bootstrap","delta":"Delta Method"}

for index, technique in enumerate(["point_est","bayes"]):
    country_geopandas.plot(column=technique, cmap=custom_cmap, ax=axis_map[index])
    axis_map[index].set_title(title_map[technique], weight="bold")
    axis_map[index].title.set_size(20)

plt.savefig("figures/fig3.eps", bbox_inches='tight', format="eps", dpi=1000)
plt.close()