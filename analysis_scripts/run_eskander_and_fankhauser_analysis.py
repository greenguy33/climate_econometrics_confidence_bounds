import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.tsatools import add_lag
import random
import seaborn as sns

# load dataset

data = pd.read_stata("data/eskander_and_fankhauser_dataset2.dta")

data["lngdp2"] = np.square(data["lngdp"])
lagged_vars = ["slaws_mit_l3","slaws_mit_lt","rle1","gdp_hp","import_share","service_share","dtemp","federal2","lngdp","lngdp2"]
for var in lagged_vars:
    data[f"{var}_lag"] = np.insert(add_lag(data[var])[:,1], 0, np.NaN)
data["year"] = [int(str(val).split("-")[0]) for val in data["year"]]
data = data.drop(data[data["year"] < 1999].index).reset_index(drop=True)

# load Bayesian models

co2_model = pd.read_pickle("models/eskander_and_fankhauser_co2_model_full_covar_coefs.pkl")
bayes_co2_coef1 = co2_model["slaws_mit_l3_lag"].data.flatten() * np.std(data["lnco2"]) / np.std(data["slaws_mit_l3_lag"])
bayes_co2_coef2 = co2_model["slaws_mit_lt_lag"].data.flatten() * np.std(data["lnco2"]) / np.std(data["slaws_mit_lt_lag"])

ghg_model = pd.read_pickle("models/eskander_and_fankhauser_ghg_model_full_covar_coefs.pkl")
bayes_ghg_coef1 = ghg_model["slaws_mit_l3_lag"].data.flatten() * np.std(data["lnghg"]) / np.std(data["slaws_mit_l3_lag"])
bayes_ghg_coef2 = ghg_model["slaws_mit_lt_lag"].data.flatten() * np.std(data["lnghg"]) / np.std(data["slaws_mit_lt_lag"])

# plot figure 5

ghg_no_laws_data, co2_no_laws_data = [], []

co2_year_mean = data.groupby('year')['co2'].sum()
ghg_year_mean = data.groupby('year')['ghg'].sum()

co2_mean_term1 = np.mean(bayes_co2_coef1) * data["slaws_mit_l3_lag"]
co2_mean_term2 = np.mean(bayes_co2_coef2) * data["slaws_mit_lt_lag"]

ghg_mean_term1 = np.mean(bayes_ghg_coef1) * data["slaws_mit_l3_lag"]
ghg_mean_term2 = np.mean(bayes_ghg_coef2) * data["slaws_mit_lt_lag"]

ghg_no_laws_year_mean_list, co2_no_laws_year_mean_list = [], []

for sample in range(len(bayes_co2_coef1)):

    co2_term1 = bayes_co2_coef1[sample] * data["slaws_mit_l3_lag"]
    co2_term2 = bayes_co2_coef2[sample] * data["slaws_mit_lt_lag"]
    
    co21 = data["gdp2011"]*data["co2_gdp"]*np.exp(-co2_term1-co2_term2)/(1000*1000*1000)
    data["co2_no_laws"] = co21
    co2_no_laws_year_mean = data.groupby('year')['co2_no_laws'].sum()
    co2_no_laws_year_mean_list.append(co2_no_laws_year_mean)

    ghg_term1 = bayes_ghg_coef1[sample] * data["slaws_mit_l3_lag"]
    ghg_term2 = bayes_ghg_coef2[sample] * data["slaws_mit_lt_lag"]

    ghg1 =(data["gdp2011"]*data["ghg_gdp"]*np.exp(-ghg_term1-ghg_term2))/(1000*1000*1000)
    data["ghg_no_laws"] = ghg1
    ghg_no_laws_year_mean = data.groupby('year')['ghg_no_laws'].sum()
    ghg_no_laws_year_mean_list.append(ghg_no_laws_year_mean)

    ghg_no_laws_data.append(np.sum(data.loc[data.year==2016]["ghg_no_laws"]))
    co2_no_laws_data.append(np.sum(data.loc[data.year==2016]["co2_no_laws"]))

ghg_no_laws_year_mean_list = sorted(ghg_no_laws_year_mean_list, key=lambda x: list(x)[-1])
co2_no_laws_year_mean_list = sorted(co2_no_laws_year_mean_list, key=lambda x: list(x)[-1])

fig, axis = plt.subplots(1,1)

ghg_low = [val/1000 for val in ghg_no_laws_year_mean_list[int(len(ghg_no_laws_year_mean_list)*.05)]]
ghg_high = [val/1000 for val in ghg_no_laws_year_mean_list[int(len(ghg_no_laws_year_mean_list)*.95)]]
co2_low = [val/1000 for val in co2_no_laws_year_mean_list[int(len(co2_no_laws_year_mean_list)*.05)]]
co2_high = [val/1000 for val in co2_no_laws_year_mean_list[int(len(co2_no_laws_year_mean_list)*.95)]]

axis.fill_between(list(range(len(co2_low))), co2_low, co2_high, alpha=.2, color="blue")
axis.fill_between(list(range(len(ghg_low))), ghg_low, ghg_high, alpha=.2, color="blue")

co21 = data["gdp2011"]*data["co2_gdp"]*np.exp(-co2_mean_term1-co2_mean_term2)/(1000*1000*1000)
data["co2_no_laws"] = co21
co2_no_laws_year_mean = data.groupby('year')['co2_no_laws'].sum()
axis.plot([val/1000 for val in co2_no_laws_year_mean], color="orange")

ghg1 =(data["gdp2011"]*data["ghg_gdp"]*np.exp(-ghg_mean_term1-ghg_mean_term2))/(1000*1000*1000)
data["ghg_no_laws"] = ghg1
ghg_no_laws_year_mean = data.groupby('year')['ghg_no_laws'].sum()
axis.plot([val/1000 for val in ghg_no_laws_year_mean], color="orange")
    
axis.plot([val/1000 for val in co2_year_mean], color="blue")
axis.plot([val/1000 for val in ghg_year_mean], color="blue")

axis.xaxis.set_ticks(list(range(0,18)),["'"+str(int(val))[-2:] for val in list(range(1999,2017))],)

axis.set_xlabel("Year", weight="bold")
axis.set_ylabel("Emissions (GtCO2)", weight="bold")
axis.xaxis.label.set_size(20)
axis.yaxis.label.set_size(20)

plt.savefig("figures/fig5.png")

print("Mean of 2016 CO2 reduction: ", np.mean(co2_no_laws_data) - co2_year_mean[2016])
print("Lower bound of 2016 CO2 reduction:", np.quantile(co2_no_laws_data, .05) - co2_year_mean[2016])
print("Upper bound of 2016 CO2 reduction:", np.quantile(co2_no_laws_data, .95) - co2_year_mean[2016])

total_reduction = []
for year in range(1999, 2017):
    total_reduction.append([val[year] for val in co2_no_laws_year_mean_list] - co2_year_mean[year])
total_red_sum = np.sum(total_reduction,axis=0)
print("Mean of total CO2 reduction:", np.mean(total_red_sum))
print("Lower bound of total CO2 reduction:", np.quantile(total_red_sum,.05))
print("Upper bound of total CO2 reduction:", np.quantile(total_red_sum,.95))

print("Mean of 2016 other GHG reduction:", np.mean(ghg_no_laws_data) - ghg_year_mean[2016])
print("Lower bound of 2016 other GHG reduction:",np.quantile(ghg_no_laws_data, .05) - ghg_year_mean[2016])
print("Upper bound of 2016 other GHG reduction:",np.quantile(ghg_no_laws_data, .95) - ghg_year_mean[2016])

total_reduction = []
for year in range(1999, 2017):
    total_reduction.append([val[year] for val in ghg_no_laws_year_mean_list] - ghg_year_mean[year])
total_red_sum = np.sum(total_reduction,axis=0)
print("Mean of total other GHG reduction:", np.mean(total_red_sum))
print("Lower bound of total other GHG reduction:", np.quantile(total_red_sum,.05))
print("Upper bound of total other GHG reduction:", np.quantile(total_red_sum,.95))

# plot figure 6

temp_red_samples = []
ipcc_dist = np.random.normal(1.65, .65/1.645, 1000)
for index, sample in enumerate(random.sample(range(4000), 1000)):
    co2_term1 = bayes_co2_coef1[sample] * data["slaws_mit_l3_lag"]
    co2_term2 = bayes_co2_coef2[sample] * data["slaws_mit_lt_lag"]
    co21 = data["gdp2011"]*data["co2_gdp"]*np.exp(-co2_term1-co2_term2)/(1000*1000*1000)
    data["co2_no_laws"] = co21
    co2_no_laws_year_mean = np.sum(data.groupby('year')['co2_no_laws'].sum() - data.groupby('year')['co2'].sum())
    temp_red_samples.append(ipcc_dist[index] * co2_no_laws_year_mean/1000000)

fig, axis = plt.subplots(1,1)
axis.hist(temp_red_samples, bins=100, color="red")
sns.kdeplot(temp_red_samples, ax=axis, color="black")
axis.set_xlabel("Temperature Reduction (◦C)", size=15)
axis.set_ylabel("Probability Density", size=15)
plt.savefig("figures/fig6.png", bbox_inches="tight")