import pymc as pm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from pytensor import tensor as pt
import pickle as pkl
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.tsatools import add_lag

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

scalers, scaled_data = {}, {}
for var in model_vars:
    scalers[var] = StandardScaler()
    scaled_data[var] = scalers[var].fit_transform(np.array(data[var]).reshape(-1,1)).flatten()
    
scaled_df = pd.DataFrame()
for var in scaled_data:
    scaled_df[var] = scaled_data[var]
scaled_df["ID"] = data["ID"]
scaled_df["yearn"] = data["yearn"]

data_len = len(scaled_df)
year_mult_mat = [np.zeros(data_len) for year in set(scaled_df.yearn)]
region_mult_mat = [np.zeros(data_len) for region in set(data.ID)]
region_index = -1
curr_region = ""

min_year = min(scaled_df.yearn)
for row_index, row in enumerate(scaled_df.itertuples()):
    if row.ID != curr_region:
        region_index += 1
        curr_region = row.ID
    year_index = row.yearn - min_year
    region_mult_mat[region_index][row_index] = 1
    year_mult_mat[year_index][row_index] = 1

with pm.Model() as model:

    covar_coefs = pm.Normal("covar_coefs", 0, 10, shape=len(model_vars[1:]))
    covar_terms = pm.Deterministic("covar_terms", pt.sum(covar_coefs * scaled_df[model_vars[1:]], axis=1))

    year_coefs = pt.expand_dims(pm.Normal("year_coefs", 0, 10, shape=(len(set(scaled_df.yearn))-1)),axis=1)
    year_coefs = pm.math.concatenate([[[0]],year_coefs])
    year_fixed_effects = pm.Deterministic("year_fixed_effects",pt.sum(year_coefs*year_mult_mat,axis=0))

    region_coefs = pt.expand_dims(pm.Normal("region_coefs", 0, 10, shape=(len(set(scaled_df.ID))-1)),axis=1)
    region_coefs = pm.math.concatenate([[[0]],region_coefs])
    region_fixed_effects = pm.Deterministic("region_fixed_effects",pt.sum(region_coefs*region_mult_mat,axis=0))

    intercept = pm.Normal("intercept", 0, 10)
    
    gdp_prior = pm.Deterministic(
        "gdp_prior", 
        intercept +
        covar_terms +
        year_fixed_effects +
        region_fixed_effects
    )
    
    gdp_std_scale = pm.HalfNormal("gdp_std_scale", 10)
    gdp_std = pm.HalfNormal("gdp_std", sigma=gdp_std_scale)
    gdp_posterior = pm.Normal("gdp_posterior", gdp_prior, gdp_std, observed=scaled_df[model_vars[0]])

    prior = pm.sample_prior_predictive()
    trace = pm.sample(target_accept=.99, cores=4)
    posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    with open('models/kotz_et_all_model_full.pkl', 'wb') as buff:
        pkl.dump({
            "prior":prior,
            "trace":trace,
            "posterior":posterior,
            "var_list":model_vars
        },buff)
