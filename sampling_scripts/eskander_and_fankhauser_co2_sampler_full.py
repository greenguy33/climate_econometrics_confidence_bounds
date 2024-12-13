import pymc as pm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from pytensor import tensor as pt
import pickle as pkl
from statsmodels.tsa.tsatools import add_lag

data = pd.read_stata("data/eskander_and_fankhauser_dataset.dta")

data["lngdp2"] = np.square(data["lngdp"])
lagged_vars = ["slaws_mit_l3","slaws_mit_lt","rle1","gdp_hp","import_share","service_share","dtemp","federal2","lngdp","lngdp2"]
for var in lagged_vars:
    data[f"{var}_lag"] = np.insert(add_lag(data[var])[:,1], 0, np.NaN)
data["year"] = [int(str(val).split("-")[0]) for val in data["year"]]
data = data.drop(data[data["year"] < 1999].index).reset_index(drop=True)

for element in sorted(list(set(data["iso2"])))[1:]:
    data[f"fe_{element}_iso2"] = np.where(data["iso2"] == element, 1, 0)
for element in sorted(list(set(data["year"])))[1:]:
    data[f"fe_{element}_year"] = np.where(data["year"] == element, 1, 0)

model_vars = [
    "lnco2",
    "slaws_mit_l3_lag",
    "slaws_mit_lt_lag",
    "rle1_lag",
    "gdp_hp_lag",
    "lngdp_lag",
    "lngdp2_lag",
    "import_share_lag",
    "service_share_lag",
    "dtemp_lag",
    "federal2_lag"
]

non_fe_vars_count = len(model_vars)

for fe_col in [col for col in data if col.startswith("fe_")]:
    model_vars.append(fe_col)

data = data.dropna(subset=model_vars).reset_index(drop=True)

scalers, scaled_data = {}, {}
for var in model_vars[0:non_fe_vars_count]:
    scalers[var] = StandardScaler()
    scaled_data[var] = scalers[var].fit_transform(np.array(data[var]).reshape(-1,1)).flatten()

scaled_df = pd.DataFrame()
for var in scaled_data:
    scaled_df[var] = scaled_data[var]
for fe_var in model_vars[non_fe_vars_count:]:
    scaled_df[fe_var] = data[fe_var]

data_len = len(scaled_df)
assert data_len == 2394

with pm.Model() as model:

    covar_coefs = pm.Normal("covar_coefs", 0, 10, shape=len(model_vars[1:]))
    covar_terms = pm.Deterministic("covar_terms", pt.sum(covar_coefs * scaled_df[model_vars[1:]], axis=1))
    
    intercept = pm.Normal("intercept", 0, 10)
    co2_prior = pm.Deterministic(
        "co2_prior", 
        covar_terms +
        intercept 
    )
    
    co2_std_scale = pm.HalfNormal("co2_std_scale", 10)
    co2_std = pm.HalfNormal("co2_std", sigma=co2_std_scale)
    co2_posterior = pm.Normal("co2_posterior", co2_prior, co2_std, observed=scaled_df[model_vars[0]])

    prior = pm.sample_prior_predictive()
    trace = pm.sample(target_accept=.99, cores=4)
    posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    with open('models/eskander_and_fankhauser_co2_model_full.pkl', 'wb') as buff:
        pkl.dump({
            "prior":prior,
            "trace":trace,
            "posterior":posterior,
            "var_list":model_vars
        },buff)