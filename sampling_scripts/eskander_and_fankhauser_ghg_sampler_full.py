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

model_vars = [
    "lnghg",
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
data = data.dropna(subset=model_vars).reset_index(drop=True)

scalers, scaled_data = {}, {}
for var in model_vars:
    scalers[var] = StandardScaler()
    scaled_data[var] = scalers[var].fit_transform(np.array(data[var]).reshape(-1,1)).flatten()

scaled_df = pd.DataFrame()
for var in scaled_data:
    scaled_df[var] = scaled_data[var]
scaled_df["year"] = data["year"]
scaled_df["iso2"] = data["iso2"]

assert len(scaled_df) == 2394

data_len = len(scaled_df)
year_mult_mat = [np.zeros(data_len) for year in set(scaled_df.year)]
country_mult_mat = [np.zeros(data_len) for country in set(scaled_df.iso2)]
country_index = -1
curr_country = ""

min_year = min(scaled_df.year)
for row_index, row in enumerate(scaled_df.itertuples()):
    if row.iso2 != curr_country:
        country_index += 1
        curr_country = row.iso2
    year_index = row.year - min_year
    country_mult_mat[country_index][row_index] = 1
    year_mult_mat[year_index][row_index] = 1

with pm.Model() as model:

    covar_coefs = pm.Normal("covar_coefs", 0, 10, shape=len(model_vars[1:]))
    covar_terms = pm.Deterministic("covar_terms", pt.sum(covar_coefs * scaled_df[model_vars[1:]], axis=1))

    year_coefs = pt.expand_dims(pm.Normal("year_coefs", 0, 10, shape=(len(set(scaled_df.year))-1)),axis=1)
    year_coefs = pm.math.concatenate([[[0]],year_coefs])
    year_fixed_effects = pm.Deterministic("year_fixed_effects",pt.sum(year_coefs*year_mult_mat,axis=0))

    country_coefs = pt.expand_dims(pm.Normal("country_coefs", 0, 10, shape=(len(set(scaled_df.iso2))-1)),axis=1)
    country_coefs = pm.math.concatenate([[[0]],country_coefs])
    country_fixed_effects = pm.Deterministic("country_fixed_effects",pt.sum(country_coefs*country_mult_mat,axis=0))
    
    ghg_prior = pm.Deterministic(
        "ghg_prior", 
        covar_terms +
        year_fixed_effects +
        country_fixed_effects
    )
    
    ghg_std_scale = pm.HalfNormal("ghg_std_scale", 10)
    ghg_std = pm.HalfNormal("ghg_std", sigma=ghg_std_scale)
    ghg_posterior = pm.Normal("ghg_posterior", ghg_prior, ghg_std, observed=scaled_df[model_vars[0]])

    prior = pm.sample_prior_predictive()
    trace = pm.sample(target_accept=.99, cores=4)
    posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    with open('models/eskander_and_fankhauser_ghg_model_full.pkl', 'wb') as buff:
        pkl.dump({
            "prior":prior,
            "trace":trace,
            "posterior":posterior,
            "var_list":model_vars
        },buff)