import pymc as pm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from pytensor import tensor as pt
import pickle as pkl
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.tsatools import add_lag
from sklearn.model_selection import train_test_split

def add_fixed_effect_to_data(node, data):
    for element in sorted(list(set(data[node])))[1:]:
        data[f"fe_{element}_{node}"] = np.where(data[node] == element, 1, 0)
    return data

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

scaled_df = add_fixed_effect_to_data("ID", scaled_df)
scaled_df = add_fixed_effect_to_data("yearn", scaled_df)

model_vars.extend([col for col in scaled_df.columns if col.startswith("fe")])

X_train, X_test, y_train, y_test = train_test_split(scaled_df[model_vars[1:]], scaled_df[model_vars[0]], test_size=0.2, random_state=1)

with pm.Model() as model:

    covar_coefs = pm.Normal("covar_coefs", 0, 10, shape=len(model_vars[1:]))
    covar_terms = pm.Deterministic("covar_terms", pt.sum(covar_coefs * X_train, axis=1))
    intercept = pm.Normal("intercept", 0, 10)
    
    gdp_prior = pm.Deterministic(
        "gdp_prior", 
        intercept +
        covar_terms
    )
    
    gdp_std_scale = pm.HalfNormal("gdp_std_scale", 10)
    gdp_std = pm.HalfNormal("gdp_std", sigma=gdp_std_scale)
    gdp_posterior = pm.Normal("gdp_posterior", gdp_prior, gdp_std, observed=y_train)

    prior = pm.sample_prior_predictive()
    trace = pm.sample(target_accept=.99, cores=4)
    posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

with model:
    
    mu_pred = intercept + pt.sum((covar_coefs * X_test),axis=1)
    y_pred = pm.Normal("y_pred", mu=mu_pred, sigma=gdp_std)
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["y_pred"])

preds = posterior_predictive.posterior_predictive.y_pred.data
preds = preds.reshape(-1, preds.shape[-1])
preds_lower = np.quantile(preds, .025, axis=0)
preds_upper = np.quantile(preds, .975, axis=0)
in_range = 0
for index, item in enumerate(y_test):
    if preds_lower[index] < item < preds_upper[index]:
        in_range += 1
print(in_range/len(preds[0]))

bounds_store = {"preds_lower":preds_lower,"preds_upper":preds_upper,"real_y":y_test}
with open('models/kotz_et_al_prediction_intervals.pkl', 'wb') as buff:
    pkl.dump(bounds_store,buff)
