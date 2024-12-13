import pymc as pm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pytensor import tensor as pt
import pickle as pkl
from sklearn.linear_model import LinearRegression
import csv
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split

def add_fixed_effect_to_data(node, data):
    for element in sorted(list(set(data[node])))[1:]:
        data[f"fe_{element}_{node}"] = np.where(data[node] == element, 1, 0)
    return data

data = pd.read_csv("data/burke_et_al_dataset.csv").dropna().reset_index(drop=True)

# data scaling

precip_scaler, gdp_scaler, temp_scaler = StandardScaler(), StandardScaler(), StandardScaler()
precip_scaled = precip_scaler.fit_transform(np.array(data.UDel_precip_popweight).reshape(-1,1)).flatten()
gdp_scaled = gdp_scaler.fit_transform(np.array(data.growthWDI).reshape(-1,1)).flatten()
temp_scaled = temp_scaler.fit_transform(np.array(data.UDel_temp_popweight).reshape(-1,1)).flatten()

data["temp_scaled"] = temp_scaled
data["precip_scaled"] = precip_scaled
data["gdp_scaled"] = gdp_scaled
data["temp_scaled_2"] = np.square(temp_scaled)
data["precip_scaled_2"] = np.square(precip_scaled)

data = add_fixed_effect_to_data("iso", data)
data = add_fixed_effect_to_data("year", data)

model_vars = ["temp_scaled","precip_scaled","temp_scaled_2","precip_scaled_2"]
model_vars.extend([col for col in data.columns if col.startswith("_y")])
model_vars.extend([col for col in data.columns if col.startswith("fe")])

X_train, X_test, y_train, y_test = train_test_split(data[model_vars], data["gdp_scaled"], test_size=0.2, random_state=1)

# construct model and sample

with pm.Model() as model:

    gdp_intercept = pm.Normal('gdp_intercept',0,5)
    covar_coefs = pm.Normal("covar_coefs", 0, 10, shape=len(model_vars))
    covar_terms = pm.Deterministic("covar_terms", pt.sum(covar_coefs * X_train,axis=1))
    
    gdp_prior = pm.Deterministic(
        "gdp_prior",
        gdp_intercept +
        covar_terms
    )

    gdp_std = pm.HalfNormal('gdp_std', sigma=1)
    gdp_posterior = pm.Normal('gdp_posterior', mu=gdp_prior, sigma=gdp_std, observed=y_train)

    prior = pm.sample_prior_predictive()
    trace = pm.sample(target_accept=.99, cores=4)
    posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

with model:
    
    mu_pred = gdp_intercept + pt.sum((covar_coefs * X_test),axis=1)
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
with open('models/burke_et_al_prediction_intervals.pkl', 'wb') as buff:
    pkl.dump(bounds_store,buff)