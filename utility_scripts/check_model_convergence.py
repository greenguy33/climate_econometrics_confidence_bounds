import pandas as pd
import arviz as az

# check convergence via Rhat

burke_model = pd.read_pickle("models/burke_et_al_model_full.pkl")
print("Burke et al.")
print(az.summary(burke_model["trace"], var_names=["temp_gdp_coef","temp_sq_gdp_coef","precip_gdp_coef","precip_sq_gdp_coef","gdp_intercept"]))

kotz_model = pd.read_pickle("models/kotz_et_al_model_full.pkl")
print("Kotz et al.")
print(az.summary(kotz_model["trace"], var_names=["covar_coefs"]))

ef_co2 = pd.read_pickle("models/eskander_and_fankhauser_co2_model_full.pkl")
print("Eskander and Fankhauser - CO2")
print(az.summary(ef_co2["trace"], var_names=["covar_coefs"]))

ef_ghg = pd.read_pickle("models/eskander_and_fankhauser_ghg_model_full.pkl")
print("Eskander and Fankhauser - Other GHG")
print(az.summary(ef_ghg["trace"], var_names=["covar_coefs"]))