import pickle as pkl
import pandas as pd

data = pd.read_pickle("full_models/burke_et_al_model_full.pkl")
with open('models/burke_et_all_model_full_covar_coefs.pkl', 'wb') as buff:
	pkl.dump({
	    "burke_coef1_full_scaled":data["trace"].posterior.temp_gdp_coef.data.flatten(),
		"burke_coef2_full_scaled":data["trace"].posterior.temp_sq_gdp_coef.data.flatten()
	},buff)

data = pd.read_pickle("full_models/kotz_et_al_model_full.pkl")
covar_coefs = data["posterior"]["posterior"]["covar_coefs"]

with open('models/kotz_reproduction_full_covar_coefs.pkl', 'wb') as buff:
	pkl.dump({
	    "t5_varm":covar_coefs[:,:,0],
	    "t5_seas_diff_mXt5_varm":covar_coefs[:,:,1]
	},buff)


data = pd.read_pickle("full_models/eskander_and_fankhauser_ghg_model_full.pkl")
covar_coefs = data["posterior"]["posterior"]["covar_coefs"]
slaws_mit_l3_lag = covar_coefs[:,:,0]
slaws_mit_lt_lag = covar_coefs[:,:,1]

with open('models/eskander_and_fankhauser_ghg_model_full_covar_coefs.pkl', 'wb') as buff:
	pkl.dump({
	    "slaws_mit_l3_lag":slaws_mit_l3_lag,
	    "slaws_mit_lt_lag":slaws_mit_lt_lag
	},buff)
	

data = pd.read_pickle("full_models/eskander_and_fankhauser_co2_model_full.pkl")
covar_coefs = data["posterior"]["posterior"]["covar_coefs"]
slaws_mit_l3_lag = covar_coefs[:,:,0]
slaws_mit_lt_lag = covar_coefs[:,:,1]

with open('models/eskander_and_fankhauser_co2_model_full_covar_coefs.pkl', 'wb') as buff:
	pkl.dump({
	    "slaws_mit_l3_lag":slaws_mit_l3_lag,
	    "slaws_mit_lt_lag":slaws_mit_lt_lag
	},buff)

