import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from statsmodels.tsa.tsatools import add_lag

# load burke dataset

burke_data = pd.read_csv("data/burke_et_al_dataset.csv")
burke_data = burke_data.dropna(subset=["growthWDI","UDel_precip_popweight","UDel_temp_popweight"]).reset_index(drop=True)
gdp_scaler = StandardScaler()
gdp_scaled = gdp_scaler.fit_transform(np.array(burke_data.growthWDI).reshape(-1,1)).flatten()

# load burke prediction intervals

burke_pred_int = pd.read_pickle("models/burke_et_al_prediction_intervals.pkl")
idx_order =  np.flip(np.argsort(burke_pred_int["real_y"])[::-1])
burke_lowers = gdp_scaler.inverse_transform(np.array(burke_pred_int["preds_lower"][idx_order]).reshape(-1,1)).flatten()
burke_uppers = gdp_scaler.inverse_transform(np.array(burke_pred_int["preds_upper"][idx_order]).reshape(-1,1)).flatten()
burke_real_y = gdp_scaler.inverse_transform(np.array(burke_pred_int["real_y"])[idx_order].reshape(-1,1)).flatten()

print("Burke et al. prediction interval coverage (target 95%):")
burke_colors = []
in_range = 0
for i in range(len(burke_lowers)):
    if burke_lowers[i] <= burke_real_y[i] <= burke_uppers[i]:
        in_range += 1
        burke_colors.append("red")
    else:
        burke_colors.append("green")
print(in_range/len(burke_lowers))

# load kotz dataset

kotz_data = pd.read_stata("data/kotz_et_al_dataset.dta")
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
kotz_data["T5_mean_diff"] = kotz_data.groupby("ID")["T5_mean"].diff()
kotz_data["T5_mean_diff_lag"] = np.insert(add_lag(kotz_data["T5_mean_diff"])[:,1], 0, np.NaN)
t5_mean_diff_lag = []
last_region = ""
for row in kotz_data.itertuples():
    if row.ID != last_region:
        t5_mean_diff_lag.append(np.NaN)
    else:
        t5_mean_diff_lag.append(row.T5_mean_diff_lag)
    last_region = row.ID
kotz_data["T5_mean_diff_lag"] = t5_mean_diff_lag

kotz_data["T5_seas_diff_mXT5_varm"] = kotz_data["T5_seas_diff_m"] * kotz_data["T5_varm"]
kotz_data["T5_mean_mXT5_mean_diff"] = kotz_data["T5_mean_m"] * kotz_data["T5_mean_diff"]
kotz_data["T5_mean_mXT5_mean_diff_lag"] = kotz_data["T5_mean_m"] * kotz_data["T5_mean_diff_lag"]

kotz_data = kotz_data.dropna(subset=model_vars).reset_index(drop=True)

gdp_scaler = StandardScaler()
gdp_scaled = gdp_scaler.fit_transform(np.array(kotz_data.dlgdp_pc_usd).reshape(-1,1)).flatten()

# load kotz prediction intervals

kotz_pred_int = pd.read_pickle("models/kotz_et_al_prediction_intervals.pkl")
idx_order =  np.flip(np.argsort(kotz_pred_int["real_y"])[::-1])
kotz_lowers = gdp_scaler.inverse_transform(np.array(kotz_pred_int["preds_lower"][idx_order]).reshape(-1,1)).flatten()
kotz_uppers = gdp_scaler.inverse_transform(np.array(kotz_pred_int["preds_upper"][idx_order]).reshape(-1,1)).flatten()
kotz_real_y = gdp_scaler.inverse_transform(np.array(kotz_pred_int["real_y"])[idx_order].reshape(-1,1)).flatten()

print("Kotz et al. prediction interval coverage (target 95%):")
kotz_colors = []
in_range = 0
for i in range(len(kotz_lowers)):
    if kotz_lowers[i] <= kotz_real_y[i] <= kotz_uppers[i]:
        in_range += 1
        kotz_colors.append("red")
    else:
        kotz_colors.append("green")
print(in_range/len(kotz_lowers))

# load eskander and fankhauser dataset

ef_data = pd.read_stata("data/eskander_and_fankhauser_dataset.dta")

ef_data["lngdp2"] = np.square(ef_data["lngdp"])
ef_data["year"] = [int(str(val).split("-")[0]) for val in ef_data["year"]]
ef_data = ef_data.drop(ef_data[ef_data["year"] < 1999].index).reset_index(drop=True)

co2_scaler = StandardScaler()
co2_scaled = co2_scaler.fit_transform(np.array(ef_data.lnco2).reshape(-1,1)).flatten()
ghg_scaler = StandardScaler()
ghg_scaled = ghg_scaler.fit_transform(np.array(ef_data.lnghg).reshape(-1,1)).flatten()

# load eskander and fankhauser prediction intervals

ef_ghg_pred_int = pd.read_pickle("models/eskander_and_fankhauser_prediction_intervals_co2.pkl")
idx_order =  np.flip(np.argsort(ef_ghg_pred_int["real_y"])[::-1])
ef_ghg_lowers = ghg_scaler.inverse_transform(np.array(ef_ghg_pred_int["preds_lower"][idx_order]).reshape(-1,1)).flatten()
ef_ghg_uppers = ghg_scaler.inverse_transform(np.array(ef_ghg_pred_int["preds_upper"][idx_order]).reshape(-1,1)).flatten()
ef_ghg_real_y = ghg_scaler.inverse_transform(np.array(ef_ghg_pred_int["real_y"])[idx_order].reshape(-1,1)).flatten()

print("Eskander and Fankhauser GHG prediction interval coverage (target 95%):")
ef_ghg_colors = []
in_range = 0
for i in range(len(ef_ghg_lowers)):
    if ef_ghg_lowers[i] <= ef_ghg_real_y[i] <= ef_ghg_uppers[i]:
        in_range += 1
        ef_ghg_colors.append("red")
    else:
        ef_ghg_colors.append("green")
print(in_range/len(ef_ghg_lowers))

ef_co2_pred_int = pd.read_pickle("models/eskander_and_fankhauser_prediction_intervals_ghg.pkl")
idx_order =  np.flip(np.argsort(ef_co2_pred_int["real_y"])[::-1])
ef_co2_lowers = co2_scaler.inverse_transform(np.array(ef_co2_pred_int["preds_lower"][idx_order]).reshape(-1,1)).flatten()
ef_co2_uppers = co2_scaler.inverse_transform(np.array(ef_co2_pred_int["preds_upper"][idx_order]).reshape(-1,1)).flatten()
ef_co2_real_y = co2_scaler.inverse_transform(np.array(ef_co2_pred_int["real_y"])[idx_order].reshape(-1,1)).flatten()

print("Eskander and Fankhauser GHG prediction interval coverage (target 95%):")
ef_co2_colors = []
in_range = 0
for i in range(len(ef_co2_lowers)):
    if ef_co2_lowers[i] <= ef_co2_real_y[i] <= ef_co2_uppers[i]:
        in_range += 1
        ef_co2_colors.append("red")
    else:
        ef_co2_colors.append("green")
print(in_range/len(ef_co2_lowers))

# make supplementary figure 1

def make_pred_int_plots(axis, axis_label, lowers, uppers, real_y, color, dot_colors, label, upper_lim=None, lower_lim=None):
    last_line = None
    axis.scatter(lowers, list(range(len(lowers))), color=color, s=10)
    axis.scatter(uppers, list(range(len(lowers))), color=color, s=10)
    for index in range(len(lowers)):
        if last_line != None:
            axis.add_patch(
                patches.Polygon(
                    xy=[[last_line[0],index-1],[last_line[1],index-1],[uppers[index],index],[lowers[index],index]]
                )
            )
        last_line = [lowers[index],uppers[index]]
    axis.scatter(real_y, list(range(len(lowers))), color=dot_colors, s=10)
    axis.set_xlabel(axis_label, weight="bold")
    axis.set_ylabel("Withheld Row #", weight="bold")
    axis.xaxis.label.set_size(10)
    axis.yaxis.label.set_size(10)
    axis.xaxis.set_tick_params(labelsize=15)
    axis.yaxis.set_tick_params(labelsize=15)
    axis.set_title(label, weight="bold")
    axis.title.set_size(15)
    if upper_lim is not None and lower_lim is not None:
        axis.set_xlim(lower_lim, upper_lim)

fig, axes = plt.subplots(1,4, figsize=(12,4))

make_pred_int_plots(axes[0], "Δln(GDP)", burke_lowers, burke_uppers, burke_real_y, "orange", burke_colors, "Burke et al.", .3, -.3)
make_pred_int_plots(axes[1], "Δln(GDP)", kotz_lowers, kotz_uppers, kotz_real_y, "orange", kotz_colors, "Kotz et al.", 1, -1)
make_pred_int_plots(axes[2], "ln(CO2)", ef_co2_lowers, ef_co2_uppers, ef_co2_real_y, "orange", ef_co2_colors, "Eskander and \nFankhauser (CO2)")
make_pred_int_plots(axes[3], "ln(GHG)", ef_ghg_lowers, ef_ghg_uppers, ef_ghg_real_y, "orange", ef_ghg_colors, "Eskander and \nFankhauser (Other GHG)")

fig.tight_layout()

plt.savefig("figures/si_fig_1.png", bbox_inches='tight', dpi=1000)