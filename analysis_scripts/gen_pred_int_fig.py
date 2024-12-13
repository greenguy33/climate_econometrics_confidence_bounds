import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# load burke dataset

burke_data = pd.read_csv("data/burke_et_al_dataset.csv")
gdp_scaler = StandardScaler()
gdp_scaled = gdp_scaler.fit_transform(np.array(burke_data.growthWDI).reshape(-1,1)).flatten()

# load burke prediction intervals

burke_pred_int = pd.read_pickle("models/burke_et_al_prediction_intervals.pkl")
idx_order =  np.flip(np.argsort(burke_pred_int["real_y"])[::-1])
burke_lowers = gdp_scaler.inverse_transform(np.array(burke_pred_int["preds_lower"][idx_order]).reshape(-1,1)).flatten()
burke_uppers = gdp_scaler.inverse_transform(np.array(burke_pred_int["preds_upper"][idx_order]).reshape(-1,1)).flatten()
burke_real_y = gdp_scaler.inverse_transform(np.array(burke_pred_int["real_y"])[idx_order].reshape(-1,1)).flatten()

# load eskander and fankhauser dataset

ef_data = pd.read_stata("data/eskander_and_fankhauser_dataset.dta")
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

ef_co2_pred_int = pd.read_pickle("models/eskander_and_fankhauser_prediction_intervals_ghg.pkl")
idx_order =  np.flip(np.argsort(ef_co2_pred_int["real_y"])[::-1])
ef_co2_lowers = co2_scaler.inverse_transform(np.array(ef_co2_pred_int["preds_lower"][idx_order]).reshape(-1,1)).flatten()
ef_co2_uppers = co2_scaler.inverse_transform(np.array(ef_co2_pred_int["preds_upper"][idx_order]).reshape(-1,1)).flatten()
ef_co2_real_y = co2_scaler.inverse_transform(np.array(ef_co2_pred_int["real_y"])[idx_order].reshape(-1,1)).flatten()

# make supplementary figure 1

def make_pred_int_plots(axis, axis_label, lowers, uppers, real_y, color, label, upper_lim=None, lower_lim=None):
    last_line = None
    for index in range(len(lowers)):
        if last_line != None:
            axis.add_patch(
                patches.Polygon(
                    xy=[[last_line[0],index-1],[last_line[1],index-1],[uppers[index],index],[lowers[index],index]]
                )
            )
        last_line = [lowers[index],uppers[index]]
    axis.scatter(lowers, list(range(len(lowers))), color=color, s=10)
    axis.scatter(uppers, list(range(len(lowers))), color=color, s=10)
    axis.scatter(real_y, list(range(len(lowers))), color="red", s=10)
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

make_pred_int_plots(axes[0], "Δln(GDP)", burke_lowers, burke_uppers, burke_real_y, "orange", "Burke et al.", .3, -.3)
# make_pred_int_plots(axes[1], "Δln(GDP)", kotz_lowers, kotz_uppers, kotz_real_y, colors["Bayes"], "Kotz et al.")
make_pred_int_plots(axes[2], "ln(CO2)", ef_co2_lowers, ef_co2_uppers, ef_co2_real_y, "orange", "Eskander and \nFankhauser (CO2)")
make_pred_int_plots(axes[3], "ln(GHG)", ef_ghg_lowers, ef_ghg_uppers, ef_ghg_real_y, "orange", "Eskander and \nFankhauser (Other GHG)")

fig.tight_layout()

plt.savefig("figures/si_fig_1.png", bbox_inches='tight', dpi=1000)