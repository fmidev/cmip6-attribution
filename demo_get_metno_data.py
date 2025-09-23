# %%
import subroutines

# %%
metnosid = "SN18700"  # Oslo-Blindern
frost_client_id = ""  # add your personal client id here

# %%
data = subroutines.read_monthly_temps_from_frost(metnosid, frost_client_id, homogenised=True)
# %%
target_mon = 9
data_sel_month = subroutines.read_obs_temp_frost(frost_client_id, metnosid, target_mon)

# %%
print(subroutines.get_station_metadata_frost(metnosid, frost_client_id))

# %%
obs_temp = subroutines.read_obs_temp_frost(frost_client_id, metnosid, target_mon).loc[1900:]
obs_temp.plot()
