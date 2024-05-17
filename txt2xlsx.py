import pandas as pd
name = 'PID_Output_Vel_5'
df = pd.read_csv(f'{name}.txt', sep=", ")
df.to_excel(f"{name}.xlsx", index=False)