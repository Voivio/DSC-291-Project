import pandas as pd

data_path = "./dataverse_files/AER merged.dta"
df = pd.read_stata(data_path)
df.to_csv("./AER merged.csv")

print("Conversion complete.")
print(df.shape)