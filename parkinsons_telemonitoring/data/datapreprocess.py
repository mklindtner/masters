import pandas as pd
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
parkinsons_telemonitoring = fetch_ucirepo(id=189) 
  
# data (as pandas dataframes) 
X = parkinsons_telemonitoring.data.features 
y = parkinsons_telemonitoring.data.targets 
  
full_df = pd.concat([X, y], axis=1)

# 2. Define a filename for your local data file
filename = 'parkinsons_data.csv'

# 3. Save the combined dataframe to a CSV file
# index=False prevents pandas from writing the dataframe index as a new column
full_df.to_csv(filename, index=False)

print(f"Data successfully saved to '{filename}'")
print("\nHere are the first 5 rows of the saved data:")
print(full_df.head())