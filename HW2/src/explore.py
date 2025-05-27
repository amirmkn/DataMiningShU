import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('../data/adult.csv')
############### List the names of all numerical attributes in the dataset #################
numerical_attributes = df.select_dtypes(include=['number']).columns.tolist()
print(f"the numerical attts are : {numerical_attributes}")

################ Counting Number Of Unique Values For Each Attribute ###############
for column in df.columns:
    num_of_unique = df[column].nunique()
    print(f"Number of unique values in column {column} = {num_of_unique}")


################ Unique values of WorkClass Column ########################
unique_values = df['workclass'].unique()
print(f"Unique Values of workclass is : {unique_values}") 

################ Num of Missing Values of each column ##################
missing_values = {}
for column in df.columns:
    missing_sum = df[column].isnull().sum() + (df[column] == '?').sum()
    if missing_sum > 0 :
        missing_values[column] = missing_sum

missing_sum_df = pd.DataFrame(list(missing_values.items()),columns = ['Attribute','Num of Missing values'])
print(missing_sum_df)

################ Percentage of individuals from US ####################
us_natives = (df['native-country'] == 'United-States').sum()
all_natives = df['native-country'].notnull().sum()
percentage = us_natives/(all_natives) *100
print(f"The Percentage of individuals from US is : {percentage:.2f}%" )

################ Bar-plot of native-country attribute excluding US #############

non_us_list = {}
new_df = df['native-country'].replace('?',np.nan)
non_us = new_df[new_df != 'United-States'].dropna()
non_us_df = pd.DataFrame(non_us, columns=['native-country'])

country_counts = non_us_df.value_counts()

plt.figure(figsize=(12, 6))
country_counts.plot(kind='bar', color='skyblue')

# Add title and axis labels
plt.title('Number of Individuals by Native Country (Excluding United States)', fontsize=16)
plt.xlabel('Country', fontsize=14)
plt.ylabel('Number of Individuals', fontsize=14)

plt.xticks(rotation=45, ha='right', fontsize=12)

plt.tight_layout()
plt.show()