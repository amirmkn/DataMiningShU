import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

df = pd.read_csv('../data/adult.csv')


#################### Missing Value Imputation #####################
missing_columns = ['workclass','occupation','native-country']
for c in missing_columns:
    df[c]= df[c].replace('?',df[c].mode()[0])

################### Merging Infrequent Countries ###################
country_counts = df['native-country'].value_counts()
less_than_40 = country_counts[country_counts < 40].index

df['native-country'] = df['native-country'].replace(less_than_40,'Others')

unique_count = len(df['native-country'].unique())
print(f"Number of unique values of the new native-country: {unique_count}")

##################### Categorical Attributes Binarization ##############
# Get all categorical columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical Attributes:")
print(categorical_columns)

for c in categorical_columns:
    # Generate binary attributes based on the column
    if c == 'income':
        binary_atts = pd.get_dummies(df[c], prefix="income",prefix_sep='').astype(int)
    else:
        binary_atts = pd.get_dummies(df[c], prefix=f"{c}", prefix_sep="=").astype(int)
    
    # Concatenate the new binary attributes with the DataFrame
    df = pd.concat([df, binary_atts], axis=1)

    # Drop the original categorical column
    df = df.drop(columns=[c])
######################## Continuous Attributes Binarization ######################
# age binning with equal fequency and 12 intervals
age_binned = pd.qcut( df['age'], q=12)
def age_bin_label(bin):
    # Get the interval bounds
    left, right = bin.left, bin.right
    # Format the interval names as requested (rounded to 2 decimals)
    return f"{left:.2f} < age <= {right:.2f}"
df['age_bin'] = age_binned.apply(age_bin_label)
df_dummies = pd.get_dummies(df['age_bin']).astype(int)
# age_bin_boundaries = age_binned.cat.categories
# print(age_bin_boundaries)

# Concatenate the dummy columns with the original DataFrame
df = pd.concat([df, df_dummies], axis=1)

# education-num binning with equal width and 8 intervals
education_num_bin = pd.cut(df['education-num'],bins=8)
def edu_num_label(bin):
    # Get the interval bounds
    left, right = bin.left, bin.right
    # Format the interval names as requested (rounded to 2 decimals)
    return f"{left:.2f} < education-num <= {right:.2f}"
df['education_num_bin'] = education_num_bin.apply(edu_num_label)
df_dummies_edu = pd.get_dummies(df['education_num_bin']).astype(int)
df = pd.concat([df, df_dummies_edu], axis=1)

#hours-per-week with equal width and 5 intervals
hours_bin = pd.cut(df['hours-per-week'],bins=5)
def hours_label(bin):
    # Get the interval bounds
    left, right = bin.left, bin.right
    # Format the interval names as requested (rounded to 2 decimals)
    return f"{left:.2f} < hours-per-week <= {right:.2f}"
df['hours_bin'] = hours_bin.apply(hours_label)
df_dummies_hours = pd.get_dummies(df['hours_bin']).astype(int)
df = pd.concat([df, df_dummies_hours], axis=1)


# (c) Binning capital-gain with 6 intervals using custom split points
capital_gain_bins = [-float('inf'), 2000, 5700, 11500, 21500, 64000, float('inf')]

def capital_gain_label(bin):
    left, right = bin.left, bin.right
    if left == -float('inf'):
        return f"capital-gain <= {right:.2f}"
    elif right == float('inf'):
        return f"capital-gain > {left:.2f}"
    else:
        return f"{left:.2f} < capital-gain <= {right:.2f}"

# Bin and label capital-gain
df['capital_gain_bin'] = pd.cut(df['capital-gain'], bins=capital_gain_bins)
df['capital_gain_bin'] = df['capital_gain_bin'].apply(capital_gain_label)

# One-hot encode the bins
capital_gain_dummies = pd.get_dummies(df['capital_gain_bin']).astype(int)
df = pd.concat([df, capital_gain_dummies], axis=1)

# (d) Binning capital-loss with 4 intervals using custom split points
capital_loss_bins = [-float('inf'), 900, 2000, 3100, float('inf')]

def capital_loss_label(bin):
    left, right = bin.left, bin.right
    if left == -float('inf'):
        return f"capital-loss <= {right:.2f}"
    elif right == float('inf'):
        return f"capital-loss > {left:.2f}"
    else:
        return f"{left:.2f} < capital-loss <= {right:.2f}"

# Bin and label capital-loss
df['capital_loss_bin'] = pd.cut(df['capital-loss'], bins=capital_loss_bins)
df['capital_loss_bin'] = df['capital_loss_bin'].apply(capital_loss_label)

# One-hot encode the bins
capital_loss_dummies = pd.get_dummies(df['capital_loss_bin']).astype(int)
df = pd.concat([df, capital_loss_dummies], axis=1)


############## Creating strip plots ##############
# (a) Equal frequency for age with n=12 intervals
age_intervals = pd.qcut(df['age'], q=12, retbins=True)[1]

# (b) Equal width for education-num with n=8 intervals
edu_intervals = np.linspace(df['education-num'].min(), df['education-num'].max(), num=9)

# (c) Fixed intervals for capital-gain
capital_gain_intervals = [0, 2000, 5700, 11500, 21500, 64000, df['capital-gain'].max()]

# (d) Fixed intervals for capital-loss
capital_loss_intervals = [0, 900, 2000, 3100, df['capital-loss'].max()]

# (e) Equal width for hours-per-week with n=5 intervals
hours_intervals = np.linspace(df['hours-per-week'].min(), df['hours-per-week'].max(), num=6)

# Function to create strip plots
def create_enhanced_strip_plots(data, bins_and_intervals, figure_size=(20, 10)):
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=figure_size)
    axes = axes.flatten()
    
    for ax, (attribute, intervals, title, log_scale) in zip(axes, bins_and_intervals):
        sns.stripplot(
            x=attribute, 
            data=data, 
            jitter=0.2,  # Add random displacement to avoid overplotting
            size=3,      # Adjust point size
            alpha=0.5,   # Add transparency for better visibility
            ax=ax
        )
        
        # Add vertical lines for the split points
        for split in intervals[1:-1]:  # Skip the first and last points
            ax.axvline(
                x=split, 
                color='red', 
                linestyle='--', 
                label=f'Split: {split:.2f}'
            )
        
        
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f"{attribute} (log scale)" if log_scale else attribute, fontsize=10)
        
        if log_scale:  # Apply logarithmic scale to the x-axis if specified
            ax.set_xscale('log')
        
        # Add legend and grid
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True)

    plt.tight_layout()  # Adjust layout to avoid overlapping elements
    plt.show()

# Attribute configurations with log scale adjustments for wide-range features
attributes_and_bins = [
    ('age', age_intervals, 'Strip Plot for Age (Equal Frequency)', False),
    ('education-num', edu_intervals, 'Strip Plot for Education-Num (Equal Width)', False),
    ('capital-gain', capital_gain_intervals, 'Strip Plot for Capital-Gain (Fixed Intervals)', True),
    ('capital-loss', capital_loss_intervals, 'Strip Plot for Capital-Loss (Fixed Intervals)', True),
    ('hours-per-week', hours_intervals, 'Strip Plot for Hours-per-Week (Equal Width)', False),
]

# Generate all enhanced strip plots in a single figure
create_enhanced_strip_plots(df, attributes_and_bins)



df = df.drop(columns=['capital-gain', 'capital_gain_bin','age','age_bin','hours-per-week','hours_bin','capital-loss','capital_loss_bin','education_num_bin','education-num','fnlwgt'])

df.to_csv("../dist/adult_preprocessed.csv", index=False)
