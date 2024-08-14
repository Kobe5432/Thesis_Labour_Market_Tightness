import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Arial"

def moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

def calculate_contributions(X, model):
    contributions = pd.DataFrame(index=X.index)
    for var in X.columns:
        if var != 'const':  # Exclude constant term from contributions (handle separately)
            contributions[var] = X[var] * model.params.get(var, 0)
    contributions['constant'] = model.params['const']  # Constant contribution
    return contributions

def compute_average_relative_influence(contributions, residuals):
    # Include residuals in contributions
    contributions['residuals'] = residuals.abs()
    
    average_influence = contributions.abs().mean()
    total_influence = average_influence.sum()
    relative_influence = (average_influence / total_influence) * 100  # Percentage of total
    return relative_influence

def main(start_year=None, end_year=None, selected_sections=None, use_abbreviations=True):
    # Load and preprocess data
    data = pd.read_excel(r"filepath")

    if not pd.api.types.is_numeric_dtype(data['Year']):
        data['Year'] = pd.to_datetime(data['Year'], format='%Y').dt.year

    if start_year is not None and end_year is not None:
        data = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]
        data['Year'] = data['Year'].astype(int)
        
    # Create period labels
    data['Period_Label'] = data['Year'].astype(str).str[-2:] + '_' + data['Period']

    # Calculate necessary columns
    data['wage_inflation'] = data['LCI inflation']
    data['ma_wage_inflation'] = data.groupby('Economic section')['wage_inflation'].rolling(window=2).mean().reset_index(level=0, drop=True)
    data['productivity_growth'] = data['productivity growth per person']
    data['ma_productivity_inflation'] = data.groupby('Economic section')['productivity_growth'].rolling(window=2).mean().reset_index(level=0, drop=True)
    data['lagged_ma_productivity_inflation'] = data.groupby('Economic section')['ma_productivity_inflation'].shift(3)
    data['job_vacancy_rate'] = data['Job vacancy rate']
    data['ma_job_vacancy_rate'] = data.groupby('Economic section')['job_vacancy_rate'].rolling(window=2).mean().reset_index(level=0, drop=True)
    data['lagged_ma_job_vacancy_rate'] = data.groupby('Economic section')['ma_job_vacancy_rate'].shift(2)
    data['lagged_inflation_expectations'] = data.groupby('Economic section')['Backward-looking inflation'].shift(0)

    model_predictors = [
        'lagged_inflation_expectations',
        'lagged_ma_productivity_inflation',
        'lagged_ma_job_vacancy_rate'
    ]

    data = data.dropna(subset=model_predictors + ['ma_wage_inflation'])

    if selected_sections is not None:
        data = data[data['Economic section'].isin(selected_sections)]

    # Mapping of section codes to their full names
    section_code_map = {
        'B': 'Mining and quarrying',
        'C': 'Manufacturing',
        'D': 'Electricity, gas, steam and air conditioning supply',
        'E': 'Water supply; sewerage, waste management and remediation activities',
        'F': 'Construction',
        'G': 'Wholesale and retail trade; repair of motor vehicles and motorcycles',
        'H': 'Transportation and storage',
        'I': 'Accommodation and food service activities',
        'J': 'Information and communication',
        'K': 'Financial and insurance activities',
        'L': 'Real estate activities',
        'M': 'Professional, scientific and technical activities',
        'N': 'Administrative and support service activities'
    }

    # Mapping of full section names to their abbreviations
    section_abbreviation_map = {v: k for k, v in section_code_map.items()}

    # Custom labels for predictors
    predictors_name = ['Inflation Expectations', 'Productivity Growth', 'Labour Market Tightness']

    # Prepare a DataFrame to store the relative influences for each section
    relative_influences = pd.DataFrame()

    for section in data['Economic section'].unique():
        section_data = data[data['Economic section'] == section] 
        X = section_data[model_predictors]
        X = sm.add_constant(X)

        y = section_data['ma_wage_inflation']

        # Align the indices of y_ma and X by dropping NaNs
        aligned_data = pd.concat([y, X], axis=1).dropna()
        y_aligned = aligned_data['ma_wage_inflation']
        X_aligned = aligned_data.drop(columns='ma_wage_inflation')

        # Fit the regression model
        model = sm.OLS(y_aligned, X_aligned).fit()

        contributions = calculate_contributions(X, model)
        residuals = y - model.fittedvalues

        # Compute the average relative influence for this section
        relative_influence = compute_average_relative_influence(contributions, residuals)
        relative_influence['Section'] = section_abbreviation_map.get(section, section)

        relative_influences = pd.concat([relative_influences, relative_influence], axis=1)

    relative_influences = relative_influences.T
    relative_influences.set_index('Section', inplace=True)

    # Determine x-axis labels based on the use_abbreviations flag
    if use_abbreviations:
        relative_influences.index = relative_influences.index.map(lambda x: section_abbreviation_map.get(x, x))
    else:
        # Use full names
        relative_influences.index = relative_influences.index.map(lambda x: section_code_map.get(x, x))

    # Plot the relative influences as a stacked bar chart
    fig, ax = plt.subplots(figsize=(16, 10))  # Increase figure size for better readability
    bar_width = 0.65  # Adjust bar width here
    
    # Create bar plot manually to control bar width
    bars = relative_influences.plot(kind='bar', stacked=True, width=bar_width, ax=ax,
                                    color=['#156082', '#FFC000', '#4EA72E', '#808080', '#E97132'], alpha=0.7)
    
    plt.ylabel('Average Relative Influence (%)', fontsize=13)
    
    # Update the legend labels and place it below the plot
    ax.legend(predictors_name + ['Constant', 'Residuals'], loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=5, fontsize=12)
    
    plt.xticks(rotation=0, ha='right', fontsize=12)  # Rotate x-axis labels for better readability
    plt.yticks(fontsize=12)
    # ax.yaxis.set_visible(False)
    plt.xlabel('')  # Remove x-axis label

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    
    # # Annotate bars with the percentage values
    # for bars in ax.containers:
    #     ax.bar_label(bars, fmt='%.1f%%', label_type='center', fontsize=10)
    
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.show()


if __name__ == "__main__":
    selected_activities = None
    # [
    #     'Construction',
    #     'Information and communication',
    #     'Professional, scientific and technical activities',
    #     'Wholesale and retail trade; repair of motor vehicles and motorcycles',
    #     'Financial and insurance activities']  # Example selection
    main(start_year=2012, end_year=2023, selected_sections=selected_activities, use_abbreviations=True)

