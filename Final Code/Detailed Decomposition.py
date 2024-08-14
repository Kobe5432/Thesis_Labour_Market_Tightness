import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

plt.rcParams["font.family"] = "Arial"

def moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

def calculate_contributions(X, model):
    contributions = pd.DataFrame(index=X.index)
    for var in X.columns:
        if var != 'const':  # Exclude constant term from contributions
            contributions[var] = X[var] * model.params.get(var, 0)
    return contributions

def plot_contributions_and_wage_inflation(x_labels, contributions, ma_wage_inflation, residuals, predicted_wage_inflation, title):
    # Combine all data into a single DataFrame
    combined_df = pd.concat([ma_wage_inflation, predicted_wage_inflation, contributions, residuals], axis=1)
    combined_df.columns = ['Wage Inflation', 'Predicted Wage Inflation'] + list(contributions.columns) + ['Residuals']
    bar_positions = np.arange(len(combined_df))
    
    # Create a new figure
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot actual wage inflation as a line plot
    ax1.plot(bar_positions, combined_df['Wage Inflation'], color='red', label='Wage Inflation', linewidth=2)

    # Add dots for each data point of actual wage inflation
    ax1.scatter(bar_positions, combined_df['Wage Inflation'], color='red', s=25, zorder=2.5)
    
    # Plot predicted wage inflation as a line plot
    ax1.plot(bar_positions, combined_df['Predicted Wage Inflation'], color='blue', linestyle='--', label='Predicted Wage Inflation', linewidth=2)
    
    ax1.set_ylabel('Wage Inflation (in %)', fontsize=12)
    ax1.tick_params(axis='y')

    # Set x-ticks to only show 'Q1' entries
    ax1.set_xticks(bar_positions)
    labels = [label if 'Q4' in label else '' for label in x_labels]
    ax1.set_xticklabels(labels, rotation=20, ha='right', fontsize=12)  # Adjust fontsize as needed

    # Plot contributions as stacked bars
    stacked_values_pos = np.zeros(len(combined_df))  # To stack the contributions
    stacked_values_neg = np.zeros(len(combined_df))  # To stack the contributions

    colors = ['#156082', '#FFC000', '#4EA72E']
    predictors_name = ['Inflation Expectations', 'Productivity Growth', 'Labour Market Tightness']

    # Split constants into positive and negative parts
    constants_positive = (ma_wage_inflation - (contributions.sum(axis=1) + residuals)).clip(lower=0)
    constants_negative = (ma_wage_inflation - (contributions.sum(axis=1) + residuals)).clip(upper=0).abs()
    
    # Plot positive residuals
    ax1.bar(bar_positions, constants_positive, bottom=stacked_values_pos, width=0.8, alpha=0.7, color='grey')
    stacked_values_pos += constants_positive  # Update the stacked values
    
    # Plot negative residuals (below zero)
    ax1.bar(bar_positions, -constants_negative, bottom=stacked_values_neg, width=0.8, label='Constant', alpha=0.7, color='grey')
    stacked_values_neg -= constants_negative  # Update the stacked values
    
    for predictor, color, name in zip(contributions.columns, colors, predictors_name):
        predictor_positive = contributions[predictor].clip(lower=0)
        predictor_negative = contributions[predictor].clip(upper=0).abs()
        # Plot positive predictors
        ax1.bar(bar_positions, predictor_positive, bottom=stacked_values_pos, width=0.8, label=name, alpha=0.7, color=color)
        stacked_values_pos += predictor_positive  # Update the stacked values
        # Plot negative predictors
        ax1.bar(bar_positions, -predictor_negative, bottom=stacked_values_neg, width=0.8, alpha=0.7, color=color)
        stacked_values_neg -= predictor_negative  # Update the stacked values

    # Plot positive residuals
    residuals_positive = residuals.clip(lower=0)
    residuals_negative = residuals.clip(upper=0).abs()  # Get absolute value for plotting below zero
    # Plot positive residuals
    ax1.bar(bar_positions, residuals_positive, bottom=stacked_values_pos, width=0.8, label='Residuals', alpha=0.7, color='#E97132')
    stacked_values_pos += residuals_positive  # Update the stacked values
    # Plot negative residuals (below zero)
    ax1.bar(bar_positions, -residuals_negative, bottom=stacked_values_neg, width=0.8, alpha=0.7, color='#E97132')
    stacked_values_neg -= residuals_negative  # Update the stacked values

    # Add grid and title
    ax1.grid(axis='y', alpha=0.5)
    plt.title(title, fontsize=16)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def main(start_year=None, end_year=None, selected_sections=None, output_file='output_data.xlsx'):
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
        print(len(data))

    summary = pd.DataFrame()

    with pd.ExcelWriter(output_file) as writer:
        for section in data['Economic section'].unique():
            section_data = data[data['Economic section'] == section]
            # Recreate x_labels for this specific section
            x_labels = section_data['Period_Label'].tolist()

            y = section_data['ma_wage_inflation']
    
            X = section_data[model_predictors]
            X = sm.add_constant(X)

            # Align the indices of y_ma and X by dropping NaNs
            aligned_data = pd.concat([y, X], axis=1).dropna()
            y_ma_aligned = aligned_data['ma_wage_inflation']
            X_aligned = aligned_data.drop(columns='ma_wage_inflation')

            # Fit the regression model
            model = sm.OLS(y_ma_aligned, X_aligned).fit()
            contributions = calculate_contributions(X, model)
            residuals = y - model.fittedvalues

            plot_contributions_and_wage_inflation(x_labels, contributions, y, residuals, model.fittedvalues, f'{section} - Wage Inflation and Contributions')

            summary_data = []
            for var in model.params.index:
                summary_data.append({
                    'Section': section,
                    'Variable': var,
                    'Coefficient': model.params[var]
                })

            summary = pd.concat([summary, pd.DataFrame(summary_data)], ignore_index=True)

            # Write each section's data to the Excel file
            section_summary = pd.DataFrame(summary_data)
            section_summary.to_excel(writer, sheet_name=f'{section}_Summary', index=False)

            # You might want to save other dataframes as well, for example:
            contributions.to_excel(writer, sheet_name=f'{section}_Contributions', index=False)
            residuals.to_frame(name='Residuals').to_excel(writer, sheet_name=f'{section}_Residuals', index=False)
            section_data.to_excel(writer, sheet_name=f'{section}_Data', index=False)

    print(summary)

if __name__ == "__main__":
    selected_activities = ['Construction', 'Information and communication', 'Professional, scientific and technical activities', 'Financial and insurance activities', 'Wholesale and retail trade; repair of motor vehicles and motorcycles']
    main(start_year=2012, end_year=2023, selected_sections=selected_activities, output_file='output_data.xlsx')
