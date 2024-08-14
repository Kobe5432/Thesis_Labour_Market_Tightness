import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def main(start_year=None, end_year=None, selected_sections=None, output_file="correlation_vif_output.xlsx"):
    # Load and preprocess data
    data = pd.read_excel(r"filepath")

    if not pd.api.types.is_numeric_dtype(data['Year']):
        data['Year'] = pd.to_datetime(data['Year'], format='%Y').dt.year

    if start_year is not None and end_year is not None:
        data = data[(data['Year'] >= start_year) & (data['Year'] <= end_year)]
        data['Year'] = data['Year'].astype(int)
        
    # Wage inflation
    data['wage_inflation'] = data['Loonkost inflatie']
    data['ma_wage_inflation'] = data.groupby('Economic section')['wage_inflation'].rolling(window=4).mean().reset_index(level=0, drop=True)
    # MA Productivity growth
    data['ma_productivity_inflation'] = data['productivity growth per person'].rolling(window=2, min_periods=1).mean()
    data['lagged_ma_productivity_inflation'] = data.groupby('Economic section')['ma_productivity_inflation'].shift(1)
    # MA JVR
    data['ma_job_vacancy_rate'] = data['Job vacancy rate'].rolling(window=2, min_periods=1).mean()
    data['lagged_ma_job_vacancy_rate'] = data.groupby('Economic section')['ma_job_vacancy_rate'].shift(2)
    # Backward-looking inflation
    data['lagged_inflation_expectations'] = data.groupby('Economic section')['Backward-looking inflation V2'].shift(0)

    model_predictors = [
        'lagged_inflation_expectations',
        'lagged_ma_productivity_inflation',
        'lagged_ma_job_vacancy_rate'
    ]

    data = data.dropna(subset=model_predictors + ['ma_wage_inflation'])

    if selected_sections is not None:
        data = data[data['Economic section'].isin(selected_sections)]

    output_data = []

    for section in data['Economic section'].unique():
        section_data = data[data['Economic section'] == section]

        X = section_data[model_predictors]
        X = sm.add_constant(X)

        # Calculate bivariate correlations
        correlations = X[model_predictors].corr().unstack().reset_index()
        correlations.columns = ['Predictor_1', 'Predictor_2', 'Correlation']
        correlations = correlations[correlations['Predictor_1'] != correlations['Predictor_2']]

        for _, row in correlations.iterrows():
            output_data.append({
                'Section': section,
                'Predictor Pair': f"{row['Predictor_1']} & {row['Predictor_2']}",
                'Biv Corr': row['Correlation'],
                'VIF': None
            })

        # Calculate VIF
        vif_data = calculate_vif(X.drop(columns=['const']))
        for _, row in vif_data.iterrows():
            output_data.append({
                'Section': section,
                'Predictor Pair': row['feature'],
                'Biv Corr': None,
                'VIF': row['VIF']
            })

    # Convert to DataFrame
    output_df = pd.DataFrame(output_data)

    # Export to Excel
    output_df.to_excel(output_file, index=False)
    print(f"Data exported successfully to {output_file}")

if __name__ == "__main__":
    selected_activities = [
        'Construction',
        'Information and communication',
        'Professional, scientific and technical activities',
        'Wholesale and retail trade; repair of motor vehicles and motorcycles',
        'Financial and insurance activities']  # Example selection
    main(start_year=2012, end_year=2023, selected_sections=selected_activities)
