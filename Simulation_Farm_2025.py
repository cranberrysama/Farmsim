def clear_globals():
    keys_to_clear = [key for key in globals().keys() if not key.startswith("__")]
    for key in keys_to_clear:
        del globals()[key]
clear_globals()

import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
import random
import numpy as np
import numpy_financial as npf
# import math
# from scipy.stats import gaussian_kde

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)


def draw_samples_from_scenario(data, size=5):
    import re

    # Detect all variable columns (Price & Yield columns)
    price_cols = [col for col in data.columns if re.search(r'Prc', col)]
    yield_cols = [col for col in data.columns if re.search(r'Yld', col)]
    var_cols = price_cols + yield_cols  # all variables

    # Create Year_Indexed
    data = data.copy()
    data["Year_Indexed"] = np.arange(1, len(data) + 1)

    # Detrend and collect results
    results = {}
    for var in var_cols:
        X = sm.add_constant(data["Year_Indexed"])
        model = sm.OLS(data[var], X).fit()
        trend = model.predict(X)
        deviation = data[var] - trend
        percent_deviation = deviation / trend

        results[var] = {
            "Deviation": deviation,
            "Percent_Deviation": percent_deviation,
            "Trend": trend,
            "OLS_Model": model
        }

        # Store to dataframe
        data[f"{var}_Deviation"] = deviation
        data[f"{var}_Percent_Deviation"] = percent_deviation
        data[f"{var}_Trend"] = trend

    # Step 1: Prepare sorted percent deviations
    data_pd = data[[f"{var}_Percent_Deviation" for var in var_cols]]
    sorted_perc_deviations = np.sort(data_pd, axis=0)
    num_samples = len(data_pd)
    quantiles = np.linspace(0, 1, num_samples)

    # Step 2: Correlation matrix & Cholesky
    data_dev = data[[f"{var}_Deviation" for var in var_cols]]
    correlation_matrix = data_dev.corr(method="pearson")
    cholesky_matrix = np.linalg.cholesky(correlation_matrix)

    # Step 3: Generate random samples
    random_samples = np.random.normal(size=(size, len(var_cols)))
    correlated_samples = random_samples @ cholesky_matrix.T
    cusd_samples = norm.cdf(correlated_samples)  # Convert to uniform

    # Step 4: Interpolate stochastic deviations
    stochastic_deviates_percent = np.array([
        np.interp(cusd_samples[:, i], quantiles, sorted_perc_deviations[:, i])
        for i in range(len(var_cols))
    ]).T

    # Step 5: Build DataFrame
    stochastic_deviates_df = pd.DataFrame(
        stochastic_deviates_percent,
        columns=[f"{var.split()[0]}_Stochastic_Deviation" for var in var_cols]
    )

    # Step 6: Predict future values
    future_years = np.arange(data["Year_Indexed"].max() + 1, data["Year_Indexed"].max() + 1 + size)
    new_predicted_values = {year: {} for year in future_years}

    for j, var in enumerate(var_cols):
        model = results[var]["OLS_Model"]
        for i, year in enumerate(future_years):
            stochastic_percent = stochastic_deviates_df.iloc[i, j]
            new_predicted_values[year][var] = model.predict([[1, year]])[0] * (1 + stochastic_percent)

    # Step 7: Return as DataFrame
    predicted_values_df = pd.DataFrame.from_dict(new_predicted_values, orient="index")

    return predicted_values_df

def summarize_outcomes(outcomes_list):
    """
    Summarizes a list of outcome dictionaries.

    Args:
        outcomes_list (list): List of dictionaries, where each dictionary represents simulation outcomes.

    Returns:
        dict: Aggregated outcomes with mean or other summary statistics.
    """
    # Initialize a dictionary to store aggregated results
    aggregated_outcomes = {}

    # Convert list of dictionaries to DataFrame for easier handling
    outcomes_df = pd.DataFrame(outcomes_list)

    for col in outcomes_df.columns:
        # Check the type of each column
        if isinstance(outcomes_df[col].iloc[0], (list, pd.Series)):
            # Handle multi-value outcomes (lists/Series)
            # Convert to DataFrame and calculate column-wise mean
            expanded_df = pd.DataFrame(outcomes_df[col].tolist())
            aggregated_outcomes[col] = expanded_df.mean().tolist()
        elif isinstance(outcomes_df[col].iloc[0], (float, int)):
            # Handle numeric single-value outcomes
            aggregated_outcomes[col] = outcomes_df[col].mean()
        else:
            # Handle other types (e.g., boolean or categorical)
            aggregated_outcomes[col] = outcomes_df[col].mode()[0]  # Use the most common value

    return aggregated_outcomes

def simulate_scenarios(num_simulations, sim_function):
    """
    Simulates multiple scenarios and aggregates the outcomes.

    Args:
        num_simulations (int): Number of simulations to run.
        sim_function (function): Function that runs a single simulation and returns outcomes.

    Returns:
        dict: Aggregated outcomes with summary statistics.
    """
    outcomes_list = [sim_function() for _ in range(num_simulations)]
    return summarize_outcomes(outcomes_list)

def format_outcomes(outcomes):
    """
    Formats aggregated outcomes for better readability.

    Args:
        outcomes (dict): Aggregated outcomes.

    Returns:
        str: Formatted string representation of outcomes.
    """
    formatted_lines = []
    for key, value in outcomes.items():
        if isinstance(value, list):
            formatted_lines.append(f"{key}: {', '.join(map(str, value))}")
        else:
            formatted_lines.append(f"{key}: {value}")
    return "\n".join(formatted_lines)

def generate_inflation_matrix(base_rates, growth_rates, num_years):
    """
    Generate a DataFrame of rates based on base values, growth rates, and a specified number of years.

    :param base_rates: Dictionary containing the base rates for each variable.
    :param growth_rates: Dictionary containing the annual growth rates for each variable.
    :param num_years: Number of years to project.
    :return: Pandas DataFrame with rows corresponding to years and columns to variables.
    """
    # Initialize the matrix
    inflation_matrix = []

    # Iterate over the number of years
    for i in range(num_years):
        year = base_rates["Year"] + i
        row = {"Year": year}  # Start with the year
        for key in base_rates:
            if key != "Year":  # Skip the Year key for calculations
                value = base_rates[key] + i * growth_rates[key]  # Compute value for the current year
                row[key] = value
        inflation_matrix.append(row)

    # Convert the result into a DataFrame
    return pd.DataFrame(inflation_matrix)


def calculate_tax_from_bracket(taxable_income):
    """
    Calculate tax based on the given tax brackets.

    Args:
    - taxable_income (float): The taxable income.

    Returns:
    - total_tax (float): The total tax calculated.
    """
    # Tax brackets as per the provided data
    tax_brackets = [
        (0, 50000, 0, 0.15),  # From $0 to $50,000
        (50000, 75000, 7500, 0.25),  # From $50,000 to $75,000
        (75000, 10000000, 13750, 0.34),  # From $75,000 to $10,000,000
        (10000000, float('inf'), 388250, 0.35)  # From $10,000,000 upwards
    ]

    total_tax = 0
    for lower, upper, base_tax, rate in tax_brackets:
        if taxable_income > lower:
            if taxable_income <= upper:
                total_tax = base_tax + (taxable_income - lower) * rate
                break
        else:
            break

    return round(total_tax, 2)

def extract_insurance_info(insurance_data: object, scenario: object) -> object:
    """
    Extract insurance information for a specified scenario (e.g., 'Base', 'Alt1', etc.).

    :param insurance_data: Dictionary containing insurance selection data.
    :param scenario: The scenario to extract (e.g., 'Base', 'Alt1', 'Alt2').
    :return: A dictionary with Crop, Yield Fractions, and Premiums for the specified scenario.
    """
    if scenario not in ["Base", "Alt1", "Alt2", "Alt3", "Alt_NoInsurance"]:
        raise ValueError(
            f"Invalid scenario: {scenario}. Must be one of ['Base', 'Alt1', 'Alt2', 'Alt3', 'Alt_NoInsurance']")

    # Extract data for the specified scenario
    return {
        "Crop": insurance_data["Crop"],
        f"Yield Fractions": [
            yield_fraction[scenario] for yield_fraction in insurance_data["Yield Fractions"]
        ],
        f"Premiums $/acre": [
            premium[scenario] for premium in insurance_data["Premiums $/acre"]
        ]
    }


def get_scenario_info(scenario_name):
    """
    Retrieve all information for a given scenario.

    :param scenario_name: The name of the scenario (e.g., 'Base', 'Alt1').
    :return: A dictionary containing all information for the scenario, or None if not found.
    """
    # Filter the DataFrame for the specified scenario
    scenario_data = scenario_df.loc[scenario_df['Scenario'] == scenario_name]

    if scenario_data.empty:
        return f"Scenario '{scenario_name}' not found."

    # Convert the row to a dictionary and return
    return scenario_data.to_dict(orient='records')[0]
def calculate_loan_payment(principal, annual_rate, term_years):
    r = annual_rate  # Annual interest rate
    n = term_years   # Total number of years
    payment = (principal * r * (1 + r)**n) / ((1 + r)**n - 1)
    return payment

def loan_balance_and_interest(principal, annual_rate, term_years, year):
    """
    Calculate the remaining principal, interest expense, and principal payment for a specific year.

    Args:
    - principal (float): Loan principal (initial amount borrowed).
    - annual_rate (float): Annual fixed interest rate (e.g., 0.044 for 4.4%).
    - term_years (int): Total loan term in years.
    - year (int): The year of repayment to calculate (1-based index).

    Returns:
    - remaining_principal (float): The remaining principal after the given year.
    - interest_expense (float): The interest expense for the given year.
    - principal_payment (float): The principal payment for the given year.
    """
    # Calculate annual payment using amortization formula
    r = annual_rate
    n = term_years
    total_payment = (principal * r * (1 + r)**n) / ((1 + r)**n - 1)

    # Initialize variables
    remaining_principal = principal

    # Loop through each year up to the specified year
    for current_year in range(1, year + 1):
        # Interest expense for the current year
        interest_expense = remaining_principal * r
        # Principal payment for the current year
        principal_payment = total_payment - interest_expense
        # Update remaining principal
        remaining_principal -= principal_payment

    # Return values for the given year
    return round(remaining_principal, 2), round(interest_expense, 2), round(principal_payment, 2)


def probability_consecutive_negatives(cash_flow):
    """
    Calculate the probability that two consecutive years have values less than 0.

    Parameters:
    cash_flow (pandas.Series): A pandas Series containing cash flow data.

    Returns:
    float: The probability of two consecutive years both being less than 0.
    """
    # Ensure there are at least two years of data to compare
    if len(cash_flow) < 2:
        return 0.0

    # Count the number of consecutive years both less than 0
    consecutive_negatives_count = 0
    total_pairs = len(cash_flow) - 1  # Total number of consecutive pairs

    # Iterate through the cash flow values and check for consecutive negative values
    for i in range(len(cash_flow) - 1):
        if cash_flow[i] < 0 and cash_flow[i + 1] < 0:
            consecutive_negatives_count += 1  # Increment count if both years are negative

    # Calculate the probability of consecutive negative values
    probability = consecutive_negatives_count / total_pairs
    return probability


def consecutive_loss_probability(cash_flow):
    """
    Calculate the probability of consecutive losses (each year compared with the previous year).

    Parameters:
    cash_flow (pandas.Series): A pandas Series containing cash flow data for each year.

    Returns:
    list: A list containing the probability of consecutive losses for each year (except the first year).
    """
    probabilities = []  # List to store the probabilities for each year
    total_years = len(cash_flow)

    # Iterate over the years (starting from the second year)
    for i in range(1, total_years):
        # Check if both the current year and the previous year are less than 0
        if cash_flow[i] < 0 and cash_flow[i - 1] < 0:
            probabilities.append(1)  # Probability of consecutive loss (both years are negative)
        else:
            probabilities.append(0)  # No consecutive loss

    return probabilities

def calculate_payback_period(initial_investment, pv_net_returns):
    """
    Calculate the payback period based on initial investment and PV net returns.

    Args:
        initial_investment (float): The initial investment amount (negative value if it's a cash outflow).
        pv_net_returns (list or array): A list of annual Present Value (PV) Net Returns.

    Returns:
        payback_period (float): The number of years required to recover the investment, or None if not recovered.
    """
    cumulative_pv = 0  # Track cumulative PV of Net Returns
    for year, pv in enumerate(pv_net_returns, start=1):  # Start counting from year 1
        cumulative_pv += pv
        if cumulative_pv >= initial_investment:
            # Interpolate if the exact recovery happens mid-year
            previous_cumulative = cumulative_pv - pv
            interpolation = (initial_investment - previous_cumulative) / pv
            return year - 1 + interpolation  # Fractional year
    return None  # Return None if investment is not recovered

def unpack_params(params):
    """
    Unpack parameters from a dictionary and assign them as local variables.
    Args:
        params (dict): Dictionary containing all the input parameters.
    """
    globals().update(params)

def sim_selected_scenario():

    # increase_rate_selected = scenario_info_selected['increase_rate_cost']
    #
    # selected_production_costs = {
    #     crop: {key: value * (1 + increase_rate_selected) for key, value in crop_costs_base.loc[crop].items()}
    #     for crop in crop_costs_base.index
    # }

    selected_production_costs = crop_costs_base
    selected_insurance_info = extract_insurance_info(Insurance_selection_alternatives, selected_scenario)
    selected_insurance_info_df = pd.DataFrame(selected_insurance_info)
    selected_insurance_info_df.set_index('Crop', inplace=True)
    selected_planted_acres = {crop: acres[selected_scenario] for crop, acres in planted_acres.items()}


    data = {
        "Year": list(years)
    }
    Fraction_unsold2 = Fraction_unsold

    # Generate stochastic prices and yields for each crop
    sampled_values_df = draw_samples_from_scenario(data_hist, size=len(years))
    # predicted_values = draw_samples_from_scenario(data_hist, size=5)
    sampled_values = {}

    for col in sampled_values_df.columns:
        parts = col.split()  # ['Crop1', 'Prc', ...]
        crop_name = parts[0]  # 'Crop1'

        if 'Prc' in col:
            key = f"{crop_name}_Pr"
        elif 'Yld' in col:
            key = f"{crop_name}_Yld"
        else:
            continue  # skip if not price or yield

        sampled_values[key] = sampled_values_df[col].to_numpy()

    for crop in crops:
        # Generate prices and yields based on normal distributions
        data[f"{crop} Prc"] = (
                sampled_values[f"{crop}_Pr"] +
                selected_production_costs.loc[crop]["Price Wedge $/Ac"]
        )
        data[f"{crop} Yld"] = sampled_values[f"{crop}_Yld"]

        # Calculate production and market receipts
        data[f"{crop} Production"] = data[f"{crop} Yld"] * selected_planted_acres[crop]
        data[f"{crop} Market Receipts"] = data[f"{crop} Production"] * (1 - Fraction_unsold2) * data[f"{crop} Prc"]

        # Calculate insured yield and lost yield
        data[f"{crop} Insured Yield"] = (
                insurance_df.loc[crop, 'APH Yield Bu'] *
                selected_insurance_info_df.loc[crop, 'Yield Fractions']
        )
        data[f"{crop} Lost Yield"] = np.abs(
            np.minimum(data[f"{crop} Yld"] - data[f"{crop} Insured Yield"], 0)
        )

        # Calculate insurance indemnity and total premiums
        data[f"{crop} Insurance Indemnity"] = (
                insurance_df.loc[crop]['Price Election'] *
                data[f"{crop} Lost Yield"] *
                selected_planted_acres[crop]
        )
        data[f"{crop} Total Premiums"] = (
                selected_insurance_info_df.loc[crop, 'Premiums $/acre'] *
                selected_planted_acres[crop]
        )

    # Calculate totals
    data["Total Market Receipts"] = sum(data[f"{crop} Market Receipts"].sum() for crop in crops)
    data["Total Insurance Indemnity"] = sum(data[f"{crop} Insurance Indemnity"] for crop in crops)
    data["Sum Premiums"] = sum(data[f"{crop} Total Premiums"] for crop in crops)


    outcomes = calculate_outcomes(data, selected_planted_acres, selected_production_costs)

    return outcomes




def calculate_outcomes(data, selected_planted_acres,selected_production_costs):
    # Initialize data_cost dictionary
    data_cost = {
        "Year": inflation_interest_rates["YEAR"],  # 2023 to 2027
        "annual_family_living_expenses": [annual_family_living_expenses],  # Start with 2022 values
        "farm_fixed_costs": [farm_fixed_costs],
        "labor_costs": [labor_costs],
        "machinery_costs": [annual_rent_machinery],
        'land costs' : [cash_rent_total]
    }

    # Dynamically add crop-specific keys
    for crop in crops:
        data_cost[f"{crop} Variable Costs $/Ac"] = []
        data_cost[f"{crop} Harvesting Cost ($/Ac)"] = []

    # Start with the 2022 values
    prev_family_living_expenses = annual_family_living_expenses
    prev_farm_fixed_costs = farm_fixed_costs
    prev_labor_costs = labor_costs
    prev_machinery_costs = annual_rent_machinery
    prev_land_costs = cash_rent_total
    nyear = len(inflation_interest_rates["YEAR"])

    # Loop through each year (2023 to 2027) to calculate costs based on inflation
    for i in range(nyear):  # Starting from 2023
        # Family living expenses
        new_family_living_expenses = prev_family_living_expenses * (1 + inflation_interest_rates["CPI_Family_Costs_Inflation_Rate"][i])
        data_cost["annual_family_living_expenses"].append(new_family_living_expenses)

        # Farm fixed costs
        new_farm_fixed_costs = prev_farm_fixed_costs * (1 + inflation_interest_rates["PPI_Fixed_Costs_Inflation_Rate"][i])
        data_cost["farm_fixed_costs"].append(new_farm_fixed_costs)

        # Labor costs
        new_labor_costs = prev_labor_costs * (1 + inflation_interest_rates["Labor_Cost_Inflation_Rate"][i])
        data_cost["labor_costs"].append(new_labor_costs)

        new_machinery_costs = prev_machinery_costs * (1 + inflation_interest_rates["Machine_Rent_Inflation_Rate"][i])
        data_cost["machinery_costs"].append(new_machinery_costs)

        new_land_costs = prev_land_costs* (1 + inflation_interest_rates["Land_Inflation_Rate"][i])
        data_cost["land costs"].append(new_land_costs)


        # Crop-specific calculations
        for crop in crops:
            # Variable costs (Seed, Fertilizer, Herbicide, Fuel, OtherVC)
            crop_variable_costs = sum([
                selected_production_costs.loc[crop]['Seed $/Ac'],
                selected_production_costs.loc[crop]['Fertilizers $/Ac'],
                selected_production_costs.loc[crop]['Herbicide $/Ac'],
                selected_production_costs.loc[crop]['Fuel $/Ac'],
                selected_production_costs.loc[crop]['Other VC $/Ac']
            ]) * (1 + inflation_interest_rates["Crop_Variable_Cost_Inflation_Rate"][i])
            data_cost[f"{crop} Variable Costs $/Ac"].append(crop_variable_costs)

            # Harvesting costs (in $/Ac)
            crop_harvesting_costs_per_ac = selected_production_costs.loc[crop]['Harvesting $/Ac'] * (1 + inflation_interest_rates["Harvest_Cost_Inflation_Rate"][i])
            data_cost[f"{crop} Harvesting Cost ($/Ac)"].append(crop_harvesting_costs_per_ac)

        # Update previous year values for the next iteration
        prev_family_living_expenses = new_family_living_expenses
        prev_farm_fixed_costs = new_farm_fixed_costs
        prev_labor_costs = new_labor_costs
        prev_machinery_costs = new_machinery_costs
        prev_land_costs = new_land_costs


    data_cost["annual_family_living_expenses"] = data_cost["annual_family_living_expenses"][1:nyear+1]
    data_cost["farm_fixed_costs"] = data_cost["farm_fixed_costs"][1:nyear+1] + annual_depreciation
    data_cost["labor_costs"] =data_cost["labor_costs"][1:nyear+1]
    data_cost["land costs"] =data_cost["land costs"][1:nyear+1]
    data_cost["machinery_costs"] =data_cost["machinery_costs"][1:nyear+1]

    prev_land_value= value_of_cropland_and_buildings
    land_value = {
        "Year": inflation_interest_rates["YEAR"],  # 2023 to 2027
        "land_value": [prev_land_value] # Start with 2022 values

    }

    for i in range(nyear):  # Starting from 2023
        # Family living expenses
        new_land_value= prev_land_value * (1 + inflation_interest_rates["Land_Inflation_Rate"][i])
        land_value["land_value"].append(new_land_value)

        prev_family_living_expenses = new_land_value


    land_value["land_value"] = land_value["land_value"][1:nyear+1]



    data_cost["variable_crop_costs"] = {
        crop: [cost * selected_planted_acres[crop] for cost in data_cost[f"{crop} Variable Costs $/Ac"]]
        for crop in crops
    }

    # Calculate crop harvest costs for each crop dynamically
    data_cost["Crop Harvest Costs"] = {
        crop:  [cost * selected_planted_acres[crop] for cost in data_cost[f"{crop} Harvesting Cost ($/Ac)"]]

        for crop in crops
    }

    for column in ['APH Yield Bu', 'Price Election']:
        for crop in crops:
            # Repeat the value for the length of years
            data[f"{crop} {column}"] = [insurance_df.loc[crop, column]] * len(years)


    for column in ['Insured Yield', 'Total Premiums']:
        for crop in crops:
            # Repeat values for the length of years
            data[f"{crop} {column}"] = [data[f"{crop} {column}"]] * len(years)
    # Handle the sum of premiums separately
    data['Sum Premiums'] = [data['Sum Premiums']] * len(years)


    df_balanced = pd.DataFrame(data)
    df_income_statement = pd.DataFrame({
        'Year': inflation_interest_rates["YEAR"]
    })


    columns_to_initialize = ['Market Receipts', 'Insurance Indemnity']
    for crop in crops:
        for column in columns_to_initialize:
            # Replace None with data from an external calculation or existing data
            df_income_statement[f"{crop} {column}"] = data[f"{crop} {column}"]

    # Initialize other general columns
    general_columns = ['Total Receipts']
    for column in general_columns:
        df_income_statement[column] = None


    # print(df_income_statement)


    # Define the base columns that are not crop-specific
    base_columns = [
        'Insurance Premiums', 'Fixed Costs', 'Labor Costs',
        'Land Rent', 'Machinery Rent', 'Sum Total Cash Costs',
        'Operating Loan Interest', 'Cash Flow Deficit Interest',
        'Land Loan Interest', 'Total Cash Expenses','Net cash Farm Income'
    ]

    # Dynamically add crop-specific columns
    crop_specific_columns = []
    for crop in crops:
        crop_specific_columns.extend([f"{crop} VC", f"{crop} Harvest"])

    # Combine all columns
    all_columns = ['Year'] + crop_specific_columns + base_columns

    # Create the DataFrame with None values initialized
    df_expenses = pd.DataFrame({col: [None] * len(inflation_interest_rates["YEAR"]) for col in all_columns})

    # Add the 'Year' column
    df_expenses['Year'] = inflation_interest_rates["YEAR"]


    cash_flow_statement = pd.DataFrame({
        'Year': years,
        'Beginning Cash Reserves': [None] * len(years),
        'Net Cash Farm Income': [None] * len(years),
        'Interest Earned': [None] * len(years),
        'Total Cash Inflows': [None] * len(years),
        'Family Living Expenses': [None] * len(years),
        'Principal Payment Land Loan': [None] * len(years),
        'Income Taxes': [None] * len(years),
        'Repay Cash Flow Deficit Loans': [None] * len(years),
        'Total Cash Outflows': [None] * len(years),
        'Ending Cash Dec 31': [None] * len(years),
    })

    balance_sheet = pd.DataFrame({
        'Year': years,
        'Ending Cash Reserves Dec 31': [None] * len(years),
        'Land Value': [None] * len(years),
        'Total Assets': [None] * len(years),
        'Land Debt': [None] * len(years),
        'Machinery Debt': [None] * len(years),
        'Cash Flow Deficits': [None] * len(years),
        'Total Liabilities': [None] * len(years),
        'Net worth': [None] * len(years)
    })

    discount_factors = [None] * len(years)
    pv_family_living = [None] * len(years)

    for i, year in enumerate(inflation_interest_rates["YEAR"]):
        if i==0:
            last_year_cash_flow_deficits= first_year_cash_flow_deficits

        # Calculate total receipts dynamically for all crops
        total_market_receipts = sum(data[f"{crop} Market Receipts"][i] for crop in crops) + Other_revenue_1
        total_insurance_indemnity = sum(data[f"{crop} Insurance Indemnity"][i] for crop in crops)



        # Calculate total receipts
        total_receipts = total_market_receipts + total_insurance_indemnity

        # Fill the income statement for the current year
        df_income_statement.loc[i, 'Year'] = year
        df_income_statement.loc[i, 'Total Receipts'] = float(total_receipts)

        # Fill dynamically calculated columns for each crop
        for crop in crops:
            df_income_statement.loc[i, f"{crop} Market Receipts"] = data[f"{crop} Market Receipts"][i]
            df_income_statement.loc[i, f"{crop} Insurance Indemnity"] = data[f"{crop} Insurance Indemnity"][i]

        df_expenses.loc[i, 'Year'] = year

        # Assign crop-specific variable and harvest costs
        for crop in crops:
            df_expenses.loc[i, f"{crop} VC"] = data_cost['variable_crop_costs'][crop][i]
            df_expenses.loc[i, f"{crop} Harvest"] = data_cost['Crop Harvest Costs'][crop][i]

        # Assign base columns
        df_expenses.loc[i, 'Insurance Premiums'] = data.get("Sum Premiums", [0])[i]
        df_expenses.loc[i, 'Fixed Costs'] = data_cost['farm_fixed_costs'][i]
        df_expenses.loc[i, 'Labor Costs'] = data_cost['labor_costs'][i]
        df_expenses.loc[i, 'Land Rent'] = data_cost['land costs'][i]
        df_expenses.loc[i, 'Machinery Rent'] = data_cost['machinery_costs'][i]

        # Calculate Sum Total Cash Costs
        total_variable_costs = sum(df_expenses.loc[i, f"{crop} VC"] for crop in crops)
        total_harvest_costs = sum(df_expenses.loc[i, f"{crop} Harvest"] for crop in crops)
        df_expenses.loc[i, 'Sum Total Cash Costs'] = (
            total_variable_costs +
            total_harvest_costs +
            df_expenses.loc[i, 'Insurance Premiums'] +
            df_expenses.loc[i, 'Fixed Costs'] +
            df_expenses.loc[i, 'Labor Costs'] +
            df_expenses.loc[i, 'Land Rent'] +
            df_expenses.loc[i, 'Machinery Rent']
        )

        # Calculate Operating Loan Interest
        df_expenses.loc[i, 'Operating Loan Interest'] = (
            df_expenses.loc[i, 'Sum Total Cash Costs'] *
            inflation_interest_rates["Local_Interest_Rates"][i] *
            operating_loan_fraction_year
        )

        # Calculate Cash Flow Deficit Interest
        df_expenses.loc[i, 'Cash Flow Deficit Interest'] = (
            last_year_cash_flow_deficits*
            inflation_interest_rates["Local_Interest_Rates"][i]
        )

        # Assign Land Loan Interest (placeholder, adjust as needed)


        remaining_principal, interest_expense, principal_payment = loan_balance_and_interest(
            outstanding_loans_amount , loan_interest_rate , loan_term_years, (int(current_loan_year)+i)
        )


        df_expenses.loc[i, 'Land Loan Interest'] = interest_expense

        df_expenses.loc[i, 'Total Cash Expenses'] = (
                df_expenses.loc[i, 'Sum Total Cash Costs'] +
                df_expenses.loc[i, 'Operating Loan Interest'] +
                df_expenses.loc[i, 'Cash Flow Deficit Interest']+
                df_expenses.loc[i, 'Land Loan Interest']
        )

        df_expenses.loc[i, 'Net cash Farm Income'] = (
                df_income_statement.loc[i, 'Total Receipts'] -
                df_expenses.loc[i, 'Total Cash Expenses']
        )

        if i == 0:
            cash_flow_statement.loc[i, 'Beginning Cash Reserves'] = cash_on_hand
        else:
            cash_flow_statement.loc[i, 'Beginning Cash Reserves'] = balance_sheet.loc[i - 1, 'Ending Cash Reserves Dec 31']

        cash_flow_statement.loc[i, 'Net Cash Farm Income'] = (
                    df_expenses.loc[i, 'Net cash Farm Income']
            )

        cash_flow_statement.loc[i, 'Interest Earned'] = cash_flow_statement.loc[i, 'Beginning Cash Reserves'] * float(interest_rate_savings)


            # Total Cash Inflows
        cash_flow_statement.loc[i, 'Total Cash Inflows'] = (
                cash_flow_statement.loc[i, 'Net Cash Farm Income'] +
                cash_flow_statement.loc[i, 'Interest Earned'] +
                cash_flow_statement.loc[i, 'Beginning Cash Reserves']
        )

        # Total Cash Outflows
        cash_flow_statement.loc[i, 'Family Living Expenses'] = (
            data_cost['annual_family_living_expenses'][i]
        )

        cash_flow_statement.loc[i, 'Family Living Expenses'] = (
            data_cost['annual_family_living_expenses'][i]
        )

        cash_flow_statement.loc[i, 'Principal Payment Land Loan'] = principal_payment

        #**** Update the taxable_income!!
        # taxable_income = cash_flow_statement.loc[i, 'Net Cash Farm Income'] - deductions

        # taxable_income = cash_flow_statement.loc[i, 'Net Cash Farm Income']/3
        income_taxes = calculate_tax_from_bracket(cash_flow_statement.loc[i, 'Net Cash Farm Income'])
        cash_flow_statement.loc[i, 'Income Taxes'] = income_taxes
        cash_flow_statement.loc[i, 'Repay Cash Flow Deficit Loans'] = last_year_cash_flow_deficits


        cash_flow_statement.loc[i, 'Total Cash Outflows'] = (
                cash_flow_statement.loc[i, 'Family Living Expenses'] +
                cash_flow_statement.loc[i, 'Principal Payment Land Loan'] +
                cash_flow_statement.loc[i, 'Income Taxes'] +
                cash_flow_statement.loc[i, 'Repay Cash Flow Deficit Loans']
        )

        # Ending Cash Dec 31

        cash_flow_statement.loc[i, 'Ending Cash Dec 31'] = (
                cash_flow_statement.loc[i, 'Total Cash Inflows'] -
                cash_flow_statement.loc[i, 'Total Cash Outflows']
        )



        # Ending Cash Reserves (from Cash Flow Statement)
        balance_sheet.loc[i, 'Ending Cash Reserves Dec 31']= max(cash_flow_statement.loc[i, 'Ending Cash Dec 31'],0)

        # Land Value (constant or with appreciation)
        land_value_current = land_value["land_value"][i] # Assume constant; update if there's appreciation
        balance_sheet.loc[i, 'Land Value'] = land_value_current

        balance_sheet.loc[i, 'Total Assets']= (
                balance_sheet.loc[i, 'Ending Cash Reserves Dec 31'] +
                balance_sheet.loc[i, 'Land Value']
        )

        balance_sheet.loc[i, 'Land Debt'] = remaining_principal
        balance_sheet.loc[i, 'Machinery Debt'] = Machinery_Debt
        balance_sheet.loc[i, 'Cash Flow Deficits'] = max(-balance_sheet.loc[i, 'Ending Cash Reserves Dec 31'],0)
        balance_sheet.loc[i, 'Total Liabilities'] = (
                balance_sheet.loc[i, 'Land Debt'] +
                balance_sheet.loc[i, 'Machinery Debt']+
                balance_sheet.loc[i, 'Cash Flow Deficits'])
        balance_sheet.loc[i, 'Net worth'] = balance_sheet.loc[i, 'Total Assets'] - balance_sheet.loc[i, 'Total Liabilities']


        last_year_cash_flow_deficits = balance_sheet.loc[i, 'Cash Flow Deficits']

        discount_factors[i] = 1 / ((1 + discount_rate) ** (i+1))
        pv_family_living[i] = discount_factors[i] * cash_flow_statement.loc[i, 'Family Living Expenses']

    PV_Ending_Net_Worth= balance_sheet.loc[balance_sheet.index[-1], 'Net worth']*discount_factors[-1]


    # Remain principal of the year 0
    if current_loan_year >= 1:
        Begin_Land_Debt_Remain = loan_balance_and_interest(
        outstanding_loans_amount ,
        loan_interest_rate,
        int(loan_term_years),
        int(current_loan_year - 1))[0]
    else:
        Begin_Land_Debt_Remain = 0




    Begin_Net_Worth = cash_on_hand +value_of_cropland_and_buildings - Begin_Land_Debt_Remain
    NPV = -Begin_Net_Worth + sum(pv_family_living) +  PV_Ending_Net_Worth


    Pr_EC_less_zero_indi = (cash_flow_statement['Ending Cash Dec 31'] < 0)
    Pr_EC_conse_less_zero_indi = consecutive_loss_probability(cash_flow_statement['Ending Cash Dec 31'])
    Pr_EC_greater_zero = (cash_flow_statement['Ending Cash Dec 31'] > 0).sum()/nyear
    Pr_EC_conse_less_zero = probability_consecutive_negatives(cash_flow_statement['Ending Cash Dec 31'])

    IRR_df = pd.DataFrame({
        'Year': years,
        'IRR Values': [None] * len(years),
        'PV NetReturns': [None] * len(years),
    })
    #Benefits-Costs Ratio & Internal Rate of Return

    for i, year in enumerate(inflation_interest_rates["YEAR"]):
        if i == 0:
            IRR_df.loc[i, "IRR Values"] = -outstanding_loans_amount - data_cost["land costs"][0]
        else:
            IRR_df.loc[i, "IRR Values"] = cash_flow_statement.loc[i - 1, 'Net Cash Farm Income']
        IRR_df.loc[i, "PV NetReturns"] = cash_flow_statement.loc[i, 'Net Cash Farm Income'] * discount_factors[i]

    IRR_df.loc[i+1, "IRR Values"] = cash_flow_statement.loc[i - 1, 'Net Cash Farm Income']
    IRR_rate = npf.irr(IRR_df["IRR Values"])

    BC_ratio = sum(IRR_df["PV NetReturns"][:-1])/(outstanding_loans_amount + data_cost["land costs"][0])
    # print(outstanding_loans_amount)
    # print(data_cost["land costs"][0])
    # print(BC_ratio)
    NPV_alt = -outstanding_loans_amount + sum(IRR_df["PV NetReturns"][0:-1])
    # payback_period = calculate_payback_period(outstanding_loans_amount, IRR_df["PV NetReturns"].iloc[:-1])


    outcomes = {
            "NPV_from_Networth": NPV,
            "NPV_from_Netreturns": NPV_alt,
            "IRR": IRR_rate,
            # "Payback Period (Years)": str(payback_period),
            "P(EC<0 one year)":Pr_EC_less_zero_indi,
            "P(EC<0 two years)":Pr_EC_conse_less_zero_indi,

            "Ending cash each year":cash_flow_statement['Ending Cash Dec 31'],
            "Net cash Farm Income each year": df_expenses['Net cash Farm Income'],
            "PV Ending Net Worth": PV_Ending_Net_Worth,
            "BC Ratio": BC_ratio
        }
    return outcomes



file_path = "Assumption.xlsx"
all_sheets = pd.read_excel(file_path, sheet_name=None)

# Display sheet names
# print(all_sheets.keys())  # Lists all available sheet names

# Access a specific sheet by name (example: 'Sheet1')
df_sheet1 = all_sheets['Assumption (1)'].T
df_sheet1.columns = df_sheet1.iloc[0]  # Use first row as column names
df_sheet1 = df_sheet1.iloc[1:].reset_index(drop=True)  # Remove the first row and reset index
# print(df_sheet1.head())  # Print first few rows


params = {
    "first_year": int(df_sheet1['first_year'].iloc[0]),  # Convert to integer
    "last_year": int(df_sheet1['last_year'].iloc[0]),  # Convert to integer
}

# Convert all other numerical values to float
numeric_keys = [
    "cropland_acres_owned", "cropland_acres_leased", "cash_rent_per_acre",
    "cash_rent_total", "cash_on_hand", "price_of_owned_cropland_acres", "value_of_cropland_and_buildings",
    "annual_depreciation", "annual_rent_machinery", "outstanding_loans_amount",
    "loan_start_year", "loan_term_years", "loan_interest_rate", "annual_family_living_expenses",
    "farm_fixed_costs", "labor_costs", "discount_rate", "operating_loan_wedge",
    "operating_loan_fraction_year", "interest_rate_savings", "Fraction_unsold",
    "first_year_cash_flow_deficits", "Machinery_Debt", "Other_revenue_1"
]

for key in numeric_keys:
    params[key] = df_sheet1[key].astype(float).iloc[0]

unpack_params(params)
years_len = last_year -first_year+1

# Define scenarios to process
scenarios = ['Base', 'Alt1', 'Alt2', 'Alt3']

# Prepare a dict to store the results
cost_data_scenarios = {}

# Process each scenario
for scenario in scenarios:
    sheet_name = f"{scenario}_cost"
    df = all_sheets[sheet_name]

    # Check if non-zero and non-NaN data exists (except for Base which is always used)
    if scenario == "Base" or not ((df.iloc[:, 1:] == 0) | (df.iloc[:, 1:].isna())).all().all():
        df = df.set_index("Crop")
        cost_data_scenarios[scenario] = {col: df[col].tolist() for col in df.columns}
        cost_data_scenarios[scenario]["Crop"] = df.index.tolist()

scenarios =   list(cost_data_scenarios.keys())
scenario_list =   list(cost_data_scenarios.keys())

# Collect all crops
# Step 1: Get all crops
all_crops = cost_data_scenarios["Base"]["Crop"]

# Step 2: Find crops where at least one scenario has non-zero value
crops = []

for i, crop in enumerate(all_crops):
    # Check if this crop has any non-zero value across all scenarios
    has_non_zero = False
    for scenario in scenario_list:
        for col in cost_data_scenarios[scenario]:
            if col != "Crop" and cost_data_scenarios[scenario][col][i] != 0:
                has_non_zero = True
                break
        if has_non_zero:
            break
    if has_non_zero:
        crops.append(crop)
Insurance_alternatives_frac = all_sheets['Insurance_alternatives_frac']
Insurance_alternatives_frac.set_index('Crop', inplace=True)
Insurance_alternatives_prem = all_sheets['Insurance_alternatives_prem']
Insurance_alternatives_prem.set_index('Crop', inplace=True)



# Get crop list and scenario list dynamically
# crops = Insurance_alternatives_frac.index.tolist()
# scenarios = Insurance_alternatives_frac.columns.tolist()

# Build dictionary automatically
Insurance_selection_alternatives = {
    "Crop": crops,
    "Yield Fractions": [
        {scenario: Insurance_alternatives_frac.loc[crop, scenario] for scenario in scenarios}
        for crop in crops
    ],
    "Premiums $/acre": [
        {scenario: Insurance_alternatives_prem.loc[crop, scenario] for scenario in scenarios}
        for crop in crops
    ]
}

insurance_df = all_sheets['Insurance_aph_price']
insurance_df.set_index('Crop', inplace=True)


df_inflation = all_sheets['inflation_rates']

# Transpose the dataframe
df_inflation = df_inflation.T

# Set the first row as column names
df_inflation.columns = df_inflation.iloc[0]

# Drop the first row since it is now the column headers
df_inflation = df_inflation[1:].reset_index()

# Rename the first column to "YEAR"
df_inflation.rename(columns={"index": "YEAR"}, inplace=True)
for column in df_inflation.columns[1:]:
    df_inflation[column] = (1 + df_inflation[column]).cumprod() - 1
# Print to check structure
# print(df_inflation.head())
# Initialize the dictionary with "Year" column as the first entry

# Assign values from df_inflation to the dictionary for remaining columns
column_names = df_inflation.columns.tolist()
inflation_interest_rates = {}

for col in column_names:
    if col != "Year":
        inflation_interest_rates[col] = df_inflation[col].tolist()




local_interest_rates = [rate + operating_loan_wedge for rate in inflation_interest_rates["Natl_Prime_Interest_Rate"]]
inflation_interest_rates["Local_Interest_Rates"] = local_interest_rates



df_planted_acres = all_sheets['planted_acres']
if df_planted_acres.index.name != 'Crop':
    df_planted_acres.set_index('Crop', inplace=True)

# crops = df_planted_acres.index.tolist()
# scenarios = df_planted_acres.columns.tolist()

planted_acres = {
    crop: {scenario: df_planted_acres.at[crop, scenario] for scenario in scenarios}
    for crop in crops
}
for j in range(1, len(scenario_list) + 1):
    selected_scenario = scenario_list[j - 1]



    cost_data = cost_data_scenarios[selected_scenario]

    crop_costs_base = pd.DataFrame(cost_data)

    # print(crop_costs_base)

    crop_costs_base.set_index('Crop', inplace=True)
    # You can modify your insurance information and available insurance options here
    current_loan_year = first_year - loan_start_year + 1


    # selected_planted_acres = {crop: acres[selected_scenario] for crop, acres in planted_acres.items()}

    #Stochastic Prices and Yields

    #Add loop if want to simulate more than once

    # np.random.seed(42)

    # # Define the number of years and crops
    data_hist= pd.read_csv("data.csv")

    # Set the first column as 'year' (assuming the CSV has headers)
    # data['year'] = data.iloc[:, 0]



    years = range(first_year, last_year+1)  # From 2023 to 2027





    # predicted_values = draw_samples_from_scenario(data_hist, size=5)
    # print(predicted_values)

    num_year =last_year- first_year +1




    # Extract the last four columns and store them in the dictionary

    # print(sampled_values1)


    def initialize_global_variables():
        global crop_costs_base, insurance_df, inflation_interest_rates,Other_revenue_1
        global local_interest_rates, selected_scenario,data_hist
        global crops, years, sampled_values, Insurance_selection_alternatives
        global planted_acres, crop_costs_base,current_loan_year

    initialize_global_variables()






    # Example usage
    #Setting the simulation times
    num_simulations = 1000
    final_outcomes_mean = simulate_scenarios(num_simulations, sim_selected_scenario)
    formatted_output = format_outcomes(final_outcomes_mean)
    print(formatted_output)

    file_path = f"{selected_scenario}_output.txt"

    with open(file_path, "w") as file:
        file.write(formatted_output)

