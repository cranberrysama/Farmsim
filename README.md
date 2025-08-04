# Crop Simulation Program

This program performs scenario-based farm-level financial simulation to estimate farm profitability, taking into account crop sales revenue, production costs, insurance alternatives, and stochastic variations in yields and prices.

## Requirements
---

To run the program, execute `Runit.bat`.  Please make sure that both Excel files are placed in the same folder as the `.py` and `.bat` files.
If the file appears as `Runit.txt` instead of `Runit.bat`, simply rename it to `Runit.bat`.

Before running the `.bat` file, please make sure:

- You have installed Python 3.x.
- Required Python packages:
    - numpy
    - pandas
    - scipy
    - statsmodels
    - numpy_financial
    - openpyxl

You can install them via:

```bash
pip install numpy pandas scipy statsmodels numpy_financial openpyxl
```

## Model Assumptions
---

- **Depreciation**: Constant annual depreciation is applied to buildings and machinery, but cropland is not depreciated (need update mannually in Assumption.xlsx).
- **Other Revenue**: Includes additional income such as cotton seed sales (need update mannually in Assumption.xlsx).
- **Other Assumptions**:  
  For additional assumptions regarding tax computation, interest-related computation, stochastic generation of prices and yields, and the construction of financial statements (including profitability and NPV calculation), please refer to JC's Excel file.

  `FarmSim_Eco-HAWQS_Model_Test2.xlsx`


## Required Excel Sheets
---


### The file `data.csv`

- Stores historical data for stochastic sampling.
- You may add more crops and additional historical records to this file.
- The column names should follow the format:
    - `"Crop3 Prc ($/Bu)"`
    - `"Crop3 Yld (Bu/Ac)"`
    - ...
- It is acceptable to change the units (e.g., from Bu/Ac to kg/ha), but ensure that the units are consistent with those used in your other input data (such as cost, insurance, and yield data).
- `data.csv` corresponds to rows 3 to 13 of the **'Stoch'** sheet in JC's Excel file.

### The file `Assumption.xlsx`

This file stores all the assumption-related data. The following sheets are required:

| Sheet Name | Description |
|------------|-------------|
| Assumption(1) | Farm information |
| Base_cost | Base scenario crop production costs |
| Alt1_cost | Alternative scenario 1 costs (optional; leave blank if not used) |
| Alt2_cost | Alternative scenario 2 costs (optional; leave blank if not used) |
| Alt3_cost | Alternative scenario 3 costs (optional; leave blank if not used) |
| Insurance_aph_price | Insurance Actual Production History (APH) price |
| Incurance_alternatives_frac | Insurance yield fractions for all scenarios |
| Incurance_alternatives_prem | Insurance premiums for all scenarios |
| inflation_rates | Annual inflation rates |
| Planted_acres | Base and alternative scenario planted acres |

- In the newly added file “Assumption with Notes,” the corresponding locations in JC’s file are indicated.


## Notes
---

- Do not change row names or column names unless you intend to predict additional years or include more crops.
- Crops with all-zero values will be automatically removed from the analysis.
- To predict farm profitability for additional years (default is 5):
    - Update the `first_year` and `last_year` settings in the `Assumption(1)` sheet.
    - Make sure to provide the corresponding additional years' inflation rates in the `inflation_rates` sheet.
- To include more crops in the model:
    - Modify `data.csv` accordingly.
    - Provide all necessary information for the new crops in `Assumption.xlsx` (costs, insurance, acres, etc.).
- To include more scenarios:
    - Provide cost information in `AltX_cost` sheets (e.g., `Alt4_cost`).
    - Currently, the maximum number of scenarios supported is four: `Base`, `Alt1`, `Alt2`, and `Alt3`.

## Output Description
---

After executing `runit.bat`, you will obtain several output files (e.g., `Base_output.txt`, `Alt1_output.txt`, etc.).

Each output file reports farm-level financial performance under the corresponding scenario.  
The results are based on **1,000 stochastic simulations**, and the reported values are the **average across all simulations**.

> For details of calculation procedures, please refer to JC's Excel file:  
> `FarmSim_Eco-HAWQS_Model_Test2.xlsx`

### Output Variables

- **NPV_from_Networth**:  
    Net Present Value calculated from the present value of the farm's ending net worth.

- **IRR**:  
    Internal Rate of Return of the simulated farm's cash flow.

- **P(EC<0 one year)**:  
    Probability of experiencing negative ending cash during the simulation period (reported year by year).

- **P(EC<0 two years)**:  
    Probability of experiencing negative ending cash in two consecutive years during the simulation period (reported for each two-year period).

- **Ending cash each year**:  
    Average ending cash balance at the end of each year over the simulation horizon.

- **Net cash Farm Income each year**:  
    Average net cash farm income generated in each year.

- **PV Ending Net Worth**:  
    Present value of the farm's ending net worth.

- **BC Ratio**:  
    Benefit-Cost Ratio calculated as the ratio between the present value of benefits and the present value of costs.

## Contact
---


## ✅ Update: August 4, 2025

### 🔄 Run Simulation
- Execute `Simulation_Farm_2025.py`
- Toggle the stochastic process by **commenting/uncommenting**:
  - Line **401**: stochastic
  - Lines **403–410**: costant number

### ⚠️ Important:
- Don’t forget to set the number of simulations:
  ```python
  num_simulations = 1000
### 📌 Assumptions:
- Assumed inflation rate:
  ```python
  Crop_Variable_Cost_Inflation_Rate = Harvest_Cost_Inflation_Rate （can be modified）
  ```

If you have any questions, please contact:  
mengqiaoliu@tamu.edu
