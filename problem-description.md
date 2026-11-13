# Millennium BCP-FEP Collaboration: Data Science Challenge
## Predicting Daily Account Balances

### Problem Statement
In the modern banking landscape, staying ahead of the curve is essential for financial institutions seeking to provide excellent service and maintain a competitive edge. One of the key challenges facing banks is accurately predicting the balance of each client's account on a daily basis in order to:
* Enhance risk management.
* Improve customer service with more personalized and proactive offerings.
* Provide better financial planning tools for clients.

### Goal
Your task is to develop a data-driven model that can accurately predict the daily account balance for each customer for the **next 30 days** based on historical data.

### Dataset Description
* **Structure:** The dataset is organized by client and date.
* **Rows:** Each row represents a time series of balances (in Euro, €) for an individual client account.
* **Columns:** Each column corresponds to a specific day.

### Data Availability
You have access to historical account balance data for a group of customers. Use this dataset to train and validate your model. The final evaluation will involve predicting the daily account balances for the next 30 days for each customer.

### Evaluation Metric
Model performance will be evaluated using the **Mean Absolute Percentage Error (MAPE)** metric:

$$\text{MAPE} = \frac{100\%}{n} \sum_{t=1}^{n} \left| \frac{A_t - F_t}{A_t} \right|$$

Where:
* $A_t$ is the actual value.
* $F_t$ is the forecast value.
* $n$ is the number of fitted points.

*Note: A lower MAPE indicates a more accurate model, so your objective is to minimize this metric.*