Monitoring household electricity consumption is essential for identifying wasteful patterns and reducing energy costs. However, manually analyzing thousands of hourly readings across multiple appliances is impractical. Combining AI-powered analysis with a high-performance database like [GridDB](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) allows you to store, retrieve, and analyze electricity data using natural language queries.

In this article, you will see how to build an AI-powered electricity consumption chatbot using the [Anthropic](https://www.anthropic.com/) Claude API and GridDB Cloud. The chatbot analyzes 10,000 hourly appliance-level readings, identifies consumption trends, pinpoints peak and off-peak hours for each appliance, and provides data-backed recommendations to save electricity. We will also use AI to create data visualizations showing electricity consumption trends.

GridDB’s efficient handling of time-series data, combined with Claude’s analytical reasoning, provides a powerful way to turn raw electricity readings into actionable insights.

**Prerequisites**:
You will need the following to run scripts in this article:

* A [GridDB cloud account](https://www.global.toshiba/ww/products-solutions/ai-iot/griddb/product/griddb-cloud.html). Refer to this [quick start guide](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) to set up a GridDB cloud account.
* [Anthropic API Key](https://platform.claude.com/). You can obtain one from the Anthropic console.

Note: You can find the complete code for this tutorial in my [GridDB Blogs GitHub repository](https://github.com/usmanmalik57/GridDB-Blogs/tree/main/Electricity%20Consumption%20Monitoring%20Chatbot%20with%20GridDB%20and%20Anthropic%20Claude).


## Installing and Importing Required Libraries

The following installs the libraries you will need to run the scripts in this article.

```
!pip install python-dotenv
!pip install anthropic
```

The script below imports the required libraries.

```python
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
import requests
import json
import re

from dotenv import load_dotenv
load_dotenv()
```

## Importing the Dataset

The dataset used in this article provides hourly electricity consumption data recorded at the appliance level. It includes readings for six appliances: air conditioner (ac), fridge, lights, fans, washing machine, and TV. Each row represents one hourly reading for a household, along with metadata such as the season and any ongoing festival. You can [download the dataset from Kaggle](https://www.kaggle.com/datasets/frank451995/appliance-wise-hourly-electricity-consumption?resource=download).

The following script loads the dataset and selects the first 10,000 records. The original dataset contains more than 400k records; however, 10k records are sufficient for this sample analysis.

```python
dataset = pd.read_csv("appliance_usage_dataset.csv",
                      encoding = 'utf-8')
dataset = dataset[:10000]
print(dataset.shape)
dataset.head()
```

**Output:**

<img src="images\img1-dataset.png">


## Inserting Data in GridDB

We will insert the data into GridDB Cloud.

### Creating a GridDB Connection

Run the following script to test your connection with the GridDB cloud. You can retrieve the connection credentials from your GridDB cloud account.

```python
username = os.environ.get("username")
password = os.environ.get("password")
base_url = os.environ.get("base_url")


url = f"{base_url}/checkConnection"

credentials = f"{username}:{password}"
encoded_credentials = base64.b64encode(credentials.encode()).decode()

headers = {
    'Content-Type': 'application/json',  # Added this header to specify JSON content
    'Authorization': f'Basic {encoded_credentials}',
    'User-Agent': 'PostmanRuntime/7.29.0'
}

response = requests.get(url, headers=headers)

print(response.status_code)

```

**Output:**
```
200
```

If you do not see the above response, check this [quick start guide](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) for troubleshooting.

### Creating a GridDB Container for Electricity Consumption Data

Before inserting data, we need to create a container in GridDB with the correct schema. The following script maps pandas data types to GridDB types.

```python
dataset.insert(0, "SerialNo", dataset.index + 1)
dataset.columns.name = None   
# Mapping pandas dtypes to GridDB types
type_mapping = {
    "int64":          "LONG",
    "float64":        "DOUBLE",
    "bool":           "BOOL",
    'datetime64': "TIMESTAMP",
    "object":         "STRING",
    "category":       "STRING",
}

# Generate the columns part of the payload dynamically
columns = []
for col, dtype in dataset.dtypes.items():
    griddb_type = type_mapping.get(str(dtype), "STRING")  # Default to STRING if unknown
    columns.append({
        "name": col,
        "type": griddb_type
    })

print(columns)
```

**Output:**

<img src="images\img2-pandas-griddb-mapping-types.png">

Next, we create the container using the GridDB REST API.

```python
container_name = "electricity_consumption_db"

url = f"{base_url}/containers"

# Create the payload for the POST request
payload = json.dumps({
    "container_name": container_name,
    "container_type": "COLLECTION",
    "rowkey": True,  # Assuming the first column as rowkey
    "columns": columns
})


# Make the POST request to create the container
response = requests.post(url, headers=headers, data=payload)

# Print the response
print(f"Status Code: {response.status_code}")
```

**Output:**
```
Status Code: 201
```

The 201 status code confirms the container has been created successfully.

### Inserting Data in Electricity Consumption GridDB Container

With the container created, we can now insert the 10,000 rows of electricity consumption data. The `format_row` function handles type conversions, including converting NaN values to None for GridDB compatibility.

```python
url = f"{base_url}/containers/{container_name}/rows"
# Convert dataset to list of lists (row-wise) with proper formatting

def format_row(row):
    formatted = []
    for item in row:
        if pd.isna(item):
            formatted.append(None)  # Convert NaN to None
        elif isinstance(item, bool):
            formatted.append(str(item).lower())  # Convert True/False to true/false
        elif isinstance(item, (int, float)):
            formatted.append(item)  # Keep integers and floats as they are
        else:
            formatted.append(str(item))  # Convert other types to string
    return formatted

# Prepare rows with correct formatting
rows = [format_row(row) for row in dataset.values.tolist()]

# Create payload as a JSON string
payload = json.dumps(rows)

# Make the PUT request to add the rows to the container
response = requests.put(url, headers=headers, data=payload)

# Print the response
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")
```

**Output:**
```
Status Code: 200
Response Text: {"count":10000}
```

At this point, all 10,000 records have been successfully inserted into the GridDB container.


## Using AI to Monitor Electricity Consumption and Get Recommendations

Now that the data is stored in GridDB, we can retrieve it and use the Anthropic Claude API to analyze consumption patterns and provide recommendations. The approach involves three steps: retrieve data from GridDB, build a pre-computed summary of the dataset, and pass that summary to Claude as context for intelligent Q&A.


### Retrieve Data from GridDB
The following script retrieves all 10,000 records from the GridDB container and loads them into a pandas DataFrame.


```python  

url = f"{base_url}/containers/{container_name}/rows"

# Define the payload for the query
payload = json.dumps({
    "offset": 0,           # Start from the first row
    "limit": 10000,         # Limit the number of rows returned
    "condition": "",       # No filtering condition (you can customize it)
    "sort": ""             # No sorting (you can customize it)
})

# Make the POST request to read data from the container
response = requests.post(url, headers=headers, data=payload)

# Check response status and print output
print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    try:
        data = response.json()
        print("Data retrieved successfully!")

        # Convert the response to a DataFrame
        rows = data.get("rows", [])
        electricity_consumption_db = pd.DataFrame(rows, columns=[col for col in dataset.columns])

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON response.")
else:
    print(f"Error: Failed to query data from the container. Response: {response.text}")

print(electricity_consumption_db.shape)
electricity_consumption_db.head()
```

**Output:**

<img src="images\img3-griddb-retrieved-data.png">

The data has been successfully retrieved from GridDB and is ready for analysis.


### Build Dataset Summary

Since we cannot send 10,000 raw rows to Claude in a single prompt because it may exceed token limits and reduce response quality, we pre-compute comprehensive aggregations and pass those as context instead. This way, Claude reasons over actual statistics computed from all 10,000 records.

```python
import pandas as pd

APPLIANCE_COLS = ['ac', 'fridge', 'lights', 'fans', 'washing_machine', 'tv']

# Prepare the dataframe
electricity_consumption_db['timestamp'] = pd.to_datetime(electricity_consumption_db['timestamp'])
electricity_consumption_db['Hour'] = electricity_consumption_db['timestamp'].dt.hour
electricity_consumption_db['DayOfWeek'] = electricity_consumption_db['timestamp'].dt.day_name()
electricity_consumption_db['Month'] = electricity_consumption_db['timestamp'].dt.month
electricity_consumption_db['total_kwh'] = electricity_consumption_db[APPLIANCE_COLS].sum(axis=1)


def build_summary(df):
    parts = []

    # Overview
    parts.append(f"Records: {len(df)} hourly readings across {df['house_id'].nunique()} households")
    parts.append(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    parts.append(f"Seasons: {df['season'].unique().tolist()}")
    parts.append(f"Festivals: {df['festival'].unique().tolist()}")

    # Appliance stats
    parts.append(f"\n--- Appliance Stats (kWh) ---\n{df[APPLIANCE_COLS].describe().round(3).to_string()}")

    # Total energy ranking
    totals = df[APPLIANCE_COLS].sum().sort_values(ascending=False)
    pct = (totals / totals.sum() * 100).round(1)
    parts.append(f"\n--- Energy Ranking ---")
    for app in totals.index:
        parts.append(f"  {app}: {totals[app]:.1f} kWh ({pct[app]}%)")

    # Hourly patterns
    hourly = df.groupby('Hour')[APPLIANCE_COLS].mean().round(4)
    hourly_total = df.groupby('Hour')['total_kwh'].mean().round(2)
    parts.append(f"\n--- Hourly Avg Consumption (kWh) ---\n{hourly.to_string()}")
    parts.append(f"\nHourly total avg:\n{hourly_total.to_string()}")
    parts.append(f"Lowest hour: {hourly_total.idxmin()}:00 ({hourly_total.min()} kWh)")
    parts.append(f"Highest hour: {hourly_total.idxmax()}:00 ({hourly_total.max()} kWh)")

    # Peak per appliance
    parts.append(f"\n--- Peak/Low Hour per Appliance ---")
    for col in APPLIANCE_COLS:
        parts.append(f"  {col}: peak {hourly[col].idxmax()}:00 ({hourly[col].max()}), low {hourly[col].idxmin()}:00 ({hourly[col].min()})")

    # Season patterns
    seasonal = df.groupby('season')[APPLIANCE_COLS].mean().round(4)
    parts.append(f"\n--- Seasonal Avg (kWh) ---\n{seasonal.to_string()}")

    # Day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily = df.groupby('DayOfWeek')['total_kwh'].mean().reindex(day_order).round(2)
    parts.append(f"\n--- Day of Week Avg Total (kWh) ---\n{daily.to_string()}")

    # Household comparison (top 5 and bottom 5)
    hh = df.groupby('house_id')['total_kwh'].mean().sort_values(ascending=False).round(2)
    parts.append(f"\n--- Household Avg Hourly (kWh) ---")
    parts.append(f"Top 5:\n{hh.head().to_string()}")
    parts.append(f"Bottom 5:\n{hh.tail().to_string()}")

    # Festival impact
    fest = df.groupby('festival')['total_kwh'].mean().round(2)
    parts.append(f"\n--- Festival Impact (avg kWh) ---\n{fest.to_string()}")

    return "\n".join(parts)


data_summary = build_summary(electricity_consumption_db)
```

The `build_summary` function in the above script computes seven categories of aggregations: overall statistics, appliance energy rankings with percentage shares, hourly consumption patterns with peak and off-peak hours for each appliance, seasonal averages, day-of-week trends, household comparisons, and festival impact analysis. These aggregations cover all 10,000 records and give Claude a comprehensive view of the data.

### Create Chatbot

With the data summary ready, we create the chatbot. The system prompt embeds the entire summary and instructs Claude to always cite specific numbers from the data. The `ElectricityChatbot` class maintains conversation history, allowing follow-up questions that build on previous answers.

In the script below, We used the Claude 4.6 sonnet model for reasoning. You can use any other anthropic model if you want.

```python
SYSTEM_PROMPT = f"""You are an electricity consumption analyst. You have data from {electricity_consumption_db['house_id'].nunique()} households
with {len(electricity_consumption_db)} hourly readings. Appliance columns: ac, fridge, lights, fans, washing_machine, tv.
All values are in kWh. The data also has season and festival columns.

Always cite specific numbers. Estimate kWh savings when giving recommendations.

DATA:
{data_summary}
"""


class ElectricityChatbot:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.history = []

    def ask(self, question):
        self.history.append({"role": "user", "content": question})
        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=self.history
        )
        answer = response.content[0].text
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def reset(self):
        self.history = []
        print("History cleared.")


chatbot = ElectricityChatbot()
print("Chatbot ready!")
```
**Output:**

```
Chatbot ready!
```

The chatbot is initialized and ready to answer questions about the electricity data.

### Ask Questions

Let's test the chatbot with questions about electricity consumption trends, savings recommendations, and targeted advice.

#### Question 1: Electricity Consumption Trends

```python
print(chatbot.ask("What are the main electricity consumption trends? Break down by appliance and season."))
```

**Output (partial screenshot):**

<img src="images\img4-electricity-trends-partial-output.png">

The chatbot provides a detailed breakdown of consumption patterns. It identifies the air conditioner as the highest single consumer at 4,689 kWh (18.8% of total), followed closely by fans at 4,492 kWh (18.1%) and fridge at 4,491 kWh (18.0%). The analysis reveals striking seasonal patterns: AC consumption jumps to 1.058 kWh/hr in summer (a 3.5x increase over winter), while fans and washing machines surge dramatically during the rainy season. The fridge and TV remain remarkably stable across all seasons, making them low-priority targets for optimization.

#### Question 2: Savings Recommendations

```python
print(chatbot.ask("Give me specific recommendations to save electricity with estimated kWh savings for each."))
```

**Output (partial screenshot):**

<img src="images\img5-savings-recommendations-partial-output.png">

Claude provides prioritized recommendations with estimated kWh savings based on the actual data. The top recommendations include: raising the AC thermostat by 2°C in summer (saving approximately 278 kWh/year), setting AC auto-shutoff during winter nights (saving approximately 383 kWh/year), replacing non-LED lights (saving approximately 670 kWh/year), and addressing the unusual 5 AM lighting peak which suggests lights being left on overnight (saving approximately 420 kWh/year). Each recommendation includes the specific data points that support it.

#### Question 3: Best Single Change

```python
print(chatbot.ask("Which single change gives the biggest savings with least effort?"))
```

**Output (partial screenshot):**

<img src="images\img6-single-saving-advice-partial-output.png">

Because the chatbot maintains conversation history, this follow-up question builds on the previous analysis. Claude identifies the overnight lighting issue as the single best change: the data shows lights peaking at 5 AM (0.384 kWh/hr) when they should be near zero, suggesting lights are left on overnight.

### Execute Code and Generate Charts

In addition to natural language analysis, we can ask Claude to generate matplotlib code for visualizations. The `generate_and_run_chart` function sends a chart description to Claude, receives executable Python code, and runs it inline to display the chart.

```python
def generate_and_run_chart(question):
    """
    Ask Claude to generate matplotlib code for a chart, then execute it.
    The chart displays inline in the notebook.
    """
    schema = f"Columns: {list(electricity_consumption_db.columns)}\nAppliance cols: {APPLIANCE_COLS}\nSample:\n{electricity_consumption_db.head(3).to_csv(index=False)}"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""I have a pandas DataFrame `electricity_consumption_db` with this structure:

{schema}

The electricity_consumption_db already has 'Hour', 'DayOfWeek', 'Month', 'total_kwh' columns.
Appliance values (ac, fridge, lights, fans, washing_machine, tv) are in kWh.

Write Python code using pandas and matplotlib to: {question}

RULES:
- Return ONLY Python code, no markdown fences, no explanation
- df is already loaded, matplotlib.pyplot is imported as plt
- Use plt.figure(figsize=(10, 6)) for good sizing
- Always include plt.tight_layout() and plt.show()
- Add clear title, axis labels, and legend where needed"""
        }]
    )

    code = response.content[0].text
    # Strip markdown fences if Claude includes them
    code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
    code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
    code = code.strip()

    print("Generated code:")
    print("-" * 40)
    print(code)
    print("-" * 40)
    print("\nExecuting...\n")

    exec(code, globals())
```

The `generate_and_run_chart` function sends the dataframe schema and a natural language description to Claude, which returns ready-to-execute matplotlib code. The `exec(code, globals())` call runs the code in the global scope so that all variables (including electricity_consumption_db, plt, and pd) are accessible.
Let's plot some charts showing electricity consumption trends.

#### Chart 1: Hourly Consumption by Appliance

```python
generate_and_run_chart("Line chart showing average hourly consumption for each appliance, all on one plot with different colors and a legend.")
```

**Output:**

<img src="images\img7-average-hourly-consumption-by-appliance.png">


#### Chart 2: Total Energy by Appliance

```python
generate_and_run_chart("Bar chart showing total energy consumption per appliance, sorted highest to lowest, with kWh values on each bar.")
```

**Output:**

<img src="images\img8-total-energy-consumptio-per-appliance.png">

The bar chart confirms AC as the highest total consumer at 4,689 kWh, followed by fans and fridge, with washing machine consuming the least at 3,487 kWh.

#### Chart 3: Seasonal Patterns

```python
generate_and_run_chart("Bar chart showing daily average consumption of each device for all the seasons")
```

**Output:**

<img src="images\img9-avareage-daily-consumption-by-season.png">

The seasonal chart highlights the dramatic AC spike in summer and the fan and washing machine surges during the rainy season, while fridge and TV remain stable across all seasons.

## Conclusion
This article demonstrates how to use the Anthropic Claude API and GridDB Cloud to build an AI-powered electricity consumption chatbot, which delivers data-backed analysis of consumption trends, identifies peak and off-peak hours for each appliance, and provides actionable savings recommendations with estimated kWh reductions. The code generation capability extends the chatbot further by producing inline matplotlib visualizations on demand, making it easy to explore the data visually without writing chart code manually.

You can extend this approach by adding more households for comparative analysis, integrating real-time data feeds from smart meters, or combining the chatbot with GridDB's time-series querying capabilities for more granular analysis. The combination of a high-performance database and AI reasoning opens up possibilities for intelligent energy monitoring at scale.

You can find the complete code for this blog on the [GridDB Blogs GitHub repository](https://github.com/usmanmalik57/GridDB-Blogs/tree/main/Electricity%20Consumption%20Monitoring%20Chatbot%20with%20GridDB%20and%20Anthropic%20Claude). If you have any questions or queries related to GridDB, create a post on Stack Overflow with the `griddb` tag. Our engineers are more than happy to respond.
