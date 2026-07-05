Traffic accidents are one of the leading causes of injury and death worldwide, and the records they leave behind hold valuable clues about where, when, and under what conditions crashes happen. But a nationwide accident dataset can run into millions of rows, and manually sifting through them to find where accidents cluster, which hours are the most dangerous, or how weather relates to severity is slow and impractical. Pairing a high-performance database like [GridDB](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) with an AI model lets you store millions of accident records once and then explore them with plain-language questions instead of hand-written queries.

In this article, you will see how to build an AI-powered traffic accident analysis chatbot using the [Anthropic](https://www.anthropic.com/) Claude API and GridDB Cloud. The chatbot reads US traffic accident records that include the accident severity, the timestamp, the location (state and city), the weather condition, temperature, humidity, visibility, wind speed, and whether the accident happened during the day or at night. It identifies accident hotspots, surfaces time-of-day and weekly patterns, examines how weather relates to severity, and tracks severity trends over time. You will also have Claude generate the matplotlib code for a few charts, so you can explore the patterns visually as well.


**Prerequisites**:
To run the code in this article, you will need the following:

* A [GridDB Cloud account](https://www.global.toshiba/ww/products-solutions/ai-iot/griddb/product/griddb-cloud.html). The [quick start guide](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) walks you through setting one up.
* An [Anthropic API key](https://platform.claude.com/), which you can create in the Anthropic console.

Note: The complete code for this tutorial is available in my [GridDB Blogs GitHub repository](https://github.com/usmanmalik57/GridDB-Blogs/tree/main/Traffic%20Accident%20Analysis%20Chatbot%20with%20GridDB%20and%20AI).


## Installing and Importing Required Libraries

The commands below install the two libraries this tutorial adds on top of a standard data-science setup.

```
!pip install python-dotenv
!pip install anthropic
```

We then import everything the notebook uses.

```python
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
import requests
import json
import re

import anthropic
from dotenv import load_dotenv
load_dotenv()
```



## Importing the Dataset

The dataset for this article is the [US Accidents dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents), a countrywide collection of roughly 7.7 million traffic accident records gathered across the United States between 2016 and 2023. Each row describes a single accident and includes the severity (an integer from 1 to 4), the start time, the location, and a rich set of weather attributes such as temperature, humidity, visibility, wind speed, and the general weather condition. You can grab it from [its page on Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).

The full dataset is about 3 GB, which is far more than we need for a tutorial. The following script streams the CSV in chunks and draws a reproducible 100,000-row random sample so we never load the whole file into memory. It keeps only the columns relevant to accident hotspots, timing, weather, and severity, renames a few weather columns to names GridDB will accept, and derives the year, month, weekday, and hour from the timestamp.

```python
# The full US Accidents dataset is ~7.7 million rows (about 3 GB), which is far more
# than we need for a tutorial. We stream the CSV in chunks and draw a reproducible
# 100,000-row random sample (random_state=42), keeping only the columns relevant to
# accident hotspots, timing, weather, and severity. If you would rather skip the 3 GB
# download, load the included sample instead: dataset = pd.read_csv("us_accidents_sample.csv")

SRC = "US_Accidents_March23.csv"
TOTAL_ROWS = 7_728_394
SAMPLE_N = 100_000
SEED = 42

USECOLS = ["Severity", "Start_Time", "State", "City", "Temperature(F)", "Humidity(%)",
           "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "Weather_Condition",
           "Sunrise_Sunset", "Distance(mi)"]

frac = (SAMPLE_N * 1.15) / TOTAL_ROWS
parts = []
for chunk in pd.read_csv(SRC, usecols=USECOLS, chunksize=500_000):
    parts.append(chunk.sample(frac=frac, random_state=SEED))
dataset = pd.concat(parts, ignore_index=True)

# GridDB does not allow parentheses, %, or spaces in column names, so rename the
# weather/road columns to safe equivalents before we build the container schema.
dataset = dataset.rename(columns={
    "Temperature(F)": "Temperature_F", "Humidity(%)": "Humidity_pct",
    "Visibility(mi)": "Visibility_mi", "Wind_Speed(mph)": "Wind_Speed_mph",
    "Precipitation(in)": "Precipitation_in", "Distance(mi)": "Distance_mi"})

# Keep rows with the essential dimensions, parse the timestamp, and derive time features.
dataset = dataset.dropna(subset=["Start_Time", "State", "Severity"]).copy()
dataset["Start_Time"] = pd.to_datetime(dataset["Start_Time"], errors="coerce")
dataset = dataset.dropna(subset=["Start_Time"]).copy()

# Cast the integer time features to int64 so GridDB maps them to LONG. The pandas
# .dt accessors return int32, which our type map would otherwise store as STRING.
dataset["Year"]    = dataset["Start_Time"].dt.year.astype("int64")
dataset["Month"]   = dataset["Start_Time"].dt.month.astype("int64")
dataset["Weekday"] = dataset["Start_Time"].dt.day_name()
dataset["Hour"]    = dataset["Start_Time"].dt.hour.astype("int64")

dataset = dataset.sample(n=SAMPLE_N, random_state=SEED).sort_values("Start_Time").reset_index(drop=True)
dataset["Start_Time"] = dataset["Start_Time"].dt.strftime("%Y-%m-%d %H:%M:%S")

dataset = dataset[["Severity", "Start_Time", "Year", "Month", "Weekday", "Hour", "State",
                   "City", "Weather_Condition", "Temperature_F", "Humidity_pct",
                   "Visibility_mi", "Wind_Speed_mph", "Precipitation_in",
                   "Sunrise_Sunset", "Distance_mi"]]

print(dataset.shape)
dataset.head()
```

**Output:**

<img src="images\img1-dataset.png">

The sampled dataset contains 100,000 accident records described by 16 columns, a manageable slice of the full 7.7 million that still preserves the nationwide patterns we want to explore.

## Inserting Data in GridDB

Next, we load the accident records into GridDB Cloud.

### Creating a GridDB Connection

Run the script below to test your connection to GridDB Cloud, authenticating with the credentials from your GridDB Cloud account.

```python
username = os.environ.get("username")
password = os.environ.get("password")
base_url = os.environ.get("base_url")


url = f"{base_url}/checkConnection"

credentials = f"{username}:{password}"
encoded_credentials = base64.b64encode(credentials.encode()).decode()

headers = {
    'Content-Type': 'application/json',
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

A `200` status code confirms the connection is working. If you see anything else, the [quick start guide](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) covers the common connection fixes.


### Creating a GridDB Container for Traffic Accident Data

A GridDB container needs a schema before it can store any rows, so the script below maps each pandas column type to its GridDB equivalent.

```python
dataset.insert(0, "SerialNo", dataset.index + 1)
dataset.columns.name = None

# Mapping pandas dtypes to GridDB types
type_mapping = {
    "int64":      "LONG",
    "float64":    "DOUBLE",
    "bool":       "BOOL",
    "datetime64": "TIMESTAMP",
    "object":     "STRING",
    "category":   "STRING",
}

# Generate the columns part of the payload dynamically
columns = []
for col, dtype in dataset.dtypes.items():
    griddb_type = type_mapping.get(str(dtype), "STRING")
    columns.append({
        "name": col,
        "type": griddb_type
    })

print(columns)
```

**Output:**
```
[{'name': 'SerialNo', 'type': 'LONG'}, {'name': 'Severity', 'type': 'LONG'}, {'name': 'Start_Time', 'type': 'STRING'}, {'name': 'Year', 'type': 'LONG'}, {'name': 'Month', 'type': 'LONG'}, {'name': 'Weekday', 'type': 'STRING'}, {'name': 'Hour', 'type': 'LONG'}, {'name': 'State', 'type': 'STRING'}, {'name': 'City', 'type': 'STRING'}, {'name': 'Weather_Condition', 'type': 'STRING'}, {'name': 'Temperature_F', 'type': 'DOUBLE'}, {'name': 'Humidity_pct', 'type': 'DOUBLE'}, {'name': 'Visibility_mi', 'type': 'DOUBLE'}, {'name': 'Wind_Speed_mph', 'type': 'DOUBLE'}, {'name': 'Precipitation_in', 'type': 'DOUBLE'}, {'name': 'Sunrise_Sunset', 'type': 'STRING'}, {'name': 'Distance_mi', 'type': 'DOUBLE'}]
```

Notice that we renamed columns such as `Temperature(F)` to `Temperature_F` back when we loaded the data, because GridDB does not allow parentheses, spaces, or `%` characters in column names. We also cast the derived `Year`, `Month`, and `Hour` columns to 64-bit integers so the type map turns them into GridDB `LONG` columns instead of falling through to `STRING`.

We then create the container through the GridDB REST API.

```python
container_name = "traffic_accidents_db"

url = f"{base_url}/containers"

payload = json.dumps({
    "container_name": container_name,
    "container_type": "COLLECTION",
    "rowkey": True,
    "columns": columns
})

response = requests.post(url, headers=headers, data=payload)

print(f"Status Code: {response.status_code}")
```

**Output:**
```
Status Code: 201
```

A `201` status code means the container was created successfully.


### Inserting Traffic Accident Records into the GridDB Container

With the container in place, we can load the records. The `format_row` helper handles the type conversions, most importantly turning any leftover NaN values into `None` so GridDB accepts them. Since 100,000 rows is a lot to send in one call, we insert them in batches of 10,000 to keep each request a reasonable size.

```python
url = f"{base_url}/containers/{container_name}/rows"

def format_row(row):
    formatted = []
    for item in row:
        if pd.isna(item):
            formatted.append(None)
        elif isinstance(item, bool):
            formatted.append(str(item).lower())
        elif isinstance(item, (int, float)):
            formatted.append(item)
        else:
            formatted.append(str(item))
    return formatted

rows = [format_row(row) for row in dataset.values.tolist()]

# With 100,000 rows we insert in batches so each request stays a reasonable size.
BATCH_SIZE = 10000
total_inserted = 0

for i in range(0, len(rows), BATCH_SIZE):
    batch = rows[i:i + BATCH_SIZE]
    response = requests.put(url, headers=headers, data=json.dumps(batch))
    if response.status_code != 200:
        print(f"Batch starting at {i} failed: {response.status_code} - {response.text}")
        break
    total_inserted += response.json().get("count", 0)
    print(f"Inserted {total_inserted} / {len(rows)} rows")

print(f"Done. Total inserted: {total_inserted}")
```

**Output:**
```
Inserted 10000 / 100000 rows
Inserted 20000 / 100000 rows
Inserted 30000 / 100000 rows
Inserted 40000 / 100000 rows
Inserted 50000 / 100000 rows
Inserted 60000 / 100000 rows
Inserted 70000 / 100000 rows
Inserted 80000 / 100000 rows
Inserted 90000 / 100000 rows
Inserted 100000 / 100000 rows
Done. Total inserted: 100000
```

All 100,000 accident records now live inside the GridDB container.

## Using AI to Analyze Traffic Accidents

With the data stored in GridDB, we can pull it back out and feed a pre-computed summary to Claude. The flow has three stages: fetch records from GridDB, build a structured summary of the dataset, and pass that summary as context so Claude can answer questions about hotspots, timing, weather, and severity.


### Retrieve Data from GridDB

The script below fetches every accident row from the GridDB container and loads them into a pandas DataFrame. Since there are 100,000 rows, we page through the container in batches of 10,000 rather than asking for everything in a single request.

```python
url = f"{base_url}/containers/{container_name}/rows"

# With 100,000 rows we page through the container in batches rather than one request.
BATCH_SIZE = 10000
all_rows = []
offset = 0

while True:
    payload = json.dumps({
        "offset": offset,
        "limit": BATCH_SIZE,
        "condition": "",
        "sort": ""
    })
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code != 200:
        print(f"Error at offset {offset}: {response.status_code} - {response.text}")
        break
    chunk = response.json().get("rows", [])
    if not chunk:
        break
    all_rows.extend(chunk)
    offset += len(chunk)
    if len(chunk) < BATCH_SIZE:
        break

accidents_db = pd.DataFrame(all_rows, columns=[col for col in dataset.columns])
print(f"Retrieved {accidents_db.shape[0]} rows")
accidents_db.head()
```

**Output:**

<img src="images\img2-griddb-retrieved-data.png">

All 100,000 accident records have been pulled back from GridDB and are ready for analysis.

### Build Dataset Summary

Sending 100,000 raw rows to Claude in a single prompt is wasteful and can hurt response quality. Instead, we compute a structured summary of the dataset and hand that to the model. This way Claude reasons over real statistics derived from every accident in the database, not just a sampled handful.

```python
NUMERIC_COLS = ['Temperature_F', 'Humidity_pct', 'Visibility_mi',
                'Wind_Speed_mph', 'Precipitation_in', 'Distance_mi']

def build_summary(df):
    parts = []
    total = len(df)
    parts.append(f"Records: {total} accidents")
    parts.append(f"Date range: {df['Year'].min()} to {df['Year'].max()}")
    parts.append(f"States covered: {df['State'].nunique()}  |  Cities covered: {df['City'].nunique()}")

    # Severity distribution (1 = least severe, 4 = most severe)
    sev = df['Severity'].value_counts().sort_index()
    sev_tbl = pd.DataFrame({'accidents': sev, 'pct': (sev / total * 100).round(1)})
    sev_tbl.index = [f'severity {i}' for i in sev_tbl.index]
    parts.append(f"\n--- Severity Distribution (1=least, 4=most severe) ---\n{sev_tbl.to_string()}")

    # Accidents and average severity by year
    yr = df.groupby('Year').agg(accidents=('Severity', 'count'), avg_severity=('Severity', 'mean')).round(2)
    parts.append(f"\n--- Accidents and Average Severity by Year ---\n{yr.to_string()}")

    # Top states (hotspots)
    st = df['State'].value_counts().head(10)
    st_tbl = pd.DataFrame({'accidents': st, 'pct': (st / total * 100).round(1)})
    parts.append(f"\n--- Top 10 States by Accident Count (hotspots) ---\n{st_tbl.to_string()}")

    # Top cities (hotspots)
    ct = df['City'].value_counts().head(10)
    parts.append(f"\n--- Top 10 Cities by Accident Count (hotspots) ---\n{ct.to_string()}")

    # Time of day
    hr = df.groupby('Hour').size()
    parts.append(f"\n--- Accidents by Hour of Day (0-23) ---\n{hr.to_string()}")

    # Day of week
    wd_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    wd = df['Weekday'].value_counts().reindex(wd_order)
    parts.append(f"\n--- Accidents by Day of Week ---\n{wd.to_string()}")

    # Day vs night
    ds = df.groupby('Sunrise_Sunset').agg(accidents=('Severity', 'count'), avg_severity=('Severity', 'mean')).round(2)
    parts.append(f"\n--- Day vs Night (accidents + avg severity) ---\n{ds.to_string()}")

    # Weather impact
    wc = df.groupby('Weather_Condition').agg(accidents=('Severity', 'count'), avg_severity=('Severity', 'mean'))
    wc = wc[wc['accidents'] >= 200].sort_values('accidents', ascending=False).head(12).round(2)
    parts.append(f"\n--- Top Weather Conditions (>=200 accidents): count + avg severity ---\n{wc.to_string()}")

    # Numeric conditions overall
    parts.append(f"\n--- Numeric Stats (overall) ---\n{df[NUMERIC_COLS].describe().round(2).to_string()}")

    # Average conditions by severity
    by_sev = df.groupby('Severity')[NUMERIC_COLS].mean().round(2)
    parts.append(f"\n--- Average Conditions by Severity Level ---\n{by_sev.to_string()}")

    return "\n".join(parts)


data_summary = build_summary(accidents_db)
print(data_summary[:600])
```

**Output:**

```
Records: 100000 accidents
Date range: 2016 to 2023
States covered: 49  |  Cities covered: 6428

--- Severity Distribution (1=least, 4=most severe) ---
            accidents   pct
severity 1        947   0.9
severity 2      77823  77.8
severity 3      18547  18.5
severity 4       2683   2.7

--- Accidents and Average Severity by Year ---
      accidents  avg_severity
Year
2016       5871          2.39
2017      10276          2.39
2018      12807          2.39
2019      13668          2.31
2020      16590          2.18
2021      20175          2.15
2022      18193
```


The `build_summary` function divides the accident data into eleven blocks. Each block covers a different angle on the dataset, and together they give Claude a compact but complete statistical picture of all 100,000 accidents.

* **Overview counts.** The total number of accidents, the year range they span, and how many distinct states and cities appear in the sample.
* **Severity distribution.** A table showing how many accidents fall into each severity level from 1 (least severe) to 4 (most severe), along with the percentage share of each level.
* **Accidents and average severity by year.** The number of accidents recorded each year from 2016 to 2023 and the average severity for that year, so Claude can reason about how volume and severity have shifted over time.
* **Top states.** The ten states with the most accidents, with their counts and percentage shares. This is the block that lets the chatbot pinpoint geographic hotspots.
* **Top cities.** The ten cities with the most accidents, giving a finer-grained view of where crashes cluster.
* **Accidents by hour of day.** A count of accidents for every hour from 0 to 23, which reveals the rush-hour peaks.
* **Accidents by day of week.** Accident counts for Monday through Sunday, exposing the split between weekdays and weekends.
* **Day versus night.** Accident counts and average severity for daytime and nighttime accidents.
* **Weather conditions.** For every weather condition with at least 200 accidents, the number of accidents and the average severity, so Claude can compare how conditions like fog, rain, or clear skies relate to severity.
* **Numeric stats overall.** The pandas `describe()` output for temperature, humidity, visibility, wind speed, precipitation, and the affected road distance.
* **Average conditions by severity.** The same numeric features averaged separately for each severity level, so the chatbot can describe how conditions differ between minor and serious accidents.

These eleven blocks together stay well within Claude's context window even though they are computed over every accident in the database, which is the whole point of pre-aggregating before sending the data to the model.


### Create Chatbot

With the summary in place, we wire up the chatbot. The system prompt embeds the entire summary and tells Claude to back every observation with concrete numbers. The `TrafficAccidentChatbot` class keeps a running conversation history so follow-up questions naturally build on what was already discussed.

In the script below we use the Claude Sonnet 4.6 model. You can swap in any other Anthropic model if you prefer.

```python
SYSTEM_PROMPT = f"""You are a traffic-safety data analyst. You have records for {len(accidents_db)} US
traffic accidents sampled from the US Accidents dataset. Each record includes the accident severity
(Severity, an integer from 1 = least severe to 4 = most severe), the timestamp (Start_Time) and its
derived Year, Month, Weekday, and Hour, the location (State, City), the Weather_Condition, numeric
conditions (Temperature_F, Humidity_pct, Visibility_mi, Wind_Speed_mph, Precipitation_in), whether the
accident happened during Day or Night (Sunrise_Sunset), and the length of road affected (Distance_mi).

Always ground your answers in the specific numbers from the summary below. Present findings as
data-backed observations rather than established facts, and remember that these are correlations in a
sample, not proof of causation.

DATA:
{data_summary}
"""


class TrafficAccidentChatbot:
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


chatbot = TrafficAccidentChatbot()
print("Chatbot ready!")
```

**Output:**

```
Chatbot ready!
```

The chatbot is initialized and ready to answer questions about the accident records.


### Ask Questions

Let's test the chatbot with three questions covering accident hotspots, time-of-day patterns, and a follow-up on weather and severity that builds on the previous answer.

#### Question 1: Accident Hotspots

```python
print(chatbot.ask("Which states and cities are the biggest accident hotspots in this dataset, and how concentrated are accidents geographically? Cite specific numbers."))
```

**Output (partial screenshot):**

<img src="images\img3-hotspots-partial-output.png">

The chatbot highlights just how concentrated the accidents are. California alone accounts for 22,325 accidents, or 22.3% of the entire sample, more than double second-place Florida (10,859). It notes that the top three states (California, Florida, and Texas) together make up about 41% of the records, and the top ten states account for roughly 67.3%, even though the data spans 49 states. At the city level it points to Houston (2,292), Miami (2,204), and Los Angeles (2,011) as the busiest, underlining that a handful of large metros drive a disproportionate share of the records.

#### Question 2: Time-of-Day and Weekly Patterns

```python
print(chatbot.ask("What are the busiest times for accidents? Break the pattern down by hour of day and by day of week, and point out the peak periods. Cite specific numbers."))
```

**Output (partial screenshot):**

<img src="images\img4-time-of-day-partial-output.png">

Claude lays out the daily rhythm of accidents. The two clear peaks are the morning commute at 7-8 AM (7,790 and 7,863 accidents) and the afternoon rush from 3-5 PM (peaking at 7,595 accidents in the 4 PM hour), while the small hours around 3 AM (1,020 accidents) are the quietest. It also flags a stark weekday-weekend split: Friday is the busiest day at 17,741 accidents, and Monday through Friday together account for about 84.9% of all accidents, whereas Saturday and Sunday fall to roughly 8,000 and 7,000 each. This pattern strongly suggests the records are dominated by commuter traffic.

#### Question 3: Weather and Severity

```python
print(chatbot.ask("Building on those peak times, how do weather conditions and day-versus-night relate to accident severity? Which conditions show the highest average severity, and what does the year-over-year severity trend look like?"))
```

**Output (partial screenshot):**

<img src="images\img5-weather-severity-partial-output.png">

Because the chatbot retains conversation history, this follow-up builds directly on the previous answer. Claude surfaces a "clear weather paradox": the highest average severities show up under `Scattered Clouds` (2.40), `Clear` (2.37), and `Overcast` (2.37), while `Fair` weather, which produces by far the most accidents (31,547), has the lowest average severity at 2.14, and even `Fog` (2.15) sits near the bottom. Day and night accidents have essentially identical average severity (2.23 each). The clearest signals are structural rather than weather-related: the most serious accidents (severity 4) disrupt a much longer stretch of road (1.48 miles on average versus 0.4-0.5 for less severe crashes), and average severity has eased steadily from 2.39 in 2016 to 2.06 in 2023 even as the raw accident count climbed.


### Execute Code and Generate Charts

Beyond text answers, we can ask Claude to write matplotlib code for visualizations. The `generate_and_run_chart` function sends a chart description to Claude, takes the returned Python code, and runs it inline so the chart appears right under the request.

```python

client = anthropic.Anthropic()

def generate_and_run_chart(question):
    """
    Ask Claude to generate matplotlib code for a chart, then execute it.
    The chart displays inline in the notebook.
    """
    schema = f"Columns: {list(accidents_db.columns)}\nSample:\n{accidents_db.head(3).to_csv(index=False)}"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""I have a pandas DataFrame `accidents_db` with this structure:

{schema}

Severity is an integer from 1 (least severe) to 4 (most severe). Hour is 0-23. Weekday is the day
name. Weather_Condition is a string such as 'Fair', 'Cloudy', or 'Light Rain'.

Write Python code using pandas and matplotlib to: {question}

RULES:
- Return ONLY Python code, no markdown fences, no explanation
- accidents_db is already loaded, matplotlib.pyplot is imported as plt
- Use plt.figure(figsize=(10, 6)) for good sizing
- Always include plt.tight_layout() and plt.show()
- Add a clear title, axis labels, and a legend where needed"""
        }]
    )

    code = response.content[0].text
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

We create a module-level `client` at the top of the cell because `generate_and_run_chart` references it directly inside its body. The `TrafficAccidentChatbot` class earlier built its own client as an instance attribute (`self.client`), which is not visible from outside the class, so the chart helper needs its own handle.

The function passes the dataframe schema and a plain English chart description to Claude, which sends back ready-to-run matplotlib code. The `exec(code, globals())` call runs the code in the global scope so it can see `accidents_db`, `plt`, and `pd`.

Let's plot a few charts that highlight the patterns we discussed above.

#### Chart 1: Accident Hotspots by State

```python
generate_and_run_chart("A bar chart of the top 10 states by number of accidents, sorted descending. Put the count on top of each bar.")
```

**Output:**

<img src="images\img6-top-states-by-accidents.png">

The chart makes the geographic concentration obvious: California towers over every other state at 22,325 accidents, more than double second-place Florida.

#### Chart 2: Accidents by Hour of Day

```python
generate_and_run_chart("A bar chart showing the number of accidents for each hour of the day from 0 to 23, to reveal the rush-hour peaks.")
```

**Output:**

<img src="images\img7-accidents-by-hour.png">

The twin rush-hour peaks stand out clearly, with the tallest bars at 7-8 AM and again through the late afternoon, and a deep trough in the pre-dawn hours.

#### Chart 3: Average Severity by Weather Condition

```python
generate_and_run_chart("A horizontal bar chart of the average accident severity for the weather conditions that have at least 500 accidents, sorted so the highest average severity is at the top.")
```

**Output:**

<img src="images\img8-avg-severity-by-weather.png">

The chart confirms the counterintuitive finding from the chatbot: clear and lightly clouded conditions sit at the top with the highest average severity, while `Fair` and `Fog` conditions land at the bottom.

## Conclusion

This article showed how to combine the Anthropic Claude API with GridDB Cloud to build an AI-powered traffic accident analysis chatbot. The chatbot pulls a 100,000-row sample of US accident records out of GridDB, summarizes them on the fly, and lets you ask plain natural language questions about accident hotspots, time-of-day and weekly patterns, and how weather relates to severity. The chart generation step extends it further by producing matplotlib visualizations from natural language descriptions, so you can dig into the data visually without writing plotting code by hand.

You can extend this approach by storing the full 7.7 million records for a nationwide analysis, by adding the accident latitude and longitude to plot hotspots on a real map, or by hooking the chatbot up to a live accident feed for ongoing monitoring. The pairing of a fast structured database and a reasoning model makes it easy to turn a large, unwieldy dataset into an interactive analysis tool.

You can find the complete code for this blog on the [GridDB Blogs GitHub repository](https://github.com/usmanmalik57/GridDB-Blogs/tree/main/Traffic%20Accident%20Analysis%20Chatbot%20with%20GridDB%20and%20AI). If you have any questions or queries related to GridDB, create a post on Stack Overflow with the `griddb` tag. Our engineers are more than happy to respond.
