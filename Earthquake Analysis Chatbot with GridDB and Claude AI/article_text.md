Earthquakes are one of the most destructive natural hazards, and seismic networks record tens of thousands of them every year. Each record tells you where an earthquake happened, how strong it was, and how deep it started. Going through tens of thousands of these records by hand to find where earthquakes cluster, or how the strongest ones differ from the rest, is slow and difficult. Pairing a high-performance database like [GridDB](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) with an AI model lets you store the earthquake data once and then query it in natural language to find meaningful patterns.

In this article, you will see how to build an AI-powered earthquake analysis chatbot using the [Anthropic](https://www.anthropic.com/) Claude API and GridDB Cloud. The chatbot reads a year of United States Geological Survey (USGS) earthquake records that include each event's time, location (latitude, longitude, and region), magnitude, depth, how the magnitude was measured, and whether the event was a natural earthquake or something else such as a mining explosion. It finds the biggest earthquake hotspots, breaks the activity down by magnitude and depth, and looks at how the rare strong earthquakes differ from the rest. You will also use Claude to create matplotlib visualizations, including a world map of the earthquake hotspots, so you can explore the patterns visually.


**Prerequisites**:
You will need the following to run scripts in this article:

* A GridDB Cloud account. GridDB Cloud is available on the [Microsoft Azure Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/2812187.griddb_cloud_payasyougo), which offers a free plan for light testing and a pay-as-you-go plan for larger workloads. This [guide to GridDB Cloud on the Azure Marketplace](https://www.griddb.net/en/blog/griddb-cloud-azure-marketplace/) walks through the sign-up, and the [quick start guide](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) shows how to create a database user and whitelist your IP once the instance is running.
* [Anthropic API Key](https://platform.claude.com/). You can obtain one from the Anthropic console.

Once your instance is running, there are a few ways to reach it from Python: the REST-based Web API, the native GridDB client, or a JDBC/SQL driver. This article uses the Web API, so there is nothing extra to install on your machine to talk to GridDB. The connection section below explains these options.

Note: You can find the complete code for this tutorial in my [GridDB Blogs GitHub repository](https://github.com/usmanmalik57/GridDB-Blogs/tree/main/Earthquake%20Analysis%20Chatbot%20with%20GridDB%20and%20Claude%20AI).


## Installing and Importing Required Libraries

The commands below install the libraries this tutorial adds on top of a standard data-science setup. `python-dotenv` and `anthropic` handle the configuration and the Claude API, and `cartopy` is used later to plot the earthquakes on a world map.

```
!pip install python-dotenv
!pip install anthropic
!pip install cartopy
```

We then import everything the notebook uses.

```python
import pandas as pd
import numpy as np
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

The dataset for this article comes from the [USGS earthquake database](https://earthquake.usgs.gov/earthquakes/search/), a public record of earthquakes maintained by the United States Geological Survey. Instead of downloading a static file, we query the USGS [FDSN event web service](https://earthquake.usgs.gov/fdsnws/event/1/), which returns earthquake records as CSV. We pull every event of magnitude 2.5 and above for the whole of 2024.

A single FDSN query returns at most 20,000 events, and one year of magnitude 2.5+ events is larger than that. So the following script requests the data one month at a time and joins the monthly results into a single DataFrame. It also drops any duplicate events and saves the result to a CSV, so you can run the rest of the notebook without querying the API again.

```python
# Collect one year (2024) of magnitude 2.5+ events from the USGS FDSN event API.
# We request one month at a time because a single query is capped at 20,000 events,
# then concatenate the monthly slices into one DataFrame.
BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"
YEAR = 2024
MIN_MAGNITUDE = 2.5

windows = [(f"{YEAR}-{m:02d}-01", f"{YEAR}-{m + 1:02d}-01") for m in range(1, 12)]
windows.append((f"{YEAR}-12-01", f"{YEAR + 1}-01-01"))

frames = []
for start, end in windows:
    params = {
        "format": "csv",
        "starttime": start,
        "endtime": end,
        "minmagnitude": MIN_MAGNITUDE,
        "orderby": "time",
    }
    response = requests.get(BASE, params=params, timeout=90)
    frames.append(pd.read_csv(pd.io.common.StringIO(response.text)))

dataset = pd.concat(frames, ignore_index=True).drop_duplicates(subset="id").reset_index(drop=True)
dataset.to_csv("usgs_earthquakes.csv", index=False)

print(dataset.shape)
```

**Output:**
```
(25153, 22)
```

The API returns `25,153` earthquakes described by 22 raw columns.

The raw feed has more columns than we need and stores the location as a free-text `place` string. The script below keeps the columns we want, parses the timestamp, and adds a few new columns: the `Year`, `Month`, and `Hour`; a clean `region` taken from the `place` text; and readable `MagBand` and `DepthBand` buckets built with `pd.cut`.

```python
# Keep the columns we will analyze, parse the timestamp, and engineer features.
keep = ["time", "latitude", "longitude", "depth", "mag", "magType",
        "gap", "dmin", "rms", "place", "type", "status"]
dataset = dataset[keep].copy()

dataset["time"] = pd.to_datetime(dataset["time"], errors="coerce", utc=True)
dataset = dataset.dropna(subset=["time", "mag", "depth"]).copy()

# Derived time features. Cast the integer parts to int64 so GridDB maps them to
# LONG. The pandas .dt accessors return int32, which our type map would store as STRING.
dataset["Year"]    = dataset["time"].dt.year.astype("int64")
dataset["Month"]   = dataset["time"].dt.month.astype("int64")
dataset["Hour"]    = dataset["time"].dt.hour.astype("int64")
dataset["Weekday"] = dataset["time"].dt.day_name()

# The USGS "place" field is free text like "50 km NNW of Atka, Alaska". Derive a
# cleaner region (the part after the last comma) for grouping earthquakes by area.
def get_region(place):
    p = str(place)
    if " of " in p:
        p = p.split(" of ")[-1]
    if "," in p:
        return p.split(",")[-1].strip()
    return p.strip()

dataset["region"] = dataset["place"].apply(get_region)

# Bucket magnitude and depth into readable bands.
dataset["MagBand"] = pd.cut(
    dataset["mag"], bins=[0, 4, 5, 6, 20],
    labels=["minor (<4)", "light (4-5)", "moderate (5-6)", "strong (6+)"], right=False).astype(str)
dataset["DepthBand"] = pd.cut(
    dataset["depth"], bins=[-100, 70, 300, 1000],
    labels=["shallow (<70km)", "intermediate (70-300km)", "deep (>300km)"]).astype(str)

# Store the timestamp as a clean string and fix the column order.
dataset = dataset.sort_values("time").reset_index(drop=True)
dataset["time"] = dataset["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

dataset = dataset[["time", "Year", "Month", "Hour", "Weekday", "region", "place", "type",
                   "magType", "status", "mag", "MagBand", "depth", "DepthBand",
                   "latitude", "longitude", "gap", "dmin", "rms"]]

print(dataset.shape)
dataset.head()
```

**Output:**

<img src="images\img1-dataset.png">

The cleaned dataset has `25,153` earthquake records in 19 columns. Each row is one event, with its location, magnitude, depth, and the new time and band columns, ready to store and analyze.

## Inserting Data in GridDB

Next, we load the earthquake records into GridDB Cloud.

### Creating a GridDB Connection

There are several ways to connect to a GridDB Cloud instance. The Azure Marketplace edition supports the native GridDB client libraries (available for Java, C, and Python) and a JDBC/SQL driver. These give you lower-level, faster access, but they need the GridDB client installed on your machine. In this article we use the GridDB **Web API**, which talks to your cloud instance over ordinary HTTP requests. It needs nothing installed beyond the `requests` library, so every script here runs anywhere Python does.

The script below tests the connection to your GridDB Cloud instance, using the credentials from your `.env` file (stored as `azure_username`, `azure_password`, and `azure_base_url`).

```python
username = os.environ.get("azure_username")
password = os.environ.get("azure_password")
base_url = os.environ.get("azure_base_url")


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

A `200` status code confirms the connection is working. If you do not see the above response, check this [quick start guide](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) for troubleshooting.


### Creating a GridDB Container for Earthquake Data

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
[{'name': 'SerialNo', 'type': 'LONG'}, {'name': 'time', 'type': 'STRING'}, {'name': 'Year', 'type': 'LONG'}, {'name': 'Month', 'type': 'LONG'}, {'name': 'Hour', 'type': 'LONG'}, {'name': 'Weekday', 'type': 'STRING'}, {'name': 'region', 'type': 'STRING'}, {'name': 'place', 'type': 'STRING'}, {'name': 'type', 'type': 'STRING'}, {'name': 'magType', 'type': 'STRING'}, {'name': 'status', 'type': 'STRING'}, {'name': 'mag', 'type': 'DOUBLE'}, {'name': 'MagBand', 'type': 'STRING'}, {'name': 'depth', 'type': 'DOUBLE'}, {'name': 'DepthBand', 'type': 'STRING'}, {'name': 'latitude', 'type': 'DOUBLE'}, {'name': 'longitude', 'type': 'DOUBLE'}, {'name': 'gap', 'type': 'DOUBLE'}, {'name': 'dmin', 'type': 'DOUBLE'}, {'name': 'rms', 'type': 'DOUBLE'}]
```

The derived `Year`, `Month`, and `Hour` columns map to GridDB `LONG` because we cast them to 64-bit integers earlier. The pandas `.dt` accessors return 32-bit integers, which the type map would otherwise store as `STRING`. Our column names already avoid the spaces, parentheses, and `%` characters that GridDB forbids, so we do not need to rename anything.

We then create the container through the GridDB REST API.

```python
container_name = "earthquake_events_db"

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

The `201` status code confirms the container has been created successfully.


### Inserting Earthquake Records into the GridDB Container

With the container in place, we can load the records. The `format_row` helper takes care of the type conversions, most importantly turning any leftover NaN values (such as a missing azimuthal `gap`) into `None` so GridDB accepts them. Since `25,153` rows is a lot to send in one call, we insert them in batches of 10,000 to keep each request a reasonable size.

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

# With ~25,000 rows we insert in batches so each request stays a reasonable size.
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
Inserted 10000 / 25153 rows
Inserted 20000 / 25153 rows
Inserted 25153 / 25153 rows
Done. Total inserted: 25153
```

All `25,153` earthquake records now live inside the GridDB container.

## Using AI to Analyze Earthquakes

With the data stored in GridDB, we can pull it back out and feed a pre-computed summary to Claude. The flow has three stages: fetch the records from GridDB, build a structured summary of the whole dataset, and pass that summary as context so Claude can answer questions about hotspots, magnitude, and depth.


### Retrieve Data from GridDB

The script below fetches every earthquake row from the GridDB container and loads them into a pandas DataFrame. Since there are `25,153` rows, we page through the container in batches of 10,000 rather than asking for everything in a single request.

```python
url = f"{base_url}/containers/{container_name}/rows"

# With ~25,000 rows we page through the container in batches of 10,000.
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

earthquake_db = pd.DataFrame(all_rows, columns=[col for col in dataset.columns])
print(f"Retrieved {earthquake_db.shape[0]} rows")
earthquake_db.head()
```

**Output:**

<img src="images\img2-griddb-retrieved-data.png">

All `25,153` earthquake records have been pulled back from GridDB and are ready for analysis.

### Build Dataset Summary

Sending `25,153` raw rows to Claude in a single prompt is wasteful and can hurt response quality. Instead, we compute a structured summary of the whole dataset and hand that to the model. This way Claude reasons over real statistics computed from every earthquake in the database, not just a sampled handful.

```python
NUMERIC_COLS = ["mag", "depth", "gap", "dmin", "rms"]
ORDER_M = ["minor (<4)", "light (4-5)", "moderate (5-6)", "strong (6+)"]
ORDER_D = ["shallow (<70km)", "intermediate (70-300km)", "deep (>300km)"]

def build_summary(df):
    parts = []
    total = len(df)
    parts.append(f"Records: {total} earthquake events")
    parts.append(f"Date range: {df['time'].min()} to {df['time'].max()}")
    parts.append(f"Distinct regions: {df['region'].nunique()}")
    parts.append(f"Magnitude range: {df['mag'].min():.1f} to {df['mag'].max():.1f}  |  "
                 f"Depth range: {df['depth'].min():.1f} to {df['depth'].max():.1f} km")

    mb = df['MagBand'].value_counts().reindex(ORDER_M)
    mb_tbl = pd.DataFrame({'events': mb, 'pct': (mb / total * 100).round(1)})
    parts.append(f"\n--- Magnitude Band Distribution (minor<4, light 4-5, moderate 5-6, strong 6+) ---\n{mb_tbl.to_string()}")

    mo = df.groupby('Month').agg(events=('mag', 'count'), avg_magnitude=('mag', 'mean')).round(2)
    parts.append(f"\n--- Events and Average Magnitude by Month ---\n{mo.to_string()}")

    rg = df['region'].value_counts().head(12)
    rg_tbl = pd.DataFrame({'events': rg, 'pct': (rg / total * 100).round(1)})
    parts.append(f"\n--- Top 12 Regions by Event Count (hotspots) ---\n{rg_tbl.to_string()}")

    db = df.groupby('DepthBand').agg(events=('mag', 'count'), avg_magnitude=('mag', 'mean')).round(2).reindex(ORDER_D)
    parts.append(f"\n--- Depth Band Distribution (shallow<70, intermediate 70-300, deep>300 km) ---\n{db.to_string()}")

    hr = df.groupby('Hour').size()
    parts.append(f"\n--- Events by Hour of Day (UTC, 0-23) ---\n{hr.to_string()}")

    mt = df['magType'].value_counts().head(8)
    parts.append(f"\n--- Magnitude Measurement Types (magType) ---\n{mt.to_string()}")

    et = df['type'].value_counts()
    parts.append(f"\n--- Event Type Distribution ---\n{et.to_string()}")

    parts.append(f"\n--- Numeric Stats (overall) ---\n{df[NUMERIC_COLS].describe().round(2).to_string()}")

    bd = df.groupby('MagBand').agg(avg_depth=('depth', 'mean'), avg_gap=('gap', 'mean'), count=('mag', 'count')).round(2).reindex(ORDER_M)
    parts.append(f"\n--- Average Depth & Azimuthal Gap by Magnitude Band ---\n{bd.to_string()}")

    st = df['status'].value_counts()
    parts.append(f"\n--- Review Status ---\n{st.to_string()}")

    return "\n".join(parts)


data_summary = build_summary(earthquake_db)
print(data_summary[:600])
```

**Output:**

```
Records: 25153 earthquake events
Date range: 2024-01-01 00:03:15 to 2024-12-31 23:28:55
Distinct regions: 260
Magnitude range: 2.5 to 7.5  |  Depth range: -3.5 to 671.0 km

--- Magnitude Band Distribution (minor<4, light 4-5, moderate 5-6, strong 6+) ---
                events   pct
MagBand
minor (<4)       10978  43.6
light (4-5)      12668  50.4
moderate (5-6)    1408   5.6
strong (6+)         99   0.4

--- Events and Average Magnitude by Month ---
       events  avg_magnitude
Month
1        2249           3.89
2        2161           3.75
3
```

The `build_summary` function splits the data into eleven blocks. Each block covers a different view of the data, and together they give Claude a short but complete statistical picture of all `25,153` earthquakes.

* **Overview counts.** The total number of events, the date range they span, how many different regions appear, and the overall magnitude and depth ranges.
* **Magnitude band distribution.** A table showing how many earthquakes fall into each band, from minor (below 4) to strong (6 and above), with the percentage share of each.
* **Events and average magnitude by month.** The number of events recorded each month of 2024 and the average magnitude for that month, so Claude can reason about activity over the year.
* **Top regions.** The twelve regions with the most earthquakes, with their counts and percentage shares. This block is what the chatbot uses to point out seismic hotspots.
* **Depth band distribution.** Event counts and average magnitude for shallow, intermediate, and deep earthquakes.
* **Events by hour of day.** A count of events for every hour from 0 to 23 (UTC), which shows whether the recordings are spread evenly across the day.
* **Magnitude measurement types.** How many events used each `magType` (such as `mb`, `ml`, or `mww`), which reflects the instruments and methods behind the readings.
* **Event type distribution.** The split between natural earthquakes and other recorded events such as mining explosions or ice quakes.
* **Numeric stats overall.** The pandas `describe()` output for magnitude, depth, azimuthal gap, nearest-station distance, and residual.
* **Average depth and azimuthal gap by magnitude band.** The mean depth and station coverage for each magnitude band, so the chatbot can describe how the strongest events differ.
* **Review status.** How many solutions were human-reviewed versus automatic.

These eleven blocks stay well within Claude's context window even though they are computed over every earthquake in the database, which is the whole point of summarizing the data before sending it to the model.


### Create Chatbot

With the summary in place, we set up the chatbot. The system prompt includes the entire summary and tells Claude to back every observation with concrete numbers. The `EarthquakeChatbot` class keeps a running conversation history so follow-up questions build on what was already discussed.

In the script below we use the Claude Sonnet 4.6 model. You can swap in any other Anthropic model if you prefer.

```python
SYSTEM_PROMPT = f"""You are a seismology data analyst. You have records for {len(earthquake_db)} earthquakes
recorded worldwide by the USGS during 2024 (magnitude 2.5 and above). Each record includes the event
time (time, and its derived Year, Month, Hour, Weekday), the location (latitude, longitude, the free-text
place, and a derived region), the event type (type, e.g. earthquake or mining explosion), the magnitude
(mag) and its band (MagBand), how the magnitude was measured (magType), the depth in kilometres (depth)
and its band (DepthBand), the azimuthal gap (gap), the distance to the nearest station (dmin), the
residual (rms), and whether the solution was human-reviewed or automatic (status).

Always ground your answers in the specific numbers from the summary below. Present findings as
data-backed observations about this one-year sample, and remember that these are patterns in a set of
recorded events, not predictions of future earthquakes.

DATA:
{data_summary}
"""


class EarthquakeChatbot:
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


chatbot = EarthquakeChatbot()
print("Chatbot ready!")
```

**Output:**

```
Chatbot ready!
```

The chatbot is initialized and ready to answer questions about the earthquake records.


### Ask Questions

Let's test the chatbot with three questions covering seismic hotspots, the spread of magnitude and depth, and a follow-up on the strongest earthquakes that builds on the previous answer.

#### Question 1: Earthquake Hotspots

```python
print(chatbot.ask("Which regions are the biggest earthquake hotspots in this dataset, and how concentrated is seismic activity geographically? Cite specific numbers."))
```

**Output (partial screenshot):**

<img src="images\img3-hotspots-partial-output.png">

The chatbot shows how concentrated the activity is. Alaska alone accounts for `5,534` events, or `22.0%` of all the earthquakes. That is about one in every five recorded earthquakes in a single region, which reflects its position on the Aleutian subduction zone. It notes that the top five regions (Alaska, Indonesia, California, Puerto Rico, and Hawaii) together make up about `40.9%` of the records, and the top twelve regions account for roughly `58%`, even though the data spans `260` different regions. It also points out that nearly all the top hotspots, including Indonesia, Japan, the Philippines, Tonga, Papua New Guinea, Chile, and Vanuatu, sit along the Pacific Ring of Fire.

#### Question 2: Magnitude and Depth Distribution

```python
print(chatbot.ask("How are the earthquakes distributed across magnitude bands and depth bands? What share are shallow versus deep, and minor versus strong? Cite specific numbers."))
```

**Output (partial screenshot):**

<img src="images\img4-magnitude-depth-partial-output.png">

Claude lays out the two distributions. For magnitude, minor and light earthquakes together account for `23,646` events, which is `94.0%` of the data. Strong earthquakes (magnitude 6 and above) are rare at just `99` events, or `0.4%`. This steep drop from one band to the next matches the Gutenberg-Richter relationship. For depth, shallow earthquakes (under 70 km) dominate at `19,189` events (`76.3%`), intermediate events make up `18.8%`, and deep events (over 300 km) are the rarest at `4.9%`. Claude also points out that average magnitude rises with depth, from `3.73` for shallow events to `4.35` for deep ones. It explains that this is partly a detection effect, since deep earthquakes must be larger to register at distant stations.

#### Question 3: The Strongest Earthquakes

```python
print(chatbot.ask("Building on that, focus on the strong magnitude 6+ earthquakes you just mentioned. How does their average depth compare to the overall dataset, and which regions produced the most of them?"))
```

**Output (partial screenshot):**

<img src="images\img5-strongest-earthquakes-partial-output.png">

Because the chatbot keeps its conversation history, this follow-up builds directly on the previous answer and starts by recalling the `99` strong events. It reports that these strong earthquakes are `89.80` km deep on average. That is about `24.72` km deeper than the overall mean of `65.08` km, and well above the median of `17.61` km, which puts the typical strong event near the shallow-to-intermediate boundary. It adds a useful quality note: strong events have an average azimuthal gap of only `39.07°`, against `169.11°` for minor events, so their depths are among the best measured in the data. It also says clearly that the summary has no direct breakdown of magnitude by region, so it uses the regional counts and the tectonic setting to infer that the western Pacific subduction zones (Indonesia, Japan, Tonga, Papua New Guinea, the Philippines, and Vanuatu) most likely produced most of the strong events. It adds that Alaska's large total is inflated by smaller swarms.


### Execute Code and Generate Charts

Beyond text answers, we can have Claude write the matplotlib code for our charts. The `generate_and_run_chart` helper sends a chart description to Claude, runs the Python it returns inline, and keeps a running history so each new request can build on the last.

```python

client = anthropic.Anthropic()
chart_history = []

def generate_and_run_chart(question):
    """Ask Claude to generate matplotlib code for a chart, then execute it.
    The helper keeps its own conversation history, so you can ask follow-up chart
    requests (like "now zoom into Japan") and Claude remembers the previous chart."""
    if not chart_history:
        schema = f"Columns: {list(earthquake_db.columns)}\nSample:\n{earthquake_db.head(3).to_csv(index=False)}"
        content = f"""I have a pandas DataFrame `earthquake_db` with this structure:

{schema}

mag is the earthquake magnitude and depth is in kilometres. latitude and longitude give the epicentre.
region is a place name such as 'Alaska' or 'Indonesia'. cartopy is installed and may be used for maps.

Write Python code using pandas, matplotlib, and (for maps) cartopy to: {question}

RULES:
- Return ONLY Python code, no markdown fences, no explanation
- earthquake_db is already loaded, matplotlib.pyplot is imported as plt
- Use a sensible figure size (around (10, 6) for a standard chart, larger for a detailed map)
- Always include plt.tight_layout() and plt.show()
- Add a clear title, axis labels, and a legend or colorbar where needed"""
    else:
        content = question

    chart_history.append({"role": "user", "content": content})
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=chart_history
    )
    raw = response.content[0].text
    chart_history.append({"role": "assistant", "content": raw})

    code = re.sub(r'^```python\s*', '', raw, flags=re.MULTILINE)
    code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
    code = code.strip()

    print("Generated code:")
    print("-" * 40)
    print(code)
    print("-" * 40)
    print("\nExecuting...\n")

    exec(code, globals())

```

We keep a module-level `client` and a `chart_history` list. On the first call the helper sends Claude the dataframe schema and the rules. On every call after that it just forwards the new question and adds both sides of the exchange to `chart_history`. This gives the chart helper the same conversation memory as the chatbot, so we can ask a follow-up like "now zoom into Japan" and Claude still knows what "the same map" refers to.

Claude replies with ready-to-run matplotlib code, which we strip of any markdown fences and run with `exec(code, globals())` so it can see `earthquake_db`, `plt`, `pd`, and `np`. Because the request is plain English, we are not limited to bar charts. The first two charts are full maps drawn with cartopy.

Let's plot a few charts that highlight the patterns we discussed above, starting with a map of where the earthquakes actually happened.

#### Chart 1: Global Earthquake Hotspot Map

```python
generate_and_run_chart("A clean, light Google-Maps-style world map of all earthquake epicentres. Use the PlateCarree projection with a full global extent so it is a standard rectangular world map. Fill the ocean light blue and land light beige, draw thin gray coastlines and country borders, and plot the epicentres colored by magnitude (mag) across its full range with the YlOrRd colormap and a vertical colorbar labeled Magnitude on the right. Label the major countries with their names clearly and legibly, using small bold dark-gray text on a subtle white background box (like the place labels on a map). Add gridlines with latitude/longitude labels and a bold title.")
```

**Output:**

<img src="images\img6-global-hotspot-map.png">

On the world map, almost every point falls along the Pacific Ring of Fire and the mid-ocean ridges. This is the same geographic concentration that the chatbot described in its first answer.

#### Chart 2: Earthquake Epicentres in Japan

Because the chart helper keeps its history, we can treat this as a follow-up and simply ask for the same map zoomed into Japan, one of the busiest regions in the data. Claude remembers the style of the global map, applies it to the smaller area, and adds the city labels we ask for.

```python
generate_and_run_chart("Now show the same style of map but zoomed into Japan (about longitude 127 to 148, latitude 28 to 46), plotting only the earthquakes there and labelling the major Japanese cities.")
```

**Output:**

<img src="images\img7-japan-epicentres.png">

Zoomed in, most of the earthquakes sit along the Pacific coast and cluster in the offshore trench east of the islands, where the Pacific plate slides beneath Japan. This is the same subduction setting that put Japan among the top regions in the global view.

#### Chart 3: Magnitude Distribution

```python
generate_and_run_chart("A histogram of the earthquake magnitude (mag) column with about 25 bins, to show how many earthquakes fall at each magnitude level.")
```

**Output:**

<img src="images\img8-magnitude-distribution.png">

The histogram shows how quickly the counts fall off toward larger events. Most of the earthquakes sit between magnitude 4 and 5, with only a thin tail reaching past magnitude 7.

#### Chart 4: Top Regions by Earthquake Count

```python
generate_and_run_chart("A horizontal bar chart of the top 12 regions by number of earthquakes, sorted so the region with the most earthquakes is at the top, with the count labeled on each bar.")
```

**Output:**

<img src="images\img9-top-regions.png">

The chart makes Alaska's lead obvious. Its `5,534` events are far ahead of every other region, more than the next three regions combined.

#### Chart 5: Average Magnitude by Depth Band

```python
generate_and_run_chart("A bar chart of the average earthquake magnitude for each DepthBand, ordered shallow (<70km), intermediate (70-300km), then deep (>300km), with the average magnitude labeled on each bar.")
```

**Output:**

<img src="images\img10-magnitude-by-depth-band.png">

The bars confirm the pattern the chatbot pointed out. Average magnitude goes up with depth, from `3.73` for shallow earthquakes to `4.35` for deep ones. As Claude explained, this is partly caused by detection bias.

## Conclusion

This article showed how to combine the Anthropic Claude API with GridDB Cloud to build an AI-powered earthquake analysis chatbot. The chatbot pulls a full year of USGS earthquake records out of GridDB, summarizes them, and lets you ask plain natural language questions about seismic hotspots, the spread of magnitude and depth, and how the rare strong earthquakes differ from the rest. The chart step goes further and turns natural language descriptions into matplotlib visualizations, including a world map of the earthquake hotspots, so you can explore the data visually without writing plotting code by hand.

The analysis showed how concentrated earthquakes are. Alaska alone produced `22.0%` of the `25,153` events, and the top twelve regions accounted for about `58%`. While `94.0%` of the earthquakes were minor or light, the `99` strong events were noticeably deeper than the rest of the data.

You can extend this approach by storing several years of data for long-term trend analysis, by pulling live earthquake feeds from the USGS on a schedule to keep the data current, or by plotting the hotspots on an interactive map instead of a static one. Pairing a fast database with a reasoning model makes it easy to turn a large, hard-to-read dataset into a tool you can question in plain English.

You can find the complete code for this blog on the [GridDB Blogs GitHub repository](https://github.com/usmanmalik57/GridDB-Blogs/tree/main/Earthquake%20Analysis%20Chatbot%20with%20GridDB%20and%20Claude%20AI). If you have any questions or queries related to GridDB, create a post on Stack Overflow with the `griddb` tag. Our engineers are more than happy to respond.
