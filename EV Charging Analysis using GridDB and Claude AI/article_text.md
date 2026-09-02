Public electric vehicle charging networks record every session a driver starts: which station they used, when they plugged in and unplugged, and how much energy they drew. These charging networks build up hundreds of thousands of these records over the years. Answering simple questions from this data, such as which stations are busiest or what time of day demand peaks, requires manually writing SQL by hand and running one query after another against the database. Pairing a database like [GridDB](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) with an AI model like [Claude](https://platform.claude.com/docs/en/models/overview) lets you skip that step. You ask questions about the dataset in natural language, the AI model writes and runs the SQL for you and generates consolidated responses according to your queries.

In this article, you will see how to build an AI agent that answers plain-English questions about electric vehicle charging data. You will store a public dataset of charging sessions in GridDB Cloud, then give the [Anthropic](https://www.anthropic.com/) Claude API a tool that executes SQL queries on the GridDB dataset over a JDBC connection. When you ask a question in plain English, Claude turns it into a SQL query, the tool runs it against GridDB, and Claude reads the rows back and answers. The data keeps a real `TIMESTAMP` for every session, so the queries can use time functions like `EXTRACT` and `TIMESTAMP_DIFF` to group by hour, measure how long sessions last, and filter by date range. You will also use Claude to turn SQL results into matplotlib charts.


**Prerequisites**:
You will need the following to run scripts in this article:

* A GridDB Cloud account. GridDB Cloud is available on the [Microsoft Azure Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/2812187.griddb_cloud_payasyougo), which offers a free plan for light testing and a pay-as-you-go plan for larger workloads. This [guide to GridDB Cloud on the Azure Marketplace](https://www.griddb.net/en/blog/griddb-cloud-azure-marketplace/) walks through the sign-up, and the [quick start guide](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) shows how to create a database user and whitelist your IP once the database instance is running.
* [Anthropic API Key](https://platform.claude.com/). You can obtain one from the Anthropic console.
* Java installed on your machine. The JDBC query path in this article talks to GridDB through its Java driver, so you need a Java runtime and the GridDB client jar files, which you download from the Cloud dashboard. The JDBC section below explains where to get them.

Once your database instance is running, there are a few ways to reach it from Python: the REST-based Web API, the native GridDB client, or a JDBC/SQL driver. This article uses two of them together. It loads the data with the Web API, which needs nothing installed, and then runs the SQL queries over JDBC. The connection sections below explain both.

Note: You can find the complete code for this tutorial in my [GridDB Blogs GitHub repository](https://github.com/usmanmalik57/GridDB-Blogs/tree/main/EV%20Charging%20Analysis%20using%20GridDB%20and%20Claude%20AI).


## Installing and Importing Required Libraries

The commands below install the libraries this tutorial adds on top of a standard data-science setup. `python-dotenv` and `anthropic` handle the configuration and the Claude API, and `JPype1` and `jaydebeapi` let Python talk to the GridDB JDBC driver, which is a Java library.

```
!pip install python-dotenv
!pip install anthropic
!pip install JPype1
!pip install jaydebeapi
```

We then import everything the notebook uses.

```python
import os
import re
import glob
import json
import base64
import urllib.parse

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jpype
import jaydebeapi
import anthropic

from dotenv import load_dotenv
load_dotenv()

%matplotlib inline
```


## Importing and Preparing the EV Charging Dataset

The dataset for this article is the [City of Palo Alto electric vehicle charging station usage](https://github.com/yvenn-amara/ev-load-open-data/tree/master/1.%20Input%20Data/4.%20City%20of%20Palo%20Alto) record, a public log of charging sessions at the city's public stations from 2011 to 2020. Each row is one session, with the station name, the start and end times, the energy delivered, the connector type, the fee, and the station location.

The full file holds more than 250,000 sessions. The script below downloads it, keeps the columns we want, renames them to names GridDB accepts (no spaces or symbols), parses the start and end times, and takes a seeded sample of 50,000 sessions that still spans the whole 2011 to 2020 period. It formats the two timestamps as ISO 8601 strings ending in `Z`, which is the form GridDB reads into a `TIMESTAMP` column, and adds a `session_id` that will be the row key.

```python
DATA_URL = (
    "https://raw.githubusercontent.com/yvenn-amara/ev-load-open-data/master/"
    "1.%20Input%20Data/4.%20City%20of%20Palo%20Alto/ChargePoint%20Data%20CY20Q4.csv"
)

raw = pd.read_csv(DATA_URL, low_memory=False)
print("Raw shape:", raw.shape)
```

**Output:**
```
Raw shape: (259415, 33)
```

The raw file has `259,415` sessions described by 33 columns. The script below trims it to the columns we need and prepares the sample.

```python
# Keep and rename the columns we will analyze to GridDB-safe names (no spaces or symbols).
keep = {
    "Station Name": "station_name",
    "Start Date": "_start",
    "End Date": "_end",
    "Energy (kWh)": "energy_kwh",
    "GHG Savings (kg)": "ghg_savings_kg",
    "Port Type": "port_type",
    "Plug Type": "plug_type",
    "Fee": "fee",
    "Ended By": "ended_by",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "User ID": "user_id",
}
df = raw[list(keep)].rename(columns=keep).copy()

# Parse the start and end timestamps and drop rows we cannot use.
df["_start"] = pd.to_datetime(df["_start"], format="%m/%d/%Y %H:%M", errors="coerce")
df["_end"] = pd.to_datetime(df["_end"], format="%m/%d/%Y %H:%M", errors="coerce")
df = df.dropna(subset=["_start", "_end"])
df = df[df["_end"] >= df["_start"]]

# Fill a few light gaps in the text columns.
df["port_type"] = df["port_type"].fillna("Unknown")
df["ended_by"] = df["ended_by"].fillna("Unknown")
df["user_id"] = df["user_id"].fillna("Unknown").astype(str)

# The full dataset has more than 250,000 sessions. We take a seeded sample of 50,000
# that keeps the full 2011 to 2020 span so the load and the queries stay quick.
dataset = df.sample(n=50000, random_state=42).sort_values("_start").reset_index(drop=True)

# GridDB stores a TIMESTAMP as an ISO 8601 string ending in Z, so we format both columns.
dataset["start_ts"] = dataset["_start"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
dataset["end_ts"] = dataset["_end"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

# A LONG session_id is the row key. It has to be the first column.
dataset.insert(0, "session_id", np.arange(len(dataset), dtype="int64"))

for col in ["energy_kwh", "ghg_savings_kg", "fee", "latitude", "longitude"]:
    dataset[col] = dataset[col].astype("float64").round(6)

dataset = dataset[[
    "session_id", "station_name", "start_ts", "end_ts", "energy_kwh",
    "ghg_savings_kg", "port_type", "plug_type", "fee", "ended_by",
    "latitude", "longitude", "user_id",
]]

dataset.to_csv("ev_charging_palo_alto.csv", index=False)
print("Prepared shape:", dataset.shape)
dataset.head()
```

**Output:**

<img src="images\img1-dataset-head.png">

The prepared dataset has `50,000` sessions in 13 columns. Each row carries the station, the `start_ts` and `end_ts` timestamps, the energy in kilowatt hours, the connector details, and the location, ready to store and query.

## Storing the Sessions in GridDB 

We load the sessions into GridDB Cloud with the Web API, which talks to your database instance over ordinary HTTP requests and needs nothing installed beyond the `requests` library. We query the data over JDBC in the next section.

### Creating a GridDB Connection

The Azure Marketplace edition of GridDB Cloud can be reached through the native client libraries, a JDBC/SQL driver, or the REST-based **Web API**. For loading the data we use the Web API, because it runs anywhere Python does without any extra setup. The script below tests the connection to your GridDB Cloud database instance, using the credentials from your `.env` file (stored as `azure_username`, `azure_password`, and `azure_base_url`).

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

### Creating a Container with a TIMESTAMP Column

A GridDB container needs a schema before it can store any rows, so the script below maps each pandas column type to its GridDB equivalent. The `start_ts` and `end_ts` columns hold ISO 8601 strings, so we map them to `TIMESTAMP` by hand. This is the point of the whole exercise: storing the session times as real timestamps is what lets the SQL queries later group by hour and measure how long sessions run.

```python
# Mapping pandas dtypes to GridDB types
type_mapping = {
    "int64":      "LONG",
    "float64":    "DOUBLE",
    "bool":       "BOOL",
    "datetime64": "TIMESTAMP",
    "object":     "STRING",
    "category":   "STRING",
}

# start_ts and end_ts hold ISO 8601 strings, so we map them to TIMESTAMP by hand.
timestamp_cols = {"start_ts", "end_ts"}

columns = []
for col, dtype in dataset.dtypes.items():
    if col in timestamp_cols:
        griddb_type = "TIMESTAMP"
    else:
        griddb_type = type_mapping.get(str(dtype), "STRING")
    columns.append({"name": col, "type": griddb_type})

print(columns)
```

**Output:**
```
[{'name': 'session_id', 'type': 'LONG'}, {'name': 'station_name', 'type': 'STRING'}, {'name': 'start_ts', 'type': 'TIMESTAMP'}, {'name': 'end_ts', 'type': 'TIMESTAMP'}, {'name': 'energy_kwh', 'type': 'DOUBLE'}, {'name': 'ghg_savings_kg', 'type': 'DOUBLE'}, {'name': 'port_type', 'type': 'STRING'}, {'name': 'plug_type', 'type': 'STRING'}, {'name': 'fee', 'type': 'DOUBLE'}, {'name': 'ended_by', 'type': 'STRING'}, {'name': 'latitude', 'type': 'DOUBLE'}, {'name': 'longitude', 'type': 'DOUBLE'}, {'name': 'user_id', 'type': 'STRING'}]
```

The `session_id` maps to `LONG` because we cast it to a 64-bit integer, and both timestamp columns map to `TIMESTAMP`. Our column names already avoid the spaces, parentheses, and `%` characters that GridDB forbids, so we do not need to rename anything further.

We then create the container through the GridDB REST API.

```python
container_name = "ev_charging_sessions"

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

### Loading the Sessions

With the container in place, we load the sessions. The `format_row` helper turns any leftover NaN values into `None` so GridDB accepts them. Since `50,000` rows is a lot to send in one call, we insert them in batches of 10,000 to keep each request a reasonable size.

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

# With 50,000 rows we insert in batches so each request stays a reasonable size.
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
Inserted 10000 / 50000 rows
Inserted 20000 / 50000 rows
Inserted 30000 / 50000 rows
Inserted 40000 / 50000 rows
Inserted 50000 / 50000 rows
Done. Total inserted: 50000
```

All `50,000` sessions now live inside the GridDB container.

## Querying GridDB with SQL over JDBC

The data is stored. Now we switch to JDBC so we can run SQL against it, which is what the AI agent will use.

### Why JDBC, and the Web API Alternative

GridDB Cloud gives you several ways to reach your database instance directly from code: the REST-based Web API, a JDBC/SQL driver, and the native client libraries for Java, C, and Python. Any one of them can do the whole job, both writing and reading the data. In this article we use two approaches: the Web API creates the table and loads the sessions, and JDBC runs the queries. We could load and query through the Web API alone, or do everything over JDBC, but using one interface for each step shows how they fit together.

If you would rather stay on REST, GridDB's Web API also has a SQL endpoint, so you can run the same `SELECT` queries over HTTP without installing the driver.

The JDBC path needs a bit of setup:

* **The GridDB client jar files.** Download them from the Cloud dashboard, under the Support page, from the "GridDB Cloud Library and Plugin download" link. The `JDBC` folder inside the download holds the driver jars.
* **A notification provider URL.** The cluster Overview page shows a "Notification Provider URL for external connection". You pass it in the JDBC URL so the driver can find the cluster.
* **The public connection route.** Adding `connectionRoute=PUBLIC` to the JDBC URL tells the driver to reach the cluster over its public route rather than an internal one.
* **An allowed IP.** Whitelist your machine's IP under Network Access, on the **GRIDDB ACCESS** tab. This is separate from the WEBAPI ACCESS tab you use for the Web API.

For more detail, see [Connecting to GridDB Cloud v3.2 from Your Local Dev Environment](https://www.griddb.net/en/blog/connecting-to-griddb-cloud-v3-2-from-your-local-dev-environment-no-vpn-no-vnet-peering), [Using Python to interface with GridDB via JDBC with JayDeBeApi](https://griddb.net/en/blog/using-python-to-interface-with-griddb-via-jdbc-with-jaydebeapi/), and the [SQL reference in the GridDB docs](https://docs.griddb.net/sqlreference/sql-commands-supported/).

### Setting Up the GridDB Client Libraries

The download gives you several jar files. We put the four core jars on the class path and leave out the two call-logging jars, which expect an extra logging library we do not need here. Place the jar files in a `jars` folder next to the notebook.

```python
# The GridDB JDBC driver is a set of jar files downloaded from the Cloud dashboard.
# We put the four core jars on the class path and leave out the two call-logging jars,
# which expect an extra logging library we do not need here.
jar_dir = "jars"
jars = [j for j in glob.glob(os.path.join(jar_dir, "*.jar")) if "call-logging" not in j]
print("\n".join(os.path.basename(j) for j in jars))
```

**Output:**
```
gridstore-jdbc-5.8.0.jar
gridstore-conf-5.8.0.jar
gridstore-5.8.0.jar
gridstore-advanced-5.8.0.jar
```

The `gridstore-advanced` jar is the one that handles the TLS connection for the public route, so it has to be on the class path along with the main `gridstore` and `gridstore-jdbc` jars.

### Opening the JDBC Connection

The script below builds the JDBC URL, starts the Java virtual machine that the driver runs on, and opens the connection. We read the cluster name, database, and notification provider URL from the `.env` file. The provider URL carries its own query string, so we URL encode it before dropping it into the JDBC URL. We also start the JVM with its time zone set to UTC, so the timestamps read back exactly as they were stored.

```python
cluster = os.environ.get("azure_cluster_name")
database = os.environ.get("azure_database")
provider = os.environ.get("azure_provider_url")

# We start the JVM once, pinned to UTC so timestamps read back exactly as they were stored.
if not jpype.isJVMStarted():
    jpype.startJVM(jpype.getDefaultJVMPath(), "-Duser.timezone=UTC", classpath=jars)

# The provider URL carries its own query string, so it has to be URL encoded.
encoded_provider = urllib.parse.quote(provider, safe="")
jdbc_url = (
    f"jdbc:gs:///{cluster}/{database}"
    f"?notificationProvider={encoded_provider}"
    f"&connectionRoute=PUBLIC&sslMode=PREFERRED"
)

conn = jaydebeapi.connect(
    "com.toshiba.mwcloud.gs.sql.Driver",
    jdbc_url,
    [username, password],
    jars,
)

cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM ev_charging_sessions")
print("Rows in the table:", cur.fetchall()[0][0])
cur.close()
```

**Output:**
```
Rows in the table: 50000
```

The count of `50,000` confirms two things: the JDBC connection works, and it is reading the same table we created and loaded with the Web API. Both interfaces point at one store.

### Running SQL Directly

Before handing the database to Claude, we run a few queries ourselves to check the SQL. The `sql_df` helper runs a query and returns the rows as a pandas DataFrame. GridDB returns text values as Java strings, so the helper casts them to Python strings.

```python
def sql_df(query):
    """Run a SQL query on GridDB over JDBC and return the rows as a pandas DataFrame."""
    cur = conn.cursor()
    cur.execute(query)
    cols = [str(d[0]) for d in cur.description]
    # GridDB returns text values as Java strings, so we cast them to Python strings.
    data = [[(None if v is None else v if isinstance(v, (int, float)) else str(v))
             for v in row] for row in cur.fetchall()]
    cur.close()
    return pd.DataFrame(data, columns=cols)

busiest = sql_df("""
    SELECT station_name,
           COUNT(*) AS sessions,
           ROUND(AVG(energy_kwh), 2) AS avg_kwh
    FROM ev_charging_sessions
    GROUP BY station_name
    ORDER BY sessions DESC
    LIMIT 5
""")
busiest
```

**Output:**

<img src="images\img2-busiest-stations-sql.png">

The busiest station, `PALO ALTO CA / HAMILTON #2`, handled `4,584` sessions, well ahead of the next four, which each sit between `2,677` and `2,791` sessions. The average energy per session stays close across them, from `7.35` to `9.85` kilowatt hours.

The next query measures how long sessions run. The `TIMESTAMP_DIFF` function returns the gap between the two timestamp columns, and here we ask for it in minutes and average it by connector type.

```python
# The session length in minutes comes from TIMESTAMP_DIFF on the two timestamp columns.
duration = sql_df("""
    SELECT plug_type,
           COUNT(*) AS sessions,
           ROUND(AVG(TIMESTAMP_DIFF(MINUTE, end_ts, start_ts)), 1) AS avg_minutes,
           ROUND(AVG(energy_kwh), 2) AS avg_kwh
    FROM ev_charging_sessions
    GROUP BY plug_type
    ORDER BY sessions DESC
""")
print(duration.to_string(index=False))
```

**Output:**
```
 plug_type  sessions  avg_minutes  avg_kwh
     J1772     49113        152.1     8.63
NEMA 5-20R       887        257.1     3.06
```

Almost all sessions, `49,113` of them, use the `J1772` connector, the standard Level 2 plug. They run `152.1` minutes on average and deliver `8.63` kilowatt hours. The `887` sessions on the slower `NEMA 5-20R` wall outlet run longer, `257.1` minutes, but deliver far less energy, `3.06` kilowatt hours, which is what you expect from a trickle charge.

The last query filters by date range. GridDB writes timestamp literals as `TIMESTAMP('...')` around an ISO 8601 string, so we count the sessions that started during 2019.

```python
# A time range query counts the sessions that started during 2019.
sessions_2019 = sql_df("""
    SELECT COUNT(*) AS sessions_2019
    FROM ev_charging_sessions
    WHERE start_ts BETWEEN TIMESTAMP('2019-01-01T00:00:00Z')
                       AND TIMESTAMP('2019-12-31T23:59:59Z')
""")
print(sessions_2019.to_string(index=False))
```

**Output:**
```
 sessions_2019
          9358
```

`9,358` of the sampled sessions started in 2019. These three queries cover the pieces the agent will lean on: grouping and counting, the time functions on the timestamp columns, and date-range filtering.

## Letting Claude Query GridDB with SQL (Tool Use)

Now we give the database to Claude. Instead of sending the model a summary of the data, we give it a tool that runs SQL and let it query the database itself. The model decides what SQL to write, the tool runs it over the JDBC connection, and the rows come back to the model as the answer to work from.

### Defining the run_sql Tool

Tool use in the Claude API means describing a tool as a JSON schema and running it yourself when the model asks. Our tool takes one input, a SQL query, and returns the rows as text.

```python
RUN_SQL_TOOL = {
    "name": "run_sql",
    "description": "Run a single read-only GridDB SQL SELECT statement against the "
                   "ev_charging_sessions table and return the resulting rows.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "One GridDB SQL SELECT statement."}
        },
        "required": ["query"],
    },
}

def run_sql(query):
    """Execute the SQL Claude asks for and return the rows as plain text."""
    cur = conn.cursor()
    cur.execute(query)
    cols = [str(d[0]) for d in cur.description]
    rows = cur.fetchall()
    cur.close()
    lines = [" | ".join(cols)]
    for r in rows[:50]:
        lines.append(" | ".join("" if v is None else str(v) for v in r))
    if len(rows) > 50:
        lines.append(f"... ({len(rows)} rows in total)")
    return "\n".join(lines)
```

### The Agent Loop

The system prompt tells Claude what the table looks like and gives it the few SQL rules specific to GridDB that it needs, such as writing `EXTRACT(HOUR, start_ts)` with a comma rather than the standard `EXTRACT(HOUR FROM start_ts)`. The `EVChargingAgent` class runs the loop: it sends the question, and while Claude keeps asking to run SQL, it runs each query and passes the rows back, until Claude has enough to answer. It keeps the conversation history so follow-up questions build on what came before.

In the script below we use the Claude Sonnet 4.6 model. You can swap in any other Anthropic model if you prefer.

```python
SYSTEM_PROMPT = """You are a data analyst for a city electric vehicle charging network.
You answer questions by writing GridDB SQL and calling the run_sql tool. Never guess numbers,
always query the database.

The data lives in one table, ev_charging_sessions, with these columns:
- session_id (LONG): unique id for each charging session
- station_name (STRING): the charging station, for example 'PALO ALTO CA / HAMILTON #1'
- start_ts (TIMESTAMP): when the session started
- end_ts (TIMESTAMP): when the session ended
- energy_kwh (DOUBLE): energy delivered in kilowatt hours
- ghg_savings_kg (DOUBLE): greenhouse gas savings in kilograms
- port_type (STRING): 'Level 1' or 'Level 2'
- plug_type (STRING): 'J1772' or 'NEMA 5-20R'
- fee (DOUBLE): the fee charged for the session
- ended_by (STRING): how the session ended
- latitude, longitude (DOUBLE): station location
- user_id (STRING): the driver id

GridDB SQL notes you must follow:
- To read a part of a timestamp use EXTRACT with a comma: EXTRACT(HOUR, start_ts),
  EXTRACT(MONTH, start_ts), EXTRACT(YEAR, start_ts). The forms HOUR(...) and
  EXTRACT(HOUR FROM ...) do not work.
- Session length in minutes is TIMESTAMP_DIFF(MINUTE, end_ts, start_ts).
- Write timestamp literals as TIMESTAMP('2019-01-01T00:00:00Z').
- Only SELECT statements are allowed.

When you have the numbers you need, give a short, clear answer in plain words and quote the
actual figures the query returned."""

class EVChargingAgent:
    """A small agent that lets Claude answer questions by running SQL on GridDB."""

    def __init__(self, client, model="claude-sonnet-4-6"):
        self.client = client
        self.model = model
        self.history = []

    def ask(self, question):
        self.history.append({"role": "user", "content": question})
        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=[RUN_SQL_TOOL],
                messages=self.history,
            )
            self.history.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return "".join(b.text for b in response.content if b.type == "text")

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    query = block.input["query"]
                    print("Generated SQL:", " ".join(query.split()))
                    try:
                        result = run_sql(query)
                    except Exception as e:
                        result = f"SQL error: {str(e).splitlines()[0]}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            self.history.append({"role": "user", "content": tool_results})

client = anthropic.Anthropic()
agent = EVChargingAgent(client)
print("Agent ready!")
```

**Output:**
```
Agent ready!
```

The agent is ready. Each `ask` call prints the SQL Claude wrote before it prints the answer, so you can see exactly what ran against the database.

### Ask Questions

Let's put three questions to the agent, covering the busiest stations, the busiest hours, and a follow-up on session length that builds on the previous answer.

#### Question 1: The Busiest Stations

```python
# Q1: Busiest stations
answer = agent.ask(
    "Which five charging stations handled the most sessions, and what was the "
    "average energy delivered per session at each?"
)
print(answer)
```

**Output (partial screenshot):**

<img src="images\img3-question1-partial-output.png">

The query Claude wrote, and ran through the `run_sql` tool, was:

```sql
SELECT station_name, COUNT(session_id) AS total_sessions, AVG(energy_kwh) AS avg_energy_kwh
FROM ev_charging_sessions
GROUP BY station_name
ORDER BY total_sessions DESC
LIMIT 5
```

Reading the rows back, Claude reports `PALO ALTO CA / HAMILTON #2` as the busiest station with `4,584` sessions, about `64%` more than any other, and points out that `WEBSTER #1` delivers the most energy per session at `9.85` kilowatt hours even though it ranks fourth by volume. These are the same numbers we saw when we ran the query by hand, now reached by the model on its own.

#### Question 2: The Busiest Hours

```python
# Q2: Busiest hours of the day
answer = agent.ask(
    "Across the whole dataset, which three hours of the day are the busiest for "
    "starting a charge?"
)
print(answer)
```

**Output (partial screenshot):**

<img src="images\img4-question2-partial-output.png">

Claude answered by pulling the hour out of each timestamp with `EXTRACT(HOUR, start_ts)` and grouping by it:

```sql
SELECT EXTRACT(HOUR, start_ts) AS hour_of_day, COUNT(session_id) AS total_sessions
FROM ev_charging_sessions
GROUP BY hour_of_day
ORDER BY total_sessions DESC
LIMIT 3
```

The three busiest hours are `11:00` with `4,128` sessions, `12:00` with `3,986`, and `13:00` with `3,798`. All three fall in the middle of the day, which fits drivers plugging in around a mid-morning arrival or a lunch break.

#### Question 3: Session Length in the Peak Hours

```python
# Q3: Follow-up that refers back to the busiest hours from Question 2
answer = agent.ask(
    "During those three busiest hours, what is the average session length in minutes, "
    "and how does that compare with sessions that start overnight between midnight and 5 am?"
)
print(answer)
```

**Output (partial screenshot):**

<img src="images\img5-question3-partial-output.png">

Because the agent keeps its history, this follow-up builds on the previous answer. Claude remembered that the three busiest hours were 11, 12, and 13, and wrote two queries with `TIMESTAMP_DIFF`, one for those hours and one for the overnight window:

```sql
SELECT AVG(TIMESTAMP_DIFF(MINUTE, end_ts, start_ts)) AS avg_session_minutes
FROM ev_charging_sessions
WHERE EXTRACT(HOUR, start_ts) IN (11, 12, 13);

SELECT AVG(TIMESTAMP_DIFF(MINUTE, end_ts, start_ts)) AS avg_session_minutes
FROM ev_charging_sessions
WHERE EXTRACT(HOUR, start_ts) IN (0, 1, 2, 3, 4)
```

Midday sessions run `139.7` minutes on average, while overnight sessions run `289.4` minutes, more than twice as long. As Claude notes, midday charges are tied to short work or errand stops, while overnight charges sit plugged in with no reason to unplug.

## Visualizing the Results

Beyond text answers, we can have Claude turn a SQL result into a chart. The `generate_and_run_chart` helper takes a small DataFrame that we already aggregated with a SQL query and a plain description of the chart we want. It asks Claude for the matplotlib code, strips any markdown fences, and runs it. Because the data comes straight from a `GROUP BY` query, each chart is drawn from what the database returned.

```python
def generate_and_run_chart(df, request, save_path=None):
    """Ask Claude to write matplotlib code for an already aggregated DataFrame,
    then run that code. The data comes from a SQL query on GridDB, so the chart is
    built straight from what the database returned."""
    prompt = f"""I have a small pandas DataFrame named `chart_df` that I already aggregated with a GridDB SQL query.

Columns and dtypes:
{df.dtypes.to_string()}

The complete data:
{df.to_csv(index=False)}

Write Python code using pandas and matplotlib to: {request}

RULES:
- Return ONLY Python code, no markdown fences, no explanation
- The DataFrame is available as `chart_df`; matplotlib.pyplot is imported as plt
- Use a figure size around (10, 6)
- Always include plt.tight_layout() and plt.show()
- Add a clear title and axis labels"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text
    code = re.sub(r'^```python\s*', '', raw, flags=re.MULTILINE)
    code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE).strip()
    # We save and show the figure ourselves, so drop any plt.show() the model added.
    code = re.sub(r'^\s*plt\.show\(\)\s*$', '', code, flags=re.MULTILINE)

    globals()["chart_df"] = df
    exec(code, globals())
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.show()
```

We run each aggregation query with `sql_df`, then pass the result to the helper. Because the request is plain English, we are not limited to one kind of chart. Let's plot five views of the data, starting with how the network grew over the years.

#### Chart 1: Charging Sessions per Year

```python
# Chart 1: sessions per year
yearly = sql_df("""
    SELECT EXTRACT(YEAR, start_ts) AS year, COUNT(*) AS sessions
    FROM ev_charging_sessions
    GROUP BY year
    ORDER BY year
""")
generate_and_run_chart(
    yearly,
    "a line chart of charging sessions per year from 2011 to 2020, with a marker on each point",
    save_path="images/img6-sessions-per-year.png",
)
```

**Output:**

<img src="images\img6-sessions-per-year.png">

The line climbs steeply from a few hundred sessions in 2011 to a peak in the busy years around 2016 to 2019, then drops in 2020. The early growth tracks the spread of electric vehicles in the city, and the 2020 fall lines up with the drop in commuting that year.

#### Chart 2: The Busiest Hours of the Day

```python
# Chart 2: sessions by hour of day
hourly = sql_df("""
    SELECT EXTRACT(HOUR, start_ts) AS hour, COUNT(*) AS sessions
    FROM ev_charging_sessions
    GROUP BY hour
    ORDER BY hour
""")
generate_and_run_chart(
    hourly,
    "a bar chart of charging sessions by hour of the day, from hour 0 to 23",
    save_path="images/img7-sessions-by-hour.png",
)
```

**Output:**

<img src="images\img7-sessions-by-hour.png">

The bars show the daytime pattern the agent described in Question 2. Sessions build through the morning, peak at `11:00` with `4,128` starts, and fall away through the evening. The small hours see very little activity.

#### Chart 3: The Busiest Stations

```python
# Chart 3: top stations by sessions
stations = sql_df("""
    SELECT station_name, COUNT(*) AS sessions
    FROM ev_charging_sessions
    GROUP BY station_name
    ORDER BY sessions DESC
    LIMIT 10
""")
generate_and_run_chart(
    stations,
    "a horizontal bar chart of sessions per station with the busiest at the top, "
    "using short labels that drop the 'PALO ALTO CA /' prefix",
    save_path="images/img8-top-stations.png",
)
```

**Output:**

<img src="images\img8-top-stations.png">

The chart makes Hamilton #2's lead clear. Its `4,584` sessions sit well above the rest of the top ten, which taper down from about `2,800` to `2,000` sessions.

#### Chart 4: How Long Sessions Last

```python
# Chart 4: session duration buckets
duration_buckets = sql_df("""
    SELECT
      CASE
        WHEN TIMESTAMP_DIFF(MINUTE, end_ts, start_ts) < 60  THEN '0-1h'
        WHEN TIMESTAMP_DIFF(MINUTE, end_ts, start_ts) < 120 THEN '1-2h'
        WHEN TIMESTAMP_DIFF(MINUTE, end_ts, start_ts) < 240 THEN '2-4h'
        WHEN TIMESTAMP_DIFF(MINUTE, end_ts, start_ts) < 480 THEN '4-8h'
        ELSE '8h+'
      END AS duration_bucket,
      COUNT(*) AS sessions
    FROM ev_charging_sessions
    GROUP BY duration_bucket
""")
generate_and_run_chart(
    duration_buckets,
    "a bar chart of the number of sessions in each duration bucket, "
    "ordered 0-1h, 1-2h, 2-4h, 4-8h, 8h+",
    save_path="images/img9-session-duration.png",
)
```

**Output:**

<img src="images\img9-session-duration.png">

Here the `CASE` expression sorts each session into a length bucket right in the query. Most sessions run between two and four hours, with the one-to-two-hour and under-one-hour groups next. Only a small share run past eight hours.

#### Chart 5: Energy Delivered per Year

```python
# Chart 5: total energy delivered per year
energy = sql_df("""
    SELECT EXTRACT(YEAR, start_ts) AS year, ROUND(SUM(energy_kwh), 0) AS total_kwh
    FROM ev_charging_sessions
    GROUP BY year
    ORDER BY year
""")
generate_and_run_chart(
    energy,
    "a bar chart of total energy delivered in kilowatt hours per year from 2011 to 2020",
    save_path="images/img10-energy-per-year.png",
)
```

**Output:**

<img src="images\img10-energy-per-year.png">

Total energy follows the same shape as the session count, rising through the decade to a peak near 2019 before the 2020 drop. Since this is a 50,000-session sample rather than the full record, the yearly totals show the trend rather than the network's true output.

#### Chart 6: Monthly Sessions in 2019 and 2020

The yearly charts show a clear drop in 2020. This last chart looks at it more closely by putting each month of 2019 and 2020 side by side, which is the kind of time-range question the timestamp columns make easy.

```python
# Chart 6: monthly sessions in 2019 vs 2020
monthly_2yr = sql_df("""
    SELECT EXTRACT(YEAR, start_ts) AS year,
           EXTRACT(MONTH, start_ts) AS month,
           COUNT(*) AS sessions
    FROM ev_charging_sessions
    WHERE start_ts >= TIMESTAMP('2019-01-01T00:00:00Z')
    GROUP BY year, month
    ORDER BY year, month
""")
generate_and_run_chart(
    monthly_2yr,
    "a line chart comparing charging sessions by month for 2019 and 2020, with one "
    "line per year, the month number 1 to 12 on the x axis, a legend, and a marker on each point",
    save_path="images/img11-monthly-2019-2020.png",
)
```

**Output:**

<img src="images\img11-monthly-2019-2020.png">

The two lines track each other in January and February 2020, when the network ran at its usual level and even sat a little above 2019, with `849` sessions that January against `768` a year earlier. From March the 2020 line falls away. April drops to `91` sessions, down from `709` in April 2019, and the line stays far below 2019 for the rest of the year. This timing matches the Bay Area shelter-in-place order of March 2020 and the move to remote work that followed, which cut the commuting and downtown trips these public stations depend on. The data shows when the drop happened, and the timing points to the pandemic, though the records themselves do not state the cause.

## Conclusion

This article showed how to combine the Anthropic Claude API with GridDB Cloud to build an AI agent that answers questions by writing SQL. We loaded a public electric vehicle charging dataset into GridDB with the Web API, storing each session's start and end times as real `TIMESTAMP` columns, then queried it over JDBC. Instead of reading a fixed summary, Claude was given a tool that runs SQL, so it wrote its own `GROUP BY`, `EXTRACT`, and `TIMESTAMP_DIFF` queries to answer questions about the busiest stations and hours and how long sessions last. The chart step turned the same SQL results into matplotlib visualizations.

You can extend this approach by loading the full dataset for exact totals rather than a sample, by giving the agent more tables to join across, or by letting it write the chart queries as well as the text answers. Pairing a database with a model that turns plain-English questions into SQL makes a large table of records something you can explore just by asking.

You can find the complete code for this blog on the [GridDB Blogs GitHub repository](https://github.com/usmanmalik57/GridDB-Blogs/tree/main/EV%20Charging%20Analysis%20using%20GridDB%20and%20Claude%20AI). If you have any questions or queries related to GridDB, create a post on Stack Overflow with the `griddb` tag. Our engineers are more than happy to respond.
