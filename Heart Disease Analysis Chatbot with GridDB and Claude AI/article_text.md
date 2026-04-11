Heart disease is one of the leading causes of death worldwide. Uncovering risk patterns from patient records with heart disease can help clinicians make better-informed decisions. Going through hundreds of patient records by hand to spot which combinations of age, cholesterol levels, and chest pain types tend to appear together is slow and error prone. Pairing a high-performance database like [GridDB](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) with an AI model lets you store medical records once and then query them in natural language to identify meaningful patterns.

In this article, you will see how to build an AI-powered heart disease analysis chatbot using the [Anthropic](https://www.anthropic.com/) Claude API and GridDB Cloud. The chatbot reads patient medical records that include age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, maximum heart rate achieved, and other clinical attributes. It identifies which risk factors appear most often among heart disease patients, compares risk profiles across age groups, and produces AI-generated insights and recommendations. You will also use Claude to create matplotlib visualizations so you can explore the data visually.


**Prerequisites**:
You will need the following to run scripts in this article:

* A [GridDB cloud account](https://www.global.toshiba/ww/products-solutions/ai-iot/griddb/product/griddb-cloud.html). Refer to this [quick start guide](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) to set up a GridDB cloud account.
* [Anthropic API Key](https://platform.claude.com/). You can obtain one from the Anthropic console.

Note: You can find the complete code for this tutorial in my [GridDB Blogs GitHub repository](https://github.com/usmanmalik57/GridDB-Blogs/tree/main/Heart%20Disease%20Analysis%20Chatbot%20with%20GridDB%20and%20Claude%20AI).


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

import anthropic
from dotenv import load_dotenv
load_dotenv()
```



## Importing the Dataset

The dataset used in this article is the UCI Heart Disease dataset containing patient medical records. Each row represents a single patient and includes clinical measurements such as age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved during exercise, exercise-induced angina, and a target column called `num`. The `num` column ranges from 0 to 4, where 0 means no heart disease and 1, 2, 3, and 4 represent progressive stages of heart disease. You can [download the dataset from Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).

The following script loads the dataset, keeps the columns we will work with, and removes rows with missing values in those columns so the chatbot has clean numbers to reason over. We also convert the `num` target into a binary `heart_disease` flag that is 1 whenever any stage of heart disease is present.

```python
dataset = pd.read_csv("heart_disease_uci.csv", encoding='utf-8')

keep_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
             'restecg', 'thalch', 'exang', 'oldpeak', 'num']
dataset = dataset[keep_cols].dropna().reset_index(drop=True)

# Convert the multi-class target to a binary heart disease flag
dataset['heart_disease'] = (dataset['num'] > 0).astype(int)

print(dataset.shape)
dataset.head()
```

**Output:**

<img src="images\img1-dataset.png">

The dataset has roughly 740 patients after cleaning, which is well within what we can store and query through GridDB Cloud.

## Inserting Data in GridDB

We will insert the patient records into GridDB Cloud.

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

If you do not see the above response, check this [quick start guide](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) for troubleshooting.


### Creating a GridDB Container for Patient Records

Before inserting data, we need to create a container in GridDB with the correct schema. The following script maps pandas data types to GridDB types.

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

**Output:**
```
[{'name': 'SerialNo', 'type': 'LONG'}, {'name': 'age', 'type': 'LONG'}, {'name': 'sex', 'type': 'STRING'}, {'name': 'cp', 'type': 'STRING'}, {'name': 'trestbps', 'type': 'DOUBLE'}, {'name': 'chol', 'type': 'DOUBLE'}, {'name': 'fbs', 'type': 'STRING'}, {'name': 'restecg', 'type': 'STRING'}, {'name': 'thalch', 'type': 'DOUBLE'}, {'name': 'exang', 'type': 'STRING'}, {'name': 'oldpeak', 'type': 'DOUBLE'}, {'name': 'num', 'type': 'LONG'}, {'name': 'heart_disease', 'type': 'LONG'}]
```

Next, we create the container using the GridDB REST API.

```python
container_name = "heart_disease_db"

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

The 201 status code confirms the container has been created successfully.


### Inserting Patient Records into the GridDB Container

With the container ready, we can now push the patient records into it. The `format_row` function takes care of type conversions, including turning any remaining NaN values into None so GridDB accepts them.

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

payload = json.dumps(rows)

response = requests.put(url, headers=headers, data=payload)

print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")
```

**Output:**
```
Status Code: 200
Response Text: {"count":740}
```

All patient records are now sitting inside the GridDB container.

## Using AI to Analyze Patient Records and Get Insights

With the data stored in GridDB, we can pull it back out and feed a pre-computed summary to Claude. The flow has three stages: fetch records from GridDB, build a structured summary of the dataset, and pass that summary as context so Claude can answer questions about risk factors and patient groups.


### Retrieve Data from GridDB

The script below fetches every patient row from the GridDB container and loads them into a pandas DataFrame.

```python
url = f"{base_url}/containers/{container_name}/rows"

payload = json.dumps({
    "offset": 0,
    "limit": 10000,
    "condition": "",
    "sort": ""
})

response = requests.post(url, headers=headers, data=payload)

print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    try:
        data = response.json()
        print("Data retrieved successfully!")

        rows = data.get("rows", [])
        heart_disease_db = pd.DataFrame(rows, columns=[col for col in dataset.columns])

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON response.")
else:
    print(f"Error: Failed to query data from the container. Response: {response.text}")

print(heart_disease_db.shape)
heart_disease_db.head()
```

**Output:**

<img src="images\img2-griddb-retrieved-data.png">

The patient records have been pulled back from GridDB and are ready for analysis.

### Build Dataset Summary

Sending hundreds of raw rows to Claude in a single prompt is wasteful and can hurt response quality. Instead, we compute a structured summary of the dataset and hand that to the model. This way Claude reasons over real statistics derived from every patient in the database, not just a sampled handful.

```python
NUMERIC_COLS = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']

STAGE_LABELS = {
    0: 'stage 0 (no disease)',
    1: 'stage 1',
    2: 'stage 2',
    3: 'stage 3',
    4: 'stage 4'
}

def build_summary(df):
    parts = []

    # Overview
    total = len(df)
    diseased = int(df['heart_disease'].sum())
    healthy = total - diseased
    parts.append(f"Records: {total} patients")
    parts.append(f"Heart disease present (num > 0): {diseased} ({diseased/total*100:.1f}%)")
    parts.append(f"No heart disease (num = 0): {healthy} ({healthy/total*100:.1f}%)")

    # Stage breakdown using the original num column (0 to 4)
    stage_counts = df['num'].value_counts().sort_index()
    stage_table = pd.DataFrame({
        'patients': stage_counts,
        'pct': (stage_counts / total * 100).round(1)
    })
    stage_table.index = [STAGE_LABELS.get(int(i), str(i)) for i in stage_table.index]
    parts.append(f"\n--- Heart Disease Stage Distribution (num column) ---\n{stage_table.to_string()}")

    # Sex distribution
    sex_counts = df['sex'].value_counts()
    parts.append(f"\n--- Sex Distribution ---\n{sex_counts.to_string()}")

    # Numeric stats overall
    parts.append(f"\n--- Numeric Stats (overall) ---\n{df[NUMERIC_COLS].describe().round(2).to_string()}")

    # Numeric stats split by heart disease status
    grouped = df.groupby('heart_disease')[NUMERIC_COLS].mean().round(2)
    grouped.index = ['no disease', 'disease']
    parts.append(f"\n--- Avg by Heart Disease Status ---\n{grouped.to_string()}")

    # Numeric stats split by stage (0 to 4)
    stage_grouped = df.groupby('num')[NUMERIC_COLS].mean().round(2)
    stage_grouped.index = [STAGE_LABELS.get(int(i), str(i)) for i in stage_grouped.index]
    parts.append(f"\n--- Avg by Stage ---\n{stage_grouped.to_string()}")

    # Chest pain type vs disease (cp is already a string in the Kaggle file:
    # typical angina, atypical angina, non-anginal, asymptomatic)
    cp_counts = df.groupby('cp')['heart_disease'].agg(['count', 'sum'])
    cp_counts['rate_pct'] = (cp_counts['sum'] / cp_counts['count'] * 100).round(1)
    parts.append(f"\n--- Chest Pain Type vs Disease ---\n{cp_counts.to_string()}")

    # Chest pain type vs stage (cross tab)
    stage_named = df['num'].map(STAGE_LABELS)
    cp_stage = pd.crosstab(df['cp'], stage_named)
    parts.append(f"\n--- Chest Pain Type vs Stage (counts) ---\n{cp_stage.to_string()}")

    # Age groups
    bins = [0, 40, 50, 60, 70, 100]
    labels = ['<40', '40-49', '50-59', '60-69', '70+']
    df_local = df.copy()
    df_local['age_group'] = pd.cut(df_local['age'], bins=bins, labels=labels, right=False)
    age_stats = df_local.groupby('age_group').agg(
        patients=('age', 'count'),
        disease_count=('heart_disease', 'sum'),
        avg_stage=('num', 'mean'),
        avg_chol=('chol', 'mean'),
        avg_bp=('trestbps', 'mean'),
        avg_max_hr=('thalch', 'mean')
    ).round(2)
    age_stats['disease_rate_pct'] = (age_stats['disease_count'] / age_stats['patients'] * 100).round(1)
    parts.append(f"\n--- Age Group Breakdown ---\n{age_stats.to_string()}")

    # Sex vs disease
    sex_stats = df.groupby('sex')['heart_disease'].agg(['count', 'sum'])
    sex_stats['rate_pct'] = (sex_stats['sum'] / sex_stats['count'] * 100).round(1)
    parts.append(f"\n--- Sex vs Disease ---\n{sex_stats.to_string()}")

    # High cholesterol flag
    df_local['high_chol'] = df_local['chol'] >= 240
    chol_stats = df_local.groupby('high_chol')['heart_disease'].agg(['count', 'sum'])
    chol_stats['rate_pct'] = (chol_stats['sum'] / chol_stats['count'] * 100).round(1)
    chol_stats.index = ['chol < 240', 'chol >= 240']
    parts.append(f"\n--- High Cholesterol vs Disease ---\n{chol_stats.to_string()}")

    # Exercise-induced angina
    exang_stats = df.groupby('exang')['heart_disease'].agg(['count', 'sum'])
    exang_stats['rate_pct'] = (exang_stats['sum'] / exang_stats['count'] * 100).round(1)
    parts.append(f"\n--- Exercise-Induced Angina vs Disease ---\n{exang_stats.to_string()}")

    return "\n".join(parts)


data_summary = build_summary(heart_disease_db)
print(data_summary[:600])
```

**Output:**

```
Records: 740 patients
Heart disease present (num > 0): 383 (51.8%)
No heart disease (num = 0): 357 (48.2%)

--- Heart Disease Stage Distribution (num column) ---
                      patients   pct
stage 0 (no disease)       357  48.2
stage 1                    204  27.6
stage 2                     79  10.7
stage 3                     78  10.5
stage 4                     22   3.0

--- Sex Distribution ---
sex
Male      566
Female    174

--- Numeric Stats (overall) ---
          age  trestbps    chol  thalch  oldpeak
count  740.00    740.00  740.00  740.00   740.00
mean    53.10    132.75  22
```


The `build_summary` function divides the patient data into eleven blocks. Each block covers a different aspect of the dataset, and together they give Claude both the binary and the staged view of heart disease.

* **Overview counts.** Total number of patients, how many have heart disease (`num > 0`), and how many do not, with percentages for both groups.
* **Stage distribution.** A table built from the raw `num` column showing how many patients fall into stage 0, stage 1, stage 2, stage 3, and stage 4, along with the percentage share of each stage. This is the block that lets Claude reason about severity rather than just presence.
* **Sex distribution.** Counts of male and female patients in the dataset.
* **Numeric stats overall.** The pandas `describe()` output for age, resting blood pressure (trestbps), cholesterol (chol), maximum heart rate achieved (thalch), and ST depression (oldpeak). This gives Claude the mean, standard deviation, min, max, and quartiles for every numeric clinical feature.
* **Averages by binary disease status.** The same five numeric features averaged separately for the no-disease group and the disease group, so Claude can quickly point out which features differ the most when disease is present.
* **Averages by stage.** The same five numeric features averaged across each value of `num` from 0 to 4. This block is what lets the chatbot describe how cholesterol, maximum heart rate, ST depression, and the rest shift as severity increases.
* **Chest pain type vs disease.** For every chest pain category in the `cp` column (typical angina, atypical angina, non-anginal, asymptomatic), the number of patients, the count with heart disease, and the disease rate as a percentage.
* **Chest pain type vs stage.** A table counting how many patients land in each combination of chest pain type and disease stage. This is useful for spotting which chest pain category dominates in the higher stages.
* **Age group breakdown.** Patients are bucketed into <40, 40-49, 50-59, 60-69, and 70+. For each bucket the function reports the patient count, the disease count, the average stage, the average cholesterol, the average resting blood pressure, the average maximum heart rate, and the disease rate as a percentage.
* **Sex vs disease.** Disease counts and rates split by male and female patients.
* **Cholesterol threshold and exercise-induced angina.** Two short tables checking whether having cholesterol at or above 240 mg/dL changes the disease rate, and whether exercise-induced angina does the same. Both appear in the standard heart disease literature as risk markers, so they earn their own blocks in the summary.

These eleven blocks together stay well within Claude's context window even though they are computed over every patient in the database, which is the whole point of pre-aggregating before sending the data to the model.


### Create Chatbot

With the summary in place, we wire up the chatbot. The system prompt embeds the entire summary and tells Claude to back every observation with concrete numbers. The `HeartDiseaseChatbot` class keeps a running conversation history so follow-up questions naturally build on what was already discussed.

In the script below we use the Claude Sonnet 4.6 model. You can swap in any other Anthropic model if you prefer.

```python
SYSTEM_PROMPT = f"""You are a clinical data analyst. You have records for {len(heart_disease_db)} patients
from the UCI Heart Disease dataset. Each record includes age, sex, chest pain type (cp),
resting blood pressure (trestbps), cholesterol (chol), fasting blood sugar (fbs),
resting ECG (restecg), maximum heart rate achieved (thalch), exercise-induced angina (exang),
and ST depression (oldpeak). The heart_disease column is 1 when disease is present and 0 otherwise.

Always ground your answers in the specific numbers from the summary below. When you give
recommendations, frame them as data-backed observations rather than medical advice, and remind
the user that any clinical decision should be reviewed by a qualified healthcare professional.

DATA:
{data_summary}
"""


class HeartDiseaseChatbot:
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


chatbot = HeartDiseaseChatbot()
print("Chatbot ready!")
```

**Output:**

```
Chatbot ready!
```

The chatbot is initialized and ready to answer questions about the patient records.


### Ask Questions

Let's test the chatbot with three questions covering risk factor patterns, age group comparisons, and a follow-up that builds on the previous answer.

#### Question 1: Most Common Risk Factors Among Heart Disease Patients

```python
print(chatbot.ask("Which risk factors appear most often among patients who have heart disease? Cite specific numbers."))
```

**Output (partial screenshot):**

<img src="images\img3-risk-factors-partial-output.png">

The chatbot walks through the dominant risk patterns. Asymptomatic chest pain stands out as the strongest signal, with disease rates well above the rates seen for typical angina or non-anginal pain. Patients in the disease group also show a noticeably higher average ST depression value and a lower average maximum heart rate during exercise compared to patients without disease. Cholesterol differences are smaller than most people expect, while exercise-induced angina is far more common in the disease group.

#### Question 2: Age Group Comparison

```python
print(chatbot.ask("Compare heart disease rates across age groups. How do the average cholesterol and resting blood pressure shift with age?"))
```

**Output (partial screenshot):**

<img src="images\img4-age-groups-partial-output.png">

Claude lays out the age group breakdown, showing how disease rates climb from the youngest cohort through the 60-69 bracket. It also notes that resting blood pressure trends upward with age while maximum achievable heart rate drops, which lines up with what is seen in clinical practice.

#### Question 3: Best Single Indicator

```python
print(chatbot.ask("Based on what you just told me, which single feature would be the most useful early warning sign for a clinician to focus on?"))
```

**Output (partial screenshot):**

<img src="images\img5-best-indicator-partial-output.png">

Because the chatbot retains conversation history, this follow-up question builds directly on the previous analysis. Claude points to asymptomatic chest pain combined with reduced maximum heart rate during exercise as the most discriminating combination in the dataset, and frames the conclusion as a data observation rather than a clinical instruction.


### Execute Code and Generate Charts

Beyond text answers, we can ask Claude to write matplotlib code for visualizations. The `generate_and_run_chart` function sends a chart description to Claude, takes the returned Python code, and runs it inline so the chart appears right under the request.

```python

client = anthropic.Anthropic()

def generate_and_run_chart(question):
    """
    Ask Claude to generate matplotlib code for a chart, then execute it.
    The chart displays inline in the notebook.
    """
    schema = f"Columns: {list(heart_disease_db.columns)}\nSample:\n{heart_disease_db.head(3).to_csv(index=False)}"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""I have a pandas DataFrame `heart_disease_db` with this structure:

{schema}

The heart_disease column is 1 if disease is present and 0 otherwise.
Chest pain type (cp) is a string column with values: 'typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'.

Write Python code using pandas and matplotlib to: {question}

RULES:
- Return ONLY Python code, no markdown fences, no explanation
- heart_disease_db is already loaded, matplotlib.pyplot is imported as plt
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

We create a module-level `client` at the top of the cell because `generate_and_run_chart` references it directly inside its body. The `HeartDiseaseChatbot` class earlier built its own client as an instance attribute (`self.client`), which is not visible from outside the class, so the chart helper needs its own handle.

The function passes the dataframe schema and a plain English chart description to Claude, which sends back ready-to-run matplotlib code. The `exec(code, globals())` call runs the code in the global scope so it can see `heart_disease_db`, `plt`, and `pd`.

Let's plot a few charts that highlight the patterns we discussed above.

#### Chart 1: Disease Rate by Age Group

```python
generate_and_run_chart("Bar chart showing the percentage of patients with heart disease across age groups (<40, 40-49, 50-59, 60-69, 70+). Put the percentage on top of each bar.")
```

**Output:**

<img src="images\img6-disease-rate-by-age.png">

The chart confirms the steady climb in disease rate from the youngest cohort through middle and older age groups.

#### Chart 2: Disease Rate by Chest Pain Type

```python
generate_and_run_chart("Bar chart showing the percentage of patients with heart disease for each chest pain type. Use the labels typical angina, atypical angina, non-anginal, asymptomatic on the x axis.")
```

**Output:**

<img src="images\img7-disease-rate-by-cp.png">

Asymptomatic chest pain ends up with by far the highest disease rate, which matches what the chatbot pointed out in its first answer.

#### Chart 3: Cholesterol Distribution by Disease Status

```python
generate_and_run_chart("Two overlapping histograms of cholesterol values, one for patients with heart disease and one for patients without. Use different colors and add a legend.")
```

**Output:**

<img src="images\img8-cholesterol-distribution.png">

The histogram shows that while cholesterol distributions for the two groups overlap considerably, the disease group has a slightly heavier right tail.

## Conclusion

This article showed how to combine the Anthropic Claude API with GridDB Cloud to build an AI-powered heart disease analysis chatbot. The chatbot pulls patient records out of GridDB, summarizes them on the fly, and lets you ask plain natural language questions about risk factors, age group differences, and individual features. The chart generation step extends it further by producing matplotlib visualizations from natural language descriptions, so you can dig into the data visually without writing plotting code by hand.

You can extend this approach by adding more clinical features such as serum thalassemia type or number of major vessels colored by fluoroscopy, by joining records from multiple cohorts for cross-population comparison, or by hooking the chatbot up to live patient databases for ongoing monitoring. The pairing of a fast structured database and a reasoning model opens up plenty of room for clinical decision support tools, so long as the medical professionals stay firmly in the loop.

You can find the complete code for this blog on the [GridDB Blogs GitHub repository](https://github.com/usmanmalik57/GridDB-Blogs/tree/main/Heart%20Disease%20Analysis%20Chatbot%20with%20GridDB%20and%20Claude%20AI). If you have any questions or queries related to GridDB, create a post on Stack Overflow with the `griddb` tag. Our engineers are more than happy to respond.