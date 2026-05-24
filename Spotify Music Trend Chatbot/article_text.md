Music streaming platforms generate enormous catalogs of track data. For example, Spotify's library spans dozens of genres with detailed audio features for every song. Spotting which genres are pulling ahead, what makes a track popular, or how audio properties like danceability and energy line up with popularity is hard to do by eye when the catalog runs into the hundreds of thousands of records. Pairing a high-performance database like [GridDB](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) with an AI model lets you store the full catalog once and then ask plain English questions about trends, audio profiles, and genre patterns.

In this article, you will see how to build an AI-powered Spotify music trends chatbot using the [Anthropic](https://www.anthropic.com/) Claude API and GridDB Cloud. The chatbot reads Spotify track records that include popularity scores, audio features such as danceability, energy, loudness, valence, and tempo, and the track genre. It identifies which genres are trending, points out which audio features tend to go with higher popularity scores, and compares the sonic profile of one genre against another. You will also use Claude to create matplotlib visualizations so you can explore the patterns visually.


**Prerequisites**:
You will need the following to run scripts in this article:

* A [GridDB cloud account](https://www.global.toshiba/ww/products-solutions/ai-iot/griddb/product/griddb-cloud.html). Refer to this [quick start guide](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) to set up a GridDB cloud account.
* [Anthropic API Key](https://platform.claude.com/). You can obtain one from the Anthropic console.

Note: You can find the complete code for this tutorial in my [GridDB Blogs GitHub repository](https://github.com/usmanmalik57/GridDB-Blogs/tree/main/Spotify%20Music%20Trend%20Chatbot).


## Installing and Importing Required Libraries

Two pip installs cover everything we need on top of the standard data science stack. The `python-dotenv` package lets us keep our GridDB credentials and Anthropic API key in a `.env` file instead of hard-coding them, and the `anthropic` package is the official client for talking to the Claude API.

```
!pip install python-dotenv
!pip install anthropic
```

The imports below cover both halves of the project. The `pandas` and `matplotlib` libraries do the data wrangling and plotting, `base64`, `requests`, and `json` handle the GridDB REST calls, `re` is used later to strip markdown fences out of Claude-generated code, and the `load_dotenv()` call pulls our environment variables into the process.

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

We will work with the Spotify Tracks Dataset hosted on Kaggle. It is a catalog of roughly 114,000 tracks pulled from the Spotify Web API, covering 125 genres. Every row is a track, and alongside the basic metadata (track name, artist, album, genre), each one carries a popularity score from 0 to 100 and a full set of audio features computed by Spotify. Those features include danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, and tempo, all of which feed into the analysis later. You can [download the dataset from Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset).

The following script loads the dataset, drops rows with missing values in the columns we care about, and keeps the first 20,000 records. The full catalog has roughly 114k tracks but 20k is plenty for this sample analysis and keeps the GridDB inserts quick.

One important note before running: we do not keep the `Unnamed: 0` column that ships with the raw CSV. GridDB does not allow spaces or colons in column names, and that column name has both. The explicit `keep_cols` list below sidesteps the problem.

A second thing worth knowing about this dataset is that the rows are sorted alphabetically by genre. Taking the first 20,000 rows therefore captures roughly the first 20 genres (anything from acoustic and afrobeat through club and country) with about 1,000 tracks each, rather than a random cross-section of the full 125 genres. That is fine for the demo since the analysis pipeline is what we are showing off, but if you wanted a representative slice of the whole catalog you would want to shuffle the dataframe first.

```python
dataset = pd.read_csv("dataset.csv", encoding='utf-8')

keep_cols = ['track_id', 'artists', 'album_name', 'track_name', 'popularity',
             'duration_ms', 'explicit', 'danceability', 'energy', 'loudness',
             'speechiness', 'acousticness', 'instrumentalness', 'liveness',
             'valence', 'tempo', 'time_signature', 'track_genre']
dataset = dataset[keep_cols].dropna().reset_index(drop=True)
dataset = dataset[:20000]

print(dataset.shape)
dataset.head()
```

**Output:**

<img src="images\img1-dataset.png">


## Inserting Data in GridDB

With the dataframe ready, the next job is to push it into GridDB Cloud. GridDB is exposed over a REST API, so the workflow is to test the connection, define a container schema, create the container, and then write the rows into it.

### Creating a GridDB Connection

The script below confirms that we can reach the GridDB cloud instance. The username, password, and base URL are pulled from a `.env` file via `os.environ.get`, and the basic auth header is built by base64-encoding the `username:password` string. A 200 response means we are good to go.

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

### Creating a GridDB Container for Spotify Track Data

GridDB needs a schema before it will accept any rows. Each column has to be declared with both a name and a GridDB type, so we read the pandas dtypes off the dataframe and translate them into the matching GridDB types. We also prepend a `SerialNo` column at position zero, which will act as the rowkey for the container.

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

```
[{'name': 'SerialNo', 'type': 'LONG'}, {'name': 'track_id', 'type': 'STRING'}, {'name': 'artists', 'type': 'STRING'}, {'name': 'album_name', 'type': 'STRING'}, {'name': 'track_name', 'type': 'STRING'}, {'name': 'popularity', 'type': 'LONG'}, {'name': 'duration_ms', 'type': 'LONG'}, {'name': 'explicit', 'type': 'BOOL'}, {'name': 'danceability', 'type': 'DOUBLE'}, {'name': 'energy', 'type': 'DOUBLE'}, {'name': 'loudness', 'type': 'DOUBLE'}, {'name': 'speechiness', 'type': 'DOUBLE'}, {'name': 'acousticness', 'type': 'DOUBLE'}, {'name': 'instrumentalness', 'type': 'DOUBLE'}, {'name': 'liveness', 'type': 'DOUBLE'}, {'name': 'valence', 'type': 'DOUBLE'}, {'name': 'tempo', 'type': 'DOUBLE'}, {'name': 'time_signature', 'type': 'LONG'}, {'name': 'track_genre', 'type': 'STRING'}]
```

With the column list in hand, the next call creates the container itself. The payload tells GridDB the container name, that we want a `COLLECTION` (the right choice for general tabular data), that the first column should be treated as the rowkey, and the full column schema we just built.

```python
container_name = "spotify_tracks_db"

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

A 201 response means GridDB has accepted the schema and the container is live. If you instead see a 400, the most common cause is a column name with a forbidden character (a space, colon, or hyphen will all trip it). The second most common cause is trying to create a container that already exists from a previous run, in which case you can DELETE the old one and create it again.

### Inserting Data in Spotify Tracks GridDB Container

The container is empty at this point. To fill it, we send a PUT request with a JSON array of rows, where each row is itself an array of values in the same order as the column schema. The small `format_row` helper deals with the type quirks that show up at the JSON boundary: pandas NaN has to become a JSON null, Python booleans need to be lowercased to match JavaScript conventions, and numeric types can pass through untouched.

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
Response Text: {"count":20000}
```

A 200 response with `{"count":20000}` confirms that every row landed in the container. From this point on the dataset lives inside GridDB, and the rest of the article reads it back through the REST API rather than touching the CSV again.


## Using AI to Analyze Spotify Music Trends

Now comes the AI half of the project. We will fetch the catalog back from GridDB, condense it into a structured summary that fits comfortably inside a single prompt, and hand that summary to Claude as the context for an interactive question-and-answer loop. The condensed summary is the trick that makes this work at scale: it lets Claude reason over statistics computed from every row in the container without ever seeing the raw 20,000 rows directly.


### Retrieve Data from GridDB

GridDB exposes a `/rows` endpoint on each container that returns matching rows as JSON. We send an empty `condition` and `sort` so it returns everything, with a `limit` of 20000 to cover the full sample we inserted earlier. The response comes back as a `rows` array of arrays, which we wrap into a pandas DataFrame using the original column order from `dataset.columns`.


```python

url = f"{base_url}/containers/{container_name}/rows"

# Define the payload for the query
payload = json.dumps({
    "offset": 0,           # Start from the first row
    "limit": 20000,         # Limit the number of rows returned
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
        spotify_tracks_db = pd.DataFrame(rows, columns=[col for col in dataset.columns])

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON response.")
else:
    print(f"Error: Failed to query data from the container. Response: {response.text}")

print(spotify_tracks_db.shape)
spotify_tracks_db.head()
```

**Output:**

<img src="images\img2-griddb-retrieved-data.png">

The dataframe coming out of GridDB is identical to the one we sent in, which is exactly what we want.


### Build Dataset Summary

This is the most important step. The Claude API has a generous context window, but stuffing 20,000 rows of song metadata into a single prompt is still wasteful, slow, and noisy. A much better approach is to do the aggregation work upfront in pandas and pass Claude a tight, structured summary of what the catalog actually looks like. Once that summary is in the system prompt, Claude can answer almost any high-level question about trends, profiles, and rankings without ever needing to see the raw rows.

```python
AUDIO_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness',
                  'valence', 'tempo']

POPULARITY_BUCKETS = {
    'emerging (0-25)':       (0, 25),
    'up and coming (26-50)': (26, 50),
    'mainstream (51-75)':    (51, 75),
    'chart topper (76-100)': (76, 100),
}


def build_summary(df):
    parts = []

    # Overview
    total = len(df)
    n_genres = df['track_genre'].nunique()
    n_artists = df['artists'].nunique()
    parts.append(f"Records: {total} tracks")
    parts.append(f"Unique genres: {n_genres}")
    parts.append(f"Unique artist entries: {n_artists}")
    parts.append(f"Explicit tracks: {int(df['explicit'].sum())} ({df['explicit'].mean()*100:.1f}%)")

    # Popularity overview
    parts.append(f"\n--- Popularity Stats ---\n{df['popularity'].describe().round(2).to_string()}")

    # Popularity bucket breakdown
    bucket_rows = []
    for label, (low, high) in POPULARITY_BUCKETS.items():
        mask = (df['popularity'] >= low) & (df['popularity'] <= high)
        bucket_rows.append({
            'bucket': label,
            'tracks': int(mask.sum()),
            'pct': round(mask.mean() * 100, 1)
        })
    bucket_df = pd.DataFrame(bucket_rows).set_index('bucket')
    parts.append(f"\n--- Popularity Buckets ---\n{bucket_df.to_string()}")

    # Audio feature stats overall
    parts.append(f"\n--- Audio Feature Stats (overall) ---\n{df[AUDIO_FEATURES].describe().round(3).to_string()}")

    # Top 10 genres by average popularity
    genre_pop = df.groupby('track_genre').agg(
        tracks=('popularity', 'count'),
        avg_popularity=('popularity', 'mean')
    ).round(2)
    top_genres = genre_pop.sort_values('avg_popularity', ascending=False).head(10)
    bottom_genres = genre_pop.sort_values('avg_popularity', ascending=True).head(10)
    parts.append(f"\n--- Top 10 Genres by Avg Popularity ---\n{top_genres.to_string()}")
    parts.append(f"\n--- Bottom 10 Genres by Avg Popularity ---\n{bottom_genres.to_string()}")

    # Audio feature averages for the top 10 genres
    top_genre_names = top_genres.index.tolist()
    top_genre_features = df[df['track_genre'].isin(top_genre_names)] \
                            .groupby('track_genre')[AUDIO_FEATURES].mean().round(3)
    parts.append(f"\n--- Audio Features for Top 10 Genres ---\n{top_genre_features.to_string()}")

    # Average audio features by popularity bucket
    df_local = df.copy()
    def assign_bucket(p):
        for label, (low, high) in POPULARITY_BUCKETS.items():
            if low <= p <= high:
                return label
        return 'unknown'
    df_local['pop_bucket'] = df_local['popularity'].apply(assign_bucket)
    bucket_order = list(POPULARITY_BUCKETS.keys())
    feat_by_bucket = df_local.groupby('pop_bucket')[AUDIO_FEATURES].mean().reindex(bucket_order).round(3)
    parts.append(f"\n--- Avg Audio Features by Popularity Bucket ---\n{feat_by_bucket.to_string()}")

    # Correlation between audio features and popularity
    corr = df[AUDIO_FEATURES + ['popularity']].corr()['popularity'].drop('popularity').round(3)
    corr = corr.sort_values(ascending=False)
    parts.append(f"\n--- Correlation of Audio Features with Popularity ---\n{corr.to_string()}")

    # Explicit vs non-explicit popularity
    expl = df.groupby('explicit')['popularity'].agg(['count', 'mean']).round(2)
    expl.index = ['non-explicit', 'explicit']
    parts.append(f"\n--- Explicit vs Non-Explicit Popularity ---\n{expl.to_string()}")

    # Tempo bands
    tempo_bins = [0, 90, 120, 150, 300]
    tempo_labels = ['slow (<90 BPM)', 'medium (90-120 BPM)',
                    'fast (120-150 BPM)', 'very fast (150+ BPM)']
    df_local['tempo_band'] = pd.cut(df_local['tempo'], bins=tempo_bins, labels=tempo_labels, right=False)
    tempo_stats = df_local.groupby('tempo_band').agg(
        tracks=('popularity', 'count'),
        avg_popularity=('popularity', 'mean')
    ).round(2)
    parts.append(f"\n--- Popularity by Tempo Band ---\n{tempo_stats.to_string()}")

    # Top 10 most popular individual tracks
    top_tracks = df.sort_values('popularity', ascending=False) \
                   .head(10)[['track_name', 'artists', 'track_genre', 'popularity']]
    parts.append(f"\n--- Top 10 Most Popular Tracks in Sample ---\n{top_tracks.to_string(index=False)}")

    return "\n".join(parts)


data_summary = build_summary(spotify_tracks_db)
print(data_summary[:600])
```

**Output:**

```
Records: 20000 tracks
Unique genres: 20
Unique artist entries: 6169
Explicit tracks: 1635 (8.2%)

--- Popularity Stats ---
count    20000.00
mean        31.49
std         21.70
min          0.00
25%         16.00
50%         29.00
75%         49.00
max         93.00

--- Popularity Buckets ---
                       tracks   pct
bucket                             
emerging (0-25)          9115  45.6
up and coming (26-50)    6408  32.0
mainstream (51-75)       4212  21.1
chart topper (76-100)     265   1.3

--- Audio Feature Stats (overall) ---
       danceability     energy   loudness  speechi
```


The `build_summary` function splits the catalog into 11 blocks. Each block answers a different question about the data, and together they give Claude a wide view of the music trends without having to send the raw rows.

* **Overview counts.** Total number of tracks, number of unique genres, number of unique artist entries, and how many tracks are flagged as explicit.
* **Popularity stats.** The pandas `describe()` output for the popularity column, which gives Claude the mean, standard deviation, quartiles, min, and max popularity score.
* **Popularity buckets.** Tracks are grouped into four buckets called emerging, up and coming, mainstream, and chart topper, based on popularity ranges. This lets Claude reason about how the catalog is distributed across popularity tiers.
* **Audio feature stats.** The describe output for danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, and tempo. This is the block that lets Claude talk about the overall sonic profile of the catalog.
* **Top 10 and bottom 10 genres.** Two short tables sorting genres by their average popularity, with the track count alongside so Claude can flag genres that have small sample sizes.
* **Audio features for top 10 genres.** Average audio features computed only over the highest-scoring genres, which is what lets Claude describe the typical sound of a trending genre instead of just naming it.
* **Average audio features by popularity bucket.** The same audio features averaged across the four popularity buckets. This is where you start to see whether high-popularity tracks tend to be louder, faster, more danceable, or more energetic than the rest.
* **Correlation with popularity.** A short table ranking each audio feature by its Pearson correlation with the popularity score, sorted from most positive to most negative.
* **Explicit vs non-explicit popularity.** Track counts and average popularity for explicit and non-explicit tracks.
* **Tempo bands.** Tracks bucketed into slow, medium, fast, and very fast tempo ranges, with the average popularity for each band.
* **Top 10 individual tracks.** A small table of the most popular individual songs in the sample, with their artist and genre, so Claude can give concrete examples when answering questions.

All of these blocks stay well within Claude's context window even though they were computed over every track in the GridDB container, which is the point of pre-aggregating before sending the data to the model.

### Create Chatbot

The chatbot itself is a thin class around the Anthropic client. The system prompt is where most of the work happens: it tells Claude what role it is playing, what columns exist, and what the popularity buckets mean, then embeds the entire summary we just built so the model has the numbers right there in front of it. The instruction to ground every answer in concrete numbers, and to flag genres with small sample sizes, is what stops the model from drifting into vague pop-music opinions.

The `SpotifyChatbot` class keeps a list of past messages, so each call to `ask` builds on the conversation rather than starting fresh. That is what makes the follow-up question in the next section actually work.

In the script below we use the Claude Sonnet 4.6 model. You can swap in any other Anthropic model if you prefer.

```python
SYSTEM_PROMPT = f"""You are a music trends analyst. You have data for {len(spotify_tracks_db)} Spotify tracks
spanning {spotify_tracks_db['track_genre'].nunique()} genres. Each track includes a popularity score from 0 to 100
and audio features such as danceability, energy, loudness, speechiness, acousticness, instrumentalness,
liveness, valence, and tempo. Popularity buckets are emerging (0-25), up and coming (26-50),
mainstream (51-75), and chart topper (76-100).

Always ground your answers in the specific numbers from the summary below. When you make claims about a
trending genre or a popular audio profile, cite the actual averages and counts. If a genre has very few
tracks in the sample, mention that as a caveat.

DATA:
{data_summary}
"""


class SpotifyChatbot:
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


chatbot = SpotifyChatbot()
print("Chatbot ready!")
```

**Output:**

```
Chatbot ready!
```

The chatbot is initialized and ready to answer questions about the Spotify catalog.


### Ask Questions

Let's test the chatbot with three questions covering trending genres, the audio profile of popular tracks, and a follow-up that builds on the previous answer.

#### Question 1: Trending Genres

```python
print(chatbot.ask("Which genres are trending in this catalog? Cite specific average popularity scores and track counts."))
```

**Output (partial screenshot):**

<img src="images\img3-trending-genres-partial-output.png">

The chatbot reads straight off the top 10 table and reports `chill` as the clear leader at an average popularity of 53.65, the only genre in the sample that averages into the mainstream bucket. Anime follows at 48.77, with brazil (44.67), ambient (44.19), british (43.80), and acoustic (42.48) rounding out the upper tier. It also flags the floor of the catalog (`chicago-house` at 12.34 and `classical` at 13.06) and reminds the reader that every genre in this slice has exactly 1,000 tracks, so the comparisons are at least on equal footing. The overall mean popularity of 31.49 puts the leaders 12 to 22 points above the catalog average, which is what makes them stand out.

#### Question 2: Audio Profile of Popular Tracks

```python
print(chatbot.ask("What audio features tend to make a song popular? Look at the correlations and the averages across popularity buckets."))
```

**Output (partial screenshot):**

<img src="images\img4-audio-profile-partial-output.png">

Claude works through the correlation table first and immediately flags that none of the audio features carries a strong linear signal against popularity. The strongest correlation in the whole sample is instrumentalness at -0.112, with energy at -0.101 and valence at -0.096 right behind it. Acousticness is the only feature on the positive side at +0.047. The bucket averages tell a richer story than the raw correlations though. Instrumentalness drops cleanly from 0.258 in the emerging bucket down to 0.046 for chart toppers, the cleanest monotonic decline in the data. Energy is U-shaped: high in emerging (0.636), lowest in mainstream (0.510), then jumping back up to 0.692 for the chart toppers. Acousticness shows the inverse pattern, rising from 0.352 through 0.440 in mainstream and then collapsing to 0.164 at the top. And chart toppers are noticeably louder than the rest of the catalog at -6.83 dB compared to a sample average around -9.43 dB.

#### Question 3: Best Single Audio Feature to Optimize

```python
print(chatbot.ask("Based on what you just told me, if an artist could optimize for one audio feature to lift the popularity of a new track, which one should it be and why?"))
```

**Output (partial screenshot):**

<img src="images\img5-best-audio-feature-partial-output.png">

Because the chatbot keeps the conversation history, this follow-up builds directly on the previous answer. Claude picks instrumentalness as the single best lever, with the reasoning that it has both the strongest correlation in the table (-0.112) and the cleanest staircase pattern across the four buckets, going from 0.258 down to 0.046 with no zigzags along the way. The model points out that this is also the most actionable feature for an artist since adding or strengthening a vocal is a direct compositional choice, unlike tempo or valence which depend on the song itself. It closes with the appropriate caveats: the correlation is still weak in absolute terms, the chart topper bucket only has 265 tracks, and popular genres in this slice (chill, anime, british) skew vocal anyway, so genre choice may be the real driver with instrumentalness just along for the ride.


### Execute Code and Generate Charts

Beyond text answers, we can ask Claude to write matplotlib code for visualizations. The `generate_and_run_chart` function sends a chart description to Claude, takes the returned Python code, and runs it inline so the chart shows up right under the request.

```python

client = anthropic.Anthropic()

def generate_and_run_chart(question):
    """
    Ask Claude to generate matplotlib code for a chart, then execute it.
    The chart displays inline in the notebook.
    """
    schema = f"Columns: {list(spotify_tracks_db.columns)}\nAudio feature cols: {AUDIO_FEATURES}\nSample:\n{spotify_tracks_db.head(3).to_csv(index=False)}"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""I have a pandas DataFrame `spotify_tracks_db` with this structure:

{schema}

Popularity is an integer from 0 to 100. Audio features (danceability, energy, valence,
acousticness, instrumentalness, liveness, speechiness) are floats between 0 and 1.
Loudness is in decibels and tempo is in BPM. track_genre is a string column.

Write Python code using pandas and matplotlib to: {question}

RULES:
- Return ONLY Python code, no markdown fences, no explanation
- spotify_tracks_db is already loaded, matplotlib.pyplot is imported as plt
- Use plt.figure(figsize=(10, 6)) for good sizing
- Always include plt.tight_layout() and plt.show()
- Add a clear title, axis labels, and a legend where needed"""
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

We create a module-level `client` at the top of the cell because `generate_and_run_chart` references it directly inside its body. The `SpotifyChatbot` class above built its own client as an instance attribute (`self.client`), which is not visible from outside the class, so the chart helper needs its own handle.

The function passes the dataframe schema and a plain English chart description to Claude, which sends back ready-to-run matplotlib code. The `exec(code, globals())` call runs the code in the global scope so it can see `spotify_tracks_db`, `plt`, and `pd`.

Let's plot a few charts that highlight the patterns we talked about above.

#### Chart 1: Top Genres by Average Popularity

```python
generate_and_run_chart("Horizontal bar chart showing the top 10 genres by average popularity score. Sort highest to lowest and put the popularity value at the end of each bar.")
```

**Output:**

<img src="images\img6-top-genres-by-popularity.png">

The bar chart makes the gap between the leaders and the pack obvious at a glance. Chill sits at the top with an average popularity of 53.7, well clear of the next group of anime (48.8), brazil (44.7), ambient (44.2), and british (43.8), which all cluster together in the low to mid 40s. The bottom of the top 10 falls off sharply, with club at 33.3.

#### Chart 2: Audio Feature Correlations with Popularity

```python
generate_and_run_chart("Horizontal bar chart showing the Pearson correlation of each audio feature (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo) with the popularity score. Sort from most positive to most negative and use different colors for positive and negative values.")
```

**Output:**

<img src="images\img7-feature-correlation-with-popularity.png">

The correlation chart shows only three features lean positive against popularity (acousticness at 0.047, loudness at 0.027, and danceability at 0.010), and the strongest signals are all on the negative side: instrumentalness at -0.112, energy at -0.101, and valence at -0.096. The bigger takeaway is the scale of the x-axis itself. Every single bar sits between -0.12 and +0.05, which is a visual reminder that no audio feature on its own is doing much work to predict popularity in this dataset.

#### Chart 3: Average Audio Features by Popularity Bucket

```python
generate_and_run_chart("Grouped bar chart showing average danceability, energy, valence, and acousticness for each of the four popularity buckets: emerging (0-25), up and coming (26-50), mainstream (51-75), and chart topper (76-100). Put bucket on the x axis and the four features as grouped bars.")
```

**Output:**

<img src="images\img8-features-by-popularity-bucket.png">

The grouped bar chart is where the non-linear patterns become very easy to see. Danceability stays almost flat at around 0.55 across all four buckets, valence drifts down slightly, and acousticness climbs from emerging up through mainstream before collapsing at the chart topper level. Energy is the most striking shape: it falls from 0.63 in emerging down to 0.51 in mainstream and then springs back up to 0.69 for chart toppers. So the average chart topper is loud, energetic, and decidedly non-acoustic, which is a different profile from a typical mainstream-bucket track.

## Conclusion
This article showed how to combine the Anthropic Claude API with GridDB Cloud to build an AI-powered Spotify music trends chatbot. The chatbot pulls 20,000 track records out of GridDB, builds a pre-computed summary covering genres, popularity buckets, audio features, and correlations, and then lets you ask plain natural language questions about which genres are trending and what audio profile sits behind a popular track. The chart generation step extends it further by producing matplotlib visualizations from natural language descriptions, so you can dig into the patterns visually without writing plotting code by hand.

The results were a useful reminder that single audio features are weak predictors of popularity on their own (the strongest correlation in the sample was instrumentalness at just -0.112), but the bucket-level averages still tell a clear story: chart toppers in this slice tend to be vocal-forward, loud, and energetic, with very little acoustic character. The fact that `chill` was the only genre to average into the mainstream bucket was a surprise that fell straight out of the data.

You can extend this approach by pulling in fresh data from the Spotify Web API for ongoing trend tracking, by joining the track records against artist-level or album-level tables for richer analysis, or by hooking the chatbot up to a recommendation engine that uses the audio feature columns to find similar songs. The pairing of a fast database like GridDB and a reasoning model opens up plenty of room for music analytics tooling, whether the goal is artist research or just exploring a personal listening history.

You can find the complete code for this blog on the [GridDB Blogs GitHub repository](https://github.com/usmanmalik57/GridDB-Blogs/tree/main/Spotify%20Music%20Trend%20Chatbot). If you have any questions or queries related to GridDB, create a post on Stack Overflow with the `griddb` tag. Our engineers are more than happy to respond.
