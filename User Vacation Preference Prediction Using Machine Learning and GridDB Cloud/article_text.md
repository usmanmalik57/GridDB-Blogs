
User vacation location preferences reflect individual lifestyles, personality traits, and travel habits. Predicting these preferences can enhance recommendation systems and personalize travel experiences.

This article demonstrates how to build a vacation preference prediction system using machine learning techniques and the [GridDB cloud](https://www.global.toshiba/ww/products-solutions/ai-iot/griddb/product/griddb-cloud.html) database.

We will begin by retrieving a user vacation preference dataset from Kaggle, storing it in a GridDB cloud container, and using the data to train a predictive model to identify user vacation preferences.

GridDB is a high-performance NoSQL database that offers efficient in-memory processing and scalable data storage. This makes it a strong choice for machine learning applications that demand real-time interactions and cloud-native deployment.

**Note**: All the code used in this tutorial is available in my [GridDB Blogs GitHub repository](https://github.com/usmanmalik57/GridDB-Blogs/tree/main/User%20Vacation%20Preference%20Prediction%20Using%20Machine%20Learning%20and%20GridDB%20Cloud).


## Prerequisites

You can access the GridDB cloud through any REST API library. You can use the `requests` library to access the GridDB cloud in Python.

The following script installs the required libraries and modules, which you will use to access GridDB and train machine learning models to predict user vacation preferences.

```
!pip install -q pandas seaborn matplotlib scikit-learn tensorflow requests
```

The script below imports the required libraries into your Python application.

```python
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import math
import base64
import requests
```


## Downloading and Importing the Vacation Preference Dataset From Kaggle

We will use the [mountain vs. beaches dataset from Kaggle](https://www.kaggle.com/datasets/jahnavipaliwal/mountains-vs-beaches-preference?resource=download) to train our machine learning model for predicting user vacation preferences.

The following script imports the CSV file containing the dataset into our Python application and displays the dataset column types and the first five rows.

```Python
# Dataset download link
# https://www.kaggle.com/datasets/jahnavipaliwal/mountains-vs-beaches-preference

dataset = pd.read_csv("/content/mountains_vs_beaches_preferences.csv")
print(dataset.dtypes)
dataset.head()
```


**Output:**
<img src="images\img1-user-vacation-preference-dataset.png">

The above output shows the dataset header. The `preferences` column represents each user's vacation preference. A value of `1` indicates a preference for mountains, while a `0` indicates a preference for beaches.

Let's plot a bar plot depicting the distribution of user preferences.

```Python
dataset['Preference'].value_counts().plot(kind='bar',
                                          figsize=(10, 6),
                                          title='Vacation Preference Distribution')

plt.xlabel('Preference')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate labels for better readability
plt.show()
```

**Output:**

<img src="images\img2-vacation-preference-distribution.png">

The above output shows that the database is imbalanced, and most users prefer beaches for vacation.

## Inserting Vacation Preference Dataset into GridDB Cloud

The next step is to insert the dataset into the GridDB cloud database.
To do so, you first need to [sign up for a GridDB cloud account](https://www.global.toshiba/ww/products-solutions/ai-iot/griddb/product/griddb-cloud.html).

Once you sign up, the next step is to whitelist your IP address, add users who can access the GridDB cloud database, and test your connection.
To complete the above steps, check out the  [GridDB cloud quick start guide](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/).

### Establishing a Connection with GridDB GridDB Cloud

To test the connection with GridDB in Python, you need your GridDB username and password and your GridDB cloud URL, which you will receive once you sign up for GridDB cloud.

Next, you must encode your username and password in base64 format and add them to the request header.

Finally, you can pass the URL and the header containing authentication information to the `requests.get()` function.

In the following script, we call the `checkConnection` endpoint to ensure that the connection is established.


```Python
username = "your_griddb_cloud_user_name"
password = "your_griddb_cloud_password"

url = "your_griddb_cloud_url/checkConnection"

credentials = f"{username}:{password}"
encoded_credentials = base64.b64encode(credentials.encode()).decode()

headers = {
    'Content-Type': 'application/json',  # Added this header to specify JSON content
    'Authorization': f'Basic {encoded_credentials}',
    'User-Agent': 'PostmanRuntime/7.29.0'
}

response = requests.get(url, headers=headers)

print(response.status_code)
print(response.text)
```

**Output:**

```
200
```

A response code of `200` means that your connection is successfully established. If you receive a response code of `403` ensure that you have whitelisted your API and are using correctly encoded username and password.

### Create Container for User Vacation Preference Dataset in GridDB Cloud

GridDB cloud stores data in containers. To create a container, you need to send data in the JSON payload in the following format:

```
{
    "container_name": "conainer name",
    "container_type": "container type (TIME_SERIES or COLLECTION)",
    "rowkey": true,
    "columns": [
        {
            "name": "column1",
            "type": "STRING"
        },
        {
            "name": "column2",
            "type": "DOUBLE"
        },
        {
            "name": "column3",
            "type": "BOOL"
        }
        .....
      ]

```

We will first extract the column names from the dataset and map their data types to [GridDB data types](https://docs.griddb.net/tqlreference/type/).

```Python
# Clean column names to remove spaces or forbidden characters in the GridDB container
dataset.columns = [col.strip().replace(" ", "_") for col in dataset.columns]

# Mapping pandas dtypes to GridDB types
type_mapping = {
    'int64': "LONG",
    'float64': "DOUBLE",
    'bool': "BOOL",
    'datetime64': "TIMESTAMP",
    'object': "STRING",
    'category': "STRING"
}

# Generate the columns part of the payload dynamically
columns = []
for col, dtype in dataset.dtypes.items():
    griddb_type = type_mapping.get(str(dtype), "STRING")  # Default to STRING if unknown
    columns.append({
        "name": col,
        "type": griddb_type
    })

columns
```

**Output:**

<img src="images\img3-datatypes-griddb-mapping.png">

Next, we will pass the JSON payload containing container information to the `containers` endpoint.

```Python
url = "your_griddb_cloud_url/containers"
container_name = "vacation_db"
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
201
```

A response code of `201` means a container is successfully created.


### Adding Data to GridDB Cloud Container

Next, we will insert data to the container we created. You can use the `containers/{container_name}/rows` endpoint to insert data into containers.

The script below converts the dataset rows into JSON format and inserts the data into the GridDB container using the `requests.put()` method.

```Python
url = f"your_griddb_cloud_url/containers/{container_name}/rows"

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
Response Text: {"count":52444}
```

The above output shows a status code of `200`, meaning the data is successfully inserted.

## User Vacation Preference Prediction Using Machine Learning

In the next step , we will retrieve this data from the GridDB cloud container and train our machine learning model on the data for user vacation preference prediction.

### Retrieving Data From GridDB


You can use the `containers/{container_name}/rows` endpoint to retrieve data from a GridDB container. You can also pass the number of rows to retrieve, the offset value for the rows, the sorting technique, and the filtering condition for retrieving data in the JSON payload.

The following script retrieves data from the GridDB cloud container and stores it in a Pandas dataframe.

```Python
url = f"your_griddb_cloud_url/containers/{container_name}/rows"

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
        vacation_dataset = pd.DataFrame(rows, columns=[col for col in dataset.columns])


    except json.JSONDecodeError:
        print("Error: Failed to decode JSON response.")
else:
    print(f"Error: Failed to query data from the container. Response: {response.text}")

vacation_dataset.head()
```

**Output:**

<img src="images\img4-data-retrieved-from-griddb.png">


## Preprocessing Data for Machine Learning Models

Our dataset contains some categorical columns. Since machine learning models expect data in numerical format, we will convert them into one-hot encoded numerical columns.

The following script performs data preprocessing.

```Python
categorical_columns = vacation_dataset.select_dtypes(include=['object']).columns.tolist()

# Convert categorical columns to dummies
dataset_encoded = pd.get_dummies(vacation_dataset, columns=categorical_columns, drop_first=True)

# Display the transformed dataset
dataset_encoded.head()
```

**Output:**

<img src="images\img5-preprocessed-data-for-machine-learning.png">


## Predicting User Vacation Preference with Machine Learning

Next, we will divide the dataset into training and test sets and apply standard scaling. Scaling data features expedites the training process and improves model performance.

```Python
X = dataset_encoded.drop(columns=['Preference'])
y = dataset_encoded['Preference']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Finally, we will train the  [Random Forest Classifier model from the Sklearn libary](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) on our training set and make predictions on the test set.

```Python
# Initialize and train the RandomForestClassifier
rfc_model = RandomForestClassifier(random_state=42, n_estimators=1000)
rfc_model.fit(X_train, y_train)

# Make predictions
y_train_pred = rfc_model.predict(X_train)
y_test_pred = rfc_model.predict(X_test)

# Compute accuracy using sklearn.metrics.accuracy_score
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print accuracies
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
```

**Output:**

```
Training Accuracy: 1.0000
Test Accuracy: 0.8182
```

The above output shows that we achieved an accuracy of 81.82% on the test set. You can balance the dataset and see if you can improve the model accuracy.


The following script plots the classification report and the confusion matrix for the results.

```Python
print("Classification Report:\n")
print(classification_report(y_test, y_test_pred))

conf_matrix = confusion_matrix(y_test, y_test_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=rfc_model.classes_, yticklabels=rfc_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
```

**Output:**

<img src="images\img6-prediction-results.png">

## Conclusion

This article explained how to insert and retrieve data from the GridDB cloud and train your machine learning model for classification tasks such as user vacation preference prediction.

If you have any questions or need assistance with GridDB cloud, please ask on Stack Overflow using the `griddb tag`. Our team is always happy to help.

For the complete code, visit my [GridDB Blogs GitHub repository](https://github.com/usmanmalik57/GridDB-Blogs/tree/main/User%20Vacation%20Preference%20Prediction%20Using%20Machine%20Learning%20and%20GridDB%20Cloud).
