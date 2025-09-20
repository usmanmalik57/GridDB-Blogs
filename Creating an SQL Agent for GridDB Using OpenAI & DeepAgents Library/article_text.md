

Large Language Models (LLMs) can act as natural language interfaces to databases, making it possible to query, insert, update, and delete records without writing raw SQL. Combining LLM reasoning with GridDB’s high-performance cloud database allows users to turn plain English questions into reliable database operations.

In this article, you will see how to build an SQL agent for GridDB using LangGraph Deep Agents and the OpenAI GPT-4o model. We will create a GridDB SQL agent that can automatically generate and execute data insertion and CRUD queries from natural language prompts.

GridDB’s ability to handle structured data efficiently, combined with the flexibility of Deep Agents, provides a powerful way to interact with databases conversationally while maintaining accuracy and control.

**Prerequisites:**

You will need the following to run scripts in this article:

* A [GridDB cloud account](https://www.global.toshiba/ww/products-solutions/ai-iot/griddb/product/griddb-cloud.html). Refer to this [quick start guide](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) for a quick start.
* [OpenAI API Key](https://platform.openai.com/api-keys). You can use any other LLM provider, but it will require slight modifications in the script.

**Note:** You can find the complete code for this tutorial in my [GridDB Blogs GitHub repository](https://github.com/usmanmalik57/GridDB-Blogs/tree/main/Creating%20an%20SQL%20Agent%20for%20GridDB%20Using%20OpenAI%20%26%20DeepAgents%20Library).

## Importing and Installing Required Libraries

The following installs the libraries you will need to run the scripts in this article.

```
%pip install -qU deepagents
%pip install -qU langgraph
%pip install -qU langchain-openai
%pip install -qU pandas
```

The script below imports the required libraries.

```python
import os
import base64
import requests
import pandas as pd
import json
from pathlib import Path

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

```

## Establishing a Connection with GridDB

Run the following script to test your connection with the GridDB cloud.

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

print(response.text)
```

**Output:**
```
200
```

If you do not see the above response, check this [quick start guide](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/) for troubleshooting.


## Creating SQL Agent with OpenAI and Deep Agents

Before we jump into creating an SQL agent for GridDB, let's first see the basic working of the LangGraph Deep Agent library.

### Simple Example of DeepAgent

[Deep Agents](https://github.com/langchain-ai/deepagents) in LangGraph are based on [ReAct agents](https://github.com/langchain-ai/react-agent). You can provide them with tools, instructions to follow, and an LLM that reasons which tool to call based on the system instructions and the user query.

Tools are nothing but Python functions that an agent can call. The agent decides the values that are passed to these function parameters.

The following script defines a simple tool `search_db`, the instructions that the agent must follow, and an LLM (OpenAI gpt-4o in this case). The instructions outline the agent's roles and provide guidance on accessing a specific tool.

Based on the instructions, the tool description, and the user query, the agent selects one or multiple tools that it must call to generate the best possible response for the user.


In this case, the  `search_db` tool returns the query passed to it.

```python
def search_db(query: str) -> str:
    """
    This tool searches the GridDB database using the provided SQL query.
    """

    print("search_db query=", query)

system_instructions = """

<role>
You are an SQL expert and you have access to a GridDB database. You can use the tool below to search the database using SQL queries.
When you receive a user query, you should formulate an appropriate SQL query to retrieve the relevant information from the database.
</role>

<tools>
** search_db**:
- This tool allows you to search the GridDB database using SQL queries.
- Input to this tool should be a valid SQL query string.
- The output of this tool will be the results of the SQL query.
</tools>
"""

llm = ChatOpenAI(model="gpt-4o",
                 temperature=0)

agent = create_deep_agent(
    tools=[search_db],
    instructions=system_instructions,
    model=llm
)

# Invoke the agent
result = agent.invoke({"messages": [{"role": "user",
                                     "content": "Select the patients who spent more than 5 days in the hospital."}]})
```

**Output:**

```
search_db query= SELECT * FROM patients WHERE days_in_hospital > 5;
```

In the output, you can see the SQL query that the agent generated in response to the natural language query from the user. You can update the `search_db` tool to execute this query on any SQL database.

Now that you know how deep agents work, let's define tools that execute CRUD (Create, Read, Update, Delete) operations on a GridDB database.

### Defining tools for GridDB SQL Agent

We will define five tools in this section:

1. `insert_csv_to_griddb`: which inserts data from a CSV file into a GridDB container.
2. `get_container_columns`: which returns the names of all the columns in a GridDB container.
3. `sql_select_from_griddb`: executes select (read) queries on a GridDB database.
4. `sql_insert_update_griddb`: executes the insert and update (create and update) queries on GridDB.
5. `sql_delete_rows_griddb`: which deletes a single or multiple rows from a GridDB database.

These tools are basically the functions based on the [official documentation for interacting with GridDB cloud](https://griddb.net/en/blog/griddb-cloud-quick-start-guide/).

#### Tools for Inserting CSV file into GridDB

This tool accepts a CSV file path and creates a container with the same name as the CSV file. It then inserts the records from the CSV file into the GridDB container.


```python
def insert_csv_to_griddb(csv_file_path: str) -> str:

    """
    This tool inserts data from a CSV file into the GridDB database.

    Args:
        csv_file_path (str): The path to the CSV file to be inserted.
    """

    try:

        if not os.path.isfile(csv_file_path):
            return f"Error: The file '{csv_file_path}' does not exist."

        dataset = pd.read_csv(csv_file_path)

        container_name = Path(csv_file_path).stem

        ## =============================
        ## Creating Container for GridDB
        ## =============================

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

        print("Create container response:", response.status_code, response.text)
        if response.status_code != 201:
            return f"Error creating container: {response.text}"

        ## =============================
        ## Inserting data in the container
        ## =============================


        url = f"{base_url}/containers/{container_name}/rows"

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

        if response.status_code != 200:
            return f"Error inserting data: {response.text}"                 


        return f"Data inserted successfully in the container {container_name}"


    except Exception as e:
        return f"Error: {str(e)}"
```

#### Tool for Retrieving Container Columns from GridDB

This tool accepts a container name and returns all the columns in the container. This tool will be called before every other SQL request to retrieve the schema of the GridDB container.

```python
def get_container_columns(container_name: str) -> list[str] | str:

    """
    Fetches one row from the container and tries to get column names from response metadata
    Use this tool before executing any CRUD query to get the column names
    """
    try:
        if not container_name:
            return "Error: container_name must be provided"

        url = f"{base_url}/containers/{container_name}/rows"

        payload = {
            "offset": 0,
            "limit": 1,
            "condition": "",
            "sort": ""
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            return f"Error fetching rows for container {container_name}: {response.status_code} {response.text}"

        data = response.json()

        rows = data.get("rows", None)
        if rows is None or len(rows) == 0:
            return f"No rows returned from container {container_name}"

        # Try to get “columns” metadata from response, if present
        if "columns" in data and isinstance(data["columns"], list):
            cols_meta = data["columns"]
            # cols_meta: list of dicts with at least "name"
            names = [col_meta.get("name", "") for col_meta in cols_meta]
            return names

    except Exception as e:
        return f"Error: {str(e)}"

```

#### Tool for Retrieving Data from GridDB

This tool accepts a container_name and the SQL query as parameters and returns the corresponding information from the GridDB.

You can view the tool description, which explains the capabilities of this tool. The deep agent takes this description into account during the execution of the tool.

```python

def sql_select_from_griddb(container_name: str, sql_stmt: str) -> str:
    """
    Execute a SQL SELECT query on a GridDB Cloud container via Web API,
    and return a formatted string of the results.

    If the result is an aggregate (e.g. COUNT, SUM etc.) returning a single value,
    returns something like "Aggregate result: 42".

    Otherwise, returns record by record detail.
    """
    print("sql_select_from_griddb query =", sql_stmt)

    try:
        if not container_name:
            return "Error: container_name must be provided"
        if not sql_stmt.strip():
            return "Error: SQL statement must be provided"

        url = f"{base_url}/sql"
        payload = [
            {
                "type": "sql-select",
                "stmt": sql_stmt
            }
        ]
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            return f"Error executing SQL: {response.status_code}, {response.text}"

        resp_json = response.json()
        if not isinstance(resp_json, list) or len(resp_json) < 1:
            return f"Unexpected response format: {resp_json}"

        first = resp_json[0]
        columns_meta = first.get("columns")
        results = first.get("results")

        # Handle case of aggregate queries (when there are columns but rows are empty OR rows missing)
        # Or when results is present but format is such that we have just one value
        if columns_meta is not None and results is not None:
            # If only one column and one row, could be an aggregate
            if len(results) == 1 and len(columns_meta) == 1:
                # e.g. [{"name":"count", "type":"LONG"}] and [[42]]
                col_name = columns_meta[0].get("name", "value")
                val = results[0][0]
                return f"{col_name}: {val}"

            # Otherwise, return record by record
            output_lines = []
            for idx, row in enumerate(results, start=1):
                output_lines.append(f"=============================")
                output_lines.append(f"Record {idx}")
                for col_meta, cell in zip(columns_meta, row):
                    col_name = col_meta.get("name", "UnknownColumn")
                    cell_str = "None" if cell is None else str(cell)
                    output_lines.append(f"{col_name}: {cell_str}")
                output_lines.append("")  # blank line between records
            output = "\n".join(output_lines)
            return output

        # If no "results", but maybe "columns" and maybe some other field like "rows" or "rows/records"
        # Fallback: inspect other fields
        # For example, if the response has a field "rows" (old format) or "values"
        # you can try to fetch those.

        # If still nothing meaningful:
        return f"No usable result data found for query: {first}"

    except Exception as e:
        return f"Error: {e}"
```

#### Tool for Inserting and Updating Data in GridDB

Like the `sql_select_from_griddb` tool, this tool accepts a container name and SQL and performs insert and update operations on the database.

```python
def sql_insert_update_griddb(container_name: str, sql_stmt: str) -> str:
    """
    Execute an SQL INSERT or UPDATE query on GridDB Cloud via Web API,
    return a message indicating success or error.

    Args:
        container_name (str): Name of the container/table (used for readability/logging).
        sql_stmt (str): The full SQL INSERT or UPDATE statement to execute.

    Returns:
        A string: either success message or error.
    """
    print("sql_update_griddb query =", sql_stmt)

    try:
        if not container_name:
            return "Error: container_name must be provided"
        if not sql_stmt.strip():
            return "Error: SQL statement must be provided"

        # Construct the URL for SQL update (inserts or updates)
        url = f"{base_url}/sql/update"

        payload = [
            {
                "stmt": sql_stmt
            }
        ]

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            return f"Error executing SQL update: {response.status_code}, {response.text}"

        # The response may or may not include useful JSON; check and return appropriate message
        try:
            resp_json = response.json()
        except ValueError:
            # Not JSON; maybe empty or plaintext
            return f"SQL update executed successfully for container '{container_name}'."

        # If JSON and maybe return shows how many rows affected or similar
        # Depending on what the API returns; adapt as needed
        if isinstance(resp_json, list) and len(resp_json) > 0:
            # Some APIs give back something like [{"count": N}] or status info
            # Try to find a count or status field
            first = resp_json[0]
            if "count" in first:
                return f"Successfully updated/inserted {first['count']} rows into '{container_name}'."
            else:
                # If no 'count', just return the JSON for debugging
                return f"SQL update successful. Response: {resp_json}"
        else:
            return f"SQL update successful into container '{container_name}'."

    except Exception as e:
        return f"Error: {e}"

```

#### Tool for Deleting Data From GridDB

Finally, the `sql_delete_rows_griddb` tool deletes the rows from a container based on the row keys passed to it.

Notice that in the description, we mention that to select rowkeys, the agent must first call the `sql_select_from_griddb` tool.

```python
def sql_delete_rows_griddb(container_name: str, row_keys: list, convert_to_int: bool = True) -> str:
    """
    Delete one or more rows from a GridDB container via Web API.

    Args:
        container_name (str): the container to delete from.
        row_keys (list): list of rowkey values (as str or int). This can be retrieved by first calling the selectsql_select_from_griddb tool to get the rowkeys.
        convert_to_int (bool): if True, try converting keys to ints, else keep as given.

    Returns:
        A message string: success or detailed error.
    """
    try:
        if not container_name:
            return "Error: container_name must be provided"
        if not row_keys or not isinstance(row_keys, list):
            return "Error: row_keys must be a non-empty list"

        # Convert types if needed
        if convert_to_int:
            # only convert those that are numeric strings
            new_keys = []
            for k in row_keys:
                try:
                    ki = int(k)
                    new_keys.append(ki)
                except Exception:
                    # If conversion fails, keep original
                    new_keys.append(k)
            row_keys_to_use = new_keys
        else:
            row_keys_to_use = row_keys

        print("sql_delete_rows_griddb using row_keys =", row_keys_to_use)

        url = f"{base_url}/containers/{container_name}/rows"
        response = requests.delete(url, headers=headers, json=row_keys_to_use)

        print("DELETE response status:", response.status_code)
        print("DELETE response text:", response.text)

        # Acceptable success statuses
        if response.status_code not in (200, 204):
            return f"Error deleting rows: {response.status_code}, {response.text}"

        return f"Successfully requested delete of {len(row_keys_to_use)} row(s) from '{container_name}'."
    except Exception as e:
        return f"Error: {e}"
```

We have defined the tools; the next step is to define system instructions that steer the overall behavior of the agent.

### Create a Deep Agent for Tool Calling

You can write system instructions in whatever way you want. However, I personally divide system instructions into three parts:

1. The role of the agent.
2. The tools available to the agent.
3. The constraints that the agent must adhere to.

The following script defines system instructions for our agent and creates the agent.

```python

system_instructions = """

<role>
You are an SQL expert and you have access to a GridDB database. You can use the tool below to search the database using SQL queries.
When you receive a user query, you should formulate an appropriate SQL query to retrieve the relevant information from the database.
</role>

<tools>
** insert_csv_to_griddb**:
- This tool allows you to insert data from a CSV file into the GridDB database.
- Input to this tool should be the path to a valid CSV file or the name of the CSV file. The tool will add the .csv extension if not provided.
- The tool returns a success message or an error message.

** get_container_columns**:
- This tool fetches the name of all columns in the specified container.
- Input to this tool should be the name of the container (table).
- The tool returns a list of column names or an error message.

** sql_select_from_griddb**:
- This tool allows you to search the GridDB database using SQL queries.
- Input to this tool should be a valid SQL query string.
- The output of this tool will be the results of the SQL query.

** sql_insert_update_griddb**:
- This tool allows you to insert or update data in the GridDB database using SQL queries.
- Input to this tool should be a valid SQL INSERT or UPDATE query string.
- The output of this tool will be a success message or an error message.

** sql_delete_rows_griddb**:
- This tool allows you to delete one or more rows from a GridDB container using their rowkey values.
- Input to this tool should be the name of the container (table) and a list of rowkey values identifying the rows to delete.
- Always retrieve the rowkeys by first calling the `sql_select_from_griddb` tool to get the rowkeys.
- The output of this tool will be a success message or an error message.

</tools>


<constraints>
- When using the insert_csv_to_griddb tool, ensure that the CSV file exists in the current working directory or provide the full path to the file.
- Always use the get_container_columns tool to retrieve column names before constructing SQL queries to avoid errors.
- Ensure that SQL queries are syntactically correct and reference existing containers and columns in the database.
- Before using the sql_delete_rows_griddb tool, always retrieve the rowkeys by first calling the sql_select_from_griddb tool to get the rowkeys.
</constraints>

"""


llm = ChatOpenAI(model="gpt-4o",
                 temperature=0)

agent = create_deep_agent(
    tools=[insert_csv_to_griddb,
           get_container_columns,
           sql_select_from_griddb,
           sql_insert_update_griddb,
           sql_delete_rows_griddb],
    instructions=system_instructions,
    model=llm
)


```


### Testing the GridDB SQL Agent

Let's test the agent we just created. We will define a helper function that invokes the agent and returns the agent's final response.

```python
def get_response(query: str) -> str:
    # Invoke the agent
    result = agent.invoke({"messages": [{"role": "user",
                                         "content": query}]})
    return result['messages'][-1].content
```

Let's first ask the agent to insert data from a CSV file into the GridDB. Remember that you must have the CSV file in the same directory as the directory where you will run this script. For the examples in this article, I downloaded the [Titanic dataset from GitHub](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv) and renamed it `titanic_dataset.csv`.

```python
response = get_response("Insert the titanic_dataset.csv file into the database")
print(response)
```
**Output:**

```
Create container response: 201
The `titanic_dataset.csv` file has been successfully inserted into the database. If you need any further assistance or queries regarding this dataset, feel free to ask!
```

The above output confirms that the data has been inserted.

Next, we ask the agent to give us all the columns in the dataset.

```python
response = get_response("What are the column names in the titanic_dataset table?")
print(response)
```
**Output:**

```
The column names in the `titanic_dataset` table are:

1. SerialNo
2. PassengerId
3. Survived
4. Pclass
5. Name
6. Sex
7. Age
8. SibSp
9. Parch
10. Ticket
11. Fare
12. Cabin
13. Embarked
```

Next, we ask a question that the model converts into the SELECT query.

```python
response = get_response("Get the names and ages of the top 3 oldest passengers in the titanic_dataset")
print(response)
```
**Output:**

```
sql_select_from_griddb query = SELECT Name, Age FROM titanic_dataset WHERE Age IS NOT NULL ORDER BY Age DESC LIMIT 3
The names and ages of the top 3 oldest passengers in the Titanic dataset are:

1. **Barkworth, Mr. Algernon Henry Wilson** - Age: 80.0
2. **Svensson, Mr. Johan** - Age: 74.0
3. **Artagaveytia, Mr. Ramon** - Age: 71.0
```

The above output shows the query that the agent generated and the response from the agent. The query is added only for debugging purposes to demonstrate that the agent calls the `sql_select_from_griddb` tool behind the scenes. In production scenarios, you will only use the final output.

Let's ask another question.

```python
response = get_response("What is the percentage of passengers who survived in the titanic_dataset?")
print(response)

```
**Output:**

```
The percentage of passengers who survived in the Titanic dataset is approximately 38.38%.
```

We can verify the above results directly from the CSV file as follows.

```python

df = pd.read_csv("titanic_dataset.csv")

# Drop any rows where "Survived" is missing (if applicable)
df2 = df.dropna(subset=["Survived"])

total = len(df2)
survived = df2["Survived"].sum()  # assuming Survived is 1 for survived, 0 for not

percentage_survived = survived / total * 100

print(f"Total passengers: {total}")
print(f"Number who survived: {survived}")
print(f"Percentage who survived: {percentage_survived:.2f}%")
```
**Output:**

```
Total passengers: 891
Number who survived: 342
Percentage who survived: 38.38%

```

You can see that the response from the Pandas Dataframe matches the agent's response.

Next, we will test the INSERT query. We will ask the agent to add a new passenger.

```python
response = get_response("""
                        Insert a new passenger in the titanic_dataset.
                        He is a 89 years old male named "John Doe",
                        with passenger class 3, embarked from Southampton,
                        having 0 siblings/spouses aboard, 0 parents/children aboard,
                        ticket number "A123", fare 7.25, cabin as NULL
                        """
)
print(response)

```
**Output:**

```
The new passenger "John Doe" has been successfully added to the `titanic_dataset` with a `SerialNo` of 892. If you need any further assistance, feel free to ask!
```

Let's see if the new passenger is inserted..

```python
response = get_response("Who is the oldest passenger in the titanic_dataset?")
print(response)
```
**Output:**

```
The oldest passenger in the Titanic dataset is John Doe, who was 89 years old.
```

You can see that the oldest passenger is the one that we added via the insert query.

Next, we will update the age of the oldest passenger.

```python
response = get_response("Update the age of the oldest passenger in the titanic_dataset to 95")
print(response)
```
**Output:**

```
The age of the oldest passenger in the titanic_dataset has been successfully updated to 95. If you have any more requests or need further assistance, feel free to ask!
```

The output shows that the agent was intelligent enough to first retrieve the age of the oldest passenger via the `sql_select_from_griddb` tool and then update the record via the `sql_insert_update_griddb` tool.

You can again check if the record is updated via the following question.

```python
response = get_response("Who is the oldest passenger in the titanic_dataset?")
print(response)
```
**Output:**

```
The oldest passenger in the Titanic dataset is John Doe, a 95-year-old male. He was in the third class, with a ticket number A123, and embarked from Southampton (Embarked: S).
```

Finally, we try to delete all the passengers younger than 30 years.

```python
response = get_response("Delete all the passengers younger than 30 years in the titanic_dataset")
print(response)
```
**Output:**

```
sql_select_from_griddb query = SELECT SerialNo FROM titanic_dataset WHERE Age < 30
sql_delete_rows_griddb using row_keys = [1, 3, 8, 9, 10, 11, 13, 15, 17, 23, 24, 25, 28, 35, 38, 39, 40, 42, 44, 45, ...]
All passengers younger than 30 years have been successfully deleted from the `titanic_dataset`. If you have any more requests or need further assistance, feel free to ask!
```

The output shows that the agent first retrieved the `SerialNo` of all the passengers younger than 30 years old using the `sql_select_from_griddb` tool and then deleted these passengers using the `sql_delete_rows_griddb` tool.

You can verify if the passengers are deleted using the following query.


```python
response = get_response("what is the total number of passengers in the titanic_dataset?")
print(response)
```
**Output:**
```
The total number of passengers in the titanic_dataset is 507.
```

We now have 507 passengers compared to 891 that we previously had in the database.


## Conclusion

This article explains how to use the LangGraph Deep Agent library and the OpenAI API to create an SQL agent for GridDB. The agent allows users to perform complex CRUD operations on GridDB using natural language queries. Try this tool and enhance its functionality by adding functionalities to insert other data types in addition to CSV. You can also add a tool that retrieves the exact container name, even if a user makes spelling mistakes while defining container names. The possibilities are endless with AI.

You can find the complete code for this blog on the GridDB Blogs GitHub repository. If you have any questions or queries related to GridDB, create a post on Stack Overflow with the `griddb` tag. Our engineers are more than happy to respond.

For the complete code and additional examples, visit [GridDB Blogs](https://github.com/usmanmalik57/GridDB-Blogs/tree/main/Creating%20an%20SQL%20Agent%20for%20GridDB%20Using%20OpenAI%20%26%20DeepAgents%20Library) GitHub repository.
