import os
import base64
import json
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

username = os.environ.get("username")
password = os.environ.get("password")
base_url = os.environ.get("base_url")

if not all([username, password, base_url]):
    raise RuntimeError("Missing env vars: username/password/base_url")

credentials = f"{username}:{password}"
encoded_credentials = base64.b64encode(credentials.encode()).decode()
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Basic {encoded_credentials}",
    "User-Agent": "PostmanRuntime/7.29.0",
}



async def insert_csv_to_griddb(csv_file_path: str) -> str:

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
    

## ============================================================ ##
## =================== Get Container Columns Tool ============= ##
## ============================================================ ##


async def get_container_columns(container_name: str) -> list[str] | str:
    
    """
    Fetches one row from the container and tries to get column names from response metadata
    Use this tool before executing any CRUD query to get the column names

    Args:
        container_name (str): Name of the container/table to retrieve data
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
            column_names = ", ".join(names)
            return column_names

    except Exception as e:
        return f"Error: {str(e)}"
    

## ============================================================ ##
## =================== Search Data Tool ======================= ##
## ============================================================ ##

async def sql_select_from_griddb(container_name: str, sql_stmt: str) -> str:
    """
    Execute a SQL SELECT query on a GridDB Cloud container via Web API,
    and return a formatted string of the results.

    If the result is an aggregate (e.g. COUNT, SUM etc.) returning a single value,
    returns something like "Aggregate result: 42".

    Otherwise, returns record by record detail.

    Args:
        container_name (str): Name of the container/table to retrieve data 
        sql_stmt (str): The full SQL SELECT statement to execute.

    Returns:
        A string with formatted results or error message.
    """
    

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
        

        return f"No usable result data found for query: {first}"

    except Exception as e:
        return f"Error: {e}"



## ============================================================ ##
## =================== Insert/Update Data Tool =============== ##
## ============================================================ ##


async def sql_insert_update_griddb(container_name: str, sql_stmt: str) -> str:
    """
    Execute an SQL INSERT or UPDATE query on GridDB Cloud via Web API,
    return a message indicating success or error.
    
    Args:
        container_name (str): Name of the container/table (used for readability/logging).
        sql_stmt (str): The full SQL INSERT or UPDATE statement to execute.
    
    Returns:
        A string: either success message or error.
    """

    
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


## ============================================================ ##
## =================== Delete Data Tool ======================= ##
## ============================================================ ## 


async def sql_delete_rows_griddb(container_name: str, row_keys: list, convert_to_int: bool = True) -> str:
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


        url = f"{base_url}/containers/{container_name}/rows"
        response = requests.delete(url, headers=headers, json=row_keys_to_use)

        # Acceptable success statuses
        if response.status_code not in (200, 204):
            return f"Error deleting rows: {response.status_code}, {response.text}"

        return f"Successfully requested delete of {len(row_keys_to_use)} row(s) from '{container_name}'."
    except Exception as e:
        return f"Error: {e}"