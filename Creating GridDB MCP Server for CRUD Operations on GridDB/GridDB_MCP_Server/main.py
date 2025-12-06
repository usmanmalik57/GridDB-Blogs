from fastmcp import FastMCP
from griddb_tools import (
    insert_csv_to_griddb,
    get_container_columns,
    sql_select_from_griddb,
    sql_insert_update_griddb,
    sql_delete_rows_griddb,
)

mcp = FastMCP(name="GridDB MCP Server")

# Register imported functions as tools
mcp.tool()(insert_csv_to_griddb)
mcp.tool()(get_container_columns)
mcp.tool()(sql_select_from_griddb)
mcp.tool()(sql_insert_update_griddb)
mcp.tool()(sql_delete_rows_griddb)

if __name__ == "__main__":
    mcp.run()