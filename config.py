import os

snowflake_conn_prop = {
   "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
   "user": os.environ.get("SNOWFLAKE_USER"),
   "password": os.environ.get("SNOWFLAKE_PASSWORD")
 
}


 


# Snowflake Account Identifiers
# https://docs.snowflake.com/en/user-guide/admin-account-identifier.html#account-identifier-formats-by-cloud-platform-and-region
#  