import os

snowflake_conn_prop = {
   "account": os.environ.get("SNOW_ACC"),
   "user": os.environ.get("SNOW_USER"),
   "password": os.environ.get("SNOW_PASS")
 
}


 


# Snowflake Account Identifiers
# https://docs.snowflake.com/en/user-guide/admin-account-identifier.html#account-identifier-formats-by-cloud-platform-and-region
#  