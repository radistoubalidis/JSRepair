
from datasets import load_dataset
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
import os




def sqlite2postgres(df: pd.DataFrame, table_name: str):

    load_dotenv(dotenv_path='.env')
    db_username = os.getenv("db_username")
    db_password = os.getenv("db_password")
    db_host = os.getenv("db_host")
    db_port = os.getenv("db_port")
    db_name = os.getenv("db_name")
    connection_string = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(connection_string)
    df.to_sql(table_name, engine, if_exists='replace')