import os
from dotenv import load_dotenv


def connection_string():
    load_dotenv(dotenv_path='.env')
    db_username = os.getenv("db_username")
    db_password = os.getenv("db_password")
    db_host = os.getenv("db_host")
    db_port = os.getenv("db_port")
    db_name = os.getenv("db_name")
    return f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"



