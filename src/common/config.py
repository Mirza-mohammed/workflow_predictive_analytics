import os
from urllib.parse import quote_plus


def get_db_url() -> str:
    db_user = os.getenv("DB_USER", "root")
    db_password = quote_plus(os.getenv("DB_PASSWORD", ""))
    db_host = os.getenv("DB_HOST", "localhost")
    db_name = os.getenv("DB_NAME", "workflow_db")

    return f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"