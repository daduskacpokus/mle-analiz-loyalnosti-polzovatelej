import os
from dotenv import load_dotenv


def get_db_config(env_path=None):
    """Возвращает параметры подключения из .env."""
    load_dotenv(dotenv_path=env_path)
    config = {
        "user": os.getenv("DB_USER"),
        "pwd": os.getenv("DB_PWD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "db": os.getenv("DB_NAME"),
    }
    missing = [name for name, value in config.items() if not value]
    if missing:
        raise ValueError(f"Не заданы параметры БД: {', '.join(missing)}")
    config["port"] = int(config["port"])
    return config
