import json
import logging
import random
import sys
import time
from collections.abc import MutableMapping


log = logging.getLogger(__name__)


def flatten_dict(dictionary, parent_key="", separator="_"):
    items = []
    for key, value in dictionary.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def prep_cfg_for_db(cfg, to_remove):
    cfg = dict(cfg)
    for key in to_remove:
        cfg.pop(key, None)
    return flatten_dict(cfg)


class ExperimentManager:
    def __init__(self, experiment_name, parameters_dict, db_prefix, db_data):
        self.db_data = dict(db_data)
        self.disabled = bool(self.db_data.get("disable", False))
        self.db_name = f"{db_prefix}_{experiment_name}"
        self.command_args = "python " + " ".join(sys.argv)
        self.run = parameters_dict["run"]

        if self.disabled:
            log.warning("Database logging disabled; running without MySQL persistence.")
            return

        conn, cursor = self.get_connection(use_database=False)
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self._quote(self.db_name)};")
        cursor.execute(f"USE {self._quote(self.db_name)};")
        conn.commit()
        cursor.close()
        conn.close()

        self.make_table("runs", parameters_dict, ["run"])
        self.insert_value("runs", parameters_dict)

    def _connect(self, use_database=True):
        try:
            import mysql.connector
        except ImportError as exc:
            raise ImportError(
                "mysql-connector-python is required for database logging."
            ) from exc

        while True:
            try:
                conn = mysql.connector.connect(
                    host=self.db_data["ip"],
                    user=self.db_data["username"],
                    password=self.db_data["password"],
                    port=self.db_data.get("port", 3306),
                )
                break
            except Exception:
                print("Database server not responding; busy waiting")
                time.sleep((random.random() + 0.2) * 5)

        cursor = conn.cursor()
        if use_database:
            cursor.execute(f"USE {self._quote(self.db_name)};")
        return conn, cursor

    def get_connection(self, use_database=True):
        if self.disabled:
            return None, None
        return self._connect(use_database=use_database)

    @staticmethod
    def _quote(identifier):
        return f"`{identifier}`"

    @staticmethod
    def _column_type(value):
        if isinstance(value, bool):
            return "BOOLEAN"
        if isinstance(value, int):
            return "BIGINT"
        if isinstance(value, float):
            return "DOUBLE"
        if isinstance(value, (dict, list, tuple)):
            return "JSON"
        return "TEXT"

    @staticmethod
    def _serialize(value):
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value)
        return value

    def make_table(self, table_name, data_dict, primary_key):
        if self.disabled:
            return False

        conn, cursor = self.get_connection()
        columns = [
            f"{self._quote(key)} {self._column_type(value)}"
            for key, value in data_dict.items()
        ]
        if primary_key:
            quoted_keys = ", ".join(self._quote(key) for key in primary_key)
            columns.append(f"PRIMARY KEY ({quoted_keys})")
        query = (
            f"CREATE TABLE IF NOT EXISTS {self._quote(table_name)} "
            f"({', '.join(columns)});"
        )
        cursor.execute(query)
        cursor.execute(f"SHOW COLUMNS FROM {self._quote(table_name)};")
        existing_columns = {row[0] for row in cursor.fetchall()}
        for key, value in data_dict.items():
            if key not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE {self._quote(table_name)} "
                    f"ADD COLUMN {self._quote(key)} {self._column_type(value)};"
                )
        conn.commit()
        cursor.close()
        conn.close()
        return True

    def insert_value(self, table_name, data_dict):
        self.insert_values(table_name, list(data_dict.keys()), [list(data_dict.values())])

    def insert_values(self, table_name, keys, value_list):
        if self.disabled or len(value_list) == 0:
            return

        conn, cursor = self.get_connection()
        quoted_keys = ", ".join(self._quote(key) for key in keys)
        placeholders = ", ".join(["%s"] * len(keys))
        query = (
            f"INSERT INTO {self._quote(table_name)} ({quoted_keys}) "
            f"VALUES ({placeholders});"
        )
        rows = [tuple(self._serialize(value) for value in row) for row in value_list]
        cursor.executemany(query, rows)
        conn.commit()
        cursor.close()
        conn.close()


class Metric:
    def __init__(self, name, key_values_dict, primary_key, experiment_manager):
        self.experiment_manager = experiment_manager
        self.keys = list(key_values_dict.keys())
        self.table_name = name
        self.experiment_manager.make_table(name, key_values_dict, primary_key)
        self.list_of_data = []

    def commit_to_database(self):
        if len(self.list_of_data) == 0:
            return
        self.experiment_manager.insert_values(
            self.table_name, self.keys, self.list_of_data
        )
        self.list_of_data = []

    def add_data(self, list_of_values):
        self.list_of_data.append(list_of_values)

    def clear_data(self):
        self.list_of_data = []
