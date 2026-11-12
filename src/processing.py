from pathlib import Path

import numpy as np
import pandas as pd
import phik
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from config import get_db_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_CURRENCY_PATH = PROJECT_ROOT / "data" / "final_tickets_tenge_df.csv"

PURCHASES_QUERY = """
SELECT
    p.user_id,
    p.device_type_canonical,
    p.order_id,
    p.created_dt_msk AS order_dt,
    p.created_ts_msk AS order_ts,
    p.currency_code,
    p.revenue,
    p.tickets_count,
    p.event_id,
    e.event_name_code AS event_name,
    e.event_type_main,
    p.service_name,
    r.region_name,
    c.city_name
FROM afisha.purchases AS p
INNER JOIN afisha.events AS e
    ON p.event_id = e.event_id
INNER JOIN afisha.city AS c
    ON e.city_id = c.city_id
INNER JOIN afisha.regions AS r
    ON c.region_id = r.region_id
WHERE p.device_type_canonical IN ('mobile', 'desktop')
  AND e.event_type_main != 'фильм'
ORDER BY p.user_id, p.created_ts_msk
"""

CATEGORICAL_COLUMNS = [
    "device_type_canonical",
    "currency_code",
    "event_type_main",
    "service_name",
    "region_name",
    "city_name",
]

DUPLICATE_COLUMNS = [
    "user_id",
    "order_ts",
    "device_type_canonical",
    "service_name",
    "event_id",
    "revenue",
]


class DataProcessor:
    """Загружает данные Афиши, готовит их и рассчитывает метрики лояльности."""

    def __init__(self, env_path=DEFAULT_ENV_PATH, currency_path=DEFAULT_CURRENCY_PATH):
        self.env_path = Path(env_path)
        self.currency_path = Path(currency_path)
        self.data = None
        self.currency = None
        self.profiles = None
        self.preprocessing_stats = {}
        self.profile_filter_stats = {}

    def _create_engine(self):
        config = get_db_config(self.env_path)
        url = URL.create(
            drivername="postgresql+psycopg2",
            username=config["user"],
            password=config["pwd"],
            host=config["host"],
            port=config["port"],
            database=config["db"],
        )
        return create_engine(url)

    def load_data(self):
        """Загружает заказы из базы данных."""
        engine = self._create_engine()
        try:
            self.data = pd.read_sql_query(PURCHASES_QUERY, con=engine)
        finally:
            engine.dispose()
        return self.data

    def load_currency_rates(self):
        """Загружает курс тенге к рублю."""
        self.currency = pd.read_csv(self.currency_path, parse_dates=["data"])
        return self.currency

    def preprocess(self):
        """Очищает данные, пересчитывает выручку в рублях и убирает выбросы."""
        if self.data is None:
            raise ValueError("Сначала вызовите load_data()")
        if self.currency is None:
            self.load_currency_rates()

        df = self.data.copy()
        initial_rows = len(df)
        initial_users = df["user_id"].nunique()

        duplicate_mask = df.duplicated(subset=DUPLICATE_COLUMNS, keep="first")
        df = df.loc[~duplicate_mask].reset_index(drop=True)
        df["order_dt"] = pd.to_datetime(df["order_dt"])
        df["order_ts"] = pd.to_datetime(df["order_ts"])
        df = df.sort_values(["user_id", "order_ts"]).reset_index(drop=True)
        df["days_since_prev"] = (
            df.groupby("user_id", observed=True)["order_dt"].diff().dt.days
        )

        df = df.merge(
            self.currency[["data", "nominal", "curs"]],
            left_on="order_dt",
            right_on="data",
            how="left",
        )
        missing_rates = int(df["curs"].isna().sum())
        kzt_mask = df["currency_code"].eq("kzt")
        kzt_orders = int(kzt_mask.sum())
        kzt_mean_revenue = float(df.loc[kzt_mask, "revenue"].mean())
        df["revenue_rub"] = np.where(
            kzt_mask,
            df["revenue"] * df["curs"] / df["nominal"],
            df["revenue"],
        )
        df = df.drop(columns=["data", "nominal", "curs"])

        for column in CATEGORICAL_COLUMNS:
            df[column] = df[column].astype(str).str.strip()

        negative_count = int((df["revenue_rub"] < 0).sum())
        revenue_p99 = float(df["revenue_rub"].quantile(0.99))
        above_p99_count = int((df["revenue_rub"] > revenue_p99).sum())
        before_filter = len(df)
        df = df[
            (df["revenue_rub"] >= 0) & (df["revenue_rub"] <= revenue_p99)
        ].copy()
        removed_by_revenue = before_filter - len(df)

        df = df.sort_values(["user_id", "order_ts"]).reset_index(drop=True)
        df["days_since_prev"] = (
            df.groupby("user_id", observed=True)["order_dt"].diff().dt.days
        )
        for column in CATEGORICAL_COLUMNS:
            df[column] = df[column].astype("category")

        self.preprocessing_stats = {
            "initial_rows": initial_rows,
            "initial_users": initial_users,
            "duplicates": int(duplicate_mask.sum()),
            "missing_rates": missing_rates,
            "kzt_orders": kzt_orders,
            "kzt_mean_revenue": kzt_mean_revenue,
            "negative_count": negative_count,
            "revenue_p99": revenue_p99,
            "above_p99_count": above_p99_count,
            "removed_by_revenue": removed_by_revenue,
            "final_rows": len(df),
            "final_users": df["user_id"].nunique(),
            "mean_revenue": float(df["revenue_rub"].mean()),
            "median_revenue": float(df["revenue_rub"].median()),
            "period_start": df["order_dt"].min(),
            "period_end": df["order_dt"].max(),
        }
        self.data = df
        return self.data

    def build_profiles(self):
        """Строит профиль пользователя по очищенным заказам."""
        if self.data is None:
            raise ValueError("Сначала вызовите preprocess()")

        df = self.data.sort_values(["user_id", "order_ts"]).reset_index(drop=True)
        weekday_map = {
            0: "Пн",
            1: "Вт",
            2: "Ср",
            3: "Чт",
            4: "Пт",
            5: "Сб",
            6: "Вс",
        }
        profiles = (
            df.groupby("user_id", observed=True)
            .agg(
                first_order_date=("order_dt", "min"),
                last_order_date=("order_dt", "max"),
                first_device=("device_type_canonical", "first"),
                first_region=("region_name", "first"),
                first_event_type=("event_type_main", "first"),
                total_orders=("order_id", "nunique"),
                avg_revenue_rub=("revenue_rub", "mean"),
                avg_tickets_count=("tickets_count", "mean"),
                avg_days_between_orders=("days_since_prev", "mean"),
            )
            .reset_index()
        )
        profiles["is_two"] = (profiles["total_orders"] >= 2).astype("int8")
        profiles["first_order_weekday"] = profiles["first_order_date"].dt.dayofweek.map(
            weekday_map
        )
        self.profiles = profiles
        return self.profiles

    def filter_active_profiles(self):
        """Удаляет аномально активных пользователей выше 99-го перцентиля."""
        if self.profiles is None or self.data is None:
            raise ValueError("Сначала вызовите build_profiles()")

        p99 = float(self.profiles["total_orders"].quantile(0.99))
        p95 = float(self.profiles["total_orders"].quantile(0.95))
        heavy = self.profiles[self.profiles["total_orders"] > p99]
        heavy_share = float(self.data["user_id"].isin(heavy["user_id"]).mean())
        before_users = len(self.profiles)
        before_orders = len(self.data)

        self.profiles = self.profiles[
            self.profiles["total_orders"] <= p99
        ].reset_index(drop=True)
        self.data = self.data[
            self.data["user_id"].isin(self.profiles["user_id"])
        ].reset_index(drop=True)

        self.profile_filter_stats = {
            "orders_p99": p99,
            "orders_p95": p95,
            "heavy_users": len(heavy),
            "heavy_share": heavy_share,
            "removed_users": before_users - len(self.profiles),
            "removed_orders": before_orders - len(self.data),
        }
        return self.profiles

    @staticmethod
    def _segment_stats(data, column):
        return (
            data.groupby(column, observed=True)
            .agg(users=("user_id", "size"), returned=("is_two", "sum"))
            .assign(
                share=lambda frame: frame["users"] / frame["users"].sum(),
                return_rate=lambda frame: frame["returned"] / frame["users"],
            )
            .sort_values("users", ascending=False)
        )

    def retention_by_segments(self):
        """Считает размер сегмента и долю пользователей с 2+ заказами."""
        if self.profiles is None:
            raise ValueError("Сначала вызовите build_profiles()")
        return {
            "first_event_type": self._segment_stats(
                self.profiles, "first_event_type"
            ),
            "first_device": self._segment_stats(self.profiles, "first_device"),
            "first_region": self._segment_stats(self.profiles, "first_region"),
            "first_order_weekday": self._segment_stats(
                self.profiles, "first_order_weekday"
            ),
        }

    def check_hypotheses(self, segments=None):
        """Сравнивает возврат для спорта и концертов, топ-10 и остальных регионов."""
        if segments is None:
            segments = self.retention_by_segments()

        sport_concerts = segments["first_event_type"].reindex(["спорт", "концерты"])
        sport_rate = float(sport_concerts.loc["спорт", "return_rate"])
        concerts_rate = float(sport_concerts.loc["концерты", "return_rate"])

        top_regions = segments["first_region"].head(10).index
        region_groups = self.profiles.assign(
            region_group=np.where(
                self.profiles["first_region"].isin(top_regions),
                "Топ-10 регионов",
                "Остальные регионы",
            )
        )
        region_summary = (
            region_groups.groupby("region_group", observed=True)["is_two"]
            .agg(users="size", returned="sum")
            .assign(return_rate=lambda frame: frame["returned"] / frame["users"])
        )
        top_rate = float(region_summary.loc["Топ-10 регионов", "return_rate"])
        rest_rate = float(region_summary.loc["Остальные регионы", "return_rate"])

        return {
            "sport_concerts": sport_concerts,
            "sport_rate": sport_rate,
            "concerts_rate": concerts_rate,
            "sport_diff": sport_rate - concerts_rate,
            "region_summary": region_summary,
            "top_rate": top_rate,
            "rest_rate": rest_rate,
            "region_diff": top_rate - rest_rate,
        }

    def analyze_revenue(self):
        """Сравнивает среднюю выручку разовых и вернувшихся пользователей."""
        if self.profiles is None:
            raise ValueError("Сначала вызовите build_profiles()")
        single = self.profiles.loc[
            self.profiles["total_orders"] == 1, "avg_revenue_rub"
        ]
        repeated = self.profiles.loc[
            self.profiles["total_orders"] >= 2, "avg_revenue_rub"
        ]
        return {
            "single": single,
            "repeated": repeated,
            "single_mean": float(single.mean()),
            "repeated_mean": float(repeated.mean()),
            "single_median": float(single.median()),
            "repeated_median": float(repeated.median()),
            "single_std": float(single.std()),
            "repeated_std": float(repeated.std()),
            "single_q25": float(single.quantile(0.25)),
            "single_q75": float(single.quantile(0.75)),
            "repeated_q25": float(repeated.quantile(0.25)),
            "repeated_q75": float(repeated.quantile(0.75)),
        }

    def analyze_weekday(self, segments=None):
        """Возвращает метрики возврата по дню недели первого заказа."""
        if segments is None:
            segments = self.retention_by_segments()
        weekday = segments["first_order_weekday"].reindex(
            ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
        )
        return {
            "stats": weekday,
            "spread": float(
                weekday["return_rate"].max() - weekday["return_rate"].min()
            ),
        }

    def profile_segments(self):
        """Описывает сегменты пользователей по числу заказов."""
        if self.profiles is None:
            raise ValueError("Сначала вызовите build_profiles()")
        profiles = self.profiles.copy()
        profiles["orders_segment"] = pd.cut(
            profiles["total_orders"],
            bins=[0, 1, 2, 4, np.inf],
            labels=["1", "2", "3–4", "5+"],
        )
        return (
            profiles.groupby("orders_segment", observed=True)
            .agg(
                users=("user_id", "size"),
                orders=("total_orders", "sum"),
                avg_orders=("total_orders", "mean"),
                avg_revenue=("avg_revenue_rub", "mean"),
                avg_days=("avg_days_between_orders", "mean"),
            )
            .assign(
                user_share=lambda frame: frame["users"] / frame["users"].sum(),
                order_share=lambda frame: frame["orders"] / frame["orders"].sum(),
            )
        )

    def correlation_analysis(self):
        """Считает phi_k для профиля и его сегментов."""
        if self.profiles is None:
            raise ValueError("Сначала вызовите build_profiles()")

        corr_columns = [
            "first_device",
            "first_region",
            "first_event_type",
            "first_order_weekday",
            "total_orders",
            "avg_revenue_rub",
            "avg_tickets_count",
            "avg_days_between_orders",
        ]
        interval_columns = [
            "total_orders",
            "avg_revenue_rub",
            "avg_tickets_count",
            "avg_days_between_orders",
        ]
        data = self.profiles[corr_columns].copy()

        matrix = phik.phik_matrix(data, interval_cols=interval_columns)
        total_orders = matrix["total_orders"].drop("total_orders").sort_values(
            ascending=False
        )

        segmented = data.copy()
        segmented["orders_segment"] = pd.cut(
            segmented["total_orders"],
            bins=[0, 1, 2, 4, np.inf],
            labels=["1", "2", "3–4", "5+"],
        )
        segment_matrix = phik.phik_matrix(
            segmented[corr_columns + ["orders_segment"]],
            interval_cols=interval_columns,
        )
        segment_corr = segment_matrix["orders_segment"].drop(
            ["orders_segment", "total_orders"]
        ).sort_values(ascending=False)

        segment_results = {}
        for name, mask in [
            ("3–4 заказа", segmented["total_orders"].between(3, 4)),
            ("5+ заказов", segmented["total_orders"] >= 5),
        ]:
            segment_data = segmented.loc[mask, corr_columns]
            segment_results[name] = phik.phik_matrix(
                segment_data, interval_cols=interval_columns
            )["total_orders"].drop("total_orders")

        return {
            "matrix": matrix,
            "total_orders": total_orders,
            "segment_corr": segment_corr,
            "segment_results": pd.DataFrame(segment_results),
        }
