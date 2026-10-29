"""
PySpark workflow: data validation, cleaning and aggregation of the Bangalore traffic CSV.

What this really does: it runs Spark in LOCAL mode (master="local[*]", i.e. threads on this one computer) on the
8,936-row CSV. The code is standard Spark DataFrame code and would scale to a cluster / larger files, but nothing here
is distributed across machines and there is no streaming. The dataset is daily, so there is no hour-of-day
("rush hour") information; the time analysis is by weekday, weekend vs weekday and month instead.

Run from the project root (needs Java):   python -m src.bigdata.spark_analysis
Outputs: reports/spark/*.csv  (small result tables, used by the dashboard) and reports/spark/summary.json
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

import json
import os

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import pandas as pd

try:  # PySpark is optional (requirements-bigdata.txt); a missing install is reported clearly by check_prerequisites()
    from pyspark.sql import SparkSession, functions as F
    from pyspark.sql.types import (StructType, StructField, StringType, DoubleType, IntegerType, DateType)
    _PYSPARK_IMPORT_ERROR = None
except ImportError as _e:  # pragma: no cover - depends on the environment
    _PYSPARK_IMPORT_ERROR = _e

DATA_PATH = root_dir / "data" / "raw" / "Banglore_traffic_Dataset.csv"
OUT_DIR = root_dir / "reports" / "spark"

class SparkSetupError(RuntimeError):
    """Spark cannot run on this machine (PySpark or Java missing); the message says how to fix it."""


def build_schema():
    return StructType([
        StructField("Date", DateType(), True),
        StructField("Area Name", StringType(), True),
        StructField("Road/Intersection Name", StringType(), True),
        StructField("Traffic Volume", DoubleType(), True),
        StructField("Average Speed", DoubleType(), True),
        StructField("Travel Time Index", DoubleType(), True),
        StructField("Congestion Level", DoubleType(), True),
        StructField("Road Capacity Utilization", DoubleType(), True),
        StructField("Incident Reports", IntegerType(), True),
        StructField("Environmental Impact", DoubleType(), True),
        StructField("Public Transport Usage", DoubleType(), True),
        StructField("Traffic Signal Compliance", DoubleType(), True),
        StructField("Parking Usage", DoubleType(), True),
        StructField("Pedestrian and Cyclist Count", DoubleType(), True),
        StructField("Weather Conditions", StringType(), True),
        StructField("Roadwork and Construction Activity", StringType(), True),
    ])


def check_prerequisites(data_path=DATA_PATH):
    """Raise SparkSetupError with an actionable message instead of a cryptic Py4J traceback."""
    if _PYSPARK_IMPORT_ERROR is not None:
        raise SparkSetupError("PySpark is not installed in this environment. It is optional (the dashboard does not need it). "
                              "Install it with:  pip install -r requirements-bigdata.txt")
    from src.preflight import _java_major
    exe, major = _java_major()
    if exe is None:
        raise SparkSetupError("Java was not found. PySpark needs Java 17 or newer (tested with Java 25). Install a JDK, for example "
                              "Temurin from https://adoptium.net, then open a NEW terminal and check with `java -version`. "
                              "Alternatively set JAVA_HOME. The dashboard itself does not need Java; it only reads the saved results in reports/spark/.")
    if major is not None and major < 17:
        raise SparkSetupError(f"Java {major} was found at {exe}, but PySpark 4 needs Java 17 or newer.")
    if not Path(data_path).is_file():
        raise SparkSetupError(f"Dataset not found: {data_path}. Place Banglore_traffic_Dataset.csv in data/raw/.")


def get_spark():
    return (SparkSession.builder.master("local[*]").appName("TrafficAnalysis")
            .config("spark.ui.enabled", "false").config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.shuffle.partitions", "4").getOrCreate())


def run(data_path=DATA_PATH, out_dir=OUT_DIR):
    check_prerequisites(data_path)
    try:
        spark = get_spark()
    except Exception as e:  # Py4J / Java start-up failures are unreadable, so translate them
        if "JAVA_GATEWAY_EXITED" in str(e) or "Java gateway" in str(e):
            raise SparkSetupError("Spark could not start its Java process (JAVA_GATEWAY_EXITED). Check that `java -version` works in this "
                                  "terminal and that Java is 17 or newer; then try again.") from e
        raise
    spark.sparkContext.setLogLevel("ERROR")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PERMISSIVE would silently null bad values, so FAILFAST: a malformed row stops the job.
    raw = (spark.read.option("header", True).option("mode", "FAILFAST").schema(build_schema()).csv(str(data_path)))

    # ---- validation / cleaning (nothing is imputed: invalid rows are counted and removed) ----
    n_raw = raw.count()
    key_cols = ["Date", "Road/Intersection Name", "Congestion Level"]
    n_null_keys = raw.filter(" OR ".join(f"`{c}` IS NULL" for c in key_cols)).count()
    clean = raw.dropna(subset=key_cols)
    n_dups = clean.count() - clean.dropDuplicates(["Date", "Road/Intersection Name"]).count()
    clean = clean.dropDuplicates(["Date", "Road/Intersection Name"])
    n_out_of_range = clean.filter((F.col("Congestion Level") < 0) | (F.col("Congestion Level") > 100)).count()
    clean = clean.filter(F.col("Congestion Level").between(0, 100))

    df = (clean
          .withColumn("day_of_week", F.dayofweek("Date"))  # Spark: 1 = Sunday ... 7 = Saturday
          .withColumn("is_weekend", F.col("day_of_week").isin(1, 7))
          .withColumn("month", F.month("Date")).withColumn("year", F.year("Date"))
          .withColumn("is_100", (F.col("Congestion Level") >= 100).cast("int"))
          .cache())

    def agg(group_cols):
        result = (df.groupBy(*group_cols).agg(
            F.count("*").alias("n_obs"),
            F.round(F.avg("Congestion Level"), 3).alias("mean_congestion"),
            F.round(F.expr("percentile_approx(`Congestion Level`, 0.9)"), 3).alias("p90_congestion"),
            F.round(F.avg("is_100") * 100, 2).alias("pct_at_100"),
            F.round(F.avg("Average Speed"), 3).alias("mean_speed"),
            F.round(F.avg("Traffic Volume"), 1).alias("mean_volume"),
            F.round(F.avg("Road Capacity Utilization"), 3).alias("mean_capacity_util"),
            F.round(F.avg("Incident Reports"), 3).alias("mean_incidents"),
        ).orderBy(*group_cols))
        # collect() instead of toPandas(): Spark 4.2's toPandas needs pandas>=2.2; the tables are tiny anyway
        return pd.DataFrame([r.asDict() for r in result.collect()])

    tables = {
        "by_road": agg(["Area Name", "Road/Intersection Name"]),
        "by_area": agg(["Area Name"]),
        "by_weekday": agg(["day_of_week"]),
        "by_weekend": agg(["is_weekend"]),
        "by_month": agg(["year", "month"]),
        "by_weather": agg(["Weather Conditions"]),
    }
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)  # pandas writes the (tiny) result tables

    bounds = df.agg(F.min("Date").alias("d0"), F.max("Date").alias("d1")).first()
    summary = {
        "engine": f"PySpark {spark.version}, local[*] on a single machine (not a cluster, not streaming)",
        "input": str(Path(data_path).name),
        "rows_read": n_raw, "rows_null_keys": n_null_keys, "rows_duplicate_date_road": n_dups,
        "rows_out_of_range_congestion": n_out_of_range, "rows_after_cleaning": df.count(),
        "date_min": str(bounds["d0"]), "date_max": str(bounds["d1"]),
        "n_roads": df.select("Road/Intersection Name").distinct().count(),
        "has_hour_of_day": False,
        "note": "Dataset is daily: weekday/month analysis is possible, hour-of-day rush-hour analysis is not.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    spark.stop()
    return summary, tables


if __name__ == "__main__":
    try:
        s, _ = run()
    except SparkSetupError as err:
        print(f"\nCannot run the Spark analysis: {err}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(s, indent=2))
    print(f"Wrote tables to {OUT_DIR}")
