from pyspark.sql import SparkSession
import sys
import os

_spark = None

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
def get_spark():
    global _spark
    if _spark is None:
        _spark = (SparkSession.builder
            .appName("PlayerValidator")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.pyspark.python", sys.executable)
            .config("spark.pyspark.driver.python", sys.executable)
            .config("spark.python.worker.faulthandler.enabled", "true")
            .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
            .config("spark.executor.memory", "2g")
            .config("spark.driver.memory", "4g")
            .config("spark.master", "local[2]")
            .getOrCreate())
    return _spark

spark = get_spark()

def graceful_shutdown(sig, frame):
    spark.stop()
    raise SystemExit(0)

