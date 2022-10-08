from pyspark import StorageLevel
from pyspark.sql import SparkSession

appName = "PySpark Example - MariaDB Example"
master = "local"

# Create Spark session
spark = SparkSession.builder.appName(appName).master(master).getOrCreate()

sql = "select Hit, atBat, batter, game_id from baseball.batter_counts"
database = "baseball"
user = "root"
password = "root"  # pragma: allowlist secret
server = "localhost"
port = 3306
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"

# Create a data frame by reading data from Oracle via JDBC
df = (
    spark.read.format("jdbc")
    .option("url", jdbc_url)
    .option("query", sql)
    .option("user", user)
    .option("password", password)
    .option("driver", jdbc_driver)
    .load()
)

# df.show()

sql2 = "select game_id, local_date from baseball.game"
database = "baseball"
user = "root"
password = "root"  # pragma: allowlist secret
server = "localhost"
port = 3306
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"

# Create a data frame by reading data from Oracle via JDBC
df2 = (
    spark.read.format("jdbc")
    .option("url", jdbc_url)
    .option("query", sql2)
    .option("user", user)
    .option("password", password)
    .option("driver", jdbc_driver)
    .load()
)
# df2.show()

# Historic Average

df.createOrReplaceTempView("batter_counts")
df.persist(StorageLevel.DISK_ONLY)

df2.createOrReplaceTempView("game")
df2.persist(StorageLevel.DISK_ONLY)

# Batter_Count + Game Tables

Averages_table = spark.sql(
    """
    select * from batter_counts tbl1, game tbl2
    WHERE tbl1.game_id == tbl2.game_id
    """
)
# Averages_table.show()

Averages_table.createOrReplaceTempView("Averages_Table")
Averages_table.persist(StorageLevel.DISK_ONLY)

# Seems to be running quick, let me know if you think I should add an index

Rolling_Avg = spark.sql(
    """
    SELECT a.batter, a.local_date, (SUM(b.Hit)/SUM(b.atBat))
        FROM Averages_Table a
        JOIN Averages_Table b
            ON a.batter = b.batter
        WHERE b.local_date BETWEEN a.local_date - INTERVAL 100 DAY AND a.local_date AND a.batter = 110029
        GROUP BY a.batter, a.local_date
    """
)
Rolling_Avg.show()

Rolling_Avg.createOrReplaceTempView("Rolling_Avg")
Rolling_Avg.persist(StorageLevel.DISK_ONLY)
