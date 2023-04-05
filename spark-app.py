print("########################### Spark-APP #################")

from pyspark.sql import SparkSession

spark = SparkSession.builder \
        .master("local") \
        .appName("SparkSession Name") \
        .config("spark.executor.memory", '8g') \
        .config("spark.executor.core", 2) \
        .getOrCreate()

print("############# Spark: ", spark.version)
print("############# Hadoop: ", spark.sparkContext._gateway.jvm.org.apache.hadoop.util.VersionInfo.getVersion())

data = [1,2,3,4,5,6]
distData =spark.sparkContext.parallelize(data,2)
print(type(data))
print(type(distData))
print(distData.getNumPartitions())
print(spark.sparkContext.defaultParallelism)

print(distData.count())
print(distData.sum())
print(distData.reduce(lambda a,b: a + b))
print("####### ReadFile")
lines = spark.sparkContext.textFile('/user/anshulmax1/Lec24_Book.txt')
lineLength = lines.map(lambda s: len(s))
totalLength = lineLength.reduce(lambda a,b: a+b)
print("######Total Length:", totalLength)
print(lineLength.collect()[0])
print(lineLength.take(3))

#lineLength.saveAsTextFile("/user/anshulmax1/results_book/")


def myFunc(l):
    words = l.split(" ")
    return(len(words))
wordLine =lines.map(myFunc)
print("#",wordLine.take(5))
print("## words: ", wordLine.reduce(lambda a,b: a+b))

count = lines.flatMap(lambda line: line.split(" "))\
        .map(lambda word: (word, 1))\
        .reduceByKey(lambda a, b: a + b).sortBy(lambda a: a[1],0)\
        .collect()
print("# count type: ",type(count))
print("# count: ", count[0:5])

rangeVar = spark.range(1000).toDF("number")
print("# print type: ", type(rangeVar))
print("# print types: ", rangeVar.dtypes)

print("# print type: ", rangeVar.show(5))
print("# print type: ", rangeVar.schema)

print("# ",rangeVar.schema)
rangeVar.printSchema()

even = rangeVar.where(" number % 2 = 0")
print("# count of even: ", even.count())


fd = spark.read.format("csv")\
       .option("inferSchema","true")\
       .option("header", "true")\
       .load("/user/anshulmax1/Lec24_Flight.csv")
fd.explain()
fd.printSchema()

fd.createOrReplaceTempView("fd")
qUsingSQL = spark.sql("""
        select DEST_COUNTRY_NAME, count(*) as cnt
        from fd
        group by DEST_COUNTRY_NAME
        sort by cnt desc
        """)
print("# SQL QUERY ANSWER:")
qUsingSQL.show(5)

qUsingSQL = spark.sql("""
        select * from fd  Where DEST_COUNTRY_NAME = "United States"
        """)

print("# SQL QUERY ANSWER:")
qUsingSQL.show(5)

qUsingSQL = fd.filter(fd.DEST_COUNTRY_NAME == "United States")
print("# Function Call: ")
qUsingSQL.show(5)

from  pyspark.sql import functions as F
null = fd.filter(F.col("DEST_COUNTRY_NAME").isNull())
print("# isNull: ")
qUsingSQL.show(5)

qUsingDot = fd.groupby("DEST_COUNTRY_NAME")\
        .sum("count")\
        .withColumnRenamed("sum(count)","dest_count")\
        .sort(F.desc("dest_count"))\
        .limit(5)
qUsingDot.show()

mixQuery = spark.sql("""
        Select DEST_COUNTRY_NAME, sum(count)
        from fd
        group by DEST_COUNTRY_NAME
        """)\
                .where("DEST_COUNTRY_NAME like 'So%' or DEST_COUNTRY_NAME like 'UN%'")\
                .sort(F.asc("DEST_COUNTRY_NAME"))
mixQuery.show(5)


jfk = spark.read.format("csv")\
       .option("inferSchema","true")\
       .option("header", "true")\
       .load("/user/anshulmax1/jfk_weather_cleaned.csv")

jfk.show(5)
jfk.printSchema()


from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols=["HOURLYWindDirectionSin","HOURLYWindDirectionCos", "HOURLYStationPressure"],outputCol = "features")
df_transformed = vectorAssembler.transform(jfk)
df_transformed.show(5)

from pyspark.ml.feature import Normalizer
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(labelCol="HOURLYWindSpeed", featuresCol='features_norm', maxIter=100, regParam=0.0,elasticNetParam=0.0)

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, normalizer,lr])


splits = jfk.randomSplit([0.8, 0.2])
df_train = splits[0]
df_test = splits[1]

model = pipeline.fit(df_train)
prediction = model.transform(df_test)


from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(labelCol="HOURLYWindSpeed", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(prediction)
print("RMSE on test data = %g" % rmse)






