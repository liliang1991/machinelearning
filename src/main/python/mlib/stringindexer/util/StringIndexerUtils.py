from pyspark.ml.feature import StringIndexer
from pyspark.shell import spark

df = spark.createDataFrame(
    [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
    ["id", "category"])
df.show()
indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
indexed = indexer.fit(df).transform(df)
indexed.show()

def StringIndexerModel(input_col, output_col,input_data):
    indexer = StringIndexer(inputCol=input_col, outputCol=output_col)
    result = indexer.fit(input_data).transform(input_data)
    return result