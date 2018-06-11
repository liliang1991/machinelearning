from pyspark.ml.feature import CountVectorizer
from pyspark.shell import spark
from pyspark.ml.feature import StandardScaler
from mlib.countvectorizer.util.countvectorizer_utils import *

df = spark.createDataFrame([
    (0, "a b c".split(" ")),
    (1, "a b b c a".split(" "))
], ["id", "words"])
# mindf 必须在文档中出现的最少次数
# vocabSize 词典大小
result = CountVectorizerModel("words", "features", 3, 2.0, df)

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(result)

# Normalize each feature to have unit standard deviation.
scaledResult = scalerModel.transform(result)
scaledResult.show(truncate=False)
