from pyspark.ml.feature import CountVectorizer
from pyspark.shell import spark

df = spark.createDataFrame([
    (0, "a b c".split(" ")),
    (1, "a b b c a".split(" "))
], ["id", "words"])
#mindf 必须在文档中出现的最少次数
# vocabSize 词典大小
cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=3, minDF=2.0)
model = cv.fit(df)
result = model.transform(df)
result.show(truncate=False)