from pyspark.ml.feature import CountVectorizer
from pyspark.shell import spark
from mlib.countvectorizer.util.countvectorizer_utils import *

df = spark.createDataFrame([
    (0, "a b c".split(" ")),
    (1, "a b b c a".split(" "))
], ["id", "words"])
# mindf 必须在文档中出现的最少次数
# vocabSize 词典大小
result = CountVectorizerModel("words", "features", 3, 2.0, df)
result.show(truncate=False)
