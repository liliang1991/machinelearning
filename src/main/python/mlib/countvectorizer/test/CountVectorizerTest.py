from mlib.countvectorizer.util.countvectorizer_utils import *
from mlib.standard.util.StandardScalerUtils import *
df = spark.createDataFrame([
    (0, "a b c".split(" ")),
    (1, "a b b c a".split(" "))
], ["id", "words"])
# mindf 必须在文档中出现的最少次数
# vocabSize 词典大小
cvResult = CountVectorizerModel("words", "features", 3, 2.0, df)

scaledResult= StandardScalerModel("features", "scaledFeatures",
                        True, False,cvResult)
# Normalize each feature to have unit standard deviation.
scaledResult.show(truncate=False)
