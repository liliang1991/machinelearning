from mlib.countvectorizer.util.CountVectorizerUtils import *
from mlib.standard.util.StandardScalerUtils import *
from mlib.stringindexer.util.StringIndexerUtils import *
df = spark.createDataFrame([
    ("交通", "专家给武汉交通支招：建设快速公交通"),
    ("交通", "随着北方气温的迅速回升，大量省内"),
    ("交通", "武汉铁路东线将提速改造 浏览次数")
], ["label", "text"])
sindex=StringIndexerModel("words","sindexout",df)
sindex.show()
# mindf 必须在文档中出现的最少次数
# vocabSize 词典大小
cvResult = CountVectorizerModel("words", "features", 3, 2.0, df)

scaledResult= StandardScalerModel("features", "scaledFeatures",
                        True, False,cvResult)
# Normalize each feature to have unit standard deviation.
scaledResult.show(truncate=False)
