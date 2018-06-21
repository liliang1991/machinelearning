from pyspark.ml.feature import Word2Vec
from pyspark.shell import spark
# Input data: Each row is a bag of words from a sentence or document.
documentDF = spark.createDataFrame([
    ("Hi I heard about Spark".split(" "), ),
    ("I wish Java could use case classes".split(" "), ),
    ("Logistic regression models are neat".split(" "), )
], ["text"])

def Word2VecModel(input_col, output_col, vocab_size, minCount, input_data):
    # mindf 必须在文档中出现的最少次数
    # vocabSize 词典大小
    wv = Word2Vec(inputCol=input_col, outputCol=output_col, vocabSize=vocab_size, minDF=minCount)
    model = wv.fit(input_data)
    result = model.transform(input_data)
    return result;



# for row in result.collect():
#     text, vector = row
#     print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))