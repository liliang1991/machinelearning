from pyspark.ml.feature import CountVectorizer
from pyspark.shell import spark


def CountVectorizerModel(input_col, output_col, vocab_size, min_df, input_data):
    # mindf 必须在文档中出现的最少次数
    # vocabSize 词典大小
    cv = CountVectorizer(inputCol=input_col, outputCol=output_col, vocabSize=vocab_size, minDF=min_df)
    model = cv.fit(input_data)
    result = model.transform(input_data)
    return result;
