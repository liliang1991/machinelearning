import nltk
import jieba
#nltk.download()

filepath="/home/moon/work/tianchi/data"
#fdist1=nltk.FreqDist('ddsas')
#fdist1.plot(50, cumulative=True)
with open(filepath) as file_object:
    contents = file_object.read()
    print(contents)
    text=nltk.text.Text(jieba.lcut(contents))
    print(text.concordance(u'我'))