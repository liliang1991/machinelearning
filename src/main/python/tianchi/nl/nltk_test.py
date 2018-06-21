import nltk
import jieba
#nltk.download()

filepath="/home/moon/data/1.txt"
fdist1=nltk.FreqDist('ddsas')
fdist1.plot(50, cumulative=True)
with open(filepath) as file_object:
    contents = file_object.read()
    text=nltk.text.Text(jieba.lcut(contents))
    print(text.concordance(u'我'))