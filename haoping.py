import jieba
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
text = open('haoping1.txt','r',encoding='GBK').read()
word_jieba = jieba.cut(text,cut_all=True)
word_split = " ".join(word_jieba)
wc = WordCloud(font_path="msyh.ttc",background_color="white",max_words=1000,max_font_size=300,width=3000,height=2000).generate(text)
plt.imshow(wc)
plt.axis("off")
plt.show()