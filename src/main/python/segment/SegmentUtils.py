# -*- coding: utf-8 -*-
import pandas as pd
import jieba
def segment(value,type):
    seg_list = jieba.cut(value, cut_all=type)
    print("Full Mode: " + "/ ".join(seg_list))



if __name__ == '__main__':
    segment("我来到北京清华大学",False)