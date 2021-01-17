import json
import pprint
import sys
import pandas as pd
from gensim.models import FastText
from konlpy.tag import Kkma

kkma = Kkma()

def loadparagraphset(ptxt):
    f = open(ptxt, "r", encoding='utf-8')
    paraset = f.readlines()
    f.close()
    preparaset = []
    for i in paraset:
        for k in i.split("다."):
            if k.replace("\n", "") == '' or k.replace("\n", "") == ' ':
                continue
            preparaset.append(k.replace("\n", "")+"다.")
    return preparaset


def trainmodel(paragraphset, fs, fw, fc):
    embedding_model = FastText(
        paragraphset, size=fs, window=fw, min_count=fc, workers=4, sg=1)
    mname = str(fs)+"_"+str(fw)+"_"+str(fc)+".model"
    embedding_model.save(mname)
    print(mname+"save done")
    return embedding_model


def hyungextrac(text):
    pos = kkma.pos(text)
    morph = []
    for i in pos:
        if i[1] == "NNG" or i[1] == "NNP" or i[1] == "VV":
            morph.append(i[0])
    return morph


if __name__ == "__main__":
    paraset = loadparagraphset(sys.argv[1])
    paraexhyung = []
    for i in paraset:
        paraexhyung.append(hyungextrac(i))
    femmodel = trainmodel(paraexhyung, 400, 10, 40)
    # parameter를 수정 하기 바람
