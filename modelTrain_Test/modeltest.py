import json
import pprint
import sys
import pandas as pd
from gensim.models import FastText
from konlpy.tag import Kkma
import operator


def QAreader(QAcsv):
    QA = pd.read_csv(QAcsv)
    testdiclst = []
    cate_1 = []
    cate_2 = []
    for i in range(len(QA)):
        if QA['1차 category'][i] not in cate_1:
            cate_1.append(QA['1차 category'][i])
        if QA['2차 category'][i] not in cate_2:
            cate_2.append(QA['2차 category'][i])
        testdiclst.append({"Q": QA['Q'][i], "A": QA['A'][i], "cate1": QA['1차 category']
                           [i], "cate2": QA['2차 category'][i], "paragraph": QA['paragraph'][i]})

    return testdiclst, cate_1, cate_2


def loadmodel(mname):
    model = FastText.load(mname)

    return model


def hyungextrac(text):
    pos = kkma.pos(text)
    morph = []
    for i in pos:
        if i[1] == "NNG" or i[1] == "NNP" or i[1] == "VV":
            morph.append(i[0])
    return morph


def calc_similarity(word, model, catelst):
    moon = hyungextrac(word)
    dislst = {}
    for i in catelst:
        sc = 0
        for t in moon:
            simsc = model.similarity(i, t)
            sc += simsc
        dislst[i] = sc
    return dislst


if __name__ == "__main__":
    kkma = Kkma()
    QAdic, cate_1, cate_2 = QAreader(sys.argv[1])
    loaded_model = loadmodel(sys.argv[2])
    cc = 0
    for i in QAdic:
        temp = calc_similarity(i['Q'], loaded_model, cate_1)
        print("Q:", i['Q'])
        print("Realcate:", i['cate1'])
        print("predic:", max(temp.items(), key=operator.itemgetter(1))[0])
        print("s_matrix", temp)

        if max(temp.items(), key=operator.itemgetter(1))[0] in i['cate1']:
            cc += 1
        print()

    print("Accuracy:", cc/len(QAdic))
