# -*- coding: utf-8 -*-
#Packages
from konlpy.tag import Mecab
from collections import Counter
import pandas as pd
import numpy as np
import regex
import nltk
import gensim

resultCounter = Counter()
mecab = Mecab()


###CSV read
master = pd.read_csv("master.csv",sep="\t",encoding="UTF-8")
#master = pd.read_csv("test2.txt",sep="\t",encoding="UTF-8")
master["내용"] = master["내용"].fillna("0") #형태소 분석할 내용 탭에 NaN 값이 있으면 "0"으로 전환
master["제목"] = master["제목"].fillna("0")
#hangul = re.compile('[^ \u3131-\u3163\uac00-\ud7a3]+')
hangul = regex.compile('[^\p{Hangul}]+')
test = " 한강엔 통근배를 선착장dfs3엔 통근자전거 3대를 ■ 상상의 배경 ○ 우리 서울에는 수량이 풍부하고 잔잔한 한강이 있습니다. ○ 서울은 교통난과 대기오염이라는 병을 앓고 있습니다. Welcome to"
mecab.morphs(hangul.sub('',test))


for j in range(len(master)):
    master["내용"][j] = hangul.sub('',master["내용"][j]) #한글만 남기기
    if(j%10000==0):
    	print(j)

#master.iloc[132585]["내용"] = re.sub("[\xa0,\x1c]"," ",master.iloc[132585]["내용"])

#리스트 형태로 명사만 추출 & 1음절 이하 제거
morp_title_list = []
morp_contents_list = []
for k in range(len(master)):
    title = mecab.nouns(master['제목'][k])
    contents = mecab.nouns(master['내용'][k])
    #title = mecab.morphs(master['제목'][k])
    #contents = mecab.morphs(master['내용'][k])

    morp_title_list.append(regex.findall(r'\p{Hangul}{2,10}',' '.join(title)))
    morp_contents_list.append(regex.findall(r'\p{Hangul}{2,10}',' '.join(contents)))
    if(k%10000==0):
        print(k)
morp_title_list_morphs= morp_title_list
morp_contents_list_morphs= morp_contents_list


#master에 morp 컬럼 추가
master["morp_title"] = morp_title_list
master["morp_contents"] = morp_contents_list
#master.to_csv("master_morp_V2_morphs.csv",sep="\t",index=False)
master.to_csv("master_morp_V3_nouns.csv",sep="\t",index=False)

##비전컬럼에 내용이 있는 것만 추출 -> 학습데이터
master.dropna(subset=["비전"]).to_csv("master_morp_vision_V2.csv",sep="\t",index=False)
train_data = master.dropna(subset=["비전"])
test_data = master[pd.isna(master['비전'])]

##컬럼순서 전환
#master = master[['글번호','공개','제목','작성자','득표수','작성일','진행상태','제안종류','업무분류','비전',
#'주관부서','내용','부서의견','morp_title','morp_contents']]
