# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 18:15:40 2018

@author: Bigvalue-Data
"""

#Packages
import pandas as pd
import numpy as np
import nltk
import gensim
import regex
from gensim.models import word2vec
from tqdm import *
from multiprocessing import pool


##data reload
master = pd.read_csv("master_morp_V3_nouns.csv",sep="\t",encoding="UTF-8",engine='python') #명사
#master = pd.read_csv("master_morp_V2_nouns_delete.csv",sep="\t",encoding="UTF-8",engine='python') #명사
#master = pd.read_csv("master_morp_V2_morphs.csv",sep="\t",encoding="UTF-8",engine='python') #형태소 전체

master = master.dropna(subset=["morp_contents"]) #형태소가 하나도 없는 경우 제외
master = master.reset_index(drop=True)

###stopwords 제거
stopwords = ['아래', '상상', '제안', '까지', '닷컴', '포털', '사이트', '천만', '오아시스', '이벤트', '접수','서울시','서울','특별시',
             '천만상상','파일','첨부','응모','슬로건','공모','공모전','응모전','신청','경우','때문','정도','사항',
                   '해당','호선','겁니다','이것','저것','그것','돋움','신명', '태명', '한컴', '돋움',
                   '동안','거기','저기','여기','대부분','누구','무엇','고딕','만큼','굴림','감사','건지','텐데',
                   '안녕','이번','걸로','수고','겁니까','그간','그건','그때','글쓴이','누가','니다','다면',
                   '뭔가','상상오아시스',
                   'ㅋ',' ','\p{Hangul}{1}'] #한글자, 공백 제거
stops = set(stopwords)

#for i in tqdm(range(len(master))):
#    #master['morp_title'][i]= master['morp_title'][i].split()
#    master['morp_contents'].iloc[i]= [w for w in master['morp_contents'].iloc[i].split() if not w in stops]

#불용어 제거 & contents = title + contents
title_text = master["morp_title"].tolist()
contents_text = master["morp_contents"].tolist()
for j in tqdm(range(len(contents_text))):
    #if (type(title_text[j])==str) & (title_text[j]!='nan'):
    if (type(title_text[j])==str):
        contents_text[j]= [w for w in title_text[j].split() if not w in stops] + [w for w in contents_text[j].split() if not w in stops]
    else:
        contents_text[j]= [w for w in contents_text[j].split() if not w in stops]

#contents에 덮어쓰기
master['morp_contents']=pd.Series(contents_text)
del title_text, contents_text #raw 텍스트 삭제

#year 컬럼 추가
master['year'] = master['작성일'].str[:4]

#문서당 최소 형태소 수 필터링
#master = master[master['morp_contents'].map(len)>0] #morp_contents 빈칸 제거
master = master[master['morp_contents'].map(len)>=20] #morp_contents len(20) 이상만 활용
master = master.reset_index(drop=True)

#'morp_contents' unique 처리
master["res"] = master["morp_contents"].astype(str) #list to string
master = master.drop_duplicates(subset="res")
master = master.reset_index(drop=True)
master = master.drop('res',1)


###########################
##학습 및 테스트 데이터 설정
train_data = master.dropna(subset=["비전"])
train_data = train_data.reset_index(drop=True) #index 재부여
#비전-기획 제외해서 학습데이터 설정
train_data[train_data['비전'].str.contains('기획|기타|건설')] #기획|기타|건설 4588개
train_data_without_기획 = train_data[train_data['비전'].str.contains('기획|기타|건설')==False]
train_data_without_기획 = train_data_without_기획.reset_index(drop=True) #index 재부여

test_data = master[pd.isnull(master['비전'])]
test_data = test_data.reset_index(drop=True) #index 재부여

test_data_with_기획 = pd.concat([test_data,train_data[train_data['비전'].str.contains('기획')]])
test_data_with_기획 = test_data_with_기획.reset_index(drop=True) #index 재부여

#학습,테스트 데이터 텍스트
train_text = train_data["morp_contents"].tolist()
train_text_without_기획 = train_data_without_기획["morp_contents"].tolist()
#train_raw_text = train_data["내용"].tolist()
test_text = test_data["morp_contents"].tolist()
#test_raw_text = test_data["내용"].tolist()
test_text_with_기획 = test_data_with_기획["morp_contents"].tolist()


#학습데이터 13개 피처(문서분류)
train_data["비전"].unique() #환경, 교통, 기획, 주택, 안전, 경제, 건강, 여성, 문화, 복지, 기타, 세금, 건설
train_data_without_기획["비전"].unique() #기획, 기타, 건설 없음=10개

###Vectorize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

vectorizer = CountVectorizer(analyzer = 'word', 
                             tokenizer = lambda doc: doc,
                             preprocessor = None, 
                             stop_words = None, 
                             min_df = 10, # 토큰이 나타날 최소 문서 개수
                             ngram_range=(1, 5),
                             max_features = 20000,
                             lowercase = False
                            )
# 속도 개선을 위해 파이프라인을 사용하도록 개선
# 참고 : https://stackoverflow.com/questions/28160335/plot-a-document-tfidf-2d-graph
pipeline = Pipeline([
    ('vect', vectorizer),
])
    
%time train_data_features = pipeline.fit_transform(train_text)

train_data_features
train_data_features.shape

vocab = vectorizer.get_feature_names()
vocab

# 벡터화된 피처를 확인해 봄
dist = np.sum(train_data_features, axis=0)

for tag, count in zip(vocab, dist):
    print(count, tag)

pd.DataFrame(dist, columns=vocab)
pd.DataFrame(train_data_features[:100].toarray(), columns=vocab).head()