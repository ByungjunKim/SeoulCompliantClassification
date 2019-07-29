# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 15:33:50 2018

@author: Bigvalue-Data
"""

import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO)

#기술통계량
train_data_without_기획["morp_contents"].str.len().mean() #학습데이터 평균 92단어
test_data_with_기획["morp_contents"].str.len().mean() #학습데이터 평균 67단어


# 파라메터값 지정
num_features = 5000 # 문자 벡터 차원 수
min_word_count = 60 # 최소 문자 수
num_workers = 6 # 병렬 처리 스레드 수
context = 20 # 문자열 창 크기
downsampling = 1e-3 # 문자 빈도수 Downsample

# 초기화 및 모델 학습
from gensim.models import word2vec

# 모델 학습
model = word2vec.Word2Vec(train_text_without_기획,  #train_text or train_text_without_기획
                          workers=num_workers, 
                          size=num_features, 
                          min_count=min_word_count,
                          window=context,
                          sample=downsampling)
model
# 학습이 완료 되면 필요없는 메모리를 unload 시킨다.
model.init_sims(replace=True)
model_name = '5000features_60minwords_20text'
model.save(model_name)

# 유사도가 없는 단어 추출
model.wv.doesnt_match('자전거 버스 택시'.split())
# 가장 유사한 단어를 추출
#model.wv.most_similar("스크린도어")
model.wv.most_similar("미세먼지")
#model.wv.most_similar("오세훈")

####Word2Vec으로 벡터화 한 단어를 t-SNE 를 통해 시각화
# 참고 https://stackoverflow.com/questions/43776572/visualise-word2vec-generated-from-gensim
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import gensim 
import gensim.models as g
from matplotlib import font_manager, rc
import matplotlib.font_manager as fm

##한글 폰트 설정
#windows
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

#mac
#font_list_mac = fm.OSXInstalledFonts()
#font_name = font_manager.FontProperties(fname="/Library/Fonts/NanumGothicBold.otf").get_name()
#rc('font', family=font_name)

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

#model_name = '300features_40minwords_10text'
model = g.Doc2Vec.load(model_name)

vocab = list(model.wv.vocab)
X = model[vocab]

print(len(X))
print(X[0][:10])
tsne = TSNE(n_components=2)

# 100개의 단어에 대해서만 시각화
X_tsne = tsne.fit_transform(X[:100,:])
# X_tsne = tsne.fit_transform(X)

df = pd.DataFrame(X_tsne, index=vocab[:100], columns=['x', 'y'])
df.shape
df.head(10)

fig = plt.figure()
fig.set_size_inches(40, 20)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=30)
plt.show()

###plot 그리기
fig


###평균 feature 계산
import numpy as np
def makeFeatureVec(words, model, num_features):
    """
    주어진 문장에서 단어 벡터의 평균을 구하는 함수
    """
    # 속도를 위해 0으로 채운 배열로 초기화한다.
    featureVec = np.zeros((num_features,),dtype="float32")

    nwords = 0.
    # Index2word는 모델의 사전에 있는 단어 명을 담은 리스트이다.
    # 속도를 위해 set 형태로 초기화한다.
    index2word_set = set(model.wv.index2word)
    # 루프를 돌며 모델 사전에 포함이 되는 단어라면 피처에 추가한다.
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 결과를 단어 수로 나누어 평균을 구한다.
    np.seterr(divide='ignore') #0 or NaN 나눌때 생기는 오류 방지
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    # 리뷰 단어 목록의 각각에 대한 평균 feature 벡터를 계산하고 
    # 2D numpy 배열을 반환한다.

    # 카운터를 초기화한다.
    counter = 0.
    # 속도를 위해 2D 넘파이 배열을 미리 할당한다.
    reviewFeatureVecs = np.zeros(
        (len(reviews),num_features),dtype="float32")

    for review in reviews:
       # 매 1000개 리뷰마다 상태를 출력
       if counter%4000. == 0.:
           print("민원 %d of %d" % (counter, len(reviews)))
       # 평균 피처 벡터를 만들기 위해 위에서 정의한 함수를 호출한다.
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, \
           num_features)
       # 카운터를 증가시킨다.
       counter = counter + 1.
    return reviewFeatureVecs


from sklearn.preprocessing import Imputer
#%time trainDataVecs = getAvgFeatureVecs(\
#    train_text, model, num_features ) 
#학습 데이터 벡터
%time trainDataVecs = getAvgFeatureVecs(\
    train_text_without_기획, model, num_features ) 

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(trainDataVecs)
trainDataVecs=imp.transform(trainDataVecs) ##nan, infinite 제거

#%time testDataVecs = getAvgFeatureVecs(\
#        test_text, model, num_features )
#imp.fit(testDataVecs)
#testDataVecs=imp.transform(testDataVecs) ##nan, infinite 제거

#테스트 데이터 '기획' 포함 벡터
%time testDataVecs = getAvgFeatureVecs(\
        test_text_with_기획, model, num_features )
imp.fit(testDataVecs)
testDataVecs=imp.transform(testDataVecs) ##nan, infinite 제거

###랜덤포레스트로 샘플링
from sklearn.ensemble import RandomForestClassifier

forest_w2v = RandomForestClassifier(
    n_estimators = 100, n_jobs = -1, random_state=2018)
#%time forest_w2v = forest_w2v.fit( trainDataVecs, train_data["비전"] )
%time forest_w2v = forest_w2v.fit( trainDataVecs, train_data_without_기획["비전"] )


###정확도 계산
from sklearn.model_selection import cross_val_score
#%time score = np.mean(cross_val_score(\
#    forest_w2v, trainDataVecs, \
#    train_data['비전'], cv=10))
%time score = np.mean(cross_val_score(\
    forest_w2v, trainDataVecs, \
    train_data_without_기획['비전'], cv=10))

score #0.62877774795614738 vs  0.71649097027257003

result_w2v = forest_w2v.predict( testDataVecs )
%store result_w2v

# 예측 결과를 저장하기 위해 데이터프레임에 담아 준다.
output_w2v = pd.DataFrame(data={'글번호':test_data_with_기획['글번호'], '비전':result_w2v,
                                'year':test_data_with_기획['year']})
output_w2v.head()
output_w2v.to_csv('w2v_model_output.csv', index=False, quoting=3)

## 예측결과 '비전'별 정리
output_w2v_vision = output_w2v['비전'].value_counts()
output_w2v_vision

#학습 데이터 비전 비율 vs 테스트 데이터 비전 비율
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(12,5)

sns.countplot(train_data_without_기획['비전'], ax=axes[0]) #학습 데이터
sns.countplot(output_w2v['비전'], ax=axes[1]) #테스트 데이터
