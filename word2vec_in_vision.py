# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:22:05 2018

@author: Bigvalue-Data
"""
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import gensim 
import gensim.models as g
from matplotlib import font_manager, rc

##한글 폰트 설정
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

###비전별 w2v 모델 설정
train_data_without_기획["비전"].unique()
train_data_without_기획["비전"].value_counts()
###학습데이터
#환경    2331
#교통    1979
#안전    1117
#문화    1028
#건강     855
#주택     785
#복지     640
#경제     571
#여성     474
#세금     142

output_w2v['비전'].value_counts()
###테스트데이터
#교통    26088
#환경    22154
#문화    13280
#건강     5791
#안전     4599
#경제     4501
#여성     4231
#복지     4037
#주택     3195
#세금      786

###테스트 데이터 with 트레인('기획') 데이터
#교통    26525
#환경    22938
#문화    14325
#건강     6088
#안전     5006
#경제     4852
#여성     4482
#복지     4224
#주택     3313
#세금      908

output_w2v['year'].value_counts()
#output_w2v[['비전','year']].apply(pd.value_counts)
output_w2v.groupby(['비전','year']).count()


# 초기화 및 모델 학습
from gensim.models import word2vec
vision = ['환경','교통','안전','문화','건강','주택','복지',
          '경제','여성','세금']
train_text_vision = {}
for i in tqdm(range(len(vision))):
    train_text_vision[i]=train_data[train_data['비전'].str.contains(vision[i])]["morp_contents"].tolist()

# 비전별 모델 학습
model_vision = {}
for vision_number in tqdm(range(len(vision))):
    model_vision[vision_number] = word2vec.Word2Vec(train_text_vision[vision_number],
                              workers=num_workers, 
                              size=num_features, 
                              min_count=min_word_count,
                              window=context,
                              sample=downsampling)

# 학습이 완료 되면 필요없는 메모리를 unload 시킨다.
model_vision[1].init_sims(replace=True)
model_name_1 = '2000features_40minwords_10text_1'
#model_name = '500features_80minwords_10text'
#model_name = '1000features_40minwords_20text'
model_vision[1].save(model_name_1)

model_vision[1] = g.Doc2Vec.load(model_name_1)

vocab_vision = {}
vocab_vision[1] = list(model_vision[1].wv.vocab)
X_vision_1 = model_vision[1][vocab_vision[1]]

print(len(X_vision_0))
print(X_vision_0[1][:10])
tsne_vision_1 = TSNE(n_components=2)

# 100개의 단어에 대해서만 시각화
X_tsne_vision_1 = tsne_vision_0.fit_transform(X_vision_0[:30,:])
# X_tsne = tsne.fit_transform(X)

df = pd.DataFrame(X_tsne_vision_1, index=vocab_vision[1][:30], columns=['x', 'y'])
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
