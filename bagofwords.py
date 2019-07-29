# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 12:36:52 2018

@author: Bigvalue-Data
"""

from sklearn.ensemble import RandomForestClassifier

# 랜덤포레스트 분류기를 사용
forest = RandomForestClassifier(
    n_estimators = 100, n_jobs = -1)

%time forest = forest.fit(train_data_features, train_data['비전'])

##정확도 분석
#비전이 14개이기때문에 roc_auc 분석 불가능
from sklearn.cross_validation import cross_val_score

#max_feature 30,000
%time np.mean(cross_val_score(forest, train_data_features, train_data['비전'], cv=10)) #0.606
%time np.mean(cross_val_score(forest, train_data_features, train_data['비전'], cv=10,scoring='accuracy')) #0.607

#max_feature 20,000
%time np.mean(cross_val_score(forest, train_data_features, train_data['비전'], cv=10)) #0.6082

#max_feature 20,000 & 4-grams
%time np.mean(cross_val_score(forest, train_data_features, train_data['비전'], cv=10)) #0.607

#max_feature 20,000 & 5-grams
%time np.mean(cross_val_score(forest, train_data_features, train_data['비전'], cv=10)) #0.6082

#max_feature 20,000 & 5-grams & min_df=5
%time np.mean(cross_val_score(forest, train_data_features, train_data['비전'], cv=10)) #0.6092

#max_feature 20,000 & 5-grams & min_df=10
%time np.mean(cross_val_score(forest, train_data_features, train_data['비전'], cv=10)) #06091


####테스트 데이터에 적용
# 위에서 정제해준 리뷰의 첫 번째 데이터를 확인
test_text[0]

# 테스트 데이터를 벡터화 함
%time test_data_features = pipeline.transform(test_text)
test_data_features = test_data_features.toarray()

# 테스트 데이터를 넣고 예측한다.
result = forest.predict(test_data_features)
result[:10]

# 예측 결과를 저장하기 위해 데이터프레임에 담아 준다.
output = pd.DataFrame(data={'글번호':test_data['글번호'], '비전':result})
output.head()
output.to_csv('BOW_model_output.csv', index=False, quoting=3)

## 예측결과 '비전'별 정리
output_vision = output['비전'].value_counts()
output_vision

#학습 데이터 비전 비율 vs 테스트 데이터 비전 비율
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(12,5)

sns.countplot(train_data['비전'], ax=axes[0]) #학습 데이터
sns.countplot(output['비전'], ax=axes[1]) #테스트 데이터
