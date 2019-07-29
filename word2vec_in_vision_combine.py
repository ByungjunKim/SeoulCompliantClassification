# -*- coding: utf-8 -*-
"""
Created on Tue May  1 17:56:03 2018

@author: Bigvalue-Data
"""

####학습 + 평가 데이터 combine
combine_w2v = pd.DataFrame(data={'글번호':test_data_with_기획['글번호'],'year':test_data_with_기획['year'],'비전':result_w2v,
                                 '작성자':test_data_with_기획['작성자'],'morp_contents':test_data_with_기획['morp_contents']})
train_data_without_기획[["글번호",'year',"비전","작성자","morp_contents"]]
combine_w2v=pd.concat([combine_w2v,
                       train_data_without_기획[["글번호",'year',"비전","작성자","morp_contents"]]])

combine_w2v = combine_w2v.reset_index(drop=True)
%store combine_w2v
combine_w2v["비전"].unique()
vision = ['환경','교통','안전','문화','건강','주택','복지',
          '경제','여성','세금']
vision[0]

combine_w2v[combine_w2v["비전"].str.contains(vision[0])] #환경
combine_w2v[combine_w2v["비전"].str.contains(vision[7])] #경제

###연도별 비전 분포
combine_w2v.groupby(['year','비전'])['비전'].count().to_csv('연도별 비전 변화.csv',sep='\t')


##combine export
combine_w2v.to_csv('학습 및 비전예측결과_종합데이터.csv',sep='\t',index=False,encoding="CP949")



####헤비유저 및 스팸 연구
#seouloasis(상상지기) 제외필요
combine_w2v[combine_w2v['작성자'].str.contains('seouloasis(상상지기)',regex=False)==False]
combine_w2v[combine_w2v['작성자'].str.contains('seouloasis(상상지기)',regex=False)==False]['작성자'].value_counts() #유저별 포스팅수 카운트 총작성자 24548명
user_count = combine_w2v[combine_w2v['작성자'].str.contains('seouloasis(상상지기)',regex=False)==False]['작성자'].value_counts()

user_count.mean() #평균 3.99건
user_count.median() #중앙값 1
user_count.max() #최대 934건
user_count[user_count>=2] #2건이상 작성자 7461명
user_count[user_count>=2].plot()
user_count[user_count>=100] #100건 이상 유저 확인, 118명
user_count[user_count>=200] #200건 이상 유저 확인, 57명
user_count[user_count>=300] #500건 이상 유저 확인, 14명

user_count.index[0] #1위 유저
combine_w2v[combine_w2v['작성자'].str.contains(user_count.index[0],regex=False)]

#100건이상 유저들의 민원 분류
combine_w2v[combine_w2v['작성자'].isin(user_count[user_count>=100].index)] #총 32631 건
combine_w2v[combine_w2v['작성자'].isin(user_count[user_count>=200].index)] #총 24259 건
combine_w2v[combine_w2v['작성자'].isin(user_count[user_count>=300].index)] #총 10184 건
combine_w2v[combine_w2v['작성자'].isin(user_count[user_count>=100].index)]['비전'].value_counts()
#환경    8592
#교통    7438
#문화    4298
#건강    2610
#안전    2255
#경제    2231
#복지    1952
#여성    1510
#주택    1429
#세금     316

combine_w2v[combine_w2v['작성자'].isin(user_count[user_count>=200].index)]['비전'].value_counts()
#환경    6295
#교통    5338
#문화    3285
#건강    1894
#안전    1832
#경제    1625
#복지    1515
#주택    1132
#여성    1113
#세금     230

#export
combine_w2v[combine_w2v['작성자'].isin(user_count[user_count>=200].index)].to_csv('200건이상 작성_헤비유저.csv',sep="\t",encoding="CP949",index=False)

#100건이상 유저들의 단어 네트워크