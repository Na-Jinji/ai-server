# !/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_soup(x):
    return x['category'] + ' ' + x['tag']


class Model:
    def __init__(self, tokenizer=None):
        gyeongju_data = pd.read_csv('model/data.csv')
        self.metaData = gyeongju_data[['category', 'title', 'tag']].drop_duplicates()
        self.metaData['soup'] = self.metaData.apply(create_soup, axis=1)

        # 이름:index - 예) 로라커피:0, 이스트앵글:1
        self.indices = pd.Series(self.metaData.index, index=self.metaData['title']).drop_duplicates()

        # BOW 인코딩
        if tokenizer:
            count = CountVectorizer(analyzer='word', tokenizer=tokenizer.morphs)
        else:
            count = CountVectorizer(analyzer='word')
        count_matrix = count.fit_transform(self.metaData['soup'])

        # 코사인 유사도 구하기
        self.cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

        # index 초기화
        self.metaData = self.metaData.reset_index()
        self.indices = pd.Series(self.metaData.index, index=self.metaData['title'])

    # title의 정확한 명칭 얻어오기
    def set_exact_title(self, t):
        if t in self.indices:  # 정확한 명칭이면 통과
            return t

        df_title = self.metaData[self.metaData['title'].str.contains(t)]
        values = df_title['title'].values
        if len(values) <= 0:
            return ''
        return values[0]

    # 10개의 추천리스트 가져오기
    def get_recommendations(self, title):
        title = self.set_exact_title(title)
        if len(title) <= 0 or title == '':
            return []

        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim2[idx]))  # 유사도 측정
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # 내림차순

        sim_scores = sim_scores[1:6]  # 5개
        attraction_indices = [i[0] for i in sim_scores]     # 장소 index
        scores = [i[1] for i in sim_scores]     # 유사도

        # debug
        for i in scores:
            print(i)

        result_data = self.metaData[['title', 'tag']].iloc[attraction_indices]
        result_data['scores'] = np.array(scores)
        return result_data['title'].values.tolist()


def return_recommendations(target):
    m = Model()
    expected_recommendations = m.get_recommendations(target)
    return expected_recommendations

