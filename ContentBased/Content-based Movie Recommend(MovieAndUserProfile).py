import pandas as pd
import numpy as np
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from functools import reduce
import collections

"""
利用tags.csv中每部电影的标签作为电影的候选关键词，利用TF-IDF计算每部电影的标签的tf-idf值，选取Top-N个
关键词作为电影画像标签，并将电影的分类词直接作为每部电影的画像标签。
"""


def get_movie_dataset():
    # 加载基于所有电影的标签，all-tag.csv来自ml-latest数据集中
    _tags = pd.read_csv("tags_all.csv", index_col="movieId", usecols=range(1, 3)).dropna()
    tags = _tags.groupby("movieId").agg(list)

    # 加载电影列表数据集
    movies = pd.read_csv("movies.csv", index_col="movieId")
    # 将类别词分开
    movies["genres"] = movies["genres"].apply(lambda x: x.split("|"))
    # 为每部电影匹配对应的标签数据，如果没有将会使nan
    movies_index = set(movies.index) & set(tags.index)
    new_tags = tags.loc[list(movies_index)]
    ret = movies.join(new_tags)
    """
    构建电影数据集，包含电影id、电影名称、类别、标签四个字段
    如果电影没有标签数据，那么就替换为空列表，map(fun,可迭代对象)
    """
    movie_dataset = pd.DataFrame(map(lambda x: (x[0], x[1], x[2], x[2] + x[3]) if x[3] is not np.nan
    else (x[0], x[1], x[2], []), ret.itertuples()), columns=["movieId", "title", "genres", "tags"])
    movie_dataset.set_index("movieId", inplace=True)
    return movie_dataset


def create_movie_profile(movie_dataset):
    """
    使用tfidf，分析提取topn关键字
    :param movie_dataset: 电影数据集-电影id，电影名，分类，标签
    :return: 电影画像
    """
    dataset = movie_dataset["tags"].values
    # 根据数据集建立词袋，并统计词频，将所有词放入一个词典，使用索引进行获取
    dct = Dictionary(dataset)
    corpus = [dct.doc2bow(line) for line in dataset]
    # 训练TF-IDF模型，即计算TD-IDF值
    model = TfidfModel(corpus)
    _movie_profile = []
    for i, data in enumerate(movie_dataset.itertuples()):
        mid = data[0]
        title = data[1]
        genres = data[2]
        vector = model[corpus[i]]
        movie_tags = sorted(vector, key=lambda x: x[1], reverse=True)[:30]
        topN_tags_weights = dict(map(lambda x: (dct[x[0]], x[1]), movie_tags))
        # 将类别词添加进去，并设置权重值为1
        for g in genres:
            topN_tags_weights[g] = 1
        topN_tags = [i[0] for i in topN_tags_weights.items()]
        _movie_profile.append((mid, title, topN_tags, topN_tags_weights))
    movie_profile = pd.DataFrame(_movie_profile, columns=["movieId", "title", "profile", "weights"])
    movie_profile.set_index("movieId", inplace=True)
    return movie_profile


def create_inverted_table(movie_profile):
    inverted_table = {}
    for mid, weights in movie_profile["weights"].iteritems():
        for tag, weight in weights.items():
            # 到inverted_table字典中用tag作为key取值，如果取不到返回[]
            _ = inverted_table.get(tag, [])
            _.append((mid, weight))
            inverted_table.setdefault(tag, _)
    return inverted_table


"""
user profile画像建立：
1. 提取用户观看列表
2. 根据观看列表和物品画像为用户匹配关键词，并统计词频
3. 根据词频顺序，最多保留topK个词，设k为50，作为用户的标签
"""


def create_user_profile():
    watch_record = pd.read_csv("../Collaborate Filtering/ratings.csv", usecols=range(2), dtype={"userId": np.int32, "movieId": np.int32})
    watch_record = watch_record.groupby("userId").agg(list)
    movie_dataset = get_movie_dataset()
    movie_profile = create_movie_profile(movie_dataset)
    user_profile = {}
    for uid, mids in watch_record.itertuples():
        record_movie_profile = movie_profile.loc[list(mids)]
        counter = collections.Counter(reduce(lambda x, y: list(x) + list(y), record_movie_profile["profile"].values))
        # 最感兴趣的前50个词
        interset_words = counter.most_common(50)
        maxcount = interset_words[0][1]
        interset_words = [(w, round(c / maxcount, 4)) for w, c in interset_words]
        user_profile[uid] = interset_words
    return user_profile


def give_results(user_profile, inverted_tables):
    # 每位用户推荐的10部电影id以及相应的关键词权重和
    all_result = {}
    for uid, interest_words in user_profile.items():
        result_table = {}
        for interest_word, interest_weight in interest_words:
            related_movies = inverted_tables[interest_word]
            for mid, related_weight in related_movies:
                _ = result_table.get(mid, [])
                _.append(interest_weight)
                result_table.setdefault(mid, _)
        rs_result = map(lambda x: (x[0], round(sum(x[1]), 2)), result_table.items())
        rs_result = sorted(rs_result, key=lambda x: x[1], reverse=True)[:10]
        all_result.setdefault(uid, rs_result)
    return all_result


if __name__ == "__main__":
    movieDataset = get_movie_dataset()
    movieProfile = create_movie_profile(movieDataset)
    # pprint(movieProfile)
    # pprint(create_inverted_table(movieProfile))
    invertedTable = create_inverted_table(movieProfile)
    userProfile = create_user_profile()
    allResult = give_results(userProfile, invertedTable)
    print(allResult)
