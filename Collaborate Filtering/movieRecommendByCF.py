import os
import pandas as pd
import numpy as np

DATA_PATH = "./ratings.csv"
CACHE_DIR = "./cache/"


# 数据加载
def load_data(data_path):
    """
    :param data_path: 数据集路径
    :return: 用户-物品评分矩阵
    """

    # 数据集缓存地址
    cache_path = os.path.join(CACHE_DIR, "ratings_matrix.cache")

    print("开始加载数据集...")
    if os.path.exists(cache_path):  # 判断是否存在缓存文件
        print("加载缓存文件...")
        ratings_matrix = pd.read_pickle(cache_path)
        print("从缓存加载数据集完毕")
    else:
        print("加载新数据中...")
        # 设置要加载的数据字段的类型
        dtype = {"userId": np.int32, "movieId": np.int32, "rating": np.float32}
        # 家在数据，只用前三列数据，分别是用户ID，电影ID，以及用户对电影的对应评分
        ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(3))
        # 透视表，讲电影ID转换为列名称，转化成一个User-Movie的评分矩阵
        ratings_matrix = ratings.pivot_table(index=["userId"], columns=["movieId"], values=["rating"])
        # 存入缓存文件
        ratings_matrix.to_pickle(cache_path)
        print("数据加载完毕")
    return ratings_matrix


# 相似度计算（计算用户或物品两两相似度）
def compute_pearson_similarity(ratings_matrix, based="user"):
    """
    计算皮尔逊相关系数
    :param ratings_matrix:  用户-物品评分矩阵
    :param based:   "user" or "item"
    :return: 相似度矩阵
    """
    user_similarity_cache_path = os.path.join(CACHE_DIR, "user_similarity.cache")
    item_similarity_cache_path = os.path.join(CACHE_DIR, "item_similarity.cache")
    # 基于皮尔逊相关系数计算相似度
    # 用户相似度
    if based == "user":
        if os.path.exists(user_similarity_cache_path):
            print("正从缓存加载用户相似度矩阵")
            similarity = pd.read_pickle(user_similarity_cache_path)
        else:
            print("开始计算用户相似度矩阵")
            similarity = ratings_matrix.T.corr()
            similarity.to_pickle(user_similarity_cache_path)
    elif based == "item":
        if os.path.exists(item_similarity_cache_path):
            print("正从缓存加载物品相似度矩阵")
            similarity = pd.read_pickle(item_similarity_cache_path)
        else:
            print("开始计算物品相似度矩阵")
            similarity = ratings_matrix.corr()
            similarity.to_pickle(item_similarity_cache_path)
    else:
        raise Exception("Unhandled 'based' value: %s" % based)
    print("相似度矩阵计算/加载完毕")
    return similarity


# User-Based CF评分预测
def predict(uid, iid, ratings_matrix, user_similar):
    """
    预测给定用户对给定物品的评分值
    :param uid: 用户ID
    :param iid: 物品ID
    :param ratings_matrix: 用户-物品评分矩阵
    :param user_similar: 用户两两相似度矩阵
    :return: 预测的评分值
    """
    print("开始预测用户<%d>对电影<%d>的评分" % (uid, iid))
    # 1.找出uid用户的相似用户
    similar_users = user_similar[uid].drop([uid]).dropna()
    # 相似用户筛选规则：正相关的用户
    similar_users = similar_users.where(similar_users > 0).dropna()
    if similar_users.empty is True:
        raise Exception("用户<%d>没有相似的用户")
    # 2.从uid用户的近邻相似用户中筛选出对iid物品有评分记录的近邻用户
    ids = set(ratings_matrix.iloc[:, iid - 1].dropna().index) & set(similar_users.index)
    final_similar_users = similar_users.loc[list(ids)]
    # 3.结合uid用户与其近邻用户的相似度预测uid用户对iid物品的评分
    sum_up, sum_down = 0, 0
    for sim_uid, similarity in final_similar_users.iteritems():
        # 近邻用户的评分数据
        sim_user_rated_movies = ratings_matrix.loc[sim_uid].dropna()
        # 近邻用户对iid物品的评分
        sim_user_rating_for_item = sim_user_rated_movies[iid - 1]
        # 计算分子
        sum_up += similarity * sim_user_rating_for_item
        sum_down += similarity
    # 计算预测的评分值并返回
    predict_rating = sum_up / sum_down
    print("预测出用户<%d>对电影<%d>的评分:%0.2f" % (uid, iid, predict_rating))
    return round(predict_rating, 2)


if __name__ == '__main__':
    ratings_Matrix = load_data(DATA_PATH)
    user_Similar = compute_pearson_similarity(ratings_Matrix, based="user")
    print(predict(1, 1, ratings_Matrix, user_Similar))
