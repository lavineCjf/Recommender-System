import pandas as pd
import numpy as np


def data_split(data_path, x=0.8, random=False):
    """
    切分数据集，为了保证用户数量保持不变，将每个用户的评分数据按比例进行拆分
    :param data_path: 数据集路径
    :param x: 训练集的比例，如x=0.8，则0.2是测试集
    :param random: 是否随机切分
    :return: 用户-物品评分矩阵
    """
    print("开始切分数据集...")
    # 设置要加载的数据字段的类型
    dtype = {'userId': np.int32, 'movieId': np.int32, 'rating': np.float32}
    # 加载数据。只用前三列数据，分别是用户ID，电影ID，以及用户对电影的对应评分
    ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(3))
    # 测试集的索引
    testset_index = []
    # 为保证每个用户在测试集和训练集都有数据，因此按userId聚合
    for uid in ratings.groupby('userId').any().index:
        user_rating_data = ratings.where(ratings['userId'] == uid).dropna()
        if random:
            index = list(user_rating_data.index)
            np.random.shuffle(index)
            _index = round(len(user_rating_data) * x)
            testset_index += list(index[_index:])
        else:
            index = round(len(user_rating_data) * x)
            testset_index += list(user_rating_data.index.values[index:])
    testset = ratings.loc[testset_index]
    trainset = ratings.drop(testset_index)
    print("完成数据集切分...")
    return trainset, testset


def accuracy(predict_results, method="all"):
    """
    准确性指标
    :param predict_results: 预测结果，类型为容器，每个元素是一个包含uid,iid,real_rating,pred_rating的序列
    :param method: 指标方法，类型为字符串，rmse或mae，否则返回两者rmse和mae
    :return: 指标值
    """

    def rmse(predict_results):
        length = 0
        _rmse_sum = 0
        for uid, iid, real_rating, pred_rating in predict_results:
            length += 1
            _rmse_sum += (pred_rating - real_rating) ** 2
        return round(np.sqrt(_rmse_sum / length), 4)

    def mae(predict_results):
        length = 0
        _mae_sum = 0
        for uid, iid, real_rating, pred_rating in predict_results:
            length += 1
            _mae_sum += abs(pred_rating - real_rating)
        return round(_mae_sum / length, 4)

    def rmse_mae(predict_results):
        length = 0
        _rmse_sum = 0
        _mae_sum = 0
        for uid, iid, real_rating, pred_rating in predict_results:
            length += 1
            _rmse_sum += (pred_rating - real_rating) ** 2
            _mae_sum += abs(pred_rating - real_rating)
        return round(np.sqrt(_rmse_sum / length), 4), round(_mae_sum / length, 4)

    if method.lower() == "rmse":
        rmse(predict_results)
    elif method.lower() == "mae":
        mae(predict_results)
    else:
        return rmse_mae(predict_results)


class BaselineCFBySGD(object):
    def __init__(self, number_epochs, reg_bu, reg_bi, columns=None):
        """
        :param number_epochs: 梯度下降最高迭代次数
        :param reg_bu: 用户偏置参数
        :param reg_bi: 物品偏置参数
        :param columns: 数据集中user-item-rating字段的名称
        """

        if columns is None:
            columns = ["uid", "iid", "rating"]
        self.number_epochs = number_epochs
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.columns = columns

    def fit(self, dataset):
        """
        :param dataset: 用户评分数据
        :return:
        """
        self.dataset = dataset
        # 用户评分数据
        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        # 物品评分数据
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        # 计算全局平均分
        self.global_mean = self.dataset[self.columns[2]].mean()
        # 调用als方法训练模型参数
        self.bu, self.bi = self.als()

    def als(self):
        """
        利用交替最小二乘法，优化bu, bi的值
        :return: bu, bi
        """
        bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))
        for i in range(self.number_epochs):
            print("iter%d" % i)
            for iid, uids, ratings in self.items_ratings.itertuples(index=True):
                _sum = 0
                for uid, rating in zip(uids, ratings):
                    _sum += rating - self.global_mean - bu[uid]
                bi[iid] = _sum / (self.reg_bi + len(uids))
            for uid, iids, ratings in self.users_ratings.itertuples(index=True):
                _sum = 0
                for iid, rating in zip(iids, ratings):
                    _sum += rating - self.global_mean - bi[iid]
                bu[uid] = _sum / (self.reg_bu + len(iids))
        return bu, bi

    def predict(self, uid, iid):
        # 评分预测
        if iid not in self.items_ratings.index:
            raise Exception("无法预测用户<{uid}>对电影<{iid}>的评分，因为训练集中缺失<{iid}>的数据".format(uid=uid, iid=iid))
        predict_rating = self.global_mean + self.bu[uid] + self.bi[iid]
        return predict_rating

    def test(self, testset):
        # 预测测试集数据
        for uid, iid, real_rating in testset.itertuples(index=False):
            try:
                pred_rating = self.predict(uid, iid)
            except Exception as e:
                print(e)
            else:
                yield uid, iid, real_rating, pred_rating


if __name__ == '__main__':
    trainset, testset = data_split("ratings.csv", random=True)
    bcf = BaselineCFBySGD(20, 0.1, 0.1, ["userId", "movieId", "rating"])
    bcf.fit(trainset)
    pred_results = bcf.test(testset)
    rmse, mae = accuracy(pred_results)
    print("rmse: ", rmse, "mae: ", mae)
