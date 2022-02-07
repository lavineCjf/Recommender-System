import pandas as pd
import numpy as np

class LFM(object):
    def __init__(self,alpha,reg_p,reg_q,number_LatentFactors=10,num_epochs=10,columns=["uid","iid","rating"]):
        self.alpha = alpha #学习率
        self.reg_p = reg_p #P矩阵正则
        self.reg_q = reg_q #Q矩阵正则
        self.number_LatentFactors = number_LatentFactors #隐式类别数量
        self.number_epochs = num_epochs #最大迭代次数
        self.columns = columns

    def fit(self,datasets):
        """
        训练数据集
        :param datasets: uid,iid,rating
        :return:
        """
        self.dataset = pd.DataFrame(datasets)
        self.user_ratings = datasets.groupby(self.columns[0]).agg([list])[[self.columns[1],self.columns[2]]]
        self.item_ratings = datasets.groupby(self.columns[1]).agg([list])[[self.columns[0],self.columns[2]]]
        self.globalMean = self.dataset[self.columns[2]].mean()
        self.P,self.Q = self.sgd()

    def _init_matrix(self):
        """
        初始化P和Q矩阵，同时设置为0-1之间的随机值作为初始值
        :return:
        """
        P = dict(zip(self.user_ratings.index,np.random.rand(len(self.user_ratings),self.number_LatentFactors).astype(np.float32)))
        Q = dict(zip(self.item_ratings.index,np.random.rand(len(self.item_ratings),self.number_LatentFactors).astype(np.float32)))
        return P,Q

    def sgd(self):
        """
        使用随机梯度下降优化
        :return:
        """
        P,Q = self._init_matrix()
        for i in range(self.number_epochs):
            print("iter%d"%i)
            error_list = []
            for uid,iid,r_ui in self.dataset.itertuples(index=False):
                v_pu = P[uid] #用户向量
                v_qi = Q[iid] #物品向量
                err = np.float32(r_ui-np.dot(v_pu,v_qi))
                v_pu += self.alpha * (err * v_qi - self.reg_p * v_pu)
                v_qi += self.alpha * (err * v_pu - self.reg_q * v_pu)
                P[uid] = v_pu
                Q[iid] = v_qi
                error_list.append(err**2)
            print(np.sqrt(np.mean(error_list)))
        return P,Q

    def predict(self,uid,iid):
        # 如果uid或iid不在，使用全局平均分作为预测结果返回
        if uid not in self.user_ratings.index or iid not in self.item_ratings.index:
            return self.globalMean
        p_u = self.P[uid]
        q_i = self.Q[iid]
        return np.dot(p_u,q_i)

    def test(self,testset):
        """预测测试集数据"""
        for uid,iid,real_rating in testset.itertuples(index=False):
            try:
                pred_rating = self.predict(uid,iid)
            except Exception as e:
                print(e)
            else:
                yield uid,iid,real_rating,pred_rating


if __name__=='__main__':
    dtype = [("userId",np.int32),('movieId',np.int32),('rating',np.float32)]
    dataset = pd.read_csv('ratings.csv',dtype=dict(dtype),usecols=range(3))
    lfm = LFM(0.02,0.01,0.01,10,15,['userId','movieId','rating'])
    lfm.fit(dataset)
    print(lfm.predict(1,1))