import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances


class CollaborateFilteringJaccard(object):
    def __init__(self, user, item, data):
        self.users = user
        self.items = item
        self.dataset = data

    def df_dataset(self):
        self.dataset = np.where(np.isin(self.dataset, ['buy']), np.array([1]), self.dataset)
        self.dataset = np.where(np.isin(self.dataset, [None]), np.array([0]), self.dataset)
        self.dataset = pd.DataFrame(self.dataset, index=self.users, columns=self.items)
        return self.dataset

    def calculate_user_similarity(self, df):
        user_similarity = 1 - pairwise_distances(np.array(df), metric='jaccard')
        user_similarity = pd.DataFrame(user_similarity, index=self.users, columns=self.users)
        return user_similarity

    def calculate_item_similarity(self, df):
        item_similarity = 1 - pairwise_distances(np.array(df).T, metric='jaccard')
        item_similarity = pd.DataFrame(item_similarity, index=self.items, columns=self.items)
        return item_similarity

    def user_basedcf(self, user_sim):
        top2_users = {}
        for i in user_sim.index:
            _df = user_sim.loc[i].drop([i])
            _df_sorted = _df.sort_values(ascending=False)
            top2 = list(_df_sorted.index[:2])
            top2_users[i] = top2
        user_basedcf_results = {}
        for user, sim_users in top2_users.items():
            user_basedcf_result = set()
            for sim_user in sim_users:
                user_basedcf_result = user_basedcf_result.union(
                    set(self.dataset.loc[sim_user].replace(0, np.nan).dropna().index))
            user_basedcf_result -= set(self.dataset.loc[user].replace(0, np.nan).dropna().index)
            user_basedcf_results[user] = user_basedcf_result
        return user_basedcf_results

    def item_basedcf(self, item_sim):
        top2_items = {}
        for i in item_sim.index:
            _df = item_sim.loc[i].drop([i])
            _df_sorted = _df.sort_values(ascending=False)
            top2 = list(_df_sorted.index[:2])
            top2_items[i] = top2
        item_basedcf_results = {}
        for user in self.dataset.index:
            item_basedcf_result = set()
            for item in self.dataset.loc[user].replace(0, np.nan).dropna().index:
                item_basedcf_result = item_basedcf_result.union(set(top2_items[item]))
            item_basedcf_result -= set(self.dataset.loc[user].replace(0, np.nan).dropna().index)
            item_basedcf_results[user] = item_basedcf_result
        return item_basedcf_results


users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['ItemA', 'ItemB', 'ItemC', 'ItemD', 'ItemE']
dataset = [["buy", None, "buy", "buy", None],
           ["buy", None, None, "buy", "buy"],
           ["buy", None, "buy", None, None],
           [None, "buy", None, "buy", "buy"],
           ["buy", "buy", "buy", None, "buy"]]

model = CollaborateFilteringJaccard(user=users, item=items, data=dataset)

# 基于用户的协同过滤推荐（User-based CF）
dataset = model.df_dataset()
user_Similarity = model.calculate_user_similarity(dataset)
user_basedcf = model.user_basedcf(user_Similarity)
print(user_basedcf)

# 基于物品的协同过滤推荐（Item-based CF）
item_Similarity = model.calculate_item_similarity(dataset)
item_basedcf = model.item_basedcf(item_Similarity)
print(item_basedcf)
