import pandas as pd


class CollaborateFilteringPearson(object):
    def __init__(self, user, item, data):
        self.users = user
        self.items = item
        self.dataset = pd.DataFrame(data, index=self.users, columns=self.items)

    def predict_user_based_score(self, given_user, given_item):
        user_similarity = self.dataset.T.corr().round(4)
        top2_users = {}
        for i in user_similarity.index:
            _df = user_similarity.loc[i].drop([i])
            _df_sorted = _df.sort_values(ascending=False)
            top2 = list(_df_sorted.index[:2])
            top2_users[i] = top2
        sim_users = top2_users[given_user]
        score, abs_sum = 0, 0
        for sim_user in sim_users:
            score += user_similarity.loc[given_user, sim_user] * self.dataset.loc[sim_user, given_item]
            abs_sum += user_similarity.loc[given_user, sim_user]
        return score / abs_sum

    def predict_item_based_score(self, given_user, given_item):
        item_similarity = self.dataset.corr().round(4)
        top2_items = {}
        for i in item_similarity.index:
            _df = item_similarity.loc[i].drop([i])
            _df_sorted = _df.sort_values(ascending=False)
            top2 = list(_df_sorted.index[:2])
            top2_items[i] = top2
        sim_items = top2_items[given_item]
        score, item_sum = 0, 0
        for sim_item in sim_items:
            score += item_similarity.loc[sim_item, given_item] * self.dataset.loc[given_user, sim_item]
            item_sum += item_similarity.loc[sim_item, given_item]
        return score / item_sum


users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['ItemA', 'ItemB', 'ItemC', 'ItemD', 'ItemE']
datasets = [
    [5, 3, 4, 4, None],
    [3, 1, 2, 3, 3],
    [4, 3, 4, 3, 5],
    [3, 3, 1, 5, 4],
    [1, 5, 5, 2, 1],
]
pre_user = 'User1'
pre_item = 'ItemE'

model = CollaborateFilteringPearson(users, items, datasets)
predict_score_ub = model.predict_user_based_score(pre_user, pre_item)
print(predict_score_ub)

predict_score_ib = model.predict_item_based_score(pre_user, pre_item)
print(predict_score_ib)
