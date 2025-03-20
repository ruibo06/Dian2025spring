import numpy as np

#以下是决策树的构建

#构建CART分类决策树(使用基尼指数作为划分依据)
def calculate_gini(y):
    _ , counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    gini = 1 - np.sum(np.power(proportions,2))
    return gini
    
def split(X, y, feature_index, split_pos):
    is_left = X[:, feature_index] <= split_pos
    return X[is_left], X[~is_left], y[is_left], y[~is_left]

def best_split(X, y):
    best_gini = float('inf')
    best_split = (None,None)
    feature_num = X.shape[1]
    
    for feature_index in range(feature_num):
        all_split_pos = np.unique(X[:, feature_index])
        for split_pos in all_split_pos:
            X_left, X_right, y_left, y_right = split(X, y, feature_index, split_pos)
            
            if len(X_left) == 0 or len(X_right) == 0:
                continue
                
            gini_left = calculate_gini(y_left)
            gini_right = calculate_gini(y_right)
            gini = (len(X_left) / len(X)) * gini_left + (len(X_right) / len(X)) * gini_right
            
            if gini < best_gini:
                best_gini = gini
                best_split = (feature_index, split_pos)
                
    return best_split

class DecisionTree:
    def __init__(self, max_depth=5, max_features = None):
        self.max_depth = max_depth
        self.tree = None
        self.max_features = max_features

    def fit(self, X, y):    
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):  
        if len(np.unique(y)) == 1:
            return {'class': y[0]}
        
        if self.max_depth is not None and depth >= self.max_depth:
            return {'class': np.bincount(y).argmax()}  #计算众数,bincount统计0到最大值各的出现次数，argmax返回最大值的索引
        
        if self.max_features is None:
            feature_index, split_pos = best_split(X, y) 
        else:
            feature_num = X.shape[1]
            feature_chosen = np.random.choice(feature_num, self.max_features, replace=False)
            new_feature_index, split_pos = best_split(X[:,feature_chosen], y)
            feature_index = feature_chosen[new_feature_index]
   
        if feature_index is None or new_feature_index is None:
            return {'class': np.bincount(y).argmax()}  #未能成功划分时（当某特征所有样本值都一样时）返回众数
        
        X_left, X_right, y_left, y_right = split(X, y, feature_index, split_pos)
        
        left_tree = self._build_tree(X_left, y_left, depth + 1)
        right_tree = self._build_tree(X_right, y_right, depth + 1)
        
        return {'feature_index': feature_index, 'split_pos': split_pos, 'left': left_tree, 'right': right_tree}
    
    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self._predict_one(x,self.tree))
        return np.array(prediction)
    
    def _predict_one(self, x, tree):
        if 'class' in tree:        
            return tree['class']
        
        feature_value = x[tree['feature_index']]
        if feature_value <= tree['split_pos']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])



#以下是随机森林的构建
class RandomForest:
    def __init__(self, tree_num = 100, max_features = None, max_depth = None):
        self.tree_num = tree_num
        self.max_features = max_features
        self.max_depth = max_depth
        self.trees = []

    def _boostrap(self, X, y):   #boostrap方法的实现
        sample_num = X.shape[0]
        sample_chosen = np.random.choice(sample_num,sample_num,replace=True)
        return X[sample_chosen,:], y[sample_chosen]

    def fit(self, X, y):
        if self.max_features is None:
            self.max_features = int(np.floor(np.sqrt(X.shape[1])))

        for _ in range(self.tree_num):
            tree = DecisionTree(self.max_depth, self.max_features)
            X_boostrap, y_boostrap = self._boostrap(X, y)
            tree.fit(X_boostrap, y_boostrap)
            self.trees.append(tree)

    def predict(self, X):          #投票
        predictions = []
        final_prediction = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
            
        predictions = np.array(predictions).T #转置，使一棵树对不同X的预测变成不同树对一个X的预测
        for predictions_for_one in predictions:
            final_prediction.append(np.bincount(predictions_for_one).argmax()) #取投票众数
        return np.array(final_prediction)