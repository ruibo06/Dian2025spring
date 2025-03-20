import pandas as pd
import numpy as np
from forest import RandomForest

data = pd.read_csv("./iris/bezdekIris.data", header=None)   #pandas用于读取数据
#print(data.head())
attributes = np.array(data.iloc[:,:3])
names,classes = np.unique(np.array(data.iloc[:,4]), return_inverse=True)

forest = RandomForest()
forest.fit(attributes,classes)
result = forest.predict(attributes)
print("分类结果：",result)
accuracy = np.mean(result == classes)
print("拟合准确率：",accuracy)