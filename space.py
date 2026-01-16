#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[3]:


import numpy as np
import pandas as pd

# 定义步长
step = 0.01

# 定义元素的取值范围
ranges = {
    'Bi_Na': np.arange(0.15, 0.29+step, step),
    'Ba': np.arange(0, 0.24+step, step),
    'Sr': np.arange(0, 0.24+step, step),
    'Ca': np.arange(0, 0.19+step, step),
    'Zr': np.arange(0, 0.19+step, step),
    'Nb': np.arange(0, 0.19+step, step)
}

# 使用 meshgrid 生成坐标网格
grids = np.meshgrid(*ranges.values(), indexing='ij')

# 计算每个点的元素和
sums1 = 2*grids[0] + grids[1] + grids[2] + grids[3]+ grids[5]
sums2 = grids[4] + grids[5]

# 使用布尔索引选择满足约束条件的点
valid_indices_1 = (np.isclose(sums1, 1, atol=1e-5) & (sums2 > 0) & (sums2 <= 0.2))

# 获取满足约束条件的点的坐标
results_1 = [grid[valid_indices_1] for grid in grids]

# 创建 DataFrame 并移除重复值
df = pd.DataFrame({
    'Bi': results_1[0],
    'Na': results_1[0]+ results_1[5],
    'Ba': results_1[1],
    'Sr': results_1[2],
    'Ca': results_1[3],
    'Ti': 1-results_1[4]-results_1[5],
    'Zr': results_1[4],
    'Nb': results_1[5],
    'O': 3,
}).drop_duplicates()

# 打印结果数量
print(f"找到 {len(results_1[0])} 种可能的组合。")
df.to_excel('D:/DATASET9/contents.xlsx', index=False)


# In[ ]:




