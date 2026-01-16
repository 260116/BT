#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
df = pd.read_excel('D:\\DATASET9\\BNT_14.xlsx')
array = df.values
X = array[:,1:13].astype(float)
y = array[:,-1]
y.shape
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)


# In[2]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

# 自定义颜色列表
colors = ["#C9667D", "#F46D43", "#FEE090", "#ABD9E9", "#74ADD1", "#648DC3"]
n_bins = 100  # 颜色过渡的平滑度
cmap_name = 'custom_div_cmap'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# 设置图像尺寸（英寸），8厘米约为3.15英寸
plt.figure(figsize=(5, 5))

# 设置字体为Times New Roman，字体大小为10磅，加粗
rcParams.update({'font.size':10, 'font.weight': 'bold', 'font.family': 'Times New Roman'})

# 假设df是已经定义并包含数值型列的DataFrame
# 选择数据框中的数值型列
numeric_cols = df.select_dtypes(include=[np.number])

# 删除指定列
if 'W' in numeric_cols.columns:
    numeric_cols = numeric_cols.drop(columns=['W'])

# 生成热图，使用自定义颜色映射
ax = sns.heatmap(numeric_cols.corr(), cmap=custom_cmap, annot=False)

# 设置x轴标签旋转90度
plt.xticks(rotation=90)
plt.savefig('D:\\DATASET9\\fig\\pearson_W.png', dpi=600, bbox_inches='tight')
plt.savefig('D:\\DATASET9\\fig\\pearson_W.svg', dpi=600, bbox_inches='tight')
plt.show()


# In[3]:


print(numeric_cols.corr())


# In[4]:


X_1 = scaled_data[:,[1,3,4,9,11]]


# In[5]:


from sklearn.model_selection import LeaveOneOut, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Define the RMSE scoring function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Create a scorer that is a higher-is-better scorer
rmse_scorer = make_scorer(rmse, greater_is_better=False)

# Model definitions and parameter grids
models = {
    'rf': RandomForestRegressor(),
    'gp': make_pipeline(PolynomialFeatures(), LinearRegression()),
    'svm.r': SVR(kernel='rbf'),
    'gb': GradientBoostingRegressor(),
    'lr': LinearRegression(),
    'lasso': Lasso(),
    'ridge': Ridge()
}

param_grids = {
    'rf': {
        'n_estimators': [10, 50, 100],
        'max_features': ['auto', 'sqrt'],
        'min_samples_split': [4, 6, 8]
    },
    'gp': {
        'polynomialfeatures__degree': [2, 3, 4],
        'linearregression__fit_intercept': [True, False]
    },
    'svm.r': {
        'C': [0.1, 1, 10, 50],
        'gamma': ['scale', 'auto', 0.1, 0.5]
    },
    'gb': {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5]
    },
    'lr': {
        # 线性模型通常没有需要调整的超参数
    },
    'lasso': {
        'alpha': [0.001, 0.01, 0.1, 1]
    },
    'ridge': {
        'alpha': [0.1, 1, 10, 100]
    }
}

# Leave-One-Out cross-validation setup
loo = LeaveOneOut()
model_performance = {}

for model_name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[model_name], cv=loo, scoring=rmse_scorer)
    grid_search.fit(X_1, y)
    best_model = grid_search.best_estimator_

    # Calculate test RMSE using cross-validation
    test_rmse_scores = cross_val_score(best_model, X_1, y, cv=loo, scoring=rmse_scorer)
    test_rmse = -np.mean(test_rmse_scores)  # Use negative because scores are negatives

    # Manually calculate training RMSE
    train_rmse_list = []
    for train_index, test_index in loo.split(X_1):
        X_train, X_test = X_1[train_index], X_1[test_index]
        y_train, y_test = y[train_index], y[test_index]
        best_model.fit(X_train, y_train)
        y_pred_train = best_model.predict(X_train)
        train_rmse_list.append(rmse(y_train, y_pred_train))
    train_rmse = np.mean(train_rmse_list)

    # Store performance and best parameters
    model_performance[model_name] = {
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Best Parameters': grid_search.best_params_
    }

# Print the performance and best parameters for each model
for model_name, performance in model_performance.items():
    print(f"{model_name} performance:")
    print(f"  Train RMSE: {performance['Train RMSE']}")
    print(f"  Test RMSE: {performance['Test RMSE']}")
    print(f"  Best Parameters: {performance['Best Parameters']}\n")

labels = list(model_performance.keys())
train_rmse = [model_performance[model]['Train RMSE'] for model in labels]
test_rmse = [model_performance[model]['Test RMSE'] for model in labels]

x = np.arange(len(labels))  # label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, train_rmse, width, label='Train RMSE')
rects2 = ax.bar(x + width/2, test_rmse, width, label='Test RMSE')

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('RMSE')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[6]:


# 假设 model_performance 是一个字典，包含模型及其对应的RMSE值
labels = list(model_performance.keys())
train_rmse = [model_performance[model]['Train RMSE'] for model in labels]
test_rmse = [model_performance[model]['Test RMSE'] for model in labels]

x = np.arange(len(labels))  # label locations
width = 0.45  # 增加柱子的宽度使得柱状图更加粗

# 设置图形尺寸（宽度：4，高度：4）
fig, ax = plt.subplots(figsize=(3.85, 3.8))
rects1 = ax.bar(x - width/2, train_rmse, width, label='Train RMSE', color='#e74c3c', alpha=0.8)  # 红色调的柱子
rects2 = ax.bar(x + width/2, test_rmse, width, label='Test RMSE', color='#3498db', alpha=0.8)  # 蓝色调的柱子

# 添加标签、标题和自定义的x轴刻度标签等，设置字体大小为10磅，加粗
ax.set_ylabel('RMSE', fontname='Times New Roman', fontsize=12, fontweight='bold')
ax.set_xlabel('ML Model', fontname='Times New Roman', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontname='Times New Roman', fontsize=12, fontweight='bold')
ax.legend()

# 设置图例的字体大小和加粗
legend = ax.legend(prop={'family': 'Times New Roman', 'size': 10, 'weight': 'bold'})
legend.get_frame().set_linewidth(0.0)
# 格式化标签显示为两位小数，设置标签字体大小和加粗
ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=9, fontweight='bold')
ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=9, fontweight='bold')

# 设置x轴和y轴的刻度字体，大小，和加粗
plt.xticks(fontname='Times New Roman', fontsize=10, fontweight='bold')
plt.yticks(fontname='Times New Roman', fontsize=10, fontweight='bold')
ax.set_ylim([0, 1.5])
# 自定义刻度标记向内，设置刻度字体大小和加粗
ax.tick_params(axis='both', which='both', direction='in', labelsize=12)

plt.tight_layout()
plt.savefig('D:\\DATASET9\\fig\\model_rmse_W.png', dpi=600)
plt.savefig('D:\\DATASET9\\fig\\model_rmse_W.svg', dpi=600)
plt.show()


# In[ ]:





# In[7]:


from itertools import combinations
model_1 =RandomForestRegressor(max_features='sqrt',min_samples_split=4, n_estimators=100)
# 存储每个特征子集的最小 RMSE
min_rmse_per_feature_num = {}

# 存储所有的 RMSE
all_rmse = []

# 存储 RMSE 最小的特征子集
best_features = None

# 对于每个可能的特征子集
for num_features in range(1, X_1.shape[1] + 1):
    min_rmse = np.inf
    for features in combinations(range(X_1.shape[1]), num_features):
        # 训练模型
        model_1.fit(X_1[:, features], y)
        
        # 预测
        y_pred = model_1.predict(X_1[:, features])
        
        # 计算 RMSE
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # 更新最小 RMSE 和最佳特征
        if rmse < min_rmse:
            min_rmse = rmse
            if num_features == 4:
                best_features = features
        
        # 存储 RMSE
        all_rmse.append((num_features, rmse))
    
    # 存储每个特征子集的最小 RMSE
    min_rmse_per_feature_num[num_features] = min_rmse

# 输出 RMSE 最小的特征子集
print(f"特征子集个数为三且 RMSE 值最小的特征为：{best_features}")

# 画出所有的 RMSE
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Times New Roman',
    'font.weight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold'
})

# 设置图形尺寸为8厘米乘8厘米，约3.15英寸乘3.15英寸
fig, ax = plt.subplots(figsize=(4, 4))

ax.scatter(*zip(*all_rmse), alpha=0.5, color='#3498db')  # 蓝色表示所有子集
ax.plot(list(min_rmse_per_feature_num.keys()), list(min_rmse_per_feature_num.values()), 'r-o', color='#e74c3c')  # 红色表示每个子集大小的最小 RMSE

ax.set_xlabel('Number of Features', weight='bold')
ax.set_ylabel('RMSE', weight='bold')

ax.tick_params(axis='both', which='both', direction='in', labelsize=12, width=1)
plt.savefig('D:\\DATASET9\\fig\\features_rmse_W.png', dpi=600)
plt.savefig('D:\\DATASET9\\fig\\features_rmse_W.svg', dpi=600)
plt.show()


# In[ ]:





# In[ ]:





# In[8]:


from itertools import combinations
model_1 = RandomForestRegressor(max_features='sqrt',min_samples_split=4, n_estimators=100)

# 存储每个特征子集的详细信息
all_rmse_details = []

# 存储每个特征子集的最小 RMSE
min_rmse_per_feature_num = {}

# 存储 RMSE 最小的特征子集
best_features = None

for num_features in range(1, X_1.shape[1] + 1):
    min_rmse = np.inf
    for features in combinations(range(X_1.shape[1]), num_features):
        model_1.fit(X_1[:, features], y)
        y_pred = model_1.predict(X_1[:, features])
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # 记录详细信息（特征数，特征组合，RMSE）
        all_rmse_details.append((num_features, features, rmse))
        
        if rmse < min_rmse:
            min_rmse = rmse
            if num_features == 4:  # 这里保持原有逻辑
                best_features = features
                
    min_rmse_per_feature_num[num_features] = min_rmse

# 输出所有特征子集的RMSE
print("\n所有特征子集的RMSE值：")
for detail in all_rmse_details:
    print(f"特征数: {detail[0]} | 特征组合: {detail[1]} | RMSE: {detail[2]:.4f}")

# 输出到文件（可选）
with open('D:\\DATASET7\\fig\\features_rmse_details.txt', 'w') as f:
    f.write("特征数 | 特征组合 | RMSE\n")
    for detail in all_rmse_details:
        f.write(f"{detail[0]} | {detail[1]} | {detail[2]:.4f}\n")

# 保持原有绘图代码不变...


# In[9]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

X_2 = X_1[:,[0,1,2,3]]
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_2, y, test_size=0.2, random_state=42)

model_2 =RandomForestRegressor(max_features='sqrt',min_samples_split=4, n_estimators=100)
model_2.fit(X_train_1, y_train_1)

y_train_pred = model_2.predict(X_train_1)
y_test_pred = model_2.predict(X_test_1)

# 计算训练集和测试集的 RMSE
train_rmse = np.sqrt(mean_squared_error(y_train_1, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_1, y_test_pred))
print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
# 设置字体和其他样式
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Times New Roman',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'legend.frameon': False
})

# 设置图形尺寸为8厘米乘8厘米，约3.15英寸乘3.15英寸
fig, ax = plt.subplots(figsize=(4, 4))

# 绘制训练数据散点图
ax.scatter(y_train_1, y_train_pred, color='#3498db', label='Train Data')  # 改为#3498db蓝色
# 绘制测试数据散点图
ax.scatter(y_test_1, y_test_pred, color='#e74c3c', label='Test Data')  # 改为#e74c3c红色
# 绘制拟合线
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Fit')

# 设置x和y轴标签
ax.set_xlabel('True Values ($J/cm^3$)')
ax.set_ylabel('Predictions ($J/cm^3$)')

# 设置坐标轴刻度朝内
ax.tick_params(axis='both', which='both', direction='in', labelsize=12, width=1)

# 显示图例，并设置字体大小
legend = ax.legend(fontsize='small')
plt.savefig('D:\\DATASET9\\fig\\RandomForestRegressor_W.png', dpi=600)
plt.savefig('D:\\DATASET9\\fig\\RandomForestRegressor_W.svg', dpi=600)
plt.show()


# In[10]:


# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
# import pandas as pd

# # 假设 X_1 和 y 已经定义
# X_2 = X_1[:, [0, 1, 2, 3]]

# # 定义要测试的训练集比例
# test_sizes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

# # 存储结果
# results = []

# for test_size in test_sizes:
#     # 划分训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_2, y, test_size=test_size, random_state=42
#     )
    
#     # 训练模型
#     model = RandomForestRegressor(
#         max_features='auto',
#         min_samples_split=4,
#         n_estimators=10,
#         random_state=42
#     )
#     model.fit(X_train, y_train)
    
#     # 预测
#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)
    
#     # 计算评估指标
#     train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#     test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#     train_r2 = r2_score(y_train, y_train_pred)
#     test_r2 = r2_score(y_test, y_test_pred)
    
#     # 记录结果
#     results.append({
#         'train_size': 1 - test_size,
#         'test_size': test_size,
#         'train_rmse': train_rmse,
#         'test_rmse': test_rmse,
#         'train_r2': train_r2,
#         'test_r2': test_r2,
#         'train_samples': len(X_train),
#         'test_samples': len(X_test)
#     })

# # 转换为DataFrame以便更好查看
# results_df = pd.DataFrame(results)

# # 打印详细结果
# print("随机森林回归 - 不同训练集/测试集比例结果")
# print("=" * 80)

# for idx, row in results_df.iterrows():
#     print(f"\n配置 {idx+1}:")
#     print(f"  训练集比例: {row['train_size']:.1%} ({row['train_samples']} 个样本)")
#     print(f"  测试集比例: {row['test_size']:.1%} ({row['test_samples']} 个样本)")
#     print(f"  训练集 RMSE: {row['train_rmse']:.6f}")
#     print(f"  测试集 RMSE: {row['test_rmse']:.6f}")
#     print(f"  训练集 R²: {row['train_r2']:.4f}")
#     print(f"  测试集 R²: {row['test_r2']:.4f}")

# # 输出汇总表格
# print("\n" + "=" * 80)
# print("汇总表格:")
# print("-" * 80)
# print(f"{'训练集比例':<12} {'测试集比例':<12} {'训练集RMSE':<15} {'测试集RMSE':<15} {'RMSE差异':<12}")
# print("-" * 80)

# for idx, row in results_df.iterrows():
#     rmse_diff = row['test_rmse'] - row['train_rmse']
#     print(f"{row['train_size']:<12.1%} {row['test_size']:<12.1%} "
#           f"{row['train_rmse']:<15.6f} {row['test_rmse']:<15.6f} "
#           f"{rmse_diff:<12.6f}")


# In[11]:


# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
# import pandas as pd

# # 假设 X_1 和 y 已经定义
# X_2 = X_1[:, [0, 1, 2, 3]]

# # 定义要测试的训练集比例
# test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]

# # 存储所有结果
# all_results = []

# for test_size in test_sizes:
#     print(f"\n{'='*60}")
#     print(f"训练集比例: {1-test_size:.1%}, 测试集比例: {test_size:.1%}")
#     print(f"{'='*60}")
    
#     # 划分训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_2, y, test_size=test_size, random_state=42
#     )
    
#     # 创建模型
#     model = RandomForestRegressor(
#         max_features='auto',
#         min_samples_split=4,
#         n_estimators=10,
#         random_state=42
#     )
    
#     # 设置交叉验证策略
#     cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
#     # 进行交叉验证
#     cv_scores = cross_val_score(
#         model, 
#         X_train, 
#         y_train, 
#         cv=cv,
#         scoring='neg_mean_squared_error',
#         n_jobs=-1  # 使用所有可用的CPU核心
#     )
    
#     # 转换为RMSE
#     cv_rmse_scores = np.sqrt(-cv_scores)
#     cv_rmse_mean = np.mean(cv_rmse_scores)
#     cv_rmse_std = np.std(cv_rmse_scores)
#     cv_rmse_min = np.min(cv_rmse_scores)
#     cv_rmse_max = np.max(cv_rmse_scores)
    
#     # 输出交叉验证的详细结果
#     print(f"交叉验证 (5折) 结果:")
#     for i, score in enumerate(cv_rmse_scores):
#         print(f"  折{i+1}: RMSE = {score:.6f}")
    
#     print(f"  平均值: {cv_rmse_mean:.6f}")
#     print(f"  标准差: {cv_rmse_std:.6f}")
#     print(f"  最小值: {cv_rmse_min:.6f}")
#     print(f"  最大值: {cv_rmse_max:.6f}")
    
#     # 训练最终模型
#     model.fit(X_train, y_train)
    
#     # 在测试集上预测
#     y_test_pred = model.predict(X_test)
    
#     # 计算测试集的评估指标
#     test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#     test_r2 = r2_score(y_test, y_test_pred)
    
#     print(f"\n独立测试集结果:")
#     print(f"  RMSE: {test_rmse:.6f}")
#     print(f"  R²: {test_r2:.4f}")
    
#     # 存储结果
#     all_results.append({
#         'train_size': 1 - test_size,
#         'test_size': test_size,
#         'cv_rmse_mean': cv_rmse_mean,
#         'cv_rmse_std': cv_rmse_std,
#         'cv_rmse_min': cv_rmse_min,
#         'cv_rmse_max': cv_rmse_max,
#         'test_rmse': test_rmse,
#         'test_r2': test_r2
#     })

# # 创建结果DataFrame
# results_df = pd.DataFrame(all_results)

# print(f"\n{'='*80}")
# print("所有配置的结果汇总:")
# print('='*80)
# print(f"{'训练集比例':<12} {'测试集比例':<12} {'CV平均RMSE':<15} {'CV标准差':<12} {'测试集RMSE':<15} {'测试集R²':<10}")
# print('-'*80)

# for _, row in results_df.iterrows():
#     print(f"{row['train_size']:<12.1%} {row['test_size']:<12.1%} "
#           f"{row['cv_rmse_mean']:<15.6f} {row['cv_rmse_std']:<12.6f} "
#           f"{row['test_rmse']:<15.6f} {row['test_r2']:<10.4f}")


# In[12]:


import pandas as pd

# 训练集坐标输出
train_coords = pd.DataFrame({
    "True_Value": y_train_1,      # 横坐标
    "Predicted_Value": y_train_pred  # 纵坐标
})
train_coords.to_csv("D:\\DATASET9\\fig\\train_coordinates.csv", index=False)

# 测试集坐标输出
test_coords = pd.DataFrame({
    "True_Value": y_test_1,       # 横坐标
    "Predicted_Value": y_test_pred  # 纵坐标
})
test_coords.to_csv("D:\\DATASET9\\fig\\test_coordinates.csv", index=False)


# In[13]:


import numpy as np
from sklearn.base import clone

# 假设 X_2 是已经标准化的特征数据，y 是目标数据
# 用来存储训练过的模型的列表
models = []

for i in range(1000):
    # 自举法抽样
    sample_indices = np.random.choice(len(X_2), size=71, replace=True)
    X_sample = X_2[sample_indices]
    y_sample = y[sample_indices]
    
    # 克隆原始模型
    model = clone(model_2)
    
    # 训练模型
    model.fit(X_sample, y_sample)
    
    # 将训练好的模型存储
    models.append(model)

# 读取新数据集
new_df = pd.read_excel('D:\\DATASET9\\contents_TiZrNb.xlsx')
array_1 = new_df.values
# 提取新数据集中的特征
new_features = array_1[:,9:21].astype(float) 
new_features_1 = scaler.transform(new_features)
new_features_2=new_features_1[:,[1,3,4,9]]
# 使用模型进行预测并计算均值和方差
predictions = np.array([model.predict(new_features_2) for model in models])
print(predictions.shape)
# 对于每个特征组合，计算1000个模型的预测结果的均值和方差
mean_predictions = predictions.mean(axis=0)
variance_predictions = predictions.var(axis=0)
std_predictions = predictions.std(axis=0)
# 将均值和方差添加到new_df中作为新列
new_df['Mean_Prediction'] = mean_predictions
new_df['Variance_Prediction'] = variance_predictions
new_df['Std_Prediction'] = std_predictions
# 保存新的DataFrame到Excel文件
new_df.to_excel('D:\\DATASET9\\contents_ZrNb_pre.xlsx', index=False)


# In[14]:


import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
# 读取Excel文件
df = pd.read_excel('D:\\DATASET9\\contents_ZrNb_pre.xlsx')

# Assuming the columns for mu and sigma are indeed the 16th and 17th columns
mu = df.iloc[:, 21]  # Assuming mu is in the 16th column
sigma = df.iloc[:, 23]  # Assuming sigma is in the 17th column

# 已知观测点的最佳数据
y_best = 6.6

# 定义计算期望改进的函数
def expected_improvement(mu, sigma, y_best):
    Z = (mu - y_best) / sigma
    return (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)

# 计算每个样本点的EI
df['EI'] = expected_improvement(mu, sigma, y_best)
# 将结果保存到新的Excel文件
df.to_excel('D:\\DATASET9\\EI_w_0.xlsx', index=False)


# In[15]:


import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
file_path = 'D:\\DATASET9\\EI_w_0.xlsx'
df = pd.read_excel(file_path)

# 提取wrec列
wrec_column = df['EI']

# 找到最大的前1个值
top_1_values = wrec_column.nlargest(1)

# 显示结果
print(top_1_values.index)
print(top_1_values)


# In[ ]:




