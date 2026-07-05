"""Bike Sharing Demand — 独立可复现脚本，生成 submission.csv"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

# ==================== 加载数据 ====================
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

print(f'训练集: {train.shape}, 测试集: {test.shape}')

# ==================== 合并构造特征 ====================
all_data = pd.concat([train.drop(['casual', 'registered', 'count'], axis=1), test],
                     axis=0, sort=False, ignore_index=True)

# 时间分解
all_data['hour'] = all_data['datetime'].dt.hour
all_data['weekday'] = all_data['datetime'].dt.dayofweek
all_data['month'] = all_data['datetime'].dt.month
all_data['year'] = all_data['datetime'].dt.year
all_data['day'] = all_data['datetime'].dt.day

# 周末标记
all_data['is_weekend'] = (all_data['weekday'] >= 5).astype(int)

# 时段分类
def classify_period(h):
    if h <= 5: return 0
    elif h <= 11: return 1
    elif h <= 17: return 2
    else: return 3
all_data['day_period'] = all_data['hour'].apply(classify_period)

# 高峰标记
all_data['is_rush_morning'] = ((all_data['hour'] >= 7) & (all_data['hour'] <= 9) &
                                (all_data['workingday'] == 1)).astype(int)
all_data['is_rush_evening'] = ((all_data['hour'] >= 17) & (all_data['hour'] <= 19) &
                               (all_data['workingday'] == 1)).astype(int)

# 交互特征
all_data['hour_x_workingday'] = all_data['hour'] * all_data['workingday']
all_data['hour_x_season'] = all_data['hour'] * all_data['season']
all_data['temp_x_humidity'] = all_data['temp'] * all_data['humidity']

# ==================== 数据清洗 ====================
# windspeed=0 填充
for weather_type in all_data['weather'].unique():
    mask = (all_data['weather'] == weather_type) & (all_data['windspeed'] == 0)
    median_wind = all_data.loc[(all_data['weather'] == weather_type) & (all_data['windspeed'] > 0), 'windspeed'].median()
    if pd.notna(median_wind):
        all_data.loc[mask, 'windspeed'] = median_wind

# humidity=0 填充
for weather_type in all_data['weather'].unique():
    mask = (all_data['weather'] == weather_type) & (all_data['humidity'] == 0)
    median_h = all_data.loc[(all_data['weather'] == weather_type) & (all_data['humidity'] > 0), 'humidity'].median()
    if pd.notna(median_h):
        all_data.loc[mask, 'humidity'] = median_h

# ==================== 特征列表 ====================
features = [
    'season', 'holiday', 'workingday', 'weather',
    'temp', 'humidity', 'windspeed',
    'hour', 'weekday', 'month', 'year', 'day',
    'is_weekend', 'day_period',
    'is_rush_morning', 'is_rush_evening',
    'hour_x_workingday', 'hour_x_season',
    'temp_x_humidity',
]

# ==================== 数据准备 ====================
train_len = len(train)
train_processed = all_data[:train_len].copy()
test_processed = all_data[train_len:].copy()

y_count = np.log1p(train['count'].values)
y_casual = np.log1p(train['casual'].values)
y_registered = np.log1p(train['registered'].values)

X = train_processed[features].values
X_test = test_processed[features].values

print(f'X: {X.shape}, X_test: {X_test.shape}')

# ==================== CV 配置 ====================
tscv = TimeSeriesSplit(n_splits=5)

def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

def rmsle_metric(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

# ==================== XGBoost 调参 (casual) ====================
print('\n=== XGBoost casual ===')
xgb_params = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_alpha': uniform(0, 2),
    'reg_lambda': uniform(0.5, 5),
    'min_child_weight': randint(1, 10),
}
xgb_c = RandomizedSearchCV(
    XGBRegressor(random_state=42, verbosity=0), xgb_params,
    n_iter=50, cv=tscv, scoring='neg_root_mean_squared_error',
    random_state=42, n_jobs=-1, verbose=0
).fit(X, y_casual)
print(f'Best RMSLE: {-xgb_c.best_score_:.4f}')

# XGBoost 调参 (registered)
print('=== XGBoost registered ===')
xgb_r = RandomizedSearchCV(
    XGBRegressor(random_state=42, verbosity=0), xgb_params,
    n_iter=50, cv=tscv, scoring='neg_root_mean_squared_error',
    random_state=42, n_jobs=-1, verbose=0
).fit(X, y_registered)
print(f'Best RMSLE: {-xgb_r.best_score_:.4f}')

# ==================== LightGBM 调参 ====================
print('\n=== LightGBM casual ===')
lgb_params = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 12),
    'num_leaves': randint(20, 100),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_alpha': uniform(0, 2),
    'reg_lambda': uniform(0.5, 5),
    'min_child_samples': randint(10, 50),
}
lgb_c = RandomizedSearchCV(
    LGBMRegressor(random_state=42, verbose=-1), lgb_params,
    n_iter=50, cv=tscv, scoring='neg_root_mean_squared_error',
    random_state=42, n_jobs=-1, verbose=0
).fit(X, y_casual)
print(f'Best RMSLE: {-lgb_c.best_score_:.4f}')

print('=== LightGBM registered ===')
lgb_r = RandomizedSearchCV(
    LGBMRegressor(random_state=42, verbose=-1), lgb_params,
    n_iter=50, cv=tscv, scoring='neg_root_mean_squared_error',
    random_state=42, n_jobs=-1, verbose=0
).fit(X, y_registered)
print(f'Best RMSLE: {-lgb_r.best_score_:.4f}')

# ==================== RF 调参 ====================
print('\n=== RF casual ===')
rf_params = {
    'n_estimators': randint(100, 400),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 15),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
}
rf_c = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1), rf_params,
    n_iter=30, cv=tscv, scoring='neg_root_mean_squared_error',
    random_state=42, n_jobs=-1, verbose=0
).fit(X, y_casual)
print(f'Best RMSLE: {-rf_c.best_score_:.4f}')

print('=== RF registered ===')
rf_r = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1), rf_params,
    n_iter=30, cv=tscv, scoring='neg_root_mean_squared_error',
    random_state=42, n_jobs=-1, verbose=0
).fit(X, y_registered)
print(f'Best RMSLE: {-rf_r.best_score_:.4f}')

# ==================== Stacking 集成 ====================
print('\n=== Stacking Ensemble ===')
stack_casual = StackingRegressor(
    estimators=[('xgb', xgb_c.best_estimator_), ('lgb', lgb_c.best_estimator_), ('rf', rf_c.best_estimator_)],
    final_estimator=Ridge(alpha=1.0, random_state=42),
    cv=tscv, n_jobs=-1
)
stack_registered = StackingRegressor(
    estimators=[('xgb', xgb_r.best_estimator_), ('lgb', lgb_r.best_estimator_), ('rf', rf_r.best_estimator_)],
    final_estimator=Ridge(alpha=1.0, random_state=42),
    cv=tscv, n_jobs=-1
)

# CV 评估
scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_val = X[train_idx], X[val_idx]
    stack_casual.fit(X_tr, y_casual[train_idx])
    stack_registered.fit(X_tr, y_registered[train_idx])
    pred_total = np.expm1(stack_casual.predict(X_val)) + np.expm1(stack_registered.predict(X_val))
    score = rmsle(np.expm1(y_count[val_idx]), pred_total)
    scores.append(score)
    print(f'  Fold{fold+1}: RMSLE={score:.4f}')
print(f'  CV Avg RMSLE: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})')

# ==================== 全量训练 + 生成提交 ====================
print('\n=== 全量训练，生成提交文件 ===')
stack_casual.fit(X, y_casual)
stack_registered.fit(X, y_registered)

test_pred = np.expm1(stack_casual.predict(X_test)) + np.expm1(stack_registered.predict(X_test))
test_pred = np.maximum(test_pred, 0)

submission = pd.DataFrame({
    'datetime': test['datetime'],
    'count': test_pred
})
submission.to_csv('output/submission.csv', index=False)

print(f'预测范围: {test_pred.min():.0f} ~ {test_pred.max():.0f}')
print(f'预测均值: {test_pred.mean():.0f}, 中位数: {np.median(test_pred):.0f}')
print(f'提交文件: output/submission.csv ({len(submission)} 条)')
