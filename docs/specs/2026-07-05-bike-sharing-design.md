# Bike Sharing Demand — 项目设计文档

> 日期：2026-07-05 | 状态：设计中

## 1. 项目概述

Kaggle Bike Sharing Demand 竞赛。基于华盛顿 DC 的 Capital Bikeshare 租车数据，预测每小时的自行车租用量。

- **任务类型**：回归（时间序列）
- **数据集**：训练 10,886 条 / 测试 6,493 条（小时级，2 年跨度）
- **评估指标**：RMSLE（Root Mean Squared Logarithmic Error）
- **目标**：Kaggle 提交分数进入前 20%

### 与 Titanic 项目的差异化

| 维度 | Titanic | Bike Sharing |
|------|---------|---------------|
| 任务 | 二分类 | 回归 |
| 数据结构 | 横截面 | 时间序列 |
| 核心技能 | 特征编码 + 集成 | 时间特征工程 + 时序 CV |
| 叙事 | 阶层与幸存 | 城市出行规律 |

---

## 2. 项目结构

```
bike-sharing-analysis/
├── README.md
├── bike-sharing-analysis.ipynb    # 8 章完整分析
├── run.py                         # 独立可复现脚本
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── train.csv                  # 10,886 条
│   ├── test.csv                   # 6,493 条
│   └── sampleSubmission.csv
├── output/
│   ├── submission.csv
│   ├── kaggle-score.png
│   └── figures/
└── docs/
    └── specs/
        └── 2026-07-05-bike-sharing-design.md
```

---

## 3. 章节规划（8 章）

### 第 1 章 · 问题定义
- 竞争背景：共享单车运营商需要预测车辆调配
- RMSLE 指标解读：为什么用对数误差 → 对大值和小值误差同等对待
- 分析视角：从租车数据中看到一座城市的作息规律

### 第 2 章 · EDA
- datetime 字段分解 → 小时/天/月/季节模式可视化
- `count` 的时间分布：早高峰/晚高峰/周末
- `casual` vs `registered` 行为差异（这是建模策略的关键依据）
- 天气变量与租车量的关系：temp vs atemp 共线性、极端天气的影响
- 异常值检测：windspeed=0 的比例、humidity 极端值

### 第 3 章 · 特征工程
常规特征：
- 时间分解：hour, weekday, month, season, year
- 时段标记：day_period（凌晨/上午/下午/晚上/深夜）
- 高峰标记：is_rush_morning(7-9), is_rush_evening(17-19)
- 周末标记：is_weekend

交互特征：
- hour × workingday（工作日和周末的小时模式完全不同）
- hour × season（夏季和冬季的高峰形状不同）
- temp × humidity（体感温度效果）

滚动/滞后特征：不引入。原因是测试集没有历史 count，迭代预测会放大误差且显著增加复杂度。时间分解 + 交互特征已能捕获大部分周期性模式，收益/成本比不划算

### 第 4 章 · 数据清洗
- windspeed=0 的处理（可能是传感器故障，用天气条件分组中位数填充）
- humidity 异常值筛选
- temp 和 atemp 的相关性分析（决定是否删除一个）
- 确认：本数据集无缺失值

### 第 5 章 · Baseline 建模
- RMSLE 评估函数实现
- 目标变换：`y = np.log1p(count)`（RMSLE 等价于 log 空间的 RMSE）
- 时间序列交叉验证：`TimeSeriesSplit(n_splits=5)`
- 基础模型：Ridge、RandomForest、XGBoost、LightGBM
- 重点对比：直接预测 count vs 分头预测 casual/registered

### 第 6 章 · 深度调参
- 对 XGBoost 和 LightGBM 执行 RandomizedSearchCV
- 超参搜索空间：
  - XGBoost：n_estimators, max_depth, learning_rate, subsample, reg_alpha, reg_lambda, min_child_weight
  - LightGBM：n_estimators, max_depth, num_leaves, learning_rate, subsample, reg_alpha, reg_lambda
- 使用 TimeSeriesSplit 作为交叉验证
- 记录调参前后 RMSLE 变化

### 第 7 章 · 集成与 Stacking
- 两级集成策略：
  1. 基础层：XGBoost + LightGBM + RandomForest + Ridge（调参后最优版本）
  2. Meta 层：Ridge 回归作为 blender
- 关键对比实验：
  - 直接预测 count（单一目标）
  - 分头预测 casual + registered 再求和（双目标）
  - 分头预测 + stacking（双目标集成）
- 选择 RMSLE 最优策略生成最终提交

### 第 8 章 · 复盘与总结
- 最优模型和策略总结
- 特征重要性排名 + 业务解读
- SHAP 分析（如适用）
- 城市出行洞察：
  - 通勤族 vs 休闲族的骑行行为差异
  - 天气有多大的影响
  - 时间比天气更重要吗？
- 后续改进方向

---

## 4. 建模策略

### 4.1 目标变换

RMSLE 的定义：
```
RMSLE = sqrt(1/n * Σ (log(y_pred + 1) - log(y_true + 1))²)
```

这等价于先在 log 空间做回归，再 expm1 还原。因此：
- 训练时：`y = np.log1p(count)`，用 RMSE 优化
- 预测时：`pred = np.expm1(model.predict(X))` 还原
- 分头预测时：`y_casual = np.log1p(casual)`, `y_registered = np.log1p(registered)`

### 4.2 分头预测策略

casual 和 registered 用户画像：
- **casual**：周末、下午、天气好 → 休闲型
- **registered**：工作日、早晚高峰 → 通勤型

分开建模让每个模型专注各自的模式。最后：
```
pred_count = expm1(pred_log_casual) + expm1(pred_log_registered)
```

### 4.3 时间序列 CV

使用 `TimeSeriesSplit(n_splits=5)`，按时间顺序划分：

```
Fold 1: Train[0:2177]  → Val[2177:4354]
Fold 2: Train[0:4354]  → Val[4354:6531]
Fold 3: Train[0:6531]  → Val[6531:8708]
Fold 4: Train[0:8708]  → Val[8708:10886]
Fold 5: Train[0:10886] → (hold-out test set)
```

不能随机打乱——用历史预测未来才能模拟真实上线场景。

### 4.4 Stacking 架构

```
                    ┌──────────────┐
                    │   Test Data  │
                    └──────┬───────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │  XGBoost   │  │ LightGBM   │  │    RF      │
    │ (tuned)    │  │ (tuned)    │  │ (tuned)    │
    └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
          │               │               │
          ▼               ▼               ▼
    ┌─────────────────────────────────────────────┐
    │         Meta Layer: Ridge Regression        │
    │         (blend base predictions)            │
    └────────────────────┬────────────────────────┘
                         │
                         ▼
                 ┌──────────────┐
                 │  Submission  │
                 └──────────────┘
```

---

## 5. 技术栈

| 用途 | 工具 |
|------|------|
| 数据操作 | pandas, numpy |
| 可视化 | matplotlib, seaborn |
| 中文渲染 | SimHei 字体，fontweight='normal' |
| 时间处理 | pandas datetime 属性 |
| 基础模型 | sklearn: Ridge, RandomForestRegressor |
| 梯度提升 | XGBoost, LightGBM |
| 集成 | sklearn.ensemble.StackingRegressor |
| 交叉验证 | sklearn.model_selection.TimeSeriesSplit |
| 超参搜索 | sklearn.model_selection.RandomizedSearchCV |
| 模型解释 | SHAP |
| 缺失值可视化 | missingno |

---

## 6. 输出质量标准

### Notebook
- 每个操作都有解释：为什么这样做、不这样做会怎样
- 图表标题/注释用中文（fontweight='normal' 避免 tofu）
- 关键决策点有对比实验（例如分头 vs 不分头的 RMSLE 对比）
- 代码和文字比例约 4:6（偏分析叙事）

### README
- 核心发现用具体数字说话，不要空洞的宣传语言
- 流程图清晰展示分析步骤
- 避免 AI 痕迹：不出现"此外""值得注意的是""起到了至关重要的作用"等词汇
- 用 humanizer-zh 做最终润色

### 可复现性
- `run.py` 从原始数据到 submission.csv 一步跑通
- `requirements.txt` 锁定依赖版本

---

## 7. 验收标准

1. Kaggle 提交分数 ≥ 目标（Public LB RMSLE < 0.4 为合格，< 0.35 为优秀）
2. Notebook Run All 无报错，图表中文正常显示
3. README 通过 humanizer-zh 审核，无 AI 痕迹
4. Git 历史干净，无 Co-Authored-By 等 AI 标记
5. 分头预测策略有明确的对比实验证明其优于直接预测
