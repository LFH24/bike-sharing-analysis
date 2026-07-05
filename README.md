# Bike Sharing Demand：从时间序列看城市出行规律

基于 Kaggle Bike Sharing Demand 数据集，走了一遍完整的时间序列回归分析流程，包括 EDA、时间特征工程、异常值处理、分头建模和 Stacking 集成。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-0.39970-orange)](https://www.kaggle.com/competitions/bike-sharing-demand)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 核心发现

华盛顿 DC 两年（2011-2012）的共享单车数据里，影响租车量的几个主要因素：

- **时间是最大的驱动力** — 工作日早晚高峰（7-9 点、17-19 点）的租车量是深夜的 10 倍以上。但周末没有尖峰，只有一个从上午延续到傍晚的缓坡
- **通勤族和休闲族的骑行逻辑完全不同** — casual 用户（休闲）看天气和周末，registered 用户（通勤）看早晚高峰。坏天气来了，casual 骤降 60%，registered 只降 20%——通勤是刚需
- **2012 年比 2011 年增长了约 70%** — 共享单车在 DC 进入快速普及期，年份本身就是强特征
- **温度和湿度的影响有天花板** — 太热（>35°C）之后租车量反而下降。极端天气（大雨/暴雪）样本极少，模型在这个区间基本靠猜

一个具体的数字：工作日上午 8 点的注册用户平均租车量是凌晨 3 点的约 50 倍。

---

## 分析流程

```
第 1 章 · 问题定义    →  RMSLE 指标解读 + 业务视角切入
第 2 章 · EDA         →  时间模式（小时/天/月/季节） → 天气影响 → casual vs registered 行为分化
第 3 章 · 特征工程     →  时间分解 + 时段标记 + 交互特征（hour×workingday / hour×season）
第 4 章 · 数据清洗     →  windspeed=0 传感器故障修复 + humidity 异常值 + 共线性处理
第 5 章 · Baseline     →  TimeSeriesSplit + 4 模型 baseline + 直接预测 vs 分头预测对比
第 6 章 · 深度调参     →  XGBoost / LightGBM / RF RandomizedSearchCV（分头优化）
第 7 章 · 集成         →  StackingRegressor（XGB + LGB + RF → Ridge）+ 三策略对比
第 8 章 · 复盘         →  特征重要性 + SHAP + 城市出行洞察 + 框架迁移
```

跟 Titanic 项目的主要区别是引入了时间序列 CV（TimeSeriesSplit）和分头预测策略（casual / registered 分开建模），这两个技巧对任何时序回归任务都适用。

---

## 技术栈

Python、pandas、numpy、scikit-learn、XGBoost、LightGBM、StackingRegressor、matplotlib、seaborn、SHAP、scipy

---

## Kaggle 提交结果

| 指标 | v1 |
|------|-----|
| RMSLE | 0.39970 |
| 模型 | Stacking (XGB + LGB + RF) |
| CV 策略 | TimeSeriesSplit(5) |
| 特征数 | 18（含 3 个交互特征） |
| 预测策略 | 分头预测 casual + registered |

---

## 复现

```bash
# 安装依赖
pip install -r requirements.txt

# 打开 Notebook
jupyter notebook bike-sharing-analysis.ipynb

# 或者直接运行脚本
python run.py
```

---

## 文件结构

```
bike-sharing-analysis/
├── README.md
├── bike-sharing-analysis.ipynb    # 完整分析过程（8 章）
├── run.py                         # 独立可复现脚本
├── requirements.txt
├── .gitignore
├── data/
│   ├── train.csv                  # 训练集（10,886 条）
│   ├── test.csv                   # 测试集（6,493 条）
│   └── sampleSubmission.csv
└── output/
    ├── submission.csv             # 最终预测结果
    ├── kaggle-score.png           # Kaggle 提交分数截图
    └── figures/                   # 导出图表
```

---

## 从 Titanic 到这个项目

Titanic 是分类（横截面），这个是回归（时间序列）。两个项目加在一起，基本覆盖了数据分析日常碰到的两类问题。这个项目里新碰的东西——TimeSeriesSplit、分头建模、LightGBM、StackingRegressor——都是换了场景也能直接用的技术。

---

## 接下来

- 回归分析项目（Bike Sharing 是第一个，House Prices 后续上线）
- 更多数据科学项目陆续更新

---

LFH24 · 2026.07
