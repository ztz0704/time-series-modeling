# time-series-modeling：时间序列学习与建模

按课程顺序整理时间序列学习脚本，用模拟实验理解模型，并复现 Box–Jenkins 经典案例。内容涵盖平稳性、频域分析、AR/MA/ARMA/ARIMA、季节模型、滚动预测、状态空间与 Kalman 滤波。

## 实际目录结构

下面列出当前 Git 仓库中的文件。本地的虚拟环境、工具缓存和编辑器设置由 `.gitignore` 管理。

```text
time-series-modeling/
├── .gitignore
├── .python-version
├── README.md
├── pyproject.toml
├── uv.lock
├── lessons/
│   ├── week_01/
│   │   ├── day_01_foundations.py               # 平稳性、样本 ACF、差分与 ADF
│   │   ├── day_02_box_jenkins_series_f.py       # Series F：AR(2) 拟合、预测与残差诊断
│   │   ├── day_03_frequency_analysis.py        # 频域分析
│   │   ├── day_03_compare_spectra.py           # 不同过程的谱比较
│   │   ├── day_03_welch_spectrum.py            # 周期图与 Welch 谱估计
│   │   ├── day_04_ma1.py                      # MA(1) 模拟与可逆性
│   │   ├── day_05_ar.py                       # AR(2) 模拟与平稳性
│   │   ├── day_06_arma.py                     # ARMA 模拟、拟合与诊断
│   │   └── day_07_arima.py                    # ARIMA(0,1,1) 与预测
│   └── week_02/
│       ├── day_08_sarima.py                   # 季节过程与 SARIMAX
│       ├── day_11_rolling_oos.py              # 滚动预测与样本外评估
│       ├── day_12_state_space.py              # 状态空间模型
│       ├── day_13_kalman.py                   # Kalman 滤波递推
│       ├── day_14_kalman2.py                  # Kalman 滤波与似然参数估计
│       └── day_15_exogenous.py                # 外生变量与动态预测
├── practice/
│   └── day_01_baselines.py                    # 模拟季节序列的三种基线比较
├── case_studies/
│   └── box_jenkins/
│       └── series_g/
│           ├── day_01_series_g_replication.py # 航空旅客数据变换与基线预测
│           └── findings.md                   # 案例学习记录
├── data/
│   ├── raw/
│   │   └── box_jenkins/
│   │       └── series_g_airline.csv
│   └── processed/
│       └── box_jenkins/
│           └── series_g_log.csv
├── results/
│   ├── figures/
│   │   ├── day_01_baseline_comparison.png
│   │   ├── day_03_spectra.png
│   │   ├── day_03_time_paths.png
│   │   ├── series_g_baseline_forecasts.png
│   │   ├── series_g_log.png
│   │   └── series_g_raw.png
│   └── tables/
│       ├── day_01_baseline_forecasts.csv
│       ├── day_01_baseline_metrics.csv
│       ├── series_g_baseline_forecasts.csv
│       └── series_g_baseline_metrics.csv
├── src/
│   └── time_series/
│       ├── __init__.py
│       ├── simulation/
│       │   ├── __init__.py
│       │   └── synthetic.py                  # 季节随机游走模拟
│       ├── models/
│       │   ├── __init__.py
│       │   └── baselines.py                  # 均值、朴素与季节朴素预测
│       └── evaluation/
│           ├── __init__.py
│           ├── split.py                      # 按时间顺序划分训练集与测试集
│           └── metrics.py                    # MAE、RMSE、MASE 与批量评估
└── tests/
    ├── test_baselines.py
    ├── test_metrics.py
    └── test_split.py
```

课程按 `week_01` 和 `week_02` 组织，`day_03` 包含频域分析、谱比较和 Welch 谱估计三个实验。

## 学习入口

1. 从 [day_01_foundations.py](lessons/week_01/day_01_foundations.py) 开始，观察不同生成过程的路径、ACF、差分和 ADF 检验结果。
2. 阅读 [Series F 案例](lessons/week_01/day_02_box_jenkins_series_f.py)，再学习 `day_03` 的三个频域实验，以及 `day_04` 到 `day_08` 的模型脚本。
3. 按 `day_11` 到 `day_15` 的顺序学习滚动预测、状态空间、Kalman 滤波和外生变量。
4. 运行 [基线练习](practice/day_01_baselines.py) 和 [Series G 案例](case_studies/box_jenkins/series_g/day_01_series_g_replication.py)，结合预测图、误差表和 [案例记录](case_studies/box_jenkins/series_g/findings.md) 复盘。

## 环境与运行

项目使用 `uv` 管理依赖，`.python-version` 指定 Python 3.12。以下命令均在仓库根目录执行。

安装项目及开发依赖：

```powershell
uv sync --dev
```

运行课程脚本，例如基础实验或 Kalman 滤波：

```powershell
uv run python lessons/week_01/day_01_foundations.py
uv run python lessons/week_02/day_13_kalman.py
```

运行基线练习和 Series G 案例：

```powershell
uv run python practice/day_01_baselines.py
uv run python case_studies/box_jenkins/series_g/day_01_series_g_replication.py
```

运行公共函数的测试：

```powershell
uv run pytest -q
```

课程脚本以终端输出和交互式图形为主，使用 `plt.show()` 的脚本需要关闭图形窗口后继续执行。基线练习和 Series G 案例会把图表保存到 `results/figures/`，把预测值和指标保存到 `results/tables/`；Series G 案例还会写入 `data/processed/box_jenkins/series_g_log.csv`。

## 数据与已有结果

Series F 的数据直接写在对应课程脚本中。Series G 使用仓库内的航空旅客月度数据，共 144 条观测，覆盖 1949 年 1 月至 1960 年 12 月；案例脚本读取本地 CSV，并使用最后 24 个月作为测试集。

| 内容 | 文件 |
|---|---|
| Series G 原始数据 | [series_g_airline.csv](data/raw/box_jenkins/series_g_airline.csv) |
| Series G 对数变换数据 | [series_g_log.csv](data/processed/box_jenkins/series_g_log.csv) |
| 模拟序列基线指标 | [day_01_baseline_metrics.csv](results/tables/day_01_baseline_metrics.csv) |
| 模拟序列预测对比图 | [day_01_baseline_comparison.png](results/figures/day_01_baseline_comparison.png) |
| Series G 基线指标 | [series_g_baseline_metrics.csv](results/tables/series_g_baseline_metrics.csv) |
| Series G 预测对比图 | [series_g_baseline_forecasts.png](results/figures/series_g_baseline_forecasts.png) |

`results/` 已保存基线比较、Series G 和频域分析的图表，可直接查看。案例中的数据变换、季节性和预测误差讨论见 [findings.md](case_studies/box_jenkins/series_g/findings.md)。
