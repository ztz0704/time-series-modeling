# 时间序列学习仓库

这个仓库只用于学习时间序列：按课程顺序保存练习代码，用小实验验证概念，并复现经典案例。它不是一个待发布的业务项目。

## 目录结构

```text
time-series-modeling/
├── lessons/          # 按周、按天排列的学习脚本
│   ├── week_01/      # 平稳性、频域、MA、AR、ARMA、ARIMA
│   └── week_02/      # SARIMA、滚动样本外评估、状态空间
├── practice/         # 独立的小型练习与对照实验
├── case_studies/     # 教材或经典数据案例复现
├── data/             # 原始数据与处理后的数据
├── results/          # 脚本生成的图片和表格
├── src/time_series/  # 学习过程中沉淀的可复用函数
└── tests/            # 可复用函数的测试
```

## 建议学习顺序

1. 依次阅读并运行 `lessons/week_01/` 和 `lessons/week_02/` 中的脚本。
2. 用 `practice/` 中的小实验巩固预测基线和评估方法。
3. 在 `case_studies/` 中把完整流程应用到真实数据。
4. 把重复出现的实现整理进 `src/time_series/`，并在 `tests/` 中补测试。

## 运行方式

安装依赖：

```powershell
uv sync
```

运行单个学习脚本：

```powershell
uv run python lessons/week_01/day_01_foundations.py
```

运行练习或案例：

```powershell
uv run python practice/day_01_baselines.py
uv run python case_studies/box_jenkins/series_g/day_01_series_g_replication.py
```

运行测试：

```powershell
uv run pytest -q
```

脚本产生的图片和表格统一写入 `results/`。原始数据保存在 `data/raw/`，处理后的数据保存在 `data/processed/`。
