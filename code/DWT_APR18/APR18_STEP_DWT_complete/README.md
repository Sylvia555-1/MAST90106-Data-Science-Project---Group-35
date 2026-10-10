# APR18 STEP DWT 完整实验代码

本包对应已汇报的 APR18 0.1 → 1 s⁻¹ 实验，真实数据范围为 steps!BH4:BK68，共65个原始点。
包含完整 Python 代码、英文代码注释、原始输入、复现所需人造对照、检查结果和四张结果图。

## 运行

使用 Python 3.12（本次核对环境），进入解压后的目录执行：

```bash
python -m pip install -r requirements.txt
python run_experiment.py
```

运行会重新生成本目录下 01_source_audit 至 05_comparison 的结果，不修改 inputs 中的输入文件。
代码使用相对路径，可以移动整个文件夹。首次完整运行必须从 run_experiment.py 开始，以更新来源记录。

## 建议阅读顺序

1. `run_experiment.py`：按顺序执行完整实验。
2. `01_audit_sources.py`：核对原始Excel、采样时间、重复表格，导出原始点。
3. `dwt_features.py`：DWT重构、窗口特征、边界标记和特征比较。
4. `02_extract_and_compare.py`：真实/人造数据特征提取、网格加密和对齐/边界敏感性检查。
5. `03_plot_results.py`：生成四张结果图和结果摘要。
6. `RESULTS_ZH.md`：原实验的中文结果解释。

## 方法范围

- 保留全部65个原始点作为线性插值依据；不做幅值归一化或阈值去噪。
- 离线 db2 DWT，早期 A5、R6/R7 与后期 A13 等特征。
- 2倍/4倍网格加密、不同边界方式和分析起点检查。
- 包含人造 STEP 的尺度匹配和稀疏采样诊断，因此 inputs 中也有必需的人造数据。
- A 是重构近似分量的窗口均值；R 是重构细节的窗口RMS；rate是相邻窗口均值的变化率。
- `_interior` 是工程边界筛选标记，并非独立验证的事件真值。

当前没有变化点自动检测器或实测准确率，也没有将细节直接判断为噪声的方法。

## 目录

- `inputs/`：APR18原始Excel、运行所需3个人造CSV和manifest、历史特征对照。
- `01_source_audit/`：原始点、表格清单、采样支持信息。
- `02_features/`：应力和剪切率特征、人造对照特征。
- `03_checks/`：重构、插值保真、加密、边界及对齐检查。
- `04_figures/`：四张结果图。
- `05_comparison/`：尺度对应、区间摘要和采样诊断。
- `original_code/`：未修改的原始脚本，保留历史绝对路径，不作为便携运行入口。

便携版仅调整两个脚本的输入路径，计算方法没有改动。历史 notebook 仅作为原实验阅读材料，推荐使用上述脚本入口。
