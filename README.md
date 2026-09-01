# 三维不可压缩浸入边界 GPU 求解器：2026 年 8 月快照

> [!IMPORTANT]
> **2026 年 8 月研究进展快照**<br>
> 本仓库固定保存截至 2026 年 8 月 31 日的阶段性工作，只包含核心求解器代码和一份中文论文工作稿 PDF。仓库发布后将归档，不再持续更新；后续研究进展不会回写到本快照。

**研究者：Zhaoyue Xu**<br>
**版本：`v2026.08-snapshot`**

## 快照性质

这是用于标记研究进度的开发快照，不是经过同行评议的论文版本，也不是完整的可复现数据包。仓库不包含算例、曲面或体网格、检查点、流场结果、绘图数据、超算作业文件及第三方资产。

## 仓库内容

- [`paper/IB_GPU_Solver_Paper_CN_JCP.pdf`](paper/IB_GPU_Solver_Paper_CN_JCP.pdf)：截至 2026 年 8 月 31 日的中文论文工作稿；
- [`make_euler_mesh.py`](make_euler_mesh.py)：静态嵌套笛卡尔网格生成器；
- [`solver/backend.py`](solver/backend.py)：NumPy/CuPy 数组后端；
- [`solver/ib_core_np.py`](solver/ib_core_np.py)：离散几何与相容算子；
- [`solver/ib_driver_np.py`](solver/ib_driver_np.py)：时间推进、边界条件与离散流函数线性求解；
- [`solver/ib_immersed.py`](solver/ib_immersed.py)：欧拉—拉格朗日插值、力铺展与直接强迫；
- [`solver/ib_kinematics.py`](solver/ib_kinematics.py)：规定运动学接口；
- [`solver/ib_sparse.py`](solver/ib_sparse.py)：稀疏复合算子组装；
- [`solver/ib_run.py`](solver/ib_run.py)：通用命令行入口。

## 数值方法概述

求解器面向三维不可压缩流动，以边上的离散流函数自由度为主变量，通过离散旋度构造面通量，并由有限体积几何重构得到控制体中心速度。浸入边界采用基于正则化离散 \(\delta\) 函数的直接强迫。代码使用同一套数组级实现支持 NumPy 与 CuPy 后端，并保留静态嵌套网格、非周期外边界和规定运动边界所需的离散结构。

## 环境

CPU 路径的基础依赖见 [`requirements.txt`](requirements.txt)。GPU 路径另需与本机 CUDA 版本匹配的 CuPy；示例依赖文件见 [`requirements-gpu.txt`](requirements-gpu.txt)。

```bash
python -m pip install -r requirements.txt
python solver/ib_run.py --dx 0.125 --steps 10 --re 100 --sparse
```

GPU 路径可在安装兼容的 CuPy 后运行：

```bash
python -m pip install -r requirements-gpu.txt
python solver/ib_run.py --gpu --dx 0.125 --steps 10 --re 100
```

上述命令只用于检查程序入口和运行环境；正式论文算例与原始计算数据不在本仓库中。

## 数据与复现边界

本快照有意不发布任何算例或结果数据，因此不能仅依靠本仓库重新生成论文全部图表。论文中涉及的扑翼几何、参考 Fortran 输出和其他可能受第三方权利约束的资产均未收录。

## 权利声明

本快照未附开源许可证。除非权利人另行书面授权，公开可见不表示授予复制、修改、再分发或商业使用许可。论文工作稿仅用于展示截至 2026 年 8 月 31 日的研究进展。

## 版本留痕

仓库唯一计划版本为 `v2026.08-snapshot`。发布时的标签、提交哈希和 GitHub Release 将共同对应同一份文件集合；完成核验后仓库将设为 Archived。
