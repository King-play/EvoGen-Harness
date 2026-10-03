<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/readme/logo-dark.svg">
    <img src="assets/readme/logo-light.svg" width="250" alt="EvoGen">
  </picture>
</p>
<h1 align="center">EvoGen-Harness</h1>
<p align="center"><strong>Learning Where and How to Evolve Image-Generation Harnesses</strong><br>保持生成器冻结，让外部系统学会在哪里、如何演化。</p>
<p align="center"><a href="https://arxiv.org/abs/2610.00383">论文</a> · <a href="https://king-play.github.io/EvoGen-Harness/">项目网页</a> · <a href="documentation/README.md">使用文档</a> · <a href="README.md">English</a></p>

**EvoGen-Harness** 通过演化图像生成器外部的持久化系统改善生成效果，不更新模型参数。**TRACE**（Trajectory-Relative Attribution and Coordinated Evolution）聚合多次随机执行的视觉证据，定位值得修改的职责，提出局部更新，并对残余失败重新归因。只有通过验证的更新才会保留；**No-Patch** 表示不应进行持久化修改。

## 方法概览

![EvoGen-Harness 方法总览：冻结生成器、五种外部职责和 TRACE 验证闭环](assets/readme/overview.png)

图 1 来自论文。五种职责分别为 **Policy**（需求与约束）、**Tools**（能力知识）、**Skills**（可复用过程）、**Middleware**（运行编排）和 **Memory**（持久经验）。Tools 的更新不是修改工具实现或模型权重。

## 渐进修复

![不同职责的连续更新修复多个视觉约束](assets/readme/progressive-repair.jpg)

图 4 展示逐步修复多个视觉约束。更多原始案例与交互对比见[项目网页](https://king-play.github.io/EvoGen-Harness/)。

## 论文结果

以下为论文表 1–3 的报告结果，不是本发布包 CPU 测试重新测出的分数；增益是绝对分数差。

| Benchmark | 最强已评估基线 | EvoGen-Harness | 增益 |
| :--- | ---: | ---: | ---: |
| GenEval2（Overall Soft-TIFA GM） | 0.4456 | **0.7089** | **+0.2633** |
| T2I-CompBench++（八项均值） | 0.6087 | **0.6807** | **+0.0720** |
| WISE（Overall WiScore） | 0.6028 | **0.6780** | **+0.0752** |

三个表中的最强已评估基线均为 Nano Banana。默认系统使用 FLUX.1-dev、GPT-4.1 和 OWLv2 + NVILA。不同骨干的 harness 分别进行适配，不应理解为同一个最终 harness 在所有模型上零样本迁移。相同骨干对比与计算开销见论文表 4、5 和 9。

## 快速开始：先检查代码，再配置模型

在仓库根目录运行：

```bash
git clone https://github.com/King-play/EvoGen-Harness.git
cd EvoGen-Harness
conda create -n genharness python=3.11 -y
conda activate genharness
pip install --upgrade pip
pip install -r requirements.txt
```


[英文 README 的预算检查命令](README.md#quick-start)可继续检查演化流程的输入规模。真正生成图像前，请完成[模型、GPU 与 API 配置](documentation/INSTALL.md)；不要直接删除预算命令里的 `--dry-run-budget` 就开始正式实验。

## 从使用到评估

| 需求 | 文档 |
| :--- | :--- |
| 安装完整依赖，设置模型路径和 LLM 服务 | [INSTALL.md](documentation/INSTALL.md) |
| 复制 harness 工作目录、执行 TRACE、保存和校验快照 | [EVOLUTION.md](documentation/EVOLUTION.md) |
| 三个 benchmark 的提示词准备、生成、导出与独立评估 | [EVALUATION.md](documentation/EVALUATION.md) |
| 核心模块与五种职责的边界 | [ARCHITECTURE.md](documentation/ARCHITECTURE.md) |
| 本次实际交付材料与实验记录的对应关系 | [REPRODUCIBILITY.md](documentation/REPRODUCIBILITY.md) |
| 保留网页，上传代码并替换仓库 README | [UPLOAD_GITHUB_zh.md](UPLOAD_GITHUB_zh.md) |

`examples/visual_harness/` 是随源代码提供的配置目录，本包没有将其声明为历史论文最终评估快照。快照工具保存的是你明确选中的真实目录，不能凭空生成过去实验的结果。

## 引用

```bibtex
@misc{luo2026evogenharness,
  title         = {{EvoGen-Harness}: Learning Where and How to Evolve Image-Generation Harnesses},
  author        = {Jiabin Luo and Yinan Liu and Chunlei Meng and Yufei Guo},
  year          = {2026},
  eprint        = {2610.00383},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.00383}
}
```

## 许可与贡献

代码保留原始 [MIT License](LICENSE)。第三方数据与模型使用各自许可，不因项目代码开源而变为 MIT，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。贡献代码前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，不要在公开 Issue 中提交密钥、私人路径或未授权公开的材料。
