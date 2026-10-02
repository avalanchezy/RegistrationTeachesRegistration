# 第二轮打磨：把 KBS / MedIA 主张变成可证伪实验

**当前不能认定达到 KBS 或 MedIA 的论文水平。** 源码、测试和合成实验说明研究方案可以执行，尚未证明真实病例收益、新颖性充分或临床可靠。建议先以 KBS 为主要目标，保留 MedIA 路线；这是基于当前方法和证据的研究判断，不是录用预测。

## 1. 投稿定位和创新边界

[KBS 的 Elsevier 官方介绍](https://shop.elsevier.com/journals/knowledge-based-systems/0950-7051)强调人工智能方法、知识驱动系统及理论与实际研究。[MedIA 的官方介绍](https://shop.elsevier.com/journals/medical-image-analysis/1361-8415)强调医学图像处理与分析的基础方法贡献。与两者范围都有关，并不意味着同一批工程改进足以投稿。ScienceDirect 作者指南仍未成功读取，本文不设想期刊规定的扩刊比例。

应主动比较已有工作：

- [SDFReg](https://arxiv.org/abs/2304.08929)已研究神经隐式函数配准，不能把“implicit / differentiable registration”本身作为首次贡献。
- [MICCAI 2026：Keypoint Correspondence Learning for CBCT–IOS Registration](https://papers.miccai.org/miccai-2026/0528-Paper5399.html)直接处理 CBCT–IOS 对应学习。官网摘要与作者反馈已核对；其对应监督也利用参考刚体变换。因此“由 transform 派生监督”单独不足以区分本工作。尚未在本仓库复现该方法，官网未提供代码/数据链接；不能声称已完成公平对比。
- [Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning](https://arxiv.org/abs/1908.02983)讨论伪标签确认偏差。教师在留出点上预测较小距离，仍不是独立正确性证据。

更有价值的核心问题是：**如何让有限人工配准知识形成更适合搜索与优化的图像条件场，并识别哪些新增配准能提供有效监督？** 至少应检验以下三条；若不成立，就缩减对应贡献，而不是用更多模块掩盖。

| 假设 | 公平对照 | 失败时的结论 |
| --- | --- | --- |
| H1 候选判别监督改善实际选择 | C1/C2 同候选、backbone、预算和 seed；比较 selected D、regret、真实 refinement | 只有场回归下降不足以支持 registration-aware |
| H2 空间权重比同标签的均匀权重更有效 | 同 accepted cases/transforms、初始化、查询随机数和更新数；仅改变正权重幅值 | C3/C5 的总体差异不能单独归因于空间权重 |
| H3 固定 gate 的接受质量能跨患者保持并帮助 student | 开发集冻结，独立人工参考审计，再做 matched-budget SSL | “多次收敛一致”不能直接称可信配准 |

当前 u(x) 是多次**刚体变换分歧**诱导的空间量，不是局部金属伪影或牙面真实性的独立预测。不要将权重描述成已经学会解剖区域置信度。

## 2. 距离场是否真的帮助优化

新增命令对 dense/implicit checkpoint 使用完全相同的物理扰动、点子集和优化预算。默认围绕人工参考变换后的固定 anchors 质心施加旋转，再加指定长度的平移；保持参考 parity。随机序列绑定病例身份和 seed，不随 manifest 排序或方法名变化。

```bash
python scripts/benchmark_journal_landscape.py \
  --manifest /data/journal/experiment/manifest.json --split val \
  --checkpoint dense=/data/runs/dense_seed0/best.pt \
  --checkpoint implicit=/data/runs/c2_seed0/best.pt \
  --rotation-degrees 0 5 10 20 40 --translation-mm 0 1 2 4 8 \
  --repeats 5 --refinement-steps 20 --learning-rate .25 \
  --point-budget 4096 --seed 20261002 --bootstrap-samples 2000 \
  --reference-kind manual --output-dir /data/diagnostics/landscape_val
```

`report.json` 保存每个起点、最终变换、D 和能量变化，按旋转/平移网格给成功、改善、恶化、无效比例，以及患者层面的配对汇总。失败不能只从误差均值中消失：均值以有效结果为条件，失败率保留所有起点；配对误差明确使用共同有效结果。默认 25 个网格 × 5 次 × 每个 jaw，先缩小网格测实际成本。

**这些是围绕参考构造的 oracle 初始化，只用于机制诊断。** 主结果仍须由真实候选缓存启动，不能把此实验的成功率当成自动部署成功率。重点查看能量下降而真实 D 上升的区域；这直接暴露场能量与真实配准目标的不一致。

新增 `validation_metric="refined_selected_D_mm"` 可按精修后重选的 D 选择 checkpoint；相关设置是 `validation_refinement_steps`、`validation_point_budget`、`validation_refinement_learning_rate`。默认旧 `selected_D_mm` 仍只评价原始候选，是速度较快的代理指标。对以 refinement 为主要贡献的正式实验，应统一使用前者，并与最终优化预算一致。该模式每轮明显更昂贵，不能把两种选模协议混在同一个公平比较里。

## 3. 独立审计伪标签验证器

### 3.1 开发审计

```bash
python scripts/audit_journal_verification.py collect \
  --manifest /data/journal/experiment/manifest.json \
  --checkpoint /data/runs/c2_seed0/best.pt \
  --gate-config configs/journal/verification_initial.json \
  --purpose development --split val \
  --starts 8 --sectors 4 --refinement-steps 20 --point-budget 4096 \
  --seed 20261002 --success-threshold-mm 2 --alpha .05 --risk-target .05 \
  --output-dir /data/audits/gate_dev_v1
```

验证器只收到 image、affine、points、anchors、candidates 及必要身份信息；transform、label 和其他数组都不传入。完成接受/拒绝及 medoid 选择后，评价器才使用保存的人工参考。`evidence/` 保存原始决策，`audit.json` 保存事后 D、parity、拒绝原因及 U95，足以绘制 U95–D 散点并检查“低不确定但配错”。单例数值失败记作拒绝并保留；坏输入使运行明确停止，部分报告状态为 running，不能冻结。

这一审计只接受 manual reference；silver 或相同算法生成的参考不作为独立真值。算法不会替代人工核验参考的质量。

先要求同一病例现有上下颌都接受，才计入可导出病例；单颌数据仍允许。统计独立单位为患者：同一患者任何已接受 jaw 配错（D 超阈值或 parity 错）就记该患者失败。报告总病例/患者数、病例/患者覆盖率、错误接受率和单侧精确二项上界。来源分别汇总；来源间结果不是已经校正多重比较的联合保证。

使用 [NIST 介绍的精确二项界](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm)：若 n 位被接受患者零错误，95% 单侧上界为 `1 - 0.05**(1/n)`。30 位约 9.5%，59 位约 4.95%；需要的是**被接受的独立患者数**，不是总 jaw 数、优化起点数或训练 seed 数。无人被接受时上界为 1，绝不标为成功。2 mm 和 5% 都是可配置研究阈值，不是已证实的临床安全阈值。

### 3.2 冻结后才打开独立队列

开发集允许调 gate，但其风险界仅用于描述；如果用于选择过 checkpoint，也不是独立认证队列。选定设置后冻结：

```bash
python scripts/audit_journal_verification.py freeze \
  --selected-audit /data/audits/gate_dev_v1/audit.json \
  --output /data/protocols/gate_policy_v1.json

python scripts/audit_journal_verification.py collect \
  --manifest /data/journal/experiment/manifest.json \
  --checkpoint /data/runs/c2_seed0/best.pt \
  --gate-config configs/journal/verification_initial.json \
  --purpose frozen --split external_test \
  --policy /data/protocols/gate_policy_v1.json \
  --starts 8 --sectors 4 --refinement-steps 20 --point-budget 4096 \
  --seed 20261002 --success-threshold-mm 2 --alpha .05 --risk-target .05 \
  --output-dir /data/audits/gate_external_v1
```

若开发时看过其他 cohort，冻结时用可重复的 `--additional-development-audit` 列全；不要只声明最后一次。policy 绑定 checkpoint、gate、运行设置、风险终点及代码 hash，拒绝修改后直接复用。冻结文件不可原地覆盖。测试患者及已知图像 hash 不能参与训练、checkpoint 选择或 gate 开发；external source 也不能参与这些阶段。新输入的身份/hash 仍需上游诚实且完整，工具不能识别所有经过重采样、改名的重复患者。

风险上界的解释依赖固定策略、独立且代表目标分布的患者，以及无适应性重复查看。跨来源分布变化、反复试 gate 后挑外部最好结果、按 U95 跨病例选 top-K 都不自动满足这些条件。**本工具审计未加 `max_cases` 截断的 gate；不声称其界覆盖任意后续 top-K 伪标签集。** 如果需要等量 SSL，从冻结入选集定义单独的预算对照并公开名单，另报 all-accepted 结果。

审计与导出已共用按 `(source, case_id, jaw, seed)` 派生的随机化。导出仍需显式使用同 gate 和运行参数；不自动把政策文件变成临床许可。实际导出的域若不同于审计域，也不能直接继承风险结论。

### 3.3 必做负控制

后续真实实验应包含错患者 IOS、错误 parity、重复牙列附近的错误 basin、低 overlap 和金属伪影分层，并保留原始完整分母。错配对没有合法正确变换，应单独报告拒绝率，不能把同患者参考当成其 GT。当前测试覆盖一致错误被事后审计检出和错误 parity；**尚未完成真实错配/牙列重复/伪影队列**，不能把单元测试写成论文实验。

## 4. 拆清 SSL 机制

现 C3/C5 同时改变 support/field 目标、选择策略和点权重，是系统对照，不能独立证明“空间权重有效”。新增严格权重控制：

```bash
python scripts/ablate_field_pseudo_weights.py \
  --manifest /data/journal/pseudo_c6_seed0/manifest.json \
  --output-dir /data/journal/pseudo_c6_uniform_seed0
```

该控制保持同一组 accepted cases、变换、点、有效 mask、未知空间半径及教师来源；只把原正点权重置为 1，零权重仍为零。用原 C6 配置、同一 C2 初始化、seed 和 epoch 训练两个学生，只更换 manifest 和 output-dir。转换另写 pseudo NPZ，会增加磁盘占用，原数据不修改。它检验平滑权重幅值，而非零权重筛选本身；后者需要另做同目标的筛选消融。

核心六组之外，至少报告：监督 continuation、同 accepted transform 的 uniform/spatial、dense/implicit 同优化预算，以及候选原始/精修后选择。各组总训练步数须把教师阶段计入。多 seed 不等于更多独立患者。

## 5. 完整分母和模型来源

正式评价使用预先固定 manifest，而不是仅相信输出文件中出现的病例：

```bash
python scripts/evaluate_journal_registration.py \
  --input /data/runs/c6_seed0/test/evaluation_input.json \
  --expected-manifest /data/journal/experiment/manifest.json --split test \
  --expected-method journal_field --reference-kind manual \
  --missing-as-failure --output /data/runs/c6_seed0/test/report.json
```

多方法指定全部 `--expected-method`；额外病例/方法、患者身份或参考类型不符会报错。没有 expected roster 时，多方法也必须有相同病例集合；单方法无法仅凭输出知道哪个病例消失了，所以正式结果必须给 roster。同分 confidence 整组进入 risk–coverage，缺 score 最后一组保留；不会产生同阈值却只挑好病例的虚假低风险点。

新 checkpoint 使用 `provenance_schema_version=2`，分别保存 training 和 selection 的 patient IDs、content hashes、sources；初始化/续训会保留祖先历史。旧 checkpoint 没有 selection 历史仍可读取作开发研究，但不能假设它未看过测试集。旧结果不能直接精确 resume 到新协议；若有完整历史，可用 `--initialize-field` 配合 `--initialization-provenance` 开新实验，sidecar 必须包含：

```json
{
  "provenance_schema_version": 2,
  "checkpoint_sha256": "ACTUAL_CHECKPOINT_SHA256",
  "training_patient_ids": ["ACTUAL_TRAIN_PATIENT"],
  "training_content_hashes": ["ACTUAL_TRAIN_HASH"],
  "training_sources": ["ACTUAL_TRAIN_SOURCE"],
  "selection_patient_ids": ["ACTUAL_VALIDATION_PATIENT"],
  "selection_content_hashes": ["ACTUAL_VALIDATION_HASH"],
  "selection_sources": ["ACTUAL_VALIDATION_SOURCE"],
  "excluded_patient_ids": ["ACTUAL_EXCLUDED_PATIENT"]
}
```

这些是必须替换的格式示例。空列表只能表示确实为空，不能表示未知。无法重建来源时从头按新协议训练。

## 6. 何时再判断投稿

| 条件 | KBS 路线建议 | MedIA 路线还应重点回答 |
| --- | --- | --- |
| 实质新增 | H1–H3 中有清晰、可归因的成立部分 | 对医学图像配准基础方法的贡献是否超出特定工程组合 |
| 强对照 | RTR-General、binary support、dense/implicit、匹配预算 SSL、可复现学习型基线 | 直接跨模态对应等近期方法，透明处理不可获得的代码/数据 |
| 独立证据 | 患者锁定和外部来源、完整失败、多 seed、患者配对 CI | 独立医学参考、观察者差异和代表实际使用的困难分层 |
| 可靠性 | gate 真错误率、拒绝覆盖率、student 实际收益 | 安全相关主张另需临床设计，不能只依赖 D 或教师残差 |

这张表是研究团队的 go/no-go 标准，不是任何期刊公布的录用规则。目前这些真实结果均待另一台数据机器完成。若主假设不成立，应如实收缩论文贡献；若只在内部来源有效，应收缩泛化结论。继续写代码无法替代这一步。
