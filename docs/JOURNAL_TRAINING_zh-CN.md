# KBS 扩刊分支：另一台电脑的训练与复现手册

本分支交付的是可以运行的研究代码、配置、数据适配器和实验协议。没有本机
真实数据、训练好的新权重或新方法精度结果。正式训练前先读
[数据手册](JOURNAL_DATA_zh-CN.md)和[核心方法及直接对照](JOURNAL_METHOD_zh-CN.md)。
旧挑战赛推理继续使用旧入口；新研究入口从缓存候选出发，不查找 reference bank。

## 1. 获取分支与环境

推荐 Linux、Python 3.11。先选与训练机驱动匹配的 PyTorch 构建；已核验的
实现接口为 PyTorch 2.6.0。以下是 CUDA 12.4 示例，不代表你们必须使用该驱动：

```bash
git clone --branch research/kbs-registration-field \
  https://github.com/avalanchezy/RegistrationTeachesRegistration.git
cd RegistrationTeachesRegistration
python3.11 -m venv /path/to/envs/rtr-journal
source /path/to/envs/rtr-journal/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.journal.txt
python -m pip install -e . --no-deps
python scripts/journal_preflight.py --device cuda
python -m pytest -q
```

CPU 检查可把 torch index-url 改为 `https://download.pytorch.org/whl/cpu`。
安装时如果共享旧环境，优先建立独立环境；不要修改旧挑战赛训练环境。
原有 Docker 是挑战赛部署容器，不是新训练入口。

主配置宽度仍为 24/48/96/192/384，输入由旧预处理决定，通常 128³。
暂未测量真实数据显存峰值：24 GB 或其他显存均不能保证直接容纳启用二阶项的配置。
显存不足先减 `queries`、`candidate_points`、`max_candidates`；保留 backbone
便于公平比较。`base_channels=8` 可用于开发，但论文对照必须同宽度。
初始六组均关闭 Eikonal/展开 pose loss，二阶优化单独消融。

## 2. 首先跑不需要数据的完整检查

把生成物放在仓库之外，源码发布审计会扫描工作树内的二进制文件，包括被忽略目录。

```bash
OMP_NUM_THREADS=2 python scripts/smoke_registration_field.py \
  --output-dir /tmp/rtr-journal-smoke --device cpu --task-potential

# 可选 CUDA + AMP 检查
OMP_NUM_THREADS=2 python scripts/smoke_registration_field.py \
  --output-dir /tmp/rtr-journal-smoke-cuda --device cuda --amp --task-potential
```

该检查构造合成体积、已知变换和错误候选，执行训练、断点恢复、双头 warm-start、
局部 secant 监督、候选 refinement、
固定点评价和伪标签验证。合成病例被验证器拒绝是正常结果；不应为了让它被接受而放宽
生产 gate。检查 `smoke_result.json`，并查看 `training/metrics.jsonl` 的真实优化步数。
合成数据上的零误差或可运行性不能作为医学数据实验结果。

## 3. 接入现有数据并冻结实验划分

沿用旧 `build_manifest.py`、ROI 和 crown NPZ 预处理。新增工作主要是提供显式
患者/来源/划分 CSV 和固定上游 IOS crown 点集，再转换成 journal manifest。
具体命令、字段及多来源合并见[数据手册](JOURNAL_DATA_zh-CN.md)。例如：

```bash
python scripts/prepare_registration_field_data.py \
  --manifest /data/manifests/task2.csv \
  --data-dir /data/rtr/crown/data \
  --metadata /data/manifests/journal_metadata.csv \
  --source STS --candidate-runs /data/rtr/candidates_oof \
  --candidate-provenance /data/manifests/candidate_provenance.json \
  --output-dir /data/journal/STS

python scripts/merge_journal_manifests.py \
  --inputs /data/journal/STS/manifest.json /data/journal/other/manifest.json \
  --output /data/journal/experiment/manifest.json

python scripts/journal_preflight.py \
  --manifest /data/journal/experiment/manifest.json \
  --config configs/journal/c2_implicit_rank.json --check-data --device cuda
```

每个实验使用包含 train/val/test/external_test/unlabeled 全部角色的完整 manifest，
不能只把训练子集传入以绕过排除规则。监督验证必须是人工参考；自动 silver reference
只能作为单独的 test/external_test 证据。不同 fold 各有一份冻结 manifest。

## 4. 先诊断旧系统，再训练新模型

旧 `run_geometry_benchmark.py` 直接产生候选，不加载 reference banks。几何-only
起点可使用如下命令；`--chirality-mode both` 避免把旧扫描协议当跨来源先验：

```bash
python scripts/run_geometry_benchmark.py \
  --manifest /data/manifests/task2.csv --split Train-Labeled \
  --target-mode threshold --chirality-mode both --stable-record-seeds \
  --no-visualizations --output-dir /data/rtr/geometry_only
```

强 RTR-General 对照应使用患者排除的旧 support 模型生成 crown mask，再用
`--target-mode crown --crown-mask-dir ...`；其 learned prior、候选排序模型也须
仅来自允许的训练患者。几何-only 命令只是初始诊断，不等同于复现完整冠军 generic
排序系统。旧 `run_submission_inference.py` 会尝试 bank，不作为无 bank 实验入口。
旧 ranker 的复现命令保持在 [REPRODUCE.md](REPRODUCE.md)。

```bash
python scripts/diagnose_journal_candidates.py \
  --manifest /data/journal/experiment/manifest.json \
  --candidate-run /data/rtr/candidates_oof --source STS --split val \
  --output-dir /data/runs/oracle_diagnostic
```

诊断从 `result.json` 读取旧系统实际选择，从 `candidates.json` 计算 oracle，
使用统一固定 anchors 重新计算 D，不把缓存中的 `mean_tre_mm` 当独立 landmark TRE。
oracle 差优先解决候选/优化盆地；oracle 好而 selected 差优先看排序。

## 5. 监督基线、rank 和表示消融

```bash
python scripts/train_registration_field.py \
  --manifest /data/journal/experiment/manifest.json \
  --config configs/journal/c1_implicit_supervised.json \
  --output-dir /data/runs/c1_seed0 --seed 20261002

python scripts/train_registration_field.py \
  --manifest /data/journal/experiment/manifest.json \
  --config configs/journal/c2_implicit_rank.json \
  --output-dir /data/runs/c2_seed0 --seed 20261002
```

`dense_supervised.json` 对照 dense TUDF；`implicit_rank_pose.json` 单独启用
Eikonal 与两步可微 refinement。六组核心配置均为 120 个新增训练 epochs，
首先训练 C1/C2 teacher；SSL C3–C6 从对应监督 checkpoint 初始化后再训练。
**六组主表中的 C1/C2 也必须从同一 teacher 再训练相同 120 epochs**，使用
`c1_supervised_continuation.json` / `c2_rank_continuation.json` 和
`--initialize-field`。否则 SSL 组多出一倍优化步数，不能把差异归于伪监督。
可另外报告只有初始 teacher 的结果。论文列出总 steps，包括 teacher 预训练。
运行 seeds `20261002/20261003/20261004`。

训练日志每 epoch 输出 `global_step`、loss、验证指标、AMP 跳步数；`last.pt`
保存 model、optimizer、scaler、RNG、config、数据指纹、Git 版本和源码哈希。
`best.pt` 默认以患者平均 `selected_D_mm` 选择：对固定验证候选，仅由 field energy
选最优候选，再用人工参考评估 D。该标准不受训练 rank warmup 或 loss 权重影响。
`field_loss` 是显式可选开发标准，不得在消融之间混用。记录中的
`validation_loss`/`best_validation_loss` 是通用历史字段名，具体单位以
`validation_metric` 为准；`selected_D_mm` 的单位是 mm。

```bash
python scripts/train_registration_field.py \
  --manifest /data/journal/experiment/manifest.json \
  --config configs/journal/c2_implicit_rank.json \
  --output-dir /data/runs/c2_seed0 \
  --resume /data/runs/c2_seed0/last.pt --epochs 160
```

resume 仅允许改变总 epochs/device；输入、其余配置或内容改变会拒绝，路径搬迁但
内容一致允许。CUDA/CPU 或不同硬件之间不承诺逐位一致；同环境 CPU 恢复已经验证。
若换数据进入 SSL，使用 `--initialize-field` 新建学生实验，不使用 resume。
checkpoint 含 Python 优化器/RNG 状态，只加载可信的本地文件。

旧 support 权重可经 `--initialize-support` 加载，但必须额外传
`--initialization-provenance`，其中包括 `excluded_patient_ids`、
`training_patient_ids`、`training_sources`、`training_content_hashes`，以及
`provenance_schema_version=2`、`selection_patient_ids`、`selection_sources`、
`selection_content_hashes`。历史未知不能写成空列表；格式见
[协议附录](JOURNAL_READINESS_zh-CN.md#5-完整分母和模型来源)。
不能用训练过全部 30 例的旧最终模型初始化新交叉验证却声称患者未见。
默认配置不依赖旧权重。

## 6. 伪标签与六组对照

`verification_initial.json` 是未校准起点，须在开发/OOF 数据上定标后另存冻结版本。
不要以外部测试结果反调。C5 的 teacher 用 C1，C6 的 teacher 用 C2：

```bash
python scripts/build_verified_field_pseudolabels.py \
  --manifest /data/journal/experiment/manifest.json \
  --checkpoint /data/runs/c2_seed0/best.pt \
  --gate-config /data/protocol/gate_frozen.json \
  --starts 8 --sectors 4 --refinement-steps 20 --point-budget 4096 \
  --output-dir /data/journal/pseudo_c6_seed0

python scripts/train_registration_field.py \
  --manifest /data/journal/pseudo_c6_seed0/manifest.json \
  --config configs/journal/c6_implicit_rank_verified_ssl.json \
  --initialize-field /data/runs/c2_seed0/best.pt \
  --output-dir /data/runs/c6_seed0 --seed 20261002
```

验证器检查多起点 medoid/位姿不确定性、留出 sector、全局及各区覆盖、镜像一致、
越界比例和不同盆地歧义，记录所有拒绝原因。留出轨迹从原始缓存初始化，仅用拟合
sector 排序和优化；上游候选若来自整颌 ICP，仍不是整个搜索链的独立空间留出。
需要更严格证据时，离线生成每个 sector 的搜索初始化，再扩展协议；不能把现有
teacher holdout 称作独立真值核验。

接受病例生成 transform+point_weights；远离可靠表面的查询不当作可信背景。
病例已有的上下颌记录一起入选，单颌数据仍允许，预算按 source/case 计，同时报告
实际 jaw 数。`--max-cases` 限制入选量；拒绝所有病例也会正常输出审计 manifest，
随后 SSL trainer 会明确报无可用 pseudo，不能偷偷转成纯监督。

C3/C4 使用旧 support 伪标签选择器的实际输出，经
`import_legacy_support_pseudolabels.py` 转换，命令见[数据手册](JOURNAL_DATA_zh-CN.md)。
训练配置分别为 `c3_implicit_legacy_ssl.json`、`c4_implicit_rank_legacy_ssl.json`，
初始化分别用 C1、C2。旧控制仅施加 support loss；新空间 pseudo 仅施加 weighted
field loss。新损失已自定义扩展，不能描述成字节级复现旧训练过程。

主对照两种 SSL 使用相同病例预算、labeled:pseudo 抽样比例（每 labeled jaw
配一个 pseudo jaw）、pseudo_weight、epochs 和 seeds。默认 `max_pseudo_cases=0`
表示所有可用病例，不等于已经预算匹配；在每组运行前按较小可用数设定相同上限，
冻结各自选入列表并另报 all-accepted。teacher 固定一轮；第二轮明确生成新 manifest
和新学生实验，当前未实现在线 EMA 更新。

## 7. 相同初始化的配准评价

```bash
python scripts/predict_registration_field.py \
  --manifest /data/journal/experiment/manifest.json \
  --checkpoint /data/runs/c6_seed0/best.pt \
  --split test --refinement-steps 20 --evaluate \
  --output-dir /data/runs/c6_seed0/test

python scripts/evaluate_journal_registration.py \
  --input /data/runs/c6_seed0/test/evaluation_input.json \
  --expected-manifest /data/journal/experiment/manifest.json --split test \
  --expected-method journal_field --reference-kind manual --missing-as-failure \
  --output /data/runs/c6_seed0/test/report.json \
  --bootstrap-samples 2000 --seed 20261002
```

在冻结协议后，`--split external_test` 跑完全留出的来源。checkpoint 的患者、
来源及已知图像 hash 均参与排除检查。训练时已使用的来源不能换 manifest 后
重新命名为“external”。外部人工参考与 silver reference 分别运行评价和报告，
不可把它们混成一个 ground-truth 主结果。

同一 split 同时含有两种参考时，评价器默认拒绝混合汇总。分别追加
`--reference-kind manual` 和 `--reference-kind silver`，保存成两个报告；
可追加 `--source SOURCE` 按来源报告。失败记录也保留参考类型并参与分母。

`--refinement-steps 0/2/5/10/20` 比较优化步数；同模型、相同候选列表和 seed。
推理不读取 GT 决策，只有 `--evaluate` 导出人工参考用于事后度量。
`predictions.json` 保存所有候选、所选变换和 field 特征；单例数值失败保留 failed
记录。confidence 是未校准的能量差，并非正确率概率。

报告区分 initial/final selected、oracle、selection regret、matching-start 改变量、
无效输出率、超阈值注册失败率、parity mismatch、1/2/3 mm 阈值，按患者 bootstrap。
把多方法 `evaluation_input.json` 的 records 合并、改为唯一 method 名即可作 paired
比较；同病例 anchors/reference 必须一致。D 是固定 IOS 点变换位移，不是 landmark TRE。
真正解剖 landmark TRE、临床复核、独立外部来源和强学习对应基线仍须后续采集/运行。

## 8. 保存、搬迁与最终实验检查

保留数据许可允许的私有存储：原始 manifest/metadata、固定 splits、候选及其模型
排除记录、每次 gate/config、`pip freeze`、checkpoint、日志、验证拒绝清单、
所有方法的评价输入和报告。新 manifest 使用相对路径；搬迁时保持目录结构，或
使用支持的 `--data-root` 指向共同根目录。旧 CSV 绝对路径需在新机器重建。

先确认 oracle/selected 诊断，再看 C1→C2，再看 dense/implicit 和 basin，再比较
预算匹配 C3–C6，最后打开 locked external test。若 field regression 更好但真实
配准未改善，先分析能量与几何的一致性，不追加 ensemble 来掩盖问题。

```bash
python -m pytest -q
python scripts/audit_source_release.py
git rev-parse HEAD
python -m pip freeze > /data/runs/environment.txt
```

所有新数据、checkpoint 和结果应放仓库外。源码分支无需也不应上传这些私有产物。

## 当前工程边界

正式 GPU 吞吐、128³ 显存峰值和真实病例精度尚未测量；本地证据限于单元测试和
合成 CPU/CUDA 全流程。训练器为单设备，多个 GPU 可独立运行不同 seed/fold，
没有实现 DDP。当前主要方法实验应先按[方法手册](JOURNAL_METHOD_zh-CN.md)比较
M1–M6 的优化方向、收敛范围和真实候选配准。SSL 阈值与统计工具另见
[协议附录](JOURNAL_READINESS_zh-CN.md)，不能替代方法有效性实验。

训练支持断点恢复，但伪标签验证导出目前按顺序完成整批后写汇总，没有病例级
断点续验。单例数值异常会终止这次导出，需修复后重跑；建议先验证小型开发批次，
并始终保留昂贵的上游候选缓存。独立人工参考建立、外部学习型基线适配和临床审阅
属于后续研究工作，不能由这次合成验证替代。
