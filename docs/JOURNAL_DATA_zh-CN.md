# 期刊分支的数据接口与跨机器转换

本分支沿用旧 RTR 的 CBCT、IOS 和配准矩阵接口。新增部分是明确的患者分组、
数据来源、实验划分、固定 IOS 点集及上游模型来源记录。无需移动原始数据；在训练
机器上生成旧清单和预处理结果，再运行转换器即可。

本文描述可执行的数据流程。它不表示新方法已经在真实病例上验证，也不把自动参考
或伪标签当成人工真值。训练与实验配置见 [JOURNAL_TRAINING_zh-CN.md](JOURNAL_TRAINING_zh-CN.md)。

## 1. 旧接口的实际目录结构

`scripts/build_manifest.py` 调用的 `task2reg.data.build_manifest` 实际读取以下结构。
目录名支持 `images`/`Images`、`labels`/`Labels` 两种写法：

```text
<DATA_ROOT>/
  Train-Labeled/
    images/
      003/
        CBCT.nii.gz
        upper.stl
        lower.stl
    labels/
      003/
        upper_gt.npy
        lower_gt.npy
  Train-Unlabeled/
    images/
      060/
        CBCT.nii.gz
        upper.stl
        lower.stl
  Validation/
    images/
      350/
        CBCT.nii.gz
        upper.stl
        lower.stl
    labels/                       # 没有可用参考时可以没有此目录
      350/
        upper_gt.npy
        lower_gt.npy
```

旧扫描器要求三个 split 下的 images 目录存在。其他数据集可直接提供同字段的
CaseRecord CSV，不必伪装为竞赛目录。每行对应一个颌，字段为：

```csv
split,case_id,jaw,cbct_path,ios_path,transform_path,complete
Train-Labeled,003,upper,/data/STS/Train-Labeled/images/003/CBCT.nii.gz,/data/STS/Train-Labeled/images/003/upper.stl,/data/STS/Train-Labeled/labels/003/upper_gt.npy,True
Train-Unlabeled,060,upper,/data/STS/Train-Unlabeled/images/060/CBCT.nii.gz,/data/STS/Train-Unlabeled/images/060/upper.stl,,True
```

`transform_path` 是 IOS→CBCT 的 `4×4` 齐次矩阵，不是分割 mask。`complete`
表示 CBCT/IOS 是否完整；数据来源和患者身份不在旧 CSV 中，必须另外提供。

旧清单和旧 CBCT hash cache 使用机器上的绝对路径。换电脑后应重建清单和 hash
cache；不要直接沿用原电脑盘符或挂载点。重建清单不会修改原始影像。

## 2. 复用旧预处理

以下路径只是示例。建议把病例、NPZ 和模型输出放在仓库外。完整旧方法命令见
[REPRODUCE.md](REPRODUCE.md) 第 2–4 节。

```bash
export DATA_ROOT=/data/STS
export RUN_ROOT=/data/rtr_kbs
mkdir -p "$RUN_ROOT/manifests"

python scripts/build_manifest.py \
  --data-root "$DATA_ROOT" \
  --output "$RUN_ROOT/manifests/task2.csv"

python scripts/build_cbct_hash_cache.py \
  --manifest "$RUN_ROOT/manifests/task2.csv" \
  --output "$RUN_ROOT/manifests/cbct_hashes.json"

python scripts/prepare_threshold_dental_roi.py \
  --manifest "$RUN_ROOT/manifests/task2.csv" \
  --split Train-Labeled --dataset-name LabeledDentalROI \
  --work-root "$RUN_ROOT/work" --threshold 1600 --margin-mm 20

python scripts/prepare_crown_localizer_data.py \
  --manifest "$RUN_ROOT/manifests/task2.csv" \
  --roi-dir "$RUN_ROOT/work/inputs/LabeledDentalROI/imagesTs" \
  --output-dir "$RUN_ROOT/crown_labeled" --split Train-Labeled \
  --grid-size 128 --spacing-mm 1.25 --surface-radius-mm 0.7 \
  --minimum-hu -1000 --crown-fraction 0.35 --ios-points 120000

python scripts/prepare_threshold_dental_roi.py \
  --manifest "$RUN_ROOT/manifests/task2.csv" \
  --split Train-Unlabeled --dataset-name UnlabeledDentalROI \
  --work-root "$RUN_ROOT/work" --threshold 1600 --margin-mm 20

python scripts/prepare_crown_localizer_inference_data.py \
  --manifest "$RUN_ROOT/manifests/task2.csv" \
  --roi-dir "$RUN_ROOT/work/inputs/UnlabeledDentalROI/imagesTs" \
  --output-dir "$RUN_ROOT/crown_unlabeled" --split Train-Unlabeled \
  --grid-size 128 --spacing-mm 1.25
```

有参考的预处理 NPZ 含 `image`、`affine`、可选 `label`；无标签预处理必须仅提供
影像及 affine，不携带隐藏 GT。旧 labeled preparation 用真实变换建立辅助 support
标签，这一用途属于训练监督；新的场点集和评估锚点不会由测试 GT 选择。

上述旧 labeled 脚本的随机种子计算使用 `int(case_id)`。非数字病例 ID 的其他来源
应先建立稳定的本地数字 ID 映射，或自行提供符合下述 image/affine 接口的 NPZ，
并通过 `prepared_npz_path` 指定路径。期刊转换之后的 namespaced ID 不需要是数字。

## 3. 显式 metadata CSV

必需列是 `source,case_id,patient_id,split`。建议每行都明确填写 `reference_kind`，
避免把其他来源的自动配准矩阵误当成人工真值。以下 ID 和划分仅演示格式：

```csv
source,case_id,patient_id,split,reference_kind,jaw,crown_points_path
STS,003,patient_0003,train,manual,upper,crops/003_upper.npz
STS,003,patient_0003,train,manual,lower,crops/003_lower.npz
STS,010,patient_0010,val,manual,upper,crops/010_upper.npz
STS,010,patient_0010,val,manual,lower,crops/010_lower.npz
STS,060,patient_0060,unlabeled,none,upper,crops/060_upper.npz
STS,060,patient_0060,unlabeled,none,lower,crops/060_lower.npz
```

`case_id` 在这个 CSV 中仍是旧 ID。转换后使用 `source:case_id`，例如 `STS:003`，
并保留 `original_case_id=003` 供旧候选和旧伪标签匹配。不同来源都叫 `003` 时，
用 `--source STS` 一次导入一个来源，然后合并期刊清单。

`patient_id` 必须跨来源统一：同一患者出现在 STS 和另一数据源时，应使用同一个
患者 ID。给不同数据源机械添加不同患者前缀会隐藏重复患者，不能代替患者核对。

| 可选列 | 用途 |
|---|---|
| `reference_kind` | `manual`、`silver`、`none`；伪标签通过专用导出器建立 |
| `jaw` | `upper`/`lower`；空白时该 metadata 行匹配同病例的两个颌 |
| `original_split` | 原 CaseRecord 的 split；旧 ID 在多个旧 split 复用时用于消歧 |
| `prepared_npz_path` | 指定旧 image/affine NPZ，相对 metadata CSV 所在目录解析 |
| `crown_points_path` | 固定上游 crown 点集，相对 metadata CSV 所在目录解析 |
| `content_hash` | 经核对的影像内容分组标识；空白时转换器计算 |

一个 `crown_points_path` 对应一个颌，示例因此使用逐颌 metadata 行。没有 `jaw`
的公共 metadata 行适合两个颌共用的 image/affine、患者和 split 信息，不能给两个
颌误用同一个颌的 crop 文件。

转换器兼容旧 labeled 数据，未填写 `reference_kind` 时会按是否存在
`transform_path` 推断 `manual` 或 `none`。这不构成对矩阵质量的核验。Li、私有
数据及其他自动参考来源应显式填写角色，不能依靠这一默认值。

### 划分与参考角色

| `split` | 允许的参考与用途 |
|---|---|
| `train` | 仅 `manual`，用于模型拟合 |
| `val` | 仅 `manual`，用于开发、模型与阈值选择 |
| `test` | `manual` 或 `silver` 评估；`none` 只能推理，无法计算参考误差 |
| `external_test` | 完整来源锁定的外部评估；参考解释与 test 相同 |
| `unlabeled` | 仅 `none`，不允许 transform 或 GT label |
| `pseudo` | 仅 `pseudo`，由对应导出器替换 accepted unlabeled 记录 |

某个 `source` 一旦用于 `external_test`，该来源的其他记录也必须保持在
`external_test`，不能同时用于训练、伪标签生成或阈值选择。Silver 只允许在
`test`/`external_test` 作为注明性质的参考；它既不参与核心训练，也不用于 val。
论文中应把人工参考与 silver 结果分开报告。

同一患者或相同内容 hash 不得跨越 manual train、val、test、external test 或
无标签训练池。`unlabeled` 与 `pseudo` 共享训练池分区：预算只接受某患者的一次
扫描时，可保留同患者另一病例为 unlabeled。相同病例的可用上、下颌仍须原子地
转换为同一 split，不能只把其中一颌变成 pseudo。只有一个可用颌的数据可以使用；
预算单位是 case，报告时同时给出实际颌数与患者数。

## 4. 固定 IOS 点集与无 GT 评估锚点

`crown_points_path` 接受：

- `.npy`：`float[N,3]` 点坐标；
- `.npz`：必需 `points[N,3]`，可选 `point_weights[N]`。

这些点必须仍在原 IOS 坐标系，单位为 mm。不能把已经用 GT 对齐至 CBCT 的点写入
此字段，否则后续应用 transform 会再次对齐。上游分割、牙冠裁剪方法应在实验开始
前固定并记录版本；测试集的 crop 不得根据 GT、最终配准结果或测试指标调整。

转换器不调用 GT 辅助的 crown 侧面选择。若未提供固定 crop，会对完整 IOS 三角网格
按面面积采样，发出明确警告，并写入 `surface_definition="full_ios"`。这能验证
接口和提供完整 IOS 对照，但含牙龈等非目标区域，不能被描述为 crown-only 实验。

用于位姿误差的 `anchors[M,3]` 始终从完整 IOS 按面面积独立采样。它们与训练查询
使用不同随机流，与 GT 无关；所有比较方法必须使用同一锚点 NPZ。固定锚点误差
应称为固定点位姿位移，不能在没有独立标志点的情况下写成人工 landmark TRE。

## 5. 从旧 NPZ 转换

若 labeled 和 unlabeled 分别在不同目录，可在 metadata 中用 `prepared_npz_path`
逐病例指定路径，或分两次导入后合并。不要把不同原 split 的同名文件覆盖到一起。

```bash
python scripts/prepare_registration_field_data.py \
  --manifest "$RUN_ROOT/manifests/task2.csv" \
  --data-dir "$RUN_ROOT/crown_labeled/data" \
  --metadata "$RUN_ROOT/manifests/sts_labeled_metadata.csv" \
  --source STS \
  --output-dir "$RUN_ROOT/journal/sts_labeled" \
  --points 12000 --anchors 4096 --seed 20261002

python scripts/prepare_registration_field_data.py \
  --manifest "$RUN_ROOT/manifests/task2.csv" \
  --data-dir "$RUN_ROOT/crown_unlabeled/data" \
  --metadata "$RUN_ROOT/manifests/sts_unlabeled_metadata.csv" \
  --source STS \
  --output-dir "$RUN_ROOT/journal/sts_unlabeled" \
  --points 12000 --anchors 4096 --seed 20261002
```

上面两个 metadata 文件分别包含各自要导入的记录；转换器会忽略旧 CSV 中未被本次
metadata 选择的记录，但 metadata 中每一行必须能匹配一个完整旧记录。重复 ID
需要 `original_split` 区分；预处理文件需显式 `prepared_npz_path`，或放在
`<data-dir>/<original_split>/<case_id>.npz` 中。

输出是 `manifest.json` 和 `data/*.npz`。文件名使用身份 hash，真实病例身份保存在
清单中。数据按颌逐个暂存，避免在内存保留整个数据集；最终清单在验证后写入。
已有输出必须显式 `--overwrite`，不要把覆盖用于改变一个已经锁定的实验版本。

## 6. 期刊清单与 NPZ 合同

下面是单条记录的结构示例；实际训练清单还需要独立 train/val 记录及完整评估分组。

```json
{
  "schema_version": 1,
  "records": [
    {
      "case_id": "STS:003",
      "original_case_id": "003",
      "patient_id": "patient_0003",
      "source": "STS",
      "split": "train",
      "jaw": "upper",
      "npz_path": "data/example.npz",
      "reference_kind": "manual",
      "candidate_provenance": {"kind": "none"},
      "surface_definition": "fixed_upstream_crop",
      "anchor_definition": "area_uniform_full_ios"
    }
  ]
}
```

| NPZ 字段 | 形状/含义 |
|---|---|
| `image` | `[I,J,K]`，有限数值 HU；旧准备通常为 int16 |
| `affine` | `[4,4]`，voxel index→CBCT 世界坐标，mm |
| `points` | `[N,3]`，固定源 IOS 点，至少 3 点 |
| `anchors` | `[M,3]`，固定完整 IOS 锚点，至少 3 点 |
| `transform` | 可选 `[4,4]`，人工/silver 参考或已接受的场伪配准 |
| `candidates` | 可选 `[C,4,4]`，旧候选变换；有候选必须声明来源 |
| `point_weights` | 可选 `[N]`，有限的 `[0,1]` 权重，至少有正权重 |
| `label` | 可选 `[I,J,K]`，0 背景、1 upper、2 lower |

体积保持 NIfTI 数组轴顺序 `[I,J,K]`；输入网络时为 `[B,1,I,J,K]`。世界坐标关系是：

```text
[x_mm, y_mm, z_mm, 1]^T = affine @ [i, j, k, 1]^T
points_cbct = points_ios @ transform[:3, :3].T + transform[:3, 3]
```

不要因为 PyTorch 插值接口的坐标顺序而手动交换 NPZ 轴。Affine 必须可逆；代码按
完整 affine 处理斜向与非等方网格，不能用“统一 spacing + 平移”替代原矩阵。
配准矩阵要求刚性线性块，允许行列式为 `+1` 或 `-1`，以兼容旧数据的 parity 约定。
不要为强行得到 `+1` 而事后翻转导出的矩阵。

有标签的 `manual`/`silver` 记录必须包含 transform。`reference_kind=none` 不允许
transform 或 label。原注册工具的 NIfTI 标签可出现 `17`，不能直接作为这里的
训练 label；此接口只接受 `0,1,2`。

训练查询按 surface、near-surface、candidate、outer、uniform ROI 的混合采样。
目标为相对于 `transform(points)` 的最近点距离，单位 mm，并按 truncation 截断。
开放牙冠面使用 unsigned distance，不定义 inside/outside。新增空间增强必须同时
更新查询点和 affine；不能把旧 `CrownDataset` 的数组翻转直接用于这些物理监督。

## 7. 多来源合并与搬迁

所有来源合并后必须再次运行协议检查。单独验证 STS 清单无法发现另一来源中重复的
患者，也无法要求候选教师排除尚未列出的外部患者。

```bash
python scripts/merge_journal_manifests.py \
  --inputs "$RUN_ROOT/journal/sts_labeled/manifest.json" \
           "$RUN_ROOT/journal/sts_unlabeled/manifest.json" \
           "$RUN_ROOT/journal/external/manifest.json" \
  --output "$RUN_ROOT/journal/experiment/manifest.json"
```

合并器保留数据所在位置，重新计算相对路径并检查患者、内容、来源与 provenance
约束。不要用简单列表拼接绕过检查，也不要通过改写患者 ID 消除报错。

`load_journal_manifest(path, data_root=None)` 默认相对清单目录解析 `npz_path`。
提供 `data_root` 时，**相对路径**改为相对此根目录解析；绝对路径不会被改写。
这不是按旧前缀做字符串替换。复制整个 `journal/` 目录树可保留各清单之间的
`../` 相对引用；只复制合并后的 manifest 而不复制其引用数据仍会缺文件。

训练/推理命令的 `--data-root` 采用相同语义。若改变目录层次，应重新合并或显式
更新相对路径。旧 CaseRecord CSV 与旧 hash cache 的绝对路径仍应在新电脑重建。

### 内容 hash 的边界

未在 metadata 指定 hash 时，转换器优先计算 CBCT 解压后的 NIfTI 字节 SHA256，
标记为 `nifti:<hash>`。原 CBCT 不可用时，退回 prepared image、shape、affine 的
SHA256，标记为 `prepared:<hash>`。这两个命名空间的 hash 不能直接互相比较。

NIfTI hash 忽略 gzip 重压缩差异，但不保证识别重采样、裁剪、头信息变化或另一时点
扫描。Prepared hash 也只识别完全一致的准备结果。患者跨来源匹配仍必须根据研究
管理信息核对；hash 只是补充屏障。metadata 提供的 hash 属于调用者声明，应记录
生成规则，不能为了通过协议检查而填入随机值。

## 8. 导入旧候选与 OOF 来源

旧候选路径为 `<run>/<old_case_id>_<jaw>/candidates.json`，内容是列表；转换器读取
每个元素的 `transform`。多个 run 的候选依次合并，其他训练/评估诊断字段不会变成
场输入。追加到第 5 节转换命令：

```bash
  --candidate-runs "$RUN_ROOT/candidates_seed1" "$RUN_ROOT/candidates_seed2" \
  --candidate-provenance "$RUN_ROOT/manifests/candidate_provenance.json"
```

该 JSON 可以是一个覆盖全部导入记录的声明，也可以逐记录提供：

```json
{
  "records": [
    {
      "source": "STS",
      "case_id": "003",
      "jaw": "upper",
      "candidate_provenance": {
        "kind": "learned",
        "model_id": "outer0-inner0-support-and-prior",
        "excluded_patient_ids": ["patient_0003", "patient_0010", "external_patient_0042"]
      }
    }
  ]
}
```

逐记录形式必须覆盖实际导入的每个颌。没有候选使用 `{"kind":"none"}`。
完全不使用学习模型、学习先验或学习 support 的候选可声明 `geometry_only`。
只要候选依赖学习的 support mask、统计先验或提议模型，就必须使用 `learned`。
不能因为最终步骤是 ICP 就声明为 geometry-only；评估候选也不得依赖该病例的
GT-aligned target 或 GT 变换。

`learned` 要求排除本病例患者，以及**完整当前实验清单里的所有 val/test/external
患者**。旧五折 OOF 标记本身不足以证明这一条件：某个旧教师可能排除了当前病例，
却训练过新的 journal validation 患者。此时应按新的 outer/inner 划分重新生成
support、先验和候选；不要把未排除的患者手动写进 exclusion list。

`model_id` 和 exclusion list 是可审计声明，不是代码自动证明。应另存教师配置、
训练患者列表、模型 hash 与生成日志，并在实验冻结前核对。

## 9. 两种伪标签合同

两种伪标签都必须是 `split=pseudo, reference_kind=pseudo`，并附：

```json
{
  "pseudo_provenance": {
    "teacher_id": "frozen-round0-teacher",
    "excluded_patient_ids": ["patient_0010", "external_patient_0042"]
  }
}
```

排除列表需覆盖完整清单中的 val/test/external 患者。数据装载只校验声明与协议，
不能替代对教师训练历史的审计。

| `pseudo_kind` | NPZ 要求 | 用途 |
|---|---|---|
| `field`（默认） | transform + point_weights；禁止 label | 已验证配准的空间监督 |
| `support` | label；禁止 transform | 旧 crown-confidence 自训练对照 |

Field pseudo 用原 IOS 点及接受的变换生成世界空间监督。记录可设置
`supervision_radius_mm`，默认 `2.0`；采样点距已接受表面超出此范围时权重为零，
未知空间不会被解释为确定背景。权重继承最近表面点的 uncertainty weight。
半径和接受阈值都需要在开发集上校准，然后冻结。

Support pseudo 只监督辅助 support 分支，不应被当作经过几何验证的场伪标签。
具体训练方式由 `ssl_strategy` 配置控制。

## 10. 旧 crown-confidence 伪标签对照导入

旧 `select_crown_pseudo_labels.py` 没有 `selected.csv`。实际产物是：

```text
crown_pseudo/
  audit.csv                 # 每 fold、每 case 的过滤诊断
  summary.csv               # 汇总数量，没有逐病例选择列表
  summary.json
  top_80/
    fold_0/
      060.npz               # 真正选中后复制的 label + affine
    fold_1/
      ...
```

`audit.csv` 的实际列为
`fold,case_id,upper_voxels,lower_voxels,confidence,entropy,selection_score,accepted_filter`。
`accepted_filter=1` 只表示通过体积/置信度/熵过滤，最终 top-N 和内容去重的成员还必须
以 `top_N/fold_F/` 中存在的 NPZ 为准。

准备教师声明文件，例如：

```json
{
  "teacher_id": "legacy-crown-outer0-fold0",
  "fold": 0,
  "excluded_patient_ids": ["patient_0010", "external_patient_0042"]
}
```

然后运行：

```bash
python scripts/import_legacy_support_pseudolabels.py \
  --manifest "$RUN_ROOT/journal/experiment/manifest.json" \
  --pseudo-label-dir "$RUN_ROOT/crown_pseudo/top_80/fold_0" \
  --selection-csv "$RUN_ROOT/crown_pseudo/audit.csv" \
  --teacher-provenance "$RUN_ROOT/manifests/legacy_teacher.json" \
  --source STS --max-cases 20 \
  --output-dir "$RUN_ROOT/journal/legacy_support_pseudo"
```

Fold 从 `fold_0` 目录名推断；若目录无此命名且 CSV 包含多个 fold，必须加 `--fold 0`。
若提供的是原预测 `label_arrays/`，导入的是通过 confidence gate 的池，尚未复现旧
selector 的 top-N/内容去重；需要明确报告这一差别。优先使用复制后的 selected 目录。

导入器按 `selection_score` 降序、旧 case ID 打破同分，再限制 `--max-cases`；`0`
表示不额外限制。只转换清单里的 unlabeled 记录，通过 `original_case_id` 匹配，
跨来源旧 ID 重复时要求 `--source`。两个可用颌一起替换，其他记录保持原路径引用。
它检查标签形状、affine、`0/1/2` 值及对应颌是否存在，拒绝包含 transform 的旧标签，
并从 support pseudo 输出中移除候选矩阵。

新清单不会同时保留同一病例的 unlabeled 和 pseudo 副本。`import_summary.json`
记录实际导入病例数和颌数。与 verified-field 自训练比较时，应匹配**实际接受数量**，
而不只是给两个命令填写同一个最大预算；还应报告来源、患者数、教师、筛选与采样规则。

## 11. 转换后需要核对的事实

- 数据载入与协议检查通过，只证明格式与声明的分组一致，不证明上游标签正确。
- 记录固定 crop 的上游版本，抽查 IOS 坐标、变换方向、affine 与冠部覆盖。
- 用患者管理信息核对跨来源身份，再用内容 hash 补充检查重复影像。
- 合并完整实验清单后检查 provenance；不要删除评估记录来消除 exclusion 报错。
- 锁定 external source、split、固定锚点和阈值；后续修改形成新的实验版本。
- Git 分支只提交代码、测试、配置和说明，不提交病例、点集、NPZ、模型权重或私有预测。
