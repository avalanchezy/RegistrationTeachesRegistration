# Registration Teaches Registration

这是 `avalanchezy` 团队在 **STSR 2026 Task 2 获得第一名**的 CBCT-IOS
半监督配准方法源码。

## 方法概要

Task 2 没有牙齿分割标签。我们把已标注病例的刚体变换用于对齐 IOS，
再把牙冠侧 IOS 表面栅格化为 CBCT 中的上下颌弱监督。五个监督 3D U-Net
与五个自训练 3D U-Net 等权融合，预测牙冠支持区域。随后通过允许
`det(R)=+1/-1` 的 PCA 初始化、裁剪多尺度 ICP、随机盆地搜索、ExtraTrees
候选排序和上下颌联合选择得到最终变换。

最终提交没有使用 Task 1 mask、ToothSeg 权重、外部牙科预训练分割模型或
外部牙科数据集。

## 官方结果

- 隐藏测试集：**Task 2 第一名**
- 公共验证集平均平移误差：**5.7848 mm**
- 公共验证集平均旋转误差：**3.7495 deg**
- 成功输出：**100/100 个牙颌**

主办方的第一名通知未提供隐藏测试集数值，因此没有把公共验证分数冒充为
隐藏测试分数。

## 作者

- [Yi Zhu](https://orcid.org/0009-0001-1159-6853)，CREATIS，第一作者兼通讯作者
- [Razmig Kéchichian](https://orcid.org/0000-0001-7974-8705)，CREATIS
- [Raphaël Richert](https://orcid.org/0000-0002-9298-1293)，Hospices Civils de Lyon
- [Sébastien Valette](https://orcid.org/0000-0001-7549-4808)，CREATIS

## 快速使用

先安装与显卡匹配的 PyTorch，再执行：

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

模型资产不会直接上传到 GitHub，因为模板库包含由主办方数据采样得到的
几何信息。注册参赛者可以按照 [完整复现说明](docs/REPRODUCE.md) 从官方
Task 2 数据重建；团队内部也可以把经过 SHA256 校验的最终资产放入
`model_assets/`。

```powershell
python scripts\verify_assets.py
python scripts\run_submission_inference.py `
  --input-dir C:\path\to\inputs `
  --output-dir C:\path\to\outputs `
  --model-dir model_assets
python scripts\validate_outputs.py `
  --input-dir C:\path\to\inputs `
  --output-dir C:\path\to\outputs
```

输出为：

```text
outputs/<case_id>/upper_gt.npy
outputs/<case_id>/lower_gt.npy
```

每个文件都是有限的 NumPy `float64 (4,4)` 刚体变换矩阵。

## 进一步阅读

- [方法细节](docs/METHOD.md)
- [最终实现映射](docs/IMPLEMENTATION_MAP.md)
- [完整训练复现](docs/REPRODUCE.md)
- [数据结构](docs/DATA.md)
- [Docker 构建与运行](docs/DOCKER.md)
- [模型资产说明](model_assets/README.md)
