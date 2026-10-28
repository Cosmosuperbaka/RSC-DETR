# 数据准备

此目录只包含准备说明与 M3FD 复现材料；数据集图片和标注不随源码分发。

默认路径与两套最终配置保持一致：

```text
datasets/
├── VEDAI/
│   ├── Vehicules1024/
│   └── annotations/
│       ├── vedai_fold01_train_class8.json
│       ├── vedai_fold01_val_class8.json
│       └── vedai_fold01_test_class8.json
├── M3FD lt20/
│   ├── visible/
│   ├── infrared/
│   └── annotations/
│       ├── instances_train.json
│       ├── instances_val.json
│       └── instances_test.json
├── M3FD/reproducibility/          # tracked: fixed lists, manifest, counts, dedupe report
├── DVTOD dataset/          (directory name contains spaces)
│   ├── vis1/images/{train,val}/
│   └── ir1/images/{train,val}/
└── RTDOD_HBB_3class/
    └── annotations/
        ├── instances_train.json
        └── instances_val.json
```

`M3FD lt20` 与 `DVTOD dataset` 的目录名包含空格。如果数据放在其他位置，请同时更新配置内所有 `img_folder`、`ir_folder`、`ann_file` 和 `annotation_file`。

VEDAI 使用 fold-1、1024 HBB、8 个标注类别协议；沿用原配置的类别编号与输出维度，不要仅依据类别数量修改 `num_classes`。图像读取与 RGB/IR 配对规则见 [VEDAI 数据加载器](../src/data/dataset/vedai_dataset.py)。

M3FD-LT20 使用长尾训练划分，训练、验证和测试路径分别配置；图像读取与模态配对规则见 [M3FD 数据加载器](../src/data/dataset/m3fd_dataset.py)。固定划分列表、构建脚本、统计清单与去重报告见 [M3FD 复现材料](M3FD/reproducibility/README.md) 与 [scripts/m3fd](../scripts/m3fd/)；原始数据仍需从官方渠道获取，本包不随源码分发图像或逐框标注。

DVTOD 使用 Person / Car / Bicycle 三类、1088×1920 输入协议，可见光与红外图像分别位于 `DVTOD dataset/vis1` 与 `DVTOD dataset/ir1`，标注为 COCO 格式（`RTDOD_HBB_3class/annotations/instances_{train,val}.json`）；报告数字为验证集结果。该数据集不公开分发，目录布局与 [RSC-DETR_DVTOD.yml](../configs/RSC-DETR_DVTOD.yml) 保持一致。

`.gitignore` 会忽略此目录中的数据实体，仅保留本说明与 `M3FD/reproducibility/` 下的固定列表、清单和统计文件。

## 官方数据集下载

| 数据集 | 官方来源 | 说明 |
| --- | --- | --- |
| VEDAI | <https://downloads.greyc.fr/vedai/> | GREYC 官方发布页（Razakarivony & Jurie, 2015）。下载 1024×1024 图像（part1–part5）与官方 annotations，另附 devkit 与使用条款。 |
| M3FD | <https://github.com/JinyuanLiu-CV/TarDAL> | TarDAL 项目页（Liu et al., ACCV 2022）提供 M3FD 的下载入口与使用说明。 |

请注意：上述官方发布只包含原始图像与官方标注。本仓库使用的 COCO 格式标注文件（VEDAI 的 `vedai_fold01_*_class8.json` 与 M3FD 的 `instances_{train,val,test}.json`）是本论文自行整理的结果，需要按照以下规则从官方数据构建，且不随本仓库分发：

- VEDAI：采用 fold-1 划分、1024 HBB、8 个标注类别协议；RGB/IR 配对与类别编号规则见 [VEDAI 数据加载器](../src/data/dataset/vedai_dataset.py)。
- M3FD-LT20：项目自行生成的固定长尾划分（非 M3FD 官方划分）。数据划分采样种子为 42（基础验证集）、1042（基础测试集）与 42（长尾训练子集）；模型训练种子是另一组独立参数，论文使用 42 / 3407 / 2024。`lt20_seed3407`、`lt20_seed1234` 是未用于论文的备用划分，不能与训练种子 `s3407` 混淆。最终训练/验证/测试为 1428/420/419 对，测试集 `00539` 因 IR 文件与验证集 `01300` 字节相同而被整对移除；最终 IF=20.59。详细协议、哈希与重建命令见 [M3FD 复现材料](M3FD/reproducibility/README.md)。

## 权重下载

论文使用的模型权重以 Git LFS 形式存放在 [`weights/`](../weights/) 目录；克隆后若未自动拉取，请运行 `git lfs pull`。权重与配置的对应关系见 [registry/experiments.yaml](../registry/experiments.yaml)。
