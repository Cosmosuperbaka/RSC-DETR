# 数据准备

此目录只包含准备说明，数据集图片和标注不随源码分发。

默认路径与两套最终配置保持一致：

```text
datasets/
├── VEDAI/
│   ├── Vehicules1024/
│   └── annotations/
│       ├── vedai_fold01_train_class8.json
│       ├── vedai_fold01_val_class8.json
│       └── vedai_fold01_test_class8.json
└── M3FD lt20/
    ├── visible/
    ├── infrared/
    └── annotations/
        ├── instances_train.json
        ├── instances_val.json
        └── instances_test.json
```

`M3FD lt20` 的目录名包含空格。如果数据放在其他位置，请同时更新配置内所有 `img_folder`、`ir_folder`、`ann_file` 和 `annotation_file`。

VEDAI 使用 fold-1、1024 HBB、8 个标注类别协议；沿用原配置的类别编号与输出维度，不要仅依据类别数量修改 `num_classes`。图像读取与 RGB/IR 配对规则见 [VEDAI 数据加载器](../src/data/dataset/vedai_dataset.py)。

M3FD-LT20 使用长尾训练划分，训练、验证和测试路径分别配置；图像读取与模态配对规则见 [M3FD 数据加载器](../src/data/dataset/m3fd_dataset.py)。此包不提供划分生成脚本或原始数据下载。

`.gitignore` 会忽略此目录中的数据实体，仅保留本说明。

## 官方数据集下载

| 数据集 | 官方来源 | 说明 |
| --- | --- | --- |
| VEDAI | <https://downloads.greyc.fr/vedai/> | GREYC 官方发布页（Razakarivony & Jurie, 2015）。下载 1024×1024 图像（part1–part5）与官方 annotations，另附 devkit 与使用条款。 |
| M3FD | <https://github.com/JinyuanLiu-CV/TarDAL> | TarDAL 项目页（Liu et al., ACCV 2022）提供 M3FD 的下载入口与使用说明。 |

请注意：上述官方发布只包含原始图像与官方标注。本仓库使用的 COCO 格式标注文件（VEDAI 的 `vedai_fold01_*_class8.json` 与 M3FD 的 `instances_{train,val,test}.json`）是本论文自行整理的结果，需要按照以下规则从官方数据构建，且不随本仓库分发：

- VEDAI：采用 fold-1 划分、1024 HBB、8 个标注类别协议；RGB/IR 配对与类别编号规则见 [VEDAI 数据加载器](../src/data/dataset/vedai_dataset.py)。
- M3FD-LT20：seed 42 的长尾训练划分（长尾系数 20）；划分与配对规则见 [M3FD 数据加载器](../src/data/dataset/m3fd_dataset.py)。

## 权重下载

论文使用的模型权重以 Git LFS 形式存放在 [`weights/`](../weights/) 目录；克隆后若未自动拉取，请运行 `git lfs pull`。权重与配置的对应关系见 [registry/experiments.yaml](../registry/experiments.yaml)。
