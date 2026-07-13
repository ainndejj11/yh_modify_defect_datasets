

# yh_modify_defect_datasets

缺陷识别数据集构建的相关代码：

1、总库裁切缺陷子图

2、缺陷数据清洗后，标签映射回总库





## 新增：漏裁缺陷分析

新增共享模块 `1-crop_defect_new/defect_coverage_analyzer.py` 和两个入口，用于对比裁切结果与原图缺陷标注，统计未裁切到的缺陷并按原因归集。

### 漏裁原因分类

| 原因 | 含义 |
|---|---|
| `component_not_covered` | 缺陷未与任何部件检测框相交 |
| `incomplete_or_filtered` | 缺陷与部件框有交集，但最大面积占比低于该类阈值 |
| `data_inconsistency` | 缺陷本应被裁切（重叠比达标），但未在 `crop_mapping_正样本.json` 中找到匹配记录 |

### 1. 裁切程序集成（基础报告）

`1-crop_defect_new/统一裁切程序_正样本.py` 在裁切完成后会自动生成 `漏裁统计报告.txt`，输出到 `--output` 目录。

### 2. 独立分析脚本（详细报告 + 可视化）

```bash
python 1-crop_defect_new/分析漏裁.py ddx \
  --images-dir /path/to/JPEGImages \
  --component-ann /path/to/部件_导地线_xml \
  --defect-ann /path/to/Annotations \
  --output /path/to/output_ddx \
  --viz
```

参数说明：
- `component_type`：部件类型（`ddx`/`gt`/`jyz`/`gd`）
- `--images-dir`：原图目录（可选，可视化时需要）
- `--image-index`：pkl 图片索引（可选）
- `--component-ann`：部件检测 XML 目录
- `--defect-ann`：缺陷标注 XML 目录
- `--crop-mapping`：`crop_mapping_正样本.json` 路径（可选，默认从 `--output` 目录推断）
- `--output`：输出目录
- `--config`：配置文件路径（默认 `./config.yaml`）
- `--workers`：线程数（默认 24）
- `--viz`：生成漏裁可视化图
- `--viz-max`：可视化图数量上限，`0` 表示不限制（默认 `0`）

输出文件：
- `missed_defects.json`：每个漏裁缺陷的完整记录
- `missed_defects.csv`：CSV 格式，便于 Excel 分析
- `漏裁统计报告.txt`：按原因、类别、图片聚合的统计
- `missed_images_<reason>.txt`：按原因分文件的漏裁原图绝对路径列表，每行一张图
- `viz/`：按原因分目录的可视化图，图中同时画出所有部件检测框（青色虚线）和漏裁缺陷

### 设计要点

- 以缺陷 XML 为遍历入口，确保部件模型未检出的图片也能被统计。
- 缺陷与 `crop_mapping` 采用三级匹配：精确匹配 → IoU≥0.9 兜底 → 两坐标匹配兜底。
- 挂点（gd）的 `class_mapping` 会和裁切程序保持一致，在过滤后应用。
- 配置外缺陷、无效 bbox 会被忽略并打印警告。

## 裁切缺陷子图，保存的json字段含义

### 正样本 `crop_mapping_正样本.json`
```json
{
  "original_img_path": "原图完整路径",
  "original_defect_xml": "原图对应的缺陷XML文件名",
  "part_bbox": "在原图中裁切的部件区域坐标 [xmin, ymin, xmax, ymax]",
  "defects_at_crop_time": [
    {
      "name": "缺陷类别名称",
      "bbox_in_crop": "缺陷在裁切后子图中的坐标 [xmin, ymin, xmax, ymax]",
      "bbox_in_source": "缺陷在原图中裁切后的坐标 [xmin, ymin, xmax, ymax]",
      "bbox_in_source_original": "缺陷在原图中的原始标注坐标（未裁切） [xmin, ymin, xmax, ymax]",
      "was_clipped": "缺陷框是否被裁切过",
      "overlap_ratio": "裁切面积占比（只有 >= threshold 的缺陷才会被保留）"
    }
  ]
}
```

### 负样本 `crop_mapping_负样本.json`
```json
{
  "original_img_path": "原图完整路径",
  "original_defect_xml": "原图对应的缺陷XML文件名",
  "part_bbox": "在原图中裁切的部件区域坐标 [xmin, ymin, xmax, ymax]",
  "defects_at_crop_time": "裁切时保留的所有缺陷详细信息；负样本没缺陷（默认空列表）",
  "is_negative_sample": "true,标识这是个负样本（正样本没有此参数）"
}
```