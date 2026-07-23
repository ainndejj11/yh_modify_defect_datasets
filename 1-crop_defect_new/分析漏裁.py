#!/usr/bin/env python3
"""
漏裁缺陷独立分析脚本

基于原图缺陷标注、部件检测标注和 crop_mapping_正样本.json，
统计原图中未被裁切到的缺陷目标，并按原因归集：
  - component_not_covered: 部件模型未覆盖
  - incomplete_or_filtered: 部件已包含但缺陷不完整/未达面积阈值
  - data_inconsistency: 按理应裁切但未在 crop_mapping 中匹配到

输出：
  - missed_defects.json
  - missed_defects.csv
  - 漏裁统计报告.txt
  - viz/（可选）
"""

import os
import sys
import argparse
from pathlib import Path

from defect_coverage_analyzer import run_analysis


def infer_crop_mapping_path(output_dir):
    """从输出目录推断 crop_mapping_正样本.json 路径"""
    output_dir = Path(output_dir)
    candidate = output_dir / 'crop_mapping_正样本.json'
    if candidate.exists():
        return str(candidate)
    return None


def main():
    parser = argparse.ArgumentParser(
        description='漏裁缺陷独立分析脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整分析（含可视化）
  python 分析漏裁.py ddx \
    --images-dir /raid/datasets_defect_2026/全图测试集/images \
    --component-ann /raid/datasets_defect_2026/全图测试集/部件_导地线_xml\
    --defect-ann /raid/datasets_defect_2026/全图测试集/Annotations  \
    --crop-mapping /raid/datasets_defect_2026/datasets_val/全量_正样本/dx_data/crop_mapping_正样本.json \
    --output /raid/datasets_defect_2026/全图测试集/测试wtj \
    --viz

    
  # 只输出报告，不生成可视化（显式指定crop_mapping）
  python 分析漏裁.py gd \
    --images-dir /raid/Nas-122/项目数据/输电项目/缺陷/标记样本库/训练集/JPEGImages \
    --image-index /raid/wtj/ultralytics-8.4.6/缺陷识别-模型优化v7.0/1-总库图像进行部件检测/image_indexs_20260114.pkl \
    --component-ann /raid/Nas-122/项目数据/输电项目/缺陷/标记样本库/训练集/部件xmls \
    --defect-ann /raid/Nas-122/项目数据/输电项目/缺陷/标记样本库/训练集/Annotations \
    --crop-mapping /raid/datasets_defect_2026/datasets_train/全量_正样本/gd_data/crop_mapping_正样本.json \
    --output /raid/datasets_defect_2026/datasets_train/全量_正样本/gd_data/输出漏裁信息0723


支持的部件类型: ddx(导地线), gt(杆塔), jyz(绝缘子), gd(挂点)
        """
    )

    parser.add_argument(
        'component_type',
        choices=['ddx', 'gt', 'jyz', 'gd'],
        help='部件类型: ddx(导地线), gt(杆塔), jyz(绝缘子), gd(挂点)'
    )

    parser.add_argument(
        '--images-dir',
        required=False,
        default=None,
        help='原图目录路径'
    )

    parser.add_argument(
        '--image-index',
        required=False,
        default=None,
        help='图片索引pkl文件路径（可选，不指定时自动扫描images-dir文件夹）'
    )

    parser.add_argument(
        '--component-ann',
        required=True,
        help='部件检测XML目录路径'
    )

    parser.add_argument(
        '--defect-ann',
        required=True,
        help='缺陷标注XML目录路径'
    )

    parser.add_argument(
        '--crop-mapping',
        required=False,
        default=None,
        help='crop_mapping_正样本.json 路径（可选，默认从 --output 目录推断）'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='输出目录路径'
    )

    parser.add_argument(
        '--config',
        default='./config.yaml',
        help='配置文件路径 (默认: ./config.yaml)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=32,
        help='线程数 (默认: 24)'
    )

    parser.add_argument(
        '--viz',
        action='store_true',
        help='生成漏裁可视化图'
    )

    parser.add_argument(
        '--viz-max',
        type=int,
        default=0,
        help='最多生成多少张可视化图，0 表示不限制 (默认: 0)'
    )

    args = parser.parse_args()

    # 检查路径
    if not os.path.exists(args.component_ann):
        print(f"❌ 错误: 部件检测XML目录不存在: {args.component_ann}")
        sys.exit(1)

    if not os.path.exists(args.defect_ann):
        print(f"❌ 错误: 缺陷标注XML目录不存在: {args.defect_ann}")
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"❌ 错误: 配置文件不存在: {args.config}")
        sys.exit(1)

    # 推断 crop_mapping 路径
    crop_mapping_path = args.crop_mapping
    if crop_mapping_path is None:
        crop_mapping_path = infer_crop_mapping_path(args.output)
        if crop_mapping_path is None:
            print(f"❌ 错误: 无法从 {args.output} 推断 crop_mapping_正样本.json，请显式指定 --crop-mapping")
            sys.exit(1)
        print(f"📦 自动推断 crop_mapping: {crop_mapping_path}")

    if not os.path.exists(crop_mapping_path):
        print(f"❌ 错误: crop_mapping 文件不存在: {crop_mapping_path}")
        sys.exit(1)

    # 可视化需要图片目录
    if args.viz and not args.images_dir:
        print("❌ 错误: 开启 --viz 时必须指定 --images-dir")
        sys.exit(1)

    if args.images_dir and not os.path.exists(args.images_dir):
        print(f"❌ 错误: 原图目录不存在: {args.images_dir}")
        sys.exit(1)

    if args.image_index and not os.path.exists(args.image_index):
        print(f"❌ 错误: 图片索引文件不存在: {args.image_index}")
        sys.exit(1)

    # 打印参数
    print("\n" + "=" * 70)
    print("🔧 漏裁分析运行参数")
    print("=" * 70)
    print(f"  部件类型: {args.component_type}")
    print(f"  原图目录: {args.images_dir if args.images_dir else '未指定'}")
    if args.image_index:
        print(f"  图片索引: {args.image_index} (pkl模式)")
    elif args.images_dir:
        print(f"  图片索引: 从原图目录扫描 (文件夹模式)")
    print(f"  部件XML目录: {args.component_ann}")
    print(f"  缺陷XML目录: {args.defect_ann}")
    print(f"  crop_mapping: {crop_mapping_path}")
    print(f"  输出目录: {args.output}")
    print(f"  配置文件: {args.config}")
    print(f"  线程数: {args.workers}")
    print(f"  可视化: {'是' if args.viz else '否'}")
    if args.viz:
        print(f"  可视化上限: {args.viz_max}")
    print("=" * 70 + "\n")

    # 执行分析
    run_analysis(
        component_type=args.component_type,
        defect_ann_dir=args.defect_ann,
        component_ann_dir=args.component_ann,
        crop_mapping_path=crop_mapping_path,
        config_path=args.config,
        output_dir=args.output,
        images_dir=args.images_dir,
        image_pkl_path=args.image_index,
        max_workers=args.workers,
        enable_viz=args.viz,
        viz_max=args.viz_max
    )

    print("\n" + "=" * 70)
    print("🎉 漏裁分析完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
