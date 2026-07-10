# yh_modify_defect_datasets

缺陷识别数据集构建的相关代码
1、总库裁切缺陷子图
2、缺陷数据清洗后，标签映射回总库


# 裁切缺陷子图，保存的json字段含义
#正样本  crop_mapping_正样本.json
{
  "original_img_path":       # 原图完整路径
  "original_defect_xml":     # 原图对应的缺陷XML文件名
  "part_bbox":       # 在原图中裁切的部件区域坐标 [xmin, ymin, xmax, ymax] 
  "defects_at_crop_time": [         #含义：裁切时保留的所有缺陷详细信息；   类型: 列表，每个元素是一个缺陷对象
    {
	  "name": 		# 缺陷类别名称
	  "bbox_in_crop":      # 缺陷在裁切后子图中的坐标 [xmin, ymin, xmax, ymax] 
	  "bbox_in_source":   # 缺陷在原图中裁切后的坐标[xmin, ymin, xmax, ymax]
	  "bbox_in_source_original":       # 缺陷在原图中的原始标注坐标（未裁切）[xmin, ymin, xmax, ymax]
	  "was_clipped":        # 缺陷框是否被裁切过
	  "overlap_ratio":      # 缺陷有效区域 / 原始缺陷总面积 裁切面积占比（只有 >= threshold 的缺陷才会被保留）
    }
  ]
}


#负样本  crop_mapping_负样本.json
{
  "original_img_path":       # 原图完整路径
  "original_defect_xml":     # 原图对应的缺陷XML文件名
  "part_bbox":       # 在原图中裁切的部件区域坐标 [xmin, ymin, xmax, ymax] 
  "defects_at_crop_time": [],         #裁切时保留的所有缺陷详细信息；负样本没缺陷（默认空列表）
  "is_negative_sample": true 	  #标识这是个负样本（正样本没有此参数）
}