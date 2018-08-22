# 布匹缺陷检测解题思路

https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11165268.5678.1.421964aaxoXX96&raceId=231666

- 目标检测方法
- 目标分类方法

## 训练数据：
- 数据统计

|批次|(有缺陷)图片数|缺陷种类（包括正常）数|目标数|
|:---:|:---:|:---:|:---:|
|round1|706|47|971|
|round2|462|48|613|
|总计|2336|60|1584|

|批次|扎洞/边扎洞|毛斑|擦洞|毛洞|织稀|吊经|缺经|跳花|油渍/污渍/黄渍/油污|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|round1|64/16|46|221|67|62|154|49|75|23/29/14/
|round2|8/17|10|127|33|47|123|20|75|17/8/4/1

正常图片：1316

- 数据增广
  - 裁剪，长宽原图1/n，步长为原图1/(n+1)
  
- 检测训练
  - 原图
  - 裁剪：1/2、1/3，若缺陷面积占原缺陷面积>0.5
  - 水平翻转
  - 网络参数：
    - 网络retina-x-101-fpn
    
    ```
    FPN:
      FPN_ON: True
      MULTILEVEL_RPN: True
      RPN_MAX_LEVEL: 5
      RPN_MIN_LEVEL: 3
      COARSEST_STRIDE: 128
      EXTRA_CONV_LEVELS: True
    RESNETS:
      STRIDE_1X1: False  # default True for MSRA; False for C2 or Torch models
      TRANS_FUNC: bottleneck_transformation
      NUM_GROUPS: 64
      WIDTH_PER_GROUP: 4
    RETINANET:
      RETINANET_ON: True
      NUM_CONVS: 2
      ASPECT_RATIOS: (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
      SCALES_PER_OCTAVE: 3
      ANCHOR_SCALE: 4
      LOSS_GAMMA: 2.0
      LOSS_ALPHA: 0.25
    ```
    - 训练参数
    ```
    TRAIN:
      WEIGHTS:  ret-x-101-backbone.pkl
      DATASETS: ('cloth_train',)
      SCALES: (500,800)
      MAX_SIZE: 2333
      IMS_PER_BATCH: 1
      BATCH_SIZE_PER_IM: 32
      RPN_BATCH_SIZE_PER_IM: 32
      RPN_PRE_NMS_TOP_N: 2000  # Per FPN level
      DATASET_CACHE_PATH: 'detectron/datasets/data/cloth/cache/'
      RPN_STRADDLE_THRESH: -1  # default 0
    ```
  
    - 测试参数
      - 原图检测，尺度[1200, 1500, 1920]
      - 裁剪1/2检测
    ```
    TEST:
      DATASETS: ('cloth_train',)
      NMS: 0.3
      RPN_PRE_NMS_TOP_N: 1000  # Per FPN level
      RPN_POST_NMS_TOP_N: 100
    
      # -- Test time augmentation example -- #
      BBOX_AUG:
        ENABLED: True
        SCORE_HEUR: UNION  # AVG NOTE: cannot use AVG for e2e model
        COORD_HEUR: UNION  # AVG NOTE: cannot use AVG for e2e model
        H_FLIP: True
        SCALES: ( 500, 800, 960, 1200, 1920)
        MAX_SIZE: 8000
    
        SCALE_H_FLIP: True
        SCALE_SIZE_DEP: False
        AREA_TH_LO: 50   # 50^2
        AREA_TH_HI: 25000000  # 180^2
        ASPECT_RATIOS: ()
        ASPECT_RATIO_H_FLIP: False
      BBOX_VOTE:
        ENABLED: True
        VOTE_TH: 0.5
    ```
- 分类训练  
  
  训练正常缺陷的二分类器，以得到正常的概率
  
  - 训练数据：
    - 裁剪：1/3、1/4、1/5、1/6，若缺陷面积占原缺陷面积>0.5
  - 训练网络：resnext-50
    - 输入大小：400*400
    - 均值:120，120，120
    - 从头训
  - 测试数据：
    - 裁剪：1/5
    
- 结果处理：
  - 单个检测结果：
    - 每张图获取每个缺陷的最大值
    - 对于裁剪过的图片，丢弃重叠部分在边缘的检测结果
    - 正常概率为：1-\prod(1-defect[i])
  - 结果融合：
    - 每张图获取每个缺陷的最大值
    - 正常概率为：1-\prod(1-defect[i])
  - 分类结果
    
    
    
    
    
  
