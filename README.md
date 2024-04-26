# dtapars
## Introduction
- dtapars 是对于yolo的数据集进行处理和划分的脚本集合
## installer
    - git clone https://github.com/zheng0116/dtapars.git
    - pip install -r requirements.txt
    - cd parsing
## using
    - 使用以下命令来运行单个脚本：
        - python extract_dataname_to_labels.py #车牌数据集提取labels
        - python xml_to_txt.py #xml to yolo 划分
        - python json_to_yolo.py #json to yolo pose
