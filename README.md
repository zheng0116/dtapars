# dtapars
## 介绍
dtapars 是对于yolo的数据集进行处理和划分的脚本集合
## 安装
### 克隆项目
```bash 
git clone https://github.com/zheng0116/dtapars.git
```
### 安装依赖
```bash
pip install -r requirements.txt
```
### 切换路径
```bach
cd cd parsing
```
## 使用方法
### 使用以下命令来运行单个脚本：
```bach
 python extract_dataname_to_labels.py #车牌数据集提取labels 
```
### xml 转换成yolo
```bach
 python xml_to_txt.py 
```
### json转换成yolo
```bach
 python json_to_yolo.py #json to yolo pose
```


