# 模型信息分析工具

## 功能

生成三张onnx模型相关信息的统计数据。
* model_infomation表统计给定的所有onnx模型路径中的模型名称、opset、ir_version、producer、输入数据类型等信息
* op_info表统计各种类型算子在某一个onnx模型中的出现次数、其opset、是否为原生op
* op_type_list表统计所有onnx模型的算子类型、是否为原生op

## 使用方式

支持用户不同方式的输入1、支持配置文件指定模型路径  （用户保证传入绝对路径，可能是目录，也可能是带路径的具体模型）2、若未传入配置文件，默认获取当前脚本所在路径所有模型

比如`python onnx_analyse.py --config model_config.yaml`