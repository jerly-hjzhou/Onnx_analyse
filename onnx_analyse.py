import onnx
from onnx import TensorProto
import os
import yaml
import argparse
import csv
from abc import ABC, abstractmethod

class OpInfo:
    def __init__(self, op_name, freq, since_version, model_name):
        self.op_name = op_name
        self.freq = freq
        self.since_version = since_version
        self.model_name = model_name


class ONNXModelInfo:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = onnx.load(model_path)

    def get_model_name(self):
        model_name = os.path.basename(self.model_path)
        return model_name

    def get_opset(self):
        imports = self.model.opset_import
        opset = []
        for imp in imports:
            domain = imp.domain if imp.domain else "ai.onnx"
            version = imp.version
            opset.append(f"{domain} v{version}")
        opset = [f"ai.onnx v{imp.version}" for imp in imports]
        model_opset = ",".join(opset)
        return model_opset

    def get_ir_version(self):
        ir_version = self.model.ir_version
        return f"ONNX v{ir_version}"

    def get_producer(self):
        name = self.model.producer_name
        version = self.model.producer_version
        if name == "" and version == "": producer = "Absence"
        else: producer = f"{name} {version}"
        return producer

    @staticmethod
    def get_readable_data_type(data_type):
        data_type_names = {
            TensorProto.FLOAT: "float32",
            TensorProto.UINT8: "uint8",
            TensorProto.INT8: "int8",
            TensorProto.UINT16: "uint16",
            TensorProto.INT16: "int16",
            TensorProto.INT32: "int32",
            TensorProto.INT64: "int64",
            TensorProto.STRING: "string",
            TensorProto.BOOL: "bool",
            TensorProto.FLOAT16: "float16",
            TensorProto.DOUBLE: "double",
            TensorProto.UINT32: "uint32",
            TensorProto.UINT64: "uint64",
        }
        return data_type_names.get(data_type, "Absence")

    def get_input_dtype(self):
        input_types = set()
        for input in self.model.graph.input:
            elem_type = input.type.tensor_type.elem_type
            readable_type = self.get_readable_data_type(elem_type)
            input_types.add(readable_type)
        return ",".join(input_types)

    def get_op_counts(self):
        op_counts = {}  
        
        for node in self.model.graph.node:
            op_type = node.op_type
            if op_type in op_counts:
                op_counts[op_type] += 1
            else:
                op_counts[op_type] = 1
        
        return op_counts


class CSVExporter(ABC):

    def __init__(self, model_paths, output_name):
        self.model_paths = model_paths
        self.output_name = output_name

    @abstractmethod
    def get_header_data(self):
        """
        get header and data for csv file from onnx model_paths
        
        Returns:
        header[list] and datas[2d list]
        """
        pass

    def export(self):
        header, datas = self.get_header_data()
        with open(self.output_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(datas)


class ModelInfosCSV(CSVExporter):

    def __init__(self, model_paths, output_name):
        super().__init__(model_paths, output_name)

    def get_header_data(self):

        header = [
            'idx', 'model_name', 'opset', 'ir_version', 'producer',
            'input_dtype'
        ]
        datas = []
        idx = 1
        for path in self.model_paths:
            data = []
            model = ONNXModelInfo(path)
            data.append(idx)
            data.append(model.get_model_name())
            data.append(model.get_opset())
            data.append(model.get_ir_version())
            data.append(model.get_producer())
            data.append(model.get_input_dtype())
            datas.append(data)
            idx += 1

        return header, datas


class OpsListCSV(CSVExporter):

    def __init__(self, model_paths, output_name):
        super().__init__(model_paths, output_name)
        schemas = onnx.defs.get_all_schemas()
        self.onnx_ops = {schema.name for schema in schemas}

    def get_header_data(self):
        header = ['idx', 'op_type', 'is_standard']
        datas = []
        op_names = dict()
        for path in self.model_paths:
            model = ONNXModelInfo(path)
            op_names.update(model.get_op_counts())
        idx = 1
        for op in sorted(op_names):
            data = []
            data.append(idx)
            data.append(op)
            data.append("Yes" if op in self.onnx_ops else "No")
            datas.append(data)
            idx += 1

        return header, datas

class OpsInfoCSV(CSVExporter):

    def __init__(self, model_paths, output_name):
        super().__init__(model_paths, output_name)
        schemas = onnx.defs.get_all_schemas()
        self.onnx_ops = {schema.name for schema in schemas}

    def get_header_data(self):
        header = ['idx', 'op_type', 'freq', 'opsets', 'from_model', "is_standard"]
        datas = []
        ops = []
        for path in self.model_paths:
            model = ONNXModelInfo(path)
            ops_dict = model.get_op_counts()
            for op_name in ops_dict:
                op_info=OpInfo(op_name, ops_dict[op_name], model.get_opset(), model.get_model_name())
                ops.append(op_info)

        idx = 1
        for op in sorted(ops, key=lambda op: op.freq, reverse=True):
            data = []
            data.append(idx)
            data.append(op.op_name)
            data.append(op.freq)
            data.append(op.since_version)
            data.append(op.model_name)
            data.append("Yes" if op.op_name in self.onnx_ops else "No")
            datas.append(data)
            idx += 1

        return header, datas

class Config:

    def __init__(self, config_path):
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        self.__model_paths = []
        for file_path in config:
            temp_paths = self.find_onnx_paths(file_path)
            for model_path in temp_paths:
                self.__model_paths.append(model_path)

    def find_onnx_paths(self, file_path):
        """
        extract paths of onnx model from folder
        
        Parameters:
        filepath - may be onnx model path or folder path which contains onnx model
        
        Returns[list]:
        onnx model path
        """
        if os.path.isdir(file_path):
            onnx_files = []
            for root, dirs, files in os.walk(file_path):
                for file in files:
                    if file.endswith('.onnx'):
                        onnx_files.append(os.path.join(root, file))
            return onnx_files
        elif file_path.endswith('.onnx'):
            return [file_path]
        else:
            return []

    def get_model_paths(self):
        return self.__model_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        help='configure path of onnx models')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.config is None:
        model_paths = [f for f in os.listdir('.') if f.endswith('.onnx')]
    else:
        model_paths = Config(args.config).get_model_paths()
    model_info = ModelInfosCSV(model_paths, "model_information.csv")
    model_info.export()
    op_list = OpsListCSV(model_paths, "op_type_list.csv")
    op_list.export()
    op_info = OpsInfoCSV(model_paths, "op_info.csv")
    op_info.export()