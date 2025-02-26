import os
import json
import numpy as np
import h5py
from joblib import Parallel, delayed
import sys
sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.macro import *

DATA_ROOT = "data"  # 调整为项目根目录下的 data
RAW_DATA = os.path.join(DATA_ROOT, "cad_json")
RECORD_FILE = os.path.join(DATA_ROOT, "train_val_test_split.json")
SAVE_DIR = os.path.join(DATA_ROOT, "cad_vec")

MAX_N_EXT = 10    # 最大拉伸操作数
MAX_N_LOOPS = 5   # 最大循环操作数
MAX_N_CURVES = 20 # 最大曲线数
MAX_TOTAL_LEN = 100 # 最大序列长度

print(SAVE_DIR)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def process_one(data_id):
    print(f"Processing ID: {data_id}")
    json_path = os.path.join(RAW_DATA, data_id + ".json")
    # 尝试按子目录格式处理
    if "/" in data_id:
        folder, file = data_id.split("/")
        json_path = os.path.join(RAW_DATA, folder, f"{file}.json")
    print(f"Looking for file: {json_path}")
    try:
        with open(json_path, "r") as fp:
            data = json.load(fp)
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        cad_seq.numericalize()
        cad_vec = cad_seq.to_vector(MAX_N_EXT, MAX_N_LOOPS, MAX_N_CURVES, MAX_TOTAL_LEN, pad=True)
    except FileNotFoundError as e:
        print(f"File not found: {json_path}, Error: {str(e)}")
        return None
    except Exception as e:
        print(f"failed: {data_id}, Error: {str(e)}")
        return None
    if MAX_TOTAL_LEN < cad_vec.shape[0] or cad_vec is None:
        print(f"exceed length condition: {data_id}, length: {cad_vec.shape[0]}")
        return None
    return cad_vec

# 收集所有向量并保存到单个 vectors.h5 文件
with open(RECORD_FILE, "r") as fp:
    all_data = json.load(fp)

all_vectors = []
for split in ["train", "validation", "test"]:
    vectors = Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data[split])
    all_vectors.extend([v for v in vectors if v is not None])

if all_vectors:
    with h5py.File(os.path.join(SAVE_DIR, "vectors.h5"), 'w') as fp:
        fp.create_dataset("vectors", data=np.stack(all_vectors, axis=0), dtype=np.int)
    print(f"Generated vectors.h5 with shape {np.stack(all_vectors, axis=0).shape}")
else:
    print("No valid vectors generated.")