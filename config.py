#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Created on 2021/2/19 16:40
# @Author: CaoYugang
START_TAG = "<START>"
STOP_TAG = "<STOP>"
SPACE_WORD = "o"  # 无关字符标识符
EMBEDDING_DIM = 256  # Embedding层数
HIDDEN_DIM = 256  # LSTM隐藏层数

TAG_IO_IX = {"o": 0, "$": 1, START_TAG: 2, STOP_TAG: 3}  # 标签对应字典
TAG_IO_CN = {1: "实体"}  # 标签值对应标签中文

LR = 0.001
TRAIN_FILE_PATH = "data/train_data.txt"  # 训练数据集
MODEL_PATH = "model/"
EPOCH = 100  # 训练循环次数
LOG_PRINT_INDEX = 100  # 日志打印统计
MODEL_SAVE_INDEX = 5000  # 模型保存训练次数间隔

# 测试选用模型
TEST_MODEL = "bi-lstm-crf-10-0.pkl"
