"""
モデル訓練スクリプト

このスクリプトは、MNISTモデルの訓練を行うために使用されます。訓練データローダーを作成し、指定された設定でモデルの訓練を実行します。

使用方法:
    1. スクリプトをインポート: `from scripts.train import train`
    2. 訓練を実行:
       - コンフィグファイルをロード: `config = load_config("config/config.json")`                                                                                                                                                                                
       - 学習を実行: `train(config)`
"""

import json
from utils.data_loader import get_mnist_data_loader
from models.model import NeuralNetworkModel
import utils.trainer as trainer
from config.config import load_config
import torch

def train(config):
    """
    モデルの訓練を実行します。

    この関数は、指定された設定でMNISTモデルの訓練を行います。
    
    Args:
        config (dict): 訓練のための設定が含まれる辞書

    Returns:
        None
    """
    train_loader = get_mnist_data_loader(config["batch_size"], is_train=True)

    # モデルの初期化
    model = NeuralNetworkModel()

    # モデルの訓練
    trainer.train_model(model, train_loader, config)
