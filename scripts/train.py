"""
モデルトレーニングスクリプト

このスクリプトは、MNISTモデルのトレーニングを行うために使用されます。トレーニングデータローダーを作成し、指定された設定でモデルのトレーニングを実行します。

使用方法:
    1. スクリプトをインポート: `from scripts.train import train`
    2. トレーニングを実行:
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
    モデルのトレーニングを実行します。

    この関数は、指定された設定でMNISTモデルのトレーニングを行います。
    
    Args:
        config (dict): トレーニングのための設定が含まれる辞書

    Returns:
        None
    """
    train_loader = get_mnist_data_loader(config["batch_size"], is_train=True)

    # モデルの初期化
    model = NeuralNetworkModel()

    # モデルのトレーニング
    trainer.train_model(model, train_loader, config)
