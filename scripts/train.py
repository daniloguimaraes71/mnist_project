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
    
    # データセットを訓練データと検証データに分割
    total_size = len(train_loader.dataset)
    val_size = int(total_size * config["validation_split"])
    train_size = total_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_loader.dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # モデルの初期化
    model = NeuralNetworkModel()

    # モデルの訓練
    trainer.train_model(model, train_loader, val_loader, config)  # Pass val_loader to train_model
