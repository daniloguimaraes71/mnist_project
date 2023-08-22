"""
モデルテストスクリプト

このスクリプトは、指定されたモデルバージョンのMNISTモデルをテストするために使用されます。テストデータローダーを作成し、指定されたモデルバージョンのモデルを読み込み、テストを実行します。

使用方法:
    1. スクリプトをインポート: `from scripts.test import test`
    2. テストを実行:
       - コンフィグファイルをロード: `config = load_config("config/config.json")`
       - テストするモデルバージョンを指定: `model_version = '1'`
       - テストを実行: `test(config, model_version)`
"""

import json
from utils.data_loader import get_mnist_data_loader
from models.model import NeuralNetworkModel
import utils.tester as tester
from config.config import load_config
import torch
import logging

def test(config, model_version):
    """
    モデルのテストを実行します。

    この関数は、指定されたモデルバージョンのMNISTモデルに対してテストを実行します。
    
    Args:
        config (dict): テストのための設定が含まれる辞書
        model_version (str): テストするモデルのバージョン

    Returns:
        None
    """
    test_loader = get_mnist_data_loader(config["batch_size"], is_train=False)

    # モデルの初期化
    model = NeuralNetworkModel()

    # モデルのテスト
    tester.test_model(model, test_loader, config, model_version)
