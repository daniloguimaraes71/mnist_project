"""
Neural Network モデル定義

このモジュールには、MNIST データセットの画像分類に使用するニューラルネットワークモデルが定義されています。
モデルは、3つの全結合層（fully connected layers）から構成されており、入力画像を10個のクラスに分類します。

使用方法:
    1. モデルをインポート: `from models.model import NeuralNetworkModel`
    2. モデルのインスタンスを作成: `model = NeuralNetworkModel()`
    3. フォワードパスの実行: `outputs = model(inputs)`
"""

import torch.nn as nn

class NeuralNetworkModel(nn.Module):
    def __init__(self):
        # モデルの層を定義
        super(NeuralNetworkModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        # フォワードパスの実装
        x = x.view(-1, 28 * 28)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
