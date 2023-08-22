"""
MNISTデータローダーユーティリティモジュール

このモジュールには、MNISTデータセットの前処理とデータローダーの取得を行う関数が含まれています。
トレーニングやテストデータのデータローダーを簡単に取得するために使用されます。

使用方法:
    1. モジュールをインポート: `import utils.data_loader as data_loader`
    2. データローダーを取得:
       - トレーニングデータローダー: `train_loader = data_loader.get_mnist_data_loader(batch_size, is_train=True)`
       - テストデータローダー: `test_loader = data_loader.get_mnist_data_loader(batch_size, is_train=False)`

このモジュールは、MNISTデータセットの前処理を行い、データローダーを取得するのに役立ちます。
"""

import torch
from torchvision import datasets, transforms

def get_mnist_data_loader(batch_size, is_train=True):
    """
    MNISTデータローダーを取得します。

    Args:
        batch_size (int): バッチサイズ
        is_train (bool): トレーニングデータかどうかのフラグ

    Returns:
        DataLoader: データローダーオブジェクト
    """
    # 前処理の定義
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # MNISTデータセットの読み込みとデータローダーの作成
    dataset = datasets.MNIST('data/mnist_data', train=is_train, transform=transform, download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return data_loader
