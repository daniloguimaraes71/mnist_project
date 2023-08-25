"""
モデルユーティリティモジュール

このモジュールには、MNIST Deep Learning プロジェクトのためのモデル関連のユーティリティ関数が含まれています。
モデルのバージョン管理、新しいバージョンの生成、モデルウェイトの保存・読み込みなどの機能を提供します。

使用方法:
    1. モジュールをインポート: `import utils.model_utils as model_utils`
    2. 新しいモデルバージョンの取得: `new_version = model_utils.get_model_new_version(save_dir)`
    3. すべてのモデルバージョンの取得: `all_versions = model_utils.get_all_model_versions(model_save_dir)`
    4. 最新のモデルバージョンの取得: `latest_version = model_utils.get_latest_model_version(save_dir)`
    5. モデルウェイトの保存: `model_utils.save_model(model, save_dir, version)`
    6. モデルウェイトの読み込み: `model_utils.load_model(model, model_path)`

このモジュールは、モデルのバージョン管理やウェイトの操作を効率的かつ一貫性のある方法で行うのに役立ちます。
"""

import os
import glob
import torch
import datetime

def get_model_new_version(save_dir):
    """
    新しいモデルバージョンを計算します。

    Args:
        save_dir (str): モデルが保存されるディレクトリ

    Returns:
        int: 新しいモデルバージョン
    """
    os.makedirs(save_dir, exist_ok=True)
    existing_versions = [int(version) for version in os.listdir(save_dir) if version.isdigit()]
    if existing_versions:
        return max(existing_versions) + 1
    return 1

def get_all_model_versions(model_save_dir):
    """
    指定されたディレクトリ内のすべてのモデルバージョンのリストを取得します。

    Args:
        model_save_dir (str): モデルが保存されるディレクトリ

    Returns:
        list: すべてのモデルバージョンのリスト
    """
    model_versions = []
    for filename in os.listdir(model_save_dir):
        if filename.startswith('model_v'):
            version = int(filename.split('_')[1][1:])
            model_versions.append(version)
    return sorted(model_versions)


def get_latest_model_version(save_dir):
    """
    最新のモデルバージョンを取得します。

    Args:
        save_dir (str): モデルが保存されるディレクトリ

    Returns:
        int: 最新のモデルバージョン
    """
    existing_versions = [int(version) for version in os.listdir(save_dir) if version.isdigit()]
    if existing_versions:
        return max(existing_versions)
    return None

def save_model(model, save_dir, version, epoch):
    """
    モデルウェイトを保存します。

    Args:
        model (nn.Module): 保存するモデル
        save_dir (str): モデルが保存されるディレクトリ
        version (int): 保存するモデルのバージョン
        epoch (int): 保存するモデルの学習エポック数

    Returns:
        None
    """
    version_folder = os.path.join(save_dir, str(version))
    os.makedirs(version_folder, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = os.path.join(version_folder, f"model_version_{version}_epoch{epoch}_{timestamp}.pt")
    torch.save(model.state_dict(), model_filename)

def load_model(model, model_path):
    """
    モデルウェイトをファイルから読み込みます。

    Args:
        model (nn.Module): 読み込まれるモデル
        model_path (str): モデルウェイトファイルのパス

    Returns:
        None
    """

    if torch.cuda.is_available():
        # GPUが利用可能な場合
        device = torch.device('cuda')
        model.to(device)
        try:
            model.load_state_dict(torch.load(model_path))
        except RuntimeError as e:
            if "Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False" in str(e):
                print("An error occurred: CUDA device is not available. Loading model on CPU.")
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            else:
                raise e
    else:
        # GPUが利用不可の場合
        device = torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()
