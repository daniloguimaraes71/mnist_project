"""
設定ファイル読み込みユーティリティモジュール

このモジュールには、コンフィグファイルから設定値を読み込むための関数が含まれています。
モデルのトレーニングやテストに必要な設定情報をコンフィグファイルから取得する際に使用されます。

使用方法:
    1. モジュールをインポート: `import config.config as config`
    2. コンフィグファイルのパスを指定して設定を読み込む: `config_data = config.load_config('path/to/config.json')`

コンフィグファイルは、モデルのパラメータ、データパス、バッチサイズなどの設定を保持します。
"""

import json

def load_config(config_path):
    """
    コンフィグファイルから設定値を読み込む

    指定されたコンフィグファイルから設定情報を読み込み、辞書形式で返します。

    Args:
        config_path (str): コンフィグファイルのパス

    Returns:
        dict: 設定情報の辞書
    """
    
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config
