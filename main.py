"""
MNIST Deep Learning Model

このスクリプトは、MNISTデータセットを使用してディープラーニングモデルをトレーニングおよびテストするためのコードです。

Usage:
    python main.py --mode train
    python main.py --mode test --model_version <model_version>

Arguments:
    --mode (str): 'train'または'test'を指定して、トレーニングモードまたはテストモードを選択します。
    --model_version (int): テストするモデルのバージョンを指定します。指定しないと、最新のモデルがテストされます。

Example:
    トレーニングモードの実行:
    python main.py --mode train

    特定のモデルバージョンでのテスト:
    python main.py --mode test --model_version 3

    最新のモデルでのテスト:
    python main.py --mode test

"""

import argparse
import json
from scripts.train import train
from scripts.test import test
from utils import logger
import logging
from config.config import load_config
import os

def main():
    try:
        # コマンドライン引数のパーサーを作成
        parser = argparse.ArgumentParser(description="MNIST Deep Learning Model")
        parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="Choose 'train' for training mode or 'test' for testing mode")
        parser.add_argument('--model_version', type=str, help='Model version for testing')
        args = parser.parse_args()

        # コンフィグファイルをロード
        config = load_config("config/config.json")

        # ログを設定
        logger.setup_logger(config["log_dir"])
        
        # トレーニングまたはテストモードの実行
        if args.mode == "train":
            train(config)

        elif args.mode == "test":
            if args.model_version is not None:
                logging.info(f'Testing model version: {args.model_version}')
                model_version = args.model_version
                model_folder = os.path.join(config["model_save_dir"], str(model_version))
                if not os.path.exists(model_folder):
                    logging.error(f'Model version {model_version} does not exist.')
                    return
                test(config, model_version)
            else:
                test(config, model_version='latest')

    except Exception as e:
        logging.error(f'An error occurred: {e}')

if __name__ == "__main__":
    main()
