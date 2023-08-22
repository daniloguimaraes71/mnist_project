"""
ロギングユーティリティモジュール

このモジュールには、MNIST Deep Learning プロジェクトのためのログ設定関数が含まれています。
ログの出力、フォーマット、およびログレベルのカスタマイズオプションを提供します。

使用方法:
    1. モジュールをインポート: `import utils.logger as logger`
    2. ロギングの設定: `logger.setup_logger(log_dir)`
       ここで、`log_dir` はログファイルが保存されるディレクトリです。
    3. ロガーの使用: `logging.info('ログメッセージ')` など。

このモジュールは、プロジェクト全体で一貫性のある情報を提供するログを維持するのに役立ちます。
"""

import logging
import os

def setup_logger(log_dir):
    """
    ロガーの設定を行います。

    Args:
        log_dir (str): ログファイルが保存されるディレクトリです。

    Returns:
        None
    """
    # ログの設定
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, 'output.log')

    # ログの書き込み先とレベルの設定
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[
                            logging.FileHandler(log_filename),
                            logging.StreamHandler()
                        ])

    logger = logging.getLogger()
    return logger
