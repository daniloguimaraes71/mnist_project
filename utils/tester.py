"""
モデルテスターユーティリティモジュール

このモジュールには、モデルのテストを行うための関数が含まれています。
テストデータでモデルを評価し、テスト損失と正解率を計算して保存します。

使用方法:
    1. モジュールをインポート: `import utils.tester as tester`
    2. model: テスト対象のモデルを用意
    3. test_loader: テストデータのデータローダーを用意
    4. config: 設定情報を含む辞書を準備
    5. model_version: モデルのバージョンを指定。デフォルト値は最新バージョン(latest)
    5. テストを実行: `tester.test_model(model, test_loader, config, model_version)`

このモジュールは、モデルのテストプロセスを自動化し、テスト結果を保存するのに役立ちます。
"""

import os
import glob
import torch
import torch.nn as nn
import utils.model_utils as model_utils
import logging
import os
import json

def test_model(model, test_loader, config, model_version='latest'):
    """
    テストデータでモデルのテストを実行します。

    Args:
        model (nn.Module): テスト対象のモデル
        test_loader (DataLoader): テストデータのデータローダー
        config (dict): 設定情報が含まれる辞書
        model_version (str): テスト対象のモデルバージョン（最新の'latest'または特定のバージョン）

    Returns:
        None
    """
    # GPUが利用可能かどうかを確認し、デバイスを設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 損失関数の設定                                                           
    criterion = nn.CrossEntropyLoss()

    if model_version == 'latest':
        # 最新のモデルバージョンを取得
        model_version = model_utils.get_latest_model_version(config["model_save_dir"])
    
    # モデルバージョンごとにテストを実行
    model_folder = os.path.join(config["model_save_dir"], str(model_version))
    model_paths = glob.glob(os.path.join(model_folder, f"model_version{model_version}_*.pt"))

    model_paths.sort(key=lambda f: (os.path.getmtime(f)))

    # テストを開始
    logging.info('Testing started...')
    logging.info(f'Model Version {model_version}')
    test_losses = []
    test_accuracy = []
    
    for model_path in model_paths:
        # モデルウェイトを読み込む
        model_utils.load_model(model, model_path)
    
        correct = 0
        total = 0
        predictions = []

        with torch.no_grad():
            running_loss = 0.0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predictions.extend(predicted.cpu().numpy())
                running_loss += loss.item()
            accuracy = correct / total
            average_loss = running_loss / len(test_loader)
            model_file_name = model_path.split('/')[-1]
            model_epoch_number = int(model_file_name.split('_')[2][5:])
            
            logging.info(f'Model epoch {model_epoch_number} - Test Accuracy: {accuracy:.4f} Test loss: {average_loss:.4f}')

            test_losses_dict = {f'epoch{model_epoch_number}':average_loss}
            test_accuracy_dict = {f'epoch{model_epoch_number}':accuracy}

            # 各モデルのテスト損失と正解率を保存
            test_losses.append(test_losses_dict)
            test_accuracy.append(test_accuracy_dict)

            metrics_folder = f"{config['model_save_dir']}{model_version}/metrics/"
            os.makedirs(metrics_folder, exist_ok=True)

            # テスト損失をJSONファイルに保存 
            losses_filename = os.path.join(metrics_folder, f'test_losses.json')
            with open(losses_filename, 'w') as f:
                json.dump(test_losses, f)

            # テスト精度をJSONファイルに保存
            accuracy_filename = os.path.join(metrics_folder, f'test_accuracy.json')
            with open(accuracy_filename, 'w') as f:
                json.dump(test_accuracy, f)
    logging.info('Testing for all weights completed.')
