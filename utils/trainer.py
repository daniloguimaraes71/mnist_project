"""
モデルトレーナーユーティリティモジュール

このモジュールには、モデルのトレーニングを行うための関数が含まれています。
トレーニングデータを使用してモデルを学習し、トレーニング損失を記録して保存します。

使用方法:
    1. モジュールをインポート: `import utils.trainer as trainer`
    2. model: トレーニング対象のモデルを用意
    3. train_loader: トレーニングデータのデータローダーを用意
    4. config: 設定情報を含む辞書を準備
    5. トレーニングを実行: `trainer.train_model(model, train_loader, config)`

このモジュールは、モデルのトレーニングプロセスを自動化し、トレーニング損失を記録して保存するのに役立ちます。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import utils.model_utils as model_utils
import logging
import os
import json

def train_model(model, train_loader, config):
    """
    モデルのトレーニングを実行します。

    Args:
        model (nn.Module): トレーニング対象のモデル
        train_loader (DataLoader): トレーニングデータのデータローダー
        config (dict): 設定情報が含まれる辞書

    Returns:
        None
    """
    # GPUが利用可能かどうかを確認し、デバイスを設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f'Device: {device}')

    # 損失関数と最適化アルゴリズムの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # トレーニングを開始
    logging.info('Training started...')
    training_losses = []
    
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_loss = running_loss / len(train_loader)

        # ログにトレーニング損失を追記
        logging.info(f"Epoch [{epoch+1}/{config['epochs']}], Average Loss: {average_loss:.4f}")

        training_losses_dict = {f'epoch{epoch+1}':average_loss}
        
        # 各エポックのトレーニング損失を保存
        training_losses.append(training_losses_dict)
        
        # モデルのバージョンを取得し、保存
        if epoch==0:
            model_version = model_utils.get_model_new_version(config["model_save_dir"])
        else:
            model_version = model_utils.get_latest_model_version(config["model_save_dir"])
            
        model_utils.save_model(model, config["model_save_dir"], model_version, epoch+1)

    metrics_folder = f"{config['model_save_dir']}{model_version}/metrics/"                                                                                                                                                                                
    os.makedirs(metrics_folder, exist_ok=True) 
        
    # トレーニング損失をJSONファイルに保存
    losses_filename = os.path.join(metrics_folder, f'training_losses.json')
    with open(losses_filename, 'w') as f:
        json.dump(training_losses, f)
        
    # ログにトレーニング結果を追記
    logging.info('Training completed.')
    logging.info(f"Model weights of version {model_version} saved in path ../{config['model_save_dir']}{model_version}/")
