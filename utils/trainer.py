"""
モデルトレーナーユーティリティモジュール

このモジュールには、モデルの訓練を行うための関数が含まれています。
訓練データを使用してモデルを学習し、訓練損失を記録して保存します。

使用方法:
    1. モジュールをインポート: `import utils.trainer as trainer`
    2. model: 訓練対象のモデルを用意
    3. train_loader: 訓練データのデータローダーを用意
    4. config: 設定情報を含む辞書を準備
    5. 訓練を実行: `trainer.train_model(model, train_loader, config)`

このモジュールは、モデルの訓練プロセスを自動化し、訓練損失を記録して保存するのに役立ちます。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import utils.model_utils as model_utils
import logging
import os
import json

def train_model(model, train_loader, val_loader, config):
    """
    モデルの訓練を実行します。

    Args:
        model (nn.Module): 訓練対象のモデル
        train_loader (DataLoader): 訓練データのデータローダー
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

    # 訓練を開始
    logging.info('Training started...')
    training_losses = []
    validation_losses = []
    
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

        # ログに訓練損失を追記
        #logging.info(f"Epoch [{epoch+1}/{config['epochs']}], Train Loss: {average_loss:.4f}")
        
        # バリデーション
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
        average_val_loss = val_loss / len(val_loader)

        # ログに検証損失を追記
        logging.info(f"Epoch [{epoch+1}/{config['epochs']}], Train Loss: {average_loss:.4f}, Validation Loss: {average_val_loss:.4f}")
        
        training_losses_dict = {f'epoch{epoch+1}': average_loss}
        validation_losses_dict = {f'epoch{epoch+1}': average_val_loss}

        # 各エポックの訓練損失と検証損失を保存
        training_losses.append(training_losses_dict)
        validation_losses.append(validation_losses_dict)
        
        # モデルのバージョンを取得し、保存
        if epoch == 0:
            model_version = model_utils.get_model_new_version(config["model_save_dir"])
        else:
            model_version = model_utils.get_latest_model_version(config["model_save_dir"])
            
        model_utils.save_model(model, config["model_save_dir"], model_version, epoch+1)

    metrics_folder = f"{config['model_save_dir']}{model_version}/metrics/"
    os.makedirs(metrics_folder, exist_ok=True)

    # 訓練損失と検証損失をJSONファイルに保存
    losses_filename = os.path.join(metrics_folder, f'training_losses.json')
    with open(losses_filename, 'w') as f:
        json.dump(training_losses, f)
        
    validation_losses_filename = os.path.join(metrics_folder, f'validation_losses.json')
    with open(validation_losses_filename, 'w') as f:
        json.dump(validation_losses, f)

    # ログに訓練結果を追記
    logging.info('Training completed.')
    logging.info(f"Model weights of version {model_version} saved in path ../{config['model_save_dir']}{model_version}/")