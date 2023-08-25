"""
メトリクスプロットユーティリティモジュール

このモジュールには、指定されたメトリクスデータフォルダから、訓練損失、テスト損失、およびテスト精度のグラフをプロットする関数が含まれています。

使用方法:
    1. モジュールをインポート: `import utils.plot_metrics as plot_metrics`
    2. メトリクスのグラフをプロット:
       - メトリクスデータフォルダへのパスを指定してプロット: `plot_metrics.plot_metrics_from_json(metrics_path)`

このモジュールは、訓練とテストのメトリクスデータを読み込み、それをグラフにプロットするのに役立ちます。
"""

import json
import matplotlib.pyplot as plt
import os

def plot_metrics_from_json(metrics_path):
    """
    指定されたメトリクスデータフォルダから、訓練損失、テスト損失、およびテスト精度のグラフをプロットします。

    Args:
        metrics_path (str): メトリクスデータフォルダへのパス

    Returns:
        None
    """
    try:
        # JSONファイルの確認
        test_accuracy_path = os.path.join(metrics_path, 'test_accuracy.json')
        test_losses_path = os.path.join(metrics_path, 'test_losses.json')
        training_losses_path = os.path.join(metrics_path, 'training_losses.json')
        
        available_graphs = []
        # Test AccuracyのJSONファイルが存在する場合、データを読み込み
        if os.path.exists(test_accuracy_path):
            with open(test_accuracy_path, 'r') as f:
                test_accuracy_data = json.load(f)
            test_accuracy = [(int(list(value_dict.keys())[0][5:]), list(value_dict.values())[0]) for value_dict in test_accuracy_data]
            available_graphs.append('Test Accuracy')
            
        # Test LossのJSONファイルが存在する場合、データを読み込み
        if os.path.exists(test_losses_path):
            with open(test_losses_path, 'r') as f:
                test_losses_data = json.load(f)
            test_losses = [(int(list(value_dict.keys())[0][5:]), list(value_dict.values())[0]) for value_dict in test_losses_data]
            available_graphs.append('Test Loss')

        # Training LossのJSONファイルが存在する場合、データを読み込み
        if os.path.exists(training_losses_path):
            with open(training_losses_path, 'r') as f:
                training_losses_data = json.load(f)
            training_losses = [(int(list(value_dict.keys())[0][5:]), list(value_dict.values())[0]) for value_dict in training_losses_data]
            available_graphs.append('Training Loss')

        # 利用可能なグラフがない場合、メッセージを表示して終了
        if len(available_graphs) == 0:
            print("No JSON files found for plotting.")
            return

        epochs = [epoch for epoch, _ in training_losses]
        
        # 利用可能なグラフをプロット
        if 'Training Loss' in available_graphs or 'Test Loss' in available_graphs:
            plt.figure(figsize=(10, 5))
            if 'Training Loss' in available_graphs:
                plt.plot(epochs, [value for _, value in training_losses], label='Training Loss')
            if 'Test Loss' in available_graphs:
                plt.plot(epochs, [value for _, value in test_losses], label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Test Losses')
            plt.legend()
            plt.grid()
            plt.show()
        
        if 'Test Accuracy' in available_graphs:
            plt.figure(figsize=(10, 5))
            plt.plot([epoch for epoch, _ in test_accuracy], [value for _, value in test_accuracy], label='Test Accuracy', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Test Accuracy')
            plt.legend()
            plt.grid()
            plt.show()
        
    except Exception as e:
        print(f"An error occurred while plotting metrics: {str(e)}")
