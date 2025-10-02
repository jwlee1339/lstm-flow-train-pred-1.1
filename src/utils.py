# utils.py
# 2025-05-20
import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

def save_prediction_to_txt(df:pd.DataFrame, filename:str, header=None, output_dir="./Result"):
    """儲存預測結果到文字檔案

    Args:
        df (pd.DataFrame): _description_
        filename (str): 檔案名稱
        header (str, optional): 標題. Defaults to None.
        output_dir (str, optional): _description_. Defaults to "./Result".
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            if header:
                f.write(f"{header}\n")
            # 逐行写入时间戳与预测值
            for timestamp, level in df.itertuples():
                f.write(f"{timestamp.strftime('%Y-%m-%d %H:%M')},{level:.2f}\n")
        print(f"預測結果已保存至 {filename}")
    except Exception as e:
        print(f"保存失敗: {str(e)}")
    return

        
def SaveToTxt(obstime:np.ndarray, y_true:np.ndarray, predictions:np.ndarray, filename:str, header=None, output_dir="./Result"):
    """儲存到文字檔案

    Args:
        y_true (np.ndarray): 觀測值
        predicions(np.ndarray): 預測值
        filename (str): 檔案名稱
        header (str, optional): 標題. Defaults to None.
        output_dir (str, optional): _description_. Defaults to "./Result".
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            if header:
                f.write(f"{header}\n")
            # 逐行寫入資料
            index = 1
            for t, value1, value2 in zip(obstime, y_true, predictions):
                f.write(f"{index},{t},{value1:.2f},{value2:.2f}\n")
                index +=1
        print(f"預測結果已保存至 {filename}")
    except Exception as e:
        print(f"保存失敗: {str(e)}")
        
        
def SavePredToTxt(obstime:list, predictions:list, filename:str, header=None, output_dir="./Result"):
    """儲存預測結果到文字檔案

    Args:
        obstime (list): 預測起始時間的列表。
        predictions (list): 包含多步預測結果的列表 (list of lists)。
        filename (str): 檔案名稱
        header (str, optional): 標題. Defaults to None.
        output_dir (str, optional): _description_. Defaults to "./Result".
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            if header:
                f.write(f"{header}\n")
            # 逐行寫入資料
            index = 1
            for time, pred_item in zip(obstime, predictions):
                pred_str = ' '.join([f'{p:9.2f}' for p in pred_item])
                f.write(f"{index:3d} {time} {pred_str}\n")
                index += 1
        print(f"預測結果已保存至 {filename}")
    except Exception as e:
        print(f"保存失敗: {str(e)}")
        
        
# Save to CSV text file
def SaveObsPredToCSV(df_to_save: pd.DataFrame, filename:str, header=None, output_dir="./Result"):
    """儲存到CSV文字檔案

    Args:
        df_to_save (pd.DataFrame): 包含觀測值和預測值的DataFrame，索引應為datetime。
        filename (str): 檔案名稱
        header (str, optional): 標題. Defaults to None.
        output_dir (str, optional): _description_. Defaults to "./Result".
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        # 確保索引是 datetime 格式
        if not isinstance(df_to_save.index, pd.DatetimeIndex):
            print("[Warning] DataFrame index is not a DatetimeIndex. Saving without specific time format.")
            df_to_save.to_csv(filepath, float_format='%.2f', header=True if header else False)
        else:
            # 儲存時格式化時間戳
            df_to_save.to_csv(filepath, float_format='%.2f', date_format='%Y-%m-%d %H:%M:%S', header=True if header else False)

        print(f"預測結果已保存至 {filename}")
    except Exception as e:
        print(f"保存失敗: {str(e)}")
        

def AddHoursToArray(obstime_test:np.ndarray, hours_add:int=6):
    """ndarray 字串轉換為datetime, 並加上6小時

    Args:
        obstime_test (np.ndarray): _description_

    Returns:
        np.ndarray: 日期時間ndarray串列
    """
    # 假設 ndarray 中的日期時間格式為 "%Y-%m-%d %H:%M:%S.%f"
    # date_strs = np.array(['2025-05-20 08:30:00.1', '2025-05-21 15:45:00.2'])

    # 自定義轉換函數
    def str_to_datetime_add_6h(s):
        # dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return dt + timedelta(hours=hours_add)

    # 向量化操作轉換並增加時間
    vectorized_func = np.vectorize(str_to_datetime_add_6h)
    result_datetimes = vectorized_func(obstime_test)

    # print("轉換後結果:", result_datetimes)
    return result_datetimes

def inverse_transform_flow(scaled_data: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    僅對流量（第一列）進行反正規化。

    Args:
        scaled_data (np.ndarray): 正規化後的數據，流量應在第一列。
        scaler (MinMaxScaler): 用於訓練的 scaler 物件。

    Returns:
        np.ndarray: 反正規化後的流量數據。
    """
    # 創建一個與原始特徵維度相同的虛擬陣列
    dummy_array = np.zeros((len(scaled_data), len(scaler.scale_)))
    # 將目標數據（流量）放置在第一列
    dummy_array[:, 0] = scaled_data.flatten()
    # 執行反轉換並只取第一列
    return scaler.inverse_transform(dummy_array)[:, 0]


def create_prediction_dataset(data: np.ndarray, lookback: int, forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """從時間序列數據創建監督式學習的 X 和 y_future_rainfall。"""
    X, y_future_rainfall = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:(i + lookback)])
        # y_future_rainfall 只需要未來時間點的雨量（第二列）
        y_future_rainfall.append(data[i + lookback : i + lookback + forecast_horizon, 1])
    return np.array(X), np.array(y_future_rainfall)

def create_dataset(data, lookback, forecast_horizon):
    """從時間序列數據創建監督式學習的數據集"""
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:(i + lookback)])
        # 目標 y 只需要流量，即第0個特徵
        y.append(data[(i + lookback):(i + lookback + forecast_horizon), 0])
    return np.array(X), np.array(y)