#!/usr/bin/env python
# coding: utf-8

# LSTM類神經網路降雨逕流預測
# - 讀取前期降雨及入流量。
# - 採用 shift 1 小時觀測雨量為輸入值。
# - 逐時預測未來流量。

import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from FlowLSTM import FlowLSTM
from src.plot_chart import plot_prediction_vs_observation
from src.ModelEV import ModelTest, PrintEvaluationFactors
from src.utils import SaveObsPredToCSV, create_dataset, inverse_transform_flow


# 1. 加載保存的模型
def load_model(model_path):
    """
    從指定路徑加載已訓練好的 PyTorch 模型、MinMaxScaler 和模型參數。

    Args:
        model_path (str): 模型檔案 (.pth) 的路徑。

    Returns:
        tuple: 包含以下元素的元組：
            - model (FlowLSTM): 加載並設定為評估模式的模型。
            - scaler (MinMaxScaler): 重建的數據正規化器。
            - lookback (int): 模型的 lookback 參數。
            - forecast_horizon (int): 模型的 forecast_horizon 參數。
    """
    # 注意weights_only=False，因為我們需要載入 scaler 的 numpy 陣列
    checkpoint = torch.load(model_path, weights_only=False)

    model_params = checkpoint.get('model_params', {})
    model = FlowLSTM(
        input_size=model_params.get('input_size', 2),
        hidden_size=model_params.get('hidden_size', 64),
        num_layers=model_params.get('num_layers', 2),
        output_size=model_params.get('output_size', 1)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 重建scaler
    scaler = MinMaxScaler()
    scaler.min_ = checkpoint['scaler_min']
    scaler.scale_ = checkpoint['scaler_scale']

    return model, scaler, model_params.get('lookback'), model_params.get('forecast_horizon')


# 2. 預測函數
def predict_future_flow(model, scaler, last_observed_data):
    """
    對單一 lookback 窗口的數據進行一次流量預測。

    Args:
        model (FlowLSTM): 已訓練的模型。
        scaler (MinMaxScaler): 用於數據正規化的轉換器。
        last_observed_data (np.ndarray): 用於預測的輸入數據，形狀為 [lookback, num_features]。

    Returns:
        np.ndarray: 預測出的流量值（原始尺度）。
    """
    # 標準化輸入數據
    scaled_input = scaler.transform(last_observed_data)

    # 準備模型輸入
    device = next(model.parameters()).device
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)

    # 進行預測
    with torch.no_grad():
        scaled_predictions = model(input_tensor).cpu().numpy()[0]

    # 使用 utils 中的函式反標準化預測結果 (只針對流量)
    prediction = inverse_transform_flow(scaled_predictions, scaler)

    return prediction


def load_and_prepare_data(data_path, feature_cols, target_col, shift_col, shift_hours):
    """
    加載並預處理用於預測的數據。

    Args:
        data_path (str): CSV 數據檔案的路徑。
        feature_cols (list): 特徵欄位名稱列表。
        target_col (str): 目標欄位名稱。
        shift_col (str): 要進行位移以創建新特徵的欄位名稱（通常是雨量）。
        shift_hours (int): 位移的時數。

    Returns:
        pd.DataFrame: 經過預處理的 DataFrame。
    """
    df = pd.read_csv(data_path)
    df['date_column'] = pd.to_datetime(df['obstime'])
    df = df.set_index('date_column')
    print("原始數據讀取完成:")
    print(df.head())

    # 製作預測雨量特徵
    df[f'fcstrain_{shift_hours}h'] = df[shift_col].shift(periods=-shift_hours, fill_value=0)
    print("\n加入 shifted rainfall feature 後的數據:")
    print(df.head())
    
    return df


def perform_iterative_forecast(df, model, scaler, X, lookback, forecast_horizon, lead_hours):
    """
    執行迭代式（滾動）預測。

    此函式模擬真實世界的預測情境：使用一個歷史數據窗口進行預測，
    然後將預測結果作為下一個時間點的輸入，並結合已知的未來降雨，
    持續滾動預測直到達到指定的總預測時長。

    Args:
        df (pd.DataFrame): 包含所有數據的 DataFrame。
        model (FlowLSTM): 預測模型。
        scaler (MinMaxScaler): 數據正規化器。
        X (np.ndarray): 初始的輸入數據集，形狀為 [num_samples, lookback, num_features]。
        lookback (int): 模型的回看時長。
        forecast_horizon (int): 模型單次預測的步長。
        lead_hours (int): 總共要預測的未來時數。

    Returns:
        pd.DataFrame: 一個 DataFrame，索引為預測起始時間，欄位為 h1, h2, ... 表示未來第 n 小時的預測值。
    """
    all_predictions = []
    initial_times = []

    print(f"\n模型單次預測 {forecast_horizon} 小時, 迭代預測未來 {lead_hours} 小時:")

    for index, initial_input in enumerate(X):
        # 檢查是否有足夠的未來觀測雨量來進行多步預測
        # 迭代預測需要 lookback + index + lead_hours 長度的資料
        if lookback + index + lead_hours > len(df):
            print(f"數據長度不足，於索引 {index} 停止預測。")
            break

        current_input = initial_input.copy()
        step_predictions = []

        # 每次迭代預測 forecast_horizon 個步長，直到達到 lead_hours
        for i in range(0, lead_hours, forecast_horizon):
            # 預測未來 forecast_horizon 小時的流量
            pred_values = predict_future_flow(model, scaler, current_input)
            
            num_preds = len(pred_values)
            step_predictions.extend(pred_values)

            # 準備下一次迭代的輸入
            # 透過 np.roll 高效地移除最舊的 num_preds 筆數據
            current_input = np.roll(current_input, -num_preds, axis=0)

            # 更新最後 num_preds 筆數據：使用預測出的流量和已知的未來降雨
            for j in range(num_preds):
                next_rainfall_index = lookback + index + i + j
                next_rainfall = df['fcstrain_1h'].iloc[next_rainfall_index]
                current_input[-(num_preds-j)] = [pred_values[j], next_rainfall]

        # 只取到我們需要的 lead_hours
        all_predictions.append(step_predictions[:lead_hours])
        # 記錄這次預測的起始時間
        initial_time = df.index[lookback + index]
        initial_times.append(initial_time)

        # 打印預測結果
        pred_str = ' '.join([f'{p:9.2f}' for p in step_predictions])
        print(f"{index+1:3d} {initial_time.strftime('%Y-%m-%d %H:%M')} {pred_str}")
    
    # 將預測結果轉換為 DataFrame
    pred_columns = [f'h{i+1}' for i in range(min(lead_hours, len(all_predictions[0])))]
    df_pred = pd.DataFrame(all_predictions, columns=pred_columns, index=initial_times)
    
    return df_pred


def evaluate_and_plot_results(df_obs, df_pred, lead_times_to_eval):
    """
    對指定領先時間的預測結果進行評估、繪圖並儲存。

    Args:
        df_obs (pd.DataFrame): 包含觀測流量和降雨的 DataFrame。
        df_pred (pd.DataFrame): 包含迭代預測結果的 DataFrame。
        lead_times_to_eval (list): 一個包含要評估的領先時間（小時）的列表，例如 [1, 3, 6]。
    """
    print("\n--- 預測結果摘要 ---")
    print(df_pred.head())

    for pred_hour in lead_times_to_eval:
        print(f"\n----- 評估 Lead Time: {pred_hour} 小時 -----")

        # 1. 準備預測數據
        # 1. 準備預測數據：選取對應欄位，並將時間索引向後平移以匹配觀測時間
        pred_series = df_pred[f'h{pred_hour}'].copy()
        pred_series.index += pd.to_timedelta(pred_hour, unit='h')
        pred_series.name = 'pred_flow'

        # 2. 準備觀測數據
        obs_series = df_obs[['flow', 'rainfall']].copy()

        # 3. 合併觀測與預測數據 (基於時間索引，自動對齊)
        eval_df = pd.merge(obs_series, pred_series, left_index=True, right_index=True, how='inner')

        if eval_df.empty:
            print(f"無法對齊 h{pred_hour} 的數據，跳過評估。")
            continue

        # 4. 計算評估指標
        obs_flow = eval_df['flow'].to_numpy()
        pred_flow = eval_df['pred_flow'].to_numpy()
        model_eva = ModelTest(qo=obs_flow, preq=pred_flow)
        PrintEvaluationFactors(model_eva)

        # 5. 繪製觀測與預測的對比圖
        year = df_obs.index[0].year
        plot_prediction_vs_observation(
            obs_df=eval_df[['flow', 'rainfall']],
            pred_df=eval_df[['pred_flow']],
            pred_hour=pred_hour,
            metrics=model_eva,
            save_path=f'{year}_pred_result_h{pred_hour}.png'
        )

        # 6. 儲存觀測與預測的對照表
        output_df = eval_df[['flow', 'pred_flow']].copy()
        output_df.columns = ['obs_flow', 'pred_flow']
        SaveObsPredToCSV(df_to_save=output_df, 
                         filename=f'pred_vs_obs_h{pred_hour}.csv',
                         header=True)

def main():
    """
    主執行函式。
    
    協調整個預測流程：設定參數、加載模型與數據、執行迭代預測、最後評估與呈現結果。
    """
    # --- 設定 ---
    MODEL_PATH = 'flow_prediction_model.pth'
    DATA_PATH = 'data/2023pred1.csv'
    LEAD_HOURS_TO_FORECAST = 6
    LEAD_TIMES_TO_EVALUATE = [1, 3, 6]
    FEATURE_COLS = ['flow', 'fcstrain_1h']

    # --- 1. 加載模型和數據 ---
    model, scaler, lookback, forecast_horizon = load_model(MODEL_PATH)
    print(f"Model loaded: lookback={lookback}, forecast_horizon={forecast_horizon}")

    df = load_and_prepare_data(
        data_path=DATA_PATH,
        feature_cols=['flow', 'rainfall'],
        target_col='flow',
        shift_col='rainfall',
        shift_hours=1
    )

    # --- 2. 準備預測數據集 ---
    raw_data_values = df[FEATURE_COLS].values
    X, y = create_dataset(raw_data_values, lookback, forecast_horizon)
    print(f"\nCreated dataset with X shape: {X.shape} and y shape: {y.shape}")

    # --- 3. 執行迭代預測 ---
    df_pred = perform_iterative_forecast(df, model, scaler, X, lookback, forecast_horizon, LEAD_HOURS_TO_FORECAST)

    # --- 4. 評估與繪圖 ---
    evaluate_and_plot_results(df, df_pred, LEAD_TIMES_TO_EVALUATE)


if __name__ == "__main__":
    main()
