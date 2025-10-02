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
from src.utils import SaveObsPredToCSV, create_dataset


# 1. 加載保存的模型
def load_model(model_path):
    # 注意weights_only=False 預設值為True
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
    對單一 lookback 窗口進行一次預測。

    參數:
    model: 模型
    scaler: 數據標準化器
    last_observed_data: 最後觀察到的原始數據 (shape: [lookback, num_features])

    返回:
    prediction: 預測的流量值 (原始尺度)
    """
    # 標準化輸入數據
    scaled_input = scaler.transform(last_observed_data)

    # 準備模型輸入
    device = next(model.parameters()).device
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)

    # 進行預測
    with torch.no_grad():
        scaled_predictions = model(input_tensor).cpu().numpy()[0]

    # 反標準化預測結果 (只針對流量)
    # 創建一個與原始數據維度相同的虛擬陣列來進行反正規化
    dummy_array = np.zeros((len(scaled_predictions), len(scaler.scale_)))
    dummy_array[:, 0] = scaled_predictions
    prediction = scaler.inverse_transform(dummy_array)[:, 0]

    return prediction


def load_and_prepare_data(data_path, feature_cols, target_col, shift_col, shift_hours):
    """加載並預處理數據"""
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


def perform_iterative_forecast(df, model, scaler, X, y, lookback, lead_hours):
    """執行迭代預測"""
    all_predictions = []
    initial_times = []

    print(f"\n模型預測 {model.fc.out_features} 小時, 逐步預測 {lead_hours} 小時:")

    for index, initial_input in enumerate(X):
        # 檢查是否有足夠的未來觀測雨量來進行多步預測
        if index + lead_hours >= len(y):
            print(f"Reached end of data. Stopping prediction at index {index}.")
            break

        current_input = initial_input.copy()
        step_predictions = []

        for i in range(lead_hours):
            # 預測未來 1 小時流量 (假設 forecast_horizon=1)
            pred_value = predict_future_flow(model, scaler, current_input)

            # 準備下一步的輸入：滾動更新觀測數據
            # 1. 移除最舊的一筆數據
            current_input = np.delete(current_input, 0, axis=0) # type: ignore
            # 2. 取得下一步的觀測雨量，應從原始 DataFrame 中獲取
            # 索引 = 當前樣本的起始索引 + lookback長度 + 迭代步數
            next_rainfall_index = lookback + index + i
            next_rainfall = df['fcstrain_1h'].iloc[next_rainfall_index]
            # 3. 將 (預測流量, 觀測雨量) 作為新的輸入加入
            new_row = [[pred_value[0], next_rainfall]]
            current_input = np.append(current_input, new_row, axis=0)

            step_predictions.append(pred_value[0])

        all_predictions.append(step_predictions)
        # 記錄這次預測的起始時間
        initial_time = df.index[lookback + index]
        initial_times.append(initial_time)

        # 打印預測結果
        pred_str = ' '.join([f'{p:9.2f}' for p in step_predictions])
        print(f"{index+1:3d} {initial_time.strftime('%Y-%m-%d %H:%M')} {pred_str}")
    
    # 將預測結果轉換為 DataFrame
    pred_columns = [f'h{i+1}' for i in range(lead_hours)]
    df_pred = pd.DataFrame(all_predictions, columns=pred_columns, index=initial_times)
    
    return df_pred


def evaluate_and_plot_results(df_obs, df_pred, lead_times_to_eval):
    """評估、繪圖並儲存結果"""
    print("\n--- 預測結果摘要 ---")
    print(df_pred.head())

    for pred_hour in lead_times_to_eval:
        print(f"\n----- 評估 Lead Time: {pred_hour} 小時 -----")

        # 1. 準備預測數據
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

        # 5. 繪圖
        year = df_obs.index[0].year
        plot_prediction_vs_observation(
            obs_df=eval_df[['flow', 'rainfall']],
            pred_df=eval_df[['pred_flow']],
            pred_hour=pred_hour,
            metrics=model_eva,
            save_path=f'{year}_pred_result_h{pred_hour}.png'
        )

        # 6. 儲存結果
        output_df = eval_df[['flow', 'pred_flow']].copy()
        output_df.columns = ['obs_flow', 'pred_flow']
        SaveObsPredToCSV(df_to_save=output_df, 
                         filename=f'pred_vs_obs_h{pred_hour}.csv',
                         header=True)

def main():
    """主執行函式"""
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
    df_pred = perform_iterative_forecast(df, model, scaler, X, y, lookback, LEAD_HOURS_TO_FORECAST)

    # --- 4. 評估與繪圖 ---
    evaluate_and_plot_results(df, df_pred, LEAD_TIMES_TO_EVALUATE)


if __name__ == "__main__":
    main()
