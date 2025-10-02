# D:\home\2_MIT\14_FromYaYun\2025環境部\LSTM\01_lstm_pred1\train1.py
# coding: utf-8
# 使用方法: python train1.py --action train --lookback 12 --epochs 50 --lr 0.001

# 降雨逕流LSTM類神經網路模型
# - 模型訓練
# - 模型驗證
# - 採用shift 1小時觀測雨量為輸入值

# 分離模型訓練與預測流程
## 第一部分: 訓練模型並保存參數 (`train.py`)

import numpy as np
import argparse
import pandas as pd
from datetime import datetime, timedelta
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from FlowLSTM import FlowLSTM
from src.ModelEV import ModelTest
from src.plot_chart import plot_raw_rainfall_ts, plot_raw_ts, plot_ts, plot_xy
from src.utils import SaveToTxt
import os


# ----從檔案讀取降雨逕流觀測資料
def load_and_preprocess_data(data_path: str, rainfall_shift_hours: int):
    """讀取CSV資料，設定時間索引，並新增位移後的降雨特徵。"""
    # index,INFO_DATE,OBS_RAIN,Q_IN
    df = pd.read_csv(data_path, parse_dates=["INFO_DATE"])
    df = df.drop(columns="index")
    # 新增index
    df["date_column"] = pd.to_datetime(df["INFO_DATE"])
    df = df.set_index("date_column")
    print(df)
    return df

def add_shifted_rainfall_feature(df: pd.DataFrame, shift_hours: int) -> pd.DataFrame:
    """為DataFrame新增一個未來N小時的降雨特徵欄位。"""
    feature_name = f"fcstrain_{shift_hours}h"
    df[feature_name] = df["OBS_RAIN"].shift(periods=-shift_hours, fill_value=0)
    if feature_name in df.columns:
        print(f"\n--- 新增雨量特徵 '{feature_name}' ---")
        print(df.head())
    else:
        print(f"錯誤: 無法新增預測雨量特徵 '{feature_name}'。請檢查資料。")
    return df

def GetRainRunoffData(df: pd.DataFrame):
    """從pandas取出原始資料串列

    Args:
        df (pd.DataFrame): 從檔案讀取的原始資料pandad dataframe

    Returns:
        dict: _description_
    """
    raw_flow: np.ndarray = df["Q_IN"].to_numpy(dtype=float)
    raw_rainfall: np.ndarray = df["OBS_RAIN"].to_numpy(dtype=float)
    # print(raw_flow)
    # print(raw_rainfall)
    obstime: list = df.index.tolist()
    # print(type(obstime[0]))
    return {"raw_flow": raw_flow, "raw_rainfall": raw_rainfall, "obstime": obstime}



def create_dataset(data, lookback, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon):
        X.append(data[i : (i + lookback)])
        y.append(data[(i + lookback) : (i + lookback + forecast_horizon), 0])
    return np.array(X), np.array(y)


# 2. 數據預處理
def prepare_data(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    lookback: int = 12,
    train_split_ratio: float = 0.8,
    forecast_horizon: int = 1,
):
    """
    準備訓練和測試數據集。
    1. 分割數據為訓練集和測試集。
    2. 在訓練集上擬合scaler，並用它來轉換訓練集和測試集，以防止數據洩漏。
    3. 創建時間序列數據集。
    4. 將數據轉換為PyTorch Tensors。
    """
    # 選擇特徵
    data = df[["Q_IN", "fcstrain_1h"]].values

    # 1. 先按比例分割原始數據，避免數據洩漏
    train_data, test_data = train_test_split(
        data, train_size=train_split_ratio, shuffle=False
    )

    # 2. 在訓練集上擬合scaler，並轉換訓練集和測試集
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)

    # 3. 為訓練集和測試集分別創建序列數據
    X_train, y_train = create_dataset(scaled_train_data, lookback, forecast_horizon)
    X_test, y_test = create_dataset(scaled_test_data, lookback, forecast_horizon)

    # 4. 轉換為 PyTorch Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    print(f"原始資料筆數: {len(data)}")
    print(f"訓練資料筆數: {len(train_data)}, 測試資料筆數: {len(test_data)}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_train, y_train, X_test, y_test


# 4. 訓練模型
from torch.utils.data import TensorDataset, DataLoader


def Train(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    scaler: MinMaxScaler,
    lookback: int,
    forecast_horizon: int,
    num_epochs: int,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    learning_rate: float,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 使用 DataLoader 進行批次處理
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )  # 時序數據通常不隨機排序

    model = FlowLSTM(
        input_size=2, hidden_size=hidden_size, num_layers=num_layers, output_size=forecast_horizon
    ).to(device)

    # 檢查是否有可用的既有模型，並驗證參數是否一致
    model_path = "flow_prediction_model.pth"
    if os.path.exists(model_path):
        print(f"\n發現先前儲存的模型檔案: '{model_path}'，正在檢查參數...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 驗證模型架構參數是否與當前設定相符
        params_match = True
        saved_params = checkpoint.get('model_params', {}) # 舊版可能沒有 model_params
        if saved_params.get('lookback') != lookback:
            print(f"  - 參數不符: lookback (已儲存: {saved_params.get('lookback')}, 目前: {lookback})")
            params_match = False
        if saved_params.get('forecast_horizon') != forecast_horizon:
            print(f"  - 參數不符: forecast_horizon (已儲存: {saved_params.get('forecast_horizon')}, 目前: {forecast_horizon})")
            params_match = False
        if saved_params.get('hidden_size') != hidden_size:
            print(f"  - 參數不符: hidden_size (已儲存: {saved_params.get('hidden_size')}, 目前: {hidden_size})")
            params_match = False
        if saved_params.get('num_layers') != num_layers:
            print(f"  - 參數不符: num_layers (已儲存: {saved_params.get('num_layers')}, 目前: {num_layers})")
            params_match = False

        if params_match:
            print("參數相符，載入既有模型權重進行暖啟動訓練...")
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                print(f"載入權重失敗: {e}")
                print("將從頭開始訓練。")
        else:
            print("由於模型參數不符，將忽略既有檔案，從頭開始訓練。")
    else:
        print("\n未發現既有模型，將從頭開始訓練。")


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- 學習率排程器 ---
    # 當驗證損失連續5個epoch沒有改善時，將學習率乘以0.1
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    loss_history = [] # 用於儲存每個 epoch 的 loss

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        # DataLoader 會自動處理批次劃分
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        loss_history.append(avg_train_loss)
        print(f"Epoch [{epoch+1:03d}/{num_epochs}], Loss: {avg_train_loss:.6f}")
        # --- 更新學習率 ---
        scheduler.step(avg_train_loss)

    output_dir = "Result"
    os.makedirs(output_dir, exist_ok=True)

    # --- 儲存並繪製損失歷史 ---
    loss_df = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        'loss': loss_history
    })
    loss_csv_path = os.path.join(output_dir, 'training_loss_history.csv')
    loss_df.to_csv(loss_csv_path, index=False)
    print(f"訓練損失歷史已儲存至 {loss_csv_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(loss_df['epoch'], loss_df['loss'], label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_curve_path = os.path.join(output_dir, 'training_loss_curve.png')
    plt.savefig(loss_curve_path)
    print(f"訓練損失曲線圖已儲存至 {loss_curve_path}")

    # 保存模型和必要組件
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'scaler_min': scaler.min_,
            'scaler_scale': scaler.scale_,
            'model_params': {
                'lookback': lookback,
                'forecast_horizon': forecast_horizon,
                'input_size': 2,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'output_size': forecast_horizon,
            }
        },
        "flow_prediction_model.pth",
    )

    print("模型訓練完成並已保存為 flow_prediction_model.pth")
    return model


# 還原
def inverse_transform(scaled_data: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    僅對流量（第一列）進行反正規化。

    Args:
        scaled_data (np.ndarray): 正規化後的數據。
        scaler (MinMaxScaler): 用於訓練的 scaler 物件。

    Returns:
        np.ndarray: 反正規化後的流量數據。
    """
    dummy_array = np.zeros((len(scaled_data), len(scaler.scale_)))
    dummy_array[:, 0] = scaled_data.flatten()
    return scaler.inverse_transform(dummy_array)[:, 0]


def evaluate_and_plot(
    model, scaler, X_test, y_test, obstime_test, forecast_horizon, year
):
    """
    使用模型進行評估，並繪製結果圖表。

    Args:
        model: 已訓練或載入的模型。
        scaler: 用於反正規化的 MinMaxScaler。
        X_test (torch.Tensor): 測試特徵。
        y_test (torch.Tensor): 測試目標。
        obstime_test (list): 測試集對應的時間戳。
        forecast_horizon (int): 預測期長。
        year (int): 資料年份，用於圖檔命名。
    """
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)

    # 遍歷每一個預測的時間步長
    for step_index in range(forecast_horizon):
        pred_hour = step_index + 1
        print(f"\n--- 正在評估與繪製 預測第 {pred_hour} 小時 的結果 ---")

        # 提取對應步長的真實值與預測值
        y_true_scaled = y_test.cpu().numpy()[:, step_index]
        predictions_scaled = test_predictions.cpu().numpy()[:, step_index]

        # 反正規化
        y_true = inverse_transform(y_true_scaled, scaler)
        predictions = inverse_transform(predictions_scaled, scaler)

        # 調整時間戳以匹配預測步長
        # 預測第 n 小時的結果，其對應的觀測時間應向後平移 n 小時
        adjusted_obstime = [t + timedelta(hours=pred_hour) for t in obstime_test]

        # 確保時間戳與數據長度一致
        # 因為 create_dataset 的關係，y_test 會比 obstime_test 短
        # 我們需要截取時間戳以匹配 y_true 的長度
        final_obstime = adjusted_obstime[:len(y_true)]

        # 輸出對照表
        SaveToTxt(
            obstime=final_obstime,
            y_true=y_true,
            predictions=predictions,
            filename=f"predictions_h{pred_hour}.csv",
            header="index,time,y_true,prediction",
        )

        # 繪製xy圖
        plot_xy(
            y_true=y_true,
            predictions=predictions,
            forecast_horizon=pred_hour,
            save_path=f"train_result_xy_h{pred_hour}.png",
        )

        # 計算評估指標
        model_eva = ModelTest(qo=y_true, preq=predictions)
        print(f"評估指標 (h{pred_hour}): {model_eva}")

        # 繪製驗證結果圖
        plot_ts(
            obs_time=final_obstime,
            raw_flow=y_true,
            predictions=predictions,
            forecast_horizon=pred_hour,
            CE=model_eva["CE"],
            EQP=model_eva["EQP"],
            save_path=f"{year}_validation_result_h{pred_hour}.png",
        )


def load_model_for_validation(model_path="flow_prediction_model.pth"):
    """從檔案載入模型與 Scaler 以進行驗證"""
    if not os.path.exists(model_path):
        print(f"錯誤: 找不到模型檔案 '{model_path}'。請先執行訓練模式。")
        return None, None

    # 由於 .pth 檔案中包含 NumPy 陣列 (scaler)，需要將 weights_only 設為 False
    # 這是 PyTorch 2.6+ 的安全更新，因為我們信任此檔案來源，所以可以安全地設為 False
    checkpoint = torch.load(
        model_path, map_location=torch.device('cpu'), weights_only=False
    )

    model_params = checkpoint.get('model_params', {})
    model = FlowLSTM(
        input_size=model_params.get('input_size', 2),
        hidden_size=model_params.get('hidden_size', 64),
        num_layers=model_params.get('num_layers', 2),
        output_size=model_params.get('output_size', 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    scaler = MinMaxScaler()
    scaler.min_ = checkpoint["scaler_min"]
    scaler.scale_ = checkpoint["scaler_scale"]

    print(f"已從 '{model_path}' 載入模型與 Scaler。")
    return model, scaler


def main():
    parser = argparse.ArgumentParser(description="降雨逕流LSTM模型訓練與驗證")
    parser.add_argument(
        "--action",
        type=str,
        default="train",
        choices=["train", "validate"],
        help='執行模式: "train" (重新訓練與驗證) 或 "validate" (僅使用現有模型驗證)。',
    )
    parser.add_argument('--data_path', type=str, default='data/2024降雨逕流_filled.csv',
                        help='包含觀測資料的 CSV 檔案路徑。')
    parser.add_argument('--lookback', type=int, default=12,
                        help='模型回看的時間步長（小時）。')
    parser.add_argument('--epochs', type=int, default=50,
                        help='訓練的週期數量。')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='訓練時的批次大小。')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='LSTM 隱藏層的大小。')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='LSTM 的層數。')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='優化器的學習率。')
    parser.add_argument('--train_split_ratio', type=float, default=0.8,
                        help='訓練集所佔的比例。')
    parser.add_argument('--rainfall_shift_hours', type=int, default=1,
                        help='用於特徵工程的降雨位移時數。')
    parser.add_argument('--forecast_horizon', type=int, default=1,
                        help='模型預測的未來時間步長。')

    args = parser.parse_args()

    # 讀取與預處理資料
    df = load_and_preprocess_data(args.data_path, args.rainfall_shift_hours) # type: ignore
    df = add_shifted_rainfall_feature(df, shift_hours=args.rainfall_shift_hours)
    if f"fcstrain_{args.rainfall_shift_hours}h" not in df.columns:
        return

    # 取得原始資料並繪製初始圖表
    data = GetRainRunoffData(df=df)
    obstime = data["obstime"]
    
    # 繪製原始流量時間序列
    plot_raw_ts(
        obs_time=obstime,
        raw_flow=data["raw_flow"],
        title="翡翠水庫集水區原始流量時間序列圖",
        save_path="raw_flow.png",
        plot_show=False,
    )

    # 繪製原始觀測雨量組體圖
    plot_raw_rainfall_ts(
        obs_time=obstime,
        raw_rainfall=data["raw_rainfall"],
        basin_area=303.0,  # km^2
        save_path="raw_rainfall.png",
        plot_show=False,
    )

    year = obstime[0].year

    if args.action == "train":
        print("\n=== 執行模式: 訓練與驗證 ===")
        print(f"超參數: lookback={args.lookback}, epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
        # 重新擬合 Scaler 並準備資料
        scaler = MinMaxScaler()
        X_train, y_train, X_test, y_test = prepare_data(
            df=df,
            scaler=scaler,
            lookback=args.lookback,
            train_split_ratio=args.train_split_ratio,
            forecast_horizon=args.forecast_horizon,
        )

        # 重新計算測試集對應的時間戳
        train_size = int(len(df) * args.train_split_ratio)
        test_start_index = train_size
        obstime_test = obstime[test_start_index : test_start_index + len(y_test)]

        # -----訓練
        model = Train(
            X_train=X_train,
            y_train=y_train,
            scaler=scaler,
            lookback=args.lookback,
            forecast_horizon=args.forecast_horizon,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            learning_rate=args.lr,
        )

        # -----評估
        evaluate_and_plot(
            model, scaler, X_test, y_test, obstime_test, args.forecast_horizon, year
        )

    elif args.action == "validate":
        print("\n=== 執行模式: 僅驗證 ===")
        model, scaler = load_model_for_validation()
        if model and scaler:
            # 準備驗證資料集
            data_for_scaling = df[["Q_IN", "fcstrain_1h"]].values
            train_data, test_data = train_test_split(
                data_for_scaling, train_size=args.train_split_ratio, shuffle=False
            )

            # 使用載入的 scaler 轉換測試集
            scaled_test_data = scaler.transform(test_data)
            X_test_raw, y_test_raw = create_dataset(
                scaled_test_data, args.lookback, args.forecast_horizon
            )

            X_test = torch.tensor(X_test_raw, dtype=torch.float32)
            y_test = torch.tensor(y_test_raw, dtype=torch.float32)

            test_start_index = len(train_data)
            obstime_test = obstime[test_start_index : test_start_index + len(y_test)]

            evaluate_and_plot(
                model, scaler, X_test, y_test, obstime_test, args.forecast_horizon, year
            )


if __name__ == "__main__":
    # 取得現在時間
    start_time = datetime.now()
    main()
    # 計算耗費時間
    elapsed_time = datetime.now() - start_time
    print(f"\n總耗費時間: {elapsed_time}")
    print("程式執行結束")
