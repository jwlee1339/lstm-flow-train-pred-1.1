# plot_chart.py
# 2025-05-20
# 繪製原始資料

import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# 繪製時間序列
def plot_raw_ts(obs_time:list[datetime], raw_flow:np.ndarray, output_dir="./Result", **kwargs):
    """繪製原始流量觀測值的時間序列圖。

    Args:
        obs_time (list[datetime]): 觀測時間的列表。
        raw_flow (np.ndarray): 原始流量數據的 NumPy 陣列 (單位: CMS)。
        output_dir (str, optional): 圖片輸出的資料夾路徑。預設為 "./Result"。
        **kwargs: 其他可選參數，例如:
            - save_path (str): 圖片儲存的完整檔案名稱。
            - plot_show (bool): 是否顯示圖表，若為 False 則不顯示。
    """
    # 建立一個新的圖形，並設定其大小
    fig = plt.figure(figsize=(18,6))
    # 繪製流量時間序列線圖
    plt.plot(obs_time, raw_flow, '-', label='flow') # type: ignore

    # 計算統計數據
    maxValue = raw_flow.max()
    minValue = raw_flow.min()
    # 計算總逕流體積 (百萬立方公尺, MCM)
    totalVolume = raw_flow.sum() * 3600 / 1000000 # (cms-hour * 3600s/hr) / 1,000,000 m^3/MCM

    # 在圖表上新增文字框，顯示統計資訊
    plt.text(
        x=0.02, y=0.8,                        # 位置（相對坐標）
        s=f"流量平均值:{np.mean(raw_flow):.2f}CMS\n最大值:{maxValue:.2f}CMS\n最小值:{minValue:.2f}CMS\n總逕流體積:{totalVolume:.2f}MCM",  # 文本内容
        transform=plt.gca().transAxes,          # 坐标系转换
        bbox=dict(facecolor='white', alpha=0.8) # 白色背景框
    )

    # 設定圖表標題與座標軸標籤
    year = obs_time[0].year
    if 'title' in kwargs:
        title = f'{year}_{kwargs['title']}'
    else:
        title = f'{year}原始流量時間序列圖'
    plt.title(title, fontsize=16)
    plt.xlabel('日期時間')
    plt.ylabel('流量(CMS)')

    # 設定X軸日期的顯示格式與間隔
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))  # 每30天一個主要刻度

    # 顯示網格線與圖例
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # 根據傳入參數決定儲存檔案或直接顯示
    if 'save_path' in kwargs:
        os.makedirs(output_dir, exist_ok=True)
        filename = kwargs['save_path']
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f'儲存至檔案: {filepath}')
    else:
        # 如果 'plot_show' 設為 False，則關閉圖形，避免在後端執行時跳出視窗
        if 'plot_show' in kwargs and kwargs['plot_show'] == False:
            plt.close()
    return


# 繪製時間序列
def plot_raw_rainfall_ts(obs_time:list[datetime], raw_rainfall:np.ndarray, output_dir="./Result", **kwargs):
    """繪製原始觀測雨量時間序列圖，並在圖上顯示相關統計資訊。

    Args:
        obs_time (list[datetime]): 觀測時間的列表。
        raw_rainfall (np.ndarray): 原始雨量數據的 NumPy 陣列。
        output_dir (str, optional): 圖片輸出的資料夾路徑。預設為 "./Result"。
        **kwargs: 其他可選參數，例如:
            - basin_area (float): 集水區面積 (km^2)，用於計算總降雨體積。
            - save_path (str): 圖片儲存的完整檔案名稱。
            - plot_show (bool): 是否顯示圖表，若為 False 則不顯示。
    """
    # 建立一個新的圖形，並設定其大小
    fig = plt.figure(figsize=(18,6))
    # 繪製雨量時間序列線圖
    plt.plot(obs_time, raw_rainfall, '-', label='平均雨量') # type: ignore

    # 檢查是否傳入集水區面積參數，用於計算降雨體積
    basin_area = 0.0 # km^2
    if 'basin_area' in kwargs:
        basin_area = kwargs['basin_area']
        
    # 計算統計數據
    maxValue = raw_rainfall.max()
    minValue = raw_rainfall.min()
    rain_sum = raw_rainfall.sum()
    rain_vol = basin_area * 1000 * 1000 * rain_sum / 1000.0 # 降雨體積 (m^3)
    total_rain_vol = rain_vol / 1000000 # 總降雨體積 (百萬立方公尺, MCM)

    # 在圖表上新增文字框，顯示統計資訊
    plt.text(
        x=0.02, y=0.8,                        # 位置（相對坐標）
        s=f"雨量平均值:{np.mean(raw_rainfall):.2f}MM\n最大值:{maxValue:.2f}MM\n最小值:{minValue:.2f}MM\n總降雨體積:{total_rain_vol:.2f}MCM",  # 文本内容
        transform=plt.gca().transAxes,          # 坐标系转换
        bbox=dict(facecolor='white', alpha=0.8) # 白色背景框
    )

    # 設定圖表標題與座標軸標籤
    year = obs_time[0].year
    if 'title' in kwargs:
        title = f'{year}_{kwargs['title']}'
    else:
        title = f'{year}觀測平均降雨組體圖'
    plt.title(title, fontsize=16)
    plt.xlabel('日期時間')
    plt.ylabel('雨量(MM)')

    # 設定X軸日期的顯示格式與間隔
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))  # 每30天一個標籤

    # 顯示網格線與圖例
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 根據傳入參數決定儲存檔案或直接顯示
    if 'save_path' in kwargs:
        os.makedirs(output_dir, exist_ok=True)
        filename = kwargs['save_path']
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f'儲存至檔案: {filepath}')
    else:
        # 如果 'plot_show' 設為 False，則關閉圖形，避免在後端執行時跳出視窗
        if 'plot_show' in kwargs and kwargs['plot_show'] == False:
            plt.close()
    return




# 繪製時間序列
def plot_ts(obs_time:list[datetime], raw_flow:np.ndarray, predictions:np.ndarray,
            forecast_horizon:int=5,output_dir="./Result", **kwargs):
    """繪製觀測流量與預測流量的時間序列對比圖。

    Args:
        obs_time (list[datetime]): 觀測時間的列表。
        raw_flow (np.ndarray): 觀測流量數據的 NumPy 陣列 (單位: CMS)。
        predictions (np.ndarray): 預測流量數據的 NumPy 陣列 (單位: CMS)。
        forecast_horizon (int, optional): 預報領先時間，用於圖表標題。預設為 5。
        output_dir (str, optional): 圖片輸出的資料夾路徑。預設為 "./Result"。
        **kwargs: 其他可選參數，例如:
            - CE (float): 效率係數 (Nash-Sutcliffe Efficiency)。
            - EQP (float): 洪峰流量誤差百分比。
            - save_path (str): 圖片儲存的完整檔案名稱。
            - plot_show (bool): 是否顯示圖表，若為 False 則不顯示。
    """
    # 建立一個新的圖形，並設定其大小
    fig = plt.figure(figsize=(18,6))
    # 繪製觀測流量與預測流量的線圖
    plt.plot(obs_time, raw_flow, '-', label='觀測流量') # type: ignore
    plt.plot(obs_time, predictions, '--', label='預測流量') # type: ignore

    # 計算預測流量的統計數據
    maxValue = predictions.max()
    minValue = predictions.min()
    totalVolume = predictions.sum() * 3600 / 1000000 # 總逕流體積 (MCM)
    # 計算觀測與預測之間的皮爾森相關係數
    res = pearsonr(raw_flow, predictions)

    # 在圖表上新增文字框，顯示預測流量的統計資訊
    plt.text(
        x=0.02, y=0.8,                        # 位置（相對坐標）
        s=f"預測平均值:{np.mean(raw_flow):.2f}CMS\n最大值:{maxValue:.2f}CMS\n最小值:{minValue:.2f}CMS\n總逕流體積:{totalVolume:.2f}MCM",  # 文本内容
        transform=plt.gca().transAxes,          # 坐标系转换
        bbox=dict(facecolor='white', alpha=0.8) # 白色背景框
    )

    # 組合圖表標題，包含年份、相關係數及其他傳入的評估指標
    year = obs_time[0].year
    title = f'{year}翡翠水庫集水區入流量驗證(預測第{forecast_horizon}小時)\n'
    if 'CE' in kwargs:
        title += f",CE={kwargs['CE']:.2f}"
    if 'EQP' in kwargs:
        title += f",EQP={kwargs['EQP']:.2f}%"
    title += f"Corr.={res.statistic:.4f}" # type: ignore

    # 設定圖表標題與座標軸標籤
    plt.title(title, fontsize=18)
    plt.xlabel('日期')
    plt.ylabel('流量(CMS)')

    # 顯示網格線與圖例
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 根據傳入參數決定儲存檔案或直接顯示
    if 'save_path' in kwargs:
        os.makedirs(output_dir, exist_ok=True)
        filename = kwargs['save_path']
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f'儲存至檔案: {filepath}')
    else:
        # 如果 'plot_show' 設為 False，則關閉圖形
        if 'plot_show' in kwargs and kwargs['plot_show'] == False:
            plt.close()
    return


# 繪圖
def plot_xy(y_true:np.ndarray, predictions:np.ndarray, forecast_horizon:int=5,**kwargs):
    """繪製觀測值 vs. 預測值的散佈圖 (XY Plot)。

    Args:
        y_true (np.ndarray): 觀測值 (真值) 的 NumPy 陣列。
        predictions (np.ndarray): 預測值的 NumPy 陣列。
        forecast_horizon (int, optional): 預報領先時間，用於圖表標題。預設為 5。
        **kwargs: 其他可選參數，例如:
            - save_path (str): 圖片儲存的完整檔案名稱。
            - plot_show (bool): 是否顯示圖表，若為 False 則不顯示。
    """
    fig = plt.figure()

    # 繪製散佈圖
    plt.scatter(y_true, predictions, alpha=0.5)
    # 繪製一條 y=x 的紅色虛線作為參考線
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    # 設定圖表標題與座標軸標籤
    plt.title(f'觀測值vs模擬值(預測第{forecast_horizon}小時)')
    plt.xlabel('觀測值')
    plt.ylabel('模擬值')
    plt.grid(True)

    plt.tight_layout()
    # 根據傳入參數決定儲存檔案或直接顯示
    if 'save_path' in kwargs:
        fig.savefig(kwargs['save_path'], dpi=300, bbox_inches='tight')
        print(f'儲存至檔案: {kwargs["save_path"]}')
    else:
        # 如果 'plot_show' 設為 False，則關閉圖形
        if 'plot_show' in kwargs and kwargs['plot_show'] == False:
            plt.close()
    return


def plot_pred_rr_chart(obsdate:list[datetime], rainfall:np.ndarray, obs_flow:np.ndarray, pred_flow:np.ndarray, output_dir="./Result", **kwargs):
    """繪製觀測及預測降雨逕流歷線圖

    Args:
        obsdate (list[datetime]): 日期時間列表。
        rainfall (np.ndarray): 降雨數據 (MM)。
        obs_flow (np.ndarray): 觀測流量數據 (CMS)。
        pred_flow (np.ndarray): 預測流量數據 (CMS)。
        output_dir (str, optional): 圖片輸出的資料夾路徑。預設為 "./Result"。
        **kwargs: 其他可選參數，例如:
            - pred_hour (int): 預測時長。
            - RunoffCoeff (float): 逕流係數。
            - CE (float): 效率係數。
            - EQP (float): 洪峰流量誤差百分比。
            - COR (float): 相關係數。
            - save_path (str): 圖片儲存的完整檔案名稱。
    """
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()  # 創建右側 Y 軸
    maxValue = max(obs_flow.max(), pred_flow.max()) * 1.67

    # 流量歷線圖
    ax1.plot(obsdate, obs_flow, '-', label='觀測流量')
    ax1.plot(obsdate, pred_flow, '--', label='預測流量')
    ax1.set_ylabel('流量 (m³/s)', color='black', fontsize=12)
    ax1.tick_params(axis='y', colors='#1f77b4')  # 同步刻度顏色
    ax1.set_ylim(top=maxValue)  # 預留頂部空間

    # 降雨組體圖
    ax2.bar(obsdate, rainfall,  width=0.02,  # 柱寬按小時級數據調整
                   color='#2ca02c',
                   alpha=0.7,
                   label='降雨量 (mm)')
    ax2.set_ylabel('降雨量 (mm)', color='black', fontsize=12)
    ax2.tick_params(axis='y', colors='#2ca02c')
    rain_maxValue = 120
    ax2.set_ylim(top=rain_maxValue)  # 預留頂部空間
    ax2.invert_yaxis() # 反轉右側 Y 軸

    # 組合圖表標題
    year = obsdate[0].year
    title = f'{year}翡翠水庫集水區入流量預測結果圖'
    if 'pred_hour' in kwargs:
        title +=f"(預測{kwargs['pred_hour']}h)"
    
    metrics = kwargs.get('metrics', {})
    if metrics:
        ce = metrics.get('CE', -99)
        eqp = metrics.get('EQP', -99)
        cor = metrics.get('COR', -99)
        title += f"\nCE={ce:.2f}, EQP={eqp:.2f}%, COR={cor:.4f}"

    plt.title(title, fontsize=14)
    plt.xlabel('日期時間')

    # 設定X軸日期的顯示格式與間隔
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=48))

    plt.grid(True)
    ax1.legend(loc='upper left', shadow=True, fontsize='large')
    ax2.legend(loc='upper right', shadow=True, fontsize='large')
    plt.tight_layout()

    # 儲存至檔案
    if 'save_path' in kwargs:
        os.makedirs(output_dir, exist_ok=True)
        filename = kwargs['save_path']
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f'儲存至檔案: {filepath}')
    else:
        plt.show()
    return

def plot_rainfall_runoff(obsdate:list[datetime], rainfall:np.ndarray, flow:np.ndarray, output_dir="./Result", **kwargs):
    """繪製降雨逕流歷線圖

    Args:
        obsdate (list[datetime]): 日期時間列表。
        rainfall (np.ndarray): 降雨數據 (MM)。
        flow (np.ndarray): 流量數據 (CMS)。
        output_dir (str, optional): 圖片輸出的資料夾路徑。預設為 "./Result"。
        **kwargs: 其他可選參數，例如:
            - RunoffCoeff (float): 逕流係數。
            - save_path (str): 圖片儲存的完整檔案名稱。
            - title (str): 圖表標題。
    """
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()  # 創建右側 Y 軸

    # 流量歷線圖
    ax1.plot(obsdate, flow, '-', label='流量')
    ax1.set_ylabel('流量 (m³/s)', color='black', fontsize=12)
    ax1.tick_params(axis='y', colors='#1f77b4')
    ax1.set_ylim(top=flow.max() * 1.67)

    # 降雨組體圖
    ax2.bar(obsdate, rainfall, width=0.02, color='#2ca02c', alpha=0.7, label='降雨量 (mm)')
    ax2.set_ylabel('降雨量 (mm)', color='black', fontsize=12)
    ax2.tick_params(axis='y', colors='#2ca02c')
    ax2.set_ylim(top=120)  # 設定降雨量最大值
    ax2.invert_yaxis()  # 反轉右側 Y 軸

    # 標題
    default_title = f'{obsdate[0].year} 降雨逕流歷線圖'
    if 'RunoffCoeff' in kwargs:
        default_title += f"\n逕流係數={kwargs['RunoffCoeff']:.4f}"
    title = kwargs.get('title', default_title)
    plt.title(title, fontsize=14)
    plt.xlabel('日期時間')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=48))

    plt.grid(True)
    ax1.legend(loc='upper left', shadow=True, fontsize='large')
    ax2.legend(loc='upper right', shadow=True, fontsize='large')
    plt.tight_layout()

    if 'save_path' in kwargs:
        filepath = os.path.join(output_dir, kwargs['save_path'])
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f'儲存至檔案: {filepath}')
    else:
        plt.show()
    return

def plot_prediction_vs_observation(obs_df: pd.DataFrame, pred_df: pd.DataFrame, output_dir="./Result", **kwargs):
    """
    繪製觀測值與預測值的降雨逕流歷線圖。

    Args:
        obs_df (pd.DataFrame): 包含觀測流量 'flow' 和 'rainfall' 的 DataFrame，以 datetime 為索引。
        pred_df (pd.DataFrame): 包含預測流量 'pred_flow' 的 DataFrame，以 datetime 為索引。
        output_dir (str, optional): 輸出目錄. Defaults to "./Result".
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # 繪製流量
    ax1.plot(obs_df.index, obs_df['flow'], '-', label='Observed Flow', color='#1f77b4')
    ax1.plot(pred_df.index, pred_df['pred_flow'], '--', label='Predicted Flow', color='#ff7f0e')
    
    max_flow_val = max(obs_df['flow'].max(), pred_df['pred_flow'].max()) * 1.2
    ax1.set_ylabel('Flow (CMS)', color='black', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_ylim(bottom=0, top=max_flow_val)

    # 繪製降雨
    ax2.bar(obs_df.index, obs_df['rainfall'], width=0.03, color='#2ca02c', alpha=0.6, label='Rainfall (mm)')
    ax2.set_ylabel('Rainfall (mm)', color='black', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    ax2.set_ylim(top=obs_df['rainfall'].max() * 4) # 讓降雨圖有足夠空間
    ax2.invert_yaxis()

    # 設定標題和標籤
    title = f"Runoff Prediction Result"
    if 'pred_hour' in kwargs:
        title += f" (Lead Time: {kwargs['pred_hour']}h)"
    if 'metrics' in kwargs:
        metrics = kwargs['metrics']
        title += f"\nCE={metrics.get('CE', 'N/A'):.2f}, COR={metrics.get('COR', 'N/A'):.4f}, EQP={metrics.get('EQP', 'N/A'):.2f}%"
    ax1.set_title(title, fontsize=16)
    ax1.set_xlabel('Datetime')

    # 設定圖例和網格
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    ax1.grid(True)
    fig.tight_layout()

    if 'save_path' in kwargs:
        filepath = os.path.join(output_dir, kwargs['save_path'])
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f'Chart saved to: {filepath}')
    else:
        plt.show()
    return
