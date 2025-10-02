# GetRRData.py
# 2025-05-20
# 取得降雨逕流觀測資料

import numpy as np
import pandas as pd


def GetRRData(df:pd.DataFrame, start_date:str, end_date:str) -> tuple[np.ndarray, np.ndarray]:
    """取得指定開始到結束日期的觀測雨量及流量時間序列資料
       資料詳如: data/2024翡翠水庫集水區csv

    Args:
        df (pd.DataFrame): 原始資料
        start_date (str): 開始日期時間, ex. start_date = '2024-09-15 00:00'
        end_date (str): 結束日期時間, ex. end_date = '2024-10-30 23:00'

    Returns:
        _type_: _description_
    """
    filtered_df = df[(df['INFO_DATE'] >= start_date) & (df['INFO_DATE'] <= end_date)]
    rain_sum = 0.0
    flow_sum = 0.0
    raw_rainfall = filtered_df['FCST_Rainfall'].to_numpy()
    raw_flow = filtered_df['Q_IN'].to_numpy()
    for flow, rain in zip(filtered_df['Q_IN'], filtered_df['FCST_Rainfall']):
        # print(f"{rain:.4f},{flow:.4f}")
        rain_sum += rain
        flow_sum += flow
    print(f'rain_sum={rain_sum:.4f}(mm), flow_sum={flow_sum:.4f}(cms-hour)')
    basin_area = 303.0 # km^2
    rain_vol = basin_area * 1000*1000 * rain_sum / 1000.0 # m^3
    flow_vol = flow_sum * 3600.0 # m^3
    print(f'rain_vol={rain_vol:.4f}(m^3)')
    print(f"flow_vol={flow_vol:.4f}(m^3)")
    print(f"Runoff Coeff.={flow_vol/rain_vol:.4f}")
    return raw_rainfall, raw_flow