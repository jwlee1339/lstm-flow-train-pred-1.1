# ModelEV.py
# 計算降雨逕流模式評估指標 CE

# ### 降雨逕流模式評估指標

import math
import numpy as np


def ModelTest(qo:np.ndarray, preq:np.ndarray) -> dict:
    """計算觀測及預測時間序列評估指標

    Args:
        qo (list[float]): 觀測序列
        preq (list[float]): 模式預測(模擬)序列

    Returns:
        dict: CE, VER, COR, EQP, pwMSE
    """
    Qavg = xsum = ysum = qsum = esum = Nsum = 0.0
    Dsum = Dsum1 = Qeavg = 0.0
    Qp = Qp1 = -99999
    if qo.size != preq.size:
        print(f'[WARN]qo.size={qo.size} != preq.size={preq.size}')
    NoOfData = min(qo.size, preq.size)

    for i in range(0, NoOfData):
        xsum = xsum + (qo[i] - preq[i]) * (qo[i] - preq[i])
        qsum = qsum + qo[i]
        esum = esum + preq[i]
        Qp = max(Qp, qo[i])
        Qp1 = max(Qp1, preq[i])

    ysum = Nsum = Dsum = Dsum1 = 0.0
    Qavg = qsum / NoOfData    # average Qobs 
    Qeavg = esum / NoOfData   # average Qest 
    pwMSE = pwSum = 0.0
    w = 0.0

    for i in range(0, NoOfData):
        ysum += (qo[i] - Qavg) * (qo[i] - Qavg)
        Nsum += (qo[i] - Qavg) * (preq[i] - Qeavg)
        Dsum += (qo[i] - Qavg) * (qo[i] - Qavg)
        Dsum1 += (preq[i] - Qeavg) * (preq[i] - Qeavg)

        # from HEC-HMS, Peak-weighted mean square error 
        # w = Math.Abs(drh[i] + Qavg) / (2 * Qavg); //-- from HEC-HMS

        w = qo[i] / (Qavg);   # the best! do not change it unless you are very cofidence!
        pwSum += (qo[i] - preq[i]) * (qo[i] - preq[i]) * w

    # 計算ETP 洪峰時刻誤差
    obs_max_index = np.argmax(qo)
    pred_max_index = np.argmax(preq)
    ETP = pred_max_index - obs_max_index
    # 效率係數
    CE = 1.0 - xsum / ysum
    # 體積誤差係數
    VER = (esum - qsum) / qsum * 100.0
    # 相關係數
    COR = Nsum / math.sqrt(Dsum * Dsum1)
    # 洪峰誤差%
    EQp = (Qp1 - Qp) / Qp * 100
    # 洪峰流量加權MSE
    pwMSE = math.sqrt(pwSum) / NoOfData

    # store to dictionary
    res = {}
    res['CE'] = CE
    res['VER'] = VER
    res['COR'] = COR
    res['ETP'] = ETP
    res['EQP'] = EQp
    res['pwMSE'] = pwMSE
    return res


# Print Model Test Results
def PrintEvaluationFactors(res):
    print('* 模式評估指標 :')
    s = f"CE= {res['CE']:8.2f}, COR ={res['COR']:8.2f}, VER ={res['VER']:8.2f}, "
    s += f"HecObj ={res['pwMSE']:8.2f}"
    print(s)

'''
 ---------------------------------------------
 Model Objective Value
 input
   Qobs     : observed flow in cms
   Qt       : computed flow in cms
   iObjFunc : objective function index
 output
 return obj, eva_res
 ----------------------------------------------
'''
def ModelObjectiveValue(Qobs, Qcomp, iObjFunc=5):
    eva_res = ModelTest(Qobs, Qcomp)
    w = 0.55
    obj = -9999

    if iObjFunc == 1:
        obj = abs(1.0 - eva_res['CE'])
    elif iObjFunc == 2:
        obj = abs(eva_res['EQP'])
    elif iObjFunc == 3:
        obj = abs(1 - eva_res['COR'])
    elif iObjFunc == 4:      
        obj = w * abs(1 - eva_res['CE']) + (1.0 - w) * abs(eva_res['EQ'] / 100.0)
    elif iObjFunc == 5:
        obj = eva_res['pwMSE']
    return obj, eva_res


# main()
def main():
    obsFlow = np.array([0.0, 0.0, 50, 310, 590, 720, 850, 820, 600, 410, 270, 177, 114, 93, 88])
    compFlow = np.array([0.7687499999999999, 1.2365625, 55.71987890625001, 316.22973093749994, 
               597.6484270456054, 734.5773155272685, 865.7250035369163, 895.1267948904416,
                623.8426152741733, 452.8424095736908, 338.185670627656, 248.72905447369232, 
                187.4674284953931, 174.95154425023685, 161.6783751157142])

    res = ModelTest(obsFlow, compFlow)
    PrintEvaluationFactors(res)

    result = ModelObjectiveValue(obsFlow, compFlow, 1)
    print(f'目標函數值 : 1-CE = {result:9.4f}')


if __name__ == '__main__':
    main()	