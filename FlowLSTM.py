# FlowLSTM.py
# 2025-05-20

import torch
import torch.nn as nn

#  定義模型
 
class FlowLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=6):
        super(FlowLSTM, self).__init__()
        self.hidden_size = hidden_size  # LSTM隱藏層
        self.num_layers = num_layers     # LSTM堆叠的層數
        # 定義LSTM層：輸入特征維度input_size，隱藏層維度hidden_size，層數num_layers
        # batch_first=True表示輸入/輸出張量的形狀(batch_size, seq_len, feature_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 定義全連接層：將LSTM最后一個時間步的隱藏狀態映射到輸出維度output_size
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 初始化LSTM的隱藏狀態h0和細胞狀態c0，形狀(num_layers, batch_size, hidden_size)
        # 利用x.device確保張量與輸入數據在同一設備（CPU/GPU）上
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # LSTM前向傳播：輸入x，輸出out（包含所有時間步的隱藏狀態）
        out, _ = self.lstm(x, (h0, c0))
        # 取最后一個時間步的输出，利用全連接層得到最終預測結果
        return self.fc(out[:, -1, :])
    
'''參數計算
參數量=4×[(input_size+hidden_size)×hidden_size+hidden_size]×num_layers
第1層
4×[(2+64)×64+64]=4×(4224+64)=17,152
第2層
4×[(64+64)×64+64]=4×(8192+64)=33,024
總LSTM參數
17,152 + 33,024 = 50,176
全連接層參數量=hidden_size×output_size+output_size
64×6+6=384+6=390
50,176(LSTM)+390(全連接層)=50,566
'''