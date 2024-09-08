
import time
import random
import torch
import numpy as np


class State:
    def __init__(self, bars_count, commission_perc, default_slippage: float, device: torch.device):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.default_slippage = default_slippage
        self.device = device

    def _cur_close(self):
        """
            Calculate real close price for the current bar
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)

    def reset(self, prices, offset):
        assert offset >= self.bars_count-1
        self._offset = offset

        # 初始化相關資料
        self._init_cash = 10000

        self._map = {
            "last_action": 0.0,
            "last_share": 0.0,
            "diff_share": 0.0,
            "calculate_cash": self._init_cash,
            "postion_value": 0.0,
            "commission": 0.0,
            "slippage": 0.0,
            "Close": 0.0,
            "sum": self._init_cash
        }

        self._prices = prices
        self.action = 0

    def step(self, action):
        """
            # 專門為了純多單製作
            計算動態資產分配action 可能的範圍 0 ~ 1

                0.8429519115130819
                0.5133293824170155
                0.3696803100350855
                0.4621302561424804
                0.37277980115611786

            buy and sell
        """
        self.action = action  # 紀錄這次的百分比變化

        reward = 0.0
        done = False

        _cur_close = self._cur_close()

        # 取得即時下單比例
        thisusemoeny = (self._map['last_share'] * _cur_close +
                        self._map['calculate_cash']) * abs(action)
        print("step內---------------------------------------------------------")
        print("此次內容:")
        print("本次動作:",self.action)
        print("本次動作:",self.action)
        
        time.sleep(50)
        print("step內---------------------------------------------------------")
        # 預估說大概可以購買多少的部位並且不能超過原始本金
        cost = _cur_close * (1 + self.default_slippage + self.commission_perc)
        share = thisusemoeny / cost

        self._map['last_action'] = action
        diff_share = share - self._map['last_share']
        self._map['diff_share'] = diff_share

        # 計算傭金和滑價
        commission = abs(diff_share) * _cur_close * self.commission_perc
        slippage = abs(diff_share) * _cur_close * \
            self.default_slippage

        # 計算賣出獲得金額 及買入獲得金額
        thissellmoney = abs(self._map['last_share']) * _cur_close
        thisbuymoney = abs(share) * _cur_close
        self._map['calculate_cash'] = self._map['calculate_cash'] + \
            thissellmoney - thisbuymoney - commission - slippage

        self._map['last_share'] = share
        self._map['commission'] = commission
        self._map['slippage'] = slippage
        self._map['postion_value'] = abs(share) * _cur_close
        self._map['Close'] = _cur_close
        reward = ((self._map['calculate_cash'] + self._map['postion_value']
                   ) - self._map['sum']) / self._map['sum']
        self._map['sum'] = self._map['calculate_cash'] + \
            self._map['postion_value']

        # 上一個時步的狀態 ================================
        self._offset += 1
        # 判斷遊戲是否結束
        done |= self._offset >= self._prices.close.shape[0] - 1


        return reward, done




class State_time_step(State):
    """
        專門用於transformer的時序函數。
    """
    @property
    def shape(self):
        return (self.bars_count, 14)

    def encode(self):
        res = torch.zeros(size=self.shape, dtype=torch.float32)
        
        ofs = self.bars_count
        for bar_idx in range(self.bars_count):
            res[bar_idx][0] = self._prices.high[self._offset - ofs + bar_idx]
            res[bar_idx][1] = self._prices.low[self._offset - ofs + bar_idx]
            res[bar_idx][2] = self._prices.close[self._offset - ofs + bar_idx]
            res[bar_idx][3] = self._prices.volume[self._offset - ofs + bar_idx]
            res[bar_idx][4] = self._prices.volume2[self._offset - ofs + bar_idx]
            res[bar_idx][5] = self._prices.quote_av[self._offset - ofs + bar_idx]
            res[bar_idx][6] = self._prices.quote_av2[self._offset - ofs + bar_idx]
            res[bar_idx][7] = self._prices.trades[self._offset - ofs + bar_idx]
            res[bar_idx][8] = self._prices.trades2[self._offset - ofs + bar_idx]
            res[bar_idx][9] = self._prices.tb_base_av[self._offset - ofs + bar_idx]
            res[bar_idx][10] = self._prices.tb_base_av2[self._offset - ofs + bar_idx]
            res[bar_idx][11] = self._prices.tb_quote_av[self._offset - ofs + bar_idx]
            res[bar_idx][12] = self._prices.tb_quote_av2[self._offset - ofs + bar_idx]

        res[:, 13] = self._map['last_action']
        return res
    
class State2D(State):
    """
        用於處理 2D 數據，如圖像。輸入數據的形狀通常是 (N, C, H, W)，其中 N 是批次大小，C 是通道數，H 是高度，W 是寬度。
        典型應用：圖像處理、計算機視覺任務（如圖像分類、物體檢測等）。
    """
    @property
    def shape(self):
        return (5, self.bars_count)

    def encode(self):
        res = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        ofs = self.bars_count-1
        res[0] = self._prices.high[self._offset-ofs:self._offset+1]
        res[1] = self._prices.low[self._offset-ofs:self._offset+1]
        res[2] = self._prices.close[self._offset-ofs:self._offset+1]
        res[3] = self._prices.volume[self._offset-ofs:self._offset+1]
        res[4] = self.action

        res = res.unsqueeze(0)
        return res
    
class Env:
    def __init__(self, keyword:str, bars_count: int, commission_perc: float, default_slippage: float, prices: dict, random_ofs_on_reset: bool, device: torch.device):
        self.bars_count = bars_count
        
        
        if keyword == 'Transformer':
            self._count_state = State_time_step(bars_count=self.bars_count,
                                        commission_perc=commission_perc,
                                        default_slippage=default_slippage,
                                        device=device
                                        )
        else:
            self._count_state = State2D(bars_count=self.bars_count,
                                        commission_perc=commission_perc,
                                        default_slippage=default_slippage,
                                        device=device
                                        )
        
        self.done = False
        self._prices = prices
        self.random_ofs_on_reset = random_ofs_on_reset

    def reset(self):
        self._instrument = random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]

        bars = self._count_state.bars_count
        if self.random_ofs_on_reset:
            offset = random.randint(0, prices.high.size()[0] - bars * 10) + bars
        else:
            offset = bars

        print("目前步數:", offset)
        self._count_state.reset(prices, offset)
        return self._count_state.encode()

    def step(self, action):
        reward, done = self._count_state.step(action)  # 這邊會更新步數

        
        obs = self._count_state.encode()  # 呼叫這裡的時候就會取得新的狀態
        info = {
            "instrument": self._instrument,
            "offset": self._count_state._offset,

        }
        
        return obs, reward, done, info

    def render(self):
        # 簡單打印當前狀態
        pass

    def close(self):
        pass

    def env_info(self):
        """
            用來儲存這個環境的資訊
        """
        return {
            "input_size":self._count_state.shape[1],
            "action_size":1
        }
        
    def action_sample(self):
        """

        Returns:
            _type_: _description_
        """
        return torch.rand(1, dtype=torch.float32)
# app = State_time_step(bars_count=300, commission_perc=0.002)
# app.reset(prices=DataFeature().get_train_net_work_data_by_path(['TRBUSDT'])['TRBUSDT'], offset=300)


# while True:
#     app.step(action=np.random.uniform(0, 1))
#     time.sleep(1)
