import os
import json
import atexit
from functools import wraps
from typing import Any, Dict, List
import numpy as np
import pprint
from Brain.DQN.lib.reward import Actions
import time

class StepAuditLogger:
    """
    DQN 環境 State_time_step.step 專用審計裝飾器。
    緩衝 50 筆批次寫入本地端點 JSONL，專門用於驗證部位轉移、多空損益與滑價。
    """

    def __init__(
        self,
        log_path: str = "audit_logs/dqn_step_audit.jsonl",
        batch_size: int = 50,
        enabled: bool = True,
    ):
        self.log_path = log_path
        self.batch_size = batch_size
        self.enabled = enabled
        self.buffer: List[Dict[str, Any]] = []
        self._registered_atexit = False

        if self.enabled:
            self._ensure_log_dir()
            atexit.register(self.flush)
            self._registered_atexit = True

    def _ensure_log_dir(self):
        log_dir = os.path.dirname(os.path.abspath(self.log_path))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def flush(self):
        """將記憶體中累積的審計數據批次刷入硬碟"""
        if not self.buffer:
            return
        try:
            self._ensure_log_dir()
            with open(self.log_path, "a", encoding="utf-8") as f:
                for record in self.buffer:
                    f.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            default=self._json_serializer,
                        )
                        + "\n"
                    )
            self.buffer.clear()
        except Exception as e:
            print(f"[StepAuditLogger Warning] Flush failed: {e}")

    @staticmethod
    def _json_serializer(obj):
        """轉換 numpy / Enum 資料型態為 JSON 標準型態"""
        if isinstance(obj, (np.floating, float)):
            return round(float(obj), 6)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Actions):
            return f"Actions.{obj.name} ({obj.value})"
        return str(obj)

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)

            # 鎖定 State_time_step 實例 (選項 A)
            state = args[0]
            action = args[1] if len(args) > 1 else kwargs.get("action")
            action_enum = Actions(action) if isinstance(action, int) else action

            # 1. 擷取進入前的狀態 (Input Snapshot)
            pre_step = getattr(state, "game_steps", 0)
            pre_offset = getattr(state, "_offset", 0)
            pre_pos = state.position
            pre_open = float(state.open_price)
            pre_bar = int(state.trade_bar)
            pre_equity = float(state.TotalPortfolioPercent)
            pre_cost = float(state.cost_sum)
            pre_closecash = float(state.closecash)

            cur_close = (
                float(state._prices.close[pre_offset])
                if hasattr(state, "_prices") and pre_offset < len(state._prices.close)
                else 0.0
            )
            prev_close = (
                float(state._prices.close[pre_offset - 1])
                if hasattr(state, "_prices") and 0 < pre_offset <= len(state._prices.close)
                else cur_close
            )
            pre_bench_ret = (
                float((cur_close - prev_close) / prev_close) if prev_close != 0 else 0.0
            )

            # 2. 執行原函數
            reward, done = func(*args, **kwargs)

            # 3. 擷取執行後的狀態 (Output Snapshot)
            post_pos = state.position
            post_open = float(state.open_price)
            post_bar = int(state.trade_bar)
            post_equity = float(state.TotalPortfolioPercent)
            post_cost = float(state.cost_sum)
            post_closecash = float(state.closecash)

            # 4. 放空機制自動語意分析
            op_type = "UNKNOWN"
            if pre_pos == 0 and post_pos == -1:
                op_type = "OPEN_SHORT"
            elif pre_pos == -1 and post_pos == 0:
                op_type = "COVER_SHORT"
            elif pre_pos == 0 and post_pos == 1:
                op_type = "OPEN_LONG"
            elif pre_pos == 1 and post_pos == 0:
                op_type = "CLOSE_LONG"
            elif (pre_pos == 1 and action_enum == Actions.Buy) or (
                pre_pos == -1 and action_enum == Actions.Sell
            ):
                op_type = "WRONG_TRADE"
            elif pre_pos == post_pos and action_enum == Actions.Hold:
                op_type = "HOLD"

            is_wrong = (op_type == "WRONG_TRADE")
            cost_diff = post_cost - pre_cost
            commission_charged = (cost_diff > 0)
            slip = getattr(state, "max_default_slippage", 0.0)

            # 滑價檢驗標籤
            slippage_check = "N/A"
            if op_type == "OPEN_SHORT":
                expected_entry = cur_close * (1.0 - slip)
                if abs(post_open - expected_entry) < 1e-4:
                    slippage_check = f"PASS ({cur_close} * (1 - {slip}) = {round(post_open, 6)})"
                else:
                    slippage_check = f"FAIL ({round(post_open, 6)} != {round(expected_entry, 6)})"
            elif op_type == "OPEN_LONG":
                expected_entry = cur_close * (1.0 + slip)
                if abs(post_open - expected_entry) < 1e-4:
                    slippage_check = f"PASS ({cur_close} * (1 + {slip}) = {round(post_open, 6)})"
                else:
                    slippage_check = f"FAIL ({round(post_open, 6)} != {round(expected_entry, 6)})"

            pos_name_map = {0: "FLAT (0)", 1: "LONG (1)", -1: "SHORT (-1)"}
            semantic_transition = f"{pos_name_map.get(pre_pos, pre_pos)} -> {pos_name_map.get(post_pos, post_pos)}"

            record = {
                "step_index": pre_step,
                "step": pre_step,
                "offset": pre_offset,
                "input": {
                    "action": f"Actions.{action_enum.name} ({action_enum.value})",
                    "pre_position": pre_pos,
                    "pre_trade_bar": pre_bar,
                    "pre_open_price": pre_open,
                    "current_close": cur_close,
                    "current_bar_close": cur_close,
                    "pre_benchmark_return": pre_bench_ret,
                    "pre_total_equity": pre_equity,
                    "pre_cost_sum": pre_cost,
                    "pre_closecash": pre_closecash,
                },
                "output": {
                    "reward": reward,
                    "done": done,
                    "post_position": post_pos,
                    "post_trade_bar": post_bar,
                    "post_open_price": post_open,
                    "post_total_equity": post_equity,
                    "step_equity_diff": post_equity - pre_equity,
                    "cost_diff": cost_diff,
                    "realized_diff": post_closecash - pre_closecash,
                    "realized_pnl_diff": post_closecash - pre_closecash,
                },
                "short_selling_audit": {
                    "transition": semantic_transition,
                    "operation_type": op_type,
                    "is_wrong_trade": is_wrong,
                    "entry_slippage_check": slippage_check,
                    "commission_charged": commission_charged,
                },
                "audit": {
                    "operation_type": op_type,
                    "transition": f"{pre_pos} -> {post_pos}",
                },
            }
            self.buffer.append(record)
            if len(self.buffer) >= self.batch_size:
                self.flush()

            return reward, done

        return wrapper


# 預設單例裝飾器
step_audit_logger = StepAuditLogger(
    log_path="audit_logs/dqn_step_audit.jsonl",
    batch_size=50,
    enabled=True,
)
