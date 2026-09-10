import unittest
import numpy as np
import torch
import gymnasium as gym
from Brain.Common.DataFeature import Prices
from Brain.PPO2.lib.environment import (
    State_time_step,
    BaseTradingEnv,
    ProductionEnv,
    TrainingEnv,
)


def create_dummy_prices(n_bars=200, close_prices=None):
    if close_prices is None:
        close_prices = np.linspace(100.0, 150.0, n_bars)
    else:
        n_bars = len(close_prices)
        close_prices = np.array(close_prices, dtype=np.float32)

    dummy_arr = np.zeros(n_bars, dtype=np.float32)
    return Prices(
        open=close_prices,
        high=close_prices * 1.01,
        low=close_prices * 0.99,
        close=close_prices,
        log_open=dummy_arr,
        log_high=dummy_arr,
        log_low=dummy_arr,
        log_close=dummy_arr,
        log_volume=dummy_arr,
        log_quote_av=dummy_arr,
        log_trades=dummy_arr,
        log_tb_base_av=dummy_arr,
        log_tb_quote_av=dummy_arr,
        log_ma_30=dummy_arr,
        log_ma_60=dummy_arr,
        log_ma_120=dummy_arr,
        log_ma_240=dummy_arr,
        log_ma_360=dummy_arr,
        age_log_minutes=dummy_arr,
        age_years=dummy_arr,
        month_sin=dummy_arr,
        month_cos=dummy_arr,
        day_sin=dummy_arr,
        day_cos=dummy_arr,
        hour_sin=dummy_arr,
        hour_cos=dummy_arr,
        minute_sin=dummy_arr,
        minute_cos=dummy_arr,
        dayofweek_sin=dummy_arr,
        dayofweek_cos=dummy_arr,
        week_sin=dummy_arr,
        week_cos=dummy_arr,
    )


class TestPPO2ContinuousEnvironment(unittest.TestCase):
    def setUp(self):
        self.bars_count = 10
        self.commission = 0.001
        self.slippage = 0.0005
        self.total_friction = self.commission + self.slippage
        self.n_steps = 50

    def get_state(self, deadzone=0.01):
        return State_time_step(
            bars_count=self.bars_count,
            commission_perc=self.commission,
            model_train=True,
            default_slippage=self.slippage,
            N_steps=self.n_steps,
            deadzone_threshold=deadzone,
        )

    # -------------------------------------------------------------
    # Phase 1: Spaces & Cleanup Tests
    # -------------------------------------------------------------
    def test_spaces_and_dimensions(self):
        state = self.get_state()
        env = ProductionEnv({"TEST": create_dummy_prices(100)}, state)

        # Action space must be Box(-1.0, 1.0, (1,))
        self.assertIsInstance(env.action_space, gym.spaces.Box)
        self.assertEqual(env.action_space.shape, (1,))
        self.assertEqual(env.action_space.low[0], -1.0)
        self.assertEqual(env.action_space.high[0], 1.0)

        # Observation space must be Dict containing 'states' and 'time_states'
        self.assertIsInstance(env.observation_space, gym.spaces.Dict)
        self.assertIn("states", env.observation_space.spaces)
        self.assertIn("time_states", env.observation_space.spaces)

        # Check engine_info
        info = env.engine_info()
        self.assertEqual(info["action_space_dim"], 1)
        self.assertEqual(info["data_input_size"], state.getStateShape()[1])
        self.assertEqual(info["time_input_size"], state.getTimeShape()[1])

    # -------------------------------------------------------------
    # Phase 2: Deadzone Filtering
    # -------------------------------------------------------------
    def test_deadzone_filtering(self):
        prices = create_dummy_prices(100, close_prices=[100.0] * 100)
        state = self.get_state(deadzone=0.01)
        state.reset(prices, offset=self.bars_count)

        # Action within deadzone (+0.005) -> should result in position 0.0
        reward, done = state.step(0.005)
        self.assertEqual(state.position, 0.0)
        self.assertEqual(state.open_price, 0.0)
        self.assertEqual(state.trade_bar, 0)
        self.assertEqual(state.cost_sum, 0.0)

        # Action within negative deadzone (-0.010) -> should be 0.0
        reward, done = state.step(-0.010)
        self.assertEqual(state.position, 0.0)
        self.assertEqual(state.open_price, 0.0)

        # Action exceeding deadzone (+0.015) -> should be 0.015
        reward, done = state.step(0.015)
        self.assertEqual(state.position, 0.015)
        self.assertEqual(state.open_price, 100.0)
        self.assertEqual(state.trade_bar, 1)
        self.assertAlmostEqual(state.cost_sum, 0.015 * self.total_friction, places=7)

    # -------------------------------------------------------------
    # Phase 2: Long Position & Accounting
    # -------------------------------------------------------------
    def test_long_position_lifecycle(self):
        # Step 0 (reset at 10): P[10]=100
        # Step 1 (offset 10->11): Agent actions a=1.0 at S_10=100
        # Step 2 (offset 11->12): Agent actions a=1.0 at S_11=110
        # Step 3 (offset 12->13): Agent actions a=0.0 at S_12=110 (full close)
        close_series = [100.0] * 10 + [100.0, 110.0, 110.0, 120.0] + [100.0] * 50
        prices = create_dummy_prices(close_prices=close_series)
        state = self.get_state()
        state.reset(prices, offset=10)

        # Step 1: Open Long 100% at P[10]=100
        r1, d1 = state.step(1.0)
        self.assertEqual(state.position, 1.0)
        self.assertEqual(state.open_price, 100.0)
        self.assertEqual(state.trade_bar, 1)
        expected_cost1 = 1.0 * self.total_friction
        self.assertAlmostEqual(state.cost_sum, expected_cost1, places=6)
        # Unrealized PnL at entry = 0
        expected_equity1 = 1.0 - expected_cost1
        self.assertAlmostEqual(state.TotalPortfolioPercent, expected_equity1, places=6)

        # Step 2: Hold Long at P[11]=110 (10% gain)
        r2, d2 = state.step(1.0)
        self.assertEqual(state.position, 1.0)
        self.assertEqual(state.open_price, 100.0)
        self.assertEqual(state.trade_bar, 2)
        # Cost unchanged (no turnover)
        self.assertAlmostEqual(state.cost_sum, expected_cost1, places=6)
        # Open profit = 1.0 * (110 - 100)/100 = 0.10
        expected_equity2 = 1.0 - expected_cost1 + 0.10
        self.assertAlmostEqual(state.TotalPortfolioPercent, expected_equity2, places=6)
        self.assertAlmostEqual(state.return_history[-1], 0.10, places=6)
        expected_downside2 = state.calculate_downside_risk_numpy([-expected_cost1, 0.10])
        self.assertAlmostEqual(r2, 0.10 - expected_downside2, places=6)

        # Step 3: Close Long at P[12]=110
        r3, d3 = state.step(0.0)
        self.assertEqual(state.position, 0.0)
        self.assertEqual(state.open_price, 0.0)
        self.assertEqual(state.trade_bar, 0)
        expected_cost3 = expected_cost1 + 1.0 * self.total_friction
        self.assertAlmostEqual(state.cost_sum, expected_cost3, places=6)
        self.assertAlmostEqual(state.closecash, 0.10, places=6)
        expected_equity3 = 1.0 - expected_cost3 + 0.10
        self.assertAlmostEqual(state.TotalPortfolioPercent, expected_equity3, places=6)

    # -------------------------------------------------------------
    # Phase 2: Short Position & Accounting
    # -------------------------------------------------------------
    def test_short_position_lifecycle(self):
        # Step 0: reset at 10, P[10]=100
        # Step 1: short 0.5 at S_10=100
        # Step 2: price drops to 90 at S_11=90 -> short gains 0.5 * (100 - 90)/100 = 0.05
        # Step 3: close at S_12=90
        close_series = [100.0] * 10 + [100.0, 90.0, 90.0] + [100.0] * 50
        prices = create_dummy_prices(close_prices=close_series)
        state = self.get_state()
        state.reset(prices, offset=10)

        # Step 1: Open Short 50%
        state.step(-0.5)
        self.assertEqual(state.position, -0.5)
        self.assertEqual(state.open_price, 100.0)
        self.assertEqual(state.trade_bar, 1)

        # Step 2: Hold Short as price drops to 90
        r2, _ = state.step(-0.5)
        self.assertEqual(state.trade_bar, 2)
        # Open profit for short = 0.5 * (100 - 90)/100 = +0.05
        expected_cost = 0.5 * self.total_friction
        expected_equity = 1.0 - expected_cost + 0.05
        self.assertAlmostEqual(state.TotalPortfolioPercent, expected_equity, places=6)
        self.assertAlmostEqual(state.return_history[-1], 0.05, places=6)
        expected_downside2 = state.calculate_downside_risk_numpy([-expected_cost, 0.05])
        self.assertAlmostEqual(r2, 0.05 - expected_downside2, places=6)

        # Step 3: Close short
        state.step(0.0)
        self.assertEqual(state.position, 0.0)
        self.assertEqual(state.open_price, 0.0)
        self.assertEqual(state.trade_bar, 0)
        self.assertAlmostEqual(state.closecash, 0.05, places=6)

    # -------------------------------------------------------------
    # Phase 2: Position Addition (Weighted Average Cost) & Partial Close
    # -------------------------------------------------------------
    def test_add_and_partial_close(self):
        # Step 1: buy 0.4 at 100 -> open_price = 100
        # Step 2: add 0.4 at 120 -> new pos = 0.8, weighted avg = (0.4*100 + 0.4*120)/0.8 = 110
        # Step 3: partial sell 0.3 at 130 -> new pos = 0.5, open_price remains 110, realized PnL = 0.3 * (130-110)/110
        close_series = [100.0] * 10 + [100.0, 120.0, 130.0] + [100.0] * 50
        prices = create_dummy_prices(close_prices=close_series)
        state = self.get_state()
        state.reset(prices, offset=10)

        state.step(0.4)
        self.assertEqual(state.position, 0.4)
        self.assertEqual(state.open_price, 100.0)

        state.step(0.8)
        self.assertEqual(state.position, 0.8)
        self.assertAlmostEqual(state.open_price, 110.0, places=6)
        self.assertEqual(state.trade_bar, 2)

        state.step(0.5)
        self.assertEqual(state.position, 0.5)
        # open_price must remain unchanged on reduction
        self.assertAlmostEqual(state.open_price, 110.0, places=6)
        # Realized profit on 0.3 closed at 130
        expected_pnl = 0.3 * (130.0 - 110.0) / 110.0
        self.assertAlmostEqual(state.closecash, expected_pnl, places=6)
        self.assertEqual(state.trade_bar, 3)

    # -------------------------------------------------------------
    # Phase 2: Position Flip (Long to Short)
    # -------------------------------------------------------------
    def test_position_flip(self):
        # Step 1: long +1.0 at 100
        # Step 2: flip to short -1.0 at 110
        # Old long closed at 110 -> realized gain = 1.0 * 10/100 = 0.10
        # Turnover = | -1.0 - 1.0 | = 2.0 -> cost = 2.0 * friction
        # New short opened at 110 -> open_price = 110, trade_bar = 1
        close_series = [100.0] * 10 + [100.0, 110.0] + [100.0] * 50
        prices = create_dummy_prices(close_prices=close_series)
        state = self.get_state()
        state.reset(prices, offset=10)

        state.step(1.0)
        state.step(-1.0)

        self.assertEqual(state.position, -1.0)
        self.assertAlmostEqual(state.closecash, 0.10, places=6)
        self.assertEqual(state.open_price, 110.0)
        self.assertEqual(state.trade_bar, 1)
        expected_cost = (1.0 + 2.0) * self.total_friction
        self.assertAlmostEqual(state.cost_sum, expected_cost, places=6)

    # -------------------------------------------------------------
    # Phase 3: Rolling Downside Risk Penalty
    # -------------------------------------------------------------
    def test_downside_risk_penalty(self):
        state = self.get_state()
        # Test calculate_downside_risk_numpy
        returns = [0.05, -0.02, -0.04, 0.03]
        # downside diff: [0, 0.02, 0.04, 0] -> sq: [0, 0.0004, 0.0016, 0] -> mean: 0.0005 -> sqrt: 0.02236
        expected_ds = np.sqrt(np.mean([0.0, 0.02**2, 0.04**2, 0.0]))
        self.assertAlmostEqual(
            state.calculate_downside_risk_numpy(returns), expected_ds, places=5
        )

        # Empty returns should be 0.0
        self.assertEqual(state.calculate_downside_risk_numpy([]), 0.0)

    # -------------------------------------------------------------
    # Phase 4: Observation Encoding & Look-Ahead Bias Prevention
    # -------------------------------------------------------------
    def test_observation_encoding_no_lookahead(self):
        close_series = [100.0] * 10 + [100.0, 110.0, 120.0] + [100.0] * 50
        prices = create_dummy_prices(close_prices=close_series)
        state = self.get_state()
        state.reset(prices, offset=10)

        # Initial obs at reset
        obs0 = state.encode()
        self.assertIn("states", obs0)
        self.assertIn("time_states", obs0)
        self.assertEqual(obs0["states"].shape, (self.bars_count, len(state.info_list) + 3))
        self.assertEqual(obs0["time_states"].shape, (self.bars_count, len(state.timelist)))

        # Feature col len(info_list): pos = 0.0
        self.assertEqual(obs0["states"][0, len(state.info_list)], 0.0)
        self.assertEqual(obs0["states"][0, len(state.info_list) + 1], 0.0)
        self.assertEqual(obs0["states"][0, len(state.info_list) + 2], 0.0)

        # Step 1: Open Long 0.8 at S_10 = 100.0
        state.step(0.8)
        # In step(), offset became 11.
        # Now encode(): S_{t-1} is prices.close[offset - 1] = prices.close[10] = 100.0
        obs1 = state.encode()
        self.assertEqual(obs1["states"][0, len(state.info_list)], 0.8)
        # Unrealized return with S_{offset-1}=100 and open_price=100 should be 0.0
        self.assertEqual(obs1["states"][0, len(state.info_list) + 1], 0.0)
        self.assertEqual(obs1["states"][0, len(state.info_list) + 2], 1.0)

        # Step 2: Hold 0.8 at S_11 = 110.0
        state.step(0.8)
        # In step(), offset became 12.
        # Now encode(): S_{offset-1} is prices.close[11] = 110.0
        # Unrealized return = 1 * (110 - 100)/100 * 0.8 = 0.08
        obs2 = state.encode()
        self.assertAlmostEqual(obs2["states"][0, len(state.info_list) + 1], 0.08, places=6)
        self.assertEqual(obs2["states"][0, len(state.info_list) + 2], 2.0)

    # -------------------------------------------------------------
    # Phase 5: BaseTradingEnv Input Types
    # -------------------------------------------------------------
    def test_env_action_types_and_clipping(self):
        prices = create_dummy_prices(100)
        state = self.get_state()
        env = ProductionEnv({"SYM": prices}, state)
        env.reset()

        # Float scalar
        obs, r, d, info = env.step(0.5)
        self.assertAlmostEqual(info["position"], 0.5)

        # 1D Numpy array
        obs, r, d, info = env.step(np.array([0.8], dtype=np.float32))
        self.assertAlmostEqual(info["position"], 0.8)

        # Torch Tensor
        obs, r, d, info = env.step(torch.tensor([-0.6]))
        self.assertAlmostEqual(info["position"], -0.6)

        # Action exceeding bounds -> should be clipped to 1.0
        obs, r, d, info = env.step(2.5)
        self.assertAlmostEqual(info["position"], 1.0)

        obs, r, d, info = env.step(-5.0)
        self.assertAlmostEqual(info["position"], -1.0)

    # -------------------------------------------------------------
    # Phase 2: Short to Long Flip
    # -------------------------------------------------------------
    def test_short_to_long_flip(self):
        # Step 1: short -0.8 at 100
        # Step 2: flip to long +0.6 at 90
        # Realized gain on short = 0.8 * (100 - 90)/100 = 0.08
        # Turnover = | 0.6 - (-0.8) | = 1.4 -> friction = 1.4 * total_friction
        # New long opened at 90, trade_bar = 1
        close_series = [100.0] * 10 + [100.0, 90.0] + [100.0] * 50
        prices = create_dummy_prices(close_prices=close_series)
        state = self.get_state()
        state.reset(prices, offset=10)

        state.step(-0.8)
        state.step(0.6)

        self.assertEqual(state.position, 0.6)
        self.assertAlmostEqual(state.closecash, 0.08, places=6)
        self.assertEqual(state.open_price, 90.0)
        self.assertEqual(state.trade_bar, 1)
        expected_cost = (0.8 + 1.4) * self.total_friction
        self.assertAlmostEqual(state.cost_sum, expected_cost, places=6)

    # -------------------------------------------------------------
    # Phase 2: Episode Termination Conditions
    # -------------------------------------------------------------
    def test_episode_termination_on_n_steps(self):
        # N_steps = 15, prices length = 100
        n_steps = 15
        state = State_time_step(
            bars_count=self.bars_count,
            commission_perc=self.commission,
            model_train=True,
            default_slippage=self.slippage,
            N_steps=n_steps,
        )
        prices = create_dummy_prices(100)
        state.reset(prices, offset=10)

        for step_i in range(1, n_steps + 1):
            r, done = state.step(0.5)
            if step_i < n_steps:
                self.assertFalse(done, f"Step {step_i} should not be done")
            else:
                self.assertTrue(done, f"Step {step_i} should be done")

    def test_episode_termination_on_data_end(self):
        # Data has 25 bars, offset starts at 20, N_steps = 50 (larger than data)
        # Should terminate when offset reaches len - 1 (offset = 24)
        prices = create_dummy_prices(25)
        state = State_time_step(
            bars_count=self.bars_count,
            commission_perc=self.commission,
            model_train=False,  # Test non-train mode data end
            default_slippage=self.slippage,
            N_steps=100,
        )
        state.reset(prices, offset=20)
        done = False
        steps_taken = 0
        while not done:
            _, done = state.step(0.1)
            steps_taken += 1
            if steps_taken > 30:
                self.fail("Infinite loop detected: env did not terminate at data end")
        self.assertTrue(done)
        self.assertGreaterEqual(state._offset, 24)

    # -------------------------------------------------------------
    # Phase 3: Downside Penalty Dynamics (Only Positive Delta Penalized)
    # -------------------------------------------------------------
    def test_downside_penalty_dynamics(self):
        state = self.get_state()
        state.downside_window_size = 5

        # Initial: no history -> penalty = 0
        self.assertEqual(state.calculate_step_downside_penalty(), 0.0)

        # Append small loss
        state.return_history.append(-0.01)
        state.return_history.append(-0.02)
        # Downside risk increases from 0 -> penalty > 0
        p1 = state.calculate_step_downside_penalty()
        self.assertGreater(p1, 0.0)
        self.assertEqual(state.prev_downside_risk, state.calculate_downside_risk_numpy(list(state.return_history)))

        # Append profit -> downside risk should not increase, penalty should be 0.0
        state.return_history.append(0.05)
        state.return_history.append(0.05)
        p2 = state.calculate_step_downside_penalty()
        self.assertEqual(p2, 0.0)

    # -------------------------------------------------------------
    # Phase 6: RolloutBuffer Integration
    # -------------------------------------------------------------
    def test_rollout_buffer_compatibility(self):
        from Brain.PPO2.lib.experience import RolloutBuffer
        prices = create_dummy_prices(100)
        state = self.get_state()
        env = ProductionEnv({"SYM": prices}, state)
        obs = env.reset()

        buffer = RolloutBuffer(gamma=0.99, lam=0.95)
        for _ in range(5):
            action = np.array([0.5], dtype=np.float32)
            next_obs, reward, done, info = env.step(action)
            value = torch.tensor([0.1])
            logp = torch.tensor([-0.5])
            buffer.store(obs, action, logp, reward, next_obs, done, value)
            obs = next_obs
            if done:
                break

        self.assertEqual(len(buffer.buffer), 5)
        # Compute GAE
        transitions = buffer.compute_gae(last_value=0.0)
        self.assertEqual(len(transitions), 5)
        for tr in transitions:
            self.assertIn("states", tr.state)
            self.assertIn("time_states", tr.state)
            self.assertIsInstance(tr.reward, float)

    # -------------------------------------------------------------
    # Quality & Robustness Tests
    # -------------------------------------------------------------
    def test_add_position_no_phantom_loss(self):
        # Initial buy 0.4 at 100 -> price rises to 120 (gain = +0.08)
        # Add 0.4 at 120 (price remains 120) -> equity must reflect true gain of +0.08 minus costs, NO phantom drop!
        close_series = [100.0] * 10 + [100.0, 120.0, 120.0] + [100.0] * 50
        prices = create_dummy_prices(close_prices=close_series)
        state = self.get_state()
        state.reset(prices, offset=10)

        # Step 1: Open 0.4 at 100
        state.step(0.4)
        cost1 = 0.4 * self.total_friction
        self.assertAlmostEqual(state.TotalPortfolioPercent, 1.0 - cost1, places=6)

        # Step 2: Price moves to 120, agent holds 0.4
        state.step(0.4)
        expected_gain = 0.4 * (120.0 - 100.0) / 100.0  # +0.08
        self.assertAlmostEqual(
            state.TotalPortfolioPercent, 1.0 - cost1 + expected_gain, places=6
        )

        # Step 3: Price is STILL 120, agent adds 0.4 -> position becomes 0.8
        state.step(0.8)
        cost2 = 0.4 * self.total_friction
        # Price did not move between step 2 and step 3! Equity should only drop by cost2!
        expected_equity = 1.0 - (cost1 + cost2) + expected_gain
        self.assertAlmostEqual(state.TotalPortfolioPercent, expected_equity, places=6)

    def test_production_env_reset_symbol(self):
        prices1 = create_dummy_prices(50)
        prices2 = create_dummy_prices(50)
        state = self.get_state()
        env = ProductionEnv({"SYM1": prices1, "SYM2": prices2}, state)

        # Default reset uses SYM1
        env.reset()
        self.assertEqual(env._instrument, "SYM1")

        # Explicit reset with SYM2
        env.reset(symbol="SYM2")
        self.assertEqual(env._instrument, "SYM2")

    def test_multidim_action_inputs(self):
        prices = create_dummy_prices(50)
        state = self.get_state()
        env = ProductionEnv({"SYM": prices}, state)
        env.reset()

        # 2D torch tensor
        obs, r, d, info = env.step(torch.tensor([[0.3]]))
        self.assertAlmostEqual(info["position"], 0.3)

        # 2D numpy array
        obs, r, d, info = env.step(np.array([[-0.7]]))
        self.assertAlmostEqual(info["position"], -0.7)

    def test_downside_queue_length_safety(self):
        # Even if N_steps is small (e.g. 15), return_history maxlen should be at least downside_window_size (60)
        state = State_time_step(
            bars_count=10,
            commission_perc=0.001,
            model_train=True,
            default_slippage=0.0005,
            N_steps=15,
            downside_window_size=60,
        )
        self.assertGreaterEqual(state.return_history.maxlen, 60)


if __name__ == "__main__":
    unittest.main()


