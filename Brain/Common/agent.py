import torch
import torch.nn.functional as F
import numpy as np
import time


class BaseAgent:
    """
    Abstract Agent interface
    """

    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError
    
class DQNAgent(BaseAgent):
    """
        DQNAgent is a memoryless DQN agent which calculates Q values
        from the observations and  converts them into the actions using action_selector
    """

    def __init__(self, dqn_model, action_selector, device="cpu"):
        """
            初始化方法接受以下參數：
            dqn_model: 一個模型，用於預測給定狀態下每個可能行動的價值。
            action_selector: 根據dqn_model的輸出選擇行動的策略。
            device: 用於運算的設備（例如，"cpu"或"cuda"）。
            preprocessor: 用於預處理狀態的函數。默認使用上面的default_states_preprocessor。

        Args:
            dqn_model (_type_): _description_
            action_selector (_type_): _description_
            device (str, optional): _description_. Defaults to "cpu".
            preprocessor (_type_, optional): _description_. Defaults to default_states_preprocessor.
        """

        self.net = dqn_model
        self.action_selector = action_selector            
        self.device = device

    def preprocessor(self, state, time_state):
        """
            Convert list of states into the form suitable for model. By default we assume Variable
            :param states: list of numpy arrays with states
            :return: Variable
        """
        states_v = torch.from_numpy(np.array(state, dtype=np.float32))
        states_v = states_v.to(self.device)

        time_states_v = torch.from_numpy(np.array(time_state, dtype=np.float32))
        time_states_v = time_states_v.to(self.device)


        return states_v, time_states_v
    
    def states_preprocessor(self, states):
        """
            Convert list of states into the form suitable for model. By default we assume Variable
            :param states: list of numpy arrays with states
            :return: Variable
        """
        out_put_states_v = []
        out_put_time_states_v = []
        
        for s in states:
            state, time_state = s

            states_v, time_states_v = self.preprocessor(state, time_state)
            out_put_states_v.append(states_v)
            out_put_time_states_v.append(time_states_v)
        

        return torch.stack(out_put_states_v), torch.stack(out_put_time_states_v)

    def __call__(self, states):
        states_v_s, time_states_v_s = self.states_preprocessor(states)
       
        with torch.no_grad():
            q_values = self.net(states_v_s, time_states_v_s)


        q = q_values.data.cpu().numpy()
        actions = self.action_selector(q)
        return actions