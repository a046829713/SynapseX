"""
    Agent is something which converts states into actions and has state
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
import time



class TargetNet:
    """
        複製目標網絡
    """

    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        """
            將主網絡的參數複製過來
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


# class PolicyAgent(BaseAgent):
#     """
#     Policy agent gets action probabilities from the model and samples actions from it
#     """
#     # TODO: unify code with DQNAgent, as only action selector is differs.

#     def __init__(self, model, action_selector=actions.ProbabilityActionSelector(), device="cpu",
#                  apply_softmax=False, preprocessor=default_states_preprocessor):
#         self.model = model
#         self.action_selector = action_selector
#         self.device = device
#         self.apply_softmax = apply_softmax
#         self.preprocessor = preprocessor

#     def __call__(self, states, agent_states=None):
#         """
#         Return actions from given list of states
#         :param states: list of states
#         :return: list of actions
#         """
#         if agent_states is None:
#             agent_states = [None] * len(states)
#         if self.preprocessor is not None:
#             states = self.preprocessor(states)
#             if torch.is_tensor(states):
#                 states = states.to(self.device)
#         probs_v = self.model(states)
#         if self.apply_softmax:
#             probs_v = F.softmax(probs_v, dim=1)
#         probs = probs_v.data.cpu().numpy()
#         actions = self.action_selector(probs)
#         return np.array(actions), agent_states


# class ActorCriticAgent(BaseAgent):
#     """
#     Policy agent which returns policy and value tensors from observations. Value are stored in agent's state
#     and could be reused for rollouts calculations by ExperienceSource.
#     """

#     def __init__(self, model, action_selector=actions.ProbabilityActionSelector(), device="cpu",
#                  apply_softmax=False, preprocessor=default_states_preprocessor):
#         self.model = model
#         self.action_selector = action_selector
#         self.device = device
#         self.apply_softmax = apply_softmax
#         self.preprocessor = preprocessor

#     def __call__(self, states, agent_states=None):
#         """
#         Return actions from given list of states
#         :param states: list of states
#         :return: list of actions
#         """
#         if self.preprocessor is not None:
#             states = self.preprocessor(states)
#             if torch.is_tensor(states):
#                 states = states.to(self.device)
#         probs_v, values_v = self.model(states)
#         if self.apply_softmax:
#             probs_v = F.softmax(probs_v, dim=1)
#         probs = probs_v.data.cpu().numpy()
#         actions = self.action_selector(probs)
#         agent_states = values_v.data.squeeze().cpu().numpy().tolist()
#         return np.array(actions), agent_states
