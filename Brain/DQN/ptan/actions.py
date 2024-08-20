import numpy as np
import time

class ActionSelector:
    """
    這是一個抽象基類，它要求所有衍生的子類都必須實現__call__方法。這樣，所有的行動選擇器都可以像函數一樣被調用。
    Abstract class which converts scores to the actions
    """
    def __call__(self, scores):
        raise NotImplementedError


class ArgmaxActionSelector(ActionSelector):
    """
    此選擇器選擇具有最高得分的行動。例如，如果神經網絡為每個行動輸出一個價值，這個選擇器會選擇價值最高的行動。
    Selects actions using argmax
    """
    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)


class EpsilonGreedyActionSelector(ActionSelector):
    """這是強化學習中非常流行的一種策略，被稱為ε-greedy策略。
    大部分時間（1-ε 的機率），它會選擇具有最高得分的行動（即貪心選擇）。
    但是有ε的機率，它會隨機選擇一個行動（這有助於探索未知的狀態和行動）。
    這樣的策略允許代理在探索和利用之間進行權衡。

    Args:
        ActionSelector (_type_): _description_
    """
    def __init__(self, epsilon=0.05, selector=None):
        self.epsilon = epsilon
        self.selector = selector if selector is not None else ArgmaxActionSelector()

    # def __call__(self, scores, marketpositions):
    #     assert isinstance(scores, np.ndarray)
    #     batch_size, n_actions = scores.shape
        
    #     q_values_modified = np.copy(scores)
    #     actions = np.empty(batch_size,dtype='int64')
        
        
    #     # 生成一次隨機數陣列
    #     mask = np.random.random(size=batch_size) < self.epsilon
        
    #     for i in range(batch_size):
    #         # 根據市場位置調整分數
    #         if marketpositions[i].item() == 1:
    #             q_values_modified[i, 1] = -np.inf
    #             valid_actions = np.array([0, 2],dtype='int64')  # 排除買入動作
    #         else:
    #             q_values_modified[i, 2] = -np.inf
    #             valid_actions = np.array([0, 1],dtype='int64')  # 排除賣出動作

    #         # ε-greedy 策略
    #         if mask[i]:        
    #             actions[i] = np.random.choice(valid_actions)
    #         else:  
    #             actions[i] = self.selector(np.expand_dims(q_values_modified[i],axis=0))

    #     return actions


    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape
        actions = self.selector(scores)        
        mask = np.random.random(size=batch_size) < self.epsilon        
        rand_actions = np.random.choice(n_actions, sum(mask))        
        actions[mask] = rand_actions
        return actions
    
class ProbabilityActionSelector(ActionSelector):
    """
    此選擇器根據給定的機率分布隨機選擇行動。這在某些策略上非常有用，特別是當行動的選擇是基於某種機率分布的時候，如在某些策略梯度方法中。
    Converts probabilities of actions into action by sampling them
    """
    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)
