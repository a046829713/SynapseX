class Example:
    def __init__(self, data):
        self.data = data

    def __getstate__(self):
        state = self.__dict__.copy()
        state['data'] = state['data'] * 2  # 修改狀態
        return state

    def __setstate__(self, state):
        print("測試進入")
        state['data'] = state['data'] // 2  # 恢復狀態時還原修改
        self.__dict__.update(state)



app = Example(data=[1,2,3,4])
app = list(app)
print(app)