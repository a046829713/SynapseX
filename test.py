from collections import deque


a = deque(maxlen=5)


a.append(1)
a.append(2)

for i in a:
    print(i)