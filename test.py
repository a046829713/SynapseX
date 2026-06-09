def test():
    i = 0
    while True:
        i+=1
        yield i **2


genration = test()


for i in range(5):
    a =  next(genration)
    print(a)        


# import dis

# def test():
#     i = 0
#     while True:
#         i += 1
#         yield i ** 2




