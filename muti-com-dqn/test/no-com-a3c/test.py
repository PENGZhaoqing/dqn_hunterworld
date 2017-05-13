class Test:
    def __init__(self, id):
        self.id = id

a = []
for i in range(10000):
    a.append(Test(i))

for i in range(1000):
    for j, item in enumerate(a):
        if j != item.id:
            print "haha"
