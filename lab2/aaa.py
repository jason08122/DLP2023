f = open("test.txt","w")

tmp = []
for i in range(0, 10000):
    tmp.append(int((i*(i+1))/2))

print(tmp, file = f)

f.close()