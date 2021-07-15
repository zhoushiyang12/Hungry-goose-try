a = [1,2,3,4]
b = [1,2,5]
for j in b:
    # for i in a :
        
    print("b",b)
    if j in a:
        b.remove(j)   ####oh my god!!
        print("b",b)


c = set(b).difference(set(a))
print('c',c)