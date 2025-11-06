import random

A=[random.randint(-5,5) for i in range(10)]
while min(A)!=abs(min(A)):
    for i in range(len(A)-1):
        #print(A[i+1]==min(A))
        if A[i]!=abs(A[i]) and A[i+1]==min(A):
            print(1)
            A=A.remove(A[i])
print(A)