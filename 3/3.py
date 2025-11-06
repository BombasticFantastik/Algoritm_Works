import random
A=[random.randint(-10,10) for i in range(10)]
print("Начальный массив:",A)
A=A[::-1]
while True:
    last_mas=A.copy()
    ln=len(A)-1
    for i in range(0,ln):
        if A[i]!=abs(A[i]) and A[i+1]==min(A):
            A.remove(A[i])
            break
    if last_mas!=A:
        last_mas=A.copy()
    else:
        break
A=A[::-1]
print("Массив после операций:",A)