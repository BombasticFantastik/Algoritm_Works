def square_quest():
    print("Введите a")
    a=int(input())
    print("Введите b")
    b=int(input())
    print("Введите c")
    c=int(input())
    D=(b**2)-(4*a*c)

    if D<0:
        print('Решение уравнения нет')

    elif a==0:
        x1=-c/b
        print(f'Решение уравнения x имеет корень {x1}')

    else:
        x1=(-b+(D**0.5))/(2*a)
        
        if D==0:
            print(f'Уравнение имеет один корень: {x1}')
        else:
            x2=(-b-(D**0.5))/(2*a)
            print(f' Уравнение имеет два корня: X1:{x1} X2:{x2}')
square_quest()