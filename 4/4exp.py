from random import randint
def turing_machine_delete_neg_after_min():
    A=[randint(-10,5) for i in range(15)]
    tape = ['_'] + A + ['_']
    n = len(A)
    min_val = tape[1]
    min_pos = 1
    for i in range(2, n + 1):
        if tape[i] < min_val:
            min_val = tape[i]
            min_pos = i

    result = []
    state = 'q0'  
    head = 1
    step = 0

    while head <= n:
        step += 1
        current = tape[head]
        pos = head
        if state == 'q0':
            if pos == min_pos:
                result.append(current)
                state = 'q1'
            else:
                result.append(current)
                
        elif state == 'q1':  
            if current < 0:
                
                state = 'q2'
            else:
                result.append(current)
            
                state = 'q3'
        elif state == 'q2':
            if current < 0:
                pass
            else:
                result.append(current)
                #print(f"Шаг {step}: позиция {pos}, элемент {current} — конец блока, добавлен")
                state = 'q3'

        elif state == 'q3':  
            result.append(current)
            #print(f"Шаг {step}: позиция {pos}, элемент {current} — после блока, добавлен")
        head += 1

    print("-" * 60)
    print("\n" + "="*70)
    print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
    print(f"Исходный массив: {A}")
    print(f"Конечный результат: {result}")
    print(f"Минимальный элемент: {min_val}")
    print(f"Всего выполнено шагов: {step}")
    #return result

#-8, -7, -6, 3, 2, -7
turing_machine_delete_neg_after_min()