from random import randint

def post_machine():
    #A = list(map(int, input("Введите элементы массива через пробел: ").split()))
    A=[randint(-10,5) for i in range(15)]
    
    tape = [str(x) for x in A] + ['_']
    head = 0
    command = 1 

    min_val = None
    #min_pos = -1
    #deleting_negative_run = False
    result = []

    step = 0
    max_steps = 1000

    while command != 0 and step < max_steps:
        step += 1
        current = tape[head] if head < len(tape) else '_'
        print(f"Шаг {step}: команда={command}, головка={head}, ячейка='{current}'")

        if command == 1:
            if current == '_':
                command = 2 
                head = 0
            else:
                num = int(current)
                if min_val is None or num < min_val:
                    min_val = num
                head += 1

        elif command == 2:
            if tape[head] == '_':
                command = 0  
            else:
                if int(tape[head]) == min_val:
                    #min_pos = head
                    command = 3  
                    head += 1  
                else:
                    head += 1

        elif command == 3:
            if head >= len(tape) or tape[head] == '_':
                command = 4  
            else:
                num = int(tape[head])
                if num < 0:
                    tape[head] = '0'  
                    head += 1
                else:
                    command = 4  

        elif command == 4:
            cleaned = [x for x in tape if x not in ['0', '_']]
            result = [int(x) for x in cleaned]
            print("НАЧАЛЬНЫЙ МАССИВ",A)
            print("ФИНАЛЬНЫЙ МАССИВ:", result)
            command = 0

    if step >= max_steps:
        print("Превышено максимальное число шагов!")
    return result if command == 0 else None

post_machine()