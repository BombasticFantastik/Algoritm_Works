def turing_machine_multiple_states():
    A = list(map(int, input("Введите элементы массива через пробел: ").split()))
    
    tape = ['_'] + A + ['_']
    head = 1
    state = 'q0'  # q0 - обработка элементов, q1 - завершение работы
    
    result = []
    positive_count = 0
    temp_storage = []
    
    step = 0
    
    print("Реализация с несколькими состояниями (как в вашем примере):")
    print(f"Исходный массив: {A}")
    print()
    
    while state != 'HALT':
        step += 1
        current = tape[head]
        
        print(f"Шаг {step}: состояние={state}, голова={head}, ячейка={current}")
        print(f"Результат: {result}, Временные: {temp_storage}")
        print(f"Счетчик положительных: {positive_count}")
        
        if state == 'q0':
            if current == '_':
                # Достигли конца ленты - переходим в состояние завершения
                print("→ Достигнут конец ленты, переходим в состояние q1")
                state = 'q1'
                head = 1  # сбрасываем головку для следующей фазы
            elif isinstance(current, int):
                if current > 0:
                    positive_count += 1
                    if positive_count == 1:
                        print(f"→ Найден первый положительный: {current}, добавляем в результат")
                        result.append(current)
                    else:
                        print(f"→ Найден положительный №{positive_count}: {current}, сохраняем во временное хранилище")
                        temp_storage.append(current)
                    head += 1
                else:
                    print(f"→ Отрицательный/нулевой элемент: {current}, добавляем в результат")
                    result.append(current)
                    head += 1
            print("→ Переход к следующей ячейке")
        
        elif state == 'q1':
            # Фаза завершения - добавляем временные элементы в хвост
            if temp_storage:
                print(f"→ Добавляем временные элементы в хвост: {temp_storage}")
                result.extend(temp_storage)
                temp_storage = []
            state = 'HALT'
            print("→ Завершаем работу")
        
        print("-" * 60)
    
    print("\n" + "="*70)
    print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
    print(f"Исходный массив: {A}")
    print(f"Конечный результат: {result}")
    print(f"Всего выполнено шагов: {step}")
    
    return result
# Тестирование
if __name__ == "__main__":
    turing_machine_multiple_states()
