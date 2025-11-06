from random import randint

print('')

n = 5
a =[-10,-9,-8,-10,-2]

print('Исходный массив:')
print(*a)

el_min = a[0]
i_min = 0

for i in range(1, n):
  if a[i] < el_min:
    el_min = a[i]
    i_min = i

print(f'Минимальный элемент: {el_min} по индексу {i_min}')

def delete(a, i):
  for i in range(i, n - 1):
    a[i] = a[i + 1]

i = 0
while a[i] != el_min:
  if a[i] < 0:
    delete(a, i)
  else:
    i += 1

print('Результат обработки:')
print(*a)