print("Задание 1(количество четных и нечетных чисел в списке: ")
import random

a = []
for i in range(10):
    a.append(int(random.random() * 100))

print(a)

even = 0
odd = 0

for i in a:
    if i%2 == 0:
        even += 1
    else:
        odd += 1

print("Нечетных:", even)#количество нечетных чисел
print("Четных:", odd)#количество четных чисел
print("--------------------------------")
print("Задание 2(разделить элементы списка на положительные и отритцательные): ")
b = []
for i in range(20):
    b.append(int(random.random() * 20) - 10)
print(b)
neg = []
pos = []
for i in b:
    if i < 0:
        neg.append(i)
    elif i > 0:
        pos.append(i)
print(neg)#список отритцательных чисел
print(pos)#писок положительных чисел
print("--------------------------------")
print("Задание 3(замена элементов списка): ")
listOrigin = [10, -15, 3, 8, 0, 9, -6, 13, -1, 5]#заполнение нового списка в зависимости от исходного
listMask = []
for item in listOrigin:
    if item > 0:
        listMask.append(1)
    elif item < 0:
        listMask.append(-1)
    else:
        listMask.append(0)

print(listOrigin)
print(listMask)

listOne = [10, -15, 3, 8, 0, 9, -6, 13, -1, 5]#замена элементов непосредственно в исходном списке
print(listOne)

for i in range(len(listOne)):
    if listOne[i] > 0:
        listOne[i] = 1
    elif listOne[i] < 0:
        listOne[i] = -1

print(listOne)
print("--------------------------------")
print("Задание 4(преобразование текста в список слов с удалением знаков препинания): ")
str = input("Write down or insert some text:\n")
punctuation = ['.',',',':',';','!','?','(',')']
wordList = str.split()
i = 0
for word in wordList:
    if word[-1] in punctuation:
        wordList[i] = word[:-1]
        word = wordList[i]
    if word[0] in punctuation:
        wordList[i] = word[1:]
    i += 1
i = 0
while i < len(wordList):
    print(wordList[i], end=' ')
    i += 1
    if i%5 == 0:
        print()
print("--------------------------------")
print("Задание 5(строка и столбец с максимальными суммами элементов): ")
from random import random
matrix = []
for i in range(5):
    row = []
    for j in range(5):
        row.append(int(random()*10))
    matrix.append(row)
for row in matrix:
    print(row)
maxRow = 0
idRow = 0
i = 0
for row in matrix:
    if sum(row) > maxRow:
        maxRow = sum(row)
        idRow = i
    i += 1
print(idRow, '-', maxRow)
maxCol = 0
idCol = 0
for i in range(5):
    colSum = 0
    for j in range(5):
        colSum += matrix[j][i]
    if colSum > maxCol:
        maxCol = colSum
        idCol = i

print(idCol, '-', maxCol)#строка и столбец с максимальной суммой элементов
print("--------------------------------")
print("Задание 6(сумма элементов главной и побочной диагоналей): ")

N = 5
matrix = []
for i in range(N):
    row = []
    for j in range(N):
        row.append(int(random()*10))
    matrix.append(row)
for row in matrix:
    print(row)
sumMain = 0
sumSecondary = 0
for i in range(N):
    sumMain += matrix[i][i]
    sumSecondary += matrix[i][N-i-1]
print(sumMain)#сумма элементов главной диагонали
print(sumSecondary)#сумма элементов побочной диагонали
print("--------------------------------")
print("Задание 7(в каким строках и столбцах содержиться элемент): ")
from random import random
N = 5
M = 10
matrix = []
for i in range(N):
    row = []
    for j in range(M):
        row.append(int(random()*40)+10)
    matrix.append(row)
for row in matrix:
    print(row)
item = int(input("Number range: "))
print("Rows (index):", end=" ")
for i in range(N):
    if item in matrix[i]:
        print(i, end=" ")
print()
print("Columns (index):", end=" ")
for j in range(M):
    for i in range(N):
        if matrix[i][j] == item:
            print(j, end=" ")
            break
print()
print("--------------------------------")
print("Задание 8(найти значение списка, которое встречается чаще всего): ")
from random import random
mass = [int(random()*5) for i in range(15)]
print(a)
massSet = set(a)#список в множество
Comm = None#часто встречаемое значение
qtyComm = 0#количество
for item in massSet:
    qty = mass.count(item)
    if qty > qtyComm:
        qtyComm = qty
        Comm= item

print(Comm)
print("--------------------------------")
