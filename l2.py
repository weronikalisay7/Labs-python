from math import sqrt


def distance(x1, y1, x2, y2): #расстояние между точками
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def power(a, n):#возведение в степень
    res = 1
    for i in range(abs(n)):
        res *= a
    if n >= 0:
        return res
    else:
        return 1 / res


def capitalize(word): #замена маленькой буквы на прописную
    first_letter_small = word[0]
    first_letter_big = chr(ord(first_letter_small) - ord('a') + ord('A'))
    return first_letter_big + word[1:]


def power(a, n): #возведение в степень(2)
    if n == 0:
        return 1
    else:
        return a * power(a, n - 1)


def reverse(): #последовательность в обратном порядке с рекурсией
    x = int(input())
    if x != 0:
        reverse()
    print(x)


def fib(n): #возврат n-го числа Фибоначчи по заданому n
    if n == 1 or n == 2:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)


print("Задача 1:")
x1 = float(input())
x2 = float(input())
y1 = float(input())
y2 = float(input())
print("Расстояние между точками: ", distance(x1, x2, y1, y2))
print("--------------------------")

print("Задача 2:")
print("Число a в степени n равно: ", power(float(input()), int(input())))
print("--------------------------")


print("Задача 3:")
source = input().split()
res = []
for word in source:
    res.append(capitalize(word))
print(' '.join(res))
print("--------------------------")


print("Задача 4:")
t = power(float(input()), int(input()))
print("Число a в степени n равно: ", t)
print("--------------------------")


print("Задача 5:")
reverse()
print("--------------------------")


print("Задача 6:")
print(fib(int(input())))
print("--------------------------")

