print("Excersice 1:")
from random import random
n = random() * 900 + 100
n = int(n)
print(n)
a = n // 100
b = (n // 10) % 10
c = n % 10
print("Sum of digits of a three-digit number:")
print(a + b + c)
print("------------------")

print("Excersice 2:")
print("Input temperature(F or C):")
t = input()
sign = t[-1]
t = int(t[0:-1])
if sign == 'C' or sign == 'c':
    t = round(t * (9/5) + 32)
    print(str(t) + 'F')
elif sign == 'F' or sign == 'f':
    t = round((t - 32) * (5/9))
    print(str(t) + 'C')
print("------------------")

print("Excersice 3:")
print("Input year:")
year = int(input())
if year % 4 != 0 or (year % 100 == 0 and year % 400 != 0):
    print("Usual year")
else:
    print("Intercalary year")
print("------------------")

print("Excersice 4:")
print("Input two integers:")
a = int(input())
b = int(input())

while a != 0 and b != 0:
    if a > b:
        a %= b
    else:
        b %= a

gcd = a + b
print(gcd)
print("------------------")

print("Excersice 5:")
from random import random
N = 20
array = []
for i in range(N):
    array.append(int(random()*100))
array.sort()
print(array)

print("Enter the number of the element you want to find:")
number = int(input())

low = 0
high = N-1
while low <= high:
    mid = (low + high) // 2
    if number < array[mid]:
        high = mid - 1
    elif number > array[mid]:
        low = mid + 1
    else:
        print("ID =", mid)
        break
else:
    print("No the number")
print("------------------")

print("Excercise 6:")
import math

print("Enter the upper and lower boundaries of the coefficient values:")
a1 = int(input('a1: '))
a2 = int(input('a2: '))
b1 = int(input('b1: '))
b2 = int(input('b2: '))
c1 = int(input('c1: '))
c2 = int(input('c2: '))

a_1 = range(a1, a2 + 1)
b_1 = range(b1, b2 + 1)
c_1 = range(c1, c2 + 1)

for i in a_1:
    if i == 0:
        continue
    for j in b_1:
        for k in c_1:
            print(i, j, k, end=' ')
            D = j * j - 4 * i * k
            if D >= 0:
                x1 = (-j - math.sqrt(D)) / (2 * i)
                x2 = (-j + math.sqrt(D)) / (2 * i)
                print('Yes', round(x1, 2), round(x2, 2))
            else:
                print('No')
print("------------------")

print("Excercise 7:")
from random import random

num = round(random() * 1000, 3)
print(num)

strNum = str(num)

maxDigit = -1

for i in range(len(strNum)):
    if strNum[i] == '.':
        continue
    elif maxDigit < int(strNum[i]):
        maxDigit = int(strNum[i])

print(maxDigit)
print("------------------")

print("Excercise 8:")
print("Input your string:")
s = input()

l = len(s)

for i in range(l//2):
    if s[i] != s[-1-i]:
        print("It's not palindrome")
        quit()

print("String '"+s+"' is a palindrome")
print("------------------")

print("Excercise 9:")
str = "Tree, box, chair, lamp, desk, cat, dog, grass, pig, box, lamp, shelf"
print("Your string: "+str)

subStrOld = input("Old substring: ")
subStrNew = input("New substring: ")
lenStrOld = len(subStrOld)

while str.find(subStrOld) > 0:
    i = str.find(subStrOld)
    str = str[:i] + subStrNew + str[i+lenStrOld:]

print(str)
print("------------------")

print("Excercise 10:")
str = "I have to go to the university now but I will come back soon."
print(str)

listWords = str.split()

idLongestWord = 0

for i in range(1,len(listWords)):
    if len(listWords[idLongestWord]) < len(listWords[i]):
        idLongestWord = i
print("The longest word:")
print(listWords[idLongestWord])
print("------------------")





