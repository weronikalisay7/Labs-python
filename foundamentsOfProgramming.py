print("Excercise 1:")
print("Input number from 1 to 9:")
x=int(input("num="))
if x==1 or x==2 or x==3:
    print("Enter string: ")
    s=input("Your string: ")
    print("Enter the number of repetitions: ")
    n=int(input("Nember of repetitions:"))
    i=0
    while i<n:
        print(s)
elif x==4 or x==5 or x==6:
    print("Enter the power of the number:")
    m=int(input("m="))
    print(x**m)
elif x==7 or x==8 or x==9:
    i=0
    while i<10:
        ch=x+1
        print(ch)
        i+=1
else:
    print("ERROR!")
print("---------------------------------------")

print("Excersice 2:")
print("Общество в начале XXI века")
print("Введите ваш возраст:")
v=int(input("Ваш возраст:"))
if 0<=v<7:
    print("Вам в детсткий сад")
elif 7<=v<18:
    print("Вам в школу")
elif 18<=v<25:
    print("Вам в профессиональное учебное заведение")
elif 25<=v<60:
    print("Вам на работу")
elif 60<=v<120:
    print("Вам предоставляется выбор")
else:
    i=0
    while i<5:
        print("Ошибка!Это программа для людей!")
        i+=1
print("---------------------------------------")


print("Excercise 3:")
xf=int(input("Введите число от 1 до 100:"))
fact=1
i=0
while i<xf:
    i += 1
    fact=fact*i
print(fact)
print("---------------------------------------")

print("Excercise 4:")
f1=0
f2=1
i=2
while i<100:
    f_sum=f1+f2
    f1=f2
    f2=f_sum
    i+=1
    print(f2)
print("---------------------------------------")


print("Excercise 5:")
import math
N=int(input("Введите номер числа Фибоначчи:"))
fi=(1+math.sqrt(5))/2
Fn=(fi**N)/(math.sqrt(5))
print(Fn)


