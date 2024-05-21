n = 5

print("\n Lower Triangular Pattern \n")
for i in range(1, n + 1):
    print('* ' * i)
print("\n")


print("\n Upper Triangular Pattern \n")
for i in range(n, 0, -1):
    print('* ' * i)
print("\n")

print("\n Pyramid Pattern \n")
for i in range(1, n + 1):
    print(' ' * (n - i) + '* ' * i)
print("\n")

