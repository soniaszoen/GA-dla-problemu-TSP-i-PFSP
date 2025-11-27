import copy
a = [10, 20, 30]
b = a.copy()  # tworzy nową listę, ale z tymi samymi elementami (referencje)

print("a:", a)  # a: [10, 20, 30]
print("b:", b)  # b: [10, 20, 30]

b[0] = 100  # modyfikujesz tylko listę b
print("b po zmianie:", b)  # b po zmianie: [100, 20, 30]

a=[[1,2], [3,4]]
b=a.copy()  # płytka kopia listy a
b[0][0] = 100  # modyfikujesz element wewnętr
print("a po zmianie:", a)  # a po zmianie: [[100, 2], [3, 4]]
print("b po zmianie:", b)  # b po zmianie: [[100, 2], [3, 4]]


x = [1, 2]
b[0] = x          # używasz ISTNIEJĄCEGO obiektu, nie tworzysz nowego
b[0] = a[1]       # to jest referencja do elementu z listy a

print("a:", a)  # a: [10, 20, 30]
print("b:", b)  # b: [20, 20, 30]