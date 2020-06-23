import math

a = 3 
b = 7 
c = 9


def angle(directly_opposite_side, remaining_side_1, remaining_side_2):
    return math.degrees(math.acos((remaining_side_2**2 - remaining_side_1**2 - directly_opposite_side**2)/(-2.0 * directly_opposite_side * remaining_side_1)))

angA = angle(a,b,c)
angB = angle(b,c,a)
angC = angle(c,a,b)

assert angA + angB + angC == 180.0

print(angA)
print(angB)
print(angC)