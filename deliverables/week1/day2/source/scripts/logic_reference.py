from itertools import product


solutions = []
for a, b, c in product((False, True), repeat=3):
    a_statement = not b
    b_statement = a == c
    c_statement = a
    if a == a_statement and b == b_statement and c == c_statement:
        solutions.append((a, b, c))

assert solutions == [(False, True, False)], solutions
print("A=liar B=truth C=liar")
print("PASS unique solution")
