from sympy import symbols, diff

J, w = symbols('J, w')
J = w**2
dJ_dw = diff(J, w)

print(f"derivatives for {J} is {dJ_dw}, examples:")
print("at 2 ->", dJ_dw.subs([(w, 2)]))
print("at 3 ->", dJ_dw.subs([(w, 3)]))
print("at -3 ->", dJ_dw.subs([(w, -3)]))
