"""Numbers for 연구 설계 정리 d 절: Berkson (collider) bias when only hospitalized people are studied.
Two conditions A and B independent in the population (10% each, n = 10,000). Run: python3 gen/rv03_d.py"""
def berkson(pA_only, pB_only, pAB, pNone, n=10000, a=0.10, b=0.10):
    AB, Ao, Bo, N = n*a*b, n*a*(1-b), n*(1-a)*b, n*(1-a)*(1-b)
    h = dict(AB=AB*pAB, Ao=Ao*pA_only, Bo=Bo*pB_only, N=N*pNone)
    OR = (h["AB"]*h["N"]) / (h["Ao"]*h["Bo"])
    print(f"population OR {(AB*N)/(Ao*Bo):.2f}; admitted AB {h['AB']:.0f} A-only {h['Ao']:.0f} B-only {h['Bo']:.0f} neither {h['N']:.0f} -> OR among admitted {OR:.3f}")
berkson(1.0, 1.0, 1.0, 0.05)   # body example
berkson(0.5, 0.5, 1.0, 0.02)   # practice example
