%Define primitive, universal, and nominal concepts:
#const k=4.
conc_A(prim(P)) :- pred(P), arity(P, 1).
conc_A(univ).
conc_A(nomi(X)) :- const(X).

base(C, C) :- conc_A(C). 
%Define "basic" concepts and their negation:

_conc(C, 1) :- conc_A(C).
_conc(neg(C), 2) :- conc_A(C).

base(neg(C), C) :- conc_A(C).
%Define conjunction:

before(C1, C2):- base(C1, N1), base(C2, N2), N1 < N2.

conc(and(C1, C2), N1 + N2 + 1) :- _conc(C1, N1), _conc(C2, N2), N1 + N2 + 1<= k, before(C1, C2).
conc(and(C1, and(C2, C3)), N1 + N2 + 1) :- _conc(C1, N1), conc(and(C2, C3), N2), N1 + N2 + 1 <= k, before(C1, C2).

base(and(C1, C2), N) :- conc(and(C1, C2), _), base(C1, N).

conc(X, N) :- _conc(X, N).

#show conc/2.