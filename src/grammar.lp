%Define primitive, universal, and nominal concepts:
#const k=8.
conc_A(prim(P)) :- pred(P), arity(P, 1).
conc_A(univ).
conc_A(nomi(X)) :- const(X).


%Define "basic" concepts and their negation:

_conc(C, 1) :- conc_A(C).
_conc(neg(C), 2) :- conc_A(C).


%Define conjunction:

conc(and(C1, C2), N1 + N2) :- _conc(C1, N1), _conc(C2, N2), N1 + N2 <= k, C1 < C2.
conc(and(C1, and(C2, C3)), N1 + N2) :- _conc(C1, N1), conc(and(C2, C3), N2), N1 + N2 <= k, C1 < C2.

conc(X, N) :- _conc(X, N).

#show conc/2.