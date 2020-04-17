%Define primitive, universal, and nominal concepts:
#const k=6.
conc_A(prim(P)) :- pred(P), arity(P, 1).
conc_A(univ).
conc_A(nomi(X)) :- const(X).

tot_ord(C, C) :- conc_A(C). 
%Define "basic" concepts and their negation:

_conc(C, 1) :- conc_A(C).
_conc(neg(C), 2) :- conc_A(C).

tot_ord(neg(C), C) :- conc_A(C).
%Define conjunction:

before(C1, C2):- tot_ord(C1, N1), tot_ord(C2, N2), N1 < N2.

conc(and(C1, C2), N1 + N2 + 1) :- _conc(C1, N1), _conc(C2, N2), N1 + N2 + 1<= k, before(C1, C2).
conc(and(C1, and(C2, C3)), N1 + N2 + 1) :- _conc(C1, N1), conc(and(C2, C3), N2), N1 + N2 + 1 <= k, before(C1, C2).

tot_ord(and(C1, C2), N) :- conc(and(C1, C2), _), tot_ord(C1, N).

conc(X, N) :- _conc(X, N).

%Define roles

role((prim(P); inv(prim(P)); plus(prim(P)); plus(inv(prim(P)))), 1) :- pred(P), arity(P, 2).

%Define role equivalence, existential, and universal role:concepts

_conc(eq(R1, R2), N1 + N2 + 1) :- role(R1, N1), role(R2, N2), N1 + N2 + 1 <= k.
_conc(ex(R, C), N1 + N2 + 1) :- role(R, N1), conc(C, N2), N1 + N2 + 1 <= k.
_conc(un(R, C), N1 + N2 + 1) :- role(R, N1), conc(C, N2), N1 + N2 + 1 <= k.

tot_ord(C, C) :- conc(C, _), C = (eq(_, _); ex(_,_); un(_,_)).

#show conc/2.