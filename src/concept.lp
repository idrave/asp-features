#const k=5.

%Rule for primitive concepts from unary predicates
belong(X, prim(P), S) :- const(X), pred(P), arity(P, 1), state(S), hold(S, P, X).
cost(prim(P), 1) :- belong(_, prim(P), _).

%Rule for universal concept
belong(X, univ, S) :- const(X), state(S).
cost(univ, 1).

%Rule for nominals
belong(X, nomi(X), S) :- const(X), state(S).
cost(nomi(X), 1) :- belong(_, nomi(X), _).

cbasic(C) :- C = (prim(_); univ; nomi(_)), cost(C, _).
valid(C) :- cbasic(C).
conc(C) :- cbasic(C), not redundant(C).

%Negation

belong(X, neg(C), S) :- const(X), cbasic(C), state(S), not belong(X, C, S).
cost(neg(C), 2) :- valid(neg(C)).
valid(neg(C)) :- cbasic(C), belong(_, neg(C), _).
conc(neg(C)) :- valid(neg(C)), not redundant(neg(C)).

%Rule for conjunction
belong(X, and(C1, C2), S) :- const(X), valid_and(C1, C2), state(S), belong(X, C1, S), belong(X, C2, S).

valid(and(C1, C2)) :- belong(_, and(C1, C2), _).
conc(and(C1, C2)) :- valid(and(C1, C2)), not redundant(and(C1, C2)).

is_and(and(C1, C2)) :- conc(and(C1, C2)).

valid_and(C1, C2) :- conc(C1), cost(C1, N1), conc(C2), cost(C2, N2), N1 + N2 < k, not is_and(C1),
                    not is_and(C2), C1 < C2.

valid_and(C1, and(C2, C3)) :- conc(C1), cost(C1, N1), conc(and(C2, C3)), cost(and(C2, C3), N2),
                    N1 + N2 < k, not is_and(C1), C1 < C2.


cost(and(C1, C2), N1 + N2 + 1) :- valid(and(C1, C2)), cost(C1, N1), cost(C2, N2).

%Primitive roles and their inverse

belong((X, Y), r_prim(P), S) :- const(X), const(Y), pred(P), arity(P, 2), state(S), hold(S, P, (X, Y)).
belong((X, Y), inv(r_prim(P)), S) :- const(X), const(Y), pred(P), arity(P, 2), state(S), hold(S, P, (Y, X)).

%Define basic roles as primitive roles and their inverse
r_basic(r_prim(P)) :- belong(_, r_prim(P), _).
r_basic(inv(R)) :- belong(_, inv(R), _).
role(R) :- r_basic(R).

%Transitive roles based on basic roles
belong(XY, tran(R), S) :- r_basic(R), belong(XY, R, S).
belong((X, Y), tran(R), S) :- r_basic(R), belong((X, Z), R, S), belong((Z, Y), tran(R), S).
role(tran(R)) :- belong(_, tran(R), _).

%Cost of basic and trnsitive roles is set to 1
cost(R, 1) :- r_basic(R).
cost(tran(R), 1) :- belong(_, tran(R), _).

%Define existential and universal roles

%role_conc: equivalent to R:C notation
belong((X, Y), role_conc(R, C), S) :- role(R), conc(C), belong((X, Y), R, S), belong(Y, C, S).
belong((X, Y), neg(role_conc(R, C)), S) :- const(X), const(Y), role(R), conc(C), state(S),
                                        not belong((X, Y), role_conc(R, C), S).

belong(X, neg(uni(R, C)), S) :- conc(C), belong((X, Y), neg(role_conc(R, C)), S), cost(R, NR), cost(C, NC), NR + NC < k.
belong(X, uni(R, C), S) :- const(X), role(R), conc(C), state(S), not belong(X, neg(uni(R, C)), S),
                            cost(R, NR), cost(C, NC), NR + NC < k.

conc(uni(R, C)) :- valid(uni(R, C)), not redundant(uni(R, C)).
cost(uni(R, C), NR + NC + 1) :- valid(uni(R, C)), cost(R, NR), cost(C, NC).
valid(uni(R, C)) :- role(R), conc(C), belong(_, uni(R, C), _).

belong(X, exi(R, C), S) :- belong((X, Y), role_conc(R, C), S), cost(R, NR), cost(C, NC), NR + NC < k.

conc(exi(R, C)) :- valid(exi(R, C)), not redundant(exi(R, C)).
cost(exi(R, C), NR + NC + 1) :- valid(exi(R, C)), cost(R, NR), cost(C, NC).
valid(exi(R, C)) :- role(R), conc(C), belong(_, exi(R, C), _).

% Guarantee non-redundancy in the set of concepts

differ(C1, C2) :- valid(C1), valid(C2), const(X), state(S), belong(X, C1, S), not belong(X, C2, S).
differ(C1, C2) :- valid(C1), valid(C2), const(X), state(S), not belong(X, C1, S), belong(X, C2, S).

equiv(C1, C2) :- valid(C1), valid(C2), not differ(C1, C2).
before(C1, C2) :- cost(C1, N1), cost(C2, N2), N1 < N2.
before(C1, C2) :- cost(C1, N1), cost(C2, N2), N1 = N2, C1 < C2.

redundant(C1) :- conc(C2), equiv(C1, C2), before(C2, C1).

%#show cost/2.

%debug(X) :- belong(_, X, _), not role(X).
%debug2(X, N) :- cost(X, N), not role(X).
%#show const/1.
#show conc/1.
%#show belong/3.