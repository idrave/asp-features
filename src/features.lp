#program features.
%general boolean rule:
hasValue(F, false, S) :- bool(F), state(S), not hasValue(F, true, S).

%nullary primitive predicate features:
bool(pred(P)) :- pred(P), arity(P, 0).
cost(pred(P), 0) :- bool(pred(P)).

hasValue(pred(P), true, S) :- bool(pred(P)), state(S), hold(S, P).

cardinality(conc(C), N, S) :- conc(C, _), state(S), N = #count{X : belong(X, C, S)}.

num(conc(C)) :- cardinality(conc(C), N, S), N > 1, conc(C, M).
bool(conc(C)) :- conc(C, _), not num(conc(C), M).

cost(conc(C), N) :- conc(C, N), feature(conc(C)).

hasValue(conc(C), true, S) :- bool(conc(C)), const(X), state(S), belong(X, C, S).
hasValue(conc(C), N, S) :- num(conc(C)), cardinality(conc(C), N, S).

num(dist(C1, R, C, C2)) :- conc(C1, M), role(R), cost(R, N), conc(C, O), conc(C2, P),
                            M+N+O+P <= k.

cost(dist(C1, R, C, C2), A) :- num(dist(C1, R, C, C2)), conc(C1, M), role(R),
                            cost(R, N), conc(C, O), conc(C2, P), A=M+N+O+P.

hasValue(dist(C1, R, C, C2), 2, S) :- num(dist(C1, R, C, C2)), state(S), const(X), const(Y),
                        belong(X, C1, S), belong((X, Y), R, S), belong(Y, C, S), belong(Y, C2, S).

hasValue(dist(C1, R, C, C2), N, S) :- num(dist(C1, R, C, C2)), state(S), not hasValue(dist(C1, R, C, C2), 2, S),
                        N-1 = #min{M : const(X), const(Y), belong(X, C1, S), belong((X, Y), R, S),
                        belong(Y, C, S), hasValue(dist(nomi(Y), R, C, C2), M, S)}.

feature(F) :- bool(F);num(F).