#defined cardinality/3.
#defined feature/1.
#defined const/1.
#defined state/1.
#defined cardinality/3.
#defined hasValue/3.
#defined num/1.
#defined bool/1.
#defined transition/2.
#defined cost/2.
#defined conc/2.
#defined qualValue/3.
#defined delta/4.
#show.
#program feature.

%Determine qualitative value of features over states
qualValue(F, V, S) :- bool(F), hasValue(F, V, S).
qualValue(F, eq0, S) :- num(F), hasValue(F, 0, S).
qualValue(F, gr0, S) :- num(F), hasValue(F, N, S), N > 0.

%Determine qualitative change in features over state transitions
delta(S1, S2, F, plus) :- transition(S1, S2), bool(F), hasValue(F, false, S1), hasValue(F, true, S2).
delta(S1, S2, F, minus) :- transition(S1, S2), bool(F), hasValue(F, true, S1), hasValue(F, false, S2).
delta(S1, S2, F, inc) :- transition(S1, S2), num(F), hasValue(F, M, S1), hasValue(F, N, S2), M < N.
delta(S1, S2, F, dec) :- transition(S1, S2), num(F), hasValue(F, M, S1), hasValue(F, N, S2), M > N.
delta(S1, S2, F, null) :- transition(S1, S2), prefeature(F), hasValue(F, M, S1), hasValue(F, M, S2).

prefeature(F) :- bool(F).
prefeature(F) :- num(F).


#show bool(F) : bool(F).
#show num(F) : num(F).
#show prefeature(F) : prefeature(F).
#show cost(F, N) : cost(F, N).
#show qualValue(F, V, S) : qualValue(F, V, S).
#show delta(S, T, F, V) : delta(S, T, F, V).

#program primitiveFeature.
%general boolean rule:
hasValue(F, false, S) :- bool(F), state(S), not hasValue(F, true, S).
%nullary primitive predicate features:
bool(pred(P)) :- pred(P), arity(P, 0).
cost(pred(P), 0) :- bool(pred(P)).

hasValue(pred(P), true, S) :- bool(pred(P)), state(S), hold(S, P).

#program conceptFeature.
%general boolean rule:
hasValue(F, false, S) :- bool(F), state(S), not hasValue(F, true, S).
num(conc(C)) :- cardinality(C, N, S), N > 1, conc(C, _).
bool(conc(C)) :- conc(C, _), not num(conc(C)).

cost(conc(C), N) :- conc(C, N), bool(conc(C)).
cost(conc(C), N) :- conc(C, N), num(conc(C)).

hasValue(conc(C), true, S) :- bool(conc(C)), cardinality(C, 1, S).
hasValue(conc(C), N, S) :- num(conc(C)), cardinality(C, N, S).

#program distFeature(k).
num(dist(C1, R, C, C2)) :- conc(C1, M), role(R), cost(R, N), conc(C, O), conc(C2, P),
                            M+N+O+P <= k.

cost(dist(C1, R, C, C2), A) :- num(dist(C1, R, C, C2)), conc(C1, M), role(R),
                            cost(R, N), conc(C, O), conc(C2, P), A=M+N+O+P.

hasValue(dist(C1, R, C, C2), 2, S) :- num(dist(C1, R, C, C2)), state(S), const(X), const(Y),
                        belong(X, C1, S), belong((X, Y), R, S), belong(Y, C, S), belong(Y, C2, S).

hasValue(dist(C1, R, C, C2), N, S) :- num(dist(C1, R, C, C2)), state(S), not hasValue(dist(C1, R, C, C2), 2, S),
                        N-1 = #min{M : const(X), const(Y), belong(X, C1, S), belong((X, Y), R, S),
                        belong(Y, C, S), hasValue(dist(nomi(Y), R, C, C2), M, S)}.


