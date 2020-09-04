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
#defined dist_const/6.
#defined valid/1.
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
%*
#show bool(F) : bool(F).
#show num(F) : num(F).
#show prefeature(F) : prefeature(F).
#show cost(F, N) : cost(F, N).
#show qualValue(F, V, S) : qualValue(F, V, S).
#show delta(S, T, F, V) : delta(S, T, F, V).
*%
1 = {show(Feat) : prefeature(Feat)}.

#show bool(F) : bool(F), show(F).
#show num(F) : num(F), show(F).
#show prefeature(F) : prefeature(F), show(F).
#show cost(F, N) : cost(F, N), show(F).
#show qualValue(F, V, S) : qualValue(F, V, S), show(F).
#show delta(S, T, F, V) : delta(S, T, F, V), show(F).

#program to_feature(id).
:- 1 != #count{Feat : prefeature(Feat)}.
#show featureId(F, id) : prefeature(F).
#show feature(id).
#show bool(id) : bool(F), feature(F).
#show num(id) : num(F), feature(F).
#show cost(id, C) : cost(F, C), prefeature(F).
#show qualValue(id, V, S) : qualValue(F, V, S), prefeature(F).
#show delta(S, T, id, V) : delta(S, T, F, V), prefeature(F).

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

#program dist_pos(pos).
#show pos(Conc, pos) : conc(Conc, _).

#program distFeature(k).
valid(dist(C1, R, C, C2)) :-
    conc(C1, M), role(R, N), conc(C, O), conc(C2, P),
    pos(C1, 0), pos(C, 1), pos(C2, 2), M+N+O+P <= k.

#program dist_const(dist).

dist_const(Const, Role, Conc, Conc2, 0, State) :-
    const(Const), valid(dist(_, Role, Conc, Conc2)), state(State), dist = 0,
    belong(Const, Conc2, State), pos(C, 0), belong(Const, C, State).

dist_const(Const, Role, Conc, Conc2, dist, State) :- dist > 0,
    not dist_const(Const, Role, Conc, Conc2, Dist, State) : Dist = 0..dist-1;
    const(Const), valid(dist(_, Role, Conc, Conc2)), state(State),
    const(Other), belong((Const, Other), Role, State),
    belong(Other, Conc, State), dist_const(Other, Role, Conc, Conc2, dist-1, State).
#show continue(dist) : dist_const(_,_,_,_,dist,_).

#program value_dist.

num(dist(C1, R, C, C2)) :-
    valid(dist(C1, R, C, C2)), const(Const), state(State),
    belong(Const, C1, State), dist_const(Const, R, C, C2, _, State).

hasValue(dist(C1, R, C, C2), Dist, S) :-
    num(dist(C1, R, C, C2)), state(S),
    Dist = #min{D : const(Const), belong(Const, C1, S), dist_const(Const, R, C, C2, D, S)}.

cost(dist(C1, R, C, C2), A) :-
    num(dist(C1, R, C, C2)), conc(C1, M), role(R, N), conc(C, O), conc(C2, P),
    pos(C1, 0), pos(C, 1), pos(C2, 2), A=M+N+O+P.
