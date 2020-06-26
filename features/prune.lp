#defined keep__/1.
#defined conc/2.
#defined exp/2.
#defined belong/3.
#defined cardinality/3.
#defined keep__/2.
#defined keep/1.
#defined role/2.
#defined ord/2.
#defined prev/2.
#defined group/2.
#defined count/1.
#defined const/1.
#defined state/1.
#defined model/1.
#defined qualValue/3.
#defined delta/4.
#defined cost/2.
#defined num/1.
#defined bool/1.
#defined prefeature/1.
#defined feature/1.
#defined hold/2.
#defined hold/3.
#defined stateId/2.
#defined goal/1.
#defined relevant/2.
#show.

#program index_state(start).
stateId(S, N+start) :- state(S), N = #count{T : state(T), T < S}.
#show stateId(S, ID) : stateId(S, ID).

#show state(M) : stateId(S, M).
#show transition(M, N) : stateId(S, M), stateId(T, N), transition(S, T).
#show relevant(M, N) : stateId(S, M), stateId(T, N), relevant(S, T).
#show goal(N) : stateId(S, N), goal(S).
#show hold(M, P, X) : hold(S, P, X), stateId(S, M).
#show hold(M, P) : hold(S, P), stateId(S, M).
#show arity(P, N) : arity(P, N).
#show const(X): const(X).
#show pred(P) : pred(P).

#program index_role(start).
roleId(Role, Id) :- role(Role, _), Id = #count{R : role(R, _), R < Role}.
#show roleId(Role, Id) : roleId(Role, Id).
#show role(Id, Cost) : role(Role, Cost), roleId(Role, Id).
#show belong(Const, Id, State) : role(Role, _), roleId(Role, Id), belong(Const, Role, State).

#program simplify.

#show const(C) : const(C).
#show state(S) : state(S).

#program transitions.

#show transition(S, T) : transition(S, T).

#program get_roles.

#show role(R, N) : role(R, N).
#show belong(XY, R, S) : role(R, N), belong(XY, R, S).

%%%% Concepts %%%%%

#program compare_exp.

compare(C1, C2) :- 
    exp(C1, _), exp(C2, _), C2 < C1.

#program keep_exp.

cardinality(C, N, S) :- exp(C, _), state(S), N = #count{X : belong(X, C, S)}.

keep(exp(C, N)) :- exp(C, N), belong(_, C, _).
#show exp(C, N) : keep(exp(C, N)).
#show belong(X, C, S) : keep(exp(C, _)), belong(X, C, S).
#show cardinality(C, N, S) : keep(exp(C, _)), cardinality(C, N, S).

#program prune_exp.


keep(exp(C, N)) :- exp(C, N), belong(_, C, _), differ(C, D) : compare(C, D).
#show exp(C, N) : keep(exp(C, N)).
#show belong(X, C, S) : keep(exp(C, _)), belong(X, C, S).
#show cardinality(C, N, S) : keep(exp(C, _)), cardinality(C, N, S).

#program compare_exp_conc.

compare(C, D) :-
    exp(C, _), conc(D, _).

#program number_conc(start, first, gsize).

conceptId(C, N + start) :- exp(C, _), N = #count{D : D < C, exp(D, _)}.
conc(I, N) :- conceptId(C, I), exp(C, N).
belong(X, I, S) :- conceptId(C, I), belong(X, C, S).
cardinality(I, N, S) :- conceptId(C, I), cardinality(C, N, S).

count(N) :- N = #count{C : conc(C, _)}.
min(M) :- M = #min{C : conc(C,_)}.
groups(0) :- count(0).
groups(1) :- count(N), N < first, N > 0.
groups(1 + (N-first+gsize-1)/gsize) :- count(N), N >= first.

group(C, 0) :- conc(C, _), min(M), C - M < first.
group(C, (C - M - first) / gsize + 1) :- conc(C, _), min(M), C - M >= first.

#program classify.
#external show(X) : X=0..N-1, groups(N).
#show conceptId(C, I) : conceptId(C, I), group(I, N), show(N).
#show conc(C, M) : conc(C, M), group(C, N), show(N).
#show belong(X, C, S) : belong(X, C, S), group(C, N), show(N).
#show cardinality(C, M, S) : cardinality(C, M, S), group(C, N), show(N).

#program enumerate_exp(start, gsize).

count(N) :- N = #count{C : exp(C, _)}.
models(0) :- count(0).
models(1) :- count(N), N < start, N > 0.
models(1+(N-start+gsize-1)/gsize) :- count(N), N >= start.
ord(C, N) :- exp(C, _), N = #count{D : D < C, exp(D, _)}.

group(C, 1) :- ord(C, N), N < start.
group(C, (N-start)/gsize+2) :- ord(C, N), N >= start.
#show models(N) : models(N).

#program classify_exp.
#external model(X) : X=0..N, models(N).

#show count(N) : model(0), count(N).
keep(exp(C, M)) :- model(N), group(C, N), exp(C, M).
#show exp(C, M) : keep(exp(C, M)).
#show belong(X, C, S) : keep(exp(C, _)), belong(X, C, S).
#show cardinality(C, N, S) : keep(exp(C, _)), cardinality(C, N, S).

%%%%%%%%%%%% Features %%%%%%%%%%%% 

#program compare_prefeature.

compare(F1, F2) :- prefeature(F1), prefeature(F2), F1 < F2.

#program compare_feature.

compare(F1, F2) :- prefeature(F1), feature(F2).

#program prune_feature.

keep(prefeature(F)) :- prefeature(F), differ(F, G) : compare(F, G).

#show bool(F) : bool(F), keep(prefeature(F)).
#show num(F) : num(F), keep(prefeature(F)).
#show prefeature(F) : keep(prefeature(F)).
#show cost(F, N) : cost(F, N),keep(prefeature(F)).
#show qualValue(F, V, S) : qualValue(F, V, S),keep(prefeature(F)).
#show delta(S, T, F, V) : delta(S, T, F, V),keep(prefeature(F)).

#program to_feature.

#show feature(F) : prefeature(F).
#show bool(F) : bool(F), prefeature(F).
#show num(F) : num(F), prefeature(F).
#show cost(F, N) : cost(F, N),prefeature(F).
#show qualValue(F, V, S) : qualValue(F, V, S),prefeature(F).
#show delta(S, T, F, V) : delta(S, T, F, V),prefeature(F).

#program enumerate_feat(start, gsize).

count(N) :- N = #count{F : feature(F)}.
models(0) :- count(0).
models(1) :- count(N), N < start, N > 0.
models(1+(N-start+gsize-1)/gsize) :- count(N), N >= start.
ord(F, N) :- feature(F), N = #count{G : G < F, feature(G)}.

group(F, 1) :- ord(F, N), N < start.
group(F, (N-start)/gsize+2) :- ord(F, N), N >= start.
#show models(N) : models(N).

#program classify_feat.
#external model(X) : X=0..N, models(N).

#show count(N) : model(0), count(N).
keep(feature(F)) :- model(N), group(F, N), feature(F).
#show feature(F) : keep(feature(F)).
#show bool(F) : bool(F), keep(feature(F)).
#show num(F) : num(F), keep(feature(F)).
#show cost(F, N) : cost(F, N),keep(feature(F)).
#show qualValue(F, V, S) : qualValue(F, V, S),keep(feature(F)).
#show delta(S, T, F, V) : delta(S, T, F, V),keep(feature(F)).