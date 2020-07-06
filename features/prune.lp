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

%%%%%%%%%%%% Advanced Prunning %%%%%%%%%%%%

%Split expressions in sets of maximum size max_exp
#program split_exp_conc(max_exp, max_conc).
set(C, X) :- exp(C, _),
    Before = #count{D : exp(D, _), D < C},
    X = Before/max_exp.
atom_groups((Count + max_exp - 1)/max_exp) :- Count = #count{C : exp(C, _)}.

set(C, ExpG + X) :- conc(C, _), atom_groups(ExpG),
    Before = #count{D : conc(D, _), D < C},
    X = Before/max_conc.
comp_groups((Count + max_conc - 1)/max_conc) :- Count = #count{C : conc(C, _)}.
%Prune redundant expressions inside set num and add output to set out
#program join_set(num_1, num_2, out).
candidate(C, out) :- set(C, num_1).
candidate(C, out) :- set(C, num_2).
compare(C, D, out) :-
    set(C, num_1), set(D, num_2), D < C.
compare(C, D, out) :-
    set(C, num_2), set(D, num_1), D < C.

%Prune redundant expressions inside set num_exp wrt set num_conc and add output to set out
#program difference_set(num_in, num_other, out).

candidate(C, out) :- set(C, num_in).
compare(C, D, out) :-
    set(C, num_in), set(D, num_other).

#program prune_set(num).

set(C, num) :- candidate(C, num), differ(C, D) : compare(C, D, num).

%Show symbols of expressions in set num
#program show_exp_set(num).
#show exp(C, N) : exp(C, N), set(C, num).
#show belong(X, C, S) : belong(X, C, S), exp(C, _), set(C, num).
#show cardinality(C, N, S) : cardinality(C, N, S), exp(C, _), set(C, num).
%%%%%%%%%%%% Features %%%%%%%%%%%% 

#program compare_prefeature.

compare(F1, F2) :- prefeature(F1), prefeature(F2), F2 < F1.

#program compare_feature.

compare(F1, F2) :- prefeature(F1), feature(F2).

#program prune_feature.%TODO add hasValue

keep(prefeature(F)) :- prefeature(F), differ(F, G) : compare(F, G).

#show bool(F) : bool(F), keep(prefeature(F)).
#show num(F) : num(F), keep(prefeature(F)).
#show prefeature(F) : keep(prefeature(F)).
#show cost(F, N) : cost(F, N),keep(prefeature(F)).
#show qualValue(F, V, S) : qualValue(F, V, S),keep(prefeature(F)).
#show delta(S, T, F, V) : delta(S, T, F, V),keep(prefeature(F)).

#program number_feat(start).

featureId(F, N + start) :- prefeature(F), N = #count{G : G < F, prefeature(G)}.
feature(I) :- featureId(F, I), prefeature(F).
bool(I) :- featureId(F, I), bool(F).
num(I) :- featureId(F, I), num(F).
cost(I, N) :- cost(F, N), featureId(F, I).
qualValue(I, V, S) :- qualValue(F, V, S), featureId(F, I).
delta(S, T, I, V) :- delta(S, T, F, V), featureId(F, I).

count(N) :- N = #count{F : feature(F)}.

#show featureId(F, I) : featureId(F, I).
#show feature(F) : feature(F).
#show bool(F) : bool(F), feature(F).
#show num(F) : num(F), feature(F).
#show cost(F, C) : cost(F, C), feature(F).
#show qualValue(F, V, S) : qualValue(F, V, S), feature(F).
#show delta(S, T, F, V) : delta(S, T, F, V), feature(F).

#program divide_feat(start, first, gsize).%TODO can this be deleted?
%*
featureId(F, N + start) :- prefeature(F), N = #count{G : G < F, prefeature(G)}.
feature(I) :- featureId(F, I), prefeature(F).
bool(I) : featureId(F, I), bool(F).
num(I) : featureId(F, I), num(F).
cost(I, N) : cost(F, N), featureId(F, I).
qualValue(I, V, S) : qualValue(F, V, S), featureId(F, I).
delta(S, T, I, V) : delta(S, T, F, V), featureId(F, I).
*%
count(N) :- N = #count{F : feature(F)}.
min(M) :- M = #min{F : feature(F)}.
groups(0) :- count(0).
groups(1) :- count(N), N < first, N > 0.
groups(1 + (N-first+gsize-1)/gsize) :- count(N), N >= first.

group(F, 0) :- feature(F), min(M), F - M < first.
group(F, (F - M - first) / gsize + 1) :- feature(F), min(M), F - M >= first.

#program classify_feat.%TODO add hasValue
#external show(X) : X=0..N-1, groups(N).
#show featureId(F, I) : featureId(F, I), group(I, N), show(N).
#show feature(F) : feature(F), group(F, N), show(N).
#show bool(F) : bool(F), group(F, N), show(N).
#show num(F) : num(F), group(F, N), show(N).
#show cost(F, C) : cost(F, C), group(F, N), show(N).
#show qualValue(F, V, S) : qualValue(F, V, S), group(F, N), show(N).
#show delta(S, T, F, V) : delta(S, T, F, V), group(F, N), show(N).