%prune_exp
%prune_new
%prune_pool
%prune_candidate

#program prune_exp.

cardinality(C, N, S) :- exp(C, _), state(S), N = #count{X : belong(X, C, S)}.

compare(C1, C2) :- 
    exp(C1, _), exp(C2, _), cost(C1, N1), cost(C2, N2), N2 < N1.

compare(C1, C2) :-
    new(C1, _), new(C2, _), cost(C1, N1), cost(C2, N2), N2 = N1, C2 < C1.

#program keep_exp.

keep__(exp(C, N)) :- exp(C, N), belong(_, C, _), differ(C, D) : compare(C, D).
keep__(belong(X, C, S)) :- keep__(exp(C, _)), belong(X, C, S).
keep__(cardinality(C, N, S)) :- keep__(exp(C, _)), cardinality(C, N, S).

#program prun_exp_conc.

compare(C, D) :-
    exp(C, _), conc(D, _).

#program exp2conc.

keep__(conc(C, N)) :- exp(C, N), belong(_, C, _).
keep__(belong(X, C, S)) :- keep__(conc(C, _)), belong(X, C, S).
keep__(cardinality(C, N, S)) :- keep__(conc(C, _)), cardinality(C, N, S).

#program simplify.
keep__(const(X)) :- const(X).
keep__(state(S)) :- state(S).

#program get_roles.

keep__(role(R, N)) :- role(R, N), belong(_, R, _).
keep__(belong(XY, R, S)) :- keep__(role(R, N)), belong(XY, R, S).

%*
#program prune_exp(i).

cardinality(C, N, S) :- exp(C, i), state(S), N = #count{X : belong(X, C, S)}.
non_empty(C, i) :- exp(C, i), belong(_, C, _).

compare(C1, C2, i, "prune_exp") :-
    non_empty(C1, i), conc(C2, j), j < i.

#program prune_new(i).

new(C, i) :- non_empty(C, i), differ(C, CC) : compare(C, CC, i, "prune_exp").

compare(C1, C2, i, "prune_new") :- 
    new(C1, i), new(C2, i), cost(C1, N1), cost(C2, N2), N2 < N1.
compare(C1, C2, i, "prune_new") :- 
    new(C1, i), new(C2, i), cost(C1, N1), cost(C2, N2), N2 = N1, C2 < C1.

#program prune_pool(i).

pool(C, i) :- new(C, i),
    differ(C, Other) : compare(C, Other, i, "prune_new").

pool(C, i) :- candidate(C, i-1).

compare(C1, C2, i, "prune_pool") :-
    pool(C1, i), pool(C2, i), cost(C1, N1), cost(C2, N2), N2 < N1.
compare(C1, C2, i, "prune_pool") :- 
    pool(C1, i), pool(C2, i), cost(C1, N1), cost(C2, N2), N2 = N1, C2 < C1.

#program prune_candidate(i).

candidate(C, i) :- pool(C, i),
    differ(C, Other) : compare(C, Other, i, "prune_pool").

conc(C, i) :- candidate(C, i), cost(C, i).

#program relevant(i).
keep__(conc(C, N)) :- conc(C, N).
keep__(candidate(C, i)) :- candidate(C, i).
keep__(role(R)) :- role(R).
keep__(cost(C, N)) :- conc(C, _), cost(C, N).
keep__(cost(C, N)) :- candidate(C, i), cost(C, N).
keep__(cost(R, N)) :- role(R), cost(R, N).
keep__(belong(X, C, S)) :- belong(X, C, S), conc(C, _).
keep__(belong(X, C, S)) :- belong(X, C, S), candidate(C, i).
keep__(belong(X, R, S)) :- belong(X, R, S), role(R).
keep__(cardinality(C, N, S)) :- cardinality(C, N, S), candidate(C, i).
keep__(cardinality(C, N, S)) :- cardinality(C, N, S), conc(C, _).*%