%prune_exp
%prune_new
%prune_pool
%prune_candidate
#defined keep__/1.
#defined conc/2.
#defined exp/2.
#defined belong/3.
#defined cardinality/3.

#program prune_exp.

cardinality(C, N, S) :- exp(C, _), state(S), N = #count{X : belong(X, C, S)}.

compare(C1, C2) :- 
    exp(C1, _), exp(C2, _), C2 < C1.

#program keep_exp.

keep__(exp(C, N)) :- exp(C, N), belong(_, C, _), differ(C, D) : compare(C, D).
keep__(belong(X, C, S)) :- keep__(exp(C, _)), belong(X, C, S).
keep__(cardinality(C, N, S)) :- keep__(exp(C, _)), cardinality(C, N, S).

#program prune_exp_conc.

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