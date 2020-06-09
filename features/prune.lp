%prune_exp
%prune_new
%prune_pool
%prune_candidate

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
#show.

#program compare_exp.

cardinality(C, N, S) :- exp(C, _), state(S), N = #count{X : belong(X, C, S)}.

compare(C1, C2) :- 
    exp(C1, _), exp(C2, _), C2 < C1.



#program keep_exp.

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

#program exp2conc.


keep(conc(C, N)) :- exp(C, N), belong(_, C, _).
#show conc(C, N) : keep(conc(C, N)).
#show belong(X, C, S) : keep(conc(C, _)), belong(X, C, S).
#show cardinality(C, N, S) : keep(conc(C, _)), cardinality(C, N, S).

#program simplify.

#show const(C) : const(C).
#show state(S) : state(S).

#program get_roles.

#show role(R, N) : role(R, N).
#show belong(XY, R, S) : role(R, N), belong(XY, R, S).

#program divide.

keep__(belong(X, C, S), G) :- keep__(conc(C, _), G), belong(X, C, S).
keep__(cardinality(C, N, S), G) :- keep__(conc(C, _), G), cardinality(C, N, S).

#program enumerate(start, gsize).

count(N) :- N = #count{C : conc(C, _)}.
models(0) :- count(0).
models(1) :- count(N), N < start, N > 0.
models(1+(N-start+gsize-1)/gsize) :- count(N), N >= start.
%prev(C, D) :- conc(C, _), conc(D, _), C < D, E < C : conc(E, _), E < D, E != C.
%st(start).
%ord(C, 0) :- conc(C, _), C < D : conc(D, _), C != D.
%ord(C, P+1) :- ord(D, P), prev(D, C).
ord(C, N) :- conc(C, _), N = #count{D : D < C, conc(D, _)}.

group(C, 1) :- ord(C, N), N < start.
group(C, (N-start)/gsize+2) :- ord(C, N), N >= start.
%#show count(N) : count(N).
#show models(N) : models(N).
%#show conc(C, N) : conc(C, N).

#program classify.
#external model(X) : X=0..N, models(N).

#show count(N) : model(0), count(N).
keep(conc(C, M)) :- model(N), group(C, N), conc(C, M).
#show conc(C, M) : keep(conc(C, M)).
#show belong(X, C, S) : keep(conc(C, _)), belong(X, C, S).
#show cardinality(C, N, S) : keep(conc(C, _)), cardinality(C, N, S).

