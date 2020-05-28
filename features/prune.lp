#program reduce(i).

new(C, i) :- exp(C, i), belong(_, C, _), differ(C, CC) : conc(CC, j), j < i.
new(C, i) :- candidate(C, i-1).
%candidate(C, i) :- candidate(C, i-1), not conc(C, i-1).

before(C1, C2) :- new(C1, i), new(C2, i), cost(C1, N1), cost(C2, N2), N1 < N2.
before(C1, C2) :- new(C1, i), new(C2, i), cost(C1, N1), cost(C2, N2), N1 = N2, C1 < C2.

%there does not exist a D s.t. equiv(C, D) and before(D, C)
redundant(C, i) :- new(C, i), new(D, i), equiv(C, D), before(D, C).
candidate(C, i) :- new(C, i), not redundant(C, i).

compare(C1, C2) :- exp(C1, i), conc(C2, j), j < i.
compare(C1, C2) :- new(C1, i), new(C2, i).

cardinality(C, N, S) :- exp(C, i), state(S), N = #count{X : belong(X, C, S)}.
equiv(C1, C2) :- compare(C1, C2), not differ(C1, C2).

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
keep__(cardinality(C, N, S)) :- cardinality(C, N, S), conc(C, _).