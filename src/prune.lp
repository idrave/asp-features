#program reduce(i).

nonempty(C) :- exp(C, i), belong(_, C, _).
-new(C, i) :- exp(C, i), not nonempty(C).
-new(C, i) :- exp(C, i), conc(D, j), j < i, equiv(C, D).
new(C, i) :- exp(C, i), not -new(C, i).
new(C, i) :- candidate(C, i-1).
%candidate(C, i) :- candidate(C, i-1), not conc(C, i-1).

before(C1, C2) :- new(C1, i), new(C2, i), cost(C1, N1), cost(C2, N2), N1 < N2.
before(C1, C2) :- new(C1, i), new(C2, i), cost(C1, N1), cost(C2, N2), N1 = N2, C1 < C2.

%there does not exist a D s.t. equiv(C, D) and before(D, C)
redundant(C, i) :- new(C, i), new(D, i), equiv(C, D), before(D, C).
candidate(C, i) :- new(C, i), not redundant(C, i).

compare(C1, C2) :- exp(C1, i), conc(C2, j), j < i.
compare(C1, C2) :- new(C1, i), new(C2, i).
%
%differ(C1, C2) :-   compare(C1, C2), const(X), state(S),
%                    belong(X, C1, S), not belong(X, C2, S).
        
%differ(C1, C2) :-   compare(C1, C2), const(X), state(S),
%                    not belong(X, C1, S), belong(X, C2, S).

cardinality(C, N, S) :- exp(C, i), state(S), N = #count{X : belong(X, C, S)}.
equiv(C1, C2) :- compare(C1, C2), not differ(C1, C2).

conc(C, i) :- candidate(C, i), cost(C, i).