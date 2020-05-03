%transition(S1, S2)
%relevant(S)
%relevant(S1, S2)
%delta(S1, S2, F, D) ?
%selected(F)
%d1(S, T)
%d2(S1, S2, T1, T2)
%goal(S)
%qualValue(F, V, S)

relevant(S) :- relevant(S, _).
relevant(S) :- relevant(_, S).

qualValue(F, V, S) :- bool(F), hasValue(F, V, S).
qualValue(F, eq0, S) :- num(F), hasValue(F, 0, S).
qualValue(F, gr0, S) :- num(F), hasValue(F, N, S), N > 0.

delta(S1, S2, F, plus) :- transition(S1, S2), bool(F), hasValue(F, false, S1), hasValue(F, true, S2).
delta(S1, S2, F, minus) :- transition(S1, S2), bool(F), hasValue(F, true, S1), hasValue(F, false, S2).
delta(S1, S2, F, inc) :- transition(S1, S2), num(F), hasValue(F, M, S1), hasValue(F, N, S2), M < N.
delta(S1, S2, F, dec) :- transition(S1, S2), num(F), hasValue(F, M, S1), hasValue(F, N, S2), M > N.
delta(S1, S2, F, null) :- transition(S1, S2), feature(F), hasValue(F, M, S1), hasValue(F, M, S2).

%Rule (1)
d1(S, T) :- feature(F), selected(F), relevant(S), state(T),
            qualValue(F, A, S), qualValue(F, B, T), not A = B.

1 {selected(F) : feature(F), qualValue(F, A, S),
                qualValue(F, B, T), not A = B} :- d1(S, T), relevant(S), state(T).

%Rule (2)
d2(S1, S2, T1, T2) :- feature(F), selected(F), relevant(S1, S2), transition(T1, T2),
                        delta(S1, S2, F, A), delta(T1, T2, F, B), not A = B.

%Rule (3)
1 {selected(F) : feature(F), delta(S1, S2, F, A), 
                delta(T1, T2, F, B), not A = B} :- d2(S1, S2, T1, T2), relevant(S1, S2), transition(T1, T2).

1 {-d2(S1, S2, T1, T2) : transition(T1, T2)} :- not d1(S1, T1), relevant(S1, S2), state(T1).

% Rule (4)
d1(S, T) :- relevant(S), state(T), {goal(S;T)} = 1. 