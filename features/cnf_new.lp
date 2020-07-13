
different(F, (S1, S2), (T1, T2)) :-
    good(S1, S2), transition(T1, T2), not good(T1, T2),
    feature(F), delta(S1, S2, F, A), delta(T1, T2, F, B), not A = B.

different(F, (S1, S2), (T1, T2)) :-
    good(S1, S2), transition(T1, T2), not good(T1, T2),
    feature(F), delta(S1, S2, F, null), delta(T1, T2, F, null),
    qualValue(F, V, S1), qualValue(F, W, T1), not V = W.


1 {selected(F) :
    feature(F), different(F, (S1, S2), (T1, T2))} :-
    good(S1, S2), transition(T1, T2), not good(T1, T2).

#minimize {N, F : selected(F), cost(F, N)}.
#show selected/1.