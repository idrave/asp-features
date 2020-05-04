%bool(F):               F is a boolean feature
%num(F):                F is a numeric feature
%feature(F):            F is a feature (either bool or num)
%transition(S1, S2)     S1 and S2 is a valid transition from the sample
%relevant(S1, S2)       S1 and S2 is a transition marked as relevant
%selected(F)            F is a selected feature
%goal(S)                S is a goal state
%hasValue(F, V, S)      Feature F has value V in state S
%cost(F, N)             Feature F has cost N

%A state is relevant if it is part of a relevant transition
relevant(S) :- relevant(S, _).
relevant(S) :- relevant(_, S).

%Determine qualitative value of features over states
qualValue(F, V, S) :- bool(F), hasValue(F, V, S).
qualValue(F, eq0, S) :- num(F), hasValue(F, 0, S).
qualValue(F, gr0, S) :- num(F), hasValue(F, N, S), N > 0.

%Determine qualitative change in features over state transitions
delta(S1, S2, F, plus) :- transition(S1, S2), bool(F), hasValue(F, false, S1), hasValue(F, true, S2).
delta(S1, S2, F, minus) :- transition(S1, S2), bool(F), hasValue(F, true, S1), hasValue(F, false, S2).
delta(S1, S2, F, inc) :- transition(S1, S2), num(F), hasValue(F, M, S1), hasValue(F, N, S2), M < N.
delta(S1, S2, F, dec) :- transition(S1, S2), num(F), hasValue(F, M, S1), hasValue(F, N, S2), M > N.
delta(S1, S2, F, null) :- transition(S1, S2), feature(F), hasValue(F, M, S1), hasValue(F, M, S2).

% Rule (1). Define d1
d1(S, T) :- feature(F), selected(F), relevant(S), state(T),
            qualValue(F, A, S), qualValue(F, B, T), not A = B.

1 {selected(F) :    feature(F), qualValue(F, A, S),
                    qualValue(F, B, T), not A = B} :- d1(S, T), relevant(S), state(T).

% Rule (2). Define d2
d2(S1, S2, T1, T2) :-   feature(F), selected(F), relevant(S1, S2), transition(T1, T2),
                        delta(S1, S2, F, A), delta(T1, T2, F, B), not A = B.

% Rule (3). Enforce soundness and completeness
1 {selected(F) :    feature(F), delta(S1, S2, F, A), 
                    delta(T1, T2, F, B), not A = B} :- d2(S1, S2, T1, T2), relevant(S1, S2), transition(T1, T2).

1 {-d2(S1, S2, T1, T2) : transition(T1, T2)} :- not d1(S1, T1), relevant(S1, S2), state(T1).

% Rule (4). Differentiate goal from non-goals
d1(S, T) :- relevant(S), state(T), {goal(S;T)} = 1.

% Find optimal model minimizing sum of feature costs
#minimize {N, F : selected(F), cost(F, N)}.