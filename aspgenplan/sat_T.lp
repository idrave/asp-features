%bool(F):               F is a boolean feature
%num(F):                F is a numeric feature
%feature(F):            F is a feature (either bool or num)
%transition(S1, S2)     S1 and S2 is a valid transition from the sample
%relevant(S1, S2)       S1 and S2 is a transition marked as relevant
%selected(F)            F is a selected feature
%goal(S)                S is a goal state
%hasValue(F, V, S)      Feature F has value V in state S
%cost(F, N)             Feature F has cost N


%*
#program sat.*%


d1(S, T) :- feature(F), selected(F), state(S), state(T),
            qualValue(F, A, S), qualValue(F, B, T), not A = B.

1 {selected(F) :    feature(F), qualValue(F, A, S),
                    qualValue(F, B, T), not A = B} :- d1(S, T), state(S), state(T).

% Rule (2). Define d2

d2(S1, S2, T1, T2) :-   feature(F), selected(F), transition(S1, S2), transition(T1, T2),
                        delta(S1, S2, F, A), delta(T1, T2, F, B), not A = B.

1 {selected(F) :    feature(F), delta(S1, S2, F, A), 
                    delta(T1, T2, F, B), not A = B} :- d2(S1, S2, T1, T2), transition(S1, S2), transition(T1, T2).

% Rule (3). Enforce soundness and completeness

1 {-d2(S1, S2, T1, T2) : transition(T1, T2)} :- not d1(S1, T1), transition(S1, S2), state(T1), transition(T1, _).

% Rule (4). Differentiate goal from non-goals
%d1(S, T) :- relevant(S), state(T), {goal(S;T)} = 1.
d1(S, T) :- state(S), state(T), goal(S), not goal(T).
d1(S, T) :- state(S), state(T), goal(T), not goal(S).

% Find optimal model minimizing sum of feature costs
#minimize {N, F : selected(F), cost(F, N)}.
#show selected/1.
%#show d1/2.