#show.

#program index_state(start).

#show stateID(S, N+start) : state(S), N = #count{T : state(T), T < S}.

#program process_sample.

#show state(M) : stateID(S, M).
#show transition(M, N) : stateID(S, M), stateID(T, N), transition(S, T).
#show relevant(M, N) : stateID(S, M), stateID(T, N), relevant(S, T).
#show goal(N) : stateID(S, N), goal(S).

#program index_feature(start).

#show featureID(F, N+start) : feature(F), N = #count{G : feature(G), F < G}.

#program process_feature.

#show feature(FID) : featureID(F, FID), feature(F).
#show cost(FID, N) : cost(F, N), featureID(F, FID).
#show qualValue(FID, V, SID) : qualValue(F, V, S), featureID(F, FID), stateID(S, SID).
#show delta(SID, TID, FID, V) : delta(S, T, F, V),
    featureID(F, FID), stateID(S, SID), stateID(T, TID).