#show.
#program dist_goal(dist).

dist_goal(State, 0) :- dist = 0, goal(State).

dist_goal(State, dist) :-
    state(State), not dist_goal(State, Dist) : Dist = 0..dist-1;
    transition(State, T), dist_goal(T, dist-1). %not dead-end
#show continue(dist) : dist_goal(State, dist).

%*
dist_goal(State, #sup) :-
    state(State),
    #sup = #min{N : dist_goal(T, N), transition(State, T)}. %dead-end
*%
#program mark_all.

good(St1, St2) :-
    transition(St1, St2), dist_goal(St1, Dist),
    dist_goal(St2, Dist - 1).


#program mark_some.

1 {good(St1, St2) : 
    transition(St1, St2), dist_goal(St2, Dist - 1)} :- state(St1), dist_goal(St1, Dist).