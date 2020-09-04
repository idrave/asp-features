#show.
#program relevant.
%Select one optimal plan as relevant transitions

optimalCost(Cost) :- Cost = #min{X : goal(S), state(S, X)}.

optimalGoal(Goal) :- state(Goal, Depth), optimalCost(Depth), goal(Goal),
	Goal <= X : state(X, Depth), goal(X).

relevant(State, Goal) :- optimalGoal(Goal), state(Goal, Depth), 
	transition(State, Goal), state(State, Depth-1),
	State <= X : transition(X, Goal), state(X, Depth-1).

relevant(State1, State2) :- state(State2, Depth), relevant(State2, _), 
	transition(State1, State2), state(State1, Depth-1),
	State1 <= X : transition(X, State2), state(X, Depth-1).

#show relevant(S1, S2) : relevant(S1, S2).

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