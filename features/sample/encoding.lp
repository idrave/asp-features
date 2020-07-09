%*
Convert to:
pred(P):            P is a predicate
arity(P, N)         The arity of P is N (integer)
const(X)            X is a constant (a grounded object)
state(S)            S is a state
hold(S, P, X)       Predicate P holds for tuple X in state S
transition(S1, S2)     S1 and S2 is a valid transition from the sample
relevant(S1, S2)       S1 and S2 is a transition marked as relevant
goal(S)                S is a goal state
*%

#program predicates.
arity(V, 1) :- variable(variable((V, _))).
arity(V, 2) :- variable(variable((V, _, _))).
pred(P) :- arity(P, _).
%TODO: include types as predicates

#program null_predicate(p).
arity(p, 0).

%%%% Perform the encoding %%%%

#program encode.
const(X) :- constant(constant(X)).
state(S) :- state(S, _).
goal(S) :- goalState(S).

hold(State, Pred) :- 
	holdsState(variable(Pred), value(variable(Pred), true), State), arity(Pred, 0).
hold(State, Pred, Const) :-
	holdsState(variable((Pred,constant(Const))),value(variable((Pred,constant(Const))),true),State).
hold(State, Pred, (Const1, Const2)) :-
	holdsState(variable((Pred,constant(Const1), constant(Const2))),value(variable((Pred,constant(Const1), constant(Const2))),true),State).

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

%%%% Safety constraints %%%%

:- goal(S), not state(S).
:- hold(State, Pred), not pred(Pred).
:- hold(State, Pred), not state(State).
:- hold(State, Pred, Const), not pred(Pred).
:- hold(State, Pred, Const), not state(State).
:- hold(State, Pred, Const), arity(Pred, 1), not const(Const).
pair(X) :- X = (A,B), const(A), const(B). 
:- hold(State, Pred, Const), arity(Pred, 2), not pair(Const).

%%%% Show results %%%%

#program print.
%#show arity/2.
%#show const/1.
%#show pred/1.
%#show state/1.
%#show goal/1.
%#show hold/3.
%#show hold/2.
%#show pair/1.
#show optimalGoal/1.
#show optimalCost/1.
#show relevant/2.