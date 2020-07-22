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
#show.
#defined relevant/2.
#defined holdsState/3.

%%%% Perform the encoding %%%%

#program predicates.
arity(V, 1) :- variable(variable((V, _))).
arity(V, 2) :- variable(variable((V, _, _))).
pred(P) :- arity(P, _).

#show pred(P) : pred(P).
#show arity(P, N) : arity(P, N).

#program const.
#show const(X) : constant(constant(X)).

#program hold.

hold(State, Pred) :- 
	holds(variable(Pred), value(variable(Pred), true), State), arity(Pred, 0).
hold(State, Pred, Const) :-
	holds(variable((Pred,constant(Const))),value(variable((Pred,constant(Const))),true),State).
hold(State, Pred, (Const1, Const2)) :-
	holds(variable((Pred,constant(Const1), constant(Const2))),value(variable((Pred,constant(Const1), constant(Const2))),true),State).

%#show hold(S, P) : hold(S, P).
%#show hold(S, P, X) : hold(S, P, X).

%%%% Safety constraints %%%%

:- goal(S), not state(S).
:- hold(State, Pred), not pred(Pred).
:- hold(State, Pred), not state(State).
:- hold(State, Pred, Const), not pred(Pred).
:- hold(State, Pred, Const), not state(State).
:- hold(State, Pred, Const), arity(Pred, 1), not const(Const).
pair(X) :- X = (A,B), const(A), const(B). 
:- hold(State, Pred, Const), arity(Pred, 2), not pair(Const).

#program get_encoding(max_depth, optimal).

show(State) :- state(State, Depth), Depth <= max_depth.
show(State) :- state(State), 1 {relevant(State, _); relevant(_, State)}, optimal = 1.

#show state(State) : show(State).
#show state(State, Depth) : state(State, Depth), show(State).
#show stateId(State, Id) : stateId(State, Id), show(State).
#show hold(State, Pred) : hold(State, Pred), show(State).
#show hold(State, Pred, Const) : hold(State, Pred, Const), show(State).
#show transition(St1, St2) : transition(St1, St2), show(St1).
#show goal(Goal) : goal(Goal), show (Goal).