#show.
#defined initialState/1.
#defined contains/3.
#defined mutexGroup/1.
#defined derivedVariable/1.
#defined precondition/4.
#defined postcondition/5.
#defined derivedPredicate/2.
#defined state/2.
#program expand(depth).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Establish initial state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

state(initialState, depth) :- depth = 0.
holds(Variable, Value, initialState) :- initialState(Variable, Value), state(initialState, depth).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute derived predicates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% conjunctive preconditions
satisfied(DerivedPredicate, type(and), S) :- derivedPredicate(DerivedPredicate, type(and)), holds(Variable, Value, S) : precondition(DerivedPredicate, type(and), Variable, Value); state(S, depth).

% disjunctive preconditions
satisfied(DerivedPredicate, type(or), S) :- precondition(DerivedPredicate, type(or), Variable, Value), holds(Variable, Value, S), state(S, depth).

holds(DerivedVariable, Value, S) :- satisfied(DerivedPredicate, Type, S), postcondition(DerivedPredicate, Type, effect(unconditional), DerivedVariable, Value), state(S, depth).

holds(derivedVariable(DerivedVariable), value(DerivedVariable, false), S) :- derivedVariable(DerivedVariable), not holds(derivedVariable(DerivedVariable), value(DerivedVariable, true), S), state(S, depth).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform actions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Apply effects

caused(Variable, Value, apply(Action, State)) :-
	validAction(Action, State, depth),
	postcondition(Action, Effect, Variable, Value),
	holds(VariablePre, ValuePre, State) : precondition(Effect, VariablePre, ValuePre).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inertia rules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

modified(Variable, State) :- caused(Variable, Value, State), expanded(State, depth).

holds(Variable, Value, apply(Action, State)) :- caused(Variable, Value, apply(Action, State)).

holds(variable(V), Value, apply(Action, State)) :- 
    holds(variable(V), Value, State), not modified(variable(V), apply(Action, State)),
    validAction(Action, State, depth).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variables and mutex groups
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check that variables have unique values
:- variable(Variable), not 1 {holds(Variable, Value, S) : contains(Variable, Value)} 1, expanded(S, depth).

% Check mutexes
:- mutexGroup(MutexGroup), not {holds(Variable, Value, S) : contains(MutexGroup, Variable, Value)} 1, expanded(S, depth).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Verify that goal is met
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

goal(State) :- expanded(State, depth), holds(Variable, Value, State) : goal(Variable, Value).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

validAction(Action, State, depth) :-
    state(State, depth-1), action(Action),
    not goal(State), %Goal states are terminal, no further actions can be performed
    holds(Variable, Value, State) : precondition(Action, Variable, Value).

expanded(apply(Action, State), depth) :- validAction(Action, State, depth).

#program prune(depth).
compare(Exp, State, depth) :- expanded(Exp, depth), state(State, E), before(E).
compare(New1, New2, depth) :- newState(New1, depth), newState(New2, depth).

different(S1, S2) :-
    compare(S1, S2, depth), holds(variable(V), Value1, S1), not holds(variable(V), Value1, S2).

before(0..depth-1).

newState(State, depth) :-
    expanded(State, depth),
    different(State, S) : state(S, D), before(D).

state(State, depth) :- newState(State, depth),
    different(State, S) : newState(S, depth), S < State.

transition(State, Other) :-
    state(State, depth-1), validAction(Action, State, depth), state(Other, _),
    compare(apply(Action, State), Other, depth), not different(apply(Action, State), Other).

#program show_default(depth).

#show state(S, depth) : state(S, depth).
#show state(S) : state(S, depth).
#show goal(G) : goal(G), state(G, depth).
#show transition(S, T) : transition(S, T), state(S, depth-1).
#show has(X, T) : has(X, T).
#show variable(V) : variable(V).
#show initialState(S) : initialState(S), state(S, depth).
#show constant(X) : constant(X).
#show holds(Var, V, S) : holds(Var, V, S), state(S, depth).

#program show_numbered(depth).

old(N) :- N = #count{S : state(S, D), D < depth}.
stateId(S, N + M) :- state(S, depth), old(N), M = #count{T : state(T, depth), T < S}.

#show stateId(S, Id) : stateId(S, Id), state(S, depth).
#show state(Id, depth) : state(S, depth), stateId(S, Id).
#show state(Id) : state(S, depth), stateId(S, Id).
#show goal(Id) : goal(G), state(G, depth), stateId(G, Id).
#show transition(S, Id2) : transition(S, T), state(S, depth-1), state(T, depth), stateId(T, Id2).
#show transition(S, T) : transition(S, T), state(S, depth-1), state(T, D), D < depth.
#show has(X, T) : has(X, T).
#show variable(V) : variable(V).
#show initialState(Id) : initialState(S), stateId(S, Id), state(S, depth).
#show constant(X) : constant(X).
#show holds(Var, V, Id) : holds(Var, V, S), state(S, depth), stateId(S, Id).