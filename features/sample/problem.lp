#show.
#defined initialState/1.
#defined contains/3.
#defined mutexGroup/1.
#defined derivedVariable/1.
#defined precondition/4.
#defined postcondition/5.
#defined derivedPredicate/2.
#defined state/2.
#defined validAction/3.
#defined stateId/2.
#defined derivedPredicate/1.

#program initial_state.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Establish initial state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

expanded(initialState, 0).
holds(Variable, Value, initialState) :- initialState(Variable, Value).

#program successor.

validAction(Action, State, Depth + 1) :-
    state(State, Depth), action(Action),
    holds(Variable, Value, State) : precondition(Action, Variable, Value).

%validAction(Action, State, Depth + 1) :-
%    state(State, Depth), action(Action),
%    not goal(State), %Goal states are terminal, no further actions can be performed
%    holds(Variable, Value, State) : precondition(Action, Variable, Value).

expanded(apply(Action, State), Depth) :- validAction(Action, State, Depth).

#program evaluate.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute derived predicates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% conjunctive preconditions
satisfied(DerivedPredicate, S) :- derivedPredicate(DerivedPredicate), holds(Variable, Value, S) : precondition(DerivedPredicate, Variable, Value); expanded(S, _).

% disjunctive preconditions
%satisfied(DerivedPredicate, type(or), S) :- precondition(DerivedPredicate, type(or), Variable, Value), holds(Variable, Value, S), expanded(S, _).
%TODO disjunctive preconditions not working in plasp

holds(DerivedVariable, Value, S) :- satisfied(DerivedPredicate, S), postcondition(DerivedPredicate, effect(unconditional), DerivedVariable, Value), expanded(S, _).

holds(DerivedVariable, value(DerivedVariable, false), S) :- derivedVariable(DerivedVariable), not holds(DerivedVariable, value(DerivedVariable, true), S), expanded(S, _).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform actions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Apply effects

caused(Variable, Value, apply(Action, State)) :-
	validAction(Action, State, _),
	postcondition(Action, Effect, Variable, Value),
	holds(VariablePre, ValuePre, State) : precondition(Effect, VariablePre, ValuePre).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inertia rules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

modified(Variable, State) :- caused(Variable, Value, State), expanded(State, _).

holds(Variable, Value, apply(Action, State)) :- caused(Variable, Value, apply(Action, State)).

holds(variable(V), Value, apply(Action, State)) :- 
    holds(variable(V), Value, State), not modified(variable(V), apply(Action, State)),
    validAction(Action, State, _).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variables and mutex groups
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check that variables have unique values
:- variable(Variable), not 1 {holds(Variable, Value, S) : contains(Variable, Value)} 1, expanded(S, _).

% Check mutexes
:- mutexGroup(MutexGroup), not {holds(Variable, Value, S) : contains(MutexGroup, Variable, Value)} 1, expanded(S, _).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Verify that goal is met
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

goal(State) :- expanded(State, _), holds(Variable, Value, State) : goal(Variable, Value).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program show_expanded.

%1 = {show(N) : N = 0..2}.
1 = {show(State) : expanded(State, _)}.
count(N) :- N =  #count{S: expanded(S, _)}.

#show expanded(State, Depth) : expanded(State, Depth), show(State).
#show holds(Var, V, State) : holds(Var, V, State), show(State).
%#show transition(St1, St2) : transition(St1, St2), show_all(St1).
#show goal(Goal) : goal(Goal), show(Goal).

#program to_state(id).

:- 1 != #count{State : expanded(State, _)}.
stateId(State, id) :- expanded(State, _).

#show stateId(State, id) : stateId(State, id).
#show state(id).
#show state(id, Depth) : expanded(State, Depth), stateId(State, id).
#show holds(Var, V, id) : holds(Var, V, State), stateId(State, id).
#show goal(id) : goal(Goal), stateId(Goal, id).

#program encode.

#show state(State) : state(State).
#show state(State, Depth) : state(State, Depth).
#show stateId(State, Id) : stateId(State, Id).
#show hold(State, Pred) : hold(State, Pred).
#show hold(State, Pred, Const) : hold(State, Pred, Const).
#show goal(Goal) : goal(Goal).

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

#program show_numbered(depth, start).

stateId(S, start + M) :- state(S, depth), M = #count{T : state(T, depth), T < S}.

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