%*
Atoms in plasp encoding:
type            no 
boolean         no
contains        no
action          no
precondition    no
postcondition   no
validAction     ours
caused          no
expanded        ours
modified        no
before          ours
compare         ours
newState        ours
different       ours
has(constant, type)             yes     something has a type     
variable(variable)              yes     it is a variable
initialState(state)             yes     initial state
constant(constant)              yes     something is a constant (object)
goalState(state)                yes     a state is a goal
state(state, depth)				yes     it is a state
holds(variable, value, state)   yes     variable has value
transition(state, state)        yes		there is a transition between states

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

%arity(V, 0) :- variable(variable((V))).
arity(V, 1) :- variable(variable((V, _))).
arity(V, 2) :- variable(variable((V, _, _))).

const(X) :- constant(constant(X)).



#show arity/2.
#show const/1.

#program arity0(p).
arity(p, 0).