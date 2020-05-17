%---------------------------------------
% domain
%---------------------------------------

% utils
boolean(true).
boolean(false).

% types
type(type("object")).

% variables
variable(variable(("clear", X1))) :- has(X1, type("object")).
variable(variable(("on-table", X1))) :- has(X1, type("object")).
variable(variable("arm-empty")).
variable(variable(("holding", X1))) :- has(X1, type("object")).
variable(variable(("on", X1, X2))) :- has(X1, type("object")), has(X2, type("object")).

contains(X, value(X, B)) :- variable(X), boolean(B).

% actions
action(action(("pickup", X1))) :- has(X1, type("object")).
precondition(action(("pickup", X1)), variable(("clear", X1)), value(variable(("clear", X1)), true)) :- action(action(("pickup", X1))).
precondition(action(("pickup", X1)), variable(("on-table", X1)), value(variable(("on-table", X1)), true)) :- action(action(("pickup", X1))).
precondition(action(("pickup", X1)), variable("arm-empty"), value(variable("arm-empty"), true)) :- action(action(("pickup", X1))).
postcondition(action(("pickup", X1)), effect(unconditional), variable(("holding", X1)), value(variable(("holding", X1)), true)) :- action(action(("pickup", X1))).
postcondition(action(("pickup", X1)), effect(unconditional), variable(("clear", X1)), value(variable(("clear", X1)), false)) :- action(action(("pickup", X1))).
postcondition(action(("pickup", X1)), effect(unconditional), variable(("on-table", X1)), value(variable(("on-table", X1)), false)) :- action(action(("pickup", X1))).
postcondition(action(("pickup", X1)), effect(unconditional), variable("arm-empty"), value(variable("arm-empty"), false)) :- action(action(("pickup", X1))).

action(action(("putdown", X1))) :- has(X1, type("object")).
precondition(action(("putdown", X1)), variable(("holding", X1)), value(variable(("holding", X1)), true)) :- action(action(("putdown", X1))).
postcondition(action(("putdown", X1)), effect(unconditional), variable(("clear", X1)), value(variable(("clear", X1)), true)) :- action(action(("putdown", X1))).
postcondition(action(("putdown", X1)), effect(unconditional), variable("arm-empty"), value(variable("arm-empty"), true)) :- action(action(("putdown", X1))).
postcondition(action(("putdown", X1)), effect(unconditional), variable(("on-table", X1)), value(variable(("on-table", X1)), true)) :- action(action(("putdown", X1))).
postcondition(action(("putdown", X1)), effect(unconditional), variable(("holding", X1)), value(variable(("holding", X1)), false)) :- action(action(("putdown", X1))).

action(action(("stack", X1, X2))) :- has(X1, type("object")), has(X2, type("object")).
precondition(action(("stack", X1, X2)), variable(("clear", X2)), value(variable(("clear", X2)), true)) :- action(action(("stack", X1, X2))).
precondition(action(("stack", X1, X2)), variable(("holding", X1)), value(variable(("holding", X1)), true)) :- action(action(("stack", X1, X2))).
postcondition(action(("stack", X1, X2)), effect(unconditional), variable("arm-empty"), value(variable("arm-empty"), true)) :- action(action(("stack", X1, X2))).
postcondition(action(("stack", X1, X2)), effect(unconditional), variable(("clear", X1)), value(variable(("clear", X1)), true)) :- action(action(("stack", X1, X2))).
postcondition(action(("stack", X1, X2)), effect(unconditional), variable(("on", X1, X2)), value(variable(("on", X1, X2)), true)) :- action(action(("stack", X1, X2))).
postcondition(action(("stack", X1, X2)), effect(unconditional), variable(("clear", X2)), value(variable(("clear", X2)), false)) :- action(action(("stack", X1, X2))).
postcondition(action(("stack", X1, X2)), effect(unconditional), variable(("holding", X1)), value(variable(("holding", X1)), false)) :- action(action(("stack", X1, X2))).

action(action(("unstack", X1, X2))) :- has(X1, type("object")), has(X2, type("object")).
precondition(action(("unstack", X1, X2)), variable(("on", X1, X2)), value(variable(("on", X1, X2)), true)) :- action(action(("unstack", X1, X2))).
precondition(action(("unstack", X1, X2)), variable(("clear", X1)), value(variable(("clear", X1)), true)) :- action(action(("unstack", X1, X2))).
precondition(action(("unstack", X1, X2)), variable("arm-empty"), value(variable("arm-empty"), true)) :- action(action(("unstack", X1, X2))).
postcondition(action(("unstack", X1, X2)), effect(unconditional), variable(("holding", X1)), value(variable(("holding", X1)), true)) :- action(action(("unstack", X1, X2))).
postcondition(action(("unstack", X1, X2)), effect(unconditional), variable(("clear", X2)), value(variable(("clear", X2)), true)) :- action(action(("unstack", X1, X2))).
postcondition(action(("unstack", X1, X2)), effect(unconditional), variable(("on", X1, X2)), value(variable(("on", X1, X2)), false)) :- action(action(("unstack", X1, X2))).
postcondition(action(("unstack", X1, X2)), effect(unconditional), variable(("clear", X1)), value(variable(("clear", X1)), false)) :- action(action(("unstack", X1, X2))).
postcondition(action(("unstack", X1, X2)), effect(unconditional), variable("arm-empty"), value(variable("arm-empty"), false)) :- action(action(("unstack", X1, X2))).

%---------------------------------------
% problem
%---------------------------------------

% objects
constant(constant("a")).
has(constant("a"), type("object")).

constant(constant("b")).
has(constant("b"), type("object")).

constant(constant("c")).
has(constant("c"), type("object")).

constant(constant("d")).
has(constant("d"), type("object")).

constant(constant("e")).
has(constant("e"), type("object")).

% initial state
initialState(variable(("clear", constant("b"))), value(variable(("clear", constant("b"))), true)).
initialState(variable(("clear", constant("c"))), value(variable(("clear", constant("c"))), true)).
initialState(variable(("clear", constant("e"))), value(variable(("clear", constant("e"))), true)).
initialState(variable(("on-table", constant("c"))), value(variable(("on-table", constant("c"))), true)).
initialState(variable(("on-table", constant("d"))), value(variable(("on-table", constant("d"))), true)).
initialState(variable(("on", constant("a"), constant("d"))), value(variable(("on", constant("a"), constant("d"))), true)).
initialState(variable(("on", constant("b"), constant("a"))), value(variable(("on", constant("b"), constant("a"))), true)).
initialState(variable(("on-table", constant("e"))), value(variable(("on-table", constant("e"))), true)).
initialState(variable("arm-empty"), value(variable("arm-empty"), true)).

initialState(X, value(X, false)) :- variable(X), not initialState(X, value(X, true)).

% goal
goal(variable(("clear", constant("d"))), value(variable(("clear", constant("d"))), true)).
