# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


class ImmutableState:
    """
    Note: This does use more memory but the complexity stays the same.
    Old states will be garbage-collected.
    """

    def __init__(self, state, path=None, actions=None, cost=0):
        if actions is None:
            actions = []
        if path is None:
            path = [state]
        self.state = state
        self.path = path
        self.actions = actions
        self.cost = cost

    def visited(self, state):
        return self.state_hash(state) in self.path

    def state_hash(self, state=None):
        if state is None:
            state = self.state
        hash = ""
        try:
            hash_dict = vars(state)
            hash = ", ".join(["%s: %s" % (str(k), str(v)) for k, v in
                              hash_dict.iteritems()])
        except TypeError:
            hash = state
        return hash

    def mutate(self, action, next_state, cost=0):
        return ImmutableState(next_state,
                              self.path[:] + [self.state_hash(next_state)],
                              self.actions[:] + [action],
                              self.cost + cost)


def pushWithCost(container, item, cost, useCost):
    if useCost:
        container.push(item, cost)
    else:
        container.push(item)


def searchWithContainer(problem, container,
                        reverseSequence=False,  # DFS
                        useCost=False,  # UCS, containers with cost
                        heuristic=None  # heuristic for A*
                        ):
    if heuristic is None:
        heuristic = nullHeuristic

    con = container()
    pushWithCost(con, ImmutableState(problem.getStartState()), 0, useCost)
    visited = []

    while not con.isEmpty():
        current_state = con.pop()
        if problem.isGoalState(current_state.state):
            return current_state.actions

        if current_state.state_hash() in visited:
            continue
        visited.append(current_state.state_hash())

        successors = problem.getSuccessors(current_state.state)
        if reverseSequence:
            successors = reversed(successors)

        for successor, action, cost in successors:
            next_state = current_state.mutate(action, successor, cost)
            if successor not in visited and not current_state.visited(
                    successor):
                pushWithCost(con,
                             next_state,
                             next_state.cost +
                             heuristic(next_state.state, problem),
                             useCost)


def depthFirstSearch(problem):
    """
    :type problem SearchProblem

    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    return searchWithContainer(problem, util.Stack, reverseSequence=True)


def breadthFirstSearch(problem):
    """
    @type problem: SearchProblem
    Search the shallowest nodes in the search tree first.
    """
    return searchWithContainer(problem, util.Queue)


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    return aStarSearch(problem, heuristic=nullHeuristic)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    return searchWithContainer(problem, util.PriorityQueue, useCost=True,
                               heuristic=heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
