# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    visited = set()                                                 # Set to keep track of visited states
    stack = util.Stack()                                            # Stack to store states to be explored
    stack.push((problem.getStartState(), []))                       # Push the start state and an empty list of actions

    while not stack.isEmpty():                                      # While there are states to explore
        state, actions = stack.pop()                                # Get the next state and list of actions
        if problem.isGoalState(state):                              # Check if the current state is the goal state
            return actions                                          # Return the list of actions if the goal state is reached
        if state not in visited:                                    # Check if the state hasn't been visited
            visited.add(state)                                      # Mark the state as visited
            successors = problem.getSuccessors(state)               # Get the successors of the current state
            for successor, action, _ in successors:                 # Iterate over the successors             
                if successor not in visited:                        # Check if the successor has not been visited
                    stack.push((successor, actions + [action]))     # Push the successor and the updated list of actions

    return []                                                       # Return an empty list if no solution is found

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.

    Args:
        problem: The problem to be solved.

    Returns:
        A list representing the path from the start state to the goal state.
        If no solution is found, an empty list is returned.
    """
    visited = set()                                                                 # Set to keep track of visited states
    queue = util.Queue()                                                            # Queue to store states to be explored
    queue.push((problem.getStartState(), []))                                       # Start state and empty path

    while not queue.isEmpty():                                                      # While there are states to explore
        state, path = queue.pop()                                                   # Get the next state and path
        if problem.isGoalState(state):                                              # If the goal state is reached,
            return path                                                             # return the path to the goal
        if state not in visited:                                                    # If the state has not been visited,
            visited.add(state)                                                      # mark it as visited 
            for successor, action, stepCost in problem.getSuccessors(state):        # For each successor of the state,
                queue.push((successor, path + [action]))                            # add the successor and the path to it to the queue
    
    return []                                                                       # Return empty path if no solution is found

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.

    Args:
        problem: The problem instance to be solved.

    Returns:
        A list of actions that leads to the goal state, or an empty list if no solution is found.
    """
    priorityQueue = util.PriorityQueue()                                            # Initialize Priority Queue
    priorityQueue.push((problem.getStartState(), []), 0)                            # Push the start state and an empty list of actions to the priority queue

    visited = set()                                                                 # Set to keep track of visited states
    best_cost = {}                                                                  # Dictionary to keep track of visited costs and states
    best_cost[problem.getStartState()] = 0                                          # Initialize the cost of the start state to 0                         
    parent = {}                                                                     # Dictionary to keep track of parent states
    parent[problem.getStartState()] = None                                          # Initialize the parent of the start state to None                    

    while not priorityQueue.isEmpty():                                              # While there are states to explore
        state, actions = priorityQueue.pop()                                        # Get the next state and list of actions
        if problem.isGoalState(state):                                              # Check if the current state is the goal state                    
            return actions                                                          # Return the list of actions if the goal state is reached                 
        if state not in visited:                                                    # Check if the state hasn't been visited, if not    
            visited.add(state)                                                      # Mark the state as visited       
            successors = problem.getSuccessors(state)                               # Then, get the successors of the current state
            for successor, action, cost in successors:                              # Iterate over the successors 
                new_cost = best_cost[state] + cost                                  # Calculate the new cost
                if successor not in best_cost or new_cost < best_cost[successor]:   # If the new cost is less than the best cost of the successor
                    best_cost[successor] = new_cost                                 # Update the best cost of the successor
                    priorityQueue.push((successor, actions + [action]), new_cost)   # Push the successor and the updated list of actions to the priority queue
                    parent[successor] = state                                       # Update the parent of the successor

    return []                                                                       # Return an empty list if no solution is found

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    
    states = util.PriorityQueue()                                                                 #Creates a priority queue to store the states   
    start = problem.getStartState()                                                               #Gets the start of the problem
    states.push((start, []), nullHeuristic(start, problem))                                       #Pushes the start state and the heuristic value of the start state    
    explored = []                                                                                 #List to hold explored states   
    cost = 0                                                                                      #Initializes the cost
    if (problem.isGoalState(start)):                                                              #Checks if the start state is the goal state
        return []                                                                                 #Returns an empty list if the start state is the goal state
    while not states.isEmpty():
        state, actions = states.pop()
        if problem.isGoalState(state):                                                            #Checks if the state is the goal state, returns the actions if it is
            return actions
        if state not in explored:                                                                 #Checks if the state hasn't been explored            
            successors = problem.getSuccessors(state)                                             #Get successors of the current state
            for x in successors:
                coords = x[0]
                if coords not in explored:
                    directions = x[1]                    
                    nActions = actions + [directions]                                             #Create a new list of actions by appending the current direction to the existing actions                   
                    cost = problem.getCostOfActions(nActions) + heuristic(coords, problem)        #Calculate the cost of the new actions by adding the cost of the current actions and the heuristic value of the successor state                   
                    states.push((coords, actions + [directions]), cost)                           #Push the successor state and the new actions with their cost into the priority queue        
        explored.append(state)                                                                    #Mark the current state as explored
    return actions                                                                                #Return the actions if no solution is found                 
        


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
