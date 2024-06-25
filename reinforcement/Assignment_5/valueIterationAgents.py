# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # For each iteration
        for i in range(iterations):
            # Initialize valuesCopy to a copy of self.values
            valuesCopy = self.values.copy()

            # For each state in the MDP
            for state in mdp.getStates():
                # If the state is terminal
                if mdp.isTerminal(state):
                    # Set the value of the state to 0
                    valuesCopy[state] = 0
                else:
                    # Get the possible actions for the state
                    possibleActions = mdp.getPossibleActions(state)

                    # Initialize qValues to an empty list
                    qValues = []

                    # For each possible action
                    for action in possibleActions:
                        # Compute the qValue for the action
                        qValue = self.computeQValueFromValues(state, action)

                        # Append the qValue to qValues
                        qValues.append(qValue)
                    
                    # Set the value of the state to the maximum qValue
                    valuesCopy[state] = max(qValues)
            
            # Update self.values to valuesCopy
            self.values = valuesCopy
        


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Initialize qValue to 0
        qValue = 0

        # Get the transition states and probabilities
        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)

        # For each transition state and probability
        for nextState, prob in transitionStatesAndProbs:
            # Get the reward for the transition
            reward = self.mdp.getReward(state, action, nextState)

            # Update qValue
            qValue += prob * (reward + self.discount * self.values[nextState])

        # Return the qValue
        return qValue

        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Initialize bestAction to None
        bestAction = None

        # Initialize bestQValue to negative infinity
        bestQValue = float("-inf")

        # Get the possible actions for the state
        possibleActions = self.mdp.getPossibleActions(state)

        # If there are no possible actions, return None
        if len(possibleActions) == 0:
            return None
        
        # For each possible action
        for action in possibleActions:
            # Get the qValue for the action
            qValue = self.computeQValueFromValues(state, action)

            # If the qValue is greater than the bestQValue
            if qValue > bestQValue:
                # Update bestQValue
                bestQValue = qValue

                # Update bestAction
                bestAction = action
        
        # Return the bestAction
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
