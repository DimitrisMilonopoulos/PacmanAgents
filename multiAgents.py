# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random
import util

# random
import random

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # print "Curr, Succ", currentGameState, successorGameState
        newPos = successorGameState.getPacmanPosition()
        # print newPos
        oldFood = currentGameState.getFood()
        oldFood = oldFood.asList()
        newFood = successorGameState.getFood()
        newFood = newFood.asList()  # Make the food grid as coordinate List
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = successorGameState.getGhostPositions()
        # print newGhostPositions
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        # print newScaredTimes

        "*** YOUR CODE HERE ***"
        score = 0
        ghost_distances = []
        "Calculate the ghost distances to the new position"
        for ghost in newGhostPositions:
            ghost_distances.append(util.manhattanDistance(newPos, ghost))

        "Estimate the closest food to the new position(Manhattan Distance)"
        mini = 99999
        mdistance = 0
        for snack in newFood:
            mdistance = util.manhattanDistance(newPos, snack)
            if mdistance < mini:
                mini = mdistance

        for dist in ghost_distances:
            if dist == 0:
                return -9999

        if newPos in oldFood: #add to the score if in the new state the pacman eats food
            score += 10  
        if mdistance != 0: #add higher score to the pacman if distance to the closest score is smaller
            score += 10.0/mdistance
        if newPos == currentGameState.getPacmanPosition(): #punish the pacman if it doesn't move
            score -= 7
        return score 


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def Max_Value(curr_state, agentIndex, depth):
            "Get the index value of the agent"
            

            if depth == self.depth:   #if we are in a leaf return the evaluation function
                return self.evaluationFunction(curr_state)
            else: #else get the legal actions, if there aren't any return the evaluation function
                legal_actions = curr_state.getLegalActions(agentIndex)
                if not legal_actions:
                    return self.evaluationFunction(curr_state)

            #Now lets find the move which gives the higher value
            v = -float("inf")
            best_move = 0
            for legal_action in legal_actions:
                value = Min_Value( #increment the agent index so that we check the moves of other players
                    curr_state.generateSuccessor(agentIndex, legal_action),agentIndex + 1, depth) 
                if value > v:
                    v=value
                    best_move = legal_action

            #if we are in the root we need to return the best move
            if depth == 0:
                return best_move
            else: #else return the value of the best move
                return v

        def Min_Value(curr_state, agentIndex, depth):

            legal_actions = curr_state.getLegalActions(agentIndex)
            if not legal_actions:
                return self.evaluationFunction(curr_state)

            v = float("inf")
            # that means that the next will be pacman, so we call the Max_Value
            if agentIndex == curr_state.getNumAgents() - 1:
                for legal_action in legal_actions:
                    value = Max_Value( #our agent will be pacman so agentIndex = 0
                        curr_state.generateSuccessor(agentIndex, legal_action), 0,depth+1)
                    if value < v:
                        v = value
                        

            else: #not pacman so its a ghost, so we call Min_Value, incrementing the agentIndex
                for legal_action in legal_actions:
                    value = Min_Value(
                        curr_state.generateSuccessor(agentIndex, legal_action), agentIndex + 1, depth) 
                    if value < v:
                        v = value
                        
            #just return the value in min nodes
            return v

        return Max_Value(gameState, 0, 0)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def Max_Value(curr_state, a,b,agentIndex, depth): #a,b values for a-b pruning
            "Get the index value of the agent"
            
            #same procedure as with the Minimax, for referance look above.
            # Comments only on pruning related lines
            #Following the a-b pruning algorithm as found in the questions instructions
            if depth == self.depth:
                return self.evaluationFunction(curr_state)
            else:
                legal_actions = curr_state.getLegalActions(agentIndex)
                if not legal_actions:
                    return self.evaluationFunction(curr_state)


            v = -float("inf")
            best_move = 0
            for legal_action in legal_actions:
                value = Min_Value(
                    curr_state.generateSuccessor(agentIndex, legal_action),a,b,agentIndex + 1, depth)
                if value > v: #if the value is better than the value we already have
                    v=value   #change the best value
                    best_move = legal_action #as well as the best action
                if v > b: #if v > b no need to search anymore in the moves
                    break
                a = max(a,v)

            if depth == 0:
                return best_move
            else:
                return v

        def Min_Value(curr_state,a,b, agentIndex, depth):

            legal_actions = curr_state.getLegalActions(agentIndex)
            if not legal_actions:
                return self.evaluationFunction(curr_state)

            v = float("inf")
            # that means that the next will be pacman
            if agentIndex == curr_state.getNumAgents() - 1:
                for legal_action in legal_actions:
                    value = Max_Value(
                        curr_state.generateSuccessor(agentIndex, legal_action),a,b, 0,depth+1)
                    if value < v:
                        v = value
                    if v < a: #if v < a no need to search any more moves
                        break
                    b = min(v,b)

            else:
                for legal_action in legal_actions:
                    value = Min_Value(
                        curr_state.generateSuccessor(agentIndex, legal_action),a,b, agentIndex + 1, depth)
                    if value < v:
                        v = value
                    if v < a:
                        break
                    b = min(v,b)

            return v

        return Max_Value(gameState, -float("inf"),float("inf"),0, 0)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def Max_Value(curr_state, agentIndex, depth):
            "Get the index value of the agent"
            
            #same as in minimax
            if depth == self.depth:   
                return self.evaluationFunction(curr_state)
            else:
                legal_actions = curr_state.getLegalActions(agentIndex)
                if not legal_actions:
                    return curr_state.getScore()


            v = -float("inf")
            best_move = 0
            for legal_action in legal_actions:
                value = Chance_Value(
                    curr_state.generateSuccessor(agentIndex, legal_action),agentIndex + 1, depth)
                if value > v:
                    v=value
                    best_move = legal_action

            if depth ==0:
                return best_move
            else:
                return v

        def Chance_Value(curr_state, agentIndex, depth):
            legal_actions = curr_state.getLegalActions(agentIndex)
            if not legal_actions:
                return self.evaluationFunction(curr_state)

            suma = 0 
            # that means that the next will be pacman
            # now we sum all of the move values
            if agentIndex == curr_state.getNumAgents() - 1:
                for legal_action in legal_actions:
                    value = Max_Value(
                        curr_state.generateSuccessor(agentIndex, legal_action), 0,depth+1)
                    suma += value
            else:
                for legal_action in legal_actions:
                    value = Chance_Value(
                        curr_state.generateSuccessor(agentIndex, legal_action), agentIndex + 1, depth)
                    suma+= value
            avg = float(suma)/len(legal_actions) #that's the same as calculating the same propability in all of the nodes
            return avg

        return Max_Value(gameState, 0, 0)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    PacManPosition = currentGameState.getPacmanPosition()
    FoodPositions = currentGameState.getFood()
    FoodPositions = FoodPositions.asList()
    GhostPositions = currentGameState.getGhostPositions()
    GhostStates = currentGameState.getGhostStates()
    GhostScaredTimes = [
        ghostState.scaredTimer for ghostState in GhostStates]
    ghost_distances = []
  
    #estimate the shortest distance to a food
    food_distances= []
    for snack in FoodPositions:
        food_distances.append(manhattanDistance(PacManPosition,snack))

    #make it so that the lower it is the better the value it gives
    if FoodPositions:
        minimum_dist = min(food_distances)
        minimum_dist = 10.0/minimum_dist
    else:
        minimum_dist = 0

    food = 0
    #increase the score if for the states that have lower food count
    if FoodPositions:
        food = 400.0/float(len(FoodPositions))
    else:
        food = 0

    
    
    
    if 0 not in GhostScaredTimes:#Only if all the ghosts are scared, hunt the closest ghost
        # get the minimum distance from a ghost
        for ghost in GhostPositions:
            ghost_distances.append(util.manhattanDistance(PacManPosition, ghost))
            min_ghost_dist = min(ghost_distances)
        if GhostPositions: #we give it a high value as it is high priority to eat it to increase the score
            min_ghost_dist = 30.0/min_ghost_dist
        else:
            min_ghost_dist = 0
        return currentGameState.getScore() +food + minimum_dist + min_ghost_dist

    #get the capsules-pellets so we can scare the ghosts
    CapsulePositions = currentGameState.getCapsules()
    caps_dist = []
    #find the minimum capsule distance
    minimum_caps_dist = 0
    for capsule in CapsulePositions:
        caps_dist.append(manhattanDistance(PacManPosition,capsule))
    if CapsulePositions:
        minimum_caps_dist =min(caps_dist)
        
    if minimum_caps_dist !=0: #give high priority to eating the capsules
            minimum_caps_dist = 25.0/minimum_caps_dist

    
        
    return currentGameState.getScore() + minimum_dist +food + minimum_caps_dist
    





# Abbreviation
better = betterEvaluationFunction
