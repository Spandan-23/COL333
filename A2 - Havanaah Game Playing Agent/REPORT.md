# Havannah Game Playing AI: Monte Carlo Tree Search with Strategic Enhancements


## Introduction

This report outlines the design and implementation of an AI player for the game of Havannah using Monte Carlo Tree Search (MCTS) with various optimizations. The AI is designed to efficiently explore possible game states and make decisions based on the current board configuration while considering both self and opponent strategies. Due to the large search space, MCTS is augmented with a custom evaluation function, specific state checks, and mechanisms to recognize and react to certain critical game structures like forks, bridges, rings, and kite formations.


## Key Implementation Details

### 1. Monte Carlo Tree Search (MCTS)

At the core of the AI is the MCTS algorithm, which simulates possible future moves to estimate the most promising actions. However, due to the large number of possible states in Havannah, the search cannot extend to terminal states. Instead, the following steps are taken:

- **a) Selection**: MCTS selects nodes using a UCT (Upper Confidence Bound for Trees) strategy that balances exploration of unvisited nodes and exploitation of known strong moves.
  
- **b) Expansion**: A selected node is expanded by generating valid child nodes (new board states) from available moves.
  
- **c) Simulation**: Rather than simulating until the end of the game, each state is evaluated using a custom scoring function that checks for partial structures (forks, bridges, rings) formed by both the AI and the opponent.
  
- **d) Backpropagation**: The results of the simulation are propagated back through the tree, updating the score and visit counts for each node.

### 2. Custom Evaluation Function

The evaluation function plays a pivotal role in assessing the strength of intermediate game states. It looks at partial formations of important structures:

- **a) Partial Forks**: Evaluates how close a player is to creating a fork by connecting two or more board edges.
  
- **b) Partial Bridges**: Measures proximity to creating a bridge between two corners of the board.
  
- **c) Partial Rings**: Scores potential closed loops of connected cells, which would form a ring.

The score also accounts for control of key areas (corners and edges), as well as connectivity between pieces, thus prioritizing moves that strengthen ongoing strategies or block opponent advancements.

### 3. Recognizable States

Before invoking MCTS, the AI checks for specific game states to react swiftly and decisively:

- **a) Winning Moves**: If the AI can win in the current move, it takes that move immediately.
  
- **b) Opponent’s Winning Moves**: If the opponent can win in one move, the AI blocks that move without engaging MCTS.
  
- **c) Two-Step Wins**: Both for itself and the opponent, the AI evaluates whether it can secure a win or block an opponent’s two-move strategy.

### 4. Virtual Connections and Kite Structures

An additional strategic layer involves recognizing kite structures, which represent virtual connections on the board that can either prevent or create bridges and forks. The AI identifies these structures for both players and, if necessary, takes steps to disrupt or complete them.

- **a) Self-Kite Disruption**: The AI ensures that it preserves its virtual connections by securing critical cells that maintain the integrity of its kite structures.
  
- **b) Opponent Kite Blocking**: The AI identifies potential kite structures the opponent may form and blocks them by placing pieces in key positions.

 

## Strategy and Observations

### 1. Efficiency in State Recognition

The early detection of win conditions and threats significantly reduces unnecessary MCTS simulations. This leads to faster decisions, especially in scenarios where the game can be decided in one or two moves.

### 2. Partial Structure Evaluation

The evaluation function’s focus on partial forks, bridges, and rings allows the AI to make more informed decisions in mid-game states where winning conditions are still far off. This scoring method provides a competitive edge in the absence of fully completed structures.

### 3. Handling Opponent Strategies

The AI not only focuses on building its own winning structures but also constantly evaluates the opponent's progress. This defensive aspect improves the AI’s ability to delay the opponent's advances while it works on securing its own victory.

 

## Improvements and Future Work

### 1. Heuristic Refinements

The evaluation function can be further improved by incorporating more complex heuristics for predicting future moves, especially in long-term planning for forks and rings.

### 2. Pathfinding Optimization

The current implementation of checking connectivity and evaluating partial rings through DFS could be optimized using more efficient pathfinding algorithms, especially on larger board sizes.

### 3. Endgame Strategy

As the game approaches its conclusion, the AI could shift its focus from structure building to aggressive blocking or securing key cells that prevent any sudden victories by the opponent.

 

## Conclusion

This AI player for Havannah employs a robust strategy that integrates Monte Carlo Tree Search with targeted state evaluations and strategic checks. The combination of fast win detection, partial structure evaluation, and virtual connections allows the AI to make competitive decisions within its time constraints. Although there is room for further enhancement, the current implementation demonstrates strong performance across a range of scenarios, showcasing the effectiveness of combining MCTS with specialized heuristics for a complex board game like Havannah.
