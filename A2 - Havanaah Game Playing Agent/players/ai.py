import time
import math
import random
import itertools
import numpy as np
from helper import *

from collections import defaultdict

class AIPlayer:

    def __init__(self, player_number: int, timer):
        """
        Intitialize the AIPlayer Agent

        # Parameters
        `player_number (int)`: Current player number, num==1 starts the game
        
        `timer: Timer`
            - a Timer object that can be used to fetch the remaining time for any player
            - Run `fetch_remaining_time(timer, player_number)` to fetch remaining time of a player
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        self.max_moves = 100
        self.num_simulations = 1000  
        self.max_eval_time = 100
    
    def get_move(self, state: np.array) -> Tuple[int, int]:
        
        winning_move = self.check_winning_move(state, self.player_number)
        if winning_move and state[winning_move] == 0:
            # print("wining move")
            return winning_move

        opponent_winning_move = self.check_winning_move(state, 3 - self.player_number)
        if opponent_winning_move and state[opponent_winning_move] == 0:
            # print("opponent 1 step win")
            return opponent_winning_move

        four_move = self.get_move_four(state)
        if state.shape[0] == 7 and four_move and state[four_move] == 0:
            # print("not here plz")
            return four_move
        
        self_winning_move_2 = self.two_step_win(state, self.player_number)
        if self_winning_move_2 and state[self_winning_move_2] == 0:
            # print("self 2 step win")
            return self_winning_move_2
        
        opponent_winning_move_2=self.two_step_win(state,3-self.player_number)
        if opponent_winning_move_2 and state[opponent_winning_move_2] == 0:
            # print("opponent 2 step win")
            return opponent_winning_move_2
        
        preserve_connection = self.self_kite_disrupt_move( state)
        if preserve_connection:
            i, j= preserve_connection
            
        if preserve_connection and is_valid(i,j , state.shape[0]) :
            # print("preserving self kite connection")
            return preserve_connection
        
        corners = get_all_corners(state.shape[0])
        empty_corners = [corner for corner in corners if state[corner] == 0]
        if empty_corners :
            if len(corners) - len(empty_corners) <= 2 and random.random() <0.25:  
                return random.choice(empty_corners)
            elif random.random()<0.15:
                return random.choice(empty_corners)
        near_corner_moves = self.get_near_corner_moves(state)
        if random.random() < 0.3 and near_corner_moves :  
            # print("Near corner moves ",near_corner_moves)
            # print("Near corner taken")
            move = random.choice(near_corner_moves)
            return move
        
        dim = state.shape[0]
        center = (dim // 2, dim // 2)
        if(random.random() < 0.3 and state[center] == 0):
            return center
                
        kite_move = self.check_kite_structures(state)
        # print("kite move ",kite_move)
        if kite_move and state[kite_move] == 0:
            # print("---------------------------------------------------------------------------------")
            # print(kite_move)
            return kite_move
        # Check for opponent's potential fork structures
        opponent_fork_prevention_move = self.check_opponent_fork_structures(state)
        if opponent_fork_prevention_move and state[opponent_fork_prevention_move] == 0:
            # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # print(opponent_fork_prevention_move)
            return opponent_fork_prevention_move
    
        
        
        # Check for opponent's potential kite structures
            
        opponent_kite_move = self.check_opponent_kite_structures(state)
        if opponent_kite_move and state[opponent_kite_move] == 0:
            # print("**********************************************")
            # print(opponent_kite_move)
            return opponent_kite_move
        iterations=0
        remaining_time = fetch_remaining_time(self.timer,self.player_number)
        root_node = Node(state, self.player_number)
        while iterations<self.num_simulations and remaining_time-fetch_remaining_time(self.timer,self.player_number) < 20  :

            # print(iterations)
            node = self.select_node(root_node)
            child_node=self.expand_node(node)
            moves,result = self.simulate_game(child_node)
            self.backpropagate(child_node,moves,result)
            iterations+=1
        return self.get_best_move(root_node)
        
    def get_move_four(self, state):
        opponent = 3 - self.player_number
        imp_kite_points = {
            (1,2):[(0,0),(0,3)],
            (1,4):[(0,3),(0,6)],
            (2,5):[(0,6),(3,6)],
            (4,4):[(3,6),(6,3)],
            (4,2):[(6,3),(3,0)],
            (2,1):[(3,0),(0,0)]
        }
        for point,corner_points in imp_kite_points.items():
            # print(point)
            if(state[point] == opponent):
                if state[corner_points[0]] != 0:
                    return corner_points[1]
                elif state[corner_points[1]] != 0:
                    return corner_points[0]
        return None
        
    def self_kite_disrupt_move(self,state):
        opponent = 3- self.player_number
        dim = state.shape[0]
        for i in range(dim):
            for j in range(dim):
                current_cell = (i,j)
                if state[current_cell] == opponent:
                    neighbours = self.get_valid_neighbours(dim,current_cell,state)
                    for n1 in neighbours:
                        for n2 in neighbours:
                            if n1==n2 or state[n1]!=self.player_number or state[n2]!=self.player_number:
                                continue
                            if not self.is_path(state,n1,n2):
                                common = set(self.get_valid_neighbours(dim,n1,state)) & set(self.get_valid_neighbours(dim,n2,state))
                                common.remove(current_cell)
                                if len(common)>0:
                                    ans_cell = common.pop()
                                    if state[ans_cell]==0:
                                        return ans_cell
        return None
    
    def two_step_win(self,state,player):
        dim=state.shape[0]
        move_counts = defaultdict(int) 
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] == 0:  
                    test_state = np.copy(state)
                    test_state[i, j] = player
                    
                    steps = self.check_winning_move(test_state,player)
                    if  steps :
                        move_counts[(i,j)]+=1
                        move_counts[steps]+=1

        if not move_counts :
            return None  
        return max(move_counts,key=move_counts.get)
    def get_valid_neighbours(self,dim,vertex,state):
        neighbours=get_neighbours(dim,vertex)
        neighbours=[n for n in neighbours if state[n]!=3]  
        return neighbours
    def get_near_corner_moves(self, state: np.array) -> List[Tuple[int, int]]:
        near_corner_moves = []
        dim = state.shape[0]
        corners = get_all_corners(dim)
        for c in corners:
            for vertex in self.get_valid_neighbours(dim,c,state):
                if len(self.get_valid_neighbours(dim,vertex,state))==6 and state[vertex]==0:
                    near_corner_moves.append(vertex)
        if len(near_corner_moves)>0:
            return near_corner_moves
        return None

    def check_opponent_kite_structures(self, state: np.array) -> Tuple[int, int]:
        opponent = 3 - self.player_number
        dim = state.shape[0]
        def is_connected_to_corner(x, y):
            return any(state[nx, ny] == opponent for nx, ny in get_neighbours(dim, (x, y)) if self.is_connected_to_corner(state,(nx,ny),opponent))

        def is_connected_to_edge(x, y):
            return any(state[nx, ny] == opponent for nx, ny in get_neighbours(dim, (x, y)) if self.is_connected_to_edge(state,(nx,ny),opponent))

        def find_potential_kite(x, y):
            neighbours = get_neighbours(dim,(x,y))
            for (x1,y1) in neighbours:
                for (x2,y2) in neighbours:
                    if ((x1,y1) == (x2,y2)) or (state[x1,y1]!=0 or state[x2,y2]!=0) :
                        continue
                    neighbours_x1y1 = set(get_neighbours(dim,(x1,y1)))
                    neighbours_x2y2 = set(get_neighbours(dim,(x2,y2)))
                    common = neighbours_x1y1.intersection(neighbours_x2y2)
                    if len(common)<2:
                        continue
                    common.remove((x,y))
                    kite = common.pop()
                    if state[kite] !=0 or not (is_connected_to_corner(kite[0],kite[1]) or is_connected_to_edge(kite[0],kite[1])):
                        continue
                    connect_two_corners =is_connected_to_corner(kite[0],kite[1]) and (is_connected_to_corner(x1,y1) or is_connected_to_corner(x2,y2))
                    connect_two_edges = is_connected_to_edge(kite[0],kite[1]) and (is_connected_to_edge(x1,y1) or is_connected_to_edge(x2,y2))
                    if connect_two_corners or connect_two_edges:
                        return (x,y)
            return None

        for i in range(dim):
            for j in range(dim):
                if state[i, j] == 0:
                    potential_kite = find_potential_kite(i, j)
                    if potential_kite:
                        return potential_kite
        
        return None
    
    def check_opponent_fork_structures(self, state: np.array) -> Tuple[int, int]:
        opponent = 3 - self.player_number
        dim = state.shape[0]

        def check_two_side_connection(board: np.array, move: Tuple[int, int]) -> bool:
            visited = bfs_reachable(board, move)
            sides = get_all_edges(dim)
            sides = [set(side) for side in sides]
            
            reachable_edges = [1 if len(side.intersection(visited)) > 0 else 0 for side in sides]
            return sum(reachable_edges) >= 2  

        opponent_board = (state == opponent)

        for i in range(dim):
            for j in range(dim):
                if state[i, j] == 0:  
                    test_board = opponent_board.copy()
                    test_board[i, j] = True
                    
                    if check_two_side_connection(test_board, (i, j)):
                        return (i, j) 

        return None
    
    def is_connected_to_corner(self, state, start, player):
        dim = state.shape[0]
        visited = set()
        stack = [start]
        corners = get_all_corners(dim)
        corners = [corner for corner in corners if state[corner]==player]
        while stack:
            current = stack.pop()
            if current in corners:
                return True
            visited.add(current)
            for neighbor in self.get_neighbors(current, dim):
                if neighbor not in visited and state[neighbor] == player:
                    stack.append(neighbor)

        return False
    
    def is_connected_to_edge(self, state, start, player):
        dim = state.shape[0]
        visited = set()
        stack = [start]
        edges = get_all_edges(dim)
        edges = [vertex for edge in edges for vertex in edge if state[vertex]==player]
        while stack:
            current = stack.pop()
            if current in edges:
                return True
            visited.add(current)
            for neighbor in self.get_neighbors(current, state.shape[0]):
                if neighbor not in visited and state[neighbor] == player:
                    stack.append(neighbor)

        return False
        
    def check_kite_structures(self, state: np.array) -> Tuple[int, int]:
        player = self.player_number
        dim = state.shape[0]
        all_moves=[]
        def find_potential_kite(x, y):
            kites=[]
            neighbours = get_neighbours(dim,(x,y))
            for (x1,y1) in neighbours:
                for (x2,y2) in neighbours:
                    if ((x1,y1) == (x2,y2)) or (state[x1,y1]!=0 or state[x2,y2]!=0) :
                        continue
                    neighbours_x1y1 = set(get_neighbours(dim,(x1,y1)))
                    neighbours_x2y2 = set(get_neighbours(dim,(x2,y2)))
                    common = neighbours_x1y1.intersection(neighbours_x2y2)
                    if len(common)<2:
                        continue
                    common.remove((x,y))
                    kite = common.pop()
                    if kite in neighbours:
                        continue
                    if state[kite] !=0:
                        continue
                    kites.append(kite)
            return kites
        
        for i in range(dim):
            for j in range(dim):
                if state[i, j] == player:
                    potential_kite = find_potential_kite(i, j)
                    all_moves.extend(potential_kite)
        if all_moves:
            return random.choice(all_moves)
        return None


    def get_player_neighbours(self,state,vertex,player):
        neighbours = get_neighbours(state.shape[0],vertex)
        neighbours=[n for n in neighbours if state[n]==player]
        return neighbours
    def select_node(self, node):
        while not node.is_leaf():
            node = self.get_best_child(node)
        return node

    def expand_node(self, node):
        best_children = []
        for move in get_valid_actions(node.state):
            new_state = self.simulate_move(node.state, move,node.player)
            child_node = Node(new_state, 3-node.player)
            child_node.parent = node
            child_node.move = move
            best_children.append((child_node, self.evaluate_state(child_node.state,child_node.player)))
        best_children.sort(key=lambda x: x[1], reverse=True)
        node.children = [child[0] for child in best_children] 
        return best_children[0][0]
    
    def simulate_game(self, node):
        moves_made=0
        state = np.copy(node.state)  
        player = node.player
        moves=[]
        while moves_made==0 or (not check_win(state,moves[-1],3-player)[0] and moves_made<self.max_moves):
            move = random.choice(get_valid_actions(state))
            moves.append(move)
            state = self.simulate_move(state, move,player)
            player = 3 - player
            moves_made+=1
        result = self.get_result(state, 3-player) 
        return moves,result
    def backpropagate(self, node,moves,result):
        moves=set(moves)
        while node is not None:
            node.visits += 1
            node.score += result
            node = node.parent
            result=-result

    def get_best_move(self, node):
        epsilon=1e-18
        best_child = max(node.children, key=lambda child: child.score / (child.visits + epsilon))
        return best_child.move

    def get_best_child(self, node):
        epsilon = 1e-18
        return max(node.children, key=lambda child: child.score / (child.visits + epsilon) + np.sqrt(2 * np.log(node.visits + epsilon) / (child.visits + epsilon)) if child.visits > 0 else np.inf)

    def is_path(self, state, start, end):
    # Checks if there's a path of adjacent cells b/w start and end
        queue = [(start, [start])]
        while queue:
            (x, y), path = queue.pop(0)
            if (x, y) == end:
                return True
            for n in self.get_valid_neighbours(state.shape[0],(x,y),state):
                if state[n] == state[x, y] and n not in path:
                    queue.append((n, path + [n]))
        return False
    def is_bridge_cell(self, state, i, j):
        corners = get_all_corners(state.shape[0])
        for corner in corners:
            if self.is_adjacent(corner, (i, j)):
                # Check if there's a path of adjacent cells with the same color that connects to another corner
                for other_corner in corners:
                    if other_corner != corner and self.is_path(state, (i, j), other_corner):
                        return True
        return False
    def is_fork_cell(self, state, i, j):
    # Check if the cell is part of a potential fork
    # We can check if the cell is part of a fork by checking if it's adjacent to an edge
    # and if there are two or more paths of adjacent cells with the same color that connect to different edges
        edges = get_all_edges(state.shape[0])
        for edge in edges:
            for point in edge:
                if self.is_adjacent(point, (i, j)):
                    # Check if there are two or more paths of adjacent cells with the same color that connect to different edges
                    paths = []
                    for other_edge in edges:
                        if other_edge != edge:
                            for other_point in other_edge:
                                if self.is_path(state, (i, j), other_point):
                                    paths.append(other_point)
                    if len(paths) >= 2:
                        return True
        return False
    def is_ring_cell(self, state, i, j):
        # Check if the cell is part of a potential ring
        # A ring is a closed path of adjacent cells with the same color
        # We can check if the cell is part of a ring by performing a depth-first search (DFS) from the cell
        visited = set()
        if self.is_ring_cell_dfs(state, i, j, visited):
            return True
        return False

    def is_ring_cell_dfs(self, state, i, j, visited):
        # Perform a depth-first search (DFS) from the cell
        visited.add((i, j))
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < state.shape[0] and 0 <= nj < state.shape[1] and state[ni, nj] == state[i, j]:
                if (ni, nj) == (i, j):  # If we're back at the starting cell, it's a ring
                    return True
                if (ni, nj) not in visited and self.is_ring_cell_dfs(state, ni, nj, visited):
                    return True
        return False
    def is_adjacent(self, cell1, cell2):
    # Check if two cells are adjacent
    # Two cells are adjacent if they differ by at most one in either the x or y coordinate
        return abs(cell1[0] - cell2[0]) <= 1 and abs(cell1[1] - cell2[1]) <= 1
    def simulate_move(self, state, move,player):
        # Simulate a move by creating a new state
        new_state = np.copy(state)
        new_state[move] = player
        return new_state
    def check_winning_move(self, state, player):
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] == 0:  # Empty cell
                    # Simulate the move
                    test_state = np.copy(state)
                    test_state[i, j] = player
                    
                    # Convert the state to a boolean array for the checking functions
                    bool_state = (test_state == player)
                    
                    # Check if this move creates a win
                    if check_ring(bool_state, (i, j)) or \
                    check_bridge(bool_state, (i, j)) or \
                    check_fork(bool_state, (i, j)):
                        return (i, j)
        return None

    def evaluate_state(self, state, curr_player):
        start_time = time.time()
        score = 0
        player = curr_player
        opponent = 3 - player
        multiplier = 1 if player == self.player_number else -1
        
        # Evaluate board control
        player_positions = np.argwhere(state == player)
        opponent_positions = np.argwhere(state == opponent)
        
        # Evaluate partial structures
        player_score = self.evaluate_partial_structures(state, player, start_time)
        if player_score is None:
            return 0  # Return early if time limit exceeded
        
        opponent_score = self.evaluate_partial_structures(state, opponent, start_time)
        if opponent_score is None:
            return 0  # Return early if time limit exceeded
        
        score += player_score - opponent_score
        
        # Evaluate control of corners and edges
        corners = get_all_corners(state.shape[0])
        edges = get_all_edges(state.shape[0])
        
        for corner in corners:
            if state[corner] == player:
                score += 10
            elif state[corner] == opponent:
                score -= 10
        
        for edge in edges:
            player_edge_control = sum(1 for cell in edge if state[cell] == player)
            opponent_edge_control = sum(1 for cell in edge if state[cell] == opponent)
            score += 2 * (player_edge_control - opponent_edge_control)
        
        # Evaluate connectivity
        player_connectivity = self.evaluate_connectivity(state, player_positions)
        opponent_connectivity = self.evaluate_connectivity(state, opponent_positions)
        score += 5 * (player_connectivity - opponent_connectivity)
        
        return score * multiplier

    def evaluate_partial_structures(self, state, player, start_time):
        score = 0
        bool_state = (state == player)
        
        ring_score = self.evaluate_partial_rings(bool_state, start_time)
        if ring_score is None:
            return None
        score += ring_score * 10
        
        bridge_score = self.evaluate_partial_bridges(bool_state, start_time)
        if bridge_score is None:
            return None
        score += bridge_score * 8
        
        fork_score = self.evaluate_partial_forks(bool_state, start_time)
        if fork_score is None:
            return None
        score += fork_score * 6
        
        return score


    def evaluate_connectivity(self, state, positions):
        connectivity = 0
        for pos in positions:
            neighbors = get_neighbours(state.shape[0], tuple(pos))
            connected_neighbors = sum(1 for neighbor in neighbors if tuple(neighbor) in positions)
            connectivity += connected_neighbors
        return connectivity

    def evaluate_partial_rings(self, state, start_time):
        score = 0
        dim = state.shape[0]
        for i in range(dim):
            for j in range(dim):
                if time.time() - start_time > self.max_eval_time:
                    return None
                if state[i, j]:
                    ring_size = self.find_partial_ring(state, i, j, start_time)
                    if ring_size is None:
                        return None
                    score += ring_size / 6
        return score

    def find_partial_ring(self, state, i, j, start_time):
        visited = set()
        max_length = 0
        
        def dfs(x, y, length):
            nonlocal max_length
            if time.time() - start_time > self.max_eval_time:
                return None
            if length > max_length:
                max_length = length
            
            visited.add((x, y))
            for nx, ny in get_neighbours(state.shape[0], (x, y)):
                if state[nx, ny] and (nx, ny) not in visited:
                    result = dfs(nx, ny, length + 1)
                    if result is None:
                        return None
            visited.remove((x, y))
            return max_length
        
        return dfs(i, j, 0)

    def evaluate_partial_bridges(self, state, start_time):
        score = 0
        dim = state.shape[0]
        corners = get_all_corners(dim)
        
        for corner1 in corners:
            for corner2 in corners:
                if time.time() - start_time > self.max_eval_time:
                    return None
                if corner1 != corner2:
                    path_length = self.find_longest_path(state, corner1, corner2, start_time)
                    if path_length is None:
                        return None
                    score += path_length / dim
        
        return score

    def find_longest_path(self, state, start, end, start_time):
        visited = set()
        max_length = [0]
        
        def dfs(current, length):
            if time.time() - start_time > self.max_eval_time:
                return None
            if current == end:
                max_length[0] = max(max_length[0], length)
                return
            
            visited.add(current)
            for neighbor in get_neighbours(state.shape[0], current):
                if state[neighbor] and neighbor not in visited:
                    result = dfs(neighbor, length + 1)
                    if result is None:
                        return None
            visited.remove(current)
        
        result = dfs(start, 0)
        if result is None:
            return None
        return max_length[0]

    def evaluate_partial_forks(self, state, start_time):
        score = 0
        dim = state.shape[0]
        edges = get_all_edges(dim)
        
        for i in range(len(edges)):
            for j in range(i+1, len(edges)):
                for k in range(j+1, len(edges)):
                    if time.time() - start_time > self.max_eval_time:
                        return None
                    path1 = self.find_longest_path(state, edges[i][0], edges[j][0], start_time)
                    if path1 is None:
                        return None
                    path2 = self.find_longest_path(state, edges[j][0], edges[k][0], start_time)
                    if path2 is None:
                        return None
                    path3 = self.find_longest_path(state, edges[k][0], edges[i][0], start_time)
                    if path3 is None:
                        return None
                    score += (path1 + path2 + path3) / (3 * dim)
        
        return score

    def _dfs_partial_ring(self, state, i, j, player, visited, start, depth):
        if depth > 0 and self.is_adjacent((i, j), start):
            return depth
        visited.add((i, j))
        max_ring_size = 0
        for ni, nj in self.get_neighbors((i, j), state.shape[0]):
            if (ni, nj) not in visited and state[ni, nj] == player:
                ring_size = self._dfs_partial_ring(state, ni, nj, player, visited.copy(), start, depth + 1)
                max_ring_size = max(max_ring_size, ring_size)
        return max_ring_size
    def get_neighbors(self, pos, board_size):
        i, j = pos
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < board_size and 0 <= nj < board_size:
                neighbors.append((ni, nj))
        return neighbors
    
    def is_game_over(self, state):
        for player in [1, 2]:
            for i in range(state.shape[0]):
                for j in range(state.shape[1]):
                    if state[i, j] == player:
                        if self.is_bridge_cell(state, i, j):
                            return True
                        if self.is_fork_cell(state, i, j) > 2:
                            return True
                        if self.is_ring_cell(state, i, j):
                            return True
        if np.all(state != 0):
            return True
        return False

    def get_result(self, state, player):
        if self.is_bridge_cell(state, 0, 0):
            return 1 if player == self.player_number else -1
        elif self.is_fork_cell(state, 0, 0) > 2:
            return 1 if player == self.player_number else -1
        elif self.is_ring_cell(state, 0, 0):
            return 1 if player == self.player_number else -1
        else:
            return 0

class Node:
    def __init__(self, state, player, move=None):
        self.state = state
        self.player = player
        self.children = []
        self.parent = None
        self.move = move  
        self.visits = 0
        self.score = 0

    def is_leaf(self):
        return len(self.children) == 0