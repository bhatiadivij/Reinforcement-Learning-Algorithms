import numpy as np
import sys
import time
MINUS_INF = -sys.maxint - 1

def read_input(filename):
    game = {}
    with open(filename, 'r') as f:
        game["n"] = int(next(f))
        game["num_walls"] = int(next(f))
        game["walls"] = []
        for i in range(game["num_walls"]):
            game["walls"].append([int(x)-1 for x in next(f).split(',')])
        game["num_terminals"] = int(next(f))
        game["terminals"] = []
        for i in range(game["num_terminals"]):
            game["terminals"].append([int(x)-1 for x in next(f).split(',')])
            game["terminals"][-1][2] += 1
        game["terminal_pos"] = [x[:-1] for x in game["terminals"]]
        game["p"] = float(next(f))
        game["reward"] = float(next(f))
        game["gamma"] = float(next(f))
    f.close()
    return game


def generate_cell_hash_table(game):
    index_map = {}
    k = 0
    for i in range(game["n"]):
        for j in range(game["n"]):
            index_map[(i, j)] = k
            k += 1
    for i in game["walls"]:
        del index_map[tuple(i)]
    return index_map


def generate_reward_matrix(game, index_map):
    reward_matrix = np.full((game["n"]**2), game["reward"], dtype=float)
    # reward_matrix = np.full((game["n"]**2), 0, dtype=float)
    for i in game["terminals"]:
        reward_matrix[index_map[(i[0], i[1])]] = i[2]
    return reward_matrix


def generate_state_transition_matrix(game, index_map):
    # z direction indices represent U,D,R,L
    st_matrix= np.full((game["n"] ** 2, 3, 4), 0, dtype=int)
    
    for i in range(game["n"]):
        for j in range(game["n"]):
            if [i, j] not in game["walls"] and [i,j] not in game["terminal_pos"]:
                # up operation
                st_matrix[index_map.get((i, j))][0][0] = index_map.get((i-1, j), index_map[(i, j)])
                st_matrix[index_map.get((i, j))][1][0] = index_map.get((i-1, j-1), index_map[(i, j)])
                st_matrix[index_map.get((i, j))][2][0] = index_map.get((i-1, j+1), index_map[(i, j)])
                
                # down operation
                st_matrix[index_map.get((i, j))][0][1] = index_map.get((i+1, j), index_map[(i, j)])
                st_matrix[index_map.get((i, j))][1][1] = index_map.get((i+1, j-1), index_map[(i, j)])
                st_matrix[index_map.get((i, j))][2][1] = index_map.get((i+1, j+1), index_map[(i, j)])

                # right operation
                st_matrix[index_map.get((i, j))][0][2] = index_map.get((i, j+1), index_map[(i, j)])
                st_matrix[index_map.get((i, j))][1][2] = index_map.get((i-1, j+1), index_map[(i, j)])
                st_matrix[index_map.get((i, j))][2][2] = index_map.get((i+1, j+1), index_map[(i, j)])

                # left operation
                st_matrix[index_map.get((i, j))][0][3] = index_map.get((i, j-1), index_map[(i, j)])
                st_matrix[index_map.get((i, j))][1][3] = index_map.get((i-1, j-1), index_map[(i, j)])
                st_matrix[index_map.get((i, j))][2][3] = index_map.get((i+1, j-1), index_map[(i, j)])
    return st_matrix


def value_iteration(game, index_map, reward_matrix, st_matrix,t):
    operations = np.full(4, 0, dtype=float)
    actions = ["U", "D", "R", "L"]
    policy = np.full((game["n"], game["n"]), "", dtype=str)
    max_iterations =100000
    epsilon = 1e-3
    diag_prob = (1-game["p"])/2
    prob=np.array([game["p"],diag_prob,diag_prob])
    for m in range(max_iterations):
        prev_reward = np.copy(reward_matrix)
        prev_policy=np.copy(policy)
        for i in range(game["n"]):
            for j in range(game["n"]):
                if [i, j] in game["walls"]:
                    policy[i, j] = "N"
                elif [i, j] in game["terminal_pos"]:
                    policy[i, j] = "E"
                else:
                    for k in range(4):
                        operations[k]=0
                        for l in range(3):
                            operations[k]+=(prev_reward[st_matrix[index_map[(i, j)],l, k]]*prob[l])
                    
                    reward_matrix[index_map[(i,j)]]=game["reward"] + game["gamma"] * np.max(operations)
                    policy[i, j] = actions[operations.argmax()]
                        
        if np.sum(np.fabs(prev_reward - reward_matrix)) <= epsilon  or time.time()-t>27 or np.array_equal(policy,prev_policy):
            print m+1
            break
    return policy

def policy_evaluation(game,actions,policy,prev_reward,reward_matrix,prob):
    for i in range(game["n"]):
        for j in range(game["n"]):
            if [i, j] not in game["walls"] and [i, j] not in game["terminal_pos"]:
                operations=0
                for l in range(3):
                    operations+=(prev_reward[st_matrix[index_map[(i, j)],l, actions[policy[i,j]]]]*prob[l])
                
                reward_matrix[index_map[(i,j)]]=game["reward"] + game["gamma"] * operations                    

def policy_iteration(game, index_map, reward_matrix, st_matrix,t):
    operations = np.full(4, 0, dtype=float)
    actions = ["U", "D", "R", "L"]
    action_inv={"U":0, "D":1, "R": 2, "L":3}
    # policy = np.full((game["n"], game["n"]), "U", dtype=str)
    policy = np.random.choice(actions, (game["n"], game["n"]))
    max_iterations =100000
    epsilon = 1e-3
    diag_prob = (1-game["p"])/2
    prob=np.array([game["p"],diag_prob,diag_prob])
    for m in range(max_iterations):
        prev_reward = np.copy(reward_matrix)
        prev_policy = np.copy(policy)
        policy_evaluation(game,action_inv,prev_policy,prev_reward,reward_matrix,prob)
        # print np.sum(np.fabs(prev_reward - reward_matrix))
        for i in range(game["n"]):
            for j in range(game["n"]):
                if [i, j] in game["walls"]:
                    policy[i, j] = "N"
                elif [i, j] in game["terminal_pos"]:
                    policy[i, j] = "E"
                else:
                    for k in range(4):
                        operations[k]=0
                        for l in range(3):
                            operations[k]+=(reward_matrix[st_matrix[index_map[(i, j)],l, k]]*prob[l])
                    
                    if operations[action_inv[policy[i,j]]] < np.max(operations):
                        policy[i, j] = actions[operations.argmax()]
        if np.sum(np.fabs(prev_reward - reward_matrix)) <= epsilon or time.time()-t>27 or np.array_equal(policy,prev_policy):
            break

    return policy

def print_policy(policy):
    with open("output.txt", "w") as f:
        t=""
        for i in policy:
            for j in i:
                t = t + j + ","
            t=t[:-1]+"\n"
        f.write(t[:-1])
    f.close()

if __name__ == "__main__":
    t=time.time()
    game = read_input('input6.txt')
    index_map = generate_cell_hash_table(game)
    reward_matrix = generate_reward_matrix(game, index_map)
    st_matrix = generate_state_transition_matrix(game, index_map)
    policy = value_iteration(game, index_map, reward_matrix, st_matrix, t)
    # policy = policy_iteration(game, index_map, reward_matrix, st_matrix, t)
    print_policy(policy)
    print time.time() - t