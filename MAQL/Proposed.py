from pathlib import Path
import os

# Batch launcher sets these; direct execution defaults to job_1 42.
RUN_SEED = int(os.environ.get("RL_SEED", "50"))
SCRIPT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = Path(os.environ.get("RL_RESULTS_ROOT", "proposed")).expanduser()
if not RESULTS_ROOT.is_absolute():
    RESULTS_ROOT = SCRIPT_ROOT / RESULTS_ROOT
RESULTS_ROOT = RESULTS_ROOT.resolve()
Q_TABLE_PATH = RESULTS_ROOT / f"seed{RUN_SEED}" / "Q_table.npy"


import networkx as nx
import numpy as np

import math

scale = 5350 *2.4

def routing(nodes_num,start_node_n,end_node_n):

    reward_n=[]
    M_n=[]
    path=[]
    shortest_path=[]

    reward=0
    M=0
    learning_rate=0.8
    discount_factor=0.95

    # seed1 = 8968
    #
    # seed1 = RUN_SEED
    seed1 = 50


    G = nx.random_geometric_graph(nodes_num, 0, seed=seed1)
    added_user_num =0
    step = 1
    if added_user_num != 0:
        H = nx.random_geometric_graph(step * added_user_num, 0, seed=seed1 + 10)
        mapping = dict(zip(H, range(10, 10 + step * added_user_num)))
        H = nx.relabel_nodes(H, mapping)
        U = nx.compose(G, H)
    else:
        U = G

    pos = nx.get_node_attributes(U, "pos")
    for nodeG in range(10 + step * added_user_num):
        for nodeH in range(10 + step * added_user_num):
            if nodeH != nodeG:
                distance = [pos[nodeG][0] - pos[nodeH][0], pos[nodeG][1] - pos[nodeH][1]]
                d = math.hypot(distance[0], distance[1])
                if d < 0.6:
                    U.add_edge(nodeG, nodeH)


    nodes_num = U.number_of_nodes()
    one_hop_error = np.zeros((nodes_num, nodes_num))
    for node1 in range(nodes_num):
        for node2 in range(nodes_num):
            if U.has_edge(node1, node2):
                distance = [pos[node1][0] - pos[node2][0], pos[node1][1] - pos[node2][1]]
                d = math.hypot(distance[0], distance[1])
                # scale = 5350
                d = d * scale
                pi = 3.
                lamda = 3/ 25
                np.random.seed(4)
                a = (lamda / (4 * pi * d))
                snr = (a) ** 2 * 0.1 / 1e-13

                # print(snr)
                snr2 = math.sqrt(snr)
                error = 0.5 * math.erfc(snr2)
                if error < 0.0000001:
                    error = 0.000001
                if error>0.009:
                    error=0.005
                # error=0.0003
                # one_hop_error[node1][node2] = 1-(1-error) ** (93520)  #2
                one_hop_error[node1][node2] = 1 - (1 - error) ** (153520)  #1
                # one_hop_error[node1][node2] = 1 - (1 - error) ** (553520)  #0
                Weight = -math.log2(one_hop_error[node1][node2])
                # print(error)
                U[node1][node2].update({"weight": Weight})

    import random
    random.seed(0)
    for node1 in range(10):
        for node2 in range(10):
            if one_hop_error[node1][node2] == 0:
                # one_hop_error[node1][node2] = random.uniform(0, 0.3)
                one_hop_error[node1][node2] = random.uniform(0, 0.6)
                # one_hop_error[node1][node2] = random.uniform(0.5, 0.9)
                # if node1==4  or node2==4:
                #     one_hop_error[node1][node2]=random.uniform(0, 0.2)

    if added_user_num!=0:
        for node1 in range(10 + step * added_user_num):
            for node2 in range(10 + step * added_user_num):
                if one_hop_error[node1][node2] == 0:
                    one_hop_error[node1][node2] = random.uniform(0, 0.00001)
    else:
        pass
    E = np.ones((nodes_num, nodes_num), dtype='float64')
    one_hop_error=E-one_hop_error


    for node1 in range(nodes_num):
        for node2 in range(nodes_num):
            if node1==node2:
                one_hop_error[node1][node2]=1

    q = np.zeros((nodes_num,nodes_num))

    edge_distances = np.zeros((nodes_num,nodes_num))
    edge_distances = list(edge_distances)
    edge_distances = np.array(edge_distances)
    edge_distances = edge_distances.astype(float)

    for node1 in range(nodes_num):
        for node2 in range(nodes_num):
            if one_hop_error[node1][node2]==0:
                edge_distances[node1][node2]=-1
            else:
                edge_distances[node1][node2]=-math.log2(one_hop_error[node1][node2])



    for node1 in range(nodes_num):
        for node2 in range(nodes_num):
            if one_hop_error[node1][node2]==0:
                q[node1][node2]=-9999.00
            if node1==node2:
                q[node1][node2] = -999.00
                edge_distances[node1][node2] = -1

    print(edge_distances)
    q = list(q)
    q = np.array(q)
    q_table = [[q for i in range(10)] for j in range(10)]
    q_table = list(q_table)
    q_table = np.array(q_table)

    for i in range(10):
        for j in range(10):
            q_table[i][j][end_node_n[i][j], end_node_n[i][j]]=1



    Q_table = np.load(Q_TABLE_PATH)
    for episode_n in range(50000):
        episode_n = int(episode_n)
        r_m=np.ones((10,10))
        epsilon = 0.001 + (0.95 - 0.01) * 0.9998 ** (episode_n)

        print(episode_n)
        object_node_n = [[None for j in range(10)] for i in range(10)]
        obs_n = start_node_n
        new_obs_n = start_node_n
        reward_n.append(reward)
        reward = 0
        M_n.append(M)
        M = 0
        reward2 = [[1 for i in range(10)] for i in range(10)]
        reward0=[[0 for i in range(10)] for i in range(10)]
        reward3 = [[0 for i in range(10)] for i in range(10)]

        path.append(shortest_path)
        shortest_path = [[[] for i in range(10)] for i in range(10)]  # 7个客户端，每个客户端有6个智能体
        for i in range(10):
            for j, node in enumerate(obs_n[i]):
                shortest_path[i][j].append(node)

        l_n = [0 for i in range(nodes_num)]  # 在每一轮次中记录每个节点的转发次数，累次叠加
        r2 = 0
        while (object_node_n != end_node_n):  # 所有客户端的所有智能体都到达对应的目标节点
            model_rm = np.ones((nodes_num, nodes_num))

            L_n = [0 for i in range(nodes_num)]
            k_n = [0 for i in range(
                nodes_num)]

            M += 1

            reward1 = [[] for i in range(10)]
            obs_n = new_obs_n
            new_obs_n = [[] for i in range(10)]
            print(object_node_n)
            for i in range(10):
                for j, current_node in enumerate(obs_n[i]):
                    if object_node_n[i][j] == end_node_n[i][j] or  current_node == end_node_n[i][j]:
                        action = current_node

                    # choose action
                    else:
                        if np.random.uniform(0, 1) < epsilon:
                            # Explore
                            possible_actions = np.where(edge_distances[current_node, :] >=0)[0]
                            if len(possible_actions) == 0:
                                break
                            action = np.random.choice(possible_actions)
                        else:
                            # Exploit
                            possible_actions = \
                            np.where(q_table[i][j][current_node, :] == np.max(q_table[i][j][current_node, :]))[0]
                            if len(possible_actions) == 0:
                                break
                            action = np.random.choice(possible_actions)

                    new_obs_n[i].append(action)  # 获得智能体的下一位置




            for i in range(10):
                for j, next_node in enumerate(new_obs_n[i]):
                    j = int(j)
                    next_node = int(next_node)
                    shortest_path[i][j].append(next_node)
                    if next_node != obs_n[i][j]:
                        L_n[obs_n[i][j]] += 1
                    if obs_n[i][j]==next_node:
                        reward0[i][j]=0
                    else:
                         reward0[i][j]=-edge_distances[obs_n[i][j], next_node]


            for i, node in enumerate(L_n):
                if node != 0 :
                    l_n[i] += 1

            for i in range(10):
                for j, next_node in enumerate(new_obs_n[i]):

                    reward1[i].append(reward0[i][j])
                    reward2[i][j] *= (one_hop_error[obs_n[i][j], next_node])


            # #网络拥塞控制

            kk_n=[[0 for j in range(nodes_num)]for i in range(nodes_num)]
            K_n=[[[]for j in range(nodes_num)]for i in range(nodes_num)]
            for i in range(10):  # 计算当前节点接收需要转发智能体的个数，并记录对应的客户端的智能体
                visited_nodes=set()

                for j, next_node in enumerate(new_obs_n[i]):
                    i = int(i)
                    j = int(j)
                    next_node = int(next_node)
                    if next_node != end_node_n[i][j]:
                        kk_n[next_node][i] += Q_table[i][j][obs_n[i][j], next_node]
                        K_n[next_node][i].append(
                            [i, j])

                        if next_node not in visited_nodes:
                            k_n[next_node] += 1  # 记录第next_node个节点接收来自同一个客户端数据包的个数
                            visited_nodes.add(next_node)
                          # 记录了每个发生网络拥塞的节点上所有智能体Q值总和（将来自同一个客户端的看作是一个整体）


            for node1 in range(nodes_num):  #遍历节点
                KK = []
                KKK_n = []
                KKK_nn = []
                for node2 in range(nodes_num):#客户端
                    if k_n[node1] >2:  # 节点发生网络拥塞
                        if kk_n[node1][node2] != 0:
                            for m, (i, j) in enumerate(K_n[node1][node2]):
                                # reward1[i][j] += -0.2/(len(K_n[node1][node2])**len(K_n[node1][node2]))#拥塞
                                # reward1[i][j]+=-8
                                model_rm[i][j] =2.9999999/ k_n[node1]
                                if r_m[i][j] >2.9999999/ k_n[node1]:
                                    r_m[i, :] = 2.9999999 / k_n[node1]

                    if k_n[node1] >2:  # 节点发生网络拥塞
                        if kk_n[node1][node2]!=0:
                            KKK_n.append(kk_n[node1][node2])
                            KKK_nn.append(node2)

                while len(KKK_n) >2:
                    min_value = min(KKK_n)
                    min_idx = KKK_n.index(min_value)
                    node = KKK_nn[min_idx]
                    KK.append(K_n[node1][node])  # 表示需要存储的智能体
                    KKK_n.remove(min_value)
                    KKK_nn.remove(node)
                # for node in range(len(KK)):#网络拥塞控制
                #     for m, (i, j) in enumerate(KK[node]):
                #         reward1[i][j] +=-0.1


            for i in range(10):
                for j, next_node in enumerate(new_obs_n[i]):
                    j = int(j)
                    next_node = int(next_node)
                    if obs_n[i][j]==next_node:
                        reward1[i][j]+=0
                    else:
                         reward1[i][j]+=3*math.log2(model_rm[i][j])


            for i in range(10):
                for j, next_node in enumerate(new_obs_n[i]):
                    if obs_n[i][j] == end_node_n[i][j] or object_node_n[i][j]==end_node_n[i][j]:  # 已经到达目标节点的智能体不再更新Q表
                        pass
                    else:
                       q_table[i][j][obs_n[i][j], next_node] = (1 - learning_rate) * q_table[i][j][
                            obs_n[i][j], next_node] + learning_rate * (reward1[i][j] + discount_factor * np.max(
                            q_table[i][j][next_node, :]))



            for i in range(10):
                for j, next_node in enumerate(new_obs_n[i]):
                    j = int(j)
                    next_node = int(next_node)
                    if next_node == end_node_n[i][j]:  # 如果下一节点为目标节点，则将其加入目标节点中
                        object_node_n[i][j] = next_node
                    else:
                        pass



            if object_node_n == end_node_n:

                for i in range(10):

                    for j in range(10):

                        if i==j:
                            r_m[i][j]=1
                        reward += r_m[i][j]*( reward2[i][j] )# 计算所有客户端的智能体的路由信息之和


                break

    reward_n.remove(0)

    min_value1 = min(reward_n)
    min_idx1 = reward_n.index(min_value1)
    # print(min_value1)
    # print(path[min_idx1 + 1])
    # print(q_table)
    pathbest=path[49999]


    # Define the common path prefix
    output_dir = RESULTS_ROOT / f"seed{RUN_SEED}" / "proposed"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = str(output_dir) + "/"

    # Create a dictionary to store filenames and corresponding data
    file_data = {
        'reward.txt': reward_n,
        'r_m.txt': r_m,
        'one_hop_error':one_hop_error
    }


    # Write text files
    for filename, data in file_data.items():
        with open(base_path + filename, "w") as f:
            if isinstance(data, np.ndarray):  # For numpy arrays
                np.savetxt(f, data, fmt='%.12f', delimiter=',')
            else:  # For other types (like float or string)
                f.write(str(data))

    # Save the numpy array for pathbest
    np.save(base_path + 'pathbest.npy', pathbest)
    # np.save(base_path + 'Q_table.npy', q_table)
    # logging.info(f"routing_bad one_hop_error: {one_hop_error}")

    return one_hop_error, pathbest


x = [[0 for i in range(10)], [1 for i in range(10)], [2 for i in range(10)], [3 for i in range(10)],
     [4 for i in range(10)], [5 for i in range(10)], [6 for i in range(10)], [7 for i in range(10)],
     [8 for i in range(10)], [9 for i in range(10)], ]
y = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]


one_hop_error, pathbest = routing(10, x, y)





