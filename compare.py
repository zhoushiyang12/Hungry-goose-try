import collections

import kaggle_environments
from kaggle_environments import evaluate, make, utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

kaggle_environments.__version__

list_names = [
    "dqn_3plan",
    # "simple_toward", 
    # "greedy",
    "risk_averse_greedy",
    # "simple_bfs",
    # "straightforward_bfs",
    # "boilergoose",
    "crazy_goose",
    # "smart_reinforcement_learning",
]

list_agents = [agent_name + ".py" for agent_name in list_names]

def one_on_one_with_two_simple(agents):
    n_agents = len(agents)

    scores = np.zeros((n_agents, n_agents), dtype=np.int)
    
    print("Simulation of battles. It can take some time...")

    for ind_1 in range(n_agents):
        for ind_2 in range(ind_1 + 1, n_agents):
            print(
                f"LOG: {agents[ind_1]} vs {agents[ind_2]} vs 2 X simple_toward", 
                end="\r"
            )

            current_score = evaluate(
                "hungry_geese", 
                [
                    agents[ind_1], 
                    agents[ind_2], 
                    "simple_toward.py", 
                    "simple_toward.py",
                ],
                num_episodes=100,
            )
            # print('current_score',current_score)
            episode_winners = np.argmax(current_score, axis=1)
            episode_winner_counts = collections.Counter(episode_winners)

            scores[ind_1, ind_2] = episode_winner_counts.get(0, 0)
            scores[ind_2, ind_1] = episode_winner_counts.get(1, 0)

        print()
    
    return scores

def one_against_three(agents):
    n_agents = len(agents)

    scores = np.zeros((n_agents, n_agents), dtype=np.int)
    
    print("Simulation of battles. It can take some time...")

    for ind_1 in range(n_agents):
        for ind_2 in range(n_agents):
            if ind_1 == ind_2:continue
            print(
                f"LOG: {agents[ind_1]} vs 3 X {agents[ind_2]}", 
                end="\r"
            )

            current_score = evaluate(
                "hungry_geese", 
                [
                    agents[ind_1], 
                    agents[ind_2], 
                    agents[ind_2], 
                    agents[ind_2], 
                ],
                num_episodes=100,
            )

            episode_winners = np.argmax(current_score, axis=1)
            episode_winner_counts = collections.Counter(episode_winners)

            scores[ind_1, ind_2] = episode_winner_counts.get(0, 0)
        print()
    
    return scores

def two_against_two(agents):
    n_agents = len(agents)

    scores = np.zeros((n_agents, n_agents), dtype=np.int)
    
    print("Simulation of battles. It can take some time...")

    for ind_1 in range(n_agents):
        for ind_2 in range(ind_1 + 1, n_agents):
            print(
                f"LOG: 2 X{agents[ind_1]} vs 2 X {agents[ind_2]}", 
                end="\r"
            )

            current_score = evaluate(
                "hungry_geese", 
                [
                    agents[ind_1], 
                    agents[ind_1], 
                    agents[ind_2], 
                    agents[ind_2], 
                ],
                num_episodes=100,
            )

            episode_winners = np.argmax(current_score, axis=1)
            episode_winner_counts = collections.Counter(episode_winners)

            scores[ind_1, ind_2] += episode_winner_counts.get(0, 0) + episode_winner_counts.get(1, 0)
            scores[ind_2, ind_1] += episode_winner_counts.get(2, 0) + episode_winner_counts.get(3, 0)
        print()
    
    return scores

def visualize_scores(scores, x_agents, y_agents, title, name = "fig"):
    df_scores = pd.DataFrame(
        scores, 
        index=x_agents, 
        columns=y_agents,
    )


    plt.figure(figsize=(5, 5))
    sn.heatmap(
        df_scores, annot=True, cbar=False, 
        cmap='coolwarm', linewidths=1, 
        linecolor='black', fmt="d",
    )
    plt.xticks(rotation=90, fontsize=15)
    plt.yticks(rotation=0, fontsize=15)
    plt.title(title, fontsize=18)
    
    plt.savefig(fname="{}.png".format(name) )
    # plt.show()

scores = one_on_one_with_two_simple(list_agents)
visualize_scores(scores, list_names, list_names,
 "Number of wins: one on one with two simple",
 'one_on_one_with_two_simple2')

scores = one_against_three(list_agents)
visualize_scores(
    scores, 
    list_names, 
    list(map(lambda x: "3 X " + x, list_names)),
    "Number of wins: one against three",
    'one_against_three2'
)
print("over!~")
scores = two_against_two(list_agents)
visualize_scores(
    scores, 
    list(map(lambda x: "2 X " + x, list_names)),
    list(map(lambda x: "2 X " + x, list_names)),
    "Number of wins: two_against_two2",
    'two_against_two2'
)
print("over!~")