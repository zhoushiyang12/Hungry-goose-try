import os
import webbrowser
from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, GreedyAgent,greedy_agent

env = make("hungry_geese", debug=True)

env.reset()
# env.run(['submission.py', 'submission_rule.py', 'submission_rule.py', 'submission_rule.py'])
env.run([ 'dqn_3plan.py', 'smart_reinforcement_learning.py','smart_reinforcement_learning.py', 'smart_reinforcement_learning.py'])    #白\蓝\green\red
render = env.render(mode="html")
render = render.replace('"', "&quot;")

path = os.path.abspath("temp.html")
with open(path, "w") as f:
    f.write(f'<iframe srcdoc="{render}" width="800" height="600"></iframe>')

webbrowser.open("file://" + path)