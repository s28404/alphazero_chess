import gym
import gym_chess
import random

env = gym.make('Chess-v0')
print(env.render())

env.reset()
done = False

while not done:
    action = random.sample(env.legal_moves)
    action_space = env.legal_moves
    observation_space = env.board.fen()
    print(f"Action Space: {action_space}")
    print(f"Observation Space: {observation_space}")
    env.step(action)
    print(env.render(mode='unicode'))

env.close()
