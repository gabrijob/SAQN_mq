#!./venv/bin/python3

import numpy as np
import csv
import random
import os
import time
import array

from cpython cimport array

from SAQNAgent import SparkMQEnv, SAQNAgent

CSVFILENAME = "episodes.log"

EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
MIN_REWARD = -200  # For model save

class AgentEpisode:
    def __init__(self, start_state):

        # Create models folder
        if not os.path.isdir('models'):
            os.makedirs('models')

        self.env = SparkMQEnv()
        self.agent = SAQNAgent()

        with open(CSVFILENAME, "r", encoding="utf-8") as scraped:
            reader = csv.reader(scraped, delimiter=',')
            last_episode = reader[-1]
        
            # Get epsilon number from file
            self.epsilon = last_episode[1]

            # Get episode number from file
            self.episode = last_episode[0] + 1
            self.agent.tensorboard.step = self.episode

        self.episode_reward = 0
        self.step = 1
        self.current_state = start_state

    # JUNTAR get_action() COM change_state()??????????????
    def get_action(self):

        if np.random.random_sample() > self.epsilon:
            # Get action from Q table
            action = np.argmax(self.agent.get_qs(self.current_state))
        else:
            # Get random action (MUST BE RANDOM SO THAT IT CAN EXPLORE)
            action = np.random.randint(0, self.env.ACTION_SPACE_SIZE)

        self.last_action = action
        return action

    def change_state(self, new_state, done=False):
        #new_state, reward, done = env.step(action)
        reward = self.env.calc_reward(new_state)
        
        # Transform new continous state to new discrete state and count reward
        self.episode_reward += reward

        # Every step we update replay memory and train main network
        self.agent.update_replay_memory((self.current_state, self.last_action, reward, new_state, done)) #self.last_action?
        self.agent.train(done, self.step)
        self.current_state = new_state
        self.step += 1

    def done(self, last_state):
        self.change_state(last_state, done=True)

        ep_rewards = []
        # Read list from file
        with open(CSVFILENAME, "r", encoding="utf-8") as scraped:
            reader = csv.reader(scraped, delimiter=',')

            for row in reader:
                ep_rewards.append(row[2])

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(self.episode_reward)
        if not self.episode % AGGREGATE_STATS_EVERY or self.episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            self.agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=self.epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if average_reward >= MIN_REWARD:
                self.agent.model.save(f'models/{SAQNAgent.MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)


        with open(CSVFILENAME, mode='w') as episode_file:
            writer = csv.writer(episode_file, delimiter=',')

            writer.writerow([self.agent.tensorboard.step, self.epsilon, self.episode_reward])



cdef public object createAgent(char* start_state):
    state = array.array('f', start_state)
    return AgentEpisode(state)


cdef public int infer(object agent , char* new_state):
    state = array.array('f', new_state)
    
    agent.change_state(state)

    action = agent.get_action()

    return action

cdef public void done(object agent, char* last_state):
    state = array.array('f', last_state)
    agent.done(state)

