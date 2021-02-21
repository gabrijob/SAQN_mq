#!./venv/bin/python3

import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from collections import deque
import time
import random
from tqdm import tqdm
import os
import csv
import array

from cpython cimport array

CSVFILENAME = "episodes.log"

DISCOUNT = 0.5
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

MIN_REWARD = -200  # For model save
MODEL_NAME = 'SA-64243'

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes

# MDP defs
OBSERVATION_SPACE_VALUES = (6,)
ACTION_SPACE_SIZE = 3 #(raise cache, reduce cache, do nothing)

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model
        self._train_dir = self.log_dir + '/train'
        #pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

     # added for performance?
    def on_train_batch_end(self, _, __):
        pass

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


# Class for fetching data from the Spark message queue environment
class SparkMQEnv:
    # MDP defs
    OBSERVATION_SPACE_VALUES = (6,)
    ACTION_SPACE_SIZE = 3 #(raise cache, reduce cache, do nothing)

    def __init__(self):
        # For more repetitive results
        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.set_random_seed(1)

    def reset(self):
        self.episode_step = 0

        # Make dummy state
        # [total_throughput, proc_t, sche_t, msgs_in_gb, ready_mem, spark_thresh]
        start_state = [0, 0, 0, 0, 0, 0]

        return start_state

    def step(self, action):
        self.episode_step += 1
        done = True
        # send action to spark
        # get new state

        # make dummy state, reward and done
        new_state = [100, 1, 1, 10, 1, 0.5]

        reward = 1
        if self.episode_step > 1_000:
            done = True
            print('finish')

        return new_state, reward, done

    def calc_reward(self, state):
        throughput = state[0]
        delay = state[1] + state[2]

        reward = throughput

        return reward


# SAQN Agent 
class SAQNAgentInstance:
    MODEL_NAME = 'SA-64243'
    
    def __init__(self):
        # main model, it gets trained every step
        self.model = self.create_model()

        # target model, this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir = f"logs/{MODEL_NAME}-{time.time()}")

        self.target_update_counter = 0


    def create_model(self):
        #Stacked Autoencode model
        model = Sequential()

        # Encoder part
        model.add(Dense(4, input_shape=OBSERVATION_SPACE_VALUES, activation="relu"))
        model.add(Dense(2, activation="relu"))

        # Decoder part
        model.add(Dense(4, activation='relu'))
        model.add(Dense(ACTION_SPACE_SIZE, activation="sigmoid"))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    
    def get_qs(self, state):
        state_array = np.array(state)
        return self.model.predict(state_array.reshape(-1, *state_array.shape))[0]

    
    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)


        X_train = np.asarray(X).astype(np.float32)
        y_train = np.asarray(y).astype(np.float32)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(
            X_train, 
            y_train, 
            batch_size=MINIBATCH_SIZE, 
            verbose=0, 
            shuffle=False, 
            callbacks=[self.tensorboard] if terminal_state else None
            )

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


class AgentEpisode:
    def __init__(self, start_state):

        # Create models folder
        if not os.path.isdir('models'):
            os.makedirs('models')

        self.env = SparkMQEnv()
        self.agent = SAQNAgentInstance()

        if os.path.isfile(CSVFILENAME):
            with open(CSVFILENAME, "r", encoding="utf-8") as scraped:
                reader = csv.reader(scraped, delimiter=',')
                last_episode = reader[-1]
            
                # Get epsilon number from file
                self.epsilon = last_episode[1]

                # Get episode number from file
                self.episode = last_episode[0] + 1
                self.agent.tensorboard.step = self.episode
        else:
            self.epsilon = 1
            self.episode = 1
            self.agent.tensorboard.step = self.episode

        self.episode_reward = 0
        self.step = 1
        self.current_state = start_state
        self.last_action = 0

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

    def finish(self, last_state):
        self.change_state(last_state, done=True)

        ep_rewards = []
        # Read list from file
        if os.path.isfile(CSVFILENAME):
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
                self.agent.model.save(f'models/{SAQNAgentInstance.MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)


        with open(CSVFILENAME, mode='w') as episode_file:
            writer = csv.writer(episode_file, delimiter=',')

            writer.writerow([self.agent.tensorboard.step, self.epsilon, self.episode_reward])



cdef public object createAgent(float* start_state):
    state = []
    for i in range(6):
        state.append(start_state[i])

    return AgentEpisode(state)


cdef public int infer(object agent , float* new_state):
    state = []
    for i in range(6):
        state.append(new_state[i])

    agent.change_state(state)

    action = agent.get_action()

    return action

cdef public void finish(object agent, float* last_state):
    state = []
    for i in range(6):
        state.append(last_state[i])
    
    agent.finish(state)
