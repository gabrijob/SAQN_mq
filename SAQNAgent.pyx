#!./venv/bin/python3

import tensorflow as tf
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Lambda
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

#LOAD_MODEL = 'saqn_models/saved/SA-84243__1616895138.model' # or None
LOAD_MODEL = None
MODELS_DIR = '/tmp/saqn_models'
LOGS_DIR =  '/tmp/saqn_logs'
CSVFILENAME = LOGS_DIR + "/episodes.log"

DISCOUNT = 0.5
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

MODEL_NAME = 'SA-84243'

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1.0  # not a constant, going to be decayed
EPSILON_DECAY = 0.99875
MIN_EPSILON = 0.001
DECAY_EPSILON_STEP = 5

#  Stats settings
AGGREGATE_STATS_EVERY = 20  # steps

# MDP defs
OBSERVATION_SPACE_VALUES = (8,)
ACTION_SPACE_SIZE = 6 #[(reduce cache, do nothing, raise cache) performance case
                        #(reduce cache, do nothing, raise cache)] cost case

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
    #OBSERVATION_SPACE_VALUES = (6,)
    #ACTION_SPACE_SIZE = 3 #(raise cache, reduce cache, do nothing)

    def __init__(self):
        # For more repetitive results
        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.set_random_seed(1)

    def reset(self):
        self.episode_step = 0

        # Make dummy state
        # [total_throughput, thpt_variation, proc_t, sche_t, msgs_to_spark, msgs_in_gb, ready_mem, spark_thresh]
        start_state = [0, 0, 0, 0, 0, 0, 0, 0]

        return start_state

    def step(self, action):
        self.episode_step += 1
        done = True
        # send action to spark
        # get new state

        # make dummy state, reward and done
        new_state = [100, 1, 1, 10, 1, 0.5, 3, 50.8]

        reward = 1
        if self.episode_step > 1_000:
            done = True
            print('finish')

        return new_state, reward, done

    def calc_reward(self, state):
        #throughput = state[0]
        thgpt_variation = state[1]
        mem_use = state[7]

        reward_p = thgpt_variation
        # upper limit: 30
        # lower limit: 4
        reward_c = 2*(4-mem_use)/(30-4)+1

        return (reward_p, reward_c)


# SAQN Agent 
class SAQNAgentInstance:
    MODEL_NAME = 'SA-84243'
    
    def __init__(self):
        # main model, it gets trained every step
        self.model = self.create_model()

        # target model, this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir = f"{LOGS_DIR}/{MODEL_NAME}-{time.time()}")

        self.target_update_counter = 0


    def create_model(self):
        model = None 
        if LOAD_MODEL is not None:
            # Load pre-trained model
            print('Loading model');
            model = load_model(LOAD_MODEL)
            print('Model loaded ' + LOAD_MODEL)
        else:
            # Stacked Autoencoder model
            model = Sequential()

            # Encoder part
            model.add(Dense(4, input_shape=OBSERVATION_SPACE_VALUES, activation="relu"))
            model.add(Dense(2, activation="relu"))

            # Decoder part
            model.add(Dense(4, activation='relu'))
            model.add(Dense(ACTION_SPACE_SIZE, activation="sigmoid"))

            # Normalize output from (0,2) to (-1,1)
            #model.add(Lambda(lambda x: x-1))

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
        new_current_states = np.array([transition[4] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward_p, reward_c, new_current_state, done) in enumerate(minibatch):
            # Use the correct reward for the case (performance or cost)
            reward = 0
            if index >= ACTION_SPACE_SIZE/2:
                reward = reward_c
            else:
                reward = reward_p

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
        if not os.path.isdir(MODELS_DIR):
            os.makedirs(MODELS_DIR)

        self.env = SparkMQEnv()
        self.agent = SAQNAgentInstance()

        if os.path.isfile(CSVFILENAME):
            with open(CSVFILENAME, "r", encoding="utf-8") as scraped:
                reader = csv.reader(scraped, delimiter=',')
                #last_episode = reader.pop()
                for last_episode in reader:
                    # Get epsilon number from file
                    self.epsilon = float(last_episode[1])

                    # Get episode number from file
                    self.episode = int(last_episode[0]) + 1

                    # Get avg reward from last saved episode
                    self.last_avg_reward = float(last_episode[2])
        else:
                self.epsilon = 1
                self.episode = 1
                self.last_avg_reward = 1

        self.agent.tensorboard.step = self.episode
        self.step_rewards = {}
        self.step_rewards['performance'] = []
        self.step_rewards['cost'] = []
        self.step = 1
        self.current_state = start_state
        self.last_action = 0
        self.epsilon_counter = 0

    
    def get_action(self):
        
        if self.epsilon_counter > DECAY_EPSILON_STEP:
             # Decay epsilon
            if self.epsilon > MIN_EPSILON:
                self.epsilon *= EPSILON_DECAY
                self.epsilon = max(MIN_EPSILON, self.epsilon)
            self.epsilon_counter = 0
        

        if np.random.random_sample() > self.epsilon:
            # Get action from Q table
            action = np.argmax(self.agent.get_qs(self.current_state))
        else:
            # Get random action (MUST BE RANDOM SO THAT IT CAN EXPLORE)
            action = np.random.randint(0,ACTION_SPACE_SIZE) 

        self.epsilon_counter += 1
        self.last_action = action
        return action

    def change_state(self, new_state, done=False):
        #new_state, reward, done = env.step(action)
        reward_p, reward_c = self.env.calc_reward(new_state)
        self.tensorboard_log(reward_p, reward_c)
        
        # Every step we update replay memory and train main network
        self.agent.update_replay_memory((self.current_state, self.last_action, reward_p, reward_c, new_state, done)) #self.last_action?
        self.agent.train(done, self.step)
        self.current_state = new_state
        self.step += 1


    def tensorboard_log(self, reward_p, reward_c):
        # Append episode reward to a list and log stats (every given number of episodes)
        self.step_rewards['performance'].append(reward_p)
        self.step_rewards['cost'].append(reward_c)

        if not self.step % AGGREGATE_STATS_EVERY or self.step == 1:
            average_reward_p = sum(self.step_rewards['performance'][-AGGREGATE_STATS_EVERY:])/len(self.step_rewards['performance'][-AGGREGATE_STATS_EVERY:])
            min_reward_p = min(self.step_rewards['performance'][-AGGREGATE_STATS_EVERY:])
            max_reward_p = max(self.step_rewards['performance'][-AGGREGATE_STATS_EVERY:])
            
            average_reward_c = sum(self.step_rewards['cost'][-AGGREGATE_STATS_EVERY:])/len(self.step_rewards['cost'][-AGGREGATE_STATS_EVERY:])
            min_reward_c = min(self.step_rewards['cost'][-AGGREGATE_STATS_EVERY:])
            max_reward_c = max(self.step_rewards['cost'][-AGGREGATE_STATS_EVERY:])
            
            self.agent.tensorboard.update_stats(
                    reward_avg_p=average_reward_p, 
                    reward_min_p=min_reward_p, 
                    reward_max_p=max_reward_p, 
                    reward_avg_c=average_reward_c,
                    reward_min_c=min_reward_c,
                    reward_max_c=max_reward_c,
                    epsilon=self.epsilon)


            # Save model, but only when min reward is greater or equal a set value
            if (average_reward_p + average_reward_c)**2 > self.last_avg_reward :
                self.agent.model.save(f'{MODELS_DIR}/{SAQNAgentInstance.MODEL_NAME}__{int(time.time())}.model')

    

    def finish(self, last_state):
        self.change_state(last_state, done=True)
    
        # Calculate average reward for the whole episode
        avg_reward_p = sum(self.step_rewards['performance'])/len(self.step_rewards['performance'])
        avg_reward_c = sum(self.step_rewards['cost'])/len(self.step_rewards['cost'])

        mean_squared_reward = (avg_reward_p + avg_reward_c)**2

        with open(CSVFILENAME, mode='a+') as episode_file:
            writer = csv.writer(episode_file, delimiter=',')

            writer.writerow([self.episode, self.epsilon, mean_squared_reward])



cdef public object createAgent(float* start_state):
    state = []
    for i in range(8):
        state.append(start_state[i])

    return AgentEpisode(state)


cdef public int infer(object agent , float* new_state):
    state = []
    for i in range(8):
        state.append(new_state[i])

    agent.change_state(state)

    action = agent.get_action()
    action = action % (ACTION_SPACE_SIZE/2) - 1

    return action

cdef public void finish(object agent, float* last_state):
    state = []
    for i in range(8):
        state.append(last_state[i])
    
    agent.finish(state)
