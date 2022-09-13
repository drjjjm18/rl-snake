from tf_agents.replay_buffers import tf_uniform_replay_buffer 
from tf_agents.trajectories import trajectory
import os

class train_driver:
    
    def __init__(self, env, agent, metrics=None, log_interval=50, eval_interval=250, checkpoint_interval=1000, checkpoint_dir='checkpoints/checkpoint'):
        
        self.env=env
        self.agent=agent
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                                            batch_size=env.batch_size,
                                                                            max_length=1000000)
        replay_buffer_observer = self.replay_buffer.add_batch
        
        observers = [replay_buffer_observer]
        
    def compute_avg_return(self, num_episodes=10):
        total_return = 0.0
        for _ in range(num_episodes):

            time_step = self.env.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = self.agent.policy.action(time_step)
                time_step = self.env.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]
    
    def complete_episode(self):
        time_step = self.env.reset()
        i = 0
        while not time_step.is_last():
            time_step = self.env.current_time_step()
            action_step = self.agent.policy.action(time_step)
            next_step = self.env.step(action_step.action)
            data = trajectory.from_transition(time_step,
                                              action_step,
                                              next_step)
            self.replay_buffer.add_batch(data)
            i+=1
        print('episode length: ', i)
        
    def run(self, iterations):
        
        returns = []
        losses=[]
        dataset = self.replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=16,
                num_steps=2).prefetch(3)
        
        iterator = iter(dataset)

        avg_return = self.compute_avg_return()
        returns.append(avg_return)
        
        time_step = self.env.reset()
        self.agent.train_step_counter.assign(0)

        for episode in range(iterations):
            print(episode)
            self.complete_episode()
            #time_step, _ = self.driver.run(time_step)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = self.agent.train(experience).loss
            losses.append(train_loss)
            step = self.agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % self.eval_interval == 0:
                avg_return = self.compute_avg_return()
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)
                
            if step % self.checkpoint_interval == 0:
                self.save_checkpoint(episode)
   
        avg_return = self.compute_avg_return()
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
        self.save_checkpoint(episode)
                
        return returns, losses
    
    def save_checkpoint(self, episode):
        self.agent.save_checkpoint(episode)
        
    def load_checkpoint(self, path):
        self.agent.load_checkpoint(episode)
        
        
        
