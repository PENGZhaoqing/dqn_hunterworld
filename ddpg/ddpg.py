from replay_mem import ReplayMem
from ddpg_utils import discount_return, sample_rewards
import logging as logger
import mxnet as mx
import numpy as np
from qfuncs import Qnet, Target_Qnet
from policies import PolicyNet, Target_PolicyNet
from strategies import OUStrategy
import sys


class DDPG(object):
    def __init__(
            self,
            env,
            ctx=mx.gpu(0),
            batch_size=32,
            n_epochs=1000,
            epoch_length=1000,
            memory_size=1000000,
            memory_start_size=1000,
            discount=0.99,
            max_path_length=1000,
            eval_samples=10000,
            soft_target_tau=1e-3,
            n_updates_per_sample=1,
            include_horizon_terminal=False,
            seed=12345):

        mx.random.seed(seed)
        np.random.seed(seed)

        self.env = env
        self.ctx = ctx
        self.strategy = OUStrategy(env)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.memory_size = memory_size
        self.memory_start_size = memory_start_size
        self.discount = discount
        self.max_path_length = max_path_length
        self.eval_samples = eval_samples
        self.soft_target_tau = soft_target_tau
        self.n_updates_per_sample = n_updates_per_sample
        self.include_horizon_terminal = include_horizon_terminal

        self.init_net()

        # logging
        self.qfunc_loss_averages = []
        self.policy_loss_averages = []
        self.q_averages = []
        self.y_averages = []
        self.strategy_path_returns = []

    def init_net(self):

        # Qnet init
        self.Qnet = Qnet(self.env)
        self.Qnet.define_exe(ctx=self.ctx)

        # qfunc_target init
        self.target_Qnet = Target_Qnet(env=self.env, Qnet=self.Qnet, ctx=self.ctx).getNet()

        # policyNet init
        self.policyNet = PolicyNet(self.env)
        self.policyNet.define_exe(ctx=self.ctx)

        # policyNet network and q-value network are combined to backpropage
        # gradients from the policyNet loss
        # since the loss is different, yval is not needed
        args = {}
        for name, arr in self.Qnet.arg_dict.items():
            if name != "yval":
                args[name] = arr

        args_grad = {}
        policy_grad_dict = dict(zip(self.Qnet.loss.list_arguments(), self.Qnet.exe.grad_arrays))
        for name, arr in policy_grad_dict.items():
            if name != "yval":
                args_grad[name] = arr

        loss = -1.0 / 32 * mx.symbol.sum(self.Qnet.qval)
        policy_loss = mx.symbol.MakeLoss(loss, name="policy_loss")

        self.policy_executor = policy_loss.bind(
            ctx=self.ctx,
            args=args,
            args_grad=args_grad,
            grad_req="write")

        self.policy_executor_grad_dict = dict(zip(
            policy_loss.list_arguments(),
            self.policy_executor.grad_arrays))

        # policy_target init
        # target policyNet only needs to produce actions, not loss
        # parameters are not shared but initialized the same
        self.target_policyNet = Target_PolicyNet(self.policyNet, ctx=self.ctx).getNet()

    def train(self):

        memory = ReplayMem(
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0],
            memory_size=self.memory_size)

        itr = 0
        path_length = 0
        path_return = 0
        end = False
        obs = self.env.reset()

        for epoch in xrange(self.n_epochs):
            print("epoch #%d | " % epoch)
            print("Training started")
            for _ in range(self.epoch_length):
                # run the policyNet
                if end:
                    # reset the environment and stretegy when an episode terminal_flags
                    obs = self.env.reset()
                    self.strategy.reset()
                    # self.policyNet.reset()
                    self.strategy_path_returns.append(path_return)
                    path_length = 0
                    path_return = 0
                # note action is sampled from the policyNet not the target policyNet
                act = self.strategy.get_action(obs, self.policyNet)
                nxt, rwd, end, _ = self.env.step(act.flatten())

                path_length += 1
                path_return += rwd

                if not end and path_length >= self.max_path_length:
                    end = True
                else:
                    memory.add_sample(obs, act, rwd, end)

                obs = nxt

                if memory.size >= self.memory_start_size:
                    for update_time in xrange(self.n_updates_per_sample):
                        batch = memory.get_batch(self.batch_size)
                        self.do_update(itr, batch)

                itr += 1

            print("Training finished")
            if memory.size >= self.memory_start_size:
                self.evaluate(epoch)

    def do_update(self, itr, batch):

        states, actions, rewards, terminal_flags, nextstate = batch

        self.target_policyNet.arg_dict["obs"][:] = nextstate
        self.target_policyNet.forward(is_train=False)
        next_actions = self.target_policyNet.outputs[0].asnumpy()

        self.target_Qnet.arg_dict["obs"][:] = nextstate
        self.target_Qnet.arg_dict["act"][:] = next_actions
        self.target_Qnet.forward(is_train=False)
        next_qvals = self.target_Qnet.outputs[0].asnumpy()

        # executor accepts 2D tensors
        rewards = rewards.reshape((-1, 1))
        terminal_flags = terminal_flags.reshape((-1, 1))
        ys = rewards + (1.0 - terminal_flags) * self.discount * next_qvals

        # since policy_executor shares the grad arrays with Qnet
        # the update order could not be changed


        self.Qnet.update_params(states, actions, ys)

        # in update values all computed
        # no need to recompute qfunc_loss and qvals
        Qnet_loss = self.Qnet.exe.outputs[0].asnumpy()
        qvals = self.Qnet.exe.outputs[1].asnumpy()

        # print self.policy_executor_grad_dict["act"][:5].asnumpy()
        policy_actions = self.policyNet.get_actions(states)
        self.policy_executor.arg_dict["obs"][:] = states
        self.policy_executor.arg_dict["act"][:] = policy_actions
        self.policy_executor.forward(is_train=True)
        policy_loss = self.policy_executor.outputs[0].asnumpy()
        self.policy_executor.backward()

        # print "before fc1_weight"
        # print self.policyNet.exe.grad_dict['policy_fc1_weight'].asnumpy()[:1]
        # print "com grad"
        # print self.policy_executor_grad_dict["act"].asnumpy()[:1]
        self.policyNet.update_params(self.policy_executor_grad_dict["act"])
        # print "updated fc1_weight"
        # print self.policyNet.exe.grad_dict['policy_fc1_weight'].asnumpy()[:1]
        # sys.exit()

        # update target networks
        for name, arr in self.target_policyNet.arg_dict.items():
            if name not in self.policyNet.input_shapes:
                arr[:] = (1.0 - self.soft_target_tau) * arr[:] + \
                         self.soft_target_tau * self.policyNet.arg_dict[name][:]
        for name, arr in self.target_Qnet.arg_dict.items():
            if name not in self.Qnet.input_shapes:
                arr[:] = (1.0 - self.soft_target_tau) * arr[:] + \
                         self.soft_target_tau * self.Qnet.arg_dict[name][:]

        self.qfunc_loss_averages.append(Qnet_loss)
        self.policy_loss_averages.append(policy_loss)
        self.q_averages.append(qvals)
        self.y_averages.append(ys)

    def evaluate(self, epoch):

        if epoch == self.n_epochs - 1:
            print("Collecting samples for evaluation")
            rewards = sample_rewards(env=self.env,
                                     policy=self.policyNet,
                                     eval_samples=self.eval_samples,
                                     max_path_length=self.max_path_length)
            average_discounted_return = np.mean(
                [discount_return(reward, self.discount) for reward in rewards])
            returns = [sum(reward) for reward in rewards]

        all_qs = np.concatenate(self.q_averages)
        all_ys = np.concatenate(self.y_averages)

        average_qfunc_loss = np.mean(self.qfunc_loss_averages)
        average_policy_loss = np.mean(self.policy_loss_averages)

        print('Epoch', epoch)
        if epoch == self.n_epochs - 1:
            print('AverageReturn', np.mean(returns))
            print('StdReturn', np.std(returns))
            print('MaxReturn', np.max(returns))
            print('MinReturn', np.min(returns))
            print('AverageDiscountedReturn', average_discounted_return)
        if len(self.strategy_path_returns) > 0:
            print('AverageEsReturn', np.mean(self.strategy_path_returns))
            print('StdEsReturn', np.std(self.strategy_path_returns))
            print('MaxEsReturn', np.max(self.strategy_path_returns))
            print('MinEsReturn', np.min(self.strategy_path_returns))
        print('AverageQLoss', average_qfunc_loss)
        print('AveragePolicyLoss', average_policy_loss)
        print('AverageQ', np.mean(all_qs))
        print('AverageAbsQ', np.mean(np.abs(all_qs)))
        print('AverageY', np.mean(all_ys))
        print('AverageAbsY', np.mean(np.abs(all_ys)))
        print('AverageAbsQYDiff',
              np.mean(np.abs(all_qs - all_ys)))

        self.qfunc_loss_averages = []
        self.policy_loss_averages = []
        self.q_averages = []
        self.y_averages = []
        self.strategy_path_returns = []


if __name__ == '__main__':
    from ddpg import DDPG
    from ddpg_utils import SEED
    import mxnet as mx

    # set environment, policy, qfunc, strategy
    import gym

    env = gym.make('BipedalWalker-v2')

    # set the training algorithm and train

    algo = DDPG(
        env=env,
        ctx=mx.gpu(0),
        max_path_length=100,
        epoch_length=1000,
        memory_start_size=100,
        n_epochs=1000,
        discount=0.99,
        seed=SEED)

    algo.train()
