from itertools import chain
import mxnet as mx
import numpy as np


class A3c(object):
    def __init__(self, state_dim, act_space, config):
        self.state_dim = state_dim
        self.num_envs = config.num_envs
        self.act_space = act_space
        self.config = config

        # Shared network.
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(
            data=net, name='fc1', num_hidden=config.hidden_size, no_bias=True)
        net = mx.sym.Activation(data=net, name='relu1', act_type="relu")

        # Policy network.
        policy_fc = mx.sym.FullyConnected(
            data=net, name='policy_fc', num_hidden=act_space, no_bias=True)
        policy = mx.sym.SoftmaxActivation(data=policy_fc, name='policy')
        policy = mx.sym.clip(data=policy, a_min=1e-5, a_max=1 - 1e-5)
        log_policy = mx.sym.log(data=policy, name='log_policy')
        out_policy = mx.sym.BlockGrad(data=policy, name='out_policy')

        # Negative entropy.
        neg_entropy = policy * log_policy
        neg_entropy = mx.sym.MakeLoss(
            data=neg_entropy, grad_scale=config.entropy_wt, name='neg_entropy')

        # Value network.
        value = mx.sym.FullyConnected(data=net, name='value', num_hidden=1)

        self.sym = mx.sym.Group([log_policy, value, neg_entropy, out_policy])
        self.model = mx.mod.Module(self.sym, data_names=('data',),
                                   label_names=None)

        self.paralell_num = config.num_envs * config.t_max
        self.model.bind(
            data_shapes=[('data', (self.paralell_num, state_dim))],
            label_shapes=None,
            grad_req="write")

        self.model.init_params(config.init_func)

        optimizer_params = {'learning_rate': config.learning_rate,
                            'rescale_grad': 1.0}
        if config.grad_clip:
            optimizer_params['clip_gradient'] = config.clip_magnitude

        self.model.init_optimizer(
            kvstore='local', optimizer=config.update_rule,
            optimizer_params=optimizer_params)

    def forward(self, state, is_train):
        self.model.reshape([('data', state.shape)])
        data_batch = mx.io.DataBatch(
            data=[mx.nd.array(state, ctx=self.config.ctx)], label=None)
        self.model.forward(data_batch, is_train=is_train)
        return self.model.get_outputs()

def compute_adv(action_num, action, neg_advs_v, q_ctx):
    neg_advs_np = np.zeros((len(neg_advs_v), action_num), dtype=np.float32)
    action = np.array(list(chain.from_iterable(action)))
    neg_advs_np[np.arange(neg_advs_np.shape[0]), action] = neg_advs_v
    neg_advs = mx.nd.array(neg_advs_np, ctx=q_ctx)
    return neg_advs
