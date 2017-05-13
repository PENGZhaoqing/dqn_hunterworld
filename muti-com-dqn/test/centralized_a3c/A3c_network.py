import mxnet as mx


class A3c(object):
    def __init__(self, state_dim, act_space, config):
        self.state_dim = state_dim
        self.num_envs = config.num_envs
        self.config = config

        state = mx.sym.Variable('state')
        net = mx.sym.FullyConnected(
            data=state, name='fc1', num_hidden=200, no_bias=True)
        net = mx.sym.Activation(data=net, name='relu1', act_type="relu")

        policy_fc1 = mx.sym.FullyConnected(
            data=net, name='policy_fc1', num_hidden=act_space, no_bias=True)
        policy1 = mx.sym.SoftmaxActivation(data=policy_fc1, name='policy1')
        policy1 = mx.sym.clip(data=policy1, a_min=1e-5, a_max=1 - 1e-5)
        log_policy1 = mx.sym.log(data=policy1, name='log_policy1')
        out_policy1 = mx.sym.BlockGrad(data=policy1, name='out_policy1')

        policy_fc2 = mx.sym.FullyConnected(
            data=net, name='policy_fc2', num_hidden=act_space, no_bias=True)
        policy2 = mx.sym.SoftmaxActivation(data=policy_fc2, name='policy2')
        policy2 = mx.sym.clip(data=policy2, a_min=1e-5, a_max=1 - 1e-5)
        log_policy2 = mx.sym.log(data=policy2, name='log_policy2')
        out_policy2 = mx.sym.BlockGrad(data=policy2, name='out_policy2')

        # Negative entropy.
        neg_entropy1 = policy1 * log_policy1
        neg_entropy1 = mx.sym.MakeLoss(
            data=neg_entropy1, grad_scale=config.entropy_wt, name='neg_entropy1')

        # Negative entropy.
        neg_entropy2 = policy2 * log_policy2
        neg_entropy2 = mx.sym.MakeLoss(
            data=neg_entropy2, grad_scale=config.entropy_wt, name='neg_entropy2')

        # Value network.
        value = mx.sym.FullyConnected(data=net, name='value', num_hidden=1)
        self.sym = mx.sym.Group([log_policy1, log_policy2, value, neg_entropy1, neg_entropy2, out_policy1, out_policy2])
        self.model = mx.mod.Module(self.sym, data_names=('state',),
                                   label_names=None)

        self.paralell_num = config.num_envs * config.t_max
        self.model.bind(
            data_shapes=[('state', (self.paralell_num, state_dim))],
            label_shapes=None, inputs_need_grad=True,
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
        self.model.reshape([('state', (len(state), self.state_dim))])
        data_batch = mx.io.DataBatch(
            data=[mx.nd.array(state, ctx=self.config.ctx)], label=None)
        self.model.forward(data_batch, is_train=is_train)
        return self.model.get_outputs()
