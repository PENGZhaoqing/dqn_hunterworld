import mxnet as mx


class PolicyNet():
    """
    Deterministic Multi-Layer Perceptron Policy used
    for deterministic policy training.
    """

    def __init__(self, env):
        self.env = env
        action_dim = self.env.action_space.shape[0]
        self.obs = mx.symbol.Variable("obs")
        net = mx.symbol.FullyConnected(data=self.obs, name="policy_fc1", num_hidden=32)
        net = mx.symbol.Activation(data=net, name="policy_relu1", act_type="relu")
        net = mx.symbol.FullyConnected(data=net, name="policy_fc2", num_hidden=32)
        net = mx.symbol.Activation(data=net, name="policy_relu2", act_type="relu")
        net = mx.symbol.FullyConnected(data=net, name='policy_fc3', num_hidden=action_dim)
        self.act = mx.symbol.Activation(data=net, name="act", act_type="tanh")
        # note the negative one here: the loss maximizes the average return

    def get_output_symbol(self):
        return self.act

    def define_exe(self, ctx):

        self.input_shapes = {
            "obs": (32, self.env.observation_space.shape[0])}

        # define an executor, initializer and updater for batch version
        self.exe = self.act.simple_bind(ctx=ctx, **self.input_shapes)
        self.arg_arrays = self.exe.arg_arrays
        self.grad_arrays = self.exe.grad_arrays
        self.arg_dict = self.exe.arg_dict

        init = mx.initializer.Normal()
        for name, arr in self.arg_dict.items():
            if name not in self.input_shapes:
                init(mx.init.InitDesc(name), arr)

        self.updater = mx.optimizer.get_updater(
            mx.optimizer.create('adam',
                                learning_rate=0.0001))

        # define an executor for sampled single observation
        # note the parameters are shared
        new_input_shapes = {"obs": (1, self.input_shapes["obs"][1])}
        self.exe_one = self.exe.reshape(**new_input_shapes)
        self.arg_dict_one = self.exe_one.arg_dict

    def update_params(self, grad_from_top):

        # policy accepts the gradient from the Value network
        self.exe.forward(is_train=True)
        self.exe.backward([grad_from_top])

        for i, pair in enumerate(zip(self.arg_arrays, self.grad_arrays)):
            weight, grad = pair
            self.updater(i, grad, weight)

    def get_actions(self, obs):
        # batch version
        self.arg_dict["obs"][:] = obs
        self.exe.forward(is_train=False)
        return self.exe.outputs[0].asnumpy()

    def get_action(self, obs):
        # single observation version
        self.arg_dict_one["obs"][:] = obs
        self.exe_one.forward(is_train=False)
        return self.exe_one.outputs[0].asnumpy()

class Target_PolicyNet():
    def __init__(self, policyNet, ctx):
        self.policy_target = policyNet.act.simple_bind(ctx=ctx, **policyNet.input_shapes)
        for name, arr in self.policy_target.arg_dict.items():
            if name not in policyNet.input_shapes:
                policyNet.arg_dict[name].copyto(arr)

    def getNet(self):
        return self.policy_target
