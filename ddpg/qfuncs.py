import mxnet as mx

class Qnet():

    def __init__(self, env):
        self.env = env
        self.obs = mx.symbol.Variable("obs")
        self.act = mx.symbol.Variable("act")
        net = mx.symbol.FullyConnected(data=self.obs, name="Qnet_fc1", num_hidden=32)
        net = mx.symbol.Activation(data=net, name="Qnet_relu1", act_type="relu")
        net = mx.symbol.FullyConnected(data=net, name="Qnet_fc2", num_hidden=32)
        net = mx.symbol.Activation(data=net, name="Qnet_relu2", act_type="relu")
        net = mx.symbol.Concat(net, self.act, name="Qnet_concat")
        net = mx.symbol.FullyConnected(data=net, name="Qnet_fc3", num_hidden=32)
        net = mx.symbol.Activation(data=net, name="Qnet_relu3", act_type="relu")
        self.qval = mx.symbol.FullyConnected(data=net, name="Qnet_qval", num_hidden=1)
        self.yval = mx.symbol.Variable("yval")
        loss = 1.0 / 32 * mx.symbol.sum(mx.symbol.square(self.qval - self.yval))
        loss = mx.symbol.MakeLoss(loss, name="Qnet_loss")
        self.loss = mx.symbol.Group([loss, mx.symbol.BlockGrad(self.qval)])
        print self.loss.list_arguments()

    def get_output_symbol(self):
        return self.qval

    def define_exe(self, ctx):

        self.input_shapes = {
            "obs": (32, self.env.observation_space.shape[0]),
            "act": (32, self.env.action_space.shape[0]),
            "yval": (32, 1)}

        # define an executor, initializer and updater for batch version loss
        self.exe = self.loss.simple_bind(ctx=ctx, **self.input_shapes)
        self.arg_arrays = self.exe.arg_arrays
        self.grad_arrays = self.exe.grad_arrays
        self.arg_dict = self.exe.arg_dict

        init = mx.initializer.Normal()
        for name, arr in self.arg_dict.items():
            if name not in self.input_shapes:
                init(mx.init.InitDesc(name), arr)

        self.updater = mx.optimizer.get_updater(
            mx.optimizer.create('adam', learning_rate=0.001))

    def update_params(self, obs, act, yval):

        self.arg_dict["obs"][:] = obs
        self.arg_dict["act"][:] = act
        self.arg_dict["yval"][:] = yval

        self.exe.forward(is_train=True)
        self.exe.backward()

        for i, pair in enumerate(zip(self.arg_arrays, self.grad_arrays)):
            weight, grad = pair
            self.updater(i, grad, weight)

    def get_qvals(self, obs, act):

        self.exe.arg_dict["obs"][:] = obs
        self.exe.arg_dict["act"][:] = act
        self.exe.forward(is_train=False)

        return self.exe.outputs[1].asnumpy()


class Target_Qnet():
    def __init__(self, env, Qnet, ctx):
        self.env = env
        self.target_shapes = {
            "obs": (32, self.env.observation_space.shape[0]),
            "act": (32, self.env.action_space.shape[0])
        }
        self.Qnet_target = Qnet.qval.simple_bind(ctx=ctx,
                                                 **self.target_shapes)
        # parameters are not shared but initialized the same
        for name, arr in self.Qnet_target.arg_dict.items():
            if name not in Qnet.input_shapes:
                Qnet.arg_dict[name].copyto(arr)

    def getNet(self):
        return self.Qnet_target
