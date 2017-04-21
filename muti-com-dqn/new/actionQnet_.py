import mxnet as mx
import numpy
import mxnet.ndarray as nd
from mxnet.context import Context


class DQNOutput(mx.operator.CustomOp):
    def __init__(modQ):
        super(DQNOutput, modQ).__init__()

    def forward(modQ, is_train, req, in_data, out_data, aux):
        modQ.assign(out_data[0], req[0], in_data[0])

    def backward(modQ, req, out_grad, in_data, out_data, in_grad, aux):
        out_qvalue = out_data[0].asnumpy()
        action = in_data[1].asnumpy().astype(numpy.int)
        target = in_data[2].asnumpy()
        ret = numpy.zeros(shape=out_qvalue.shape, dtype=numpy.float32)
        ret[numpy.arange(action.shape[0]), action] = numpy.clip(
            out_qvalue[numpy.arange(action.shape[0]), action] - target, -1, 1)
        modQ.assign(in_grad[0], req[0], ret)
        # sys.exit(0)


@mx.operator.register("DQNOutput")
class DQNOutputProp(mx.operator.CustomOpProp):
    def __init__(modQ):
        super(DQNOutputProp, modQ).__init__(need_top_grad=False)

    def list_arguments(modQ):
        return ['data', 'action', 'target']

    def list_outputs(modQ):
        return ['output']

    def infer_shape(modQ, in_shape):
        data_shape = in_shape[0]
        action_shape = (in_shape[0][0],)
        target_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, action_shape, target_shape], [output_shape], []

    def create_operator(modQ, ctx, shapes, dtypes):
        return DQNOutput()


class actQnet():
    def __init__(self, actions_num, ctx, signal_num):
        self.com = mx.symbol.Variable("com")
        data = mx.symbol.Variable('data')
        data = mx.symbol.Flatten(data=data)
        fc1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=128)
        act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type="relu")
        fc2 = mx.symbol.FullyConnected(data=act1, name='fc2', num_hidden=64)
        act2 = mx.symbol.Activation(data=fc2, name='relu2', act_type="relu")
        act2 = mx.symbol.Concat(act2, self.com, name="Qnet_concat")
        self.qvalue = mx.symbol.FullyConnected(data=act2, name='qvalue', num_hidden=actions_num)
        self.dqn = mx.symbol.Custom(data=self.qvalue, name='dqn', op_type='DQNOutput')

        self.input_shapes = {
            "data": (32, 74),
            'com': (32, signal_num),
            "dqn_action": (32,),
            "dqn_target": (32,)}

        # define an executor, initializer and updater for batch version loss
        self.exe = self.dqn.simple_bind(ctx=ctx, **self.input_shapes)
        self.arg_arrays = self.exe.arg_arrays
        self.grad_arrays = self.exe.grad_arrays
        self.arg_dict = self.exe.arg_dict
        self.aux_dict = self.exe.aux_dict

        init = mx.init.Xavier(factor_type="in", magnitude=2.34)
        for name, arr in self.arg_dict.items():
            if name not in self.input_shapes:
                init(name, arr)

        self.updater = mx.optimizer.get_updater(
            mx.optimizer.create('adam', learning_rate=0.0002))

    def update_params(self, state, signal, action, targets):

        self.arg_dict["data"][:] = state
        self.arg_dict["dqn_action"][:] = action
        self.arg_dict["dqn_target"][:] = targets
        self.exe.arg_dict["com"][:] = signal

        self.exe.forward(is_train=True)
        self.exe.backward()

        for i, pair in enumerate(zip(self.arg_arrays, self.grad_arrays)):
            weight, grad = pair
            self.updater(i, grad, weight)
        return self.exe.outputs[0]


class target_actQnet():
    def __init__(self, Qnet, signal_num, ctx):
        self.intput_shapes = {"data": (32, 74),
                              'com': (32, signal_num)}
        self.exe = Qnet.qvalue.simple_bind(ctx=ctx, **self.intput_shapes)
        self.arg_dict = self.exe.arg_dict
        # parameters are not shared but initialized the same
        for name, arr in self.exe.arg_dict.items():
            if name not in Qnet.input_shapes:
                Qnet.arg_dict[name].copyto(arr)

        new_input_shapes = {"data": (1, 74),
                            'com': (1, signal_num)}
        self.exe_one = self.exe.reshape(**new_input_shapes)
        self.arg_dict_one = self.exe_one.arg_dict

    def getNet(self):
        return self.exe

    def get_qvals(self, state, signal):
        # batch version
        self.arg_dict["data"][:] = state
        self.arg_dict["com"][:] = signal

        self.exe.forward(is_train=False)
        return self.exe.outputs[0]

    def get_qval(self, state, signal):
        # single version
        self.arg_dict_one["data"][:] = state
        self.arg_dict_one["com"][:] = signal

        self.exe_one.forward(is_train=False)
        return self.exe_one.outputs[0]


class comNet():
    def __init__(self, signal_num, ctx):
        data = mx.symbol.Variable('data')
        data = mx.symbol.Flatten(data=data)
        fc1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=128)
        act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type="relu")
        fc2 = mx.symbol.FullyConnected(data=act1, name='fc2', num_hidden=64)
        act2 = mx.symbol.Activation(data=fc2, name='relu2', act_type="relu")
        signal = mx.symbol.FullyConnected(data=act2, name='signal', num_hidden=signal_num)
        self.signal = mx.symbol.Activation(data=signal, name="act", act_type="tanh")

        self.input_shapes = {"data": (32, 74)}
        self.exe = self.signal.simple_bind(ctx=ctx, **self.input_shapes)
        self.arg_arrays = self.exe.arg_arrays
        self.grad_arrays = self.exe.grad_arrays
        self.arg_dict = self.exe.arg_dict
        self.aux_dict = self.exe.aux_dict

        init = mx.init.Xavier(factor_type="in", magnitude=2.34)
        for name, arr in self.arg_dict.items():
            if name not in self.input_shapes:
                init(name, arr)

        self.updater = mx.optimizer.get_updater(
            mx.optimizer.create('adam', learning_rate=0.0002))

    def update_params(self, grad_from_top):
        # policy accepts the gradient from the Value network
        self.exe.forward(is_train=True)
        self.exe.backward([grad_from_top])

        for i, pair in enumerate(zip(self.arg_arrays, self.grad_arrays)):
            weight, grad = pair
            self.updater(i, grad, weight)


    def get_signals(self, state):
        self.exe.arg_dict['data'][:] = state
        self.exe.forward(is_train=False)
        return self.exe.outputs[0]


class target_comNet():
    def __init__(self, comNet, ctx):
        self.target_shapes = {"data": (32, 74)}
        self.exe = comNet.signal.simple_bind(ctx=ctx, **self.target_shapes)
        self.arg_dict = self.exe.arg_dict

        # parameters are not shared but initialized the same
        for name, arr in self.exe.arg_dict.items():
            comNet.arg_dict[name].copyto(arr)

        new_input_shapes = {"data": (1, 74)}
        self.exe_one = self.exe.reshape(**new_input_shapes)
        self.arg_dict_one = self.exe_one.arg_dict

    def get_singal(self, state):
        # single version
        self.arg_dict_one["data"][:] = state
        self.exe_one.forward(is_train=False)
        return self.exe_one.outputs[0]

    def get_singals(self, state):
        # batch version
        self.arg_dict["data"][:] = state
        self.exe.forward(is_train=False)
        return self.exe.outputs[0]


def soft_copy_params_to(net, target, soft_target_tau):
    for name, arr in target.arg_dict.items():
        if name not in net.input_shapes:
            arr[:] = (1.0 - soft_target_tau) * arr[:] + \
                     soft_target_tau * net.arg_dict[name][:]


def copy_params_to(net, target):
    for name, arr in target.arg_dict.items():
        net.arg_dict[name].copyto(arr)


def load_params(Net, fname):
    save_dict = nd.load(fname)
    arg_params = {}
    aux_params = {}
    for k, value in save_dict.items():
        arg_type, name = k.split(':', 1)
        if arg_type == 'arg':
            arg_params[name] = value
        elif arg_type == 'aux':
            aux_params[name] = value
        else:
            raise ValueError("Invalid param file " + fname)
    Net.arg_dict = arg_params
    Net.aux_dict = aux_params


def save_params(Net, fname, ctx):
    arg_params = Net.arg_dict
    aux_params = Net.aux_dict
    # ctx = Context('cpu', 0)
    save_dict = {('arg:%s' % k): v.as_in_context(ctx) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(ctx) for k, v in aux_params.items()})
    nd.save(fname, save_dict)
