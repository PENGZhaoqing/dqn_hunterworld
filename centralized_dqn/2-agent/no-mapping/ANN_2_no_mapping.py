import mxnet as mx
import numpy
import sys


class DQNOutput(mx.operator.CustomOp):
    def __init__(self):
        super(DQNOutput, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        out_qvalue = out_data[0].asnumpy()
        action = in_data[1].asnumpy().astype(numpy.int)
        target1 = in_data[2].asnumpy()
        target2 = in_data[3].asnumpy()
        # target3 = in_data[4].asnumpy()
        ret = numpy.zeros(shape=out_qvalue.shape, dtype=numpy.float32)

        ret[numpy.arange(action.shape[0]), action[:, 0]] = numpy.clip(
            out_qvalue[numpy.arange(action.shape[0]), action[:, 0]] - target1, -1, 1)

        ret[numpy.arange(action.shape[0]), action[:, 1] + 4] = numpy.clip(
            out_qvalue[numpy.arange(action.shape[0]), action[:, 1] + 4] - target2, -1, 1)
        # print action[:, 0]
        # print target2
        # print ret
        # sys.exit(0)
        # ret[numpy.arange(action.shape[0]), action[:, 2] + 8] = numpy.clip(
        #     out_qvalue[numpy.arange(action.shape[0]), action[:, 2] + 8] - target3, -1, 1)

        self.assign(in_grad[0], req[0], ret)

@mx.operator.register("DQNOutput")
class DQNOutputProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(DQNOutputProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'action', 'target1', 'target2']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        action_shape = in_shape[1]
        target1_shape = (in_shape[0][0],)
        target2_shape = (in_shape[0][0],)
        # target3_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, action_shape, target1_shape, target2_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return DQNOutput()


def sym(actions_num, predict=False):
    data = mx.sym.Variable('data')
    data = mx.sym.Flatten(data=data)
    fc1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=256)
    act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type="relu")
    fc2 = mx.symbol.FullyConnected(data=act1, name='fc2', num_hidden=64)
    act2 = mx.symbol.Activation(data=fc2, name='relu2', act_type="relu")
    Qvalue = mx.symbol.FullyConnected(data=act2, name='qvalue', num_hidden=actions_num)
    Dqn = mx.symbol.Custom(data=Qvalue, name='dqn', op_type='DQNOutput')
    if predict:
        return Qvalue
    else:
        return Dqn


def createQNetwork(actions_num, q_ctx, bef_args=None, isTrain=True, batch_size=32):
    if isTrain:
        modQ = mx.mod.Module(symbol=sym(actions_num),
                             data_names=('data', 'dqn_action', "dqn_target1", "dqn_target2",),
                             label_names=None,
                             context=q_ctx)
        modQ.bind(
            data_shapes=[('data', (batch_size, 1, 148)), ('dqn_action', (batch_size, 2)),
                         ('dqn_target1', (batch_size,)),
                         ('dqn_target2', (batch_size,))],
            for_training=True)
        modQ.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), arg_params=bef_args)
        modQ.init_optimizer(
            optimizer='adam',
            optimizer_params={
                'learning_rate': 0.0002,
                'wd': 0.,
                'beta1': 0.5,
            })
    else:
        modQ = mx.mod.Module(symbol=sym(actions_num, predict=True), data_names=('data',), label_names=None,
                             context=q_ctx)
        modQ.bind(data_shapes=[('data', (batch_size, 1, 148))],
                  for_training=False)
        modQ.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), arg_params=bef_args)

    return modQ


def copyTargetQNetwork(Qnet, target):
    arg_params, aux_params = Qnet.get_params()
    target.init_params(initializer=None, arg_params=arg_params, aux_params=aux_params, force_init=True)
