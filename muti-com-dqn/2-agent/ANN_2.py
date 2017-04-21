import mxnet as mx
import numpy
from mxnet.module.base_module import _parse_data_desc
from mxnet.module.module import Module
from mxnet.module.executor_group import DataParallelExecutorGroup
import mxnet.ndarray as nd


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


def sym(actions_num, predict=False):
    data = mx.sym.Variable('data')
    data = mx.sym.Flatten(data=data)
    fc1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=420)
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
        modQ = mx.mod.Module(symbol=sym(actions_num), data_names=('data', 'dqn_action', 'dqn_target'), label_names=None,
                             context=q_ctx)
        bind(modQ,
             data_shapes=[('data', (batch_size, 1, 74)), ('dqn_action', (batch_size,)), ('dqn_target', (batch_size,))],
             for_training=True)

        modQ.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), arg_params=bef_args)
        modQ.init_optimizer(
            optimizer='adam',
            optimizer_params={
                'learning_rate': 0.0002
            })
    else:
        modQ = mx.mod.Module(symbol=sym(actions_num, predict=True), data_names=('data',), label_names=None,
                             context=q_ctx)
        bind(modQ, data_shapes=[('data', (batch_size, 1, 74))],
             for_training=False)
        modQ.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), arg_params=bef_args)
    return modQ


def createQNetwork1(actions_num, q_ctx, bef_args=None, isTrain=True, batch_size=32):
    if isTrain:
        modQ = mx.mod.Module(symbol=sym(actions_num), data_names=('data', 'dqn_action', 'dqn_target'), label_names=None,
                             context=q_ctx)
        bind(modQ,
             data_shapes=[('data', (batch_size, 1, 78)), ('dqn_action', (batch_size,)), ('dqn_target', (batch_size,))],
             for_training=True)

        modQ.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), arg_params=bef_args)
        modQ.init_optimizer(
            optimizer='adam',
            optimizer_params={
                'learning_rate': 0.0002
            })
    else:
        modQ = mx.mod.Module(symbol=sym(actions_num, predict=True), data_names=('data',), label_names=None,
                             context=q_ctx)
        bind(modQ, data_shapes=[('data', (batch_size, 1, 78))],
             for_training=False)
        modQ.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), arg_params=bef_args)
    return modQ


def copyTargetQNetwork(Qnet, target):
    arg_params, aux_params = Qnet.get_params()
    target.set_params(arg_params=arg_params, aux_params=aux_params)


def bind(modQ, data_shapes, label_shapes=None, for_training=True,
         inputs_need_grad=False, force_rebind=False, shared_module=None,
         grad_req='write'):
    if force_rebind:
        modQ._reset_bind()

    if modQ.binded:
        modQ.logger.warning('Already binded, ignoring bind()')
        return

    modQ.for_training = for_training
    modQ.inputs_need_grad = inputs_need_grad
    modQ.binded = True
    modQ._grad_req = grad_req

    if not for_training:
        assert not inputs_need_grad
    else:
        pass
        # this is not True, as some module might not contains a loss function
        # that consumes the labels
        # assert label_shapes is not None

    modQ._data_shapes, modQ._label_shapes = _parse_data_desc(
        modQ.data_names, modQ.label_names, data_shapes, label_shapes)

    if shared_module is not None:
        assert isinstance(shared_module, Module) and \
               shared_module.binded and shared_module.params_initialized
        shared_group = shared_module._exec_group
    else:
        shared_group = None

    modQ._exec_group = DataParallelExecutorGroup(modQ._symbol, modQ._context,
                                                 modQ._work_load_list, modQ._data_shapes,
                                                 modQ._label_shapes, modQ._param_names,
                                                 for_training, inputs_need_grad,
                                                 shared_group, logger=modQ.logger,
                                                 fixed_param_names=modQ._fixed_param_names,
                                                 grad_req=grad_req,
                                                 state_names=modQ._state_names)
    modQ._total_exec_bytes = modQ._exec_group._total_exec_bytes
    if shared_module is not None:
        modQ.params_initialized = True
        modQ._arg_params = shared_module._arg_params
        modQ._aux_params = shared_module._aux_params
    elif modQ.params_initialized:
        # if the parameters are already initialized, we are re-binding
        # so automatically copy the already initialized params
        modQ._exec_group.set_params(modQ._arg_params, modQ._aux_params)
    else:
        assert modQ._arg_params is None and modQ._aux_params is None
        param_arrays = [
            nd.zeros(x[0].shape, dtype=x[0].dtype, ctx=x[0][0].context)
            for x in modQ._exec_group.param_arrays
            ]
        modQ._arg_params = {name: arr for name, arr in zip(modQ._param_names, param_arrays)}

        aux_arrays = [
            nd.zeros(x[0].shape, dtype=x[0].dtype, ctx=x[0][0].context)
            for x in modQ._exec_group.aux_arrays
            ]
        modQ._aux_params = {name: arr for name, arr in zip(modQ._aux_names, aux_arrays)}

    if shared_module is not None and shared_module.optimizer_initialized:
        modQ.borrow_optimizer(shared_module)
