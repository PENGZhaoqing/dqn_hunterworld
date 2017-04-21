import mxnet as mx

from mxnet.module.base_module import _parse_data_desc
from mxnet.module.module import Module
from mxnet.module.executor_group import DataParallelExecutorGroup
import mxnet.ndarray as nd


def sym(actions_num, predict=False):
    data = mx.sym.Variable('data')
    yInput = mx.sym.Variable('yInput')
    actionInput = mx.sym.Variable('actionInput')
    conv1 = mx.sym.Convolution(data=data, kernel=(8, 8), stride=(4, 4), pad=(2, 2), num_filter=32, name='conv1')
    relu1 = mx.sym.Activation(data=conv1, act_type='relu', name='relu1')
    pool1 = mx.sym.Pooling(data=relu1, kernel=(2, 2), stride=(2, 2), pool_type='max', name='pool1')
    conv2 = mx.sym.Convolution(data=pool1, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=64, name='conv2')
    relu2 = mx.sym.Activation(data=conv2, act_type='relu', name='relu2')
    conv3 = mx.sym.Convolution(data=relu2, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=64, name='conv3')
    relu3 = mx.sym.Activation(data=conv3, act_type='relu', name='relu3')
    flat = mx.sym.Flatten(data=relu3, NameError='flat')
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=512, name='fc1')
    relu4 = mx.sym.Activation(data=fc1, act_type='relu', name='relu4')
    Qvalue = mx.sym.FullyConnected(data=relu4, num_hidden=actions_num, name='qvalue')
    temp = actionInput * Qvalue
    coeff = mx.sym.sum(temp, axis=1, name='temp1')
    output = mx.sym.square(coeff - yInput)
    loss = mx.sym.MakeLoss(output)

    if predict:
        return Qvalue
    else:
        return loss


def createQNetwork(actions_num, q_ctx, bef_args=None, isTrain=True, batch_size=32):
    if isTrain:
        modQ = mx.mod.Module(symbol=sym(actions_num), data_names=('data', 'actionInput'), label_names=('yInput',),
                             context=q_ctx)
        batch = batch_size
        modQ.bind( data_shapes=[('data', (batch, 4, 80, 80)), ('actionInput', (batch, actions_num))],
             label_shapes=[('yInput', (batch,))],
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
        batch = 1
        modQ.bind( data_shapes=[('data', (batch, 4, 80, 80))],
             for_training=False)
        modQ.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), arg_params=bef_args)

    return modQ


def copyTargetQNetwork(Qnet, target):
    arg_params, aux_params = Qnet.get_params()
    # arg={}
    # for k,v in arg_params.iteritems():
    #    arg[k]=arg_params[k].asnumpy()

    target.init_params(initializer=None, arg_params=arg_params, aux_params=aux_params, force_init=True)
    # args,auxs=self.target.get_params()
    # arg1={}
    # for k,v in args.iteritems():
    #    arg1[k]=args[k].asnumpy()
    # print 'time to copy'


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
