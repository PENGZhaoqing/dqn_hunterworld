import mxnet as mx


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


def dqn_sym_nature(action_num, predict=False):
    data = mx.sym.Variable('data')
    yInput = mx.sym.Variable('yInput')
    actionInput = mx.sym.Variable('actionInput')
    net = mx.symbol.Convolution(data=data, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=32)
    net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=64)
    net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv3', kernel=(3, 3), stride=(1, 1), num_filter=64)
    net = mx.symbol.Activation(data=net, name='relu3', act_type="relu")
    net = mx.symbol.Flatten(data=net)
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=512)
    net = mx.symbol.Activation(data=net, name='relu4', act_type="relu")
    Qvalue = mx.symbol.FullyConnected(data=net, name='fc5', num_hidden=action_num)
    temp = actionInput * Qvalue
    coeff = mx.sym.sum(temp, axis=1, name='temp1')
    output = mx.sym.square(coeff - yInput)
    loss = mx.sym.MakeLoss(output)
    if predict:
        return Qvalue
    else:
        return loss


class DQNInitializer(mx.init.Xavier):
    def _init_bias(self, _, arr):
        arr[:] = 0.1


def createQNetwork(actions_num, q_ctx, bef_args=None, isTrain=True, batch_size=32):
    if isTrain:
        modQ = mx.mod.Module(symbol=dqn_sym_nature(actions_num), data_names=('data', 'actionInput'),
                             label_names=('yInput',),
                             context=q_ctx)
        batch = batch_size
        modQ.bind(data_shapes=[('data', (batch, 4, 84, 84)), ('actionInput', (batch, actions_num))],
                  label_shapes=[('yInput', (batch,))],
                  for_training=True)

        modQ.init_params(initializer=mx.initializer.Xavier(factor_type="in",
                                                           magnitude=2.34),
                         arg_params=bef_args)

        modQ.init_optimizer(
            optimizer='adam',
            optimizer_params={
                'learning_rate': 0.0002,
                'wd': 0.0,
                'beta1': 0.5,
                # 'beta2': 0.999,
                # 'epsilon': 1e-08,
                # 'rescale_grad': 1.0,
                # 'clip_gradient': None
            })
        # modQ.init_optimizer(
        #     optimizer='adagrad',
        #     optimizer_params={
        #         'learning_rate': 0.01,
        # 'wd': 0.0,
        # 'eps': 0.01,
        # 'rescale_grad': 1.0
        # 'clip_gradient': None
        # })
    else:
        modQ = mx.mod.Module(symbol=dqn_sym_nature(actions_num, predict=True), data_names=('data',), label_names=None,
                             context=q_ctx)
        batch = 1
        modQ.bind(data_shapes=[('data', (batch, 4, 84, 84))],
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
