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


modQ = mx.mod.Module(symbol=sym(4, False), data_names=('data', 'actionInput'), label_names=('yInput',),
                     context=mx.gpu(0))
modQ.bind(data_shapes=[('data', (32, 4, 80, 80)), ('actionInput', (32, 4))],
          label_shapes=[('yInput', (32,))],
          for_training=True)

print modQ._context
for k, v in modQ._arg_params.items():
    print("   %s: %s" % (k, v))
