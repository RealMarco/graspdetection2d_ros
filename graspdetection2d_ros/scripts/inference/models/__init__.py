def get_network(network_name):
    network_name = network_name.lower()
    # Original GR-ConvNet
    if network_name == 'grconvnet':
        from .grconvnet import GenerativeResnet
        return GenerativeResnet
    # Configurable GR-ConvNet with multiple dropouts
    elif network_name == 'grconvnet2':
        from .grconvnet2 import GenerativeResnet
        return GenerativeResnet
    # Configurable GR-ConvNet with dropout at the end
    elif network_name == 'grconvnet3':
        from .grconvnet3 import GenerativeResnet
        return GenerativeResnet
    # GR-ConvNet without BatchNorm
    elif network_name == 'grconvnet3_nobn':
        from .grconvnet3_nobn import GenerativeResnet
        return GenerativeResnet
    # Inverted GR-ConvNet
    elif network_name == 'grconvnet4':
        from .grconvnet4 import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'gcotnet':
        from .gcotnet import GenerativeCoTNet
        return GenerativeCoTNet
    elif network_name == 'gcotnext':
        from .gcotnext import GenerativeCoTNeXt
        return GenerativeCoTNeXt
    elif network_name == 'gsecotnetd':
        from .gsecotnetd import GenerativeSECoTNetD
        return GenerativeSECoTNetD
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
