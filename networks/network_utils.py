from networks.U_Net import U_Net

def set_model(model_name, out_channels=3, out_layers=1):
    if model_name == 'U_Net':
        model  = U_Net(out_channels=out_channels, out_layers=out_layers)

    return model