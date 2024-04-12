import torch


def run_inversion(inputs, net, opts, return_intermediate_results=False, delta_t=None, condition=None, length=None, weights_delta=None):
    y_hat, latent, weights_deltas, codes, init_y_hat = None, None, None, None, None
    if return_intermediate_results:
        results_batch = {idx: [] for idx in range(inputs.shape[0])}
        results_latent = {idx: [] for idx in range(inputs.shape[0])}
        results_deltas = {idx: [] for idx in range(inputs.shape[0])}
        results_init = {idx: [] for idx in range(inputs.shape[0])}
    else:
        results_batch, results_latent, results_deltas, results_init = None, None, None, None

    for iter in range(length):
        y_hat, latent, weights_deltas, codes, init_y_hat = None, None, None, None, None
        y_hat, latent, weights_deltas, codes, init_y_hat, _, _ = net.forward(inputs,
                                                                 y_hat=y_hat,
                                                                 condition=condition[iter],
                                                                 codes=codes,
                                                                 weights_deltas=weights_deltas,
                                                                 return_latents=True,
                                                                 resize=opts.resize_outputs,
                                                                 randomize_noise=False,
                                                                 return_weight_deltas_and_codes=True)
        if "cars" in opts.dataset_type:
            if opts.resize_outputs:
                y_hat = y_hat[:, :, 32:224, :]
            else:
                y_hat = y_hat[:, :, 64:448, :]

        if return_intermediate_results:
            store_intermediate_results(results_batch, results_latent, results_deltas, results_init, y_hat, latent, weights_deltas, init_y_hat)

        # resize input to 256 before feeding into next iteration
        if "cars" in opts.dataset_type:
            y_hat = torch.nn.AdaptiveAvgPool2d((192, 256))(y_hat)
        else:
            y_hat = net.face_pool(y_hat)

    if return_intermediate_results:
        return results_batch, results_latent, results_deltas, results_init
    return y_hat, latent, weights_deltas, codes


def store_intermediate_results(results_batch, results_latent, results_deltas, results_init, y_hat, latent, weights_deltas, init_y_hat):
    for idx in range(y_hat.shape[0]):
        results_batch[idx].append(y_hat[idx])
        results_init[idx].append(init_y_hat[idx])
        results_latent[idx].append(latent[idx].cpu().numpy())
        results_deltas[idx].append([w[idx].cpu().numpy() if w is not None else None for w in weights_deltas])
