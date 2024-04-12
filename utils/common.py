from PIL import Image
import matplotlib.pyplot as plt


def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def vis_faces(log_hooks):
    display_count = len(log_hooks)
    n_outputs = len(log_hooks[0]['output_face']) if type(log_hooks[0]['output_face']) == list else 1
    fig = plt.figure(figsize=(8 + (n_outputs * 2), 3 * display_count))
    gs = fig.add_gridspec(display_count, 2)
    for i in range(display_count):
        hooks_dict = log_hooks[i]
        fig.add_subplot(gs[i, 0])
        vis_faces_iterative(hooks_dict, fig, gs, i)
    plt.tight_layout()
    return fig


def vis_faces_with_id(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['input_face'])
    plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
    fig.add_subplot(gs[i, 1])
    plt.imshow(hooks_dict['target_face'])
    plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
                                                     float(hooks_dict['diff_target'])))
    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict['output_face'])
    plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))


def vis_faces_iterative(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['w_inversion'])
    plt.title('Input')
    output_image, similarity = hooks_dict['output_face'][0]
    fig.add_subplot(gs[i, 1])
    plt.imshow(output_image)
    plt.title('Target')

