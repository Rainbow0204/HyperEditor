import math
import torch
from torch import nn
import copy
from argparse import Namespace

from models.encoders.psp import pSp
from models.stylegan2.model import Generator
from configs.paths_config import model_paths
from models.hypereditor.backbone import BackboneNet
from utils.resnet_mapping import RESNET_MAPPING
from criteria import image_embedding_loss


import clip
import torchvision.transforms as transforms


class HyperEditor(nn.Module):

    def __init__(self, opts):
        super(HyperEditor, self).__init__()
        self.set_opts(opts)
        self.device = 'cuda'
        self.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self.hypernet = self.set_hypernet()
        self.decoder = Generator(self.opts.output_size, 512, 8, channel_multiplier=2).to(self.device)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.loss_clip = image_embedding_loss.ImageEmbddingLoss()
        self.model_clip, _ = clip.load("ViT-B/32", device="cuda")
        self.transform = transforms.Compose(
            [transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        self.clip_pool = nn.AdaptiveAvgPool2d((224, 224))
        # print(1)
        # Load weights if needed
        self.load_weights()
        if self.opts.load_w_encoder:
            self.w_encoder.eval()

        self.str = '../datasets/templates.txt'
        with open(self.str, "r") as templates:
            text_prompt_templates = templates.readlines()
        self.text_prompts_templates = text_prompt_templates

        if self.opts.choose_layers is True:
            self.delta_i, _ = self.get_delta_i([self.opts.target_text, self.opts.init_text])
            # print(self.delta_i)
            self.choose_layer = self.adaptive_layer_selectors(self.delta_i)



    def set_hypernet(self):
        if self.opts.output_size == 1024:
            self.opts.n_hypernet_outputs = 26
        elif self.opts.output_size == 512:
            self.opts.n_hypernet_outputs = 23
        elif self.opts.output_size == 256:
            self.opts.n_hypernet_outputs = 20
        else:
            raise ValueError(f"Invalid Output Size! Support sizes: [1024, 512, 256]!")
        networks = BackboneNet(opts=self.opts)

        return networks

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print(f'Loading HyperEditor from checkpoint: {self.opts.checkpoint_path}')
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.hypernet.load_state_dict(self.__get_keys(ckpt, 'hypernet'), strict=True)
            self.decoder.load_state_dict(self.__get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
            if self.opts.load_w_encoder:
                self.w_encoder = self.__get_pretrained_w_encoder()
        else:
            hypernet_ckpt = self.__get_hypernet_checkpoint()
            self.hypernet.load_state_dict(hypernet_ckpt, strict=False)
            print(f'Loading decoder weights from pretrained path: {self.opts.stylegan_weights}')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=True)
            self.__load_latent_avg(ckpt, repeat=self.n_styles)
            if self.opts.load_w_encoder:
                self.w_encoder = self.__get_pretrained_w_encoder()

    def get_delta_i(self, text_prompts):
        text_features = self._get_averaged_text_features(text_prompts)
        delta_t = text_features[0] - text_features[1]
        return delta_t, text_features[0]

    def _get_averaged_text_features(self, text_prompts):
        with torch.no_grad():
            text_features_list = []
            for text_prompt in text_prompts:
                formatted_text_prompts = [template.format(text_prompt) for template in self.text_prompts_templates]  # format with class  （79）
                formatted_text_prompts = clip.tokenize(formatted_text_prompts).cuda()  # tokenize  (79 * 77)
                text_embeddings = self.model_clip.encode_text(formatted_text_prompts)  # embed with text encoder  (79 * 512)
                text_embedding = text_embeddings.mean(dim=0)
                text_features_list.append(text_embedding)
            text_features = torch.stack(text_features_list, dim=1).cuda()
        return text_features.t()

    def adaptive_layer_selectors(self, delta_i):
        sample_z = torch.randn(self.opts.batch_size, 512, device=self.device)
        sample_z = [sample_z]
        initial_w_codes = [self.decoder.style(s) for s in sample_z]
        # print(initial_w_codes)
        initial_w_codes = initial_w_codes[0].unsqueeze(1).repeat(1, 18, 1)
        w_codes = torch.Tensor(initial_w_codes.cpu().detach().numpy()).to(self.device)
        w_codes.requires_grad = True
        w_optim = torch.optim.Adam([w_codes], lr=0.01)
        with torch.no_grad():
            initial_inversion_w = initial_w_codes.unsqueeze(0)
            initial_inversion_1 = self.decoder(initial_inversion_w, input_is_latent=True)[0]
        generated_from_w = None
        for _ in range(10):
            w_codes_for_gen = w_codes.unsqueeze(0)
            generated_from_w = self.decoder(w_codes_for_gen, input_is_latent=True)[0]
            w_loss = self.loss_clip(
                masked_generated=generated_from_w,
                masked_img_tensor=initial_inversion_1,
                delta_i=delta_i, delta=True).mean()
            w_optim.zero_grad()
            w_loss.backward()
            w_optim.step()
        layer_weights = torch.abs(w_codes - initial_w_codes).mean(dim=-1).mean(dim=0)
        w_to_weight = {
            0: 0,
            1: 2,
            2: 3,
            3: 5,
            4: 6,
            5: 8,
            6: 9,
            7: 11,
            8: 12,
            9: 14,
            10: 15,
            11: 17,
            12: 18,
            13: 20,
            14: 21,
            15: 23,
            16: 24,
        }
        with torch.no_grad():
            fine_layers = []
            for i in range(len(layer_weights) - 1):
                if layer_weights[i] >= (layer_weights.mean() + self.opts.lambda_std * layer_weights.std()) and i != 17:
                    fine_layers.append(w_to_weight[i])
        # print(fine_layers)
        return fine_layers

    def forward(self, x, style_fix=None, condition=None, delta_t=None, resize=True, input_code=False, randomize_noise=True, return_latents=False,
                return_weight_deltas_and_codes=False, weights_deltas=None, y_hat=None, codes=None):

        delta_i = None
        globe_clip = None
        choose_layer = None

        if condition is not None:
            delta_i, globe_clip = self.get_delta_i(condition)
            delta_i = delta_i.float()
            globe_clip = globe_clip.float()

        if input_code:
            codes = x
            with torch.no_grad():
                y_hat, _ = self.decoder([codes],
                                         input_is_latent=True,
                                         randomize_noise=randomize_noise,
                                         return_latents=return_latents)
        else:
            if y_hat is None:
                assert self.opts.load_w_encoder, "Cannot infer latent code when e4e isn't loaded."
                y_hat, codes = self.__get_initial_inversion(x, resize=True)

        # pass through hypernet to get per-layer deltas
        if self.opts.choose_layers is True:
            hypernet_outputs = self.hypernet(y_hat, delta_i=delta_i, weight_choose=choose_layer)
        else:
            hypernet_outputs = self.hypernet(y_hat, delta_i=delta_i, weight_choose=None)

        if weights_deltas is None:
            weights_deltas = hypernet_outputs
        else:
            weights_deltas = weights_deltas

        input_is_latent = (not input_code)

        images, result_latent = self.decoder([codes],
                                             weights_deltas=weights_deltas,
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents and return_weight_deltas_and_codes:
            return images, result_latent, weights_deltas, codes, y_hat, delta_i, globe_clip
        elif return_latents:
            return images, result_latent
        elif return_weight_deltas_and_codes:
            return images, weights_deltas, codes
        else:
            return images

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

    def __get_hypernet_checkpoint(self):
        print('Loading hypernet weights from resnet34!')
        hypernet_ckpt = torch.load(model_paths['resnet34'])
        # Transfer the RGB input of the resnet34 network to the first 3 input channels of hypernet
        if self.opts.input_nc != 3:
            shape = hypernet_ckpt['conv1.weight'].shape
            altered_input_layer = torch.randn(shape[0], self.opts.input_nc, shape[2], shape[3], dtype=torch.float32)
            altered_input_layer[:, :3, :, :] = hypernet_ckpt['conv1.weight']
            hypernet_ckpt['conv1.weight'] = altered_input_layer
        mapped_hypernet_ckpt = dict(hypernet_ckpt)
        for p, v in hypernet_ckpt.items():
            for original_name, net_name in RESNET_MAPPING.items():
                if original_name in p:
                    mapped_hypernet_ckpt[p.replace(original_name, net_name)] = v
                    mapped_hypernet_ckpt.pop(p)
        return hypernet_ckpt

    @staticmethod
    def __get_keys(d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt

    def __get_pretrained_w_encoder(self):
        print("Loading pretrained W encoder...")
        opts_w_encoder = vars(copy.deepcopy(self.opts))
        opts_w_encoder['checkpoint_path'] = self.opts.w_encoder_checkpoint_path
        opts_w_encoder['encoder_type'] = self.opts.w_encoder_type
        opts_w_encoder['input_nc'] = 3
        opts_w_encoder = Namespace(**opts_w_encoder)
        w_net = pSp(opts_w_encoder)
        w_net = w_net.encoder
        w_net.eval()
        w_net.cuda()
        return w_net

    def __get_initial_inversion(self, x, resize=True):
        # get initial inversion and reconstruction of batch
        with torch.no_grad():
            return self.__get_w_inversion(x, resize)


    def __get_w_inversion(self, x, resize=True):
        if self.w_encoder.training:
            self.w_encoder.eval()
        codes = self.w_encoder.forward(x)  #(2*18*512)

        if codes.ndim == 2:
            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        y_hat, _ = self.decoder([codes],
                                weights_deltas=None,
                                input_is_latent=True,
                                randomize_noise=False,
                                return_latents=False)
        if resize:
            y_hat = self.face_pool(y_hat)
            # print(y_hat.shape)
        if "cars" in self.opts.dataset_type:
            y_hat = y_hat[:, :, 32:224, :]
        return y_hat, codes

