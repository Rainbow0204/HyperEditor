from argparse import ArgumentParser

from configs.paths_config import model_paths


class TestOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # arguments for inference script
        self.parser.add_argument('--exp_dir', type=str, default=None,
                                 help='Path to experiment output directory')
        self.parser.add_argument('--checkpoint_path', default=None,
                                 type=str,
                                 help='Path to HyperEditor model checkpoint')
        self.parser.add_argument('--data_path', type=str, default=None,
                                 help='Path to directory of images to evaluate')
        self.parser.add_argument('--resize_outputs', action='store_true',
                                 help='Whether to resize outputs to 256x256 or keep at original output resolution')
        self.parser.add_argument('--test_batch_size', default=2, type=int,
                                 help='Batch size for testing and inference')
        self.parser.add_argument('--test_workers', default=4, type=int,
                                 help='Number of test/inference dataloader workers')
        self.parser.add_argument('--n_images', type=int, default=None,
                                 help='Number of images to output. If None, run on all data')
        self.parser.add_argument('--save_weight_deltas', default=True,
                                 help='Whether to save the weight deltas of each image. Note: file weighs about 200MB.')

        # arguments for loading pre-trained encoder
        self.parser.add_argument('--load_w_encoder', default=True, help='Whether to load the w e4e encoder.')
        self.parser.add_argument('--w_encoder_checkpoint_path', default=model_paths["faces_w_encoder"], type=str,
                                 help='Path to pre-trained W-encoder.')
        self.parser.add_argument('--w_encoder_type', default='WEncoder',
                                 help='Encoder type for the encoder used to get the initial inversion')

        # text conditions
        self.parser.add_argument('--init_text', default='face', type=str,
                                 help='Initial text conditions.')
        self.parser.add_argument('--target_text', default='face with smile', type=str,
                                 help='Target text condition.')
        self.parser.add_argument('--target_text_file', default=None, type=str,
                                 help='Path to the txt file for the target text condition.')



    def parse(self):
        opts = self.parser.parse_args()
        return opts
