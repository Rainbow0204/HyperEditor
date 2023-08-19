# HyperEditor: Achieving Both Authenticity and Cross-Domain Capability in Image Editing via Hypernetworks

<br>

## Getting Started

### Prerequisites

- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
- Python 3

### Installation

- Dependencies:
  We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/).
  All dependencies for defining the environment are provided in `environment/hyperEditor_env.yaml`.

## Auxiliary Models

In addition, we provide various auxiliary models needed for training your own HyperEditor models from scratch.
These include the pretrained [e4e](https://github.com/omertov/encoder4editing) encoders into W, pretrained StyleGAN2 generators, and models used for loss computation.
<br>

### Pretrained W-Encoders

| Path                                                         | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [Faces W-Encoder](https://drive.google.com/file/d/1B_HV65_hpoGwh3-NVGU1NDFNvwP8bYgi/view?usp=sharing) | Pretrained e4e encoder trained on FFHQ into the W latent space. |

<br>

### StyleGAN2 Generator

| Path                                                         | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [FFHQ StyleGAN](https://drive.google.com/file/d/1AWtD8uflxvUwcyUSNrgtWXI42jTlvVuJ/view?usp=sharing) | StyleGAN2 model trained on FFHQ with 1024x1024 output resolution. |

Note: all StyleGAN models are converted from the official TensorFlow models to PyTorch using the conversion script from [rosinality](https://github.com/rosinality/stylegan2-pytorch).

<br>

### Other Utility Models

| Path                                                         | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [IR-SE50 Model](https://drive.google.com/file/d/1zJ5m-A1O8bL_pBFBTTPBOho2JJtNdkaH/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our ID loss and encoder backbone on human facial domain. |
| [ResNet-34 Model](https://drive.google.com/file/d/1wr__Me6XDUa4Z9eBp6iuIDXqusJWscH6/view?usp=sharing) | ResNet-34 model trained on ImageNet taken from [torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) for initializing our encoder backbone. |
| [CurricularFace Backbone](https://drive.google.com/file/d/16G0R88jvfbg2z9-K1yzCWmEdX_IfgSjI/view?usp=sharing) | Pretrained CurricularFace model taken from [HuangYG123](https://github.com/HuangYG123/CurricularFace) for use in ID similarity metric computation. |
| [MTCNN](https://drive.google.com/file/d/1vJAMpUvovGi3mSIiKbwCqqVbISyqbrpO/view?usp=sharing) | Weights for MTCNN model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in ID similarity metric computation. (Unpack the tar.gz to extract the 3 model weights.) |

By default, we assume that all auxiliary models are downloaded and saved to the directory `pretrained_models`.
However, you may use your own paths by changing the necessary values in `configs/path_configs.py`.
<br>

## Preparing your Data

In order to train HyperEditor on your own data, you should perform the following steps:

1. You need to get the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset and the [Celeba-HQ](https://github.com/tkarras/progressive_growing_of_gans) dataset
2. Update `configs/paths_config.py` with the necessary data paths and model paths for training and inference.

```
dataset_paths = {
    'ffhq': '/home/asus/stuFile/FFHQ/FFHQ',
    'celeba_test': '/home/asus/stuFile/celeba_hq'
}
```

3. Configure a new dataset under the `DATASETS` variable defined in `configs/data_configs.py`. There, you should define the source/target data paths for the train and test sets as well as the transforms to be used for training and inference.

```
DATASETS = {
	'ffhq_hypernet': {
	'transforms': transforms_config.EncodeTransforms,
	'train_source_root': dataset_paths['ffhq'],
	'train_target_root': dataset_paths['ffhq'],
	'test_source_root': dataset_paths['celeba_test'],
	'test_target_root': dataset_paths['celeba_test']
	}

}
```

4. To train with your newly defined dataset, simply use the flag `--dataset_type my_hypernet`.

<br>

## Training HyperEditor

The main training script can be found in `scripts/train.py`.
See `options/train_options.py` for all training-specific flags.
Intermediate training results are saved to `opts.exp_dir`. This includes checkpoints, train outputs, and test outputs.
Additionally, if you have tensorboard installed, you can visualize tensorboard logs in `opts.exp_dir/logs`.

Here, we provide an example for training on the human faces domain:

```
python scripts/train.py
```

### Additional Notes

- To select which generator layers to tune with the hypernetworks, you can use the `--layers_to_tune` flag.
  
  - By default, we will alter all non-toRGB convolutional layers.
  - If we use adaptive layer selector to reduce the complexity of the model, when a single model implements a single attribute edit, you can use `--choose_layers` flag.
  - The adaptive layer selector trade-off parameter can be used with `--lambda_std`, which defaults to `0.6`.
- If training a model to edit only a single attribute, fill in text pairs in `--init_text` and `--target_text`, respectively, e.g. ('face','face with smile').
- If you are training a model to edit multiple attributes, you can use the `--target_text_file flag`.
  
  - `--init_text` defaults to the initial text for multiple attributes, such as 'face'.
  - The object txt file pointed to by `--target_text_file` contains the target text condition in each line.
    <br>

## Inference

## Inference Script

You can use `scripts/inference.py` to apply a trained HyperEditor model on a set of images:
See `options/test_options.py` for all inference-specific flags.

Here, we provide an example for inference on the human faces domain:

```
python scripts/inference.py
```

### Additional Notes

- The results are saved to `--exp_dir`.
- The path to the trained HyperEditor model is stored in `--checkpoint_path`.
- The path to the test images is stored in `--data_path`.
