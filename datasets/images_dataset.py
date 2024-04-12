from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import random

class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.style_path = sorted(data_utils.make_dataset(target_root), reverse=True)
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):

		random_index = random.randint(0, len(self.source_paths) - 1)
		from_path = self.source_paths[index]
		to_path = self.target_paths[index]
		# print(self.source_paths)
		fix_path = self.style_path[random_index]
		# print(self.style_path)

		from_im = Image.open(from_path).convert('RGB')
		to_im = Image.open(to_path).convert('RGB')
		fix_im = Image.open(fix_path).convert('RGB')

		if self.target_transform:
			to_im = self.target_transform(to_im)
			fix_im = self.target_transform(fix_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im, fix_im
