import argparse
import os
import json
from diffusers.models import AutoencoderKL
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
from pathlib import Path
from torch.utils import data as torchdata
from torch.utils.data import default_collate


def read_jsonl(path: str):
    """
    Read a jsonl file.
    Args:
        path (str): Path to the jsonl file.
    Returns:
        List[Dict]: A list of dicts, each dict is one line in the jsonl file.
    """
    data = []
    with open(path, 'r') as f:
        for line in f:
            line_dict = json.loads(line)
            data.append(line_dict)
    return data


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def my_collate(batch):
    batch = [x for x in batch if x[1] is not None]
    return default_collate(batch)


class SimpleDataset(torchdata.Dataset):
    def __init__(self, data_root, image_path_list, transform):
        self.data_root = data_root
        self.image_path_list = image_path_list
        self.transform = transform

    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        try:
            image = Image.open(os.path.join(self.data_root, image_path))
            image = self.transform(image)
        except:
            print('broken image:', os.path.join(self.data_root, image_path))
            image = None
        
        return image_path, image
        

def extract_img_vae(dataset_root: str, vae_checkpoint_path: str, annotation_path: str, save_dir: str, image_resize: int = 256, batch_size: int = 64, start_idx: int = None, end_idx: int = None, device: str = 'cuda'):
    vae = AutoencoderKL.from_pretrained(f'{vae_checkpoint_path}').to(device)

    # train_data_json = json.load(open(args.json_path, 'r'))
    train_data_json = read_jsonl(annotation_path)
    image_names = set()

    vae_save_root = f'{save_dir}/{image_resize}resolution'
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(vae_save_root, exist_ok=True)

    vae_save_dir = os.path.join(vae_save_root, 'noflip')
    os.makedirs(vae_save_dir, exist_ok=True)

    for item in train_data_json:
        image_name = item['img_path']
        if image_name in image_names:
            continue
        image_names.add(image_name)
    lines = sorted(image_names)
    lines = lines[start_idx:start_idx]

    _, images_extension = os.path.splitext(lines[0])

    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB')),
        # T.Resize(image_resize),  # Image.BICUBIC
        # T.CenterCrop(image_resize),
        T.Lambda(lambda img: center_crop_arr(img, image_resize)),
        T.ToTensor(),
        T.Normalize([.5], [.5]),
    ])

    dataset = SimpleDataset(dataset_root, lines, transform)
    dataloader = torchdata.DataLoader(dataset, batch_size, num_workers=4, collate_fn=my_collate)
    
    os.umask(0o000) 

    for image_names, images in tqdm(dataloader, total=len(dataloader)):
        images = images.to(device)
        with torch.no_grad():
            posterior = vae.encode(images).latent_dist
            z = torch.cat([posterior.mean, posterior.std], dim=1).detach().cpu().numpy().squeeze()
        
        for image_name, cur_z in zip(image_names, z):
            save_path = os.path.join(vae_save_dir, os.path.dirname(image_name), Path(image_name).stem)
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            np.save(save_path, cur_z)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str)
    parser.add_argument('--device', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dataset_root = '/root/paddlejob/workspace/box5_home/disk2/vis/zhaochuyang/data/journeydb/'
    vae_checkpoint_path = '/root/paddlejob/workspace/box5_home/disk1/vis/songyuxin02/data/checkpoints/sd-vae-ft-ema'
    annotation_path = args.annotation_path
    save_dir = '/root/paddlejob/workspace/vae_features'
    device = args.device
    batch_size = 128
    extract_img_vae(dataset_root=dataset_root, vae_checkpoint_path=vae_checkpoint_path, annotation_path=annotation_path, save_dir=save_dir, batch_size=batch_size, device=device)
