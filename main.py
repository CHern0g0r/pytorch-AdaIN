import argparse
from pathlib import Path
from distutils.dir_util import copy_tree

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import shutil

import net
from function import adaptive_instance_normalization, coral

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def get_parser():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content', type=str,
                        help='File path to the content image')
    parser.add_argument('--content_dir', type=str,
                        help='Directory path to a batch of content images')
    parser.add_argument('--style', type=str,
                        help='File path to the style image, or multiple style \
                        images separated by commas if you want to do style \
                        interpolation or spatial control')
    parser.add_argument('--style_dir', type=str,
                        help='Directory path to a batch of style images')
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
    parser.add_argument('--decoder', type=str, default='models/decoder.pth')

    # Additional options
    parser.add_argument('--content_size', type=int, default=512,
                        help='New (minimum) size for the content image, \
                        keeping the original size if set to 0')
    parser.add_argument('--style_size', type=int, default=512,
                        help='New (minimum) size for the style image, \
                        keeping the original size if set to 0')
    parser.add_argument('--crop', action='store_true',
                        help='do center crop to create squared image')
    parser.add_argument('--save_ext', default='.jpg',
                        help='The extension name of the output image')
    parser.add_argument('--output', type=str, default='test_output',
                        help='Directory to save the output image(s)')

    # Advanced options
    parser.add_argument('--preserve_color', action='store_true',
                        help='If specified, preserve color of the content image')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='The weight that controls the degree of \
                                stylization. Should be between 0 and 1')
    parser.add_argument(
        '--style_interpolation_weights', type=str, default='',
        help='The weight for blending the style of multiple style images')

    return parser

if __name__ == '__main__':
    parser = get_parser()

    args = parser.parse_args()

    do_interpolation = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Either --content or --contentDir should be given.
    assert (args.content)
    content_path = Path(args.content)

    # Either --style or --styleDir should be given.
    assert (args.style)
    # style_path = args.style.split(',')
    # if len(style_path) == 1:
    style_path = Path(args.style)

    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(args.decoder))
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    content_tf = test_transform(args.content_size, args.crop)
    style_tf = test_transform(args.style_size, args.crop)

    styles = []

    for style_img in style_path.iterdir():
        if style_img.suffix == '.jpg':
            styles.append(style_tf(Image.open(str(style_img))).unsqueeze(0))

    print(f'Number of styles {len(styles)}')

    cur_style = styles[0]

    topil = transforms.ToPILImage()

    style_idx = 0

    for directory in content_path.iterdir():
        dir_name = directory.name

        new_directory = output_dir / dir_name
        new_directory.mkdir(exist_ok=True, parents=True)

        for target in directory.iterdir():
            if target.is_dir():
                inner_dir_name = target.name
                new_inner_path = new_directory / inner_dir_name
                copy_tree(str(target), str(new_inner_path))
            elif target.suffix == '.png':
                content_name = target.name
                output_path = new_directory / content_name

                cur_style = styles[style_idx % len(styles)]
                cur_style = cur_style.to(device)
                
                content_img = Image.open(str(target))
                if content_img.mode == 'L':
                    content_img = content_img.convert('RGB')
                content = content_tf(content_img)
                content = content.to(device).unsqueeze(0)
                with torch.no_grad():
                    output = style_transfer(
                        vgg, decoder, content, cur_style)

                cur_style = cur_style.to('cpu')
                style_idx += 1
                
                output = output.cpu()
                save_image(output, str(output_path))


