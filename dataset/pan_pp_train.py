import numpy as np
from PIL import Image
from torch.utils import data
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
import math
import string
import scipy.io as scio
import mmcv
import re
import os

EPS = 1e-6
tt_root_dir = './data/TrainingData/'
tt_train_data_dir = tt_root_dir + 'image/'
tt_train_gt_dir = tt_root_dir + 'txt/'

def get_img(img_path, read_type='cv2'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path).convert('RGB'))
    except Exception as e:
        print(img_path)
        raise
    return img


def check(s):
    for c in s:
        if c in list(string.printable[:-6]):
            continue
        return False
    return True

def get_ann_tt(img, gt_path):
    h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig')
        gt = line.split('\t')
        if len(gt)<5:
            print(gt_path)
            print(gt)
            continue
        word = gt[-1]
        word=word.replace('@@@','')
        if word == '###':
            continue
        words.append(word)
        
        p_bbox =[int(float(gt[j])) for j in range(len(gt)-1)]
        bbox_re=np.reshape(p_bbox,(-1,2))
        bbox=[]
        for i in range(len(bbox_re)):
            bbox.append(bbox_re[i][0])
            bbox.append(bbox_re[i][1])
        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * len(bbox_re))
        bboxes.append(bbox)
    return bboxes, words

def random_horizontal_flip(imgs):
    if random.random() < 0.1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 1
    angle = random.random() * 2 * max_angle - max_angle
    # angle = angle + 90*random.randrange(0, 4)
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    return imgs


def scale_aligned(img, h_scale, w_scale):
    h, w = img.shape[0:2]
    h = int(h * h_scale + 0.5)
    w = int(w * w_scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    '''
    from mmcv import Config
    cfg = Config.fromfile('config/pan_pp_test.py')
    from dataset import build_data_loader
    data_loader = build_data_loader(cfg.data.train)
    '''
    return img


def random_scale(img, min_size, short_size=736):
    h, w = img.shape[0:2]

    scale = np.random.choice(np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 3, 4, 5]))
    scale = (scale * short_size) / min(h, w)

    aspect = np.random.choice(np.array([0.9, 0.95, 1.0, 1.05, 1.1]))
    h_scale = scale * math.sqrt(aspect)
    w_scale = scale / math.sqrt(aspect)

    img = scale_aligned(img, h_scale, w_scale)
    return img


def random_crop_padding(imgs, target_size):
    """ using padding and the final crop size is (800, 800) """
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs


def update_word_mask(instance, instance_before_crop, word_mask):
    labels = np.unique(instance)

    for label in labels:
        if label == 0:
            continue
        ind = instance == label
        if np.sum(ind) == 0:
            word_mask[label] = 0
            continue
        ind_before_crop = instance_before_crop == label
        if float(np.sum(ind)) / np.sum(ind_before_crop) > 0.9:
            continue
        word_mask[label] = 0

    return word_mask


def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception as e:
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes


def get_vocabulary(voc_type, EOS='EOS', PADDING='PAD', UNKNOWN='UNK'):
    if voc_type == 'LOWERCASE':
        voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'ALLCASES':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        voc = list(string.printable[:-6])
    else:
        raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char


class PAN_PP_TRAIN(data.Dataset):
    def __init__(self,
                 split='train',
                 is_transform=False,
                 img_size=None,
                 short_size=736,
                 kernel_scale=0.5,
                 with_rec=False,
                 read_type='pil',
                 report_speed=False,
                 viz_mode=False):
        self.split = split
        self.is_transform = is_transform

        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_scale = kernel_scale
        self.short_size = short_size
        self.read_type = read_type

        self.img_paths = {}
        self.gts = {}
        self.texts = {}

        self.img_num = 0
        # tt
        self.img_paths['tt'] = []
        self.gts['tt'] = []
        img_names = [img_name for img_name in mmcv.utils.scandir(tt_train_data_dir, '.jpg')]
        img_names.extend([img_name for img_name in mmcv.utils.scandir(tt_train_data_dir, '.png')])
        img_names.extend([img_name for img_name in mmcv.utils.scandir(tt_train_data_dir, '.jpeg')])
        img_names.extend([img_name for img_name in mmcv.utils.scandir(tt_train_data_dir, '.tif')])
        img_names.extend([img_name for img_name in mmcv.utils.scandir(tt_train_data_dir, '.TIF')])

        for idx, img_name in enumerate(img_names):
            img_path = tt_train_data_dir + img_name
            self.img_paths['tt'].append(img_path)

            filename, file_extension = os.path.splitext(img_name)
            gt_name = filename + '.txt'
            gt_path = tt_train_gt_dir + gt_name
            self.gts['tt'].append(gt_path)
        self.img_num += len(self.img_paths['tt'])

        self.voc, self.char2id, self.id2char = get_vocabulary('LOWERCASE')
        self.max_word_num = 1000 #200
        self.max_word_len = 48 #32
        self.viz_mode = viz_mode

    def __len__(self):
        return self.img_num

    def __getitem__(self, index):
        img_path = self.img_paths['tt'][index]
        gt_path = self.gts['tt'][index]
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann_tt(img, gt_path)
        if len(bboxes) > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]
            words = words[:self.max_word_num]

        gt_words = np.full((self.max_word_num + 1, self.max_word_len), self.char2id['PAD'], dtype=np.int32)
        word_mask = np.zeros((self.max_word_num + 1, ), dtype=np.int32)
        for i, word in enumerate(words):
            if word == '###':
                continue
            if word == '???':
                continue
            word = word.lower()
            gt_word = np.full((self.max_word_len,), self.char2id['PAD'], dtype=np.int32)
            for j, char in enumerate(word):
                if j > self.max_word_len - 1:
                    break
                if char in self.char2id:
                    gt_word[j] = self.char2id[char]
                else:
                    gt_word[j] = self.char2id['UNK']
            if len(word) > self.max_word_len - 1:
                gt_word[-1] = self.char2id['EOS']
            else:
                gt_word[len(word)] = self.char2id['EOS']
            gt_words[i + 1] = gt_word
            word_mask[i + 1] = 1

        if self.is_transform:
            img = random_scale(img, self.img_size[0], self.short_size)

        gt_instance = np.zeros(img.shape[0:2], dtype='int32')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if len(bboxes) > 0:
            if type(bboxes) == list:
                for i in range(len(bboxes)):
                    bboxes[i] = np.reshape(bboxes[i] * ([img.shape[1], img.shape[0]] * (bboxes[i].shape[0] // 2)),
                                           (bboxes[i].shape[0] // 2, 2)).astype('int32')
            else:
                bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * (bboxes.shape[1] // 2)),
                                    (bboxes.shape[0], -1, 2)).astype('int32')
            for i in range(len(bboxes)):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
                if words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        gt_kernels = []
        for rate in [self.kernel_scale]:
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        if self.is_transform:
            imgs = [img, gt_instance, training_mask]
            imgs.extend(gt_kernels)
            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            gt_instance_before_crop = imgs[1].copy()
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]
            word_mask = update_word_mask(gt_instance, gt_instance_before_crop, word_mask)

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        max_instance = np.max(gt_instance)
        gt_bboxes = np.zeros((self.max_word_num + 1, 4), dtype=np.int32)
        for i in range(1, max_instance + 1):
            ind = gt_instance == i
            if np.sum(ind) == 0:
                continue
            points = np.array(np.where(ind)).transpose((1, 0))
            tl = np.min(points, axis=0)
            br = np.max(points, axis=0) + 1
            gt_bboxes[i] = (tl[0], tl[1], br[0], br[1])

        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        if not self.viz_mode:
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).long()
        gt_kernels = torch.from_numpy(gt_kernels).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()
        gt_bboxes = torch.from_numpy(gt_bboxes).long()
        gt_words = torch.from_numpy(gt_words).long()
        word_mask = torch.from_numpy(word_mask).long()

        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernels,
            training_masks=training_mask,
            gt_instances=gt_instance,
            gt_bboxes=gt_bboxes,
        )

        return data

if __name__ == "__main__":
    import random
    import shutil
    from PIL import ImageDraw


    try:
        shutil.rmtree('DataLoaderViz')
    except:
        print('Already DataLoaderViz Removed')
    os.mkdir('DataLoaderViz')
    data_loader = PAN_PP_TRAIN(split='train',
                                is_transform=True,
                                img_size=736,
                                short_size=736,
                                kernel_scale=0.5,
                                with_rec=False,
                                read_type='pil',
                                report_speed=False,
                                viz_mode=True)
    for i, tmp in enumerate(data_loader):
        img = tmp['imgs']
        gt = tmp['gt_bboxes'].numpy()
        draw = ImageDraw.Draw(img)
        for g in gt:
            if g[0] == g[2] and g[1] == g[3]:
                continue
            y1, x1, y2, x2 = g
            draw.line([(x1, y1),(x2, y1)], fill='red')
            draw.line([(x1, y1),(x1, y2)], fill='red')
            draw.line([(x2, y1),(x2, y2)], fill='red')
            draw.line([(x1, y2),(x2, y2)], fill='red')
        name = str(random.randrange(100_000, 1_000_000))
        img.save('./DataLoaderViz/' + name + '.png')
        cv2.imwrite('./DataLoaderViz/' + name + '_gt_texts.png', tmp['gt_texts'].numpy()/(tmp['gt_texts'].numpy().max()+0.00001)*255)
        cv2.imwrite('./DataLoaderViz/' + name + '_gt_kernels.png', tmp['gt_kernels'][0,:,:].numpy()/(tmp['gt_kernels'][0,:,:].numpy().max()+0.00001)*255)
        # cv2.imwrite('./DataLoaderViz/' + name + '_training_masks.png', tmp['training_masks'].numpy()/(tmp['training_masks'].numpy().max()+0.00001)*255)
        cv2.imwrite('./DataLoaderViz/' + name + '_gt_instances.png', cv2.applyColorMap((tmp['gt_instances'].numpy()/(tmp['gt_instances'].numpy().max()+0.00001)*255).astype(np.uint8), cv2.COLORMAP_JET))
        if i > 10:
            break