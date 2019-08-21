from shutil import copyfile
import os
import random
import fire


def split_image_folder(src_path, tgt_more_path, tgt_less_path, thre):
    for j, subdir in enumerate(os.listdir(src_path)):
        src_sub_path = os.path.join(src_path, subdir)
        if not os.path.isdir(src_sub_path):
            continue
        image_list = os.listdir(src_sub_path)
        n = len(image_list)
        random.shuffle(image_list)
        for i, pic in enumerate(image_list):
            src_pic_path = os.path.join(src_sub_path, pic)
            if not os.path.isfile(src_pic_path):
                continue
            tgt_big_pic_dir = os.path.join(tgt_more_path, subdir)
            tgt_small_pic_dir = os.path.join(tgt_less_path, subdir)
            os.makedirs(tgt_big_pic_dir, exist_ok=True)
            os.makedirs(tgt_small_pic_dir, exist_ok=True)
            if i < int(n*thre):
                copyfile(src_pic_path, os.path.join(tgt_big_pic_dir, pic))
            else:
                copyfile(src_pic_path, os.path.join(tgt_small_pic_dir, pic))


def entrance(src_path='', thre=0.9, tgt_more_path='', tgt_less_path=''):
    split_image_folder(src_path, tgt_more_path, tgt_less_path, thre)
    # workaround for postprocess
    copyfile(os.path.join(src_path, 'index_to_label.json'), os.path.join(tgt_more_path, 'index_to_label.json'))
    copyfile(os.path.join(src_path, 'index_to_label.json'), os.path.join(tgt_less_path, 'index_to_label.json'))
    print('Finished')


if __name__ == '__main__':
    fire.Fire(entrance)
