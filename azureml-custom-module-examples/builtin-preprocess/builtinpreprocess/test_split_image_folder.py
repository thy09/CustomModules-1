from shutil import copyfile
import os
import random
import fire


def split_image_folder(src_path, tgt_train_path, tgt_test_path, thre):
    for j, subdir in enumerate(os.listdir(src_path)):
        print(f'Split {subdir}')
        src_sub_path = os.path.join(src_path, subdir)
        if not os.path.isdir(src_sub_path):
            continue
        image_list = os.listdir(src_sub_path)
        n = len(image_list)
        random.shuffle(image_list)
        tgt_train_pic_dir = os.path.join(tgt_train_path, subdir)
        tgt_test_pic_dir = os.path.join(tgt_test_path, subdir)
        os.makedirs(tgt_train_pic_dir, exist_ok=True)
        os.makedirs(tgt_test_pic_dir, exist_ok=True)
        for i, pic in enumerate(image_list):
            src_pic_path = os.path.join(src_sub_path, pic)
            if not os.path.isfile(src_pic_path):
                continue
            if i < int(n * thre):
                copyfile(src_pic_path, os.path.join(tgt_train_pic_dir, pic))
            else:
                copyfile(src_pic_path, os.path.join(tgt_test_pic_dir, pic))


def entrance(src_path='/mnt/chjinche/data/output_transformed/',
             thre=0.9,
             tgt_train_path='/mnt/chjinche/data/output_transformed_train/',
             tgt_test_path='/mnt/chjinche/data/output_transformed_test/'):
    print('Start!')
    split_image_folder(src_path, tgt_train_path, tgt_test_path, thre)
    print('Finished')


if __name__ == '__main__':
    fire.Fire(entrance)
