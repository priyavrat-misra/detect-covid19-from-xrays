import os
import shutil
import random

class_names = ['normal', 'viral', 'covid']
root_dir = 'COVID-19 Radiography Database'
source_dirs = ['NORMAL', 'Viral Pneumonia', 'COVID-19']

if os.path.isdir(os.path.join(root_dir, source_dirs[1])):
    os.mkdir(os.path.join(root_dir, 'test'))

    for i, d in enumerate(source_dirs):
        os.rename(os.path.join(root_dir, d),
                  os.path.join(root_dir, class_names[i]))

    for c in class_names:
        os.mkdir(os.path.join(root_dir, 'test', c))

    for c in class_names:
        images = [x for x in os.listdir(os.path.join(
            root_dir, c)) if x.lower().endswith('png')]
        selected_images = random.sample(images, 40)
        for image in selected_images:
            source_path = os.path.join(root_dir, c, image)
            target_path = os.path.join(root_dir, 'test', c, image)
            shutil.move(source_path, target_path)

    os.mkdir(os.path.join(root_dir, 'train'))

    for c in class_names:
        source_path = os.path.join(root_dir, c)
        target_path = os.path.join(root_dir, 'train')
        shutil.move(source_path, target_path)
