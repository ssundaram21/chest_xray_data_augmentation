import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import PIL.Image
import imageio
import tfutil
import matplotlib.pyplot as plt
import os
import sys
import argparse

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, required=True)
FLAGS = parser.parse_args()
idx = FLAGS.idx

network_path = "/om/user/shobhita/src/chexpert/CheXpert GAN/"
output_data_path = "/om/user/shobhita/src/chexpert/gan_fake_data/"
real_data_path = "/om/user/shobhita/src/chexpert/data/"

names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices']

label_filename = "train_preprocessed_subset_50.csv" if idx == 50 else "train_preprocessed.csv"
patient_id = 64900 if idx == 50 else 65000

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# Initialize TensorFlow session.
with tf.Session():

    print("Generating GAN data for {}% dataset".format(idx))
    sys.stdout.flush()

    # Import pretrained Chexpert GAN.
    with open(network_path + "network-final.pkl", 'rb') as file:
        G, D, Gs = pickle.load(file)

    label_path = "/om/user/shobhita/src/chexpert/data/CheXpert-v1.0-small/" + label_filename
    labels = pd.read_csv(label_path)

    def get_class_split(labels, proportion=True):
        split = {}
        total = len(labels)
        for name in names:
            split[name] = sum(labels[name])/total if proportion else sum(labels[name])
        return split

    split = get_class_split(labels, proportion=False)
    total = len(labels)
    low = {name: value for name, value in split.items() if value/total <= 0.09}

    label_vectors = {}
    x = 15000 if idx==50 else 30000
    y = 17000 if idx==50 else 35000
    z = 15000 if idx==50 else 30000

    for cat, num_to_sample in zip(["Lung Lesion", "Pleural Other", "Fracture"], [x, y, z]):
        print("Generating {} for cat {}".format(num_to_sample, cat))
        relevant_labels = labels[labels[cat] == 1]
        new_labels = relevant_labels.sample(num_to_sample, replace=True)[names].to_numpy()
        label_vectors[cat] = new_labels
    sys.stdout.flush()

    output_data_path = "/om/user/shobhita/src/chexpert/data/CheXpert-v1.0-small/train/patient{}/study1/".format(patient_id)
    output_labels_path = "/om/user/shobhita/src/chexpert/data/"
    cats = ["Lung Lesion", "Pleural Other", "Fracture"]
    labels_save = {}
    for cat in cats:
        labels = label_vectors[cat]
        batch = 1
        used_labels = []
        used_imgname = []
        latents_raw = np.random.RandomState(1000).randn(labels.shape[0], *Gs.input_shapes[0][1:])
        total_num = latents_raw.shape[0]

        print("Generating {}".format(cat))
        sys.stdout.flush()

        for n in range(int(total_num / batch)):
            if n % 1000 == 0 or n == 50:
                print("{}/{}".format(n, total_num))
                sys.stdout.flush()

            latent_vec = latents_raw[n * batch: (n + 1) * batch, :]
            label_vec = labels[n * batch: (n + 1) * batch, :]
            used_labels.append(label_vec)
            images = Gs.run(latent_vec, label_vec)
            images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
            images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
            save_images = np.squeeze(images, axis=-1)

            data_dir = output_data_path
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            for idx in range(save_images.shape[0]):
                image_idx = idx + batch * n
                labels_save["{}_{}".format(cat.replace(" ", "_"), image_idx)] = labels[image_idx, :]
                store_name = '{}_{}.jpg'.format(cat.replace(" ", "_"), image_idx)
                used_imgname.append(store_name)
                store_path = os.path.join(data_dir, store_name)
                imageio.imwrite(store_path, save_images[idx])

        print("Done with {}".format(cat))
        print("Num ims to generate: ", len(labels))
        print("Num labels generated: ", len(used_labels))
        print("Num ims generated: ", len(used_imgname))
        sys.stdout.flush()

    with open(output_labels_path + "gan_labels_{}_prop.pkl".format(patient_id), "wb") as handle:
        pickle.dump(labels_save, handle)

