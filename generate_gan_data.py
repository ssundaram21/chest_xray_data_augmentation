import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import PIL.Image
import imageio
# import tfutils
import matplotlib.pyplot as plt
import os
import sys

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

network_path = "/om/user/shobhita/src/chexpert/CheXpert GAN/"
output_data_path = "/om/user/shobhita/src/chexpert/gan_fake_data/"
real_data_path = "/om/user/shobhita/src/chexpert/data/"

names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices']

tf.InteractiveSession()

# Import pretrained Chexpert GAN.
with open(network_path + "network-final.pkl", 'rb') as file:
    G, D, Gs = pickle.load(file)

real_labels = pd.read_csv(real_data_path + "CheXpert-v1.0-small/train_preprocessed.csv")

classes_to_generate = ["Lung Lesion", "Pleural Other", "Fracture"]
total = len(real_labels)

lesion =  sum(real_labels["Lung Lesion"])
pleural = sum(real_labels["Pleural Other"])
fracture = sum(real_labels["Fracture"])

lesion_n, pleural_n, fracture_n = int(lesion*1.65), int(pleural*3.65), int(fracture*1.95)
total_gen = total + lesion_n + pleural_n + fracture_n

print("Lesion: {}/{} + {} --> {}/{}".format(lesion, lesion/total, lesion_n, lesion+lesion_n, (lesion+lesion_n)/(total+total_gen)))
print("Pleural: {}/{} + {} --> {}/{}".format(pleural, pleural/total, pleural_n, pleural+pleural_n, (pleural+pleural_n)/(total + total_gen)))
print("Fracture: {}/{} + {} --> {}/{}".format(fracture, fracture/total, fracture_n, fracture+fracture_n, (fracture+fracture_n)/(total + total_gen)))
sys.stdout.flush()


label_vectors = {}
for cat, n in zip(classes_to_generate, [lesion_n, pleural_n, fracture_n]):
    relevant_labels = real_labels[real_labels[cat] == 1]
    new_labels = relevant_labels.sample(n, replace=True)[names].to_numpy()
    label_vectors[cat] = new_labels

for cat, arr in label_vectors.items():
    print("{}: {}".format(cat, arr.shape))

label_vectors = {}
for cat, n in zip(classes_to_generate, [lesion_n, pleural_n, fracture_n]):
    relevant_labels = real_labels[real_labels[cat] == 1]
    new_labels = relevant_labels.sample(n, replace=True)[names].to_numpy()
    label_vectors[cat] = new_labels

for cat, arr in label_vectors.items():
    print("{}: {}".format(cat, arr.shape))

labels_save = {}
for cat in classes_to_generate:
    labels = label_vectors[cat]
    batch = 1
    used_labels = []
    used_imgname = []
    latents_raw = np.random.RandomState(1000).randn(labels.shape[0], *Gs.input_shapes[0][1:])
    total_num = latents_raw.shape[0]

    print("Generating {}".format(cat))
    sys.stdout.flush()

    for n in range(int(total_num / batch)):
        if n % 1000 == 0:
            print("{}/{}".format(n, total_num))

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
            labels_save["{}_{}".format(cat, image_idx)] = labels[image_idx, :]
            store_name = 'fake_{}_{}.png'.format(cat, image_idx)
            used_imgname.append(store_name)
            store_path = os.path.join(data_dir, store_name)
            imageio.imwrite(store_path, save_images[idx])

    print("Done with {}".format(cat))
    print(len(labels))
    print(len(used_labels))
    print(len(used_imgname))
    sys.stdout.flush()

with open(output_data_path + "gan_image_labels.pkl", "wb") as handle:
    pickle.dump(labels_save, handle)
sys.stdout.flush()

print("Done :)")