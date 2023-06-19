import os
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2')
import csv

if __name__ == '__main__':

	BASE_DIR = os.path.join('../../data/vww/', "vw_coco2014_96")
	generate_path='../../data/vww/val_data_official/'

	names=[]
	file=open("y_labels.csv")
	reader=csv.reader(file)
	for row in reader:
		names.append('COCO_val2014_'+row[0].replace('bin','jpg'))
	file.close()

	labels=['person','non_person']

	for label in labels:
		save_dir=os.path.join(generate_path, label)
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		dataset_dir = os.path.join(BASE_DIR, label)

		all_names=os.listdir(dataset_dir)
		for image_name in names:
			if image_name in all_names:
				full_path=os.path.join(dataset_dir,image_name)

				img = tf.keras.preprocessing.image.load_img(
					full_path, color_mode='rgb').resize((96, 96))
				arr = tf.keras.preprocessing.image.img_to_array(img)
				arr = arr.reshape([1, 96, 96, 3]).transpose((0,3,1,2))
				arr = np.ascontiguousarray(arr)
				arr = arr.astype(np.uint8)

				f = open(os.path.join(save_dir, image_name[13:-3] + "bin"), "wb")
				f.write(arr)
				f.close()
