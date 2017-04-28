import pandas as pd
import os



def prep_img_label_maps(train_label_map_path, validation_label_map_path):
	if train_label_map_path != validation_label_map_path:
		return train_label_map_path, validation_label_map_path
	else:
		label_map = pd.read_csv(train_label_map_path)
		train_set=label_map.sample(frac=0.8,random_state=200)
		validation_set=label_map.drop(train_set.index)
		new_train_label_path = os.path.join(os.path.dirname(train_label_map_path), 'train_label_map.csv')
		new_validation_label_path = os.path.join(os.path.dirname(validation_label_map_path), 'validation_label_map.csv')
		train_set.to_csv(new_train_label_path, index = False)
		validation_set.to_csv(new_validation_label_path, index = False)
		return new_train_label_path, new_validation_label_path




def get_image_label_list(label_map_file_path, data_dir ): 
	label_frame = pd.read_csv(label_map_file_path)
	full_file_path = []
	labels = []
	for path, directories, files in os.walk(data_dir):
		for f in files:
			base_filename = os.path.splitext(f)[0]

			if((label_frame.applymap(str).ix[:,0]==base_filename).any()):
				idx = label_frame[label_frame.applymap(str).ix[:,0]==base_filename].index
				label_value = label_frame.ix[:,1].loc[idx].tolist()[0]
				labels.append(label_value)
				full_file_path.append(os.path.join(path,f))

	return full_file_path,labels
