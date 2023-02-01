import cv2, os

data_path = 'C:/Python Apps/MACHINE_LEARNING/Face_Mask_Detection_With_CNN/train'
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]

label_dict = dict(zip(categories, labels))

# print(label_dict)
# print(categories)
# print(labels)

img_size = 100
data = []
target = []

# for category in CATEGORIES:
#     path = os.path.join(DIRECTORY, category)
#     for img in os.listdir(path):
#         img_path = os.path.join(path, img)
#         image = load_img (img_path, target_size=(224, 224))
#         image = img_to_array(image)
#         image = preprocess_input(image)
        
#         data.append(image)
#         labels.append(category)
for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)
    
    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        
        try:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray_img, (img_size, img_size))
            data.append(resized_img)
            target.append(label_dict[category])
        except Exception as e:
            print("Exception: ", e)

import numpy as np  
data = np.array(data)/255.0
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))

target = np.array(target)  

from keras.utils import np_utils
target = np_utils.to_categorical(target)
np.save('data', data)
np.save('target', target)

