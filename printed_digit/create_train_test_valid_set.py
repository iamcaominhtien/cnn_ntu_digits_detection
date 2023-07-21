import os, cv2

from sklearn.model_selection import train_test_split

folder_number = 9
data_path = r'D:\code-example\my_research\mnist_cnn\printed_digit\data'
folder_number_path = os.path.join(data_path, str(folder_number))

file_names = os.listdir(folder_number_path)
images = [cv2.imread(os.path.join(folder_number_path, file_name)) for file_name in file_names]

# 80% of the data is used for training
# 7% of the data is used for validation
# 13% of the data is used for testing

# use train_test_split to split the data into training and others
# use train_test_split to split the others into validation and testing

(train_images, other_images) = train_test_split(images, test_size=0.2, random_state=42)
(validation_images, test_images) = train_test_split(other_images, test_size=0.65, random_state=42)

# save the images to the corresponding folders
train_folder_path = os.path.join(data_path, 'train', str(folder_number))
if not os.path.exists(train_folder_path):
    os.makedirs(train_folder_path)
validation_folder_path = os.path.join(data_path, 'validation', str(folder_number))
if not os.path.exists(validation_folder_path):
    os.makedirs(validation_folder_path)
test_folder_path = os.path.join(data_path, 'test', str(folder_number))
if not os.path.exists(test_folder_path):
    os.makedirs(test_folder_path)

print('Saving images to the corresponding folders...')
print('Train folder path: ', train_folder_path)
for idx, image in enumerate(train_images):
    cv2.imwrite(os.path.join(train_folder_path, str(idx) + '.jpg'), image)

print('Validation folder path: ', validation_folder_path)
for idx, image in enumerate(validation_images):
    cv2.imwrite(os.path.join(validation_folder_path, str(idx) + '.jpg'), image)

print('Test folder path: ', test_folder_path)
for idx, image in enumerate(test_images):
    cv2.imwrite(os.path.join(test_folder_path, str(idx) + '.jpg'), image)

print('Done')
exit()