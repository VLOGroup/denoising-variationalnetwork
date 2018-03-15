import os
import shutil
from scipy.misc import imread, toimage

import urllib.request
import tarfile


bsds_url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz'

tar_name = bsds_url.split('/')[-1]
print('start downloading training data from "{}"'.format(bsds_url))
urllib.request.urlretrieve(bsds_url, tar_name)
print('downloading done')

print('start extracting tar file')
with tarfile.open(tar_name, 'r:gz') as tar_file:
    tar_file.extractall('./tmp/')
print('extracting done')

# setup the training data
base_dir = './data'
training_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

if not os.path.exists(training_dir):
    os.makedirs(training_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# generate the training data
print('generating training data')
data_source_base_dir = './tmp/BSR/BSDS500/data/images'
for src in ['test', 'train']:
    src_dir = os.path.join(data_source_base_dir, src)
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and
             os.path.splitext(f)[1] == '.jpg']
    for file in files:
        full_file_name = os.path.join(src_dir, file)
        img = imread(full_file_name, mode='L')
        toimage(img, cmin=0, cmax=255).save(os.path.join(training_dir, file))
print('training data done')

print('generating test data')
test_set_list_url = 'http://www.visinf.tu-darmstadt.de/media/visinf/vi_data/foe_test.txt'
test_list = urllib.request.urlopen(test_set_list_url).read().decode('utf-8').split('\n')[:-1]
src_dir = os.path.join(data_source_base_dir, 'val')
for i in range(len(test_list)):
    img = imread(os.path.join(src_dir, test_list[i]), mode='L')
    toimage(img, cmin=0, cmax=255).save(os.path.join(test_dir, '{}.jpg'.format(i+1)))
print('test data done')

# cleanup
print('cleaning up')
shutil.rmtree('./tmp')
os.remove(tar_name)
print('cleaning up done')
