import json
import glob
import numpy as np
import os
import h5py
import random
import threading
import queue

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(1)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

def get_good_files(hdf5_path, train=True):
    if train:
        # https://gist.github.com/crizCraig/65677883e07c74bdc08f987e806cd95f
        with open(hdf5_path + "/good_files.json", "rb") as f:
            ids = json.loads(f.read().decode('utf8'))
        ids.remove(1)
    else:
        ids = [1]

    ret = []
    for i in ids:
        name = os.path.join(hdf5_path, "train_%04d.h5" % i)
        ret += [name]
    return set(ret)

def load_file(h5_filename):
    mean_pixel = np.array([104., 117., 123.], np.float32)
    out_images = []
    out_targets = []

    with h5py.File(h5_filename, 'r') as hf:
        images = list(hf.get('images'))
        targets = list(hf.get('targets'))
        perm = np.arange(len(images))
        for i in range(len(images)):
            idx = perm[i]
            img = images[idx].transpose((1, 2, 0))  # CHW => HWC
            img = img[:, :, ::-1]  # BGR => RGB
            img = img.astype(np.float32)
            img -= mean_pixel
            out_images.append(img)
            out_targets.append(targets[idx])
    return out_images, out_targets

def file_loader(file_stream):
    for h5_filename in file_stream:
        print('input file: {}'.format(h5_filename))
        yield load_file(h5_filename)

def batch_gen(file_stream, batch_size):
    gen = BackgroundGenerator(file_loader(file_stream))
    for images, targets in gen:
        num_iters = len(images) // batch_size
        for i in range(num_iters):
            yield images[i * batch_size:(i+1) * batch_size], targets[i * batch_size:(i+1) * batch_size]

class Dataset(object):
    def __init__(self, files):
        self._files = files

    def iterate_once(self, batch_size):
        def file_stream():
            for file_name in self._files:
                yield file_name
        yield from batch_gen(file_stream(), batch_size)

    def iterate_forever(self, batch_size):
        def file_stream():
            while True:
                random.shuffle(self._files)
                for file_name in self._files:
                    yield file_name
        yield from batch_gen(file_stream(), batch_size)


def get_dataset(hdf5_path, train=True):
    good_files = get_good_files(hdf5_path, train=train)
    file_names = glob.glob(hdf5_path + "/*.h5")
    file_names = [fname for fname in file_names if fname in good_files]
    return Dataset(file_names)

def run():
    hdf5_path = os.environ('DEEPDRIVE_HDF5_PATH')

    # print(get_good_files(hdf5_path))
    dataset = get_dataset(hdf5_path)
    print(dataset)

if __name__ == "__main__":
    run()
