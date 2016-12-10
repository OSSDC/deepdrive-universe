Run deepdrive models using Universe.

# Setup
```
git clone https://github.com/deepdrive/deep_drivers
```

Note: You will need GPU acceleration to do inference at the 8Hz that the model runs. Slower (or faster) inference may work, but is not the standard way to run the model.

## Baseline model - Tensorflow version
_Thanks to  [Rafal jozefowicz](https://github.com/rafaljozefowicz) for contributing this model_
```
cd deepdrive-tf
wget https://www.dropbox.com/s/fsummbpqlfildnq/model.ckpt-20048?dl=1
```

## Baseline model - Caffe version
Install the latest version of Caffe and download the model via
```
cd deepdrive
wget https://www.dropbox.com/s/z92c4otvyofgl3f/caffe_deep_drive_train_iter_35352.caffemodel?dl=1
```

## Directly connected machines
To connect your Windows and Ubuntu machines directly via ethernet, follow [these instructions](http://askubuntu.com/a/26770/158805) for adding the interface to Ubuntu.

Use the _Netmask_ provided by `ipconfig` in Windows for your ethernet interface.
