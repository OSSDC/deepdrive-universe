# DeepDrive in [Universe](https://universe.openai.com/)

## Setup
```
git clone https://github.com/deepdrive/deep_drivers
```

Note: You will need GPU acceleration to do inference at the 8Hz that the model runs. Slower (or faster) inference may work, but is not the standard way the model is run.

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

Baseline models were trained with the standard hood camera in GTAV. 

To enable the hood camera, hit <kbd>v</kbd> until you see something like this
![deepdrive load](https://www.dropbox.com/s/q28tce40ukurm9p/Screenshot%202016-10-30%2014.33.50.png?dl=1)

If you see the steering wheel, change the camera settings like so:
![deepdrive load](https://www.dropbox.com/s/h3xu98jz45bafld/Screenshot%202016-10-30%2014.28.42.png?dl=1)

In addition, enable borderless mode to avoid sending the window chrome through the network

![borderless](https://www.dropbox.com/s/dci8o6z3129bwpl/borderless.jpg?dl=1)

## Run the GTAV Universe environment
Follow the env setup instructions [here](https://github.com/openai/universe-windows-envs/blob/master/vnc-gtav/README.md)

## Directly connected machines
To connect your Windows and Ubuntu machines directly via ethernet, follow [these instructions](http://askubuntu.com/a/26770/158805) for adding the interface to Ubuntu.

Use the _Netmask_ provided by `ipconfig` in Windows for your ethernet interface.

## Run the model
```
python main.py -d [DeepDriver|DeepDriverTF] -r vnc://<your-env-ip>:5900+15900
```
