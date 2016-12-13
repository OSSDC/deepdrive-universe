# DeepDrive in [Universe](https://universe.openai.com/)

## Prerequites

Follow the env setup instructions [here](https://github.com/openai/universe-windows-envs/blob/master/vnc-gtav/README.md)

## Setup
```
git clone https://github.com/deepdrive/deepdrive-universe
```

Note: You will need GPU acceleration to do inference at the 8Hz that the model runs. Slower (or faster) inference may work, but is not the standard way the model is run.

## Baseline model - Caffe version
Install the latest version of Caffe and download the model via
```
cd drivers/deepdrive
wget -O caffe_deep_drive_train_iter_35352.caffemodel https://goo.gl/sVAedm
```

## (Coming soon) Baseline model - Tensorflow version
_Thanks to  [Rafal jozefowicz](https://github.com/rafaljozefowicz) for contributing this model_
```
cd drivers/deepdrive-tf
wget -O model.ckpt-20048 https://goo.gl/zanx88
```


Baseline models were trained with the standard hood camera in GTAV. 

To enable the hood camera, hit <kbd>v</kbd> until you see something like this
![deepdrive load](https://www.dropbox.com/s/q28tce40ukurm9p/Screenshot%202016-10-30%2014.33.50.png?dl=1)

If you see the steering wheel, change the camera settings like so:
![deepdrive load](https://www.dropbox.com/s/h3xu98jz45bafld/Screenshot%202016-10-30%2014.28.42.png?dl=1)

In addition, enable borderless mode to avoid sending the window chrome through the network

![borderless](https://www.dropbox.com/s/dci8o6z3129bwpl/borderless.jpg?dl=1)

## Run the model
```
python main.py -d [DeepDriver|DeepDriverTF] -r vnc://<your-env-ip>:5900+15900
```

## Directly connected machines
To connect your Windows and Ubuntu machines directly via ethernet, follow [these instructions](http://askubuntu.com/a/26770/158805) for adding the interface to Ubuntu.

Use the _Netmask_ provided by `ipconfig` in Windows for your ethernet interface.
