# DeepDrive in [Universe](https://universe.openai.com/)

_Pretrained self-driving car models that run in GTAV / Universe_

## Prerequisites

Follow the env setup instructions [here](https://github.com/openai/universe-windows-envs/blob/master/vnc-gtav/README.md)

## Setup
```
git clone https://github.com/deepdrive/deepdrive-universe
```

Note: You will need GPU acceleration to do inference at the 8Hz that the model runs. Slower (or faster) inference may work, but is not the standard way the model is run.

### Baseline model - Caffe version
Install the latest version of Caffe and download the model via
```
cd drivers/deepdrive
wget -O caffe_deep_drive_train_iter_35352.caffemodel https://goo.gl/sVAedm
```

### (Work in progress) Baseline model - Tensorflow version
_Thanks to  [Rafal jozefowicz](https://github.com/rafaljozefowicz) for helping train this model_
```
cd drivers/deepdrive_tf
wget -O model.ckpt-20048 https://goo.gl/zanx88
wget -O model.ckpt-20048.meta https://goo.gl/LNqHoj
```


Baseline models were trained with the standard hood camera in GTAV. 

To enable the hood camera, hit <kbd>v</kbd> until you see something like this
![deepdrive load](https://www.dropbox.com/s/q28tce40ukurm9p/Screenshot%202016-10-30%2014.33.50.png?dl=1)

If you see the steering wheel, change the camera settings like so:
![deepdrive load](https://www.dropbox.com/s/h3xu98jz45bafld/Screenshot%202016-10-30%2014.28.42.png?dl=1)

In addition, set the resolution to 800x600 to minimize network I/O and enable borderless mode to avoid sending the window chrome through the network

![borderless](https://www.dropbox.com/s/dci8o6z3129bwpl/borderless.jpg?dl=1)

## Run the model
```
python main.py -d [DeepDriver|DeepDriverTF] -r vnc://<your-env-ip>:5900+15900
```

## Sample data returned by the env

```
{
	"body" :
	{
		"done" : false,
		"info" :
		{
			"center_of_lane_reward" : 0,
			"distance_from_destination" : 1306.6157153344816,
			"forward_vector_x" : -0.9870644211769104,
			"forward_vector_y" : 0.15973846614360809,
			"forward_vector_z" : 0.013689413666725159,
			"game_time.day_of_month" : 6,
			"game_time.hour" : 16,
			"game_time.minute" : 8,
			"game_time.month" : 9,
			"game_time.ms_per_game_min" : 2000,
			"game_time.second" : 47,
			"game_time.year" : 2009,
			"heading" : 80.808067321777344,
			"is_game_driving" : false,
			"last_collision_time" : 1481999559,
			"last_material_collided_with" : "4201905313",
			"on_road" : true,
			"script_hook_loadtime" : 1481939095,
			"speed" : 0,
			"spin" : 0,
			"x_coord" : -2372.70068359375,
			"y_coord" : 1032.6005859375,
			"z_coord" : 195.53288269042969
		},
		"reward" : 0
	},
	"headers" :
	{
		"episode_id" : "1",
		"sent_at" : 1481999938.4742091
	},
	"method" : "v0.env.reward"
}
```

## Directly connected machines
To connect your Windows and Ubuntu machines directly via ethernet, follow [these instructions](http://askubuntu.com/a/26770/158805) for adding the interface to Ubuntu.

Use the _Netmask_ provided by `ipconfig` in Windows for your ethernet interface.

## Contributors

Several people, without which, this project would not have been possible
```
Aitor Ruano
Alec Radford
Alex Ray
Andrej Karpathy
Andrew Gray
Brad Templeton
Catherine Olsson
Christina Araiza
Dan Van Boxel
Dario Amodei
Felipe Code
Greg Brockman
Gustavo Younberg
Ilya Sutskever
Jack Clark
James Denney
Jeremy Schlatter
Jon Gautier
Jonas Schneider
Krystian Gebis
Lance Martin
Ludwig Pettersson
Matthew Kleinsmith
Matthew O'Kelly
Mike Roberts
Paul Baumgart
Pieter Abbeel 
Rafal Jozefowicz
Tom Brown
Vicki Cheung
```
