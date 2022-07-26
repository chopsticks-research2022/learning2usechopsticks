# Installation

We recommend you to create a [virtual enviroment](https://docs.python.org/3/library/venv.html) before you install the required packages.

### Operating system
1. [Windows](#installation-windows)
1. [Linux](#installation-linux)

## Installation (Windows)

### **mujoco-py**

First download Mujoco150 from [this page](https://www.roboti.us/download.html), then extract it to `%userprofile%\.mujoco`. 

Go download `mjkey.txt` from [this page](https://www.roboti.us/file/mjkey.txt) and save it into `%userprofile%\.mujoco\mjkey.txt` and `%userprofile%\.mujoco\mjpro150\bin\mjkey.txt`.

Add `%userprofile%\.mujoco\mjpro150\bin` into `PATH`.

Download and extract the source code for [mujoco-py(1.50.1.1)](https://github.com/openai/mujoco-py/releases/tag/1.50.1.1).

Make sure you have Visual Studio 2019 intalled, and also the Microsoft Visual C++ Build Tools 2019 installed before you proceed the next few steps.

Open the Visual Studio x64 Native Build Tools Command Prompt, and change the directory to `mujoco-py-1.50.1.1\`, activate your virtual environment, then run the following commands to install `mujoco-py`:
```
pip install -r requirements.txt
python setup.py install
```

### **pytorch3d**

First clone the latest [pytorch3d](https://github.com/facebookresearch/pytorch3d) repo via 
```
git clone https://github.com/facebookresearch/pytorch3d.git
```

Open the Visual Studio x64 Native Build Tools Command Prompt, change the directory to the `pytorch3d` repo, activate your virtual environment, then run the following commands:
```
python setup.py install
```

### **other dependencies**
Just run the following command to install the other dependencies.
```
pip install -r requirements.txt
```

## Installation (Linux)

### **mujoco-py**

First download Mujoco150 from [this page](https://www.roboti.us/download.html), then extract it to `~/.mujoco/`.

Go download `mjkey.txt` from [this page](https://www.roboti.us/file/mjkey.txt) and save it to `~/.mujoco/mjkey.txt` and `~/.mujoco/mjpro150/bin/mjkey.txt`.

Download and extract the source code for [mujoco-py(1.50.1.1)](https://github.com/openai/mujoco-py/releases/tag/1.50.1.1).

Change your directory to the source code for `mujoco-py-1.50.1.1`, activate your virtual environment, then run the following commands to install `mujoco-py`:
```
pip install -r requirements.txt
python setup.py install
```

### **pytorch3d**

Follow the installation guide on [pytorch3d](https://github.com/facebookresearch/pytorch3d) repo.

### **other dependencies**
Just run the following command to install the other dependencies.
```
pip install -r requirements.txt
```
