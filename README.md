# TiresIA
[Demo Video](https://drive.google.com/file/d/1Rv6MqSHVn1b-dUzVfuW7jEBa2mHf7qMs/view?usp=sharing) 

## Setu up

Clone the repo.

```
git clone https://github.com/braillefly/Blind_navigator.git
cd Blind_navigator
```
create a virtual environment
```
python -m venv venv
```
Activate the environment and install the required packages.

```
pip install openai PyQt5 opencv-python
```

## Run 

Run with a video 

```
python LLAMA_3_2_navigation.py --video <video path> --api <api key>
```
Run with a camera

```
python LLAMA_3_2_navigation.py --video <int correpsonding to the camera> --api <api key>
```
