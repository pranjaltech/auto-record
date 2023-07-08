# auto-record
All-purpose auto-record. Made this project using a Raspberry Pi (or a similar device) and a usb-mic to record an acoustic piano. 

## Description
This is a Python script that can be run as a Unix service on any linux-based platform. The script connects to a usb-microphone and keeps listening to it's input. If the input goes above a certain loudness level, it starts recording. Once it encounters a silence (audio signal below the threshold) for 5 seconds, it stops recording and saves the file — either locally or on Google Drive. 

I made this script to run on a Raspberry Pi 4, using a Zoom H1N USB Recorder — which is placed inside my Kawai Upright Piano. Whenever the piano is played, the script saves the audio clip to a Google Drive folder automatically. 

As a result, I never have to worry about pressing record while playing. 

## Installation
You are going to need Python 3 on your system. [See how to install python on your machine](/todo/). 

Clone the repository to your local machine and navigate to the cloned repository. 
```bash
git clone { repo URL }
cd auto-record
```

Install the required python packages. 
```bash
python3 -m pip install -r requirements.txt
```

To configure the parameters for the app, open `config.yml` in an editor. 
```yml
- threshold: 100
- lol: something
```

Try running the script now. 
```bash
python3 record.py
```

You will see the following output —
```
Listening... 
```

### (Optional) Install the script as a UNIX service 
```
chmod +x install_service.sh
./install_service.sh
```

## Usage

## Future Goals

## Contributing

## Tests

## License
