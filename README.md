# Grip-Classification
## Table of Contents
* [Project Overview](#project-overview)
* [Materials Required](#materials-required)
* [Setup](#setup)
* [Implementation](#implementation-instructions)
## Project Overview
When performing different types of hand grips, different forearm muscles are activated at different intensities. The objective of this project was to develop a portable device that can classify and predict types of hand grips in real time using surface electromyography signals (sEMG) on the flexor digitorum profundus. After collecing data on the muscle activity during 6 hand grip types (spherical, palmer, tip, hook, lateral, and cylindrical), a machine learning model was trained to accurately classify the particular hand grip being performed. This project utilizes a single bipolar EMG sensor and an ESP32 to collect the electrical signals of the muscle contractions. The model was implemented on a LilyGo ESP32 T-Display, making a compact and wearable, real-time, machine learning-powered EMG device.

## Materials Required 
* Muscle BioAmp BisCute
* Electrodes (x3)
* LilyGo ESP32 T-Display
* ESP32
* Arduino IDE
* Google Collab/Google Drive

## Setup
### Data Collection 
1. Apply conductive electrode gel
2. The Muscle BioAmp BisCute has (1) a positive inverting (inv +), (2) a negative inverting (inv -), and (3) a reference connection that each connect directly to the electrodes for recieving signals. The reference electrode should be placed on the palm of your hand because it is __. The inv + and inv - placements should be at each end of the flexor digitorum profundus. The inv - side should be where the muscle inserts, and the inv + should be where the muscle begins. This is because __.
To export these signals the Muscle BioAmp BisCute has (1) a ground, (2) a voltage supply, and (3) an output __. These are connected to the ESP32 which is powered through a laptop/computer. Ground is connected to the ground pin. The voltage supply is connected to 5V, and the output is connected to any analog pin.
The image below displays a visual of the setup. 
<img width="2880" height="1620" alt="image" src="https://github.com/user-attachments/assets/1b345874-f2c7-40e4-a7a0-338addeb70ff" />
[https://github.com/upsidedownlabs/Muscle-BioAmp-BisCute]

### Verification 
1. Apply conductive electrode gel
2. Follow the same instructions from [data collection setup](#data-collection), except the microcontroller will now be a LilyGo ESP32 T-Display. The connections are the same, but the microcontroller is slightly different, and it provides a display screen. This allows for portable display. THe LilyGo ESP32 T-Display should be connected to the computer and person. 


## Implementation Instructions 
### Using Existing Data 
1. Download [collected grip data](/Grip_Data_(Collected)/) onto your laptop and upload the folder to Google Drive. Ensure the folder is labeled *Data* within the Google Drive. 

### Collecting Data
1. Download the [data collection](/src/Data_Collection/) folder and upload the .ino file to Arduino IDE.
2. Follow the inital [hardware setup](#setup) instructions for [data collection](#data-collection). 
4. Perform a hand grip (spherical, palmer, tip, hook, lateral, or cylindrical) and hold that position. 
5. Flash the code onto the ESP32, and allow data to continuously collect for 5000 samples.
6. When data stops collecting, the hand grip can be released.  
7. Export the data as a .csv file, and name the file after the specific grip being performed (e.g., Spherical.csv).
8. Complete this 5 times with each hand grip.
9. Put all of the .csv files into a folder labeled *Data* and upload it into Google Drive.

### Training Model
1. [Collect data](#collecting-data) **OR** [use existing data](#using-existing-data) and ensure that it is in Google Drive.
2. Download the *Model_Training_Grip.ipynb* file from the [src](/src/) folder and upload it to Google Collab.
3. Run the program to obtain a trained model (.tflite), quantizaiton parameters, classification accuracy, and a confusion matrix.
- The .tflite file is a quantized version of the TensorFlow model and is able to fit onto a microcontroller. It will automatically download when *Model_Training_Grip.ipynb* is run. 

### Model Implementation 
1. Follow the [hardware setup](#verification) instructions for verifying the model.
2. Download the [grip classification folder](/src/Grip_Classification/) folder and open it within Arduino IDE. 
3. Download the *grip_model_s.tflite* file from the [src](/src/) folder **OR** [train a model](#training-model) to obtain a .tflite file.
4. Bash the file using "xxd -i grip_model_s.tflite > grip_model.h" to turn the .tflite file into a .h file.
5. Download the *User_Setup_Library* within the [src](/src/) folder. 
6. Manually change the *TFT_eSPI library* to be the downloaded *User_Setup_Library*
7. Ensure the .h file is within the same folder as the .ino, ensure all of the proper libraries are downloaded, and that the proper device is selected.
8. Flash the code onto the LilyGo ESP32 T-Display. Perform different hand grips and watch the classification along with the model's confidence. 
