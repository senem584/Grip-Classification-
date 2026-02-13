# Hand Grip Classification

## Table of Contents
* [Project Overview](#project-overview)
* [Materials Required](#materials-required)
* [Requirements](#requirements)
* [Setup](#setup)
  * [Data Collection](#data-collection)
  * [Verification](#verification)
* [Implementation](#implementation-instructions)
  * [Using Existing Data](#using-existing-data)
  * [Collecting Data](#collecting-data)
  * [Training Model](#training-model)
  * [Model Implementation](#model-implementation)

## Project Overview
When performing different types of hand grips, different forearm muscles are activated at different intensities. The objective of this project was to develop a portable device that can classify and predict types of hand grips in real time using surface electromyography signals (sEMG) on the flexor digitorum profundus. After collecting data on the muscle activity during 6 hand grip types (spherical, palmer, tip, hook, lateral, and cylindrical), a machine learning model was trained to accurately classify the particular hand grip being performed. This project utilizes a single bipolar EMG sensor and an ESP32 to collect the electrical signals of the muscle contractions. The model was implemented on a LilyGo ESP32 T-Display, making a compact and wearable, real-time, machine learning-powered EMG device.

## Materials Required 
* Muscle BioAmp BisCute
* Electrodes (x3)
* LilyGo ESP32 T-Display
* ESP32
* Arduino IDE
* Google Drive (& Google Colab)

## Requirements 
Use `pip install -r requirements.txt` to install the necessary dependencies. 

## Setup

### Data Collection 
1. Apply conductive electrode gel (or use pre-gelled electrodes if applicable).
2. The Muscle BioAmp BisCute has (1) a positive inverting (INV+), (2) a negative inverting (INV-), and (3) a reference (REF) connection that each connect directly to the electrodes for receiving signals.  
   * The **reference electrode** should be placed on the **palm** because it is typically a relatively low-activity area compared to the forearm muscle being measured, which helps stabilize the differential measurement and reduce noise.  
   * The **INV+ and INV-** electrodes should be placed along the **flexor digitorum profundus**, spaced a few centimeters apart and aligned with the muscle fibers. The INV- side can be placed closer to where the muscle inserts and INV+ closer to where it begins mainly for **consistent placement** across trials. In practice, swapping INV+ and INV- mostly flips signal polarity, so consistency matters more than which side is “top vs bottom.”
3. To export these signals, the Muscle BioAmp BisCute has (1) **GND**, (2) a **voltage supply (VCC)**, and (3) an **analog output (OUT)**. These are connected to the ESP32 which is powered through a laptop/computer.  
   * **GND → ESP32 GND**  
   * **VCC → 5V**  
   * **OUT → an analog/ADC pin** (any supported ESP32 analog input pin)
4. The image below displays a visual of the setup.  
<img width="2880" height="1620" alt="image" src="https://github.com/user-attachments/assets/1b345874-f2c7-40e4-a7a0-338addeb70ff" />

Sensor reference:  
[https://github.com/upsidedownlabs/Muscle-BioAmp-BisCute](https://github.com/upsidedownlabs/Muscle-BioAmp-BisCute)

### Verification 
1. Apply conductive electrode gel.
2. Follow the same instructions from [data collection setup](#data-collection), except the microcontroller will now be a **LilyGo ESP32 T-Display**. The connections are the same, but the microcontroller includes a display screen for portable output. The LilyGo ESP32 T-Display should be connected to the computer (power + programming) and the person (electrodes + sensor).

## Implementation Instructions 

### Using Existing Data 
1. Download [collected grip data](/Grip_Data_(Collected)/) onto your laptop and upload the folder to Google Drive. Ensure the folder is labeled *Data* within Google Drive. 

### Collecting Data
1. Download the [data collection](/src/Data_Collection/) folder and open the `.ino` file in Arduino IDE.
2. Follow the initial [hardware setup](#setup) instructions for [data collection](#data-collection). 
3. Perform a hand grip (spherical, palmer, tip, hook, lateral, or cylindrical) and hold that position steadily. 
4. Flash the code onto the ESP32, and allow data to continuously collect for **5000 samples**.
5. When data stops collecting, the hand grip can be released.  
6. Export the data as a `.csv` file, and name the file after the specific grip being performed (e.g., `Spherical.csv`).
7. Complete this **5 times** with each hand grip.
8. Put all of the `.csv` files into a folder labeled *Data* and upload it into Google Drive.

### Training Model
1. [Collect data](#collecting-data) **OR** [use existing data](#using-existing-data) and ensure that it is in Google Drive.
2. Download the `Model_Training_Grip.ipynb` file from the [src](/src/) folder and upload it to Google Colab.
3. Run the notebook to obtain a trained model (`.tflite`), quantization parameters, classification accuracy, and a confusion matrix.  
   * The `.tflite` file is a **quantized TensorFlow Lite model** designed to fit onto a microcontroller. It will automatically download when `Model_Training_Grip.ipynb` is run.

### Model Implementation 
1. Follow the [hardware setup](#verification) instructions for verifying the model.
2. Download the [grip classification folder](/src/Grip_Classification/) and open it within Arduino IDE. 
3. Download the `grip_model_s.tflite` file from the [src](/src/) folder **OR** [train a model](#training-model) to obtain a `.tflite` file.
4. Convert the `.tflite` file into a header file using:
   * `xxd -i grip_model_s.tflite > grip_model.h`
5. Download the `User_Setup_Library` within the [src](/src/) folder. 
6. Manually change the `TFT_eSPI` library setup to use the downloaded `User_Setup_Library`.
7. Ensure `grip_model.h` is within the same folder as the `.ino`, ensure all required libraries are installed, and that the proper device/port is selected.
8. Flash the code onto the LilyGo ESP32 T-Display. Perform different hand grips and watch the classification along with the model's confidence.

A detailed paper on this project can be found within the [docs](/docs/) folder. Verification images (confusion matrix and model accuracy reports) can be found within the [media](/media/) folder.
