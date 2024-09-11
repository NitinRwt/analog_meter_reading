

# Analog Meter Reading with CNN and Gradio Interface

This project uses a Convolutional Neural Network (CNN) model to predict the reading of an analog meter from an image input. The model has been trained to detect the position of the needle on the meter and translate that into a corresponding reading. A simple web-based interface is provided using Gradio for easy interaction.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to automate the reading of analog meters using a machine learning approach. The CNN model is trained on images of analog meters to detect the position of the needle and output the corresponding reading.

Gradio is used to create a user-friendly GUI where users can upload an image of an analog meter, and the system will output the predicted reading.

## Features
- Upload images of analog meters and get the predicted reading of the needle.
- CNN-based prediction model for accurate readings.
- User-friendly web-based interface using Gradio.

## Prerequisites
To run this project, you will need to have the following installed:
- Python 3.x
- TensorFlow (>= 2.0)
- Gradio (>= 2.0)
- Pillow (>= 8.0)

Install the dependencies using:
```bash
pip install tensorflow gradio pillow
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/NitinRwt/analog_meter_reading.git
```
2. Navigate to the project directory:
```bash
cd analog_meter_reading
```
3. Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

4. Ensure you have the trained CNN model (`CNN_Analog-Readout_Version1.h5`). If not, follow the steps in [Model Training](#model-training) to train the model.

## Usage
1. Run the Gradio interface:
```bash
python app.py
```
2. Open the provided link in your browser to access the Gradio interface.

3. Upload an image of an analog meter, and the system will output the predicted reading.

### Running the GUI
The GUI interface built using Gradio allows you to:
- Upload an analog meter image.
- Get the predicted reading.

### Example:
![Example Interface]![Screenshot (150)](https://github.com/user-attachments/assets/ad516314-e3ec-44dd-b0eb-eb8050ef7fb2)


## Model Training
The model is trained using a Convolutional Neural Network (CNN) on a dataset of analog meter images. The training steps are detailed in the `Train_CNN_Analog.ipynb` notebook.

To retrain the model:
1. Open the Jupyter notebook `Train_CNN_Analog.ipynb`.
2. Follow the steps for preprocessing, training, and saving the model.
3. The trained model will be saved as `CNN_Analog-Readout_Version1.h5`.

## File Structure
```
|-- app.py                      # Gradio interface code
|-- Train_CNN_Analog.ipynb       # Jupyter notebook for training the CNN model
|-- CNN_Analog-Readout_Version1.h5  # Pretrained model file
|-- requirements.txt             # Project dependencies
|-- README.md                    # Project documentation
|-- images/                      # Sample images
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.
