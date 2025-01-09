
# EchoShield

EchoShield is a comprehensive system that detects whether an audio file is human or AI-generated, analyzes its authenticity, and ensures the data's immutability through blockchain integration.


**This project is divided into two parts:**

1. **Frontend**: Responsible for the user interface and interaction, developed using React and TypeScript, with blockchain integration for minting NFTs and ensuring data immutability.
2. **Backend**: Handles backend processing, model training, and feature extraction, developed using Python (FastAPI, TensorFlow).

Each part is managed in a separate repository for better organization and maintenance.

**Please start with the Backend Repository first, followed by the Frontend Repository**.

- **Backend Repository**: [EchoShield-Backend](https://github.com/GVAmaresh/EchoShield-ML)
- **Frontend Repository**: [EchoShield-Frontend](https://github.com/GVAmaresh/EchoShield-Frontend)


## Table of Contents

1. [Demo](#demo)
2. [Technology Used](#technology-used)
3. [Description](#description)
4. [Getting Started](#getting-started)
    1. [Prerequisites](#prerequisites)
    2. [Using CMD (Command Line)](#using-cmd-command-line)
    3. [Using Docker](#using-docker)
5. [Meet the Team](#meet-the-team)
6. [Benefits](#benefits)
7. [License](#license)

## Demo
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/PC9OfqpwY7M/0.jpg)](https://www.youtube.com/watch?v=PC9OfqpwY7M)

## Technology Used

**Programming & Scripting Languages:**
- **Python**: For backend processing, model training, and feature extraction.
- **JavaScript**: For interactive frontend design and backend logic.
- **TypeScript**: Ensures type safety and robustness in the React components.

**Frameworks & Libraries:**
- **React**: Frontend framework for building user interfaces.
- **FastAPI**: Lightweight Python framework for serving the backend.
- **Hugging Face Transformers**: For integrating pre-trained models like wav2vec.

**Audio Processing:**
- **FFmpeg**: For audio conversion and preprocessing tasks.

**Machine Learning:**
- **TensorFlow**: Used for model implementation and training, focusing on deep learning for audio detection.
- **Model Architecture**: Sequential model with Conv2D for feature extraction, MaxPooling2D for downsampling, dense layers with ReLU, dropout for regularization, and softmax output.

**Ethereum:**
- For minting NFTs and ensuring data immutability.

**Other Tools:**
- **IPFS (InterPlanetary File System)**: To store audio files securely and immutably.


## Description

 * **Audio Detection**
EchoShield analyzes whether audio is human or AI-generated using advanced models like wav2vec, Melody Machine, and VGG16.

 * **Deepfake Detection**
The system uses feature extraction and entropy calculations to identify deepfake audio.

 * **Entropy Calculation**
EchoShield measures the unpredictability or randomness in the audio to assess its authenticity and determine its likelihood of being a deepfake.

 * **Metadata Creation**
The deepfake status, entropy value, and IPFS hash of the audio are compiled into metadata, providing a comprehensive overview of the audio's characteristics.

 * **IPFS Storage**
The audio file is securely stored on the IPFS network, providing a unique, decentralized identifier for each file.

 * **NFT Minting**
The metadata, including deepfake status, entropy, and IPFS hash, is minted as a non-fungible token (NFT) on the Ethereum blockchain.

 * **Immutability & Verification**
The minted NFT ensures that the audio and its associated data are immutable, verifiable, and securely linked to the blockchain, providing a trustworthy and transparent verification system.

    

## Getting Started

**Prerequisites**

Before getting started, make sure you have the following installed and set up:

Installing Python and Setting Up PATH on Windows

**1.Python**
- Visit the [official Python download page](https://www.python.org/downloads/) and download the latest version for Windows.

- **Important**: Ensure the **"Add Python to PATH"** option is checked and click **"Install Now"**
- Open Command Prompt (`Win + R`, type `cmd`) and check Python version:  
```bash
python --version
  ```

**2. Git**

Ensure that **Git** is installed and that you are logged in to GitHub.

Check if Git is installed:
```bash
git --version
```

If Git is not installed, download and install it from the official Git website:

- Download Git: [git-scm](https://git-scm.com/downloads)

- Git Docs: [git-scm.com/doc](https://git-scm.com/doc)


**3. Docker**

Ensure that Docker is installed on your machine if you plan to use Docker for running the project.

Check if Docker is installed:
```bash
docker --version
```

If Docker is not installed, download and install it from the official Docker website:

- Download Docker: [docker-desktop](https://www.docker.com/products/docker-desktop)
- Docker Docs: [docs.docker](https://docs.docker.com/)

---

**Running the Project**

**1. Using CMD (Command Line)**

First, clone the frontend and backend repositories using the following commands:

```bash
git clone https://github.com/GVAmaresh/EchoShield-ML
```

Navigate into the project directory and install the necessary dependencies.

```bash
cd EchoShield-ML
pip install -r requirements.txt

```
Once the installation is complete, you can start the frontend by running:

```bash
python main.py
```
Now you should be able to access the frontend on [http://localhost:3000](http://localhost:3000)

**2. Using Docker**

First, clone the frontend and backend repositories using the following commands:

```bash
git clone https://github.com/GVAmaresh/EchoShield-ML.git
```

To build the Docker images, first navigate into the projectand then build the Docker images

```bash
cd EchoShield-ML
docker build -t echoshield-backend .

```
After building the images, you can verify that they were created successfully using:
```bash
docker images
```

After building the images, run the containers for backend
```bash
docker run -d -p 8000:8000 echoshield-backend
```

Now you should be able to access the backend on [http://localhost:3000](http://localhost:8000)

## Meet the Team

- **G V Amaresh**: [GVAmaresh ](https://github.com/GVAmaresh)
- **Prateek Savanur**: [PrateekSavanur](https://github.com/PrateekSavanur)
- **Chinmayee G**: [Chi-nm](https://github.com/Chi-nm)
- **Charu Bohra**: [CharuBohra](https://github.com/CharuBohra)

## Model Information

 **1. wav2vec (Feature Extraction)**
- **Purpose**:  
  wav2vec is used for extracting high-level features from audio data. Specifically, we use it to convert raw audio signals into feature representations that can later be used by machine learning models to classify the audio as real or AI-generated.
  
- **How It's Applied**:  
  In this project, wav2vec is utilized to process the audio data before feeding it into the classification model. The extracted features capture the underlying patterns in the audio, helping to identify whether the audio is human or AI-generated.

- **Integration**:  
  The raw `.wav` files are passed through the wav2vec model, which generates a series of embeddings (features). These embeddings are then used for further analysis and classification.


**2. VGG16 (Convolutional Neural Network)**
- **Purpose**:  
  VGG16 is a well-known Convolutional Neural Network (CNN) architecture primarily used for image classification. However, in this project, we adapt VGG16 to classify audio features (extracted from wav2vec). By utilizing VGG16’s deep layers, the model can detect high-level patterns in the audio data and classify it effectively.

- **How It's Applied**:  
  After feature extraction by wav2vec, the processed features are fed into the VGG16 model. VGG16 processes the audio features (which are treated similarly to image data for classification tasks) to determine if the audio is real or fake.

- **Integration**:  
  The VGG16 architecture is fine-tuned on our audio dataset. The model learns to recognize patterns specific to human and AI-generated audio, helping to improve the accuracy of the classification task.


**3. Melody Machine (Audio Analysis)**
- **Purpose**:  
  Melody Machine is a model designed to analyze the melody and pitch of audio signals. It focuses on detecting melodic features in the audio that may vary between human and AI-generated speech, thus contributing to distinguishing the authenticity of the audio.

- **How It's Applied**:  
  This model is applied to assess the pitch, tone, and rhythm of the audio, as AI-generated voices often exhibit certain melodic patterns that differ from human speech. By analyzing these features, Melody Machine helps in the classification of the audio as either human or AI-generated.

- **Integration**:  
  Melody Machine processes the audio data alongside other feature extraction models. The output of the Melody Machine is used in conjunction with the wav2vec and VGG16 outputs for a more comprehensive classification decision.

## Benefits
- Accurate deepfake detection using state-of-the-art models.
- Secure and decentralized storage using IPFS.
- Immutable verification through blockchain technology.
- Provides transparent and verifiable metadata for each audio file.

## License
This project is licensed under the Apache-2.0 License - see the [LICENSE](https://github.com/GVAmaresh/EchoShield-ML/blob/main/LICENSE) file for details.
