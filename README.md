# Smart-Signature-Verification-

Link to the deployed App on *Heroku Platform* : https://lit-waters-92367.herokuapp.com/

**TOOLS USED**
1. Python : Overall Web-App
2. Django : Python Framework for backend
3. Heroku : Web-App Deployment Platform
4. OpenCV : Image Pre-processing
5. Tesseract : Text Extraction 
6. Tensorflow : Signature Verification Model
7. Keras : Signature Verification and OCR Model
8. HTML+CSS : Front-End
9. Javascript : Front-End
10. Ajax : Make Request to Web-Server
11. SQLite3 : Database Management System

**IDEA**
Current methods in machine learning and statistics have allowed for the reliable automation of many of these tasks (face verification, fingerprinting, iris recognition). Among the numerous tasks used for biometric authentication is signature verification, which aims to detect whether a given signature is genuine or forged.
Signature verification is essential in preventing falsification of documents in numerous financial, legal, and other commercial settings. The task presents several unique difficulties: high intra-class variability (an individual’s signature may vary greatly day-to-day), large temporal variation (signature may change completely over time) and high inter-class similarity (forgeries, by nature, attempt to be as indistinguishable from genuine signatures as possible). Our aim is to develop an AI based signature verification system that is scalable and can be easily used to detect fraud cases in various public and private sector settings such as banks, post-offices, government offices and co-operatives too and reduce manual effort.
Our idea is to extract the previously stored signature from the system using the concept of image segmentation using OpenCV and Tesseract and then apply deep learning model that we have developed and classify if the signature under scrutiny is right and matching or falsified. In the signature classification model, pre-processing of the dataset is very important wherein have done the noise removal and property adjustment part(angular rotation, resizing and exact position detection). In the signature classification model we would be using Convolutional Neural Networks and Transfer Learning techniques. Our final task will be to ensure that the model that we have developed is scalable and can be used across all the places of interests easily and can be used by even the common people without knowing indepth about the system.

**FRAMEWORK**



