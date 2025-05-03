# Cattle Identification System

This project is a web-based application for identifying cattle using muzzle patterns. It allows users to register cattle by uploading images and recognize cattle by comparing uploaded images with the registered data.

## Features

- **Cattle Registration**: Upload an image and assign a unique ID to register a cattle.
- **Cattle Recognition**: Upload an image to identify a cattle by comparing it with registered data.
- **Results Display**: View annotated images, cropped regions, and similarity scores.

## Project Structure

├── app.py
       # Main application file 
├── README.md 
       # Project documentation
├── models/
 │ └── siamese_best_model.pth
       # Pre-trained model for cattle identification
├── static/
 │ ├── scripts.js
       # JavaScript for client-side interactivity
 │ ├── styles.css # CSS for styling the application
 │ └── uploads/ 
       # Directory for uploaded images
 ├── templates/
 │ └── index.html 
       # HTML template for the web interface     
    


# Usage
1. Start the application:
    python app.py

2. Open your browser and navigate to http://127.0.0.1:5000.

3. Use the Register Form to register cattle or the Recognize Form to identify cattle.


## File Descriptions

- **`app.py`**: The main application file containing the Flask application logic.
- **`README.md`**: Documentation for the project, including setup instructions and usage details.
- **`models/siamese_best_model.pth`**: Pre-trained Siamese model used for cattle identification.
- **`static/scripts.js`**: JavaScript file for client-side interactivity, such as handling form submissions and displaying results.
- **`static/styles.css`**: CSS file for styling the web interface.
- **`static/uploads/`**: Directory for storing uploaded images.
- **`templates/index.html`**: HTML template for the web interface, defining the structure of the application pages.

