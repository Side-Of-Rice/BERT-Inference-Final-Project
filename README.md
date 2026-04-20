**Project Overview**

My project will be a supervised text classification problem using Airbnb datasets found at the Inside Airbnb website. Guests who rent a house for their trip can leave a review after their stay. They can write a comment (unstructured text) as well as give a one-to-five-star rating. It is said that Airbnb listings that score an average of 4.8 or higher are considered top performers. I want to build a model that can determine whether a written review belongs to a top performing listing or underperforming listing. Because I’ve done text classification problems before in my other classes, I’m going to add the challenge of deploying a transformer-based embedding model, BERT, to capture the deep semantic meaning with a neural network. Due to time constraint, I will only execute a fraction of the million observations for training and testing (see code for further details). 

**How to run entire pipeline (ingestion, validation, train, evaluate)**

1. Download all files in folder.

2. Go to Command Terminal.

3. Change directory to project-root folder (cd)

4. Type "dvc repro".






**How to run inference service**

1. Download all files in folder.

2. Go to Command Terminal.

3. Change directory to inference folder (cd)

4. Type "python predict.py"
