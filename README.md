# Bass Connections: Energy Data Analytics Lab 2017-2018


Welcome to the Energy Data Analytics Lab 2017-2018 Bass Connections Project!

Key Files 
1. utils.py - This library of fully documented functions will be useful for extracting and preprocessing Indian village data, as well as evaluating model performances. 
2. Feature Extraction Main - iPython Notebook that goes hand-in-hand with above utils.py to extract village features. 
3. bce.py - Library of preprocessing and classification functions 
4. Models Test - Applies scaling, upsampling and then prepares data for classification and provides code for testing a variety of models. 
4. Various Demo files show experimentations with specific techniques 


Over the duration of the academic year, our team built an **end-to-end feature extraction and machine learning pipeline** for predicting a **village's electrification level** using various kinds of satellite imagery. Our pipeline streamlines converting over 30 GB of raw image data into desired **feature statistics**, provides frameworks for subsequent**feature selection, dimensionality reduction**, and **classification**. At the time of this writing in April 2018, the optimal model for larger villages (400+ households) predicts a village-level electrification class, electrified or unelectrified, for the Indian state of Bihar with an AUC of almost 0.80. This is despite a high dimensional feature space, limited labeled data to begin with, as well as massive class-imbalance in the dataset, which had more than 99% observations being of electrified villages, and less than 1% of all observations being of the target unelectrified class.   
In the end our chosen model was a Gradient Boosting Classifier, and as this is a powerful supervised learning technique we are excited at the potential improvements to be seen with more training and more labeled data. As India completes more on-the-ground surveyal of Indian villages, more ground-truth village electrification data will be made available which can be used to train our model. Currently, we have trained on the Indian state of Bihar, and thus our model will have limited generalizability and may struggle classifying electrifications in areas that look different from Bihar. This brings room for experimentation and improvement and we're excited to see what the next Bass Connections team might come up with- we hope you find our code and work useful in your exploration of this problem. Thank you so much for reading! If you have any questions, feel free to reach out to any of the members listed below. Code-specific questions can be directed to McCourt Hu, Shamikh Hossain and Brian Wong. 

Technologies Used
This work would not be possible without amazing and free open-source data science libraries and front-end development tools. 
1. Python for Data Science ecosystem for data analysis (jupyter notebook, numpy, pandas, skimage)
2. matplotlib, seaborn for data visualization 
3. Google Earth Engine
4. VIIRS Imagery

Recognition 
1. Our research poster won 1st Place at the 2018 Duke University Research Computing Symposium!
2. We receieved an Honorable Mention at the 2018 State Energy Conference of North Carolina
3. Our team leader, Brian Wong, was awarded the Bass Connections Award for Outstanding Mentorship!