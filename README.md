# Real-time Vehicle Detection and Tracking with YOLO11 🚙🚕
Vehicle-Detection-and-Tracking-using-YOLO11  represents a personal exploration into real-time vehicle detection and tracking. Built with Ultralytics' YOLO11 framework, this proof of concept aims to showcase YOLO's potential for identifying and following vehicles in visual media.


- 📍 Detect vehicles in real-time video streams or from existing video files.
- 🚀 Track the location and movement (downwards or upwards) of vehicles frame by frame.
- 🚀 Investigate vehicle classification by category (cars, trucks etc).
- 🔍 This proof of concept can form the basis for applications in traffic analysis, surveillance, and automated vehicle monitoring.
- 🔍 Explore the capabilities of YOLO11 for accurate object detection in dynamic settings.

This project serves as a demonstration of YOLO's robust object detection abilities in practical applications like traffic monitoring and video analytics. Though not a finished product, its primary aim is to validate the chosen approach and inform future development efforts.

## Table of Contents

1. [Real-time Vehicle Detection and Tracking with YOLO11](#real-time-vehicle-detection-and-tracking-with-yolo11-)
2. [Ultralytics YOLO11](#ultralytics-yolo11)
    - [Modes at a Glance](#modes-at-a-glance)
3. [How vehicle detection using YOLO11 works](#how-vehicle-detection-using-yolo11-works-)
4. [Let's Start](#lets-start)
5. [Output](#output)
6. [Acknowledgments](#acknowledgments-)
    - [Project link](#project-link-httpsgithubcommslearner-pyvehicle-detection-and-tracking-using-yolo11treemain)

*Note: Each entry links to the corresponding section header in the README.*


# Ultralytics YOLO11
Ultralytics YOLO11 is the the latest version of the acclaimed real-time object detection and image segmentation model. YOLO11 is built on cutting-edge advancements in deep learning and computer vision, offering unparalleled performance in terms of speed and accuracy. Its streamlined design makes it suitable for various applications and easily adaptable to different hardware platforms, from edge devices to cloud APIs.

[//]:![image](https://github.com/user-attachments/assets/3d1e8650-3e49-49e6-bd3c-35c87b006bbd)

Ultralytics supports a wide range of YOLO models, from early versions like YOLOv3 to the latest YOLO11. YOLO11 models are pretrained on the COCO dataset for Detection, Segmentation, and Pose Estimation. Additionally, Classification models pretrained on the ImageNet dataset are available. Tracking mode is compatible with all Detection, Segmentation, and Pose models.

[Read more about ULTRALYTICS here!](https://github.com/ultralytics/ultralytics)

## Modes at a Glance
Understanding the different modes that Ultralytics YOLO11 supports is critical to getting the most out of your models:

- **Train mode**: Fine-tune your model on custom or preloaded datasets.
- **Val mode**: A post-training checkpoint to validate model performance.
- **Predict mode**: Unleash the predictive power of your model on real-world data.
- **Export mode**: Make your model deployment-ready in various formats.
- **Track mode**: Extend your object detection model into real-time tracking applications.
- **Benchmark mode**: Analyze the speed and accuracy of your model in diverse deployment environments.

![image](https://github.com/user-attachments/assets/569a9972-9273-475f-8932-2bf890ba38a9)


# How vehicle detection using YOLO11 works 🚀

1. Image/video Input:
YOLO11 processes video frames or still images as input. 
2. Feature and frame Extraction:
The video is a collection of sequence of frames. The model works on each individual frame. The model's architecture, including a backbone and neck, extracts relevant features from the sequence of frames. 
3. Multi-Scale Detection:
YOLO11 employs detection layers at different scales (P3, P4, and P5) to handle objects of varying sizes, ensuring accurate detection of both large and small vehicles. 
4. Object Detection:
YOLO11 identifies potential objects (vehicles) within the frames and predicts their bounding boxes. 
5. Classification:
It classifies the detected objects into different categories, such as car, truck, etc. 
6. Bounding Box Output:
YOLO11 generates bounding boxes around the detected vehicles, providing their location and size within the image. 
7. Tracking (Optional):
In some applications, like autonomous driving, YOLO11 can be used in conjunction with object tracking algorithms to follow vehicles across multiple frames, enabling the system to understand their movement and predict their future position.
8. Data output:
The processed data, including vehicle attributes and annotated frames, can be stored, analyzed, or further processed according to the requirements.
 


# Let's Start
1. Create a folder "[your folder name]"
2. Copy the video in the created folder. I have already uploaded the video [test-video.mp4](https://github.com/MsLearner-py/Vehicle-Detection-and-Tracking-using-YOLO11/blob/main/test-video.mp4)
3. Open Command prompt (cmd) and navigate to the folder that you have created.
4. Make sure that python is installed.
5. Create and activate Virtual environment (myenv) for windows: 

       Step 1: --Installing virtualenv through pip--
               pip install virtualenv
   
       Step-2: --Creating a virtualenv--
               python -m virtualenv myenv
     
       Step-3: -- Activate the virtual environment--
               myenv\Scripts\activate
   
6. Now you will find a new folder "myenv" created in the folder.
7. In this tutorial we are using "yolo11n.pt" pretrained model of YOLO11.. You can download any version of YOLO model version from the [link](https://github.com/ultralytics/ultralytics).
8. Keep this downloaded file in the created folder if not automatically downloaded.
9. To install jupyter notebook type the command--
    
        pip install jupyter notebook
10. To install jupyter notebook type the command--
    
        pip install ultralytics 
11. After installation, you can launch Jupyter Notebook with the command--

        jupyter notebook
12. Create a new file (in my case it is PythonCode) and execute the cells.
    
# Output
<img width="741" alt="output" src="https://github.com/user-attachments/assets/e00597e1-2292-4b59-90e0-07456e42e7d5" />

# Acknowledgments 🙏
- This package is powered by ultralytics YOLO for object detection.

## References

1. Ultralytics YOLO Documentation  
   [https://docs.ultralytics.com/](https://docs.ultralytics.com/)

2. Ultralytics YOLO GitHub Repository  
   [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

3. YOLO: You Only Look Once — Real-Time Object Detection Paper  
   Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016).  
   [https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640)

4. COCO: Common Objects in Context Dataset  
   Lin, T.-Y., Maire, M., Belongie, S., et al. (2014).  
   [https://cocodataset.org/](https://cocodataset.org/)

5. Virtualenv Documentation  
   [https://virtualenv.pypa.io/en/latest/](https://virtualenv.pypa.io/en/latest/)

6. Jupyter Notebook Documentation  
   [https://jupyter-notebook.readthedocs.io/en/stable/](https://jupyter-notebook.readthedocs.io/en/stable/)



   
