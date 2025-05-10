# Real-time Vehicle Detection and Tracking with YOLO11 🚙🚕
Vehicle-Detection-and-Tracking-using-YOLO11  represents a personal exploration into real-time vehicle detection and tracking. Built with Ultralytics' YOLO11 framework, this proof of concept aims to showcase YOLO's potential for identifying and following vehicles in visual media.


- 📍 Detect vehicles in real-time video streams or from existing video files.
- 🚀 Track the location and movement (downwards or upwards) of vehicles frame by frame.
- 🚀 Investigate vehicle classification by category (cars, trucks etc).
- 🔍 This proof of concept can form the basis for applications in traffic analysis, surveillance, and automated vehicle monitoring.
- 🔍 Explore the capabilities of YOLO11 for accurate object detection in dynamic settings.

This project serves as a demonstration of YOLO's robust object detection abilities in practical applications like traffic monitoring and video analytics. Though not a finished product, its primary aim is to validate the chosen approach and inform future development efforts.

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
2. Copy the video in the created folder. I have already uploaded the video [test-video.mp4]{https://github.com/MsLearner-py/Vehicle-Detection-and-Tracking-using-YOLO11/blob/main/test-video.mp4}
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
7. In this tutorial we are using "yolo11s-cls.pt" pretrained model of YOLO, which is used for classification. You can download any version of YOLO classification model version from the [link](https://github.com/ultralytics/ultralytics).
8. Keep this downloaded file in YOLO-FOOD-CLASSIFICATION folder.
9. To install jupyter notebook type the command--
    
        pip install jupyter notebook
10. To install jupyter notebook type the command--
    
        pip install ultralytics 
11. After installation, you can launch Jupyter Notebook with the command--

        jupyter notebook
12. Create a new file (in my case it is PythonCode) and execute the cells

!pip install ultralytics
import cv2
from ultralytics import YOLO
from collections import defaultdict

[//]: Load the YOLO model
model = YOLO('yolo11n.pt')
class_list = model.names 
class_list

# Open the video file

cap = cv2.VideoCapture('test-video.mp4')
import cv2
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict

# Load the YOLO model
model = YOLO('yolo11n.pt')

class_list = model.names  # List of class names

# Open the video file

cap = cv2.VideoCapture('test-video.mp4')


# Define line positions for counting
line_y_red = 298  # Red line position
line_y_blue = line_y_red + 100  # Blue line position


# Variables to store counting and tracking information
counted_ids_red_to_blue = set()
counted_ids_blue_to_red = set()

# Dictionaries to count objects by class for each direction
count_red_to_blue = defaultdict(int)  # Moving downwards
count_blue_to_red = defaultdict(int)  # Moving upwards

# State dictionaries to track which line was crossed first
crossed_red_first = {}
crossed_blue_first = {}



# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True)

    # Ensure results are not empty
    if results[0].boxes.data is not None:
        # Get the detected boxes, their class indices, and track IDs
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()


        # Draw the lines on the frame
        cv2.line(frame, (190, line_y_red), (850, line_y_red), (0, 0, 255), 3)
        cv2.putText(frame, 'Red Line', (190, line_y_red - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.line(frame, (27, line_y_blue), (960, line_y_blue), (255, 0, 0), 3)
        cv2.putText(frame, 'Blue Line', (27, line_y_blue - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        
        # Loop through each detected object
        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)

            cx = (x1 + x2) // 2  # Calculate the center point
            cy = (y1 + y2) // 2
            
            # Get the class name using the class index
            class_name = class_list[class_idx]

            # Draw a dot at the center and display the tracking ID and class name
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            
            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 


            # Check if the object crosses the red line
            if line_y_red - 5 <= cy <= line_y_red + 5:
                # Record that the object crossed the red line
                if track_id not in crossed_red_first:
                    crossed_red_first[track_id] = True

            # Check if the object crosses the blue line
            if line_y_blue - 5 <= cy <= line_y_blue + 5:
                # Record that the object crossed the blue line
                if track_id not in crossed_blue_first:
                    crossed_blue_first[track_id] = True



            # Counting logic for downward direction (red -> blue)
            if track_id in crossed_red_first and track_id not in counted_ids_red_to_blue:
                if line_y_blue - 5 <= cy <= line_y_blue + 5:
                    counted_ids_red_to_blue.add(track_id)
                    count_red_to_blue[class_name] += 1
    
            # Counting logic for upward direction (blue -> red)
            if track_id in crossed_blue_first and track_id not in counted_ids_blue_to_red:
                if line_y_red - 5 <= cy <= line_y_red + 5:
                    counted_ids_blue_to_red.add(track_id)
                    count_blue_to_red[class_name] += 1
    
    # Display the counts on the frame
    y_offset = 30
    for class_name, count in count_red_to_blue.items():
        cv2.putText(frame, f'{class_name} (Down): {count}', (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y_offset += 30

    y_offset += 20  # Add spacing for upward counts
    for class_name, count in count_blue_to_red.items():
        cv2.putText(frame, f'{class_name} (Up): {count}', (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        y_offset += 30



    
    # Show the frame
    cv2.imshow("YOLO Object Tracking & Counting", frame)

    # Exit loop if 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()












       


# Output (in runs folder)

Read more about 
- [Confusion Matrix](https://www.datacamp.com/tutorial/what-is-a-confusion-matrix-in-machine-learning) 
- [Epochs, Iteration and batch](https://medium.com/@akankshaverma136/epochs-batch-and-iterations-in-deep-learning-ed319565e85e)

![confusion_matrix_normalized](https://github.com/user-attachments/assets/e2cfb7f5-8555-49e7-bac2-6fbe8927f0c5)
![results](https://github.com/user-attachments/assets/d07d8924-375e-4120-aabc-5a4874b198aa)
![confusion_matrix](https://github.com/user-attachments/assets/cfaff543-db7f-4087-8b8f-9403f41ea3c9)


epoch	time	train/loss	metrics/accuracy_top1	metrics/accuracy_top5	val/loss	lr/pg0	lr/pg1	lr/pg2
1	1282.13	1.12327	0.85627	0.9895	0.45432	0.000221973	0.000221973	0.000221973
2	2637.63	0.61349	0.85773	0.9898	0.47545	0.000389323	0.000389323	0.000389323
3	3932.97	0.55246	0.86706	0.9898	0.4217	0.000501646	0.000501646	0.000501646
4	5334.54	0.46953	0.88105	0.99446	0.38488	0.000419376	0.000419376	0.000419376
5	6743.64	0.39123	0.89563	0.99359	0.33717	0.000336835	0.000336835	0.000336835
6	8026.88	0.31035	0.90554	0.99475	0.30944	0.000254294	0.000254294	0.000254294
7	9172.25	0.23643	0.92012	0.99679	0.26922	0.000171752	0.000171752	0.000171752
8	10363.8	0.18238	0.92449	0.99679	0.23753	8.92E-05	8.92E-05	8.92E-05
![image](https://github.com/user-attachments/assets/bc3da640-1252-4c1c-9bfc-f051eb87f18c)


# Classified output images:

![0](https://github.com/user-attachments/assets/f3278888-1729-414c-a67f-3ebacd48aa8d)
![10](https://github.com/user-attachments/assets/228fbdaf-39c7-4f18-8b5f-c5437d1ee3b2)
![11](https://github.com/user-attachments/assets/529a5718-fe09-4b59-ba18-d68a0d447bfe)
![4](https://github.com/user-attachments/assets/857cbfdd-044b-424f-88a7-f94273428b8e)








              
         
   
