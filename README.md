# Real-time Vehicle Detection and Tracking with YOLO11 🚙🚕
Vehicle-Detection-and-Tracking-using-YOLO11  represents a personal exploration into real-time vehicle detection and tracking. Built with Ultralytics' YOLO11 framework, this proof of concept aims to showcase YOLO's potential for identifying and following vehicles in visual media.


- 📍 Detect vehicles in real-time video streams or from existing video files.
- 🚀 Track the location and movement (downwards or upwards) of vehicles frame by frame.
- 🚀 Investigate vehicle classification by category (cars, trucks etc).
- 🔍 This proof of concept can form the basis for applications in traffic analysis, surveillance, and automated vehicle monitoring.
- 🔍 Explore the capabilities of YOLO11 for accurate object detection in dynamic settings.

# Ultralytics YOLO11
Ultralytics YOLO11 is the the latest version of the acclaimed real-time object detection and image segmentation model. YOLO11 is built on cutting-edge advancements in deep learning and computer vision, offering unparalleled performance in terms of speed and accuracy. Its streamlined design makes it suitable for various applications and easily adaptable to different hardware platforms, from edge devices to cloud APIs.

![image](https://github.com/user-attachments/assets/3d1e8650-3e49-49e6-bd3c-35c87b006bbd)

## Modes at a Glance
Understanding the different modes that Ultralytics YOLO11 supports is critical to getting the most out of your models:

- **Train mode**: Fine-tune your model on custom or preloaded datasets.
- **Val mode**: A post-training checkpoint to validate model performance.
- **Predict mode**: Unleash the predictive power of your model on real-world data.
- **Export mode**: Make your model deployment-ready in various formats.
- **Track mode**: Extend your object detection model into real-time tracking applications.
- **Benchmark mode**: Analyze the speed and accuracy of your model in diverse deployment environments.

# Objective:  Real-time Vehicle Detection and Tracking with YOLO11 🚙🚕
 
# DATASET
[Dataset link](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset)
Size: 1.19 GB

This dataset contains 16643 food images grouped in 11 major food categories.

## There are 3 splits in this dataset:
```bash
├──Evaluation
├──Training
├──Validation

### Each split contains 11 categories of food:

- Bread
- Dairy product
- Dessert
- Egg
- Fried food
- Meat
- Noodles-Pasta
- Rice
- Seafood
- Soup
- Vegetable-Fruit
```

# Directory Hierarchy

```bash
├── YOLO-FOOD-CLASSIFICATION FOLDER
    ├── custom_dataset
          ├── **train** (All 11 classes)
          ├── **val** (All 11 classes)
          ├── train.cache (created at runtime)
          ├── val.cache (created at runtime)
    ├── test_images (dataset of images to test)
    ├── PythonCode (jupyter notebook)
    ├── yolo11m-cls.pt/yolo11n.pt... (download the YOLO version)
    ├── myenv (virtual environment)
    ├── .ipynb_checkpoints (created at runtime)
    ├── runs (Output folder)
          ├── classify
                    ├── predict
                    ├── train
          
```

# Let's Start
1. Create a folder "YOLO-FOOD-CLASSIFICATION" (in my case it is in D drive)
2. Download the dataset [Dataset link](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset)
3. Create a folder "custom_dataset" in YOLO-FOOD-CLASSIFICATION folder.
4. Copy training and validation folder from the downloaded dataset into "custom_dataset" folder. I have renamed training folder as "train" and validation folder as "val".
5. For the easy execution, I have not copied / considered evaluation folder from the dataset. So I have only two image folders in custom_dataset.
6. Create a new folder "test_images" in YOLO-FOOD-CLASSIFICATION. Copy 2 to 3 images of each class to be tested. I have searched the images from browser.
7. Open Command prompt (cmd) and navigate to the folder "YOLO-FOOD-CLASSIFICATION".
8. Make sure that python is installed.
9. Create and activate Virtual environment (myenv): 

       Step 1: --Installing virtualenv through pip--
               pip install virtualenv
   
       Step-2: --Creating a virtualenv--
               python -m virtualenv myenv
     
       Step-3: -- Activate the virtual environment--
               myenv\Scripts\activate
   
11. Now you will find a new folder "myenv" created in YOLO-FOOD-CLASSIFICATION folder.
12. In this tutorial we are using "yolo11s-cls.pt" pretrained model of YOLO, which is used for classification. You can download any version of YOLO classification model version from the [link](https://github.com/ultralytics/ultralytics).
13. Keep this downloaded file in YOLO-FOOD-CLASSIFICATION folder.
14. To install jupyter notebook type the command--
    
        pip install jupyter notebook
15. After installation, you can launch Jupyter Notebook with the command--

        jupyter notebook
16. Once jupyter notebook is launched you can see all the folder of YOLO-FOOD-CLASSIFICATION.
17. Create a new file (in my case it is PythonCode) and execute the cells

        pip install ultralytics  
        pip install --upgrade pip

        from ultralytics import YOLO
        createdmodel=YOLO("yolo11s-cls.pt")
        results= createdmodel.train(data="custom_dataset",epochs=8,imgsz=640 )

        #USING PRETRAINED MODEL
        model_test=YOLO("runs/classify/train/weights/best.pt")
        results=model_test("test_images", save=True, imgsz=320, conf=0.7)
        results[0].show()

    
19. A new folder "runs" will be created. As we have a large dataset to process, it will take a good amount of time to get executed. 

        runs --> classify --> predict and train
    
21. In the predict folder you will get the output of the images which we have given to test from "test_images" folder.
22. In the train folder we will get weights (best.pt and last.pt), args, results, confusion matrixand other inferenced information.

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








              
         
   
