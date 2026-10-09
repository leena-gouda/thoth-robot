\# CV research (reference only)



The robot runs the vision pipeline in `ros2\_ws/src/thoth\_vision`. This folder holds the work behind it.



\- `age\_emotion\_model\_comparison.ipynb`: 3 age models and 6 emotion models tested on 34 hand-labeled faces. Selected ViT age (79.4%, the only model that detected children) and ViT-FER emotion (76.5%). Test images are not included.

\- `live\_demo.py`: webcam demo (SCRFD + ViT age + ViT-FER) without ROS. Use it to try new models before porting them to `thoth\_vision`. Run with `python live\_demo.py`.

\- `ffmpeg\_camera.py`: on Windows, OpenCV only gives slow YUY2 frames from the Rapoo C260. This reads MJPG through FFmpeg for full 30 FPS. FFmpeg must be installed. This is not needed on the robot (Ubuntu + usb\_cam).

