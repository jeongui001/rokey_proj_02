FROM ros:humble-ros-base

RUN apt-get update && apt-get install -y \
    python3-pip libgl1 libglib2.0-0 \
    ros-humble-cv-bridge ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

RUN pip install mediapipe==0.10.14 && pip install "numpy<2"

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

WORKDIR /app
COPY src/vision_node/vision_node/hand_tracking.py /app/hand_tracking.py
COPY src/vision_node/docker/hand_track_node.py /app/hand_track_node.py

CMD ["python3", "hand_track_node.py"]
