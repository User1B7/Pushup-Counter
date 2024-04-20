# Pushup-Counter
This repository contains a Python script that uses MediaPipe Pose to detect pushup movements in a live camera feed. The script counts the number of pushups performed in real-time and displays the count on the camera feed. Additionally, it creates a plot showing the pushup count over time. 

#### How to Use:
1. **Setup:**
   - Ensure you have Python installed on your system.
   - Clone this repository to your local machine by running:
     ```bash
     git clone https://github.com/User1B7/Pushup-Counter.git
     ```
   - Navigate to the cloned directory:
     ```bash
     cd Pushup-Counter
     ```
   - Install the required libraries by running:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run the Script:**
   - Execute the `main.py` script using Python:
     ```bash
     python main.py
     ```
   - A camera window will open, showing the live feed from your webcam.
   - Perform pushups in front of the camera, and the script will detect and count them in real-time.
   - The pushup count and elapsed time will be displayed on the camera feed.
   - To stop the camera feed and view the results, press 'Q'.

3. **View Results:**
   - After stopping the camera feed, the script will generate a plot showing the pushup count over time.
   - The plot will automatically open in a new window.

### Here are some Visualizations:
#### Camera Feed During Pushup Counting:
<img width="1037" alt="Bildschirmfoto 2024-04-20 um 23 29 32" src="https://github.com/User1B7/Pushup-Counter/assets/115086838/ebb8ed50-8cdb-4a47-828e-ab7182d278e7">

<img width="1033" alt="Bildschirmfoto 2024-04-20 um 23 30 04" src="https://github.com/User1B7/Pushup-Counter/assets/115086838/2cc56c4c-cda7-4a47-9346-df0d5210f847">

#### Pushup Count Plot:
<img width="510" alt="Bildschirmfoto 2024-04-20 um 23 25 29" src="https://github.com/User1B7/Pushup-Counter/assets/115086838/4d3cd373-a9c1-4fc9-b95f-d6ea8380dad7">





