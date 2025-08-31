Overview:

Deepfakes are becoming increasingly realistic, raising major challenges for trust and authenticity online. This project introduces a deepfake detection system inspired by human physiology. 
Instead of relying on visual artifacts, it focuses on neurological involuntary behaviors such as blinking, synchronized eye movements, micro-expressions, and facial muscle twitches. 
These signals are difficult for AI to replicate, making the system more robust and reliable than traditional detectors.

Key Features:

• Detection based on natural human behaviors that cannot be easily faked
• Four specialized models targeting blinks, eye synchronization, micro-expressions, and muscle activity
• A rank-weighted ensemble system that fuses multiple predictions for higher accuracy
• Real-time detection supported through a user-friendly web application
• Strong resilience against compression, occlusion, and extreme head poses

Technologies:

• Programming & Frameworks: Python, TensorFlow, Keras, OpenCV, MediaPipe
• Model Architectures: Conv1D-BiLSTM with Attention, Transformer encoders, 3D CNNs, Multi-kernel CNN with BiLSTM, Temporal Gated Aggregators
• Supporting Tools: NumPy, Scikit-learn, Real-ESRGAN (for enhancement), Flask/Django (for deployment)

Achievements

The blink detection model achieved 90% accuracy (AUC 0.89). Binocular synchronization achieved 89% accuracy (AUC 0.93). Micro-expression analysis achieved 87% accuracy (AUC 0.96). Facial muscle 
activity detection reached 73% accuracy (AUC 0.71). When combined in the ensemble, the system outperformed state-of-the-art detectors by 18 to 31 percent under adversarial conditions including 
heavy compression and ±45° head rotations.


Installation Guide:

```bash
# Clone the NeuroGuard repository
git clone https://github.com/mabmas/NeuroGuard.git

# Navigate into the project directory
cd NeuroGuard

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# If requirements.txt is not available:
# pip install tensorflow mediapipe opencv-python scikit-learn pandas

# Download pre-trained models from Google Drive and place in models/ folder
# https://drive.google.com/drive/folders/1XCkShPqMWAAPkZfCUr7Ms9c0bhg1ET25?usp=drive_link

# Run the main script (make sure video is named video.mp4)
python src/Landmark_Extraction_Algorithms.py

