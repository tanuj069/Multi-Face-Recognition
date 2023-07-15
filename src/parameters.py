"""
This module contain the default parameters used by the app
"""

# The default sender of emails
EMAIL_SENDER = 'hamza2019cs148@abesit.edu.in'

# The default receiver of emails
EMAIL_RECEIVER = 'hamzaaziz822@gmail.com'

# Path where dataset is stored (used for creating face embedding)
DATASET_PATH = 'dataset/train/pics_dlib_gril/'

# The path where dlib face encodings are stored
DLIB_FACE_ENCODING_PATH = 'dataset/dlib_face_encoding.pkl'

# The path where face recognition report will be stored
REPORT_PATH = 'src/reports/inferred_faces.csv'

# Files path where various supplementary files are stored
FILES_PATH = 'src/static/files/'

# The path where the video will be stored
VIDEO_PATH = 'src/static/files/vid.mp4'

# The path where the video will be uploaded
VIDEO_UPLOAD_PATH = 'src/static/video/vid.mp4'

# The path where logs will be stored
LOG_FILE_PATH = 'src/logs/multicam_server.log'

# face matching tolerance (distance -> less the distance, more the similarity)
FACE_MATCHING_TOLERANCE = 0.4

# face recognition model
FACE_RECOGNITION_MODEL = 'hog'  # hog -> for CPU or cnn -> for GPU (DGX)

# Number of times to upsample the image looking for faces
NUMBER_OF_TIMES_TO_UPSAMPLE = 1  # for realtime keep it to 1

# Set video frame height and width
FRAME_HEIGHT = 720  # 576
FRAME_WIDTH = 1280  # 1024

# set BATCH_SIZE for face detection
BATCH_SIZE = 1  # for DGX 32, 1 for CPU

# buffer size for video streaming to minimize inconsistent network conditions
LIVE_STREAM_BUFFER_SIZE = 256  # single camera

# buffer size for frames on which face recognition will be performed
INFERENCE_BUFFER_SIZE = 8  # 128 for DGX

# IP Camera Details
IP_CAMS = {
    'cam1': 'http://192.168.15.113:4747/video',
    'cam2': 'http://192.168.19.105:4747/video'
}
