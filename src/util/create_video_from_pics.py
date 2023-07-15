''' A module to read frames from directory and create video'''


import os

import cv2


def create_video_from_pics(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, f))]
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[0:2]+x[3:5]+x[6:8]))
    #print(files[0][0:2]+files[0][3:5]+files[0][6:8])
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


if __name__ == '__main__':
    pathIn= 'known_faces/'
    pathOut = 'video.avi'
    fps = 24.0
    create_video_from_pics(pathIn, pathOut, fps)