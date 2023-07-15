'''This module writes the data to a file'''

import time

def write_to_file(file_name, data:dict, cnt: int):
    '''This function writes the all_faces_recognized to a file'''
    #get current time
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    high_probability = ''
    low_probability = ''
    #sort data based on dict values
    data = dict(sorted(data.items(), key=lambda item: item[1][0], reverse=True))  

    #Remove wrongly classified faces older than 30 frames
    for key in list(data.keys()):
        # if the count of the face is less than equal to 3
        if data[key][0] <= 3:
            #delete items whose cnt - value[1] is greater than 30
            #that is those faces whose count is upto 3 but were detected earlier than 30 frames
            if cnt - data[key][1] > 30:
                del data[key]

    with open(file_name,'w') as f:
        str = f'-----\nAttendance At: {current_time}\n\n-----\n'
        for key, value in data.items():
            if value[0] > 3:
                high_probability += f'{key} - {value[0]}\n'
            else:
                low_probability += f'{key} - {value[0]}\n'
        str += f'High Probability:\n{high_probability}\n\nLow Probability (Can be Ignored):\n{low_probability}\n------'
        f.write(f'{str}\n')