#没检测出来
import cv2
import numpy as np

last_box_tracks_new = []
box_tracks_new = []


def match_template(last_box_tracks, box_tracks, frame1, frame2, h_tracks, flow, op_tracks,
                   optical_flow_tracking):
    #print(len(last_box_tracks))
    if(len(last_box_tracks) >= 1 ):
        for j in last_box_tracks:  # 上一帧的box  要在新的frame 里面找到
            (x2, y2, w2, h2) = j
            cropImg = frame1[y2: y2 + h2, x2: x2 + w2]
            theight, twidth = cropImg.shape[:2]
            method = cv2.TM_SQDIFF_NORMED
            result = cv2.matchTemplate(frame2, cropImg, method)
            # minMaxLoc：在给定的矩阵中寻找最大和最小值，并给出它们的位置
            # 该功能不适用于多通道阵列
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            new_center = (((min_loc[0] + min_loc[0] + twidth) // 2), ((min_loc[1] + min_loc[1] + theight) // 2))
            #print(new_center)
            x3, y3 = new_center
            x4, y4 = min_loc
            h_tracks.append(new_center)
            next_box = (x4, y4, twidth, theight)
            box_tracks.append(next_box)
            (nx, ny) = optical_flow_tracking(flow, x3, y3)
            op_tracks.append((nx, ny))
            if min_val < 0.01:
                return False
            cv2.rectangle(frame1, min_loc, (min_loc[0] + twidth, min_loc[1] + theight), (0, 0, 225), 2)
            cv2.circle(frame1, new_center, 1, (0, 0, 255), 5)
            return min_val

'''
    else:
        (x2, y2, w2, h2) = last_box_tracks[get_index]
        cropImg = frame1[y2: y2 + h2, x2: x2 + w2]
        theight, twidth = cropImg.shape[:2]
        method = cv2.TM_SQDIFF_NORMED
        result = cv2.matchTemplate(frame2, cropImg, method)
        # minMaxLoc：在给定的矩阵中寻找最大和最小值，并给出它们的位置
        # 该功能不适用于多通道阵列
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        new_center = (((min_loc[0] + min_loc[0] + twidth) // 2), ((min_loc[1] + min_loc[1] + theight) // 2))
        # print(new_center)
        x3, y3 = new_center
        x4, y4 = min_loc
        h_tracks.append(new_center)
        next_box = (x4, y4, twidth, theight)
        box_tracks.append(next_box)
        (nx, ny) = optical_flow_tracking(flow, x3, y3)
        op_tracks.append((nx, ny))
        if min_val < 0.01:
            return False
        cv2.rectangle(frame1, min_loc, (min_loc[0] + twidth, min_loc[1] + theight), (0, 0, 225), 2)
        cv2.circle(frame1, new_center, 1, (0, 0, 255), 5)

        return x3, y3
'''