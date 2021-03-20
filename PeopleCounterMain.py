# 斜着 for m4 m3
import numpy as np
import cv2
import imutils
import math
from test22 import match_template
from test37 import detect_gender
from test37 import female

cap = cv2.VideoCapture('market4new.m4v')


# fullBody_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

def testIntersectionOut(t_tracks):
    res = 627

    for j in range(len(t_tracks)):
        if len(t_tracks[j]) > 3:
            (x, y) = t_tracks[j][-1]
            (x1, y1) = t_tracks[j][-2]
            if x1 >= res and x < res and y > 100:
                return True


def testIntersectionIn(t_tracks):
    res = 627

    for j in range(len(t_tracks)):
        if len(t_tracks[j]) > 3:
            (x, y) = t_tracks[j][-1]
            (x1, y1) = t_tracks[j][-2]
            (x2, y2) = t_tracks[j][-3]
            if x2 > x1 and x > x1:
                return False
            if x1 <= res and x > res and y > 100:
                return True


def optical_flow_tracking(flow, x1, y1):
    # 返回一个两通道的光流向量，实际上是每个点的像素位移值
    # flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # 光流法算出来的坐标（nx，ny），储存到op_tracks
    dx = flow[y1, x1, 0]  # 光流法算出来在x的偏移量
    dy = flow[y1, x1, 1]  # 光流法算出来在y的偏移量
    # print(f'dx,dy = {dx},{dy}')
    nx = x1 + dx
    ny = y1 + dy
    return nx, ny


def print_object(t_tracks):
    a = 0
    if (len(t_tracks) == 1):
        print(f'object {a + 1} :{t_tracks[a]}')
    else:
        for i in t_tracks:
            print(f'object {a + 1} :{t_tracks[a]}')
            a += 1
    return


def add_to_t_tracks(h_tracks, t_tracks, min_center):
    if (len(h_tracks) == 1 and len(opn_tracks) == 1 and len(t_tracks) == 1):
        (xh, yh) = h_tracks[0]
        (xt, yt) = t_tracks[0][-1]
        d_h_t = math.sqrt((xh - xt) ** 2 + (yh - yt) ** 2)
        if (d_h_t > 30):
            t_tracks[0] = []
            t_tracks[0].append(h_tracks[0])
        else:
            t_tracks[min_index].append(min_center)
    else:  # 有多个目标
        #print(is_inserted)
        count_index = 0
        for b in t_tracks:
            if (b == []):  # 说明是新目标
                # print('yes')
                break
            (x2, y2) = t_tracks[count_index][-1]
            (x3, y3) = min_center
            d_min_t = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            temp1.append(d_min_t)
            count_index += 1  # 下一个目标的上一帧的中心点
        #print(temp1)
        min_value_t = min(temp1)
        min_index_t = temp1.index(min_value_t)
        #print(min_value_t)
        temp1.clear()

            # 是第几个index就加到第几个object里面
        if is_inserted[min_index_t] != 1:  # object还没被添加过
            t_tracks[min_index_t].append(min_center)
            is_inserted[min_index_t] = 1
        if is_inserted[min_index_t] == 1:  # 有两个相近的 取最近的
            del t_tracks[min_index_t][-1]
            t_tracks[min_index_t].append(min_center)
            is_inserted[min_index_t] = 1

    return


# 获取第一帧
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# 获取视频长宽
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width)
print(height)

h_tracks = []  # 保存旧frame的中点 每循环一次清空一次
op_tracks = []  # 保存每一帧算出来的光流法nx ny
opn_tracks = []  # 保存上一帧算出来的光流法nx ny
t_tracks = []  # 输出跟踪到的目标中点
new_tracks = []
box_tracks = []
last_box_tracks = []  # 保存上一帧的bonding box坐标以便于模版匹配
temp1 = []
temp2 = []

index = 100
frame_idx = 0
idx = 0

girl = 0
boy = 0
textIn = 0
textOut = 0
c=0
while (cap.isOpened()):
    (grabbed, frame2) = cap.read()
    if not grabbed:
        break
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # _, thresh1 = cv2.threshold(next, 8, 255, cv2.THRESH_BINARY)
    # cv2.imshow('threshold', thresh1)
    # cv2.waitKey(500)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    diff = cv2.absdiff(frame1, frame2)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # arrUpperBody = fullBody_cascade.detectMultiScale(frame1, 1.1, 2)
    # for (x, y, w, h) in arrUpperBody:
    # cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 8, 255, cv2.THRESH_BINARY)
    #cv2.imshow('threshold', thresh)
    #cv2.waitKey(500)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # fill any small holes
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)  # remove noise
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 3)
    for contour in contours:
        # print(c)
        # if the contour is too small, ignore it
        if cv2.contourArea(contour) < 900:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(contour)

        if h >= w:
            bonding_box = (x, y, w, h)

            box_tracks.append(bonding_box)

            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

            rectagleCenterPont = ((x + x + w) // 2, (y + y + h) // 2)
            (x1, y1) = rectagleCenterPont

            h_tracks.append(rectagleCenterPont)
            cv2.circle(frame1, rectagleCenterPont, 1, (0, 0, 255), 5)
            # 算光流
            nx, ny = optical_flow_tracking(flow, x1, y1)
            op_tracks.append((nx, ny))

        else:
            break
        # 解决掉帧
    # print(box_tracks)
    # print(last_box_tracks)

    # if box_tracks == [] and opn_tracks != []:
    # 单人掉帧

    if h_tracks == [] and last_box_tracks != []:
        # print('--------------down frame------------')

        min_value = match_template(last_box_tracks, box_tracks, frame1, frame2, h_tracks, flow, op_tracks,
                                   optical_flow_tracking)

        if (min_value < 0.01):
            h_tracks = []
            t_tracks = []

            '''
    if(len(last_box_tracks) > len(box_tracks)):
            print('======down frame======')
            min_value = match_template(last_box_tracks, box_tracks, frame1, frame2, h_tracks, flow, op_tracks,
                                       optical_flow_tracking)

    #多人掉帧
    for t in t_tracks:
        a = len(t)
        temp1.append(a)
        if(temp1 == temp2):
            break
        else:
            min_value = match_template(last_box_tracks, box_tracks, frame1, frame2, h_tracks, flow, op_tracks,
                                       optical_flow_tracking)
            if (min_value < 0.01):
                h_tracks = []
                t_tracks = []
            for j in temp1:
                j += 1
                temp2.append(j)
        temp1.clear()

'''
    print(h_tracks)
    # print(opn_tracks)
    # 第一帧有目标时
    if (h_tracks != [] and t_tracks == []):
        i = 0
        for ob in h_tracks:
            new_list = []  # 几个目标就创几个list
            new_list.append(ob)  # 把每个目标放到每个list里
            # print(f'object{i+1} :{new_list}')
            t_tracks.append(new_list)
            i += 1
        print_object(t_tracks)
        # 这一帧没目标
    elif h_tracks == [] and t_tracks == []:
        t_tracks = []

        print('no object')

    else:
        # 掉帧或者出去
        if (len(h_tracks) < len(opn_tracks)):
            is_inserted = is_inserted = np.zeros((len(t_tracks),), dtype=np.int)

            for i in range(len(h_tracks)):
                (x1, y1) = h_tracks[i]
                # 检查目标是否出去
                for j in range(len(opn_tracks)):
                    (nx, ny) = opn_tracks[j]
                    d_min = math.sqrt((x1 - nx) ** 2 + (y1 - ny) ** 2)
                    new_tracks.append(d_min)

                min_value = min(new_tracks)
                min_index = new_tracks.index(min_value)
                min_index = i
                min_center = h_tracks[min_index]

                if min_value > 30:  # 新目标
                    t_tracks.append([])
                    t_tracks[-1].append(min_center)
                    is_inserted = np.concatenate((is_inserted, [1]))
                   # is_inserted = np.append(is_inserted, 1)
                add_to_t_tracks(h_tracks, t_tracks, min_center)
                new_tracks.clear()

            x = 0  # 系数
            for i in is_inserted:
                if (i == 0):  # 说明没添加过中点
                    (x2, y2) = t_tracks[x][-1]
                    if (len(t_tracks) < 1):
                        break
                    print('out')
                    del (t_tracks[x])
                    i = 1

                elif (i == 1):  # 掉帧
                    x += 1
                    '''
                        last_box_tracks_new.append(last_box_tracks[x])
                        min_value = match_template(last_box_tracks_new, box_tracks, frame1, frame2, h_tracks, flow,
                                                   op_tracks,
                                                   optical_flow_tracking)
                                                   '''

        # 有新目标或上一帧掉帧
        if (len(h_tracks) > len(opn_tracks)):
            is_inserted = is_inserted = np.zeros((len(t_tracks),), dtype=np.int)
            for i in range(len(h_tracks)):
                (x1, y1) = h_tracks[i]
                for j in range(len(opn_tracks)):
                    (nx, ny) = opn_tracks[j]
                    d_min = math.sqrt((x1 - nx) ** 2 + (y1 - ny) ** 2)
                    new_tracks.append(d_min)
                # print(f'----new_tracks{new_tracks}')
                min_value = min(new_tracks)
                min_index = i  # 在h_t里面的index
                min_center = h_tracks[min_index]

                if min_value > 30:  # 新目标
                    t_tracks.append([])
                    t_tracks[-1].append(min_center)
                    is_inserted = np.concatenate((is_inserted, [1]))
                    #is_inserted = np.append(is_inserted, 1)
                add_to_t_tracks(h_tracks, t_tracks, min_center)
                new_tracks.clear()
                # print(is_inserted)
            #for o in range(len(t_tracks)):
                #if (t_tracks[o] == []):
                    #del t_tracks[o]  # 因为有相近的点而重复 加了新的【】

        # 正常跟踪
        if (len(h_tracks) == len(opn_tracks)):
            count = 0
            is_inserted = is_inserted = np.zeros((len(t_tracks),), dtype=np.int)
            for i in opn_tracks:  # 遍历上一帧光流法预计的中点
                (nx, ny) = i
                for j in h_tracks:  # 遍历这一帧的中心点
                    (x1, y1) = j
                    d_min = math.sqrt((x1 - nx) ** 2 + (y1 - ny) ** 2)
                    new_tracks.append(d_min)
                #print(new_tracks)
                min_value = min(new_tracks)
                #print(min_value)
                min_index = new_tracks.index(min_value)
                #print(min_index)
                min_center = h_tracks[min_index]

                if min_value > 30:  # 新目标
                    t_tracks.append([])
                    t_tracks[-1].append(min_center)
                    is_inserted = np.concatenate((is_inserted, [1]))
                    #is_inserted = np.append(is_inserted, 1)
                add_to_t_tracks(h_tracks, t_tracks, min_center)
                #print(is_inserted)
                new_tracks.clear()
                x = 0  # 系数
                for i in is_inserted:
                    if (i == 0):  # 说明没添加过中点
                        if (len(t_tracks) <= 1):
                            break
                        for j in opn_tracks:
                            (x2, y2) = j
                        if (x2 <= 15 or x2 >= width - 15 or y2 <= 15 or y2 >= height - 15 or x2 == width or
                                y2 == height):
                            print('out of the window')
                            # 目标出去
                            del (t_tracks[x])
                            i = 1
                            x += 1

                    else:
                        x += 1

        print_object(t_tracks)

        if (testIntersectionOut(t_tracks)):
            (boy, girl) = detect_gender(box_tracks, frame1, boy, girl)
            textOut += 1
            cv2.waitKey(500)
        if (testIntersectionIn(t_tracks)):
            (boy, girl) = detect_gender(box_tracks, frame1, boy, girl)
            textIn += 1
            cv2.waitKey(500)
        # 画轨迹
        for i in range(len(t_tracks)):
            if (len(t_tracks[i]) > 4):
                cv2.line(frame1, t_tracks[i][-4], t_tracks[i][-1], (0, 255, 0), 3)

    cv2.putText(frame1, "current object: {}".format(f'{len(t_tracks)} objects'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3)
    cv2.line(frame1, (400, int(height)), (int(width), 0), (250, 0, 1), 2)# blue line

    cv2.putText(frame1, "In: {}".format(str(textIn)), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame1, "Out: {}".format(str(textOut)), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame1, "female: {}".format(str(girl)), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame1, "male: {}".format(str(boy)), (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    last_box_tracks = box_tracks.copy()
    opn_tracks = op_tracks.copy()
    h_tracks.clear()
    op_tracks.clear()
    box_tracks.clear()
    is_inserted = np.zeros_like([len(t_tracks), 1])

    cv2.imshow('frame', frame1)
    #cv2.imwrite('test1'+str(c) + '.jpg',frame1)
    c += 1
    frame1 = frame2
    # print(tracks)

    if cv2.waitKey(1) == 27:
        break
    prvs = next
    frame_idx += 1
    print(f'the number of frame is {frame_idx}')

cap.release()
cv2.destroyAllWindows()