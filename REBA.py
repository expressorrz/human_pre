import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_angle(vector1, vector2):
    # 计算两个向量的点积
    dot_product = np.dot(vector1, vector2)

    # 计算两个向量的模长
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    # 计算夹角的余弦值
    cos_angle = dot_product / (norm_vector1 * norm_vector2)

    # 使用arccos函数计算夹角的弧度值，然后将其转换为度数
    angle = np.arccos(cos_angle)
    angle = np.degrees(angle)

    return angle

def neck_degrees(A,B,C,D,E):
    #2和20的向量在0 4 8平面的投影与垂直向量的夹角
    # 计算AB向量
    AB = B - A
    # 计算CDE平面的法向量
    CD = D - C
    CE = E - C
    normal = np.cross(CD, CE)
    # 计算AB向量在CDE平面上的投影
    projection = AB - np.dot(AB, normal) / np.linalg.norm(normal)**2 * normal
    # 计算投影向量与y轴单位向量的夹角
    y_axis = np.array([0, 1, 0])
    angle = np.arccos(np.dot(projection, y_axis) / (np.linalg.norm(projection) * np.linalg.norm(y_axis)))
    # 将弧度转换为度
    neck_flex_angle = 90 - np.degrees(angle)

    #2和20的向量与水平向量的夹角
    x_axis = np.array([1, 0, 0])
    neck_bending_angle = np.arccos(np.dot(normal, x_axis) / (np.linalg.norm(normal) * np.linalg.norm(x_axis)))
    return neck_flex_angle,neck_bending_angle

def trunk_degrees(A,B,C,D,E,F):
    #0和20的向量与0 1 20 2平面的投影与垂直向量的夹角
    # 计算AB向量
    AB = B - A
    # 计算CDE平面的法向量
    AC = C - A
    DB = B - D
    normal = np.cross(AC, DB)

    # 计算AB向量在CDE平面上的投影
    projection = AB - np.dot(AB, normal) / np.linalg.norm(normal)**2 * normal
    # 计算投影向量与y轴单位向量的夹角
    y_axis = np.array([0, 1, 0])
    angle = np.arccos(np.dot(projection, y_axis) / (np.linalg.norm(projection) * np.linalg.norm(y_axis)))
    # 将弧度转换为度
    trunk_flex_angle = 90 - np.degrees(angle)

    #0和20的向量与0 4 8平面的投影与垂直向量的夹角
    # 计算CFG平面的法向量
    AE = E - A
    AF = F - A
    normal = np.cross(AE, AF)
    # 计算AB向量在CDE平面上的投影
    projection = AB - np.dot(AB, normal) / np.linalg.norm(normal)**2 * normal
    # 计算投影向量与y轴单位向量的夹角
    angle = np.arccos(np.dot(projection, y_axis) / (np.linalg.norm(projection) * np.linalg.norm(y_axis)))
    # 将弧度转换为度
    trunk_bending_angle = 90 - np.degrees(angle)
    # print(trunk_flex_angle)
    # print(trunk_bending_angle)
    return trunk_flex_angle,trunk_bending_angle

def leg_degrees(A,B,C,D,E,F):
    #12,13向量与13，14向量的夹角
    AB = B - A
    BC = C - B
    left_leg_angle = calculate_angle(AB,BC)
    #16,17向量与17，18向量的夹角
    DE = E - D
    EF = F - E
    right_leg_angle = calculate_angle(DE,EF)
    return left_leg_angle,right_leg_angle

def ua_degrees(A,B,C,D,E,F,G,H):
    #4,5向量在0 1 20 2的投影与垂直向量的夹角
    AB = B - A
    CD = D - C
    EF = F - E
    GH = H - G
    normal = np.cross(EF, GH)
    # 计算AB向量在CDE平面上的投影
    projection = AB - np.dot(AB, normal) / np.linalg.norm(normal)**2 * normal
    # 计算投影向量与y轴单位向量的夹角
    y_axis = np.array([0, 1, 0])
    angle = np.arccos(np.dot(projection, y_axis) / (np.linalg.norm(projection) * np.linalg.norm(y_axis)))
    # 将弧度转换为度
    left_ua_angle =  90 - np.degrees(angle)
    projection = CD - np.dot(AB, normal) / np.linalg.norm(normal)**2 * normal
    angle = np.arccos(np.dot(projection, y_axis) / (np.linalg.norm(projection) * np.linalg.norm(y_axis)))
    right_ua_angle =  90- np.degrees(angle)

    #20，4向量与水平向量的夹角
    AG = G - A
    CG = G - C
    x_axis = np.array([1, 0, 0])
    left_shoulder_angle = calculate_angle(AG,x_axis)
    #20，8向量与水平向量的夹角
    right_shoulder_angle = calculate_angle(CG,x_axis)
    return left_ua_angle,right_ua_angle,left_shoulder_angle,right_shoulder_angle


def la_degrees(A,B,C,D,E,F,G,H):
    #5,6向量在0 1 20 2的投影与方向向下的垂直向量的夹角
    AB = B - A
    CD = D - C
    EF = F - E
    GH = H - G
    normal = np.cross(EF, GH)
    # 计算AB向量在CDE平面上的投影
    projection = AB - np.dot(AB, normal) / np.linalg.norm(normal)**2 * normal
    # 计算投影向量与y轴单位向量的夹角
    y_axis = np.array([0, -1, 0])
    angle = np.arccos(np.dot(projection, y_axis) / (np.linalg.norm(projection) * np.linalg.norm(y_axis)))
    left_la_angle = 90 - np.degrees(angle)
    #8,9向量在0 1 20的投影与方向向下的垂直向量的夹角
    projection = CD - np.dot(CD, normal) / np.linalg.norm(normal)**2 * normal
    angle = np.arccos(np.dot(projection, y_axis) / (np.linalg.norm(projection) * np.linalg.norm(y_axis)))
    right_la_angle = 90 - np.degrees(angle)
    return  left_la_angle,right_la_angle

def wrist_degrees(A,B,C,D,E,F,G,H):
    AB = B - A
    CD = D - C
    EF = F - E
    GH = H - G
    normal = np.cross(EF, GH)
    # 计算AB向量在CDE平面上的投影
    projection = AB - np.dot(AB, normal) / np.linalg.norm(normal)**2 * normal
    # 计算投影向量与y轴单位向量的夹角
    y_axis = np.array([0, 1, 0])
    angle = np.arccos(np.dot(projection, y_axis) / (np.linalg.norm(projection) * np.linalg.norm(y_axis)))
    left_wrist_angle = 90 - np.degrees(angle)
    projection = CD - np.dot(CD, normal) / np.linalg.norm(normal)**2 * normal
    angle = np.arccos(np.dot(projection, y_axis) / (np.linalg.norm(projection) * np.linalg.norm(y_axis)))
    right_wrist_angle = 90 - np.degrees(angle)
    return left_wrist_angle,right_wrist_angle

def neck_score(A,B):
    i = 1
    if A > 20 or A < -5:
        i = i + 1
    if abs(B) > 5 :
        i = i + 1
    return i

def trunk_score(A,B):
    i = 1
    if A < -5:
        i = i + 1
    if A > 0 and A < 20:
        i = i + 1
    if A > 20 and A < 60:
        i = i + 2
    if A < -20:
        i = i + 2
    if A > 60:
        i = i + 3
    if abs(B) > 5 :
        i = i + 1
    return i


def leg_score(A,B):
    i = 1
    if A > B:
        C = A
    else: C = B
    if C > 30 and C < 60:
        i = i + 1
    if C > 60:
        i = i + 2
    return i

def ua_score(A,B,D,E):
    i = 1
    if A > B:
        C = A
    else: C = B
    if C < -20:
        i = i + 1
    if C > 20 and C < 45:
        i = i + 2
    if C > 45 and C < 90:
        i = i + 3
    if D > E:
        F = D
    else: F = E
    if F > 10:
        i = i + 1
    return i

def la_score(A,B):
    i = 1
    if A > B:
        C = A
    else: C = B
    if C > 0 and C < 60:
        i = i + 1
    if C > 100:
        i = i + 1
    return i

def wrist_score(A,B):
    i = 1
    if abs(A) > abs(B):
        C = A
    else: C = B
    if C > 15 or C < -15:
        i = i + 1
    return i


def compute_REBA(array):

    hips_pos = array[:, 0:3]
    spine1_pos = array[:, 3:6]
    neck_pos = array[:, 6:9]
    head_pos = array[:, 9:12]
    left_shoulder_pos = array[:, 12:15]
    left_elbow_pos = array[:, 15:18]
    left_wrist_pos = array[:, 18:21]
    left_hand_pos = array[:, 21:24]
    right_shoulder_pos = array[:, 24:27]
    right_elbow_pos = array[:, 27:30]
    right_wrist_pos = array[:, 30:33]
    right_hand_pos = array[:, 33:36]
    left_upleg_pos = array[:, 36:39]
    left_knee_pos = array[:, 39:42]
    left_ankle_pos = array[:, 42:45]
    left_foot_pos = array[:, 45:48]
    right_upleg_pos = array[:, 48:51]
    right_knee_pos = array[:, 51:54]
    right_ankle_pos = array[:, 54:57]
    right_foot_pos = array[:, 57:60]
    spine3_pos = array[:, 60:63]
    left_handend_pos = array[:, 63:66]
    left_thumb_pos = array[:, 66:69]
    right_handend_pos = array[:, 69:72]
    right_thumb_pos = array[:, 72:75]
    neck_flex = [K for K in range(len(array))]
    neck_bend = [K for K in range(len(array))]
    trunk_flex = [K for K in range(len(array))]
    trunk_bend = [K for K in range(len(array))]
    left_leg = [K for K in range(len(array))]
    right_leg = [K for K in range(len(array))]
    left_ua = [K for K in range(len(array))]
    right_ua = [K for K in range(len(array))]
    left_shoulder = [K for K in range(len(array))]
    right_shoulder = [K for K in range(len(array))]
    left_la = [K for K in range(len(array))]
    right_la = [K for K in range(len(array))]
    left_wrist = [K for K in range(len(array))]
    right_wrist = [K for K in range(len(array))]
    neckscore = [K for K in range(len(array))]
    trunkscore = [K for K in range(len(array))]
    legscore = [K for K in range(len(array))]
    uascore = [K for K in range(len(array))]
    lascore = [K for K in range(len(array))]
    wristscore = [K for K in range(len(array))]
    posture_score_a = [K for K in range(len(array))]
    posture_score_b = [K for K in range(len(array))]
    posture_score_c = [K for K in range(len(array))]
    # variables = ['neck_flex', 'neck_bend', 'trunk_flex', 'trunk_bend', 'left_leg', 'right_leg', 'left_ua', 'right_ua', 'left_shoulder', 'right_shoulder', 'left_la', 'right_la', 'left_wrist', 'right_wrist']
    # locals().update([var: [K for K in range(len(array))] for var in variables])
    
    for i in range(len(array)):
        neck_flex[i],neck_bend[i] = neck_degrees(spine1_pos[i],spine3_pos[i],hips_pos[i],left_shoulder_pos[i],right_shoulder_pos[i])
        # if i == 0:
        #     print(leg_degrees(left_upleg_pos[i],left_knee_pos[i],left_ankle_pos[i],right_upleg_pos[i],right_knee_pos[i],right_ankle_pos[i]))
        trunk_flex[i],trunk_bend[i] = trunk_degrees(hips_pos[i],spine1_pos[i],spine3_pos[i],neck_pos[i],left_shoulder_pos[i],right_shoulder_pos[i])
        left_leg[i],right_leg[i] = leg_degrees(left_upleg_pos[i],left_knee_pos[i],left_ankle_pos[i],right_upleg_pos[i],right_knee_pos[i],right_ankle_pos[i])
        left_ua[i],right_ua[i],left_shoulder[i],right_shoulder[i] = ua_degrees(left_shoulder_pos[i],left_elbow_pos[i],right_shoulder_pos[i],right_elbow_pos[i],hips_pos[i],spine1_pos[i],spine3_pos[i],neck_pos[i])
        left_la[i], right_la[i] = la_degrees(left_elbow_pos[i],left_wrist_pos[i],right_elbow_pos[i],right_wrist_pos[i],hips_pos[i],spine1_pos[i],spine3_pos[i],neck_pos[i])
        left_wrist[i],right_wrist[i] = wrist_degrees(left_wrist_pos[i],left_hand_pos[i],right_wrist_pos[i],right_hand_pos[i],hips_pos[i],spine1_pos[i],spine3_pos[i],neck_pos[i])
        neckscore[i] = neck_score(neck_flex[i],neck_bend[i])
        trunkscore[i] = trunk_score(trunk_flex[i],trunk_bend[i])
        legscore[i] = leg_score(left_leg[i],right_leg[i])
        uascore[i] = ua_score(left_ua[i],right_ua[i],left_shoulder[i],right_shoulder[i])
        lascore[i] = la_score(left_la[i], right_la[i])
        wristscore[i] = wrist_score(left_wrist[i],right_wrist[i])
    
        reba_table_a = np.array([[ [1,2,3,4], [2,3,4,5], [2,4,5,6], [3,5,6,7], [4,6,7,8] ],
                        [ [1,2,3,4], [3,4,5,6], [4,5,6,7], [5,6,7,8], [6,7,8,9] ],
                        [ [3,3,5,6 ], [4,5,6,7], [5,6,7,8], [6,7,8,9], [7,8,9,9] ] ])

        reba_table_b = np.array([[ [1,2,2], [1,2,3] ],
                        [ [1,2,3], [2,3,4] ],
                        [ [3,4,5], [4,5,5] ],
                        [ [4,5,5], [5,6,7] ],
                        [ [6,7,8], [7,8,8] ],
                        [ [7,8,8], [8,9,9] ] ])

        reba_table_c = np.array([[1,1,1,2,3,3,4,5,6,7,7,7],
                        [1,2,2,3,4,4,5,6,6,7,7,8],
                        [2,3,3,3,4,5,6,7,7,8,8,8],
                        [3,4,4,4,5,6,7,8,8,9,9,9],
                        [4,4,4,5,6,7,8,8,9,9,9,9],
                        [6,6,6,7,8,8,9,9,10,10,10,10],
                        [7,7,7,8,9,9,9,10,10,11,11,11],
                        [8,8,8,9,10,10,10,10,10,11,11,11],
                        [9,9,9,10,10,10,11,11,11,12,12,12],
                        [10,10,10,11,11,11,11,12,12,12,12,12],
                        [11,11,11,11,12,12,12,12,12,12,12,12 ],
                        [12,12,12,12,12,12,12,12,12,12,12,12]])
    
        posture_score_a[i] = reba_table_a[neckscore[i] - 1][trunkscore[i] - 1][legscore[i] - 1]
        posture_score_b[i] = reba_table_b[uascore[i] - 1][lascore[i] - 1][wristscore[i] - 1]
        posture_score_c[i] = reba_table_c[posture_score_a[i] - 1][posture_score_b[i] - 1]

    # # 创建新的figure
    # plt.figure()

    # # 绘制折线图
    # plt.plot(range(len(posture_score_c)), posture_score_c, label='posture_score_c')
    # # 添加图例
    # plt.legend()

    # # 显示图形
    # plt.show()

    return posture_score_c

