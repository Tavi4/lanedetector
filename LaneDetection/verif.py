
import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video 01 1.mp4')

# Initiliaze global variables for lane line parameters

l_bs = [0, 0]
l_as = [0, 0]
r_bs = [0, 0]
r_as = [0, 0]

final_l_bs = [0, 0]
final_l_as = [0, 0]
final_r_bs = [0, 0]
final_r_as = [0, 0]

left_top = [0 ,0]
right_top = [0 ,0]
left_bottom = [0 ,0]
right_bottom = [0 ,0]

final_left_top = [0 ,0]
final_right_top = [0 ,0]
final_left_bottom = [0 ,0]
final_right_bottom = [0 ,0]

final_left_top_x= 0
final_right_top_x= 0
final_left_bottom_x= 0
final_right_bottom_x= 0

#1 OPEN THE VIDEO FILE
while True:
    ret, frame = cam.read()

    if ret is False:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    #2 SHRINK THE FRAME
    inaltime, latime, _ = frame.shape
    procentaj_scala = 37
    latime_noua = int(latime * procentaj_scala / 100)
    inaltime_noua = int(inaltime * procentaj_scala / 100)
    #latime_noua=latime #vedere imagine intreaga
    #inaltime_noua=inaltime #vedere imagine intreaga
    dimensiune_noua = (latime_noua, inaltime_noua)
    new_resolution = cv2.resize(frame, dimensiune_noua)
    cv2.imshow('small', new_resolution)

    #3 CONVERT TO GRAYSCALE
    alb_negru = cv2.cvtColor(new_resolution, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale', alb_negru)

    #4 SELECT ONLY THE ROAD
    trapez_masca = np.zeros((inaltime_noua, latime_noua), dtype=np.uint8)
    # facem colturile
    upper_left = (int(latime_noua * 0.45), int(inaltime_noua * 0.78))
    upper_right = (int(latime_noua * 0.55), int(inaltime_noua * 0.78))
    lower_left = (int(latime_noua * 0), int(inaltime_noua * 1))
    lower_right = (int(latime_noua * 1), int(inaltime_noua * 1))
    trapezoid_puncte = np.array([upper_left, upper_right, lower_right,lower_left], dtype=np.int32)

    # suprapunem masca
    cv2.fillConvexPoly(trapez_masca, trapezoid_puncte, 1)
    fill_trapez_white = np.full((inaltime_noua,latime_noua), 255, dtype=np.uint8) # in plus
    trapezoid=fill_trapez_white  * trapez_masca # in plus
    cv2.imshow('Trapezoid', trapezoid) # in plus
    masca_alb_negru = alb_negru * trapez_masca
    cv2.imshow('Road', masca_alb_negru)

    #5 GET A TOP-DOWN VIEW
    #colturi strech
    upper_left = (int(latime_noua * 0), int(inaltime_noua * 0))
    upper_right = (int(latime_noua * 1), int(inaltime_noua * 0))
    new_bounds = np.array([upper_left, upper_right, lower_right, lower_left], dtype=np.float32)

    #conversie puncte in float
    bounds=np.float32(trapezoid_puncte)

    #cream matricea magica(pentru streching)
    magic_matrix = cv2.getPerspectiveTransform(bounds, new_bounds)
    #warp perspective aplica la masca coordonatele magic pe dimensiunea noua
    top_down = cv2.warpPerspective(masca_alb_negru, magic_matrix, dimensiune_noua)
    cv2.imshow('Top-Down', top_down)

    #6 ADD A BIT OF BLUR
    ksize = (7,7)
    blurat=cv2.blur(top_down, ksize)
    cv2.imshow('Blur', blurat)

    #7 DO EDGE DETECTION
    sobel_vertical= np.float32([[-1,-2,-1],
                                [0,0,0],
                                [1,2,1]])
    sobel_orizontal= np.transpose(sobel_vertical)

    blurat=np.float32(blurat)

    sobel_vertical_blurat = cv2.filter2D(blurat, -1, sobel_vertical)
    sobel_orizontal_blurat = cv2.filter2D(blurat, -1, sobel_orizontal)

    #aplicam filtrul sobel
    sobel=np.sqrt((sobel_vertical_blurat ** 2)+(sobel_orizontal_blurat ** 2))
    sobel_vertical_blurat=cv2.convertScaleAbs(sobel_vertical_blurat) # matrix1
    sobel_orizontal_blurat=cv2.convertScaleAbs(sobel_orizontal_blurat) # matrix2
    sobel_final=cv2.convertScaleAbs(sobel) # matrix 3
    cv2.imshow('Vetical Sobel', sobel_vertical_blurat)
    cv2.imshow('Horizontal Sobel', sobel_orizontal_blurat)
    cv2.imshow('Sobel', sobel_final)


    #8 BINARIZE THE FRAME
    threshold=int(260/3) #86
    ret, binarized = cv2.threshold(sobel_final, threshold, 255, cv2.THRESH_BINARY)

    cv2.imshow('Binarized', binarized)
    #9 GET THE COORDINATES OF THE STREET MARKINGS ON EACH SIDE OF THE ROAD
    # a. inlaturam lateralele
    copie=binarized.copy()
    copie[:, 0:int(0.05*latime_noua)] = 0
    copie[:, int(0.95*latime_noua):] = 0

    # b.
    left_road_line = np.argwhere(copie[:, :int(0.5 * latime_noua)] > 255//2)
    #separam pixelii mai mari de 127 din partea stanga a imaginii
    right_road_line = np.argwhere(copie[:, int(0.5 * latime_noua):] > 255//2)
    #separam pixelii mai mari de 127 din partea dreapta a imaginii

    left_xs = left_road_line[:, 1]
    left_ys = left_road_line[:, 0]

    right_xs = right_road_line[:, 1] + latime_noua / 2
    right_ys = right_road_line[:, 0]

    #10 FIND THE LINE THAT DETECTS THE EDGES OF THE LANE

    if len(left_xs) > 0:
        l_bs, l_as = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)

    if len(right_xs) > 0:
        r_bs, r_as = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

    # calculam coordonatele de la capetele liniilor

    left_top_y = 0
    left_top_x = (left_top_y - l_bs) / l_as
    # trecem in partea dreapta elemetele ecuatiei y = ax + b

    left_bottom_y = inaltime_noua
    left_bottom_x = (left_bottom_y - l_bs) / l_as

    right_top_y = 0
    right_top_x = (right_top_y - r_bs) / r_as

    right_bottom_y = inaltime_noua
    right_bottom_x = (right_bottom_y - r_bs) / r_as

    # stocam variabilele calculate
    left_top[1] = left_top_y
    left_bottom[1] = left_bottom_y
    right_top[1] = right_top_y
    right_bottom[1] = right_bottom_y

    # convertim coordonatele in int, cele din stanga datorita unei imagini neclare trebuie
    # sa facem o verificare inainte de a le converti
    if -latime_noua / 2 < left_top_x < latime_noua / 2:
        left_top[0] = int(left_top_x)

    if -latime_noua / 2 < left_bottom_x < latime_noua / 2:
        left_bottom[0] = int(left_bottom_x)

    right_top[0] = int(right_top_x)
    right_bottom[0] = int(right_bottom_x)

    # desenam liniile gasite prin coordonate pe imaginea binarized
    frame_lines = cv2.line(copie, left_top, left_bottom,(100,0,100),5)
    frame_lines = cv2.line(frame_lines, right_top, right_bottom, (100, 0, 100), 5)

    cv2.imshow('Lines', copie)

    #11 CREATE A FINAL VISUALIZATION

    #cream o imagine goala si o desenam liniile pe care le voiam aplica pe imaginea finala
    new_magic_matrix = cv2.getPerspectiveTransform(new_bounds, bounds)

    new_left_frame = np.zeros((dimensiune_noua[1],dimensiune_noua[0]), dtype=np.uint8)
    new_left_frame = cv2.line(new_left_frame, left_top, left_bottom, (255, 0, 0), 3)
    back_to_normal_left_frame = cv2.warpPerspective(new_left_frame, new_magic_matrix, dimensiune_noua)

    new_right_frame = np.zeros((dimensiune_noua[1], dimensiune_noua[0]), dtype=np.uint8)
    new_right_frame = cv2.line(new_right_frame, right_top, right_bottom, (255, 0, 0), 3)
    back_to_normal_right_frame = cv2.warpPerspective(new_right_frame, new_magic_matrix, dimensiune_noua)

    # gasim pixelii albi din imaginea transformata
    final_left_white_pixels = np.argwhere(back_to_normal_left_frame > 0)
    final_right_white_pixels = np.argwhere(back_to_normal_right_frame > 0)

    # fixam liniile pe sobel
    final_left_xs = final_left_white_pixels[:, 1]
    final_left_ys = final_left_white_pixels[:, 0]

    final_right_xs = final_right_white_pixels[:, 1]
    final_right_ys = final_right_white_pixels[:, 0]

    #testam daca exista coordonotele punctelor(albe) de pe imagine
    if len(final_left_xs) != 0 or len(final_left_ys) != 0:
        # daca coordonatele exista atunci
        final_l_bs,final_l_as=np.polynomial.polynomial.polyfit(final_left_xs, final_left_ys, deg=1)

    if len(final_right_xs) != 0 or len(final_right_ys) != 0:
        final_r_bs,final_r_as=np.polynomial.polynomial.polyfit(final_right_xs, final_right_ys, deg=1)

    # calculam coordonatele varfului si cozii liniei finale din stanga si din dreapta
    # evitand acele coordonate care s-ar putea sa fie in afara imagini
    final_left_top_y = int(inaltime_noua * 0.8)
    if  not  (abs(int((final_left_top_y - final_l_bs)/ final_l_as)) > 10 ** 7):
        final_left_top_x = int((final_left_top_y - final_l_bs)/ final_l_as)

    final_left_bottom_y=int(inaltime_noua)
    if  not  (abs(int((final_left_bottom_y- final_l_bs)/ final_l_as)) > 10 ** 7):
        final_left_bottom_x = int((final_left_bottom_y - final_l_bs) / final_l_as)

    final_right_top_y = int(inaltime_noua * 0.8)
    if  not  (abs(int((final_right_top_y - final_r_bs)/ final_r_as)) > 10 ** 7):
        final_right_top_x = int((final_right_top_y - final_r_bs)/ final_r_as)

    final_right_bottom_y = int(inaltime_noua)
    if  not  (abs(int((final_right_bottom_y - final_r_bs)/ final_r_as)) > 10 ** 7):
        final_right_bottom_x = int((final_right_bottom_y - final_r_bs)/ final_r_as)

    # convertim si setam coordonatele finale
    final_left_top[1] = int (final_left_top_y)
    final_left_bottom[1] = int(final_left_bottom_y)
    final_right_top[1] = int(final_right_top_y)
    final_right_bottom[1] = int(final_right_bottom_y)

    final_left_top[0] = int(final_left_top_x)
    final_left_bottom[0] = int(final_left_bottom_x)

    final_right_top[0] = int(final_right_top_x)
    final_right_bottom[0] = int(final_right_bottom_x)

    # desenam liniile pe imaginea finala
    copy_final_frame = new_resolution.copy()
    copy_final_frame = cv2.line(copy_final_frame,final_left_top, final_left_bottom, (200, 50, 250),2)
    copy_final_frame = cv2.line(copy_final_frame,final_right_top,final_right_bottom,(50, 250, 50),2)

    cv2.imshow('Final', copy_final_frame)
    #print(dimensiune_noua)

cam.release()
cv2.destroyAllWindows()
