import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew= np.zeros((4,1,2) , dtype=np.int32)
    add  = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints , axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def splitBoxis(img):
    rows = np.vsplit(img , 9)
    boxes=[]
    for r in rows:
        # print(r.shape)
        cols = np.hsplit(r,9)
        for box in cols:
            # print(box.shape)
            boxes.append(box)
    return boxes

#### 6 -  TO DISPLAY THE SOLUTION ON THE IMAGE
def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        image = frame
        # image = cv2.imread("3213.png")
        # image = cv2.imread("sudoku.jpg")
        cv2.imshow("Image", image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", gray)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        cv2.imshow("blur", blur)

        thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
        cv2.imshow("thresh", thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ################################
        # allcontour = image.copy()
        # allcontour = cv2.drawContours(allcontour, contours, -1, (0, 255, 0), 3)
        # cv2.imshow("contourseeeeee", allcontour)
        biggest = np.array([])
        max_area = 0
        c = 0
        # best_cnt = contours[0]
        # print(best_cnt)
        for i in contours:
            area = cv2.contourArea(i)
            if area > 15000:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4 :
                    max_area = area
                    biggest = approx  ## The four vertex points
                    image = cv2.drawContours(image, contours, c, (0, 255, 0), 3)
            c += 1

######################################################
        widthImage = 450
        hightImage = 450
        # print(biggest.size)
        if biggest.size == 8 : # The four vertex points
            imgBigcontour = image.copy()
            biggest = reorder(biggest)
            # print(biggest)
            # print('order')
            cv2.drawContours(imgBigcontour , biggest , -1 , (0,0,255), 25)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0,0] , [widthImage , 0] , [0 , hightImage] , [widthImage , hightImage]])
            matrix = cv2.getPerspectiveTransform(pts1 , pts2)
            imagewrapcolor1 = cv2.warpPerspective(image , matrix , (widthImage , hightImage))
            cv2.imshow("imagewrapcolor", imagewrapcolor1)
            cv2.imshow("imgBigcontour", imgBigcontour)

            ################################
            #
            #REMOVE LINE
            #
            #
            ################################
            # ### remove all line
            image = imagewrapcolor1
            # image = cv2.imread('1.png')
            result = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Remove horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(result, [c], -1, (255, 255, 255), 5)

            # Remove vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(result, [c], -1, (255, 255, 255), 5)

            # cv2.imshow('thresh', thresh)
            # cv2.imshow('result', result)
            # cv2.imwrite('result.png', result)

            #########################3


            ### split the image and find digit
            imagewrapcolor = cv2.cvtColor(imagewrapcolor1, cv2.COLOR_BGR2GRAY)
            kernal = np.ones((5, 5), np.uint8)

            Open = cv2.morphologyEx(imagewrapcolor, cv2.MORPH_OPEN, kernal)
            thresh = cv2.adaptiveThreshold(Open, 255, 1, 1, 11, 2)
            # ret, thresh = cv2.threshold(Open, 127, 255, cv2.THRESH_BINARY)
            # imageCanny = cv2.Canny(Open, 50, 50)
            # kernal = np.ones((5, 5), np.uint8)
            #
            # Open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernal)
            boxes= splitBoxis(thresh)
            whiteboxes = splitBoxis(result)

            # for i in range(80):
            #     boxes[i]= cv2.dilate(boxes[i] , kernal  , iterations=1)
            # print(len(boxes))

            y = 15
            x = 15
            h = 25
            w = 25
            # boxes[9] = boxes[9][y:y + h, x:x + w]
            # n_black_pix = np.sum(boxes[9] == 0)

            # print('Number of black pixels:', n_black_pix)
            # n_white_pix = np.sum(boxes[9] == 255)
            # # print(boxes[1].shape)
            # print('Number of white pixels:', n_white_pix)
            # """for identify blank cell """
            t = 5
            r = 5
            q = 40
            c = 40

            save = boxes[4][t:t + q, r:r + c]
            zs = cv2.Canny(save, 30, 200) #Perform Edge detection
            cv2.imshow(" boxes[0] ", zs)

            cell=[]
            cell1 =[]
            for i in range(len(boxes)):

                crop = boxes[i][y:y + h, x:x + w]
                # save = whiteboxes[i]
                n_white_pix = np.sum(crop == 255)
                cell1.append(n_white_pix)
                
                ## best 130
                if n_white_pix > 90 :
                    filename = 'number/{}.jpg'.format(i)
                    save = whiteboxes[i][t:t + q, r:r + c]
                    # ret, thresh = cv2.threshold(save, 127, 255, 0)
                    #######################################
                    #
                    #extract bounding on the number
                    #
                    # img = cv2.imread(save)
                    #### (img, 140, 255, 0)   between 140 ,150,127
                    ret, thresh = cv2.threshold(save, 141, 255, 0)

                    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

                    # (2) threshold-inv and morph-open
                    th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
                    morphed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, np.ones((2, 2)))
                    # (3) find and filter contours, then draw on src
                    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                    x = y = w = h = 0
                    nh, nw = save.shape[:2]
                    # crop12=np.array()
                    for cnt in cnts:
                        x, y, w, h = bbox = cv2.boundingRect(cnt)
                        if h < 0.5 * nh or (x == 0 and y == 0):
                            continue
                        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 1, cv2.LINE_AA)
                        crop12 = threshed[y:y + h, x:x + w]
                        cv2.imwrite(filename, crop12)
                    #
                    # extract bounding on the number
                    #
                    ###################################

                    cell.append(1)

                else:
                    cell.append(0)
            cell = np.array(cell).reshape(9,9)
            # print(np.array(cell1).reshape(9,9))
            print(cell)

            ######################## display


        ######################################################
        ## exit run
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    #######################3
    ###############################3
