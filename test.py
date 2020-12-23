import cv2
import numpy as np

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

        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ################################
        # allcontour = image.copy()
        # allcontour = cv2.drawContours(allcontour, contours, -1, (0, 255, 0), 3)
        # cv2.imshow("contourseeeeee", allcontour)
        biggest = np.array([])
        max_area = 0
        c = 0
        best_cnt = contours[0]
        # print(best_cnt)
        for i in contours:
            area = cv2.contourArea(i)
            if area > 15000:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) ==4 :
                    max_area = area
                    best_cnt = i
                    biggest = approx  ## The four vertex points
                    image = cv2.drawContours(image, contours, c, (0, 255, 0), 3)
            c += 1

        mask = np.zeros((gray.shape), np.uint8)
        cv2.drawContours(mask, [best_cnt], 0, 255, -1)
        cv2.drawContours(mask, [best_cnt], 0, 0, 2)
        cv2.imshow("mask", mask)

        out = np.zeros_like(gray)
        out[mask == 255] = gray[mask == 255]
        cv2.imshow("New image", out)

        blur = cv2.GaussianBlur(out, (5, 5), 0)
        cv2.imshow("blur1", blur)

        thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
        cv2.imshow("thresh1", thresh)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imgcountour = image.copy()

        c = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 1000 / 2:
                cv2.drawContours(imgcountour, contours, c, (0, 255, 0), 3)
                print('zoooooz')
            c += 1
        cv2.imshow("Final Image", imgcountour)
        print(biggest)
        print('biggest')
######################################################3
        widthImage = 450
        hightImage = 450
        # print(biggest.size)
        if biggest.size == 8 : # The four vertex points
            imgBigcontour = image.copy()
            biggest = reorder(biggest)
            print(biggest)
            print('order')
            cv2.drawContours(imgBigcontour , biggest , -1 , (0,0,255), 25)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0,0] , [widthImage , 0] , [0 , hightImage] , [widthImage , hightImage]])
            matrix = cv2.getPerspectiveTransform(pts1 , pts2)
            imagewrapcolor = cv2.warpPerspective(image , matrix , (widthImage , hightImage))
            cv2.imshow("imagewrapcolor", imagewrapcolor)
            cv2.imshow("imgBigcontour", imgBigcontour)


            ### split the image and find digit
            imagewrapcolor = cv2.cvtColor(imagewrapcolor, cv2.COLOR_BGR2GRAY)
            kernal = np.ones((5, 5), np.uint8)
            Open = cv2.morphologyEx(imagewrapcolor, cv2.MORPH_OPEN, kernal)
            imageCanny = cv2.Canny(Open, 50, 50)
            boxes= splitBoxis(imageCanny)

            # print(len(boxes))
            cv2.imshow(" boxes[0] ", boxes[6])
            cv2.imshow(" boxes[1] ", boxes[0])
            cv2.imshow(" boxes[4] ", boxes[4])
            cv2.imshow(" boxes[5] ", boxes[5])


######################################################
        ## exit run
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



    #######################3
    ###############################3
