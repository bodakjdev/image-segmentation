import cv2
import numpy as np
import os

dir_parent = 'M:/cv/CVProj/'
dir_seg = 'segmented/'
dir_res = 'result/'
dir_col = 'color/'
dir_gt = 'ground truth/'

c = 0  # counter of files
d_sum = 0  # sum of Dice coefficient accuracies
j_sum = 0  # sum of Jaccardi index accuracies

# arrays for storing best and worst cases(accuracy, filename, image)
worst = [[1, '', 0], [1, '', 0], [1, '', 0], [1, '', 0], [1, '', 0]]
best = [[0, '', 0], [0, '', 0], [0, '', 0], [0, '', 0], [0, '', 0]]

# iterating through filenames in the folder with segmented images
for filename in os.listdir(dir_seg):
    f = os.path.join(dir_seg, filename)
    if os.path.isfile(f):

        # reading segmented image and creating a ground truth mask
        # with simple blurring and thresholding
        img_seg = cv2.imread(dir_seg + filename, 1)
        img_seg_gray = cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY)
        img_seg_blur = cv2.GaussianBlur(img_seg_gray, (3, 3), 0)
        ret, img_gt = cv2.threshold(img_seg_blur, 20, 255, cv2.THRESH_BINARY)

        # reading color image
        img_col = cv2.imread(dir_col + filename[:len(filename) - 17] + '.JPG', 1)

        # changing image to HSV colorspace
        img_hsv = cv2.cvtColor(img_col, cv2.COLOR_BGR2HSV)

        # finding HSV range adequate for leaves and thresholding
        low_val = (18, 47, 42)
        high_val = (180, 255, 255)
        img_mask = cv2.inRange(img_hsv, low_val, high_val)

        # applying noise removal
        img_morph = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), dtype=np.uint8))
        img_res = cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel=np.ones((5, 5), dtype=np.uint8))

        # finding Dice coefficient and Jaccardi index
        d = 2 * np.sum(cv2.bitwise_and(img_gt, img_res)) / (np.sum(img_gt) + np.sum(img_res))
        j = np.sum(cv2.bitwise_and(img_gt, img_res)) / np.sum(cv2.bitwise_or(img_gt, img_res))

        # checking worst and best cases
        worst.sort(key=lambda x: x[0], reverse=True)
        best.sort(key=lambda x: x[0])
        for i in range(5):
            if d < worst[i][0]:
                worst[i][0] = d
                worst[i][1] = filename
                worst[i][2] = img_res
                break
            if d > best[i][0]:
                best[i][0] = d
                best[i][1] = filename
                best[i][2] = img_res
                break

        # incrementing the counter and adding to sums
        c = c + 1
        d_sum = d_sum + d
        j_sum = j_sum + j

        # saving all the resulting ground truths and segmentations
        os.chdir(dir_gt)
        cv2.imwrite(filename, img_gt)
        os.chdir(dir_parent + dir_res)
        cv2.imwrite(filename, img_res)
        os.chdir(dir_parent)

print("Mean Dice coefficient:")
print(d_sum / c)
print("Mean Jaccardi index:")
print(j_sum / c)


# printing and saving best and worst cases
print("Best cases:")
best.sort(key=lambda x: x[0], reverse=True)
os.chdir(dir_parent + dir_res + '/best')
for i in range(5):
    cv2.imwrite(best[i][1], best[i][2])
    print(str(i+1) + '. ' + best[i][1] + ' accuracy:' + str(best[i][0]))

print("Worst cases:")
worst.sort(key=lambda x: x[0])
os.chdir(dir_parent + dir_res + '/worst')
for i in range(5):
    cv2.imwrite(worst[i][1], worst[i][2])
    print(str(i+1) + '. ' + worst[i][1] + ' accuracy: ' + str(worst[i][0]))

