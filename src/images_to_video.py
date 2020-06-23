import cv2
import os

fps = 10
size = (1920, 1080)
videowriter = cv2.VideoWriter("test.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
path = '/media/yongjie/Seagate Expansion Drive/Waymo_test_coordinate/images/'
pathlist = os.listdir(path)
print(len(pathlist))
pathlist.sort()
j=1
for i in pathlist:
    img = cv2.imread(path + i)
    print(j)
    j=j+1
    videowriter.write(img)
videowriter.release()
cv2.destroyAllWindows()
