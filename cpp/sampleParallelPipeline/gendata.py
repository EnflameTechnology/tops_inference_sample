import cv2
import struct

img = cv2.imread("./bus.jpg")
mydata = img.transpose((2,0,1)).flatten().copy()
f=open("input.data","wb")
myfmt='B'*len(mydata)
bin=struct.pack(myfmt,*mydata)
f.write(bin)
f.close()