import numpy as np
from MyConvolution import convolve
from scipy.signal import convolve2d
import numpy as np
from PIL import Image,ImageOps
from pathlib import Path
from MyHybridImages import myHybridImages
import cv2

file_dir = Path(__file__).resolve().parent


im2 = cv2.imread("data/h.jpg")
im1 = cv2.imread("data/x.jpg")



def testConvolve():
    iterTest = 20

    for i in range(iterTest):
        row, col = np.random.randint(10, 100), np.random.randint(10, 100)
        # testImage = np.random.rand(row, col, 3)
        testImage = np.random.rand(row, col)

        trow, tcol = np.random.choice([3, 5, 7]), np.random.choice([3, 5, 7])
        testKernel = np.random.randint(-5, 5, size=(trow, tcol))

        myconv = convolve(testImage, testKernel)

        # myconv = myconv[~np.all(myconv == 0, axis=(1, 2))]
        # myconv = myconv[:, ~np.all(myconv == 0, axis=(0, 2))]
        scipyconv = np.zeros(testImage.shape)
        scipyconv = convolve2d(testImage, testKernel, mode='same')
        # for j in range(3):
        # scipyconv[:, :, j] = convolve2d(testImage[:, :, j], testKernel, mode='same', boundary='fill', fillvalue=0)

        myconv = myconv.astype(np.float32)
        scipyconv = scipyconv.astype(np.float32)

        if (myconv == scipyconv).all():
            print("Convolve same ", True)
        else:
            print("Convolve same ", False)




def hybrid(sigs):
    sig1, sig2 = sigs

    test = myHybridImages(im1, sig1, im2, sig2)

    cv2.imwrite(str(file_dir/f'output/hybrid_{sig1}_{sig2}.png'), test)


# if __name__ == '__main__':
    # with Pool(3) as p:
        # p.map(hybrid, list(permutations(range(1, 21), 2)))

sig1, sig2 = 5, 6

test = myHybridImages(im1, sig1, im2, sig2)
print(test)
cv2.imwrite(f'hybrid31_{sig1}_{sig2}.jpg', test)
def testHybridImages(sigs):
    sig1, sig2 = sigs

    test = myHybridImages(im1, sig1, im2, sig2)
    return test

#testConvolve()
#tc_image = np.pad(image_array, ((1, 1), (1, 1)))
#[h,w] = [image_array.shape[0],image_array.shape[1]]
#padding_im = np.zeros([h+2*1, w+2*1, 3])

#print (tc_image)

#padding_im[1:-1, 1:-1,0:] = image_array
#print(padding_im.sum())
#print(np.array_equal(tc_image,padding_im))
#convolved image
# sig1, sig2 = 5, 6
#
# print(im1)
# test = myHybridImages(np.array(im1), sig1, np.array(im2), sig2)
# print(test)
# np_img = np.squeeze(test, axis=1)  # axis=2 is channel dimension
# pil_img = Image.fromarray(np_img)
# pil_img.show()



