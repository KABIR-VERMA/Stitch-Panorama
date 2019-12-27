import cv2
import numpy as np
import glob, sys

def getMask(img1, img2, side, window=10):
  offs = int(window / 2)
  size = img1.shape[1] - offs
  mask = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1]), dtype=np.float32)
  if side=='left':
    mask[:, size - offs : size + offs] = np.tile(np.linspace(1.0, 0.0, 2.0 * offs).T, (img1.shape[0], 1))
    mask[:, : size - offs] = 1
  else:
    mask[:, size - offs : size + offs] = np.tile(np.linspace(0.0, 1.0, 2.0 * offs).T, (img2.shape[0], 1))
    mask[:, size + offs :] = 1
  return cv2.merge([mask, mask, mask])

def laplacian(img_in1, img_in2):
  gpA = [img_in1.copy()]
  gpB = [img_in2.copy()]
  for i in range(1, 5):
    gpA.append(cv2.pyrDown(gpA[i - 1]))
    gpB.append(cv2.pyrDown(gpB[i - 1]))
  lpA = [gpA[5]]
  lpB = [gpB[5]]
  for i in range(4, 0, -1):
    sa,wa = gpA[i-1].shape
    sb,wb = gpB[i-1].shape
    sizea = (wa, sa)
    sizeb = (wb, sb)
    laplaciana = cv2.subtract(gpA[i - 1], cv2.pyrUp(gpA[i],dstsize = sizea))
    lpA.append(laplaciana)
    laplacianb = cv2.subtract(gpB[i - 1], cv2.pyrUp(gpB[i],dstsize = sizeb))
    lpB.append(laplacianb)

  laplacianPyramidComb = []
  for laplacianA, laplacianB in zip(lpA, lpB):
    rows, cols, dpt = laplacianA.shape
    size = cols/2
    laplacianComb = np.hstack((laplacianA[:, 0:size], laplacianB[:,size:]))
    laplacianPyramidComb.append(laplacianComb)

  img = laplacianPyramidComb[0]
  for i in range(1, 5):
    s,w = laplacianPyramidComb[i].shape
    size = (w, s)
    img = cv2.add(cv2.pyrUp(img,dstsize = size), laplacianPyramidComb[i])

    return img 

def getHomography(img1, img2):
  sift = cv2.xfeatures2d.SIFT_create()
  kp1, des1 = sift.detectAndCompute(img1, None)
  kp2, des2 = sift.detectAndCompute(img2, None)
  matcher = cv2.BFMatcher()
  matches = matcher.knnMatch(des1, des2, k=2)
  goodMatches = []
  for m1, m2 in matches:
    if m1.distance / m2.distance < 0.75:
      goodMatches.append(m1)
  src = np.float32([(kp2[i.trainIdx].pt[0], kp2[i.trainIdx].pt[1]) for i in goodMatches])
  dst = np.float32([(kp1[i.queryIdx].pt[0], kp1[i.queryIdx].pt[1]) for i in goodMatches])
  H = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)[0]
  return H

def blendLeft(img1, img2):
  height = img1.shape[0]
  width = img1.shape[1] + img2.shape[1]
  result = np.zeros((height, width, 3))
  H = getHomography(img1, img2)
  result[:, (width - img2.shape[1]) :, :] = img2
  # result *= getMask(img1, img2, side='right', window=0)
  HDash = np.linalg.inv(H)
  T = np.identity(3)
  T[0, 2] = img1.shape[1]
  HDash = np.matmul(T, HDash)
  result += cv2.warpPerspective(img1, HDash, (width, height)) * getMask(img1, img2, side='left', window=0)
  rows, cols = np.where(result[:, :, 0] != 0)
  minRow, maxRow = min(rows), max(rows)
  minCol, maxCol = min(cols), max(cols)
  result = result[minRow : maxRow + 1, minCol : maxCol, :]
  result = np.clip(result, 0, 255).astype('uint8')
  return result

def blendRight(img1, img2):
  height = img1.shape[0]
  width = img1.shape[1] + img2.shape[1]
  result = np.zeros((height, width, 3))
  H = getHomography(img1, img2)
  result[:, : img1.shape[1], :] = img1
  result *= getMask(img1, img2, side='left')
  img2 = cv2.warpPerspective(img2, H, (width, height)) * getMask(img1, img2, side='right',)
  result += img2
  rows, cols = np.where(result[:, :, 0] != 0)
  minRow, maxRow = min(rows), max(rows)
  minCol, maxCol = min(cols), max(cols)
  result = result[minRow : maxRow + 1, minCol : maxCol, :]
  result = np.clip(result, 0, 255).astype('uint8')
  return result

if __name__ == '__main__':
  imgs =[]
  folder = sys.argv[1]
  for image_path in glob.glob(folder + "/*.jpg"):
    imgs.append(image_path)
  imgs.sort()
  print(imgs)
  for i in range(len(imgs)):
    imgs[i] = cv2.imread(imgs[i])
  imgsLeft = imgs[: int(len(imgs) / 2)]
  imgMid = imgs[int(len(imgs) / 2)]
  imgsRight = imgs[int(len(imgs) / 2) + 1 :]
  imgLeft = imgsLeft[0]
  for i in range(len(imgsLeft) - 1):
    imgLeft = blendLeft(imgLeft, imgsLeft[i + 1])
  imgRight = imgsRight[0]
  for i in range(len(imgsRight) - 1):
    imgRight = blendRight(imgRight, imgsRight[i + 1])
  imgMid = blendLeft(imgLeft, blendRight(imgMid, imgRight))
  cv2.imwrite('output.jpg', imgMid)