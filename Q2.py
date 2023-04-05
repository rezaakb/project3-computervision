
import argparse
import numpy as np
import cv2

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--src_image', default='images/Q2_1.jpeg')
    parser.add_argument('-d', '--des_image', default='images/Q2_2.jpeg')
    parser.add_argument('-t', '--threshold', type=float, default='5')
    parser.add_argument('-m', '--mode', default='A')

    args = parser.parse_args()

    # Load the images
    img1 = cv2.imread(args.src_image)
    img2 = cv2.imread(args.des_image)

    # Copy images to another variable
    img1_c = img1.copy()
    img2_c = img2.copy()

    # Find height and width of the original image
    height, width, channels = img2.shape


    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_c, cv2.COLOR_BGR2GRAY)


    src_pts = []
    dst_pts = []

    def click_event_image_1 (event, x, y, flags, params):  
        if event == cv2.EVENT_LBUTTONDOWN:
            if (x,y) not in src_pts:
                src_pts.append((x,y))
            cv2.circle(img1, (x,y), 8, (255, 255, 0), -1)
            cv2.imshow('image 1', img1)
    

    def click_event_image_2 (event, x, y, flags, params):  
        if event == cv2.EVENT_LBUTTONDOWN:
            if (x,y) not in dst_pts:
                dst_pts.append((x,y))
            cv2.circle(img2, (x,y), 8, (255, 0, 255), -1)
            cv2.imshow('image 2', img2)   


    def draw_correspondences(img1, img2, kp1, kp2, good_matches):
        
        # Create a new image that combines the two input images side by side
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]
        new_width = width1 + width2
        new_height = max(height1, height2)
        new_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        new_img[:height1, :width1] = img1
        new_img[:height2, width1:] = img2

        overlay = new_img.copy()
        
        # Draw circles around the keypoints and lines connecting the matched keypoints
        if good_matches == "M":
            for i in range(len(kp1)):
                
                p1 = kp1[i]

                # Shift the x-coordinate of the second point to account for the offset in the new image
                p2_shifted = (kp2[i][0] + width1, kp2[i][1])

                # Draw circles around the keypoints and lines connecting the matched keypoints
                color = tuple(np.random.randint(0, 255, 3).tolist())  # Select a random color for each line
                cv2.circle(new_img, p1, 5, color, -1)
                cv2.circle(new_img, p2_shifted, 5, color, -1)
                cv2.line(new_img, p1, p2_shifted, color, 2)
        else:
            for match in good_matches:
                # Get the keypoints from the good matches
                kp1_idx = match.queryIdx
                kp2_idx = match.trainIdx
                p1 = tuple(map(int, kp1[kp1_idx].pt))
                p2 = tuple(map(int, kp2[kp2_idx].pt))

                # Shift the x-coordinate of the second point to account for the offset in the new image
                p2_shifted = (p2[0] + width1, p2[1])

                # Draw circles around the keypoints and lines connecting the matched keypoints
                color = tuple(np.random.randint(0, 255, 3).tolist())  # Select a random color for each line
                cv2.circle(new_img, p1, 5, color, -1)
                cv2.circle(new_img, p2_shifted, 5, color, -1)
                cv2.line(new_img, p1, p2_shifted, color, 2)

        alpha = 0.40
        new_img = cv2.addWeighted(overlay, alpha, new_img, 1 - alpha, 0)

        return new_img

    if args.mode == "A":

        # Find the keypoints and descriptors using SIFT
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        # Match the keypoints
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test to filter out false matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.5*n.distance:
                good_matches.append(m)


        # Estimate the homography matrix using RANSAC
        src_pts = [kp1[m.queryIdx].pt for m in good_matches]
        dst_pts = [kp2[m.trainIdx].pt for m in good_matches]

        cv2.imwrite('results/Q2-Points-threshold='+str(args.threshold)+'-mode='+args.mode+'-number_points='+str(len(src_pts))+'.jpg', draw_correspondences(img1_c,img2, kp1, kp2, good_matches))


    elif args.mode == "M":

        cv2.imshow('image 1', img1)
        cv2.imshow('image 2', img2)

        cv2.setMouseCallback('image 1', click_event_image_1)
        cv2.setMouseCallback('image 2', click_event_image_2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()   

        cv2.imwrite('results/Q2-Points-threshold='+str(args.threshold)+'-mode='+args.mode+'-number_points='+str(len(src_pts))+'.jpg', draw_correspondences(img1_c,img2, src_pts, dst_pts,"M"))
    


    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
    src_pts = np.float32(src_pts).reshape(-1, 1, 2)

    print('Number of points:', len(src_pts))


    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, args.threshold)

    # Generate final image
    height, width, _ = img1.shape
    result = cv2.warpPerspective(img1_c, M, (width, height))

    cv2.imwrite('results/Q2-Image-threshold='+str(args.threshold)+'-mode='+args.mode+'-number_points='+str(len(src_pts))+'.jpg', result)

    print("Finished")

    # Display the result
    cv2.imshow('Result', result.astype("uint8"))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()