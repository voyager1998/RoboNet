import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


tip_coord = []
SCALE = 4  # how much larger to display the image
IF_DIRECTLY_CALIBRATE = True


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global tip_coord
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        tip_coord = [x, y]


def annotate_img(img):
    go_back = False
    is_fail = False
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", img[:, :, ::-1])
        key = cv2.waitKey(1) & 0xFF
        # if the 'c' key is pressed, break from the loop
        if key == 32:   # space
            break
        elif key == ord("g"):
            is_fail = False
        elif key == ord("f"):
            is_fail = True
        elif key == ord("r"):
            go_back = True
            break
    cv2.destroyAllWindows()
    return go_back, is_fail


def display_annotation(img, labels):
    cv2.namedWindow("image")
    scaled_x = int(labels[1] * SCALE)
    scaled_y = int(labels[0] * SCALE)
    img[scaled_x - 3:scaled_x + 3, scaled_y - 3:scaled_y + 3] = [1.0, 0.0, 0.0]
    cv2.imshow("image", img)
    key = cv2.waitKey(0) & 0xFF   # half a second
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if not IF_DIRECTLY_CALIBRATE:
        states = np.load("images/states.npy")
        num_exps = states.shape[0]
        print("state shape", states.shape)
        print("There are", num_exps, "experiments")

        # TODO: automatically load all data from same viewpoint
        use_for_calibration = [0, 1, 2, 4]

        all_pixel_coords = []
        all_3d_pos = []
        for exp_id in use_for_calibration:
            all_3d_pos.append(states[exp_id])

            labels = np.empty((states.shape[1], 2))
            for t in range(states.shape[1]):
                img = cv2.imread("images/exp_" + str(exp_id) +
                                 "_img_" + str(t) + ".png")
                print(img.shape)
                img = cv2.resize(
                    img, (img.shape[1] * SCALE, img.shape[0] * SCALE))

                go_back, is_fail = annotate_img(img)
                labels[t, 0] = tip_coord[0] / SCALE
                labels[t, 1] = tip_coord[1] / SCALE

                display_annotation(img, labels[t])
                print(labels[t])

            all_pixel_coords.append(labels)

        all_pixel_coords = np.concatenate(all_pixel_coords)
        all_3d_pos = np.concatenate(all_3d_pos)
        np.save("images/all_pixel_coords", all_pixel_coords)
        np.save("images/all_3d_pos", all_3d_pos)
        print("Congrats, you're done with this one!")
    else:
        all_pixel_coords = np.load("images/all_pixel_coords.npy")
        all_3d_pos = np.load("images/all_3d_pos.npy")
        print("pixel coords shape", all_pixel_coords.shape)
        print("3d pos shape", all_3d_pos.shape)

    # calibration section starts here
    all_3d_pos = np.array(all_3d_pos[:, 0:3])
    # TODO: denormalization
    

    print("3d pos shape", all_3d_pos.shape)
    all_pixel_coords = np.array(all_pixel_coords,dtype=np.float32)

    intrinsic_guess = np.array([[300.0, 0, 160],
                                [0, 300.0, 120],
                                [0, 0, 1]])
    img_shape = (240, 320)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [all_3d_pos], [all_pixel_coords],
        img_shape, intrinsic_guess, None, flags=flags)
    print("calibrated camera intrinsic:\n", mtx)

    print(all_3d_pos)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_3d_pos[:, 0], all_3d_pos[:, 1], all_3d_pos[:, 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
