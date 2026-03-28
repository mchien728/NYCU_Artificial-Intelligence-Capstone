import cv2
import numpy as np

points = []

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """

        ### TODO ###
        # Intrinsic matrix
        fov_rad = np.deg2rad(fov)
        
        f = (self.width/2) / (np.tan(fov_rad/2))
        
        u0 = self.width / 2
        v0 = self.height / 2
        
        K = np.array([
            [f, 0, u0],
            [0, f, v0],
            [0, 0, 1]
        ])
        
        # Extrinsic matrix
        theta_rad = np.deg2rad(theta)
        
        R_bev = np.array([
                [1, 0, 0],
                [0, np.cos(theta_rad), -np.sin(theta_rad)],
                [0, np.sin(theta_rad), np.cos(theta_rad)]
        ])
        C_bev = np.array([0, 2.5, 0])
        
        R_front = np.array([
                  [1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]
        ])
        C_front = np.array([0, 1, 0])
        t_front = -R_front @ C_front
        Ex_front = np.hstack((R_front, t_front.reshape(3, 1)))
        
        new_pixels = []
        
        for (u, v) in points:
            # BEV pixels -> 3D ray in camera
            p = np.array([u, v, 1])
            ray_camera = np.linalg.inv(K) @ p
            
            ray_world = R_bev @ ray_camera
            
            if abs(ray_world[1]) < 1e-6:
                continue
            
            lamda = -C_bev[1] / ray_world[1]
            Xw = C_bev + lamda * ray_world

            Xw_append = np.append(Xw, 1)
            project = K @ (Ex_front @ Xw_append)
	    
            if abs(project[2]) <= 0:
                continue
	    	
            u_new = project[0] / project[2]
            v_new = project[1] / project[2]
	    
            new_pixels.append([int(round(u_new)), int(round(v_new))])
	    
        
        return np.array(new_pixels, dtype=np.int32)

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = -90

    front_rgb = "bev_data/front2.png"
    top_rgb = "bev_data/bev2.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)
