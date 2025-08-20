import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        
        self.left_a, self.left_b, self.left_c = [], [], []
        self.right_a, self.right_b, self.right_c = [], [], []
        self.lane_distance_threshold_m = 4.0

    def pipeline(self, img, s_thresh=(100, 255), sx_thresh=(15, 255)):
        img = np.copy(img)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(float)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        return combined_binary

    def perspective_warp(self, img, dst_size=(1280,720)):
        src = np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])
        dst = np.float32([(0,0), (1, 0), (0,1), (1,1)])
        img_size = np.float32([(img.shape[1],img.shape[0])])
        src = src * img_size
        dst = dst * np.float32(dst_size)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, dst_size)
        return warped, M

    def get_hist(self, img):
        return np.sum(img[img.shape[0]//2:,:], axis=0)

    def sliding_window(self, img, nwindows=9, margin=150, minpix=1, draw_windows=True):
        left_fit_ = np.empty(3)
        right_fit_ = np.empty(3)
        out_img = np.dstack((img, img, img))*255

        histogram = self.get_hist(img)
        midpoint = int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = int(img.shape[0]/nwindows)
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            if draw_windows:
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(100,255,255), 3)
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(100,255,255), 3)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(lefty) > 0 and len(leftx) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
            self.left_a.append(left_fit[0])
            self.left_b.append(left_fit[1])
            self.left_c.append(left_fit[2])
        else:
            left_fit = None

        if len(righty) > 0 and len(rightx) > 0:
            right_fit = np.polyfit(righty, rightx, 2)
            self.right_a.append(right_fit[0])
            self.right_b.append(right_fit[1])
            self.right_c.append(right_fit[2])
        else:
            right_fit = None

        left_fit_[0] = np.mean(self.left_a[-10:]) if self.left_a else 0
        left_fit_[1] = np.mean(self.left_b[-10:]) if self.left_b else 0
        left_fit_[2] = np.mean(self.left_c[-10:]) if self.left_c else 0

        right_fit_[0] = np.mean(self.right_a[-10:]) if self.right_a else 0
        right_fit_[1] = np.mean(self.right_b[-10:]) if self.right_b else 0
        right_fit_[2] = np.mean(self.right_c[-10:]) if self.right_c else 0

        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
        right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

        return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

    def get_curve(self, img, leftx, rightx):
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        y_eval = np.max(ploty)
        ym_per_pix = 30.5/720
        xm_per_pix = 3.7/720
         # Ensure leftx, rightx, and ploty are the same length and not empty
        min_len = min(len(leftx), len(rightx), len(ploty))
        if min_len == 0:
            # Return default values if any array is empty
            return 0, 0, 0, 0, 0

        leftx = leftx[:min_len]
        rightx = rightx[:min_len]
        ploty = ploty[:min_len]
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        car_pos = img.shape[1]/2
        l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
        r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]

        left_lane_dist_from_center = (car_pos - l_fit_x_int) * xm_per_pix
        right_lane_dist_from_center = (r_fit_x_int - car_pos) * xm_per_pix

        center_offset = (car_pos - (r_fit_x_int + l_fit_x_int)/2) * xm_per_pix

        return (left_curverad, right_curverad, center_offset, left_lane_dist_from_center, right_lane_dist_from_center)

    def draw_lanes(self, img, left_fit, right_fit, Minv):
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        color_img = np.zeros_like(img)
        
        left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
        points = np.hstack((left, right))
        
        cv2.fillPoly(color_img, np.int_(points), (0,200,255))
        inv_perspective = cv2.warpPerspective(color_img, Minv, (img.shape[1], img.shape[0]))
        inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
        return inv_perspective

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        binary_img = self.pipeline(frame_rgb)
        warped_img, M = self.perspective_warp(binary_img)
        Minv = cv2.getPerspectiveTransform(
            np.float32([(0,0), (1, 0), (0,1), (1,1)])*np.float32([(frame.shape[1],frame.shape[0])]),
            np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])*np.float32([(frame.shape[1],frame.shape[0])])
        )

        out_img, curves, lanes, ploty = self.sliding_window(warped_img, draw_windows=False)

        if curves and len(curves[0]) > 0 and len(curves[1]) > 0:
            left_curverad, right_curverad, center_offset, left_lane_dist_from_center, right_lane_dist_from_center = self.get_curve(frame, curves[0], curves[1])
            processed_frame = self.draw_lanes(frame, curves[0], curves[1], Minv)

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontSize = 1

            # Add measurements
            cv2.putText(processed_frame, f'Lane Curvature: {np.mean([left_curverad, right_curverad]):.0f} m', (50, 50), font, fontSize, (255,255,255), 2)
            cv2.putText(processed_frame, f'Vehicle offset: {center_offset:.2f} m', (50, 100), font, fontSize, (255,255,255), 2)
            cv2.putText(processed_frame, f'Left Lane Dist: {left_lane_dist_from_center:.2f} m', (50, 150), font, fontSize, (255,255,255), 2)
            cv2.putText(processed_frame, f'Right Lane Dist: {right_lane_dist_from_center:.2f} m', (50, 200), font, fontSize, (255,255,255), 2)

            # Check lane departure
            if abs(left_lane_dist_from_center) > self.lane_distance_threshold_m or abs(right_lane_dist_from_center) > self.lane_distance_threshold_m:
                if abs(left_lane_dist_from_center) > self.lane_distance_threshold_m:
                    warning_message = f"WARNING: Vehicle out of left lane by {abs(left_lane_dist_from_center) - self.lane_distance_threshold_m:.2f} m"
                    steering_message = "Steer Right"
                else:
                    warning_message = f"WARNING: Vehicle out of right lane by {abs(right_lane_dist_from_center) - self.lane_distance_threshold_m:.2f} m"
                    steering_message = "Steer Left"
                warning_color = (0, 0, 255)
            else:
                warning_message = "Good lane keeping"
                steering_message = ""
                warning_color = (0, 255, 0)

            cv2.putText(processed_frame, warning_message, (frame.shape[1] // 2 - 400, frame.shape[0] // 2), font, fontSize * 1.2, warning_color, 2)
            if steering_message:
                cv2.putText(processed_frame, steering_message, (frame.shape[1] // 2 - 100, frame.shape[0] // 2 + 50), font, fontSize * 1.5, warning_color, 3)

            return processed_frame, out_img
        else:
            cv2.putText(frame, "No lanes detected", (frame.shape[1] // 2 - 400, frame.shape[0] // 2), font, fontSize * 1.2, (255, 255, 0), 2)
            return frame, out_img