import cv2
import numpy as np
from PIL import Image


def illustrate_driving_lane(img, left_line, right_line, lane_color=(255, 0, 255), road_color=(0, 255, 0)):
    """
    #---------------------
    # This function draws lane lines and drivable area on the road
    #
    """
    # Create an empty image to draw on
    window_img = np.zeros_like(img)
    result = None

    window_margin = left_line.window_margin
    left_plotx, right_plotx = left_line.allx, right_line.allx
    ploty = left_line.ally

    if left_plotx is None:
        return result, window_img

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_plotx - window_margin / 5, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_plotx + window_margin / 5, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_plotx - window_margin / 5, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_plotx + window_margin / 5, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), lane_color)
    cv2.fillPoly(window_img, np.int_([right_line_pts]), lane_color)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_plotx + window_margin / 5, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx - window_margin / 5, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([pts]), road_color)
    result = cv2.addWeighted(img, 1, window_img, 0.3, 0)

    return result, window_img


def get_measurements(left_line, right_line):
    """
    #---------------------
    # This function calculates and returns follwing measurements:
    # - Radius of Curvature
    # - Distance from the Center
    # - Whether the lane is curving left or right
    #
    """

    # take average of radius of left curvature and right curvature
    curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2

    # calculate direction using X coordinates of left and right lanes
    direction = ((left_line.endx - left_line.startx) + (right_line.endx - right_line.startx)) / 2

    if curvature > 2000 and abs(direction) < 100:
        road_info = 'Straight'
        curvature = -1
    elif curvature <= 2000 and direction < - 50:
        road_info = 'curving to Left'
    elif curvature <= 2000 and direction > 50:
        road_info = 'curving to Right'
    else:
        if left_line.road_info is not None:
            road_info = left_line.road_info
            curvature = left_line.curvature
        else:
            road_info = 'None'
            curvature = curvature

    center_lane = (right_line.startx + left_line.startx) / 2
    lane_width = right_line.startx - left_line.startx

    center_car = 720 / 2
    if center_lane > center_car:
        deviation = str(round(abs(center_lane - center_car), 3)) + 'm Left'
    elif center_lane < center_car:
        deviation = str(round(abs(center_lane - center_car), 3)) + 'm Right'
    else:
        deviation = 'by 0 (Centered)'

    """
    center_car = 720 / 2
    if center_lane > center_car:
        deviation = 'Left by ' + str(round(abs(center_lane - center_car)/(lane_width / 2)*100, 3)) + '%' + ' from center'
    elif center_lane < center_car:
        deviation = 'Right by ' + str(round(abs(center_lane - center_car)/(lane_width / 2)*100, 3)) + '%' + ' from center'
    else:
        deviation = 'by 0 (Centered)'
    """

    left_line.road_info = road_info
    left_line.curvature = curvature
    left_line.deviation = deviation

    return road_info, curvature, deviation


def illustrate_info_panel(img, left_line, right_line):
    """
    #---------------------
    # This function illustrates details below in a panel on top left corner.
    # - Lane is curving Left/Right
    # - Radius of Curvature:
    # - Deviating Left/Right by _% from center.
    #
    """

    road_info, curvature, deviation = get_measurements(left_line, right_line)
    cv2.putText(img, 'Measurements ', (75, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (80, 80, 80), 2)

    lane_info = 'Lane is ' + road_info
    if curvature == -1:
        lane_curve = 'Radius of Curvature : <Straight line>'
    else:
        lane_curve = 'Radius of Curvature : {0:0.3f}m'.format(curvature)
    # deviate = 'Deviating ' + deviation  # deviating how much from center, in %
    deviate = 'Distance from Center : ' + deviation  # deviating how much from center

    cv2.putText(img, lane_info, (10, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1)
    cv2.putText(img, lane_curve, (10, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1)
    cv2.putText(img, deviate, (10, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1)

    return img


def illustrate_driving_lane_with_topdownview(image, left_line, right_line, img_ferrari='assets/ferrari.png'):
    """
    #---------------------
    # This function illustrates top down view of the car on the road.
    #
    """

    img = cv2.imread(img_ferrari, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (120, 246))

    rows, cols = image.shape[:2]
    window_img = np.zeros_like(image)
    road_map = None

    if right_line.startx is None or left_line.startx:
        return road_map

    window_margin = left_line.window_margin
    left_plotx, right_plotx = left_line.allx, right_line.allx
    ploty = left_line.ally
    lane_width = right_line.startx - left_line.startx
    lane_center = (right_line.startx + left_line.startx) / 2
    lane_offset = cols / 2 - (2 * left_line.startx + lane_width) / 2
    car_offset = int(lane_center - 360)

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array(
        [np.transpose(np.vstack([right_plotx + lane_offset - lane_width - window_margin / 4, ploty]))])
    left_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([right_plotx + lane_offset - lane_width + window_margin / 4, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_plotx + lane_offset - window_margin / 4, ploty]))])
    right_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([right_plotx + lane_offset + window_margin / 4, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (140, 0, 170))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (140, 0, 170))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([right_plotx + lane_offset - lane_width + window_margin / 4, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx + lane_offset - window_margin / 4, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([pts]), (0, 160, 0))

    # window_img[10:133,300:360] = img
    road_map = Image.new('RGBA', image.shape[:2], (0, 0, 0, 0))
    window_img = Image.fromarray(window_img)
    img = Image.fromarray(img)
    road_map.paste(window_img, (0, 0))
    road_map.paste(img, (300 - car_offset, 590), mask=img)
    road_map = np.array(road_map)
    road_map = cv2.resize(road_map, (95, 95))
    road_map = cv2.cvtColor(road_map, cv2.COLOR_BGRA2BGR)

    return road_map


def illustrate_final(undist_img, searching_img, left_line, right_line, Minv, c_shape, crop_size, ferrari_img='ferrari.png'):
    c_rows, c_cols = c_shape
    rows, cols = undist_img.shape[:2]

    w_comb_result, w_color_result = illustrate_driving_lane(searching_img, left_line, right_line)

    # Drawing the lines back down onto the road
    color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
    lane_color = np.zeros_like(undist_img)
    lane_color[crop_size[0]:rows + crop_size[1], 0:cols] = color_result

    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, lane_color, 0.3, 0)

    info_panel, birdeye_view_panel = np.zeros_like(result), np.zeros_like(result)
    info_panel[5:110, 5:325] = (255, 255, 255)
    birdeye_view_panel[5:110, cols - 111:cols - 6] = (255, 255, 255)

    info_panel = cv2.addWeighted(result, 1, info_panel, 0.2, 0)
    birdeye_view_panel = cv2.addWeighted(info_panel, 1, birdeye_view_panel, 0.2, 0)
    road_map = illustrate_driving_lane_with_topdownview(w_color_result, left_line, right_line, ferrari_img)
    if road_map is not None:
        birdeye_view_panel[10:105, cols - 106:cols - 11] = road_map
    birdeye_view_panel = illustrate_info_panel(birdeye_view_panel, left_line, right_line)

    return birdeye_view_panel
