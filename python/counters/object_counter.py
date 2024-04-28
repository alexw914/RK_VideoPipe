import cv2
import numpy as np

from .utils import colors
from collections import defaultdict
from shapely.geometry import LineString, Point, Polygon

class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the Counter with default values for various tracking and counting parameters."""

        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        self.reg_pts = [(20, 400), (1260, 400)]
        self.line_dist_thresh = 15
        self.counting_region = None
        self.region_color = (255, 0, 255)
        self.region_thickness = 5

        # Image and annotation Information
        self.im0 = None
        self.tf = None
        self.view_img = False
        self.view_in_counts = True
        self.view_out_counts = True

        self.names = None  # Classes names

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.counting_list = []
        self.count_txt_thickness = 0
        self.count_txt_color = (0, 0, 0)
        self.count_color = (255, 255, 255)

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False
        self.track_color = (0, 255, 0)

    def set_args(
        self,
        classes_names,
        reg_pts,
        count_reg_color=(255, 0, 255),
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        draw_tracks=True,
        count_txt_thickness=2,
        count_txt_color=(0, 0, 0),
        count_color=(255, 255, 255),
        track_color=(0, 255, 0),
        region_thickness=3,
        line_dist_thresh=15,
    ):
        """
        Configures the Counter's image, bounding box line thickness, and counting region points.

        Args:
            line_thickness (int): Line thickness for bounding boxes.
            view_img (bool): Flag to control whether to display the video stream.
            view_in_counts (bool): Flag to control whether to display the incounts on video stream.
            view_out_counts (bool): Flag to control whether to display the outcounts on video stream.
            reg_pts (list): Initial list of points defining the counting region.
            classes_names (dict): Classes names
            track_thickness (int): Track thickness
            draw_tracks (Bool): draw tracks
            count_txt_thickness (int): Text thickness for object counting display
            count_txt_color (RGB color): count text color value
            count_color (RGB color): count text background color value
            count_reg_color (RGB color): Color of object counting region
            track_color (RGB color): color for tracks
            region_thickness (int): Object counting Region thickness
            line_dist_thresh (int): Euclidean Distance threshold for line counter
        """
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks

        # Region and line selection
        if len(reg_pts) == 2:
            print("Line Counter Initiated.")
            self.reg_pts = reg_pts
            self.counting_region = LineString(self.reg_pts)
        elif len(reg_pts) == 4:
            print("Region Counter Initiated.")
            self.reg_pts = reg_pts
            self.counting_region = Polygon(self.reg_pts)
        else:
            print("Invalid Region points provided, region_points can be 2 or 4")
            print("Using Line Counter Now")
            self.counting_region = LineString(self.reg_pts)

        self.names = classes_names
        self.track_color = track_color
        self.count_txt_thickness = count_txt_thickness
        self.count_txt_color = count_txt_color
        self.count_color = count_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.line_dist_thresh = line_dist_thresh

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""

        boxes = tracks[:, :4].astype(int).tolist()
        clss = tracks[:, 6].astype(int).tolist()
        track_ids = tracks[:, 4].astype(int).tolist()

        # draw region
        cv2.polylines(self.im0, [np.array(self.reg_pts, dtype=np.int32)], isClosed=True, color=self.region_color, thickness=self.region_thickness)

        # Extract tracks
        for box, track_id, cls in zip(boxes, track_ids, clss):
            # Draw bounding box
            cv2.rectangle(self.im0, box[:2], box[2:], colors(track_id, True), self.tf)
            cv2.putText(self.im0, "({}) {}".format(track_id, self.names[cls]),
                        (box[0], box[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, colors(track_id, True), self.tf)

            # Save Tracks
            track_line = self.track_history[track_id]
            track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
            if len(track_line) > 30:
                track_line.pop(0)

            # Draw track trails
            if self.draw_tracks:
                points = np.hstack(track_line).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(self.im0, [points], isClosed=False, color=self.track_color, thickness=self.track_thickness)
                cv2.circle(self.im0, (int(track_line[-1][0]), int(track_line[-1][1])), self.track_thickness*2, self.track_color, -1)

            prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None
            # Count objects
            if len(self.reg_pts) == 4:
                if (
                    prev_position is not None
                    and self.counting_region.contains(Point(track_line[-1]))
                    and track_id not in self.counting_list
                ):
                    self.counting_list.append(track_id)
                    if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                        self.in_counts += 1
                    else:
                        self.out_counts += 1

            elif len(self.reg_pts) == 2:
                if prev_position is not None:
                    distance = Point(track_line[-1]).distance(self.counting_region)
                    if distance < self.line_dist_thresh and track_id not in self.counting_list:
                        self.counting_list.append(track_id)
                        if (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0:
                            self.in_counts += 1
                        else:
                            self.out_counts += 1

        incount_label = f"In Count : {self.in_counts}"
        outcount_label = f"OutCount : {self.out_counts}"

        # Display counts based on user choice
        counts_label = None
        if not self.view_in_counts and not self.view_out_counts:
            counts_label = None
        elif not self.view_in_counts:
            counts_label = outcount_label
        elif not self.view_out_counts:
            counts_label = incount_label
        else:
            counts_label = f"{incount_label} {outcount_label}"

        # display real-time counts
        if counts_label is not None:

            tl = self.count_txt_thickness or round(0.002 * (self.im0.shape[0] + self.im0.shape[1]) / 2) + 1
            tf = max(tl - 1, 1)
            # Get text size for in_count and out_count
            t_size_in = cv2.getTextSize(str(counts_label), 0, fontScale=tl / 2, thickness=tf)[0]
            # Calculate positions for counts label
            text_width = t_size_in[0]
            text_x = (self.im0.shape[1] - text_width) // 2  # Center x-coordinate
            text_y = t_size_in[1]

            # Create a rounded rectangle for in_count
            cv2.rectangle(
                self.im0, (text_x - 5, text_y - 5), (text_x + text_width + 7, text_y + t_size_in[1] + 7), self.count_color, -1
            )
            cv2.putText(
                self.im0, str(counts_label), (text_x, text_y + t_size_in[1]), 0, tl / 2, self.count_txt_color, tf, lineType=cv2.LINE_AA
            )

    def display_frames(self):

        """Display frame."""
        cv2.namedWindow("Object Counter")
        cv2.imshow("Object Counter", self.im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        
        if len(tracks) == 0:
            if self.view_img:
                self.display_frames()
            return

        self.extract_and_process_tracks(tracks)

        if self.view_img:
            self.display_frames()
        return self.im0


if __name__ == "__main__":
    ObjectCounter()
