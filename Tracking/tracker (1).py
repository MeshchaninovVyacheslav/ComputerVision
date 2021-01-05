import numpy as np
import os

from moviepy.editor import VideoFileClip

from detection import detection_cast, extract_detections, draw_detections
from metrics import iou_score


class Tracker:
    """Generate detections and build tracklets."""
    def __init__(self, return_images=True, lookup_tail_size=80, labels=None):
        self.return_images = return_images  # Return images or detections?
        self.frame_index = 0
        self.labels = labels  # Tracker label list
        self.detection_history = []  # Saved detection list
        self.last_detected = {}
        self.tracklet_count = 0  # Counter to enumerate tracklets

        # We will search tracklet at last lookup_tail_size frames
        self.lookup_tail_size = lookup_tail_size

    def new_label(self):
        """Get new unique label."""
        self.tracklet_count += 1
        return self.tracklet_count - 1

    def init_tracklet(self, frame):
        """Get new unique label for every detection at frame and return it."""
        # Write code here
        # Use extract_detections and new_label
        detections = extract_detections(frame)
        for i in range(len(detections)):
            detections[i][0] = self.new_label()
        return np.array(detections).reshape((-1, 5))

    @property
    def prev_detections(self):
        """Get detections at last lookup_tail_size frames from detection_history.

        One detection at one id.
        """
        detections = []
        # Write code here
        ids = set()
        #for i in range(len(self.detection_history) - 1, max(len(self.detection_history) -self.lookup_tail_size - 1, -1), -1):
        for frame in self.detection_history[::-1][:self.lookup_tail_size]:
            for det in frame:
                if det[0] not in ids:
                    detections.append(det)
                    ids.add(det[0])

        return detection_cast(detections)

    def bind_tracklet(self, detections):
        """Set id at first detection column.

        Find best fit between detections and previous detections.

        detections: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
        return: binded detections numpy int array Cx5 [[tracklet_id, xmin, ymin, xmax, ymax]]
        """
        detections = detections.copy()
        if detections.size == 0:
            return detection_cast(detections)
        prev_detections = self.prev_detections

        # Write code here

        # Step 1: calc pairwise detection IOU
        iou = []
        for i, det in enumerate(detections):
            for prev_det in prev_detections:
                iou.append([iou_score(det[1:], prev_det[1:]), prev_det[0], i])
        
        # Step 2: sort IOU list
        iou.sort()
        iou.reverse()

        # Step 3: fill detections[:, 0] with best match
        # One matching for each id
        used_prev_labels = set()
        detections[:,0] = -1
        for score, prev_label, label in iou:
            if prev_label not in used_prev_labels and detections[label,0] == -1:
                detections[label, 0] = prev_label
                used_prev_labels.add(prev_label)
        
        # Step 4: assign new tracklet id to unmatched detections
        for i, det in enumerate(detections):
            if det[0] == -1:
                detections[i,0] = self.new_label()
        return detection_cast(np.array(detections))

    def save_detections(self, detections):
        """Save last detection frame number for each label."""
        for label in detections[:, 0]:
            self.last_detected[label] = self.frame_index

    def update_frame(self, frame):
        if not self.frame_index:
            # First frame should be processed with init_tracklet function
            detections = self.init_tracklet(frame)
        else:
            # Every Nth frame should be processed with CNN (very slow)
            # First, we extract detections
            detections = extract_detections(frame, labels=self.labels)
            # Then bind them with previous frames
            # Replacing label id to tracker id is performing in bind_tracklet function
            detections = self.bind_tracklet(detections)

        # After call CNN we save frame number for each detection
        self.save_detections(detections)
        # Save detections and frame to the history, increase frame counter
        self.detection_history.append(detections)
        self.frame_index += 1

        # Return image or raw detections
        # Image usefull to visualizing, raw detections to metric
        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, 'data', 'test.mp4'))

    tracker = Tracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == '__main__':
    main()
