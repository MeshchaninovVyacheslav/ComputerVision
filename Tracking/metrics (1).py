
def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    # Write code here

    x1, y1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
    x2, y2 = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
    s_inter = max(x2 - x1, 0) * max(y2 - y1, 0)
    
    s1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    s2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    return s_inter / (s1 + s2 - s_inter)


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        frame_obj_dict = {det[0]:det[1:] for det in frame_obj}
        frame_hyp_dict = {det[0]:det[1:] for det in frame_hyp}

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for i, j in matches.items():
            if i in frame_obj_dict and j in frame_hyp_dict:
                iou = iou_score(frame_obj_dict[i], frame_hyp_dict[j])
                if iou > threshold:
                    dist_sum += iou
                    match_count += 1
                    frame_obj_dict.pop(i)
                    frame_hyp_dict.pop(j)

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        iou = []
        for i in frame_obj_dict:
            for j in frame_hyp_dict:
                iou_ = iou_score(frame_obj_dict[i], frame_hyp_dict[j])
                if iou_ > threshold:
                    iou.append([iou_, i, j])

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        # Step 5: Update matches with current matched IDs
        iou.sort()
        iou.reverse()
        for iou_, i, j in iou:
            if i in frame_obj_dict and j in frame_hyp_dict:
                dist_sum += iou_
                match_count += 1
                frame_obj_dict.pop(i)
                frame_hyp_dict.pop(j)
                matches[i] = j

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count

    return MOTP


def motp_mota   (obj, hyp, threshold=0.4):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0
    gt_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys
        frame_obj_dict = {det[0]:det[1:] for det in frame_obj}
        frame_hyp_dict = {det[0]:det[1:] for det in frame_hyp}
        gt_count += len(frame_obj_dict)

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for i, j in matches.items():
            if i in frame_obj_dict and j in frame_hyp_dict:
                iou = iou_score(frame_obj_dict[i], frame_hyp_dict[j])
                if iou > threshold:
                    dist_sum += iou
                    match_count += 1
                    frame_obj_dict.pop(i)
                    frame_hyp_dict.pop(j)

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        iou = []
        for i in frame_obj_dict:
            for j in frame_hyp_dict:
                iou_ = iou_score(frame_obj_dict[i], frame_hyp_dict[j])
                if iou_ > threshold:
                    iou.append([iou_, i, j])

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error
        # Step 6: Update matches with current matched IDs
        iou.sort(reverse=True)
        for iou_, i, j in iou:
            if i in frame_obj_dict and j in frame_hyp_dict:
                dist_sum += iou_
                match_count += 1
                frame_obj_dict.pop(i)
                frame_hyp_dict.pop(j)
                if i in matches and matches[i] != j:
                    mismatch_error += 1
                matches[i] = j

        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        false_positive += len(frame_hyp_dict)
        missed_count += len(frame_obj_dict)

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count
    MOTA = 1 - (missed_count + false_positive + mismatch_error) / gt_count

    return MOTP, MOTA
