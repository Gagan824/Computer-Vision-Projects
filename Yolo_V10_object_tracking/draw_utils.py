import cv2

def draw_label(image, text, top_left, bottom_right, color, font_color, font_scale=1.1, font_thickness=3):
    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    # Create a filled rectangle for text background
    text_background_top_left = (top_left[0]+18, top_left[1] - text_size[1])
    text_background_bottom_right = (top_left[0] + text_size[0] + 35, top_left[1]+5)
    
    cv2.rectangle(image, text_background_top_left, text_background_bottom_right, color, cv2.FILLED)
    
    # Add text on top of the rectangle
    text_position = (top_left[0] + 18, top_left[1] - 5)
    cv2.putText(image, text, text_position, font, font_scale, font_color, font_thickness)

def draw_rounded_rectangle(image, top_left, bottom_right, color, thickness, radius):
    tl = (top_left[0] + radius, top_left[1] + radius)
    tr = (bottom_right[0] - radius, top_left[1] + radius)
    bl = (top_left[0] + radius, bottom_right[1] - radius)
    br = (bottom_right[0] - radius, bottom_right[1] - radius)
    
    # image=cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

    cv2.rectangle(image, (tl[0], top_left[1]), (tr[0], bottom_right[1]), color, 2, cv2.LINE_AA)
    cv2.rectangle(image, (top_left[0], tl[1]), (bottom_right[0], bl[1]), color, thickness)
    cv2.circle(image, tl, radius, color, thickness)
    cv2.circle(image, tr, radius, color, thickness)
    cv2.circle(image, bl, radius, color, thickness)
    cv2.circle(image, br, radius, color, thickness)


def draw_text(image, text, position, background_color, font_color, font_scale=0.5, font_thickness=1):
    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    # Create a filled rectangle for text background
    text_background_top_left = (position[0] - 5, position[1] + 5)
    text_background_bottom_right = (position[0] + text_size[0] + 5, position[1] - text_size[1] - 5)
    
    cv2.rectangle(image, text_background_top_left, text_background_bottom_right, background_color, cv2.FILLED)
    
    # Add text on top of the rectangle
    text_position = (position[0], position[1] - 5)
    cv2.putText(image, text, text_position, font, font_scale, font_color, font_thickness)

def get_box_details(boxes):
    cls = boxes.cls.tolist()  # Convert tensor to list
    xyxy = boxes.xyxy
    conf = boxes.conf
    xywh = boxes.xywh

    return cls, xyxy, conf, xywh
