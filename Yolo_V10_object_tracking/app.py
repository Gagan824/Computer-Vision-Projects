from matplotlib.pylab import f
import streamlit as st
import cv2
from ultralytics import YOLOv10
import numpy as np
from deep_sort.deep_sort import DeepSort
import time
import datetime
import tempfile
from draw_utils import draw_label, draw_rounded_rectangle, draw_text, get_box_details
import pandas as pd
import ffmpeg
import uuid

model = YOLOv10('yolov10x.pt')
deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'

details = []
prev_details = {}
frames = []
unique_track_ids = set()
frame_no = 0
i = 0
counter, fps, elapsed = 0, 0, 0
start_time = time.perf_counter()

def track_video(frame, model, object_, detection_threshold, tracker, frame_no):
    og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = og_frame.copy()

    results = model(frame)

    bboxes_xywh = []
    confs = []

    class_names = list(model.names.values())
    cls, xyxy, conf, xywh = get_box_details(results[0].boxes) # type: ignore

    for c, b, co in zip(cls, xywh, conf.cpu().numpy()):
        if class_names[int(c)] == object_ and co >= detection_threshold:
            bboxes_xywh.append(b.cpu().numpy())
            confs.append(co)

    bboxes_xywh = np.array(bboxes_xywh, dtype=float)
    print(bboxes_xywh)
    if len(bboxes_xywh) >= 1:
        tracks = tracker.update(bboxes_xywh, confs, og_frame)
        
        ids = []
        for track in tracker.tracker.tracks:
            track_id = track.track_id
            hits = track.hits
            x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates in (x1, y1, x2, y2) format
            w = x2 - x1  # Calculate width
            h = y2 - y1  # Calculate height

            # Set color values for red, blue, and green
            blue_color = (0, 0, 255)  
            red_color = (255, 0, 0)  
            green_color = (0, 255, 0)  

            # Determine color based on track_id
            color_id = track_id % 3
            if color_id == 0:
                color = red_color
                color_name = 'Red'
            elif color_id == 1:
                color = blue_color
                color_name = 'Blue'
            else:
                color = green_color
                color_name = 'Green'

            draw_rounded_rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 1, 15) # type: ignore

            text_color = (255, 255, 255)  # White color for text
            draw_label(og_frame, f"{object_}-{track_id}", (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, text_color) # type: ignore
            
            print(color_name)
            if track_id not in prev_details:
                prev_details[track_id] = [time.time(), color_name]           

            # Add the track_id to the set of unique track IDs
            unique_track_ids.add(track_id)
            ids.append(track_id)

        prev_ids = list(prev_details.keys())
        ids_done = set(prev_ids)^set(ids)
        
        # Update the person count based on the number of unique track IDs
        object_counts = len(unique_track_ids)

        for id in ids_done:
            details.append([object_, id, time.time() - prev_details[id][0], prev_details[id][1], frame_no-1])
            del prev_details[id]
                        
        # Draw person count on frame
        og_frame = cv2.cvtColor(og_frame, cv2.COLOR_BGR2RGB)
        og_frame = cv2.resize(og_frame, (700, 600))

        font_color = (255, 255, 255)  # White font

        # Position to draw the text (bottom-left corner)
        position = (10, 30)
        background_color = (0, 0, 0)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        text = f'Frame: {frame_no} | Time: {timestamp} | Count: {object_counts}'
        # # Draw the text on the image
        draw_text(og_frame, text, position, background_color, font_color)

        return og_frame, object_counts
    
    else:
        og_frame = cv2.cvtColor(og_frame, cv2.COLOR_BGR2RGB)
        return og_frame, len(unique_track_ids)


st.title('Track your video here')

st.markdown(
    """
        <style>
            [data-testid="stSidebar"][area-expanded="true"] > div:first-child{
                width: 350px
            }
            [data-testid="stSidebar"][area-expanded="false"] > div:first-child{
                width: 350px
                margin-left: -350px
            }
        </style>

    """,
    unsafe_allow_html=True
)
  
st.sidebar.title("Tracker")
st.sidebar.subheader("Options")

@st.cache_data()
def frame_resize(frame, width=None, height=None, inter_=cv2.INTER_AREA):
    dim = None
    (h, w) = frame.shape[:2]
    
    if width is None and height is None:
        return frame
    
    if width is None:
        r = width/float(w)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width, int(h*r))

    resized_frame = cv2.resize(frame, dim, interpolation=inter_)

    return resized_frame

@st.cache_data()
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


app_mode = st.sidebar.selectbox('Please select', ['Run on local video', 'Run on live feed'])

object_ = st.sidebar.selectbox('Select object to track', list(model.names.values()), placeholder='Select any..')
st.sidebar.markdown('---')
st.sidebar.text('Configurations')

detection_threshold = st.sidebar.slider('Select value for detection threshold', min_value=0.1, max_value=1.0, value=0.5)
max_iou_distance = st.sidebar.slider('Select value for max iou distance', min_value=0.1, max_value=1.0, value=0.5)
min_confidence = st.sidebar.slider('Select value for min confidence', min_value=0.1, max_value=1.0, value=0.3)
max_distance = st.sidebar.slider('Select value for max distance', min_value=0.1, max_value=1.0, value=0.2)

tracker = DeepSort(model_path=deep_sort_weights, max_age=70, n_init=5, max_iou_distance=0.8, min_confidence=min_confidence, 
                   max_dist=max_distance)

if app_mode == 'Run on local video':
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.markdown(
        """
            <style>
                [data-testid="stSidebar"][area-expanded="true"] > div:first-child{
                    width: 350px
                }
                [data-testid="stSidebar"][area-expanded="false"] > div:first-child{
                    width: 350px
                    margin-left: -350px
                }
            </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('Output')
    stframe = st.empty()

    st.sidebar.markdown('---')
    video_file = st.sidebar.file_uploader('Upload your video file here', type = ['mp4', 'mov', 'avi', 'm4v'])
    t_file = tempfile.NamedTemporaryFile(delete=False)

    record = st.sidebar.checkbox('Record Video')

    if record:
        st.checkbox('Reording', value=True)

    if not video_file:
        st.error('Video not found')
    else:
        t_file.write(video_file.read())
        cap = cv2.VideoCapture(t_file.name)        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = 'output1.mp4'       
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            st.error("Error: Failed to open video writer")
            print("Error: Failed to open video writer")
        else:
            st.sidebar.text('Uploaded Video')
            st.sidebar.video(t_file.name)

            kpi1, kpi2, kpi3 = st.columns(3)

            with kpi1:
                st.markdown('**Frame rate**')
                kpi1_text = st.markdown("0")

            with kpi2:
                st.markdown('**Object Counts**')
                kpi2_text = st.markdown("0")

            with kpi3:
                st.markdown('**Frame Width**')
                kpi3_text = st.markdown("0")
            
            st.markdown('<hr/>', unsafe_allow_html=True)

            prev_time = 0
            frame_no = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    print(prev_details)
                    for id, val in prev_details.items():
                        details.append([object_, id, time.time() - val[0], val[1], frame_no-1])
                    break

                frame, object_count = track_video(frame, model, object_, detection_threshold, tracker, frame_no)

                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time

                kpi1_text.write(f"<h1 style=' color: red;'> {int(fps)} </h1>", unsafe_allow_html=True)
                kpi2_text.write(f"<h1 style=' color: red;'> {int(object_count)} </h1>", unsafe_allow_html=True)
                kpi3_text.write(f"<h1 style=' color: red;'> {int(frame_width)} </h1>", unsafe_allow_html=True)

                frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
                frame = frame_resize(frame=frame, width=1080, height=720)
                stframe.image(frame, channels='BGR', use_column_width=True)

                frame_no += 1

                if record:
                    frame_1 = cv2.resize(frame, (frame_width, frame_height))
                    out.write(frame_1)  # Ensure the frame is in BGR format

            cap.release()
            out.release()

        st.text('Video Processed')
        with st.spinner('Wait for it...'):
            name = uuid.uuid1()
            output_file = f'webmfiles/{name}.webm'
            ffmpeg.input(output_path).output(output_file).run()
            video_file = open(output_file, "rb")
            video_bytes = video_file.read()

        st.toast('Video loaded')
        st.video(video_bytes)

        st.markdown('---')
        
        details = pd.DataFrame(details, columns = ['Object', 'id', 'duration', 'color', 'frame_number'])
        st.dataframe(details)

        csv = convert_df(details)
        st.markdown(f'<a href="data:text/csv;charset=utf-8,{csv}" download="{name}.csv">Download CSV</a>', unsafe_allow_html=True)

        # Use JavaScript to simulate a click on the hidden link
        st.markdown('<script>document.querySelector("a").click();</script>', unsafe_allow_html=True)

        st.toast('Details file downloaded!!')

elif app_mode == 'Run on live feed':
    st.markdown(
        """
            <style>
                [data-testid="stSidebar"][area-expanded="true"] > div:first-child{
                    width: 350px
                }
                [data-testid="stSidebar"][area-expanded="false"] > div:first-child{
                    width: 350px
                    margin-left: -350px
                }
            </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('Output')
    stframe = st.empty()

    st.sidebar.markdown('---')
    video_file = st.sidebar.file_uploader('Upload your video file here', type = ['mp4', 'mov', 'avi', 'm4v'])
    t_file = tempfile.NamedTemporaryFile(delete=False)

    record = st.sidebar.checkbox('Record Video')

    if record:
        st.checkbox('Reording', value=True)

    video_device = st.selectbox('Select device', options=[0, 1])
    print(video_device)

    cap = cv2.VideoCapture(int(video_device))       
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'live_output1.mp4'       
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        st.error("Error: Failed to open video writer")
        print("Error: Failed to open video writer")
    else:
        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown('**Frame rate**')
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown('**Object Counts**')
            kpi2_text = st.markdown("0")

        with kpi3:
            st.markdown('**Frame Width**')
            kpi3_text = st.markdown("0")

        st.markdown('<hr/>', unsafe_allow_html=True)

        prev_time = 0
        frame_no = 0
        
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print(prev_details)
                for id, val in prev_details.items():
                    details.append([object_, id, time.time() - val[0], val[1], frame_no-1])
                break

            frame, object_count = track_video(frame, model, object_, detection_threshold, tracker, frame_no)

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            kpi1_text.write(f"<h1 style=' color: red;'> {int(fps)} </h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style=' color: red;'> {int(object_count)} </h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style=' color: red;'> {int(frame_width)} </h1>", unsafe_allow_html=True)

            frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
            frame = frame_resize(frame=frame, width=1080, height=720)
            stframe.image(frame, channels='BGR', use_column_width=True)

            frame_no += 1

            if record:
                frame_1 = cv2.resize(frame, (frame_width, frame_height))
                out.write(frame_1)  # Ensure the frame is in BGR format

        cap.release()
        out.release()

    st.text('Video Processed')
    with st.spinner('Wait for it...'):
        name = uuid.uuid1()
        output_file = f'webmfiles/{name}.webm'
        ffmpeg.input(output_path).output(output_file).run()
        video_file = open(output_file, "rb")
        video_bytes = video_file.read()

    st.toast('Video loaded')
    st.video(video_bytes)

    st.markdown('---')
    
    details = pd.DataFrame(details, columns = ['Object', 'id', 'duration', 'color', 'frame_number'])
    st.dataframe(details)

    csv = convert_df(details)
    st.markdown(f'<a href="data:text/csv;charset=utf-8,{csv}" download="{name}.csv">Download CSV</a>', unsafe_allow_html=True)

    # Use JavaScript to simulate a click on the hidden link
    st.markdown('<script>document.querySelector("a").click();</script>', unsafe_allow_html=True)

    st.toast('Details file downloaded!!')