from ultralytics.data.annotator import auto_annotate
# Testing for auto_annotation
auto_annotate(data="Processed_Images", det_model="yolov8x.pt", sam_model='sam_b.pt')
