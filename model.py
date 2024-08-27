# from roboflow import Roboflow
# rf = Roboflow(api_key="R61mAgILef91633lP9vE")
# project = rf.workspace("xcvb-t6npm").project("pencil-detector-tltqg")
# version = project.version(1)
# dataset = version.download("yolov8")




from roboflow import Roboflow
rf = Roboflow(api_key="R61mAgILef91633lP9vE")
project = rf.workspace("ar-lab").project("finger-detection-4ydhs")
version = project.version(1)
dataset = version.download("yolov8")
                