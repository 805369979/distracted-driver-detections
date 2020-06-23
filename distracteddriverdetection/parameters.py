"""
Sep, 1st, 2016
"""
import os

class Dataset:
    # name = 'Fer2013Fer2013'
    name = 'SSS'
    train_folder = './driver_feature_hogSecond2.5/train0.75'
    # train_folder = './driver_feature_small/train'
    validation_folder = './driver_feature_hogSecond2.5/test0.75'
    # validation_folder = 'driver_feature_small/test'
    test_folder = './driver_feature_hogSecond2.5/test0.75'
    # test_folder = 'driver_feature_small/test'
    shape_predictor_path='shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_lndmarks.dat'
    # trunc_trainset_to = 50000  # put the number of train images to use (-1 = all images of the train set)
    # trunc_validationset_to = 300
    # trunc_testset_to = 300

class Network:
    model = 'A'
    input_size = 224
    input_size1 = 224

    output_size = 10
    activation = 'relu'
    loss = 'categorical_crossentropy'
    use_landmarks = False
    use_hog_and_landmarks = True
    use_hog_sliding_window_and_landmarks = True
    use_batchnorm_after_conv_layers = True
    use_batchnorm_after_fully_connected_layers = True

class Hyperparams:
    # keep_prob = 0.75   # dropout = 1 - keep_prob
    # learning_rate = 0.016
    # learning_rate_decay = 0.86
    # decay_step = 50
    # optimizer = 'momentum'  # {'momentum', 'adam', 'rmsprop', 'adagrad', 'adadelta'}
    # optimizer_param = 0.95
    # 0.75 0.95

    keep_prob = 0.001  # dropout = 1 - keep_prob
    # keep_prob = 0.04814844038206214  # dropout = 1 - keep_prob
    learning_rate = 0.078
    # learning_rate = 0.06
    learning_rate_decay = 0.9886303756
    decay_step = 50
    optimizer = 'momentum'  # {'momentum', 'adam', 'rmsprop', 'adagrad', 'adadelta'}
    # optimizer_param = 0.633264271169259  # momentum value for Momentum optimizer, or beta1 value for Adam
    optimizer_param = 0.65  # momentum value for Momentum optimizer, or beta1 value for Adam



    # keep_prob = 0.8117731896183599  # dropout = 1 - keep_prob
    # learning_rate = 0.03786456902380755
    # learning_rate_decay = 0.9886235518723528
    # decay_step = 50
    # optimizer = 'momentum'  # {'momentum', 'adam', 'rmsprop', 'adagrad', 'adadelta'}
    # optimizer_param = 0.6695905203885134
class Training:

    batch_size =128
    epochs =200
    snapshot_step =8000000
    vizualize = True
    logs_dir = "logs"
    checkpoint_dir = "checkpoints/chk"
    best_checkpoint_path = "checkpoints/best/"
    max_checkpoints = 1
    checkpoint_frequency = 1.0 # in hours
    save_model = True
    save_model_path = "best_model/saved_model.bin"

class VideoPredictor:
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    print_emotions = True
    camera_source = 0
    face_detection_classifier = "lbpcascade_frontalface.xml"
    show_confidence = True
    time_to_wait_between_predictions = 0.5

class OptimizerSearchSpace:
    learning_rate = {'min': 0.05, 'max': 0.07}
    learning_rate_decay = {'min': 0.95, 'max': 0.99}
    optimizer = ['momentum']   # ['momentum', 'adam', 'rmsprop', 'adagrad', 'adadelta']
    # optimizer = ['momentum', 'adam', 'rmsprop', 'adagrad', 'adadelta']
    optimizer_param = {'min': 0.6, 'max': 0.8}
    keep_prob = {'min': 0.7, 'max': 0.7}

def make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

DATASET = Dataset()
NETWORK = Network()
TRAINING = Training()
HYPERPARAMS = Hyperparams()
VIDEO_PREDICTOR = VideoPredictor()
OPTIMIZER = OptimizerSearchSpace()

make_dir(TRAINING.logs_dir)
make_dir(TRAINING.checkpoint_dir)
