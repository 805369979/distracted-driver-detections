"""
@AmineHorseman
Sep, 1st, 2016
"""
import os

class Dataset:
    name = 'Jaffe'
    train_folder = 'jaffe_features/train'
    validation_folder = 'jaffe_features/test'
    # test_folder = 'fer2013_features/PrivateTest'
    shape_predictor_path='shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_lndmarks.dat'
    trunc_trainset_to = -1  # put the number of train images to use (-1 = all images of the train set)
    trunc_validationset_to = -1
    trunc_testset_to = -1

class Network:
    model = 'B'
    input_size = 256
    output_size = 7
    activation = 'relu'
    loss = 'categorical_crossentropy'
    use_landmarks = True
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
    keep_prob = 0.956   # dropout = 1 - keep_prob
    learning_rate = 0.016
    learning_rate_decay = 0.86
    decay_step = 50
    optimizer = 'momentum'  # {'momentum', 'adam', 'rmsprop', 'adagrad', 'adadelta'}
    optimizer_param = 0.95   # momentum value for Momentum optimizer, or beta1 value for Adam

class Training:
    batch_size = 32
    epochs = 120
    snapshot_step = 500
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
    learning_rate = {'min': 0.001, 'max': 0.1}
    learning_rate_decay = {'min': 0.5, 'max': 0.99}
    optimizer = ['momentum']   # ['momentum', 'adam', 'rmsprop', 'adagrad', 'adadelta']
    optimizer_param = {'min': 0.5, 'max': 0.99}
    keep_prob = {'min': 0.7, 'max': 0.99}

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
