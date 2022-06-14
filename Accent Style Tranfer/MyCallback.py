from tensorflow.keras.callbacks import Callback
from tensorflow.python.platform import tf_logging as logging

class MyCallback(Callback):
    
    def __init__(self, monitor='accuracy', value=0.93):
        self.monitor = monitor
        self.value = value
        
    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current > self.value:
            self.best_weights = self.model.get_weights()
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)
            print(f'Model training stopped as "{self.monitor}" reached value = {self.value}')
    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning('Early stopping conditioned on metric `%s` '
                          'which is not available. Available metrics are: %s',
                          self.monitor, ','.join(list(logs.keys())))
        return monitor_value