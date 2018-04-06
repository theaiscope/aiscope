from mxnet import gluon
from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxPrior, MultiBoxTarget
from mxnet.gluon import nn

NUM_CLASSES = 1
BATCH_SIZE = 8


class SSD(gluon.Block):
    def __init__(self, num_classes, n_scales, **kwargs):
        super(SSD, self).__init__(**kwargs)

        self.n_scales = n_scales
        self.num_classes = num_classes

        self.sizes = [[.1, .2, .272], [.1, .37, .447], [.1, .54, .619], [.1, .71, .79],
                      [.1, .88, .961]]  # * self.n_scales  # , [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        self.ratios = [[1, 0.5, 2]] * 5  # , 2, .5]]#

        self.NUM_ANCHORS = len(self.sizes[0]) + len(self.ratios[0]) - 1

        with self.name_scope():
            self.features = self.base_network()

            downsamples = nn.Sequential()

            for i in range(self.n_scales):
                downsamples.add(self.down_sample(128))

            class_preds = nn.Sequential()
            box_preds = nn.Sequential()

            for i in range(self.n_scales):
                class_preds.add(self.class_predictor(self.NUM_ANCHORS, self.num_classes))
                box_preds.add(self.box_predictor(self.NUM_ANCHORS))

            self.downsamples = downsamples
            self.class_preds = class_preds
            self.box_preds = box_preds

    def forward(self, x):
        feat = self.features(x)

        default_anchors = []
        predicted_boxes = []
        predicted_classes = []

        for i in range(self.n_scales):
            feat = self.downsamples[i](feat)

            default_anchors.append(
                MultiBoxPrior(feat,
                              clip=True,
                              sizes=self.sizes[i],
                              ratios=self.ratios[i])
            )

            bp = self.box_preds[i](feat)
            cp = self.class_preds[i](feat)

            predicted_boxes.append(self.flatten_prediction(bp))
            predicted_classes.append(self.flatten_prediction(cp))

        anchors = self.concat_predictions(default_anchors)
        box_preds = self.concat_predictions(predicted_boxes)
        class_preds = self.concat_predictions(predicted_classes)
        class_preds = nd.reshape(class_preds, shape=(0, -1, self.num_classes + 1))

        return anchors, box_preds, class_preds

    @staticmethod
    def class_predictor(num_anchors, num_classes):
        """return a layer to predict classes"""
        return nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)

    @staticmethod
    def box_predictor(num_anchors):
        """return a layer to predict delta locations"""
        return nn.Conv2D(num_anchors * 4, 3, padding=1)

    @staticmethod
    def concat_predictions(preds):
        return nd.concat(*preds, dim=1)

    @staticmethod
    def flatten_prediction(pred):
        return nd.flatten(nd.transpose(pred, axes=(0, 2, 3, 1)))

    @staticmethod
    def concat_predictions(preds):
        return nd.concat(*preds, dim=1)

    @staticmethod
    def down_sample(num_filters):
        """stack two Conv-BatchNorm-Relu blocks and then a pooling layer
        to halve the feature size"""
        out = nn.HybridSequential()
        for _ in range(2):
            out.add(nn.Conv2D(num_filters, 3, strides=1, padding=1))
            out.add(nn.BatchNorm(in_channels=num_filters))
            out.add(nn.Activation('relu'))
        out.add(nn.MaxPool2D(2))
        return out

    def base_network(self):

        features = nn.HybridSequential()

        for nfilters in [32, 64, 128]:
            features.add(self.down_sample(nfilters))

        return features
