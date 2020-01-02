import os
import string
import argparse
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from text_recognize.utils import CTCLabelConverter, AttnLabelConverter
from text_recognize.dataset import RawDataset, AlignCollate
from text_recognize.model import Model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Recognition:
    def __init__(
        self,
        image_folder,
        saved_model,
        Transformation,
        FeatureExtraction,
        SequenceModeling,
        Prediction,
        workers=4,
        batch_size=192,
        batch_max_length=25,
        imgH=32,
        imgW=100,
        rgb=False,
        character="0123456789abcdefghijklmnopqrstuvwxyz",
        sensitive=False,
        PAD=False,
        num_fiducial=20,
        input_channel=1,
        output_channel=512,
        hidden_size=256,
    ):
        self.image_folder = image_folder
        self.saved_model = saved_model
        self.Transformation = Transformation
        self.FeatureExtraction = FeatureExtraction
        self.SequenceModeling = SequenceModeling
        self.Prediction = Prediction
        self.workers = workers
        self.batch_size = batch_size
        self.batch_max_length = batch_max_length
        self.imgH = imgH
        self.imgW = imgW
        self.rgb = rgb
        self.character = character
        self.sensitive = sensitive
        self.PAD = PAD
        self.num_fiducial = num_fiducial
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_size = hidden_size

        """ vocab / character number configuration """
        if self.sensitive:
            self.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        cudnn.benchmark = True
        cudnn.deterministic = True
        self.num_gpu = torch.cuda.device_count()

        self.__loadNet()

    def __loadNet(self):
        if "CTC" in self.Prediction:
            self.converter = CTCLabelConverter(self.character)
        else:
            self.converter = AttnLabelConverter(self.character)
        self.num_class = len(self.converter.character)

        if self.rgb:
            self.input_channel = 3
        model = Model(self)
        logger.info(
            "model input parameters {} {} {} {} {} {} {} {} {} {} {} {}".format(
                self.imgH,
                self.imgW,
                self.num_fiducial,
                self.input_channel,
                self.output_channel,
                self.hidden_size,
                self.num_class,
                self.batch_max_length,
                self.Transformation,
                self.FeatureExtraction,
                self.SequenceModeling,
                self.Prediction,
            )
        )
        self.model = torch.nn.DataParallel(model).to(device)
        # load model
        print("loading pretrained model from %s" % self.saved_model)
        self.model.load_state_dict(torch.load(self.saved_model, map_location=device))
        self.model.eval()
        logger.info("Model loaded")

    def TextRecognize(self):
        results = {}
        logger.info("Loading image boxes")
        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        AlignCollate_demo = AlignCollate(
            imgH=self.imgH, imgW=self.imgW, keep_ratio_with_pad=self.PAD
        )
        demo_data = RawDataset(root=self.image_folder, opt=self)  # use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=int(self.workers),
            collate_fn=AlignCollate_demo,
            pin_memory=True,
        )

        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                # For max length prediction
                length_for_pred = torch.IntTensor([self.batch_max_length] * batch_size).to(device)
                text_for_pred = (
                    torch.LongTensor(batch_size, self.batch_max_length + 1).fill_(0).to(device)
                )

                if "CTC" in self.Prediction:
                    preds = self.model(image, text_for_pred)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    preds_index = preds_index.view(-1)
                    preds_str = self.converter.decode(preds_index.data, preds_size.data)

                else:
                    preds = self.model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index, length_for_pred)

                # dashed_line = "-" * 80
                # head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

                # print(f"{dashed_line}\n{head}\n{dashed_line}")

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for img_name, pred, pred_max_prob in zip(
                    image_path_list, preds_str, preds_max_prob
                ):
                    if "Attn" in self.Prediction:
                        pred_EOS = pred.find("[s]")
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    results[img_name] = {
                        "predicted_labels": pred,
                        "confidence_score": float(confidence_score),
                    }
                    # remove processed img
                    os.remove(img_name)
                    # print(f"{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}")
        try:
            logger.info(f"{img_name.split('_box_')[0]} recognized, total {len(results)}")
        except:
            logger.warning("No image boxes")
        return results


if __name__ == "__main__":
    path_abs = os.path.dirname(os.path.abspath(__file__))
    test = Recognition(
        image_folder=f"{path_abs}/../images/",
        saved_model=f"{path_abs}/TPS-ResNet-BiLSTM-Attn.pth",
        Transformation="TPS",
        FeatureExtraction="ResNet",
        SequenceModeling="BiLSTM",
        Prediction="Attn",
    )
    result = test.TextRecognize()
    print(result)
