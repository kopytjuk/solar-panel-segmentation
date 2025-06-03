from pathlib import Path

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from solarnet.datasets import ClassifierDataset, SegmenterDataset, make_masks
from solarnet.models import Classifier, Segmenter, train_classifier, train_segmenter
from solarnet.preprocessing import ImageSplitter, MaskMaker


class RunTask:

    @staticmethod
    def make_masks(data_folder='data'):
        """Saves masks for each .tif image in the raw dataset. Masks are saved
        in  <org_folder>_mask/<org_filename>.npy where <org_folder> should be the
        city name, as defined in `data/README.md`.

        Parameters
        ----------
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        """
        mask_maker = MaskMaker(data_folder=Path(data_folder))
        mask_maker.process()
        print("Done!")

    @staticmethod
    def split_images(data_folder='data', imsize=224, empty_ratio=2):
        """Generates images (and their corresponding masks) of height = width = imsize
        for input into the models.

        Parameters
        ----------
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        imsize: int, default: 224
            The size of the images to be generated
        empty_ratio: int, default: 2
            The ratio of images without solar panels to images with solar panels.
            Because images without solar panels are randomly sampled with limited
            patience, having this number slightly > 1 yields a roughly 1:1 ratio.
        """
        splitter = ImageSplitter(data_folder=Path(data_folder))
        splitter.process(imsize=imsize, empty_ratio=empty_ratio)
        print("Done!")

    @staticmethod
    def train_classifier(max_epochs=100, warmup=2, patience=5, val_size=0.1,
                         test_size=0.1, data_folder='data',
                         device: str | None = None,
                         retrain: bool = False
                         ):
        """Train the classifier

        Parameters
        ----------
        max_epochs: int, default: 100
            The maximum number of epochs to train for
        warmup: int, default: 2
            The number of epochs for which only the final layers (not from the ResNet base)
            should be trained
        patience: int, default: 5
            The number of epochs to keep training without an improvement in performance on the
            validation set before early stopping
        val_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the validation set
        test_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the test set
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        device: str, default:
            The device to train the models on (mps, cuda or cpu)
        """
        data_folder = Path(data_folder)

        model_dir = data_folder / 'models'
        model_path = model_dir / 'classifier.model'

        if device is None:
            device = RunTask.determine_torch_device()

        model = Classifier()
        if retrain:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model_name = "classifier_retrained.model"
        else:
            model_name = "classifier.model"

        # move weights to device
        model.to(device)

        processed_folder = data_folder / 'processed'
        dataset = ClassifierDataset(processed_folder=processed_folder, device=device)

        # make a train and val set
        train_mask, val_mask, test_mask = make_masks(len(dataset), val_size, test_size)

        dataset.add_mask(train_mask)
        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(ClassifierDataset(mask=val_mask,
                                                      processed_folder=processed_folder,
                                                      transform_images=False, device=device),
                                    batch_size=64, shuffle=True)
        test_dataloader = DataLoader(ClassifierDataset(mask=test_mask,
                                                       processed_folder=processed_folder,
                                                       transform_images=False, device=device),
                                     batch_size=64)

        train_classifier(model, train_dataloader, val_dataloader, max_epochs=max_epochs,
                         warmup=warmup, patience=patience)

        if not model_dir.exists():
            model_dir.mkdir()
        torch.save(model.state_dict(), model_dir / model_name)

        # save predictions for analysis
        print("Generating test results")
        preds, true = [], []
        with torch.no_grad():
            for test_x, test_y in tqdm(test_dataloader):
                test_preds = model(test_x)
                preds.append(test_preds.squeeze(1).cpu().numpy())
                true.append(test_y.cpu().numpy())

        np.save(model_dir / f'{model_name.split(".")[0]}_preds.npy', np.concatenate(preds))
        np.save(model_dir / f'{model_name.split(".")[0]}_true.npy', np.concatenate(true))

    @staticmethod
    def train_segmenter(max_epochs=100, val_size=0.1, test_size=0.1, warmup=2,
                        patience=5, data_folder='data', use_classifier=True,
                        device: str | None = None):
        """Train the segmentation model

        Parameters
        ----------
        max_epochs: int, default: 100
            The maximum number of epochs to train for
        warmup: int, default: 2
            The number of epochs for which only the final layers (not from the ResNet base)
            should be trained
        patience: int, default: 5
            The number of epochs to keep training without an improvement in performance on the
            validation set before early stopping
        val_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the validation set
        test_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the test set
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        use_classifier: boolean, default: True
            Whether to use the pretrained classifier (saved in data/models/classifier.model by the
            train_classifier step) as the weights for the downsampling step of the segmentation
            model
        device: str, default:
            The device to train the models on (mps, cuda or cpu)
        """
        data_folder = Path(data_folder)
        model = Segmenter()

        if device is None:
            device = RunTask.determine_torch_device()

        print(f"Using device: '{device}'")

        # move weights to device
        model.to(device)

        model_dir = data_folder / 'models'
        if use_classifier:
            classifier_sd = torch.load(model_dir / 'classifier.model')
            model.load_base(classifier_sd)
        processed_folder = data_folder / 'processed'
        dataset = SegmenterDataset(processed_folder=processed_folder, device=device)
        train_mask, val_mask, test_mask = make_masks(len(dataset), val_size, test_size)

        dataset.add_mask(train_mask)
        train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(SegmenterDataset(mask=val_mask,
                                                     processed_folder=processed_folder,
                                                     transform_images=False, device=device),
                                    batch_size=64, shuffle=True)
        test_dataloader = DataLoader(SegmenterDataset(mask=test_mask,
                                                      processed_folder=processed_folder,
                                                      transform_images=False, device=device),
                                     batch_size=64)

        train_segmenter(model, train_dataloader, val_dataloader, max_epochs=max_epochs,
                        warmup=warmup, patience=patience)

        if not model_dir.exists():
            model_dir.mkdir()
        torch.save(model.state_dict(), model_dir / 'segmenter.model')

        print("Generating test results")
        images, preds, true = [], [], []
        with torch.no_grad():
            for test_x, test_y in tqdm(test_dataloader):
                test_preds = model(test_x)
                images.append(test_x.cpu().numpy())
                preds.append(test_preds.squeeze(1).cpu().numpy())
                true.append(test_y.cpu().numpy())

        np.save(model_dir / 'segmenter_images.npy', np.concatenate(images))
        np.save(model_dir / 'segmenter_preds.npy', np.concatenate(preds))
        np.save(model_dir / 'segmenter_true.npy', np.concatenate(true))

    def train_both(self, c_max_epochs=100, c_warmup=2, c_patience=5, c_val_size=0.1,
                   c_test_size=0.1, s_max_epochs=100, s_warmup=2, s_patience=5,
                   s_val_size=0.1, s_test_size=0.1, data_folder='data',
                   device: str | None = None):
        """Train the classifier, and use it to train the segmentation model.
        """
        data_folder = Path(data_folder)
        self.train_classifier(max_epochs=c_max_epochs, val_size=c_val_size, test_size=c_test_size,
                              warmup=c_warmup, patience=c_patience, data_folder=data_folder,
                              device=device)
        self.train_segmenter(max_epochs=s_max_epochs, val_size=s_val_size, test_size=s_test_size,
                             warmup=s_warmup, patience=s_patience, use_classifier=True,
                             data_folder=data_folder, device=device)

    @classmethod
    def segment_new_data(cls, image_dir: str, output_dir: str, batch_size: int = 64,
                         image_size: int = 224, device: str | None = None):
        """Segments new data in IMAGE_DIR and stores the segmentation masks as undistored bitmaps in
        OUTPUT_DIR.

        Args:
            image_dir (str): folder with images (e.g. example-dir/unclassified/123.img). Note that
                you have to create an artificial folder level between the image directory and the
                image. This "unusual" behaviour is due to torch's `DatasetFolder` always requires
                classes.
            output_dir (str): output folder
            batch_size (int, optional): Batch size for the network, reduce
                if you get memory warnings. Defaults to 64.
            image_size (int, optional): Input image edge size used in training. Defaults to 224.
            device (str | None, optional): Device to conduct the inference on (cpu, mps, cuda).
                Defaults to None.
        """

        data_loader = cls.create_dataloader_for_inference(image_dir, batch_size, image_size)

        if device is None:
            device = cls.determine_torch_device()

        # Load the appropriate model based on the model_type parameter
        model_dir = Path("data") / 'models'
        model = Segmenter()
        model_path = model_dir / 'segmenter.model'

        # Load the model's state_dict
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set model to evaluation mode

        # send model weights to device
        model.to(device)
        model.eval()

        print("Generating test results")
        pred_masks = []
        with torch.no_grad():
            for test_x, _ in tqdm(data_loader):
                test_x = test_x.to(device)
                test_preds = model(test_x)

                pred_mask = test_preds.cpu().numpy()
                pred_masks.append(pred_mask)

        pred_masks = np.concatenate(pred_masks)

        # Get the list of image file paths
        image_file_paths = [Path(sample[0]) for sample in data_loader.dataset.imgs]

        # create output folder if necessary
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

        # iterate over each sample and store the segmentation mask as BMP
        for image_path, pred_mask in zip(image_file_paths, pred_masks):

            image_path = Path(image_path)
            
            # 0..1 -> 0..256
            pred_mask = (pred_mask[0, ...]*256).astype(np.uint8)

            # Convert the matrix to an image
            image = Image.fromarray(pred_mask)

            image_base_name = image_path.stem

            # Save the image as a BMP file
            image_output_path = output_dir / f"{image_base_name}.bmp"
            image.save(image_output_path)

        print("Done!")

    @classmethod
    def create_dataloader_for_inference(cls, image_dir, batch_size, image_size)\
            -> torch.utils.data.DataLoader:

        # define the same normalizer as it was used in training
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        image_folder_dset = datasets.ImageFolder(image_dir, transforms.Compose([
            # since image_size is only an int, the resizing does not distort the image,
            # the proportions stay the same. For non square images we need to crop.
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]))
        data_loader = torch.utils.data.DataLoader(
            image_folder_dset,
            batch_size=batch_size, shuffle=False, pin_memory=True)

        return data_loader

    @staticmethod
    def determine_torch_device() -> torch.device:
        """In case user does not provide a desired device, this functions selects
        a device based on the machine's capabilities.

        Returns:
            torch.device: selected device
        """
        if torch.backends.mps.is_available():
            print("Metal Performance Shaders (MPS) backend is available!")
            device = torch.device("mps")
        elif torch.cuda.is_available():
            print("CUDA is available!")
            device = torch.device('cuda:0')
        else:
            print("No GPU acceleration available, falling back to CPUs!")
            device = torch.device('cpu')
        return device
