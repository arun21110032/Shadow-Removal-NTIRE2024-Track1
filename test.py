import warnings

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, mean_squared_error, structural_similarity_index_measure
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from data import get_test_data
from models import *
from utils import *

warnings.filterwarnings('ignore')
TEST_DIR= "E:/SEM_6/CV/proj/finaltest"  #test directory
opt = Config('config.yml')
print(opt)

seed_everything(opt.OPTIM.SEED)


def test():
    accelerator = Accelerator()

    # Data Loader
    val_dir = TEST_DIR

    test_dataset = get_test_data(val_dir, {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H, 'ori': True})
    testloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

    # Model & Metrics
    model = Model()

    load_checkpoint(model, opt.TESTING.WEIGHT)

    model, testloader = accelerator.prepare(model, testloader)

    model.eval()

    size = len(testloader)

    if not os.path.exists("Ftestresult1"):
        os.makedirs("Ftestresult1")

    for idx, test_data in enumerate(tqdm(testloader)):

        inp = test_data[0].contiguous()

        with torch.no_grad():
            res = model(inp)
        print(os.getcwd())
        save_image(res, os.path.join(os.getcwd(), "Ftestresult1", test_data[1][0] + '.png'))
 
    print('test over')

if __name__ == '__main__':
    test()
