import os
import glob
import logging
from sklearn.utils import shuffle

class DatasetCollector:

    def __init__(self):
        pass

    def classes(self):
        return []

    def train(self, cls=None):
        return []

    def val(self, cls=None):
        return []

    def test(self, cls=None):
        return []


class SanityCollector(DatasetCollector):

    def __init__(self, *args, **kwargs):
        self.cls = ['chair']

    def classes(self):
        return self.cls

    def _gather(self):
        return [('./data/model.128.png', './data/model.shl.mat')]

    def train(self, cls=None):
        return self._gather()

    def val(self, cls=None):
        return self._gather()

    def test(self, cls=None):
        return self._gather()


class ShapeNetPTNCollector(DatasetCollector):
    """ 
    Collects samples from ShapeNet using the version of Yan et al.
    """

    def __init__(self, base_dir, crop=True):
        assert os.path.exists(base_dir), ('Base directory for PTN dataset does not exist [%s].' % base_dir)
        self.base_dir  = base_dir
        self.id_dir    = os.path.join(self.base_dir, 'shapenetcore_ids')
        self.view_dir  = os.path.join(self.base_dir, 'shapenetcore_viewdata')
        self.shape_dir = os.path.join(self.base_dir, 'shapenetcore_voxdata')
        self.crop      = crop
        self.cls       = []

        for c in sorted([d[:-12] for d in os.listdir(self.id_dir) if d.endswith('_testids.txt')]):
            if  os.path.exists(os.path.join(self.id_dir, c+'_trainids.txt')) and \
                os.path.exists(os.path.join(self.id_dir, c+'_valids.txt')) and \
                os.path.exists(os.path.join(self.view_dir, c)) and \
                os.path.exists(os.path.join(self.shape_dir, c)):
                self.cls.append(c)

    def _gather(self, subset, cls=None):
        if cls is None:
            cls = self.classes()

        samples = []    

        shape_suffix = 'model.shl.mat' if self.representation == 'shl' else 'model.vox.mat'
        for c in cls:
            logging.info('Collecting %s/%s...' % (subset, c))
            with open(os.path.join(self.id_dir, '%s_%sids.txt' % (c, subset))) as f:
                for line in f:
                    # format is class/id
                    id = line.strip().split('/')[1]
                    shapepath = os.path.join(self.shape_dir, c, id, shape_suffix)
                    # check images
                    viewdir = os.path.join(self.view_dir, c, id)                    
                    for file in sorted(os.listdir(viewdir)):
                        if self.crop and file.endswith('.128.png'):
                            samples.append((os.path.join(viewdir, file), shapepath))
                        if not self.crop and file.endswith('.png') and not file.endswith('.128.png'):
                            samples.append((os.path.join(viewdir, file), shapepath))

        return samples

    def classes(self):
        return self.cls

    def train(self, cls=None):
        return self._gather('train', cls)

    def val(self, cls=None):
        return self._gather('val', cls)

    def test(self, cls=None):
        return self._gather('test', cls)


class BlendswapOGNCollector(DatasetCollector):
    """
    OGN dataset can be taken from:
        https://github.com/lmb-freiburg/ogn
    
    """

    def __init__(self, base_dir, resolution=512):
        res2dir = {64:'64_l4', 128:'128_l4', 256:'256_l5', 512:'512_l5'}
        self.base_dir = os.path.join(base_dir, res2dir[resolution])
        assert os.path.exists(self.base_dir), ('Base directory for OGN Blendswap dataset does not exist [%s].' % self.base_dir)

    def _gather(self):
        samples = []
        shape_suffix = '.shl.mat'
        
        for file in sorted(os.listdir(self.base_dir)):
            if file.endswith(shape_suffix):             
                samples.append(os.path.join(self.base_dir, file))

        return samples

    def classes(self):
        return None

    def train(self):
        return self._gather('all')

    def val(self):
        return self._gather('all')

    def test(self):
        return self._gather('all')
    pass


class ShapeNetCarsOGNCollector(DatasetCollector):
    """
    OGN dataset can be taken from:
        https://github.com/lmb-freiburg/ogn
    Assuming that text files with sample paths are in root dir.
    """
    res2name = {
        '64': '64_l4',
        '128': '128_l4',
        '256': '256_l5',
    }
    
    def __init__(self, base_dir, shapenet_base_dir, resolution=128, crop=False):
        """
            
        base_dir: str
            Path to the OGN ShapeNet with cars dataset
        shapenet_base_dir: str
            Path to the ShapeNet folder, where cars are rendered
        """
        self.dataset_folder = base_dir
        self.base_dir = os.path.join(base_dir, ShapeNetCarsOGNCollector.res2name[str(resolution)])     
        assert os.path.exists(self.base_dir), ('Base directory for OGN ShapeNet Cars dataset does not exist [%s].' % self.base_dir)
        self.shapenet_base_dir = shapenet_base_dir
        assert os.path.exists(self.shapenet_base_dir), ('ShapeNet rendering directory for OGN ShapeNet Cars dataset does not exist [%s].' % self.shapenet_base_dir)

        self.crop = crop
        
        for s in ['train', 'validation', 'test']:
            id_path = os.path.join(self.dataset_folder, 'shapenet_cars_rendered_new_%s.txt' % s)
            assert os.path.exists(id_path), ('Could not find id list for %s set [%s].' % (s, id_path))
        assert os.path.exists(self.base_dir), ('Base directory for OGN ShapeNet Cars dataset does not exist [%s].' % self.base_dir)

    def classes(self):
        return ['car']

    def _gather(self, subset):
        samples = []
        shape_suffix = '.shl.mat'
        with open(os.path.join(self.dataset_folder, 'shapenet_cars_rendered_new_%s.txt' % subset)) as f:
            for line in f:
                img_path, id = line.strip().split(' ')
                img_id      = img_path.split('/')[-1].split('.')[0]
                shapenet_id = img_path.split('/')[-3]
                img_path    = os.path.join(self.shapenet_base_dir, '02958343', shapenet_id, \
                    'rendering', img_id + ('.128.png' if self.crop else '.png'))
                shape_path  = os.path.join(self.base_dir, str(id).zfill(4) + shape_suffix)
                samples.append((img_path, shape_path))
        return samples

    def train(self, cls=None):
        return self._gather('train')

    def val(self, cls=None):
        return self._gather('validation')

    def test(self, cls=None):
        return self._gather('test')

    
class FaustCollector(DatasetCollector):
    res2name = {
        '64': '64_l4',
        '128': '128_l4',
        '256': '256_l5',
    }
    def __init__(self, base_dir, resolution=128, **kwargs):
        res_name = FaustCollector.res2name.get(str(resolution))
        if res_name is None:
            raise Exception(f'Unknown resolution for faust dataset, resolution={resolution}')
        self.test_dir   = os.path.join(base_dir, res_name, 'test')
        self.train_dir  = os.path.join(base_dir, res_name, 'train')
        self.test_images_dir = os.path.join(base_dir, 'test', 'scans')
        self.train_images_dir = os.path.join(base_dir, 'training', 'scans')
        self.representation = 'shl'
        self.crop = False

    def classes(self):
        return self.cls

    def _gather(self, subset, cls=None):
        samples = []    
        shape_suffix = 'shl.mat' if self.representation == 'shl' else 'vox.mat'
        if subset == 'train':
            folder_path = self.train_dir
            img_path = self.train_images_dir
        elif subset == 'test' or subset == 'val':
            folder_path = self.test_dir
            img_path = self.test_images_dir
        else:
            raise Exception(f'Wrong subset={subset}')
        logging.info(f'Collecting {subset}...')
        for shapepath in glob.glob(os.path.join(folder_path, '*')):
            if shapepath.endswith(shape_suffix):
                name_file = shapepath.split('/')[-1].split('.')[0]
                if self.crop:
                    viewdir = os.path.join(img_path, f'{name_file}.128.png')
                else:
                    viewdir = os.path.join(img_path, f'{name_file}.png')
                samples.append((viewdir, shapepath))
        return samples

    def train(self, cls=None):
        return self._gather('train', cls)

    def val(self, cls=None):
        return self._gather('test', cls)   

    def test(self, cls=None):
        return self._gather('test', cls) 
            

class ShapeNet3DR2N2Collector(DatasetCollector):
    def __init__(self, base_dir, **kwargs):
        self.shape_dir = os.path.join(base_dir, 'ShapeNetVox32')
        self.view_dir  = os.path.join(base_dir, 'ShapeNetRendering')
        self.list_dir  = os.path.join(base_dir, 'ShapeNetList')
        self.representation = 'shl'
        self.crop = False
        if not os.path.exists(self.list_dir):
            self.write_split()

        self.cls = []
        for c in sorted([d.split('/')[-1].split('_')[0] for d in glob.glob(f'{self.list_dir}/*_test.txt')]):
            if  os.path.exists(os.path.join(self.list_dir, f'{c}_train.txt')) and \
                os.path.exists(os.path.join(self.view_dir, c)) and \
                os.path.exists(os.path.join(self.shape_dir, c)):
                self.cls.append(c)

    def classes(self):
        return self.cls

    def _gather(self, subset, cls=None):
        if cls is None:
            cls = self.classes()

        samples = []    

        shape_suffix = 'model.shl.mat' if self.representation == 'shl' else 'model.vox.mat'
        for c in cls:
            logging.info('Collecting %s/%s...' % (subset, c))
            with open(os.path.join(self.list_dir, '%s_%s.txt' % (c, subset))) as f:
                for line in f:
                    # format is class/id
                    id = line.strip()
                    shapepath = os.path.join(self.shape_dir, c, id, shape_suffix)
                    if not os.path.exists(shapepath):
                        continue
                    # check images
                    viewdir = os.path.join(self.view_dir, c, id, 'rendering')
                    if not os.path.exists(viewdir):
                        continue
                    for file in sorted(os.listdir(viewdir)):
                        if self.crop and file.endswith('.128.png'):
                            samples.append((os.path.join(viewdir, file), shapepath))
                        if not self.crop and file.endswith('.png') and not file.endswith('.128.png'):
                            samples.append((os.path.join(viewdir, file), shapepath))
        return samples

    def train(self, cls=None):
        return self._gather('train', cls)

    def val(self, cls=None):
        return self._gather('val', cls)   

    def test(self, cls=None):
        return self._gather('test', cls)        
    
    def write_split(self, val_split=0.1, test_split=0.2):
        os.makedirs(self.list_dir, exist_ok=True)
        # For each class
        for single_class in glob.glob(f'{self.view_dir}/*'):
            # Each class have N models (obj) - we must create train/val/test split for them
            all_folders = shuffle(glob.glob(f'{single_class}/*'))
            get_id = lambda x: x.split('/')[-1]
            to_train_indx = int(len(all_folders) * (1 - test_split - val_split))
            to_val_indx = int(len(all_folders) * (1 - test_split))
            
            train_id_list = map(get_id, all_folders[:to_train_indx])
            val_id_list = map(get_id, all_folders[to_train_indx: to_val_indx])
            test_id_list = map(get_id, all_folders[to_val_indx:])
            
            class_name = single_class.split('/')[-1]
            with open(f'{self.list_dir}/{class_name}_test.txt', 'a') as fp:
                fp.write("\n".join(test_id_list))
                
            with open(f'{self.list_dir}/{class_name}_val.txt', 'a') as fp:
                fp.write("\n".join(val_id_list))
                
            with open(f'{self.list_dir}/{class_name}_train.txt', 'a') as fp:
                fp.write("\n".join(train_id_list))
            