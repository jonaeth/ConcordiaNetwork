from Experiments.CollectiveActivity.collective import *


def return_dataset(cfg):
    if cfg.dataset_name=='collective':
        print('-----'*5)
        print('Sequence from 1-44')
        collective_read_dataset(cfg, cfg.data_path, list(range(1, 45)))
        print('Sequence from 44-72')
        collective_read_dataset(cfg, cfg.data_path, list(range(45, 73)))
        print('Training set')
        train_anns=collective_read_dataset(cfg, cfg.data_path, cfg.train_seqs)
        train_frames=collective_all_frames(train_anns)
        print('-----'*5)
        print('Test set')
        test_anns=collective_read_dataset(cfg, cfg.data_path, cfg.test_seqs)
        test_frames=collective_all_frames(test_anns)
        print('-----'*5)

        training_set=CollectiveDataset(train_anns,train_frames,
                                       cfg.data_path,cfg.image_size,cfg.out_size,
                                       num_frames=cfg.num_frames,is_training=True,is_finetune=False, is_gnn=cfg.training_stage==2)

        validation_set=CollectiveDataset(test_anns,test_frames,
                                         cfg.data_path,cfg.image_size,cfg.out_size,
                                         num_frames=1,is_training=False,is_finetune=False, is_gnn=cfg.training_stage==2)

    else:
        assert False

    
    print('Reading dataset finished...')
    print('%d train samples'%len(train_frames))
    print('%d test samples'%len(test_frames))
    return training_set, validation_set
