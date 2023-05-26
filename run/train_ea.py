import time

from utils.data_utils import *
from utils.eval_utils import format_metrics
from models.models_ea import NCModel, KEModel, RKDEA


def train_ea(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    print(f'Using: {args.device}')
    print(f'Using seed: {args.seed}')

    # Load Data
    data = load_data(args)
    args.n_nodes, args.feat_dim = data['x'].shape
    print(f'Num_nodes: {args.n_nodes}')
    print(f'Dim_feats: {args.feat_dim}')
    args.data = data
    Model = None
    args.n_classes = args.feat_dim
    if args.model == 'NC':
        Model = NCModel
    elif args.model == 'KE':
        Model = KEModel
    elif args.model == 'RKDEA':
        Model = RKDEA

    # Model and Optimizer
    model = Model(args)
    print(str(model))
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f'Total number of parameters: {tot_params}')
    if args.cuda is not None and int(args.cuda) >= 0:
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)

    # Train Model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['x'], data['adj'])
        outputs = model.decode(embeddings, data['adj'])
        # outputs, outputs_r = model.encode(data['idx_x'], data['idx_r'])
        if epoch % 50 == 0:
            model.neg_right = model.get_neg(data['train'][:, 0], outputs, args.neg_num)
            model.neg2_left = model.get_neg(data['train'][:, 1], outputs, args.neg_num)
            model.neg_triple = model.get_neg_triplet(data['triple'], data['head'], data['tail'], data['x'].shape[0])
        loss = model.get_loss(outputs, data, 'train')
        # loss = model.get_loss(outputs, outputs_r, data, 'train')
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        if (epoch + 1) % args.log_freq == 0:
            train_metrics = model.compute_metrics(outputs, data, 'train')
            print(' '.join(['Epoch: {:04d}'.format(epoch + 1),
                            'lr: {}'.format(lr_scheduler.get_lr()[0]),
                            format_metrics(train_metrics, 'train'),
                            'time: {:.4f}s'.format(time.time() - t)]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['x'], data['adj'])
            outputs = model.decode(embeddings, data['adj'])
            # outputs, outputs_r = model.encode(data['idx_x'], data['idx_r'])
            val_metrics = model.compute_metrics(outputs, data, 'val')
            print(' '.join(['Epoch: {:04d}'.format(epoch + 1),
                            format_metrics(val_metrics, 'val')]))
            if model.has_improved(best_val_metrics, val_metrics):
                best_emb = outputs
                best_test_metrics = model.compute_metrics(outputs, data, 'test')
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter >= args.patience and epoch > args.min_epochs:
                    print("Early stopping")
                    break

    print('Optimization Finished!')
    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['x'], data['adj'])
        outputs = model.decode(best_emb, data['adj'])
        best_test_metrics = model.compute_metrics(outputs, data, 'test')
    print(' '.join(['Val set results:',
                    format_metrics(best_val_metrics, 'val')]))
    print(' '.join(['Test set results:',
                    format_metrics(best_test_metrics, 'test')]))
    if args.save:
        np.save(f'data/dbp15k/{args.dataset}/{args.model}_embeddings.npy', best_emb.cpu().detach().numpy())
        args.data = []
        json.dump(vars(args), open(f'data/dbp15k/{args.dataset}/{args.model}_config.json', 'w'))
        torch.save(model.state_dict(), f'data/dbp15k/{args.dataset}/{args.model}_model.pth')
        print(f'Saved model!')

