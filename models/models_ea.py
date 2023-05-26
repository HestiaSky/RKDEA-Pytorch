from layers.att_layers import *
from models.encoders import model2encoder
from models.decoders import model2decoder
from utils.eval_utils import *
import random


class BaseModel(nn.Module):
    # Base Model for KG Embedding Task

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.n_nodes = args.n_nodes
        self.device = args.device

    def get_neg(self, ILL, output, k):
        neg = []
        t = len(ILL)
        ILL_vec = np.array([output[e].detach().cpu().numpy() for e in ILL])
        KG_vec = np.array(output.detach().cpu())
        sim = scipy.spatial.distance.cdist(ILL_vec, KG_vec, metric='cityblock')
        for i in range(t):
            rank = sim[i, :].argsort()
            neg.append(rank[0:k])
        neg = np.array(neg)
        neg = neg.reshape((t * k,))
        return neg

    def get_neg_triplet(self, triples, head, tail, ids):
        neg = []
        for triple in triples:
            (h, r, t) = triple
            h2, r2, t2 = h, r, t
            neg_scope, num = True, 0
            while True:
                nt = random.randint(0, 999)
                if nt < 500:
                    if neg_scope:
                        h2 = random.sample(head[r], 1)[0]
                    else:
                        h2 = random.sample(range(ids), 1)[0]
                else:
                    if neg_scope:
                        t2 = random.sample(tail[r], 1)[0]
                    else:
                        t2 = random.sample(range(ids), 1)[0]
                if (h2, r2, t2) not in triples:
                    break
                else:
                    num += 1
                    if num > 10:
                        neg_scope = False
            neg.append((h2, r2, t2))
        return neg

    def compute_metrics(self, outputs, data, split):
        if split == 'train':
            pair = data['train']
        else:
            pair = data['test']
        if outputs.is_cuda:
            outputs = outputs.cpu()
        return get_hits(outputs, pair)

    def has_improved(self, m1, m2):
        return (m1['Hits@10_l'] < m2['Hits@10_l']) \
               or (m1['Hits@10_r'] < m2['Hits@10_r'])

    def init_metric_dict(self):
        return {'Hits@1_l': -1, 'Hits@10_l': -1, 'Hits@50_l': -1, 'Hits@100_l': -1,
                'Hits@1_r': -1, 'Hits@10_r': -1, 'Hits@50_r': -1, 'Hits@100_r': -1}


class NCModel(BaseModel):
    # Base Model for Entity Alignment Task

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.encoder = model2encoder[args.model](args)
        self.decoder = model2decoder[args.model](args)
        ILL = args.data['train']
        t = len(ILL)
        k = args.neg_num
        self.neg_num = k
        L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
        self.neg_left = L.reshape((t * k,))
        L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
        self.neg2_right = L.reshape((t * k,))
        self.neg_right = None
        self.neg2_left = None

    def encode(self, x, adj):
        h = self.encoder.encode(x, adj)
        return h

    def decode(self, h, adj):
        output = self.decoder.decode(h, adj)
        return output

    def get_loss(self, outputs, data, split):
        ILL = data[split]
        left = ILL[:, 0]
        right = ILL[:, 1]
        t = len(ILL)
        k = self.neg_num
        left_x = outputs[left]
        right_x = outputs[right]
        A = torch.sum(torch.abs(left_x - right_x), 1)
        neg_l_x = outputs[self.neg_left]
        neg_r_x = outputs[self.neg_right]
        B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
        C = - torch.reshape(B, [t, k])
        D = A + 1.0
        L1 = F.relu(torch.add(C, torch.reshape(D, [t, 1])))
        neg_l_x = outputs[self.neg2_left]
        neg_r_x = outputs[self.neg2_right]
        B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
        C = - torch.reshape(B, [t, k])
        L2 = F.relu(torch.add(C, torch.reshape(D, [t, 1])))
        return (torch.sum(L1) + torch.sum(L2)) / (2.0 * t * k)

class KEModel(BaseModel):
    # TransE Model for Entity Alignment Task

    def __init__(self, args):
        super(KEModel, self).__init__(args)
        ILL = args.data['train']
        t = len(ILL)
        k = args.neg_num
        self.neg_num = k
        L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
        self.neg_left = L.reshape((t * k,))
        L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
        self.neg2_right = L.reshape((t * k,))
        self.neg_right = None
        self.neg2_left = None
        self.neg_triple = self.get_neg_triplet(args.data['triple'], args.data['head'], args.data['tail'], args.data['x'].shape[0])
        self.eembed = nn.Embedding.from_pretrained(args.data['x'].to_dense(), freeze=False)
        self.rembed = nn.Embedding.from_pretrained(args.data['r'].to_dense(), freeze=False)
        self.dropout = nn.Dropout(args.dropout)

    def encode(self, e, r):
        e = self.eembed(e)
        e = self.dropout(e)
        r = self.rembed(r)
        r = self.dropout(r)
        return e, r

    def get_loss(self, outputs, relation, data, split):
        pos_tri = data['triple']
        h = [t[0] for t in pos_tri]
        r = [t[1] for t in pos_tri]
        t = [t[2] for t in pos_tri]
        diff_pos = F.normalize(outputs[h] + relation[r] - outputs[t], p=2)
        neg_tri = self.neg_triple
        h = [t[0] for t in neg_tri]
        r = [t[1] for t in neg_tri]
        t = [t[2] for t in neg_tri]
        diff_neg = F.normalize(outputs[h] + relation[r] - outputs[t], p=2)
        Y = torch.ones(diff_pos.size(0), 1).to(self.device)
        loss_r = F.margin_ranking_loss(diff_pos.sum(1).view(-1, 1), diff_neg.sum(1).view(-1, 1), Y, 3)
        print(loss_r)

        return loss_r


class RKDEA(BaseModel):
    # KG Distillation Model of GCN and TransE

    def __init__(self, args):
        super(RKDEA, self).__init__(args)
        self.encoder = model2encoder[args.model](args)
        self.decoder = model2decoder[args.model](args)
        ILL = args.data['train']
        t = len(ILL)
        k = args.neg_num
        self.neg_num = k
        L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
        self.neg_left = L.reshape((t * k,))
        L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
        self.neg2_right = L.reshape((t * k,))
        self.neg_right = None
        self.neg2_left = None

    def encode(self, x, adj):
        h = self.encoder.encode(x, adj)
        return h

    def decode(self, h, adj):
        output = self.decoder.decode(h, adj)
        return output

    def get_loss(self, outputs, data, split):
        tri = data['triple']
        h = [t[0] for t in tri]
        t = [t[2] for t in tri]
        diff_gcn = outputs[h] - outputs[t]
        diff_transe = data['emb'][h] - data['emb'][t]
        loss_r = F.mse_loss(diff_gcn, diff_transe)
        print(loss_r)

        ILL = data[split]
        left = ILL[:, 0]
        right = ILL[:, 1]
        t = len(ILL)
        k = self.neg_num
        left_x = outputs[left]
        right_x = outputs[right]
        A = torch.sum(torch.abs(left_x - right_x), 1)
        neg_l_x = outputs[self.neg_left]
        neg_r_x = outputs[self.neg_right]
        B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
        C = - torch.reshape(B, [t, k])
        D = A + 1.0
        L1 = F.relu(torch.add(C, torch.reshape(D, [t, 1])))
        neg_l_x = outputs[self.neg2_left]
        neg_r_x = outputs[self.neg2_right]
        B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
        C = - torch.reshape(B, [t, k])
        L2 = F.relu(torch.add(C, torch.reshape(D, [t, 1])))
        loss_e = (torch.sum(L1) + torch.sum(L2)) / (2.0 * t * k)
        print(loss_e)

        rate = loss_r.detach() * loss_e.detach()

        return rate * loss_r + loss_e
